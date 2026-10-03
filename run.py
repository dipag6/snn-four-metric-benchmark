#!/usr/bin/env python
"""MR604 benchmark driver.

  python run.py verify                      # check operation counts, no training
  python run.py smoke                       # stage 1: every config, 2 epochs
  python run.py grid --epochs 64            # stage 2: the full 21-run grid
  python run.py priority --epochs 64        # ANN + STBP T=8 + QCFS T=4 first
  python run.py one --pathway stbp --T 8 --seed 42 --epochs 64
  python run.py eval --pathway stbp --T 8 --seed 42   # TEST ONLY, no training
  python run.py gates                       # re-apply gates to saved results
  python run.py data                        # download MNIST and verify it
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from snnbench.config import (RunConfig, full_grid, smoke_grid, priority_grid,
                             nmist_grid, EVENT_DATASETS,
                             cifar_grid, CIFAR_DATASETS, CIFAR_BATCH,  # CIFAR_IMPORTS
                             SEEDS, STBP_TIMESTEPS, QCFS_TIMESTEPS)
from snnbench.energy import verify_counts
from snnbench.engine import train_one, evaluate_four_metrics, device_of, set_seed
from snnbench.gates import apply_gates, format_gate_report
from snnbench.models import (build_model, PathB_ConvertedSNN,
                             build_converted)   # IMPORT_BUILD_CONVERTED


def save_result(cfg, result, history=None):
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if history is not None:
        result["history"] = history
    with open(out / f"{cfg.tag}.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  -> saved {out / (cfg.tag + '.json')}")


def load_results(out_dir):
    files = sorted(Path(out_dir).glob("*.json"))
    res = []
    for f in files:
        if f.name.startswith("_"):
            continue
        with open(f) as fh:
            res.append(json.load(fh))
    return res


def run_config(cfg, skip_if_done=True):
    out_json = Path(cfg.out_dir) / f"{cfg.tag}.json"
    if skip_if_done and out_json.exists():
        print(f"[{cfg.tag}] result exists -- skipping (delete the json to force a rerun)")
        return json.load(open(out_json))

    print("=" * 78)
    print(f"RUN {cfg.tag}   pathway={cfg.pathway} T={cfg.T} seed={cfg.seed} "
          f"epochs={cfg.epochs} encoding={cfg.encoding} backend={cfg.backend}")
    print("=" * 78)

    dev = device_of(cfg)
    ckpt, history = train_one(cfg)

    # rebuild and load
    set_seed(cfg.seed)
    model = build_model(cfg).to(dev)
    st = torch.load(ckpt, map_location=dev)
    model.load_state_dict(st["model"])

    # Path B: convert the trained QCFS source ANN, and evaluate under BOTH input
    # encodings. This costs nothing (conversion is inference-only) and it exposes
    # a trade-off the thesis does not currently consider:
    #
    #   rate coding   -- matched with Path A, so the comparison stays honest, but
    #                    Bernoulli sampling at small T injects large input noise
    #                    and conversion loss is severe (measured: 42 pp at T=4).
    #   direct coding -- what Bu et al. actually use; conversion loss goes to zero,
    #                    but conv1 then receives analogue values and executes MACs
    #                    at every timestep, so energy roughly doubles.
    #
    # Neither is "the" answer. Report both rows and let the four metrics decide.
    if cfg.pathway == "qcfs":
        import dataclasses
        from snnbench.engine import quick_accuracy
        from snnbench.data import get_loaders
        _, tl = get_loaders(cfg)
        src_acc = quick_accuracy(model, tl, cfg, dev)
        print(f"[{cfg.tag}] source QCFS ANN accuracy: {src_acc:.2f}%")

        source = model
        result = None
        for enc in ("rate", "direct"):
            c2 = dataclasses.replace(cfg, encoding=enc)
            snn = build_converted(source, c2).to(dev)   # USE_BUILD_CONVERTED
            r = evaluate_four_metrics(snn, c2, dev)
            r["source_ann_accuracy"] = src_acc
            r["conversion_loss_pp"] = src_acc - r["accuracy"]
            r["input_encoding"] = enc
            r["qcfs_lambdas"] = list(snn.lams)
            print(f"[{cfg.tag}/{enc}] converted acc {r['accuracy']:.2f}% "
                  f"(loss {r['conversion_loss_pp']:+.2f} pp)  "
                  f"E_th {r['e_th_uj']:.3f} uJ  E_si {r['e_si_uj']:.3f} uJ  "
                  f"MAC residue {r.get('n_mac_residue', 0):,.0f}")
            if enc == "rate":
                result = r
            else:
                r["pathway"] = "qcfs_direct"
                r["tag"] = f"qcfs_direct_T{cfg.T}_s{cfg.seed}"
                alt = dataclasses.replace(cfg, out_dir=cfg.out_dir)
                out = Path(cfg.out_dir); out.mkdir(parents=True, exist_ok=True)
                with open(out / f"{r['tag']}.json", "w") as f:
                    json.dump(r, f, indent=2, default=str)
                print(f"  -> saved {out / (r['tag'] + '.json')}")
        save_result(cfg, result, history)
        return result

    result = evaluate_four_metrics(model, cfg, dev)

    print(f"[{cfg.tag}] acc {result['accuracy']:.2f}%  "
          f"s_bar {result.get('spike_rate_mean')}  "
          f"E_th {result['e_th_uj']:.3f} uJ  E_si {result['e_si_uj']:.3f} uJ  "
          f"gpu {result['latency_gpu_ms']:.4f} ms  cpu {result['latency_cpu_ms']:.4f} ms")
    save_result(cfg, result, history)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify", "data", "smoke", "grid", "priority",
                                     "one", "eval", "gates"])
    ap.add_argument("--epochs", type=int, default=64)
    ap.add_argument("--pathway", default="stbp")
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--dataset", default="mnist",
                    choices=["mnist", "fmnist", "kmnist", "nmist",   # CIFAR_CHOICES
                             "cifar10", "cifar100"],
                    help="static 28x28x1 (mnist/fmnist/kmnist) or N-MNIST event data "
                         "(nmist; STBP-only)")
    ap.add_argument("--encoding", default="rate", choices=["rate", "direct", "ttfs"])
    ap.add_argument("--backend", default="spikingjelly", choices=["spikingjelly", "native"])
    ap.add_argument("--batch-size", type=int, default=0,   # BATCH_DEFAULT_BY_DATASET
                    help="0 = the default for the chosen dataset "
                         "(256 for the MNIST family, 128 for CIFAR)")
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--force", action="store_true", help="rerun even if a result exists")
    args = ap.parse_args()

    common = dict(out_dir=args.out_dir, ckpt_dir=args.ckpt_dir, data_root=args.data_root,
                  dataset=args.dataset,
                  encoding=args.encoding, backend=args.backend,
                  batch_size=(args.batch_size or
                              (CIFAR_BATCH if args.dataset in CIFAR_DATASETS else 256)),
                  amp=not args.no_amp)   # BATCH_RESOLVED

    if args.mode == "verify":
        ok, rows, total, msgs = verify_counts(dataset=args.dataset)   # VERIFY_WITH_DATASET
        from snnbench.config import breakeven_spike_rate, E_AC_TH_PJ, E_AC_SI_PJ
        print("\n--- break-even thresholds (gate G1) ---")
        print(f"{'T':>4}{'s* @0.9pJ':>12}{'s* @1.89pJ':>13}{'binding?':>28}")
        for T in (2, 4, 8, 16):
            a = breakeven_spike_rate(T, E_AC_TH_PJ)
            b = breakeven_spike_rate(T, E_AC_SI_PJ)
            note = ("neither binding" if a >= 1 and b >= 1 else
                    ("calibrated only" if a >= 1 else "both binding"))
            print(f"{T:>4}{a:>12.4f}{b:>13.4f}{note:>28}")
        print(f"\nANN baseline energy: {total['macs'] * 4.6 / 1e6:.3f} uJ/image")
        sys.exit(0 if ok else 1)

    if args.mode == "data":
        if args.dataset in EVENT_DATASETS:
            # N-MNIST: download + integrate event frames via SpikingJelly (first run
            # is slow and needs ~1 GB). Frame a small T just to verify the pipeline.
            from snnbench.data import get_loaders
            cfg = RunConfig(pathway="stbp", T=4, seed=42, dataset="nmist",
                            data_root=args.data_root, num_workers=0)
            tr, te = get_loaders(cfg)
            xb, yb = next(iter(te))
            print(f"dataset: {args.dataset} (SpikingJelly event frames)")
            print(f"frame batch shape: {tuple(xb.shape)}   (expect [B, T=4, 2, 34, 34])")
            print(f"train batches: {len(tr)}   test batches: {len(te)}")
            ok = (xb.dim() == 5 and xb.shape[1] == 4 and tuple(xb.shape[2:]) == (2, 34, 34))
            print(f"\ndataset check: {'PASS' if ok else 'FAIL'}")
            sys.exit(0 if ok else 1)
        # Download MNIST and prove it is intact. Nothing else touches the network,
        # so if this works the rest of the pipeline will not stall on a download.
        from torchvision import datasets, transforms
        from snnbench.config import DATASETS
        cls = getattr(datasets, DATASETS[args.dataset])
        print(f"dataset: {args.dataset} -> torchvision.datasets.{DATASETS[args.dataset]}")
        tr = cls(args.data_root, train=True, download=True,
                 transform=transforms.ToTensor())
        te = cls(args.data_root, train=False, download=True,
                 transform=transforms.ToTensor())
        x, y = tr[0]
        import collections
        print(f"\ntrain images : {len(tr):,}   (expected 60,000)")
        print(f"test images  : {len(te):,}   (expected 10,000)")
        print(f"image shape  : {tuple(x.shape)}   (expected (1, 28, 28))")
        print(f"pixel range  : [{x.min():.3f}, {x.max():.3f}]   (expected [0.000, 1.000])")
        print(f"classes      : {sorted(set(int(t) for t in tr.targets[:2000]))}")
        cnt = collections.Counter(int(t) for t in te.targets)
        print(f"test class counts: {[cnt[i] for i in range(10)]}")
        ok = (len(tr) == 60000 and len(te) == 10000 and tuple(x.shape) == (1, 28, 28))
        print(f"\ndataset check: {'PASS' if ok else 'FAIL'}")
        print(f"stored at: {Path(args.data_root).resolve()}/MNIST/raw")
        sys.exit(0 if ok else 1)

    if args.mode == "eval":
        # Evaluate an ALREADY-TRAINED checkpoint. No training, no weight changes.
        cfg = RunConfig(pathway=args.pathway, T=args.T, seed=args.seed,
                        epochs=args.epochs,
                        qcfs_levels=args.T if args.pathway == "qcfs" else 8, **common)
        ckpt = Path(cfg.ckpt_dir) / f"{cfg.tag}.pt"
        if not ckpt.exists():
            print(f"No checkpoint at {ckpt}. Train it first:\n"
                  f"  python run.py one --pathway {cfg.pathway} --T {cfg.T} "
                  f"--seed {cfg.seed} --epochs {args.epochs}")
            sys.exit(1)
        dev = device_of(cfg)
        st = torch.load(ckpt, map_location=dev)
        print(f"loaded {ckpt.name}: trained through epoch {st['epoch'] + 1}"
              f" of {st['cfg']['epochs']}")
        if st["epoch"] + 1 < st["cfg"]["epochs"]:
            print("  *** WARNING: this checkpoint is from an INCOMPLETE run. "
                  "Its numbers are not final and must not go into a thesis table. ***")
        set_seed(cfg.seed)
        model = build_model(cfg).to(dev)
        model.load_state_dict(st["model"])
        if cfg.pathway == "qcfs":
            model = build_converted(model, cfg).to(dev)   # USE_BUILD_CONVERTED
        result = evaluate_four_metrics(model, cfg, dev)
        print(json.dumps({k: v for k, v in result.items()
                          if k not in ("history", "sop_breakdown")},
                         indent=2, default=str))
        sys.exit(0)

    if args.mode == "gates":
        results = load_results(args.out_dir)
        if not results:
            print(f"no results in {args.out_dir}")
            sys.exit(1)
        per_run, summary = apply_gates(results)
        print(format_gate_report(per_run, summary))
        with open(Path(args.out_dir) / "_gates.json", "w") as f:
            json.dump(dict(per_run=per_run, summary=summary), f, indent=2, default=str)
        return

    if args.mode == "one":
        cfg = RunConfig(pathway=args.pathway, T=args.T, seed=args.seed,
                        epochs=args.epochs,
                        qcfs_levels=args.T if args.pathway == "qcfs" else 8, **common)
        run_config(cfg, skip_if_done=not args.force)
        return

    if args.mode == "smoke":
        runs = smoke_grid(**common)
    elif args.mode == "priority":
        runs = priority_grid(epochs=args.epochs, **common)
    elif args.dataset in CIFAR_DATASETS:          # CIFAR_GRID_DISPATCH
        runs = cifar_grid(dataset=args.dataset, epochs=args.epochs, seeds=args.seeds,
                          **{k: v for k, v in common.items() if k != "dataset"})
    elif args.dataset in EVENT_DATASETS:
        # N-MNIST is STBP-only; use its dedicated grid regardless of grid/priority.
        runs = nmist_grid(epochs=args.epochs, seeds=args.seeds, **common)
    else:
        runs = full_grid(epochs=args.epochs, seeds=args.seeds, **common)

    print(f"\n{len(runs)} configurations queued\n")
    ok, _, _, _ = verify_counts(dataset=args.dataset)   # VERIFY_WITH_DATASET
    if not ok:
        print("\nOperation count verification FAILED. Fix this before training -- "
              "every energy figure depends on it.")
        sys.exit(1)

    for i, cfg in enumerate(runs, 1):
        print(f"\n### [{i}/{len(runs)}]")
        try:
            run_config(cfg, skip_if_done=not args.force)
        except KeyboardInterrupt:
            print("interrupted -- checkpoints are saved, rerun the same command to resume")
            raise
        except Exception as e:
            print(f"[{cfg.tag}] FAILED: {type(e).__name__}: {e}")
            import traceback; traceback.print_exc()

    results = load_results(args.out_dir)
    per_run, summary = apply_gates(results)
    print("\n" + format_gate_report(per_run, summary))
    with open(Path(args.out_dir) / "_gates.json", "w") as f:
        json.dump(dict(per_run=per_run, summary=summary), f, indent=2, default=str)


if __name__ == "__main__":
    main()
