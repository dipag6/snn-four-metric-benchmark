#!/usr/bin/env python
"""Gate remediation and diagnosis runs for the MR604 benchmark.

This driver does NOT re-run the main grid and does not overwrite it. It writes to
its own output directory and answers the four open questions left by Chapter 5:

  arm A  reg      Can gate G1 be passed at T = 16 legitimately, by training the
                  network to fire less, rather than by relaxing the gate?
  arm B  sampling Is the G2 failure on rate-coded QCFS caused by training variance
                  or by input-sampling variance? Separates the two.
  arm C  quant    What accuracy does a post-training-quantised ANN retain at 8 and
                  4 bits? This is the measurement RQ1 is currently conditional on.
  arm D  seeds5   Honest escalation of the three G2-failing configurations to five
                  seeds, reporting whatever comes out.

None of these arms is guaranteed to produce a pass, and arms B and D are expected
not to. That is the point: a gate that cannot fail is not evidence of anything.

Usage
-----
    python run_gates.py all      --epochs 64 --out-dir ./results_gates
    python run_gates.py reg      --epochs 64 --lam 0.05
    python run_gates.py sampling --eval-seeds 10
    python run_gates.py quant
    python run_gates.py seeds5
"""
from __future__ import annotations
import argparse, json, statistics as st
from pathlib import Path

import torch

from snnbench.config import (RunConfig, breakeven_spike_rate,
                             E_AC_TH_PJ, E_AC_SI_PJ)
from snnbench.data import get_loaders, encode
from snnbench.engine import (device_of, set_seed, forward_batch, train_one,
                             evaluate_four_metrics, measure_latency)
from snnbench.models import build_model, PathB_ConvertedSNN
from snnbench.ratereg import RatePenalty
from snnbench.quantise import quantise_baseline_ann

SEEDS3 = [42, 123, 789]
SEEDS5 = [42, 123, 789, 1011, 2027]


def save(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"  -> {path}")


def hr(msg):
    print("\n" + "=" * 78 + f"\n{msg}\n" + "=" * 78)


# ---------------------------------------------------------------- arm A: reg --
def train_regularised(cfg, lam, margin, progress=True):
    """train_one with a spike-rate penalty added to the loss.

    Mirrors snnbench.engine.train_one; kept separate so the published grid's
    training path is untouched and remains byte-for-byte reproducible.
    """
    import torch.nn as nn
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from tqdm import tqdm

    dev = device_of(cfg)
    set_seed(cfg.seed)
    train_loader, test_loader = get_loaders(cfg)
    model = build_model(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.cosine_t_max)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and dev.type == "cuda")
    crit = nn.CrossEntropyLoss()
    pen = RatePenalty(model, T=cfg.T, margin=margin)
    print(f"  rate penalty: lambda={lam}  target={pen.target:.4f} "
          f"(= {margin} x calibrated s* at T={cfg.T})")

    ckpt = Path(cfg.ckpt_dir) / f"{cfg.tag}_reg.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for ep in range(cfg.epochs):
        model.train()
        tot = correct = 0
        loss_sum = pen_sum = 0.0
        for x, y in tqdm(train_loader, desc=f"{cfg.tag}_reg ep{ep+1}/{cfg.epochs}",
                         leave=False, disable=not progress):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pen.reset()
            with torch.amp.autocast("cuda", enabled=cfg.amp and dev.type == "cuda"):
                out = forward_batch(model, x, cfg, dev)
                ce = crit(out, y)
                rp = pen.penalty()
                loss = ce + lam * rp
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            loss_sum += ce.item() * y.size(0)
            pen_sum += float(rp.detach()) * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        sched.step()
        rates = pen.rates()
        history.append(dict(epoch=ep + 1, train_loss=loss_sum / tot,
                            penalty=pen_sum / tot, rates=rates))
        print(f"  [{cfg.tag}_reg] ep {ep+1}/{cfg.epochs} ce {loss_sum/tot:.4f} "
              f"pen {pen_sum/tot:.5f} rates " +
              " ".join(f"{k}={v:.4f}" for k, v in sorted(rates.items())))
        torch.save(dict(model=model.state_dict(), epoch=ep, history=history,
                        cfg=cfg.to_dict(), lam=lam, target=pen.target), ckpt)
    pen.remove()
    return model, ckpt, history


def arm_reg(a):
    hr("ARM A -- spike-rate regularisation, STBP T = 16")
    print("Question: can G1 be passed at the calibrated constant by training the\n"
          "network to fire less, without losing the accuracy that made T = 16\n"
          "worth reporting? Reported as a new arm, never as a replacement for the\n"
          "unregularised T = 16 result.")
    out = []
    for seed in SEEDS3:
        cfg = RunConfig(pathway="stbp", T=a.T, seed=seed, epochs=a.epochs,
                        out_dir=a.out_dir, ckpt_dir=a.ckpt_dir,
                        data_root=a.data_root)
        model, ckpt, hist = train_regularised(cfg, a.lam, a.margin)
        res = evaluate_four_metrics(model, cfg, device_of(cfg))
        res["tag"] = f"{cfg.tag}_reg"
        res["rate_penalty_lambda"] = a.lam
        res["rate_target"] = a.margin * breakeven_spike_rate(a.T, E_AC_SI_PJ)
        save(res, Path(a.out_dir) / f"{cfg.tag}_reg.json")
        out.append(res)

    thr = breakeven_spike_rate(a.T, E_AC_SI_PJ)
    print(f"\n  calibrated G1 threshold at T={a.T}: {thr:.4f}")
    accs = [r["accuracy"] for r in out]
    for r in out:
        lay = r.get("spike_rate_per_layer") or {}
        worst = max(lay.values()) if lay else float("nan")
        print(f"  {r['tag']:<22} acc {r['accuracy']:6.2f}%  worst layer {worst:.4f}  "
              f"{'PASS' if worst < thr else 'STILL FAILS'}")
    print(f"  accuracy mean {st.mean(accs):.2f}  s.d. {st.stdev(accs):.3f} pp"
          if len(accs) > 1 else "")
    return out


# ----------------------------------------------------- arm B: sampling split --
def arm_sampling(a):
    hr("ARM B -- separating input-sampling variance from training variance")
    print("Question: the G2 failure on rate-coded QCFS -- is it the weights or the\n"
          "Bernoulli input draw? Holds the trained network fixed and varies only\n"
          "the encoder's random stream. This DIAGNOSES the failure; it does not\n"
          "remove it, and the configuration is expected to remain unstable.")
    rows = []
    for T in a.T_list:
        for seed in SEEDS3:
            cfg = RunConfig(pathway="qcfs", T=T, seed=seed, epochs=a.epochs,
                            qcfs_levels=T, encoding="rate", out_dir=a.out_dir,
                            ckpt_dir=a.ckpt_dir, data_root=a.data_root)
            dev = device_of(cfg)
            ck = Path(cfg.ckpt_dir) / f"{cfg.tag}.pt"
            if not ck.exists():
                print(f"  [{cfg.tag}] no checkpoint -- training")
                train_one(cfg)
            set_seed(cfg.seed)
            src = build_model(cfg).to(dev)
            src.load_state_dict(torch.load(ck, map_location=dev)["model"])
            snn = PathB_ConvertedSNN(src, cfg).to(dev).eval()
            _, test_loader = get_loaders(cfg)

            accs = []
            for es in range(a.eval_seeds):
                torch.manual_seed(10_000 + es)      # varies ONLY the encoder draw
                correct = tot = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = x.to(dev), y.to(dev)
                        out = forward_batch(snn, x, cfg, dev)
                        correct += (out.argmax(1) == y).sum().item()
                        tot += y.size(0)
                accs.append(100.0 * correct / tot)
            rows.append(dict(T=T, weight_seed=seed, eval_accs=accs,
                             mean=st.mean(accs),
                             sampling_sd=st.stdev(accs) if len(accs) > 1 else 0.0))
            print(f"  T={T} weights seed {seed}: sampling s.d. over "
                  f"{a.eval_seeds} draws = {rows[-1]['sampling_sd']:.3f} pp "
                  f"(mean {rows[-1]['mean']:.2f}%)")

    summary = {}
    for T in a.T_list:
        sel = [r for r in rows if r["T"] == T]
        across_weights = st.stdev([r["mean"] for r in sel]) if len(sel) > 1 else 0.0
        within = st.mean([r["sampling_sd"] for r in sel])
        summary[f"T{T}"] = dict(sd_across_weight_seeds=across_weights,
                                mean_sd_within_weights_across_sampling=within)
        print(f"\n  T={T}:  training/weight variance {across_weights:.3f} pp   "
              f"input-sampling variance {within:.3f} pp")
    save(dict(rows=rows, summary=summary),
         Path(a.out_dir) / "_sampling_variance.json")
    return summary


# ------------------------------------------------------------- arm C: quant --
def arm_quant(a):
    hr("ARM C -- post-training quantised baselines (8-bit and 4-bit)")
    print("Question: RQ1 is currently conditional because the 4-bit ANN has an\n"
          "analytical energy of 0.305 uJ -- lower than every SNN measured -- but no\n"
          "measured accuracy. This supplies it. Method is plain PTQ, deliberately\n"
          "the weakest reasonable choice, so the comparison is conservative.")
    from snnbench.energy import ann_energy_uj
    out = []
    for bits in a.bits:
        accs, gpu_ms, cpu_ms = [], [], []
        for seed in SEEDS3:
            cfg = RunConfig(pathway="ann", T=0, seed=seed, epochs=a.epochs,
                            out_dir=a.out_dir, ckpt_dir=a.ckpt_dir,
                            data_root=a.data_root)
            dev = device_of(cfg)
            ck = Path(cfg.ckpt_dir) / f"{cfg.tag}.pt"
            if not ck.exists():
                print(f"  [{cfg.tag}] no checkpoint -- training")
                train_one(cfg)
            set_seed(seed)
            model = build_model(cfg).to(dev)
            model.load_state_dict(torch.load(ck, map_location=dev)["model"])
            train_loader, test_loader = get_loaders(cfg)
            q = quantise_baseline_ann(model, bits, train_loader, dev)
            correct = tot = 0
            with torch.no_grad():
                for x, y in test_loader:
                    x, y = x.to(dev), y.to(dev)
                    correct += (q(x).argmax(1) == y).sum().item()
                    tot += y.size(0)
            accs.append(100.0 * correct / tot)
            gpu_ms.append(measure_latency(q, cfg, dev, cfg.eval_batch_size,
                                          cfg.gpu_latency_reps, cfg.latency_warmup))
            cpu_ms.append(measure_latency(q.to("cpu"), cfg, torch.device("cpu"),
                                          1, cfg.cpu_latency_reps, cfg.latency_warmup))
        n_mac = 4_241_152
        rec = dict(tag=f"ann_int{bits}", pathway="ann", bits=bits, seeds=SEEDS3,
                   accuracy=st.mean(accs), accuracy_sd=st.stdev(accs),
                   accuracies=accs,
                   e_th_uj=ann_energy_uj(n_mac, bits),
                   e_si_uj=ann_energy_uj(n_mac, bits),
                   latency_gpu_ms=st.mean(gpu_ms), latency_cpu_ms=st.mean(cpu_ms),
                   method="post-training quantisation, symmetric per-tensor weights, "
                          "unsigned per-tensor activations, 16 calibration batches")
        save(rec, Path(a.out_dir) / f"ann_int{bits}.json")
        print(f"  {bits}-bit: acc {rec['accuracy']:.2f} +/- {rec['accuracy_sd']:.3f} pp  "
              f"energy {rec['e_th_uj']:.3f} uJ  "
              f"gpu {rec['latency_gpu_ms']:.4f} ms  cpu {rec['latency_cpu_ms']:.4f} ms")
        out.append(rec)

    print("\n  Interpretation rule, fixed in advance: the quantised baseline defeats a\n"
          "  spiking configuration only if it clears the SAME accuracy floor at LOWER\n"
          "  energy. If it does, that is the answer to RQ1 and must be reported as such.")
    return out


# ------------------------------------------------------------ arm D: seeds5 --
def arm_seeds5(a):
    hr("ARM D -- honest escalation of the G2 failures to five seeds")
    print("Gate G2 prescribes escalation when the 3-seed s.d. exceeds 1 pp. On the\n"
          "3-seed evidence two of these three are very unlikely to clear it\n"
          "(p = 0.0002 and 0.020 against a true s.d. of 1.0). Escalation improves the\n"
          "estimate; it does not reduce the variance. Report the outcome either way.")
    targets = [("qcfs", 2, "rate"), ("qcfs", 8, "rate"), ("qcfs", 8, "direct")]
    summary = {}
    for pathway, T, enc in targets:
        accs = []
        for seed in SEEDS5:
            cfg = RunConfig(pathway=pathway, T=T, seed=seed, epochs=a.epochs,
                            qcfs_levels=T, encoding=enc, out_dir=a.out_dir,
                            ckpt_dir=a.ckpt_dir, data_root=a.data_root)
            dev = device_of(cfg)
            ck = Path(cfg.ckpt_dir) / f"{cfg.tag}.pt"
            if not ck.exists():
                print(f"  [{cfg.tag}] training seed {seed}")
                train_one(cfg)
            set_seed(seed)
            src = build_model(cfg).to(dev)
            src.load_state_dict(torch.load(ck, map_location=dev)["model"])
            snn = PathB_ConvertedSNN(src, cfg).to(dev)
            res = evaluate_four_metrics(snn, cfg, dev)
            accs.append(res["accuracy"])
        sd = st.stdev(accs)
        key = f"{pathway}_T{T}_{enc}"
        summary[key] = dict(seeds=SEEDS5, accuracies=accs, mean=st.mean(accs),
                            sd=sd, G2_pass=bool(sd < 1.0))
        print(f"  {key:<20} accs {[f'{x:.2f}' for x in accs]}  "
              f"s.d. {sd:.3f}  G2 {'PASS' if sd < 1.0 else 'STILL FAILS'}")
    save(summary, Path(a.out_dir) / "_g2_escalation.json")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arm", choices=["all", "reg", "sampling", "quant", "seeds5"])
    ap.add_argument("--epochs", type=int, default=64)
    ap.add_argument("--out-dir", default="./results_gates")
    ap.add_argument("--ckpt-dir", default="./checkpoints")
    ap.add_argument("--data-root", default="./data")
    # arm A
    ap.add_argument("--T", type=int, default=16, help="timestep count for arm A")
    ap.add_argument("--lam", type=float, default=0.05, help="rate-penalty weight")
    ap.add_argument("--margin", type=float, default=0.80,
                    help="target rate as a fraction of the calibrated threshold")
    # arm B
    ap.add_argument("--eval-seeds", type=int, default=10)
    ap.add_argument("--T-list", type=int, nargs="*", default=[2, 8])
    # arm C
    ap.add_argument("--bits", type=int, nargs="*", default=[8, 4])
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FATAL: no CUDA GPU visible. These runs are not viable on CPU.")
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)

    if a.arm in ("all", "quant"):    arm_quant(a)
    if a.arm in ("all", "sampling"): arm_sampling(a)
    if a.arm in ("all", "reg"):      arm_reg(a)
    if a.arm in ("all", "seeds5"):   arm_seeds5(a)

    hr("Done")
    print(f"Results in {a.out_dir}. These are SEPARATE from the published grid in\n"
          "results/. Report them as additional arms; do not merge them into\n"
          "Tables 5.4-5.11 without saying which rows came from which run.")


if __name__ == "__main__":
    main()
