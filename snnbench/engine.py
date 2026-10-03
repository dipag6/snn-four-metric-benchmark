"""Training, evaluation and latency measurement."""
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from .config import breakeven_spike_rate, E_AC_TH_PJ, E_AC_SI_PJ, arch_for
from .data import get_loaders, encode, encoding_is_binary
from .energy import SpikeMonitor, analytical_counts, energy_report, edp, aep
from .models import build_model, PathB_ConvertedSNN
from .specnets import SpecConvertedSNN  # QCFS_SPEC_PATCH: CIFAR converted SNN is a spec model
_CONVERTED = (PathB_ConvertedSNN, SpecConvertedSNN)
from .neurons import reset_net


# --------------------------------------------------------------------------
def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        torch.backends.cudnn.benchmark = True


def device_of(cfg):
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU. "
              "STBP T=16 on CPU is not practical -- fix your runtime.")
        return torch.device("cpu")
    return torch.device(cfg.device)


# --------------------------------------------------------------------------
def forward_batch(model, x, cfg, dev):
    """One forward pass, encoder applied if the pathway is spiking."""
    if cfg.pathway == "ann" or (cfg.pathway == "qcfs" and not isinstance(model, _CONVERTED)):
        return model(x)
    spikes = encode(x, cfg)
    reset_net(model, cfg)
    return model(spikes)


def train_one(cfg, resume=True, progress=True):
    """Train a single configuration. Returns the path to the final checkpoint.

    Checkpoints every epoch. On Colab this is what saves you when the runtime
    disconnects -- point ckpt_dir at Google Drive.
    """
    dev = device_of(cfg)
    set_seed(cfg.seed)
    train_loader, test_loader = get_loaders(cfg)

    model = build_model(cfg).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.cosine_t_max)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and dev.type == "cuda")
    crit = nn.CrossEntropyLoss()

    Path(cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cfg.ckpt_dir) / f"{cfg.tag}.pt"
    start_epoch = 0
    history = []

    if resume and ckpt_path.exists():
        st = torch.load(ckpt_path, map_location=dev)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        start_epoch = st["epoch"] + 1
        history = st.get("history", [])
        if start_epoch >= cfg.epochs:
            print(f"[{cfg.tag}] already complete ({start_epoch} epochs) -- skipping")
            return ckpt_path, history
        print(f"[{cfg.tag}] resuming from epoch {start_epoch}")

    for ep in range(start_epoch, cfg.epochs):
        model.train()
        tot, correct, loss_sum = 0, 0, 0.0
        n_clip = 0
        it = tqdm(train_loader, desc=f"{cfg.tag} ep{ep+1}/{cfg.epochs}",
                  leave=False, disable=not progress)
        for x, y in it:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and dev.type == "cuda"):
                out = forward_batch(model, x, cfg, dev)
                loss = crit(out, y)
            scaler.scale(loss).backward()
            if getattr(cfg, 'grad_clip', 0.0) > 0:   # GRAD_CLIP_PATCH
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                n_clip += int((not torch.isfinite(gn)) or gn.item() > cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
            if progress:
                it.set_postfix(loss=f"{loss_sum/tot:.4f}", acc=f"{100*correct/tot:.2f}")
        sched.step()

        te_acc = quick_accuracy(model, test_loader, cfg, dev)
        history.append(dict(epoch=ep + 1, clip_frac=n_clip / max(1, len(train_loader)), grad_clip=getattr(cfg, "grad_clip", 0.0), train_loss=loss_sum / tot,
                            train_acc=100 * correct / tot, test_acc=te_acc,
                            lr=sched.get_last_lr()[0]))
        print(f"[{cfg.tag}] epoch {ep+1}/{cfg.epochs}  "
              f"loss {loss_sum/tot:.4f}  train {100*correct/tot:.2f}%  test {te_acc:.2f}%  clipped {n_clip}/{len(train_loader)}")

        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(),
                        sched=sched.state_dict(), epoch=ep, history=history,
                        cfg=cfg.to_dict()), ckpt_path)

    return ckpt_path, history


@torch.no_grad()
def quick_accuracy(model, loader, cfg, dev):
    model.eval()
    correct = tot = 0
    for x, y in loader:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=cfg.amp and dev.type == "cuda"):
            out = forward_batch(model, x, cfg, dev)
        correct += (out.argmax(1) == y).sum().item()
        tot += y.size(0)
    return 100.0 * correct / tot


# --------------------------------------------------------------------------
# Latency (gate G3: three platforms, never aggregated)
# --------------------------------------------------------------------------
def _dummy_input(cfg, batch_size, dev):
    """A representative input tensor for latency timing, correct per dataset.
    Static datasets: [B, C, H, W] uniform noise. N-MNIST: [B, T, 2, 34, 34]
    sparse binary event frames (encode() then binarises + moves T to the front)."""
    a = arch_for(getattr(cfg, "dataset", "mnist"))
    if getattr(cfg, "dataset", "mnist") == "nmist":
        return (torch.rand(batch_size, cfg.T, a["in_channels"],
                           a["input_hw"], a["input_hw"], device=dev) > 0.9).float()
    return torch.rand(batch_size, a["in_channels"], a["input_hw"], a["input_hw"], device=dev)


@torch.no_grad()
def measure_latency(model, cfg, dev, batch_size, reps, warmup):
    model.eval()
    x = _dummy_input(cfg, batch_size, dev)
    def one():
        if cfg.pathway == "ann" or not isinstance(model, _CONVERTED) and cfg.pathway == "qcfs":
            return model(x)
        if cfg.pathway == "ann":
            return model(x)
        s = encode(x, cfg)
        reset_net(model, cfg)
        return model(s)
    for _ in range(warmup):
        one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        one()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / reps * 1000.0 / batch_size   # ms per image


def project_neuromorphic_latency_ms(T, gpu_ms_per_image):
    """A PROJECTION, not a measurement (Thesis Section 3.6, gate G3).

    Neuromorphic hardware executes timesteps asynchronously rather than
    sequentially. A crude first-order projection divides out the sequential
    timestep factor. Label every use of this number as projected.
    """
    return gpu_ms_per_image / max(T, 1)


# --------------------------------------------------------------------------
# Four-metric evaluation
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate_four_metrics(model, cfg, dev, max_monitor_batches=None):
    """Accuracy, latency (3 platforms), per-layer spike rate, energy (2 constants)."""
    _, test_loader = get_loaders(cfg)
    rows, total = analytical_counts(arch_for(getattr(cfg, "dataset", "mnist")))
    model.eval()

    is_spiking = cfg.pathway in ("stbp",) or isinstance(model, _CONVERTED)

    monitor = SpikeMonitor(model) if is_spiking else None
    correct = tot = 0
    for i, (x, y) in enumerate(tqdm(test_loader, desc=f"eval {cfg.tag}", leave=False)):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        out = forward_batch(model, x, cfg, dev)
        correct += (out.argmax(1) == y).sum().item()
        tot += y.size(0)
        if monitor is not None:
            monitor.note_batch(y.size(0), cfg.T)
            if max_monitor_batches and i + 1 >= max_monitor_batches:
                monitor.remove()
                monitor_frozen = True
                break
    acc = 100.0 * correct / tot

    result = dict(tag=cfg.tag, pathway=cfg.pathway, T=cfg.T, seed=cfg.seed,
                  encoding=cfg.encoding, accuracy=acc, n_test=tot,
                  n_mac_ann=total["macs"], n_params=total["params"],
                  n_spiking_sites=total["sites"])

    # --- spikes and energy (computed BEFORE latency: the timing passes would
    #     otherwise fire the still-attached SpikeMonitor hooks and inflate the
    #     counters without incrementing samples. Restores paper Sec. 4.3 guard.)
    if monitor is not None:
        rates = monitor.per_layer_spike_rate()
        n_sop, n_mac_res, breakdown = monitor.synaptic_operations(rows)
        monitor.remove()
        result["spike_rate_per_layer"] = rates
        result["spike_rate_mean"] = sum(rates.values()) / max(len(rates), 1)
        result["sop_breakdown"] = breakdown
        result.update(energy_report(n_sop, n_mac_res, total["macs"]))
        result["encoding_binary"] = encoding_is_binary(cfg)
    else:
        from .energy import ann_energy_uj
        result["spike_rate_per_layer"] = None
        result["spike_rate_mean"] = None
        result["e_ann_uj"] = ann_energy_uj(total["macs"])
        result["e_th_uj"] = result["e_ann_uj"]
        result["e_si_uj"] = result["e_ann_uj"]
        result["n_sop"] = 0

    # --- latency, three platforms, reported separately (G3)
    gpu_ms = measure_latency(model, cfg, dev, cfg.eval_batch_size,
                             cfg.gpu_latency_reps // 10, cfg.latency_warmup // 10)
    cpu_dev = torch.device("cpu")
    model_cpu = model.to(cpu_dev)
    cpu_ms = measure_latency(model_cpu, cfg, cpu_dev, 1,
                             cfg.cpu_latency_reps, min(20, cfg.latency_warmup))
    model.to(dev)
    result["latency_gpu_ms"] = gpu_ms
    result["latency_cpu_ms"] = cpu_ms
    result["latency_neuromorphic_projected_ms"] = (
        project_neuromorphic_latency_ms(cfg.T, gpu_ms) if is_spiking else None)

    # --- sanity checks: catch the two silent failures that ruin a Chapter 5
    warnings = []
    if is_spiking:
        if not result.get("spike_rate_per_layer") or \
                all(v == 0.0 for v in result["spike_rate_per_layer"].values()):
            warnings.append(
                "ZERO SPIKES in every neuron layer at evaluation. This is NOT expected "
                "on full MNIST at any epoch count and the run must not be reported.\n"
                "      The usual cause is unconverged BatchNorm running statistics: they "
                "start at var=1.0 and decay toward the true value as 0.9^k, so at low k "
                "the eval-mode activations are divided by a variance far too large and no "
                "neuron reaches threshold, while TRAINING-mode accuracy looks healthy. "
                "Measured on this code: 16 updates -> 11.7%, 51 -> 57.1%, 96 -> 98.3%. "
                "Full MNIST at batch 256 gives 235 updates per epoch, so even one epoch "
                "clears it. If you are seeing this, you are either training on a "
                "subset, using a very large batch size, or something else is wrong.")
        elif result["spike_rate_mean"] is not None and result["spike_rate_mean"] > 0.9:
            warnings.append(
                f"Mean spike rate {result['spike_rate_mean']:.3f} is close to 1.0: nearly "
                "every neuron fires at every timestep. The network is not sparse and the "
                "energy argument does not apply. Check threshold and normalisation.")
    if acc < 20.0:
        warnings.append(
            f"Accuracy {acc:.1f}% is at or near chance (10%) for 10-class MNIST. "
            "The model did not learn, or eval-mode BatchNorm statistics have not "
            "converged. Check training-mode accuracy: if that is high while this is at "
            "chance, it is the BatchNorm effect and you are training on too few batches. "
            "Do not enter this run into any table.")
    result["warnings"] = warnings
    for w in warnings:
        print("\n  *** WARNING *** " + w + "\n")

    # --- composite indices, at both constants
    result["edp_th"] = edp(result["e_th_uj"], result["latency_cpu_ms"])
    result["edp_si"] = edp(result["e_si_uj"], result["latency_cpu_ms"])
    result["aep_th"] = aep(result["e_th_uj"], acc)
    result["aep_si"] = aep(result["e_si_uj"], acc)
    return result
