#!/usr/bin/env python
"""Verify that the native neuron implementations agree with SpikingJelly.

Run this ONCE and keep the output. If a reviewer asks whether your results are
an artefact of the framework, this is the answer. If the two disagree, do not
proceed -- one of them is wrong and you need to know which before you train
anything.

  python tools/check_backends.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from snnbench.config import RunConfig
from snnbench.neurons import NativePLIF, NativeIF


def check_plif(T=16, N=4, C=8, tol=1e-5):
    try:
        from spikingjelly.activation_based import neuron, surrogate
    except ImportError:
        print("SpikingJelly not installed -- skipping PLIF cross-check")
        return None

    torch.manual_seed(0)
    # Scale so the neuron actually fires. With init_tau=4 the decay is 0.25 and the
    # steady-state membrane std is ~0.38x the input std, so an input std of 0.5 leaves
    # the threshold ~5 sigma away and NOTHING spikes -- making "identical" vacuously
    # true. The comparison is only meaningful if both backends emit spikes.
    x = torch.randn(T, N, C) * 3.0

    sj = neuron.ParametricLIFNode(init_tau=4.0, v_threshold=1.0, v_reset=None,
                                  surrogate_function=surrogate.ATan(alpha=2.0),
                                  detach_reset=True, step_mode="s")
    nat = NativePLIF(tau_init=0.25, v_threshold=1.0, alpha=2.0)
    # align the learnable parameter
    with torch.no_grad():
        nat.w.copy_(sj.w.detach().reshape(()))

    out_sj, out_nat = [], []
    for t in range(T):
        out_sj.append(sj(x[t]))
        out_nat.append(nat(x[t]))
    a = torch.stack(out_sj); b = torch.stack(out_nat)
    same = torch.equal(a, b)
    diff = (a - b).abs().max().item()
    fired = a.sum().item() > 0
    print(f"PLIF  spike trains identical: {same}   max abs diff: {diff:.2e}   "
          f"spike counts sj={a.sum().item():.0f} native={b.sum().item():.0f}")
    if not fired:
        print("      *** VACUOUS: neither backend spiked, so 'identical' proves nothing. "
              "Increase the input scale. ***")
        return False
    return same


def check_if(T=8, N=4, C=8):
    try:
        from spikingjelly.activation_based import neuron, surrogate
    except ImportError:
        print("SpikingJelly not installed -- skipping IF cross-check")
        return None
    torch.manual_seed(0)
    x = torch.rand(T, N, C) * 0.4

    sj = neuron.IFNode(v_threshold=1.0, v_reset=None,
                       surrogate_function=surrogate.ATan(alpha=2.0), step_mode="s")
    sj.v = 0.5
    nat = NativeIF(v_threshold=1.0, v_init_frac=0.5, alpha=2.0)
    a = torch.stack([sj(x[t]) for t in range(T)])
    b = torch.stack([nat(x[t]) for t in range(T)])
    same = torch.equal(a, b)
    fired = a.sum().item() > 0
    print(f"IF    spike trains identical: {same}   "
          f"spike counts sj={a.sum().item():.0f} native={b.sum().item():.0f}")
    if not fired:
        print("      *** VACUOUS: neither backend spiked. ***")
        return False
    return same


def check_surrogate_gradient():
    """The ATan surrogate derivative at threshold should be alpha/2 = 1.0."""
    from snnbench.neurons import atan_spike
    x = torch.zeros(1, requires_grad=True)
    y = atan_spike(x, alpha=2.0)
    y.backward()
    print(f"ATan surrogate derivative at threshold: {x.grad.item():.6f} "
          f"(expected 1.000000 for alpha=2.0)")
    return abs(x.grad.item() - 1.0) < 1e-6


if __name__ == "__main__":
    print("=" * 70)
    print("BACKEND CONSISTENCY CHECK")
    print("=" * 70)
    r = [check_surrogate_gradient(), check_plif(), check_if()]
    r = [x for x in r if x is not None]
    print("=" * 70)
    print("RESULT:", "PASS" if all(r) else "FAIL -- investigate before training")
    sys.exit(0 if all(r) else 1)
