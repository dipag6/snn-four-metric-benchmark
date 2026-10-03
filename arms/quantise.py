"""Post-training quantisation of the conventional baseline.

Purpose
-------
Chapter 6 answers RQ1 conditionally because the quantised baselines have an
analytical energy figure but no measured accuracy. energy.py scales conventional
energy as (bits/32)^2, giving 1.219 microjoules at 8 bits and 0.305 at 4 bits, and
the 4-bit figure is lower than every spiking configuration in the study. Whether
that comparison is meaningful depends on whether a 4-bit network still classifies,
which was never measured.

This module supplies the missing measurement. It is POST-TRAINING quantisation
(PTQ): symmetric per-tensor weights, unsigned per-tensor activations calibrated on
a sample of training batches. It is deliberately the weakest reasonable method,
because a weak quantised baseline is the conservative choice -- if even PTQ retains
accuracy, quantisation-aware training would do better and the conclusion is
strengthened rather than weakened. State the method in the thesis; do not report
the number as if it came from an optimised pipeline.
"""
from __future__ import annotations
import torch
import torch.nn as nn


def quantise_tensor_symmetric(t: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor fake quantisation for signed values (weights)."""
    if bits >= 32:
        return t
    qmax = 2 ** (bits - 1) - 1
    scale = t.detach().abs().max() / qmax
    if scale == 0 or not torch.isfinite(scale):
        return t
    return torch.clamp(torch.round(t / scale), -qmax - 1, qmax) * scale


class ActQuant(nn.Module):
    """Unsigned per-tensor activation fake-quantiser with a calibration mode."""

    def __init__(self, bits: int):
        super().__init__()
        self.bits = bits
        self.calibrating = True
        self.register_buffer("amax", torch.zeros(()))

    def forward(self, x):
        if self.bits >= 32:
            return x
        if self.calibrating:
            self.amax = torch.maximum(self.amax, x.detach().abs().max())
            return x
        qmax = 2 ** self.bits - 1
        scale = self.amax / qmax
        if scale == 0 or not torch.isfinite(scale):
            return x
        return torch.clamp(torch.round(x / scale), 0, qmax) * scale


def quantise_baseline_ann(model: nn.Module, bits: int, calib_loader, device,
                          calib_batches: int = 16):
    """Return a quantised copy of a trained BaselineANN.

    Weights of every Conv2d and Linear are fake-quantised in place on the copy.
    An ActQuant is appended after each such layer and calibrated on
    `calib_batches` batches drawn from `calib_loader`.
    """
    import copy
    q = copy.deepcopy(model).to(device).eval()

    targets = [(n, m) for n, m in q.named_modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))]
    for _, m in targets:
        with torch.no_grad():
            m.weight.copy_(quantise_tensor_symmetric(m.weight.data, bits))
            if m.bias is not None:
                m.bias.copy_(quantise_tensor_symmetric(m.bias.data, bits))

    # attach activation quantisers via forward hooks so the module graph is untouched
    quants = []
    handles = []
    for _, m in targets:
        aq = ActQuant(bits).to(device)
        quants.append(aq)
        handles.append(m.register_forward_hook(lambda _mod, _i, out, aq=aq: aq(out)))

    # calibration pass
    with torch.no_grad():
        for i, (x, _) in enumerate(calib_loader):
            if i >= calib_batches:
                break
            q(x.to(device))
    for aq in quants:
        aq.calibrating = False

    q._quant_handles = handles      # keep alive; caller may remove
    q._quant_modules = quants
    q._quant_bits = bits
    return q
