"""Spike-rate regularisation for Path A.

Purpose
-------
Gate G1 rejects STBP T = 16 under the calibrated constant because the fully
connected spiking layer fires at ~0.194 against a threshold of 0.152. The layer's
firing rate is essentially independent of T, while the threshold falls as 1/T, so
the configuration fails by arithmetic rather than by any change in behaviour.

This module adds an optional penalty term to the training objective that pushes
per-layer firing rates below their break-even thresholds. It is a NEW EXPERIMENTAL
ARM, not a correction to the existing results: it changes the trained network, so
its outputs must be reported alongside the unregularised grid rather than replacing
it. Section 7.3 of the thesis already identifies energy-aware training
regularisation as Gap 5.

The penalty is one-sided. A layer already below target contributes nothing, so the
term does not fight the cross-entropy objective where G1 is not at risk.
"""
from __future__ import annotations
import torch

from .config import breakeven_spike_rate, E_AC_SI_PJ


class RatePenalty:
    """Accumulates per-layer firing rates during a forward pass and returns a
    one-sided quadratic penalty on any layer exceeding its target rate.

    Usage
    -----
        pen = RatePenalty(model, T=cfg.T, margin=0.80)
        ...
        pen.reset()
        out = forward_batch(model, x, cfg, dev)
        loss = criterion(out, y) + lam * pen.penalty()
        ...
        pen.remove()          # detach hooks when training finishes

    The hooks fire once per timestep per layer, so the accumulated mean over calls
    is the mean spikes per neuron per timestep -- the same quantity gate G1 tests.
    """

    def __init__(self, model, T, layer_names=("sn1", "sn2", "sn3"),
                 margin=0.80, e_ac_pj=E_AC_SI_PJ, target=None):
        if target is None:
            # default target is a margin inside the binding (calibrated) threshold
            target = margin * breakeven_spike_rate(T, e_ac_pj)
        self.target = float(target)
        self._sum = {}
        self._n = {}
        self._handles = []
        for name in layer_names:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            self._handles.append(mod.register_forward_hook(self._make_hook(name)))
        if not self._handles:
            raise ValueError(
                f"RatePenalty found none of {layer_names} on {type(model).__name__}")

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            # out is the binary spike tensor for this timestep; mean over all
            # elements is the layer's firing rate for this call. Keeps grad_fn,
            # because the surrogate gradient is attached in the backward pass.
            m = out.mean()
            if name in self._sum:
                self._sum[name] = self._sum[name] + m
                self._n[name] += 1
            else:
                self._sum[name] = m
                self._n[name] = 1
        return hook

    def reset(self):
        self._sum, self._n = {}, {}

    def rates(self):
        """Current per-layer mean rate as plain floats (no graph)."""
        return {k: float((self._sum[k] / max(self._n[k], 1)).detach())
                for k in self._sum}

    def penalty(self):
        """One-sided quadratic penalty, summed over layers. Differentiable."""
        total = None
        for k in self._sum:
            rate = self._sum[k] / max(self._n[k], 1)
            excess = torch.clamp(rate - self.target, min=0.0)
            term = excess * excess
            total = term if total is None else total + term
        if total is None:
            return torch.zeros((), dtype=torch.float32)
        return total

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False
