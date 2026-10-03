"""The three pathways, built from a spec (see backbones.py).

These mirror models.py exactly in behaviour -- same initialisation, same
normalisation placement, same readout, same threshold folding -- but read their
topology from a spec instead of hard-coding four layers. models.py is left
untouched so every MNIST-family result in the thesis still comes from the code
that produced it.

Submodules are registered under the names backbones.layer_names() returns
(conv1..convN, fc1, fc2, sn1..snN+1), because SpikeMonitor finds layers with
getattr(model, name). Do not rename them without changing that function.
"""
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import layer_names, spec_counts
from .models import xavier_init, BNTT2d, BNTT1d, QCFS
from .neurons import make_plif, make_if


def _plan(spec):
    """Forward-order plan: one entry per weight layer.

    Each entry is (kind, weight_name, norm_name, neuron_name, pool_after).
    'M' in the spec sets pool_after on the convolution it follows.
    """
    feats = spec["features"]
    plan = []
    conv_i = 0
    for k, item in enumerate(feats):
        if item == 'M':
            continue
        conv_i += 1
        pool_after = (k + 1 < len(feats) and feats[k + 1] == 'M')
        plan.append(("conv", f"conv{conv_i}", f"n{conv_i}", f"sn{conv_i}", pool_after))
    n_fc_norm = conv_i + 1
    plan.append(("fc", "fc1", f"n{n_fc_norm}", f"sn{n_fc_norm}", False))
    plan.append(("out", "fc2", None, None, False))
    return plan


def _build_weights(module, spec, plan):
    """Create the conv/linear layers and attach them under their plan names."""
    _, total = spec_counts(spec)
    c_in = spec["in_channels"]
    hw = spec["input_hw"]
    feats = [f for f in spec["features"] if f != 'M']
    n_pool = sum(1 for f in spec["features"] if f == 'M')

    for (kind, wname, _, _, _), c_out in zip(plan, feats):
        if kind != "conv":
            break
        setattr(module, wname, nn.Conv2d(c_in, int(c_out), 3, padding=1))
        c_in = int(c_out)

    final_hw = spec["input_hw"] // (2 ** n_pool)
    assert final_hw == total["final_hw"], "pool arithmetic disagrees with spec_counts"
    module.fc1 = nn.Linear(c_in * final_hw * final_hw, spec["fc_hidden"])
    module.fc2 = nn.Linear(spec["fc_hidden"], spec["num_classes"])
    return [int(f) for f in feats]


# ==========================================================================
# 1. Baseline ANN
# ==========================================================================
class SpecANN(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.plan = _plan(spec)
        chans = _build_weights(self, spec, self.plan)
        for (kind, _, nname, _, _), c in zip(self.plan, chans):
            if kind != "conv":
                break
            setattr(self, nname, nn.BatchNorm2d(c))
        setattr(self, self.plan[-2][2], nn.BatchNorm1d(spec["fc_hidden"]))
        xavier_init(self)

    def forward(self, x):
        for kind, wname, nname, _, pool in self.plan:
            w = getattr(self, wname)
            if kind == "out":
                return w(x)
            if kind == "fc":
                x = torch.flatten(x, 1)
            x = F.relu(getattr(self, nname)(w(x)))
            if pool:
                x = F.max_pool2d(x, 2)
        raise RuntimeError("plan did not terminate in an output layer")


# ==========================================================================
# 2. Path A -- STBP with PLIF + BNTT
# ==========================================================================
class SpecSTBP(nn.Module):
    def __init__(self, cfg, spec):
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.T = cfg.T
        self.plan = _plan(spec)
        chans = _build_weights(self, spec, self.plan)

        for (kind, _, nname, snname, _), c in zip(self.plan, chans):
            if kind != "conv":
                break
            setattr(self, nname, BNTT2d(c, cfg.T) if cfg.use_bntt else nn.BatchNorm2d(c))
            setattr(self, snname, make_plif(cfg))
        _, _, nfc, snfc, _ = self.plan[-2]
        setattr(self, nfc, BNTT1d(spec["fc_hidden"], cfg.T) if cfg.use_bntt
                else nn.BatchNorm1d(spec["fc_hidden"]))
        setattr(self, snfc, make_plif(cfg))
        xavier_init(self)

    def _norm(self, mod, x, t):
        return mod(x, t) if self.cfg.use_bntt else mod(x)

    def forward(self, spikes):
        """spikes: [T, B, C, H, W]. Returns mean logits over the window."""
        out = 0.0
        for t in range(self.T):
            x = spikes[t]
            for kind, wname, nname, snname, pool in self.plan:
                w = getattr(self, wname)
                if kind == "out":
                    out = out + w(x)
                    break
                if kind == "fc":
                    x = torch.flatten(x, 1)
                x = getattr(self, snname)(self._norm(getattr(self, nname), w(x), t))
                if pool:
                    x = F.max_pool2d(x, 2)
        return out / self.T


# ==========================================================================
# 3. Path B -- QCFS source and its converted SNN
# ==========================================================================
class SpecQCFSSource(nn.Module):
    def __init__(self, cfg, spec):
        super().__init__()
        self.cfg = cfg
        self.spec = spec
        self.plan = _plan(spec)
        chans = _build_weights(self, spec, self.plan)
        L, sh = cfg.qcfs_levels, cfg.qcfs_shift

        for (kind, _, nname, snname, _), c in zip(self.plan, chans):
            if kind != "conv":
                break
            setattr(self, nname, nn.BatchNorm2d(c))
            setattr(self, "act_" + snname, QCFS(L, sh))
        _, _, nfc, snfc, _ = self.plan[-2]
        setattr(self, nfc, nn.BatchNorm1d(spec["fc_hidden"]))
        setattr(self, "act_" + snfc, QCFS(L, sh))
        xavier_init(self)

    def forward(self, x):
        for kind, wname, nname, snname, pool in self.plan:
            w = getattr(self, wname)
            if kind == "out":
                return w(x)
            if kind == "fc":
                x = torch.flatten(x, 1)
            x = getattr(self, "act_" + snname)(getattr(self, nname)(w(x)))
            if pool:
                x = F.max_pool2d(x, 2)
        raise RuntimeError("plan did not terminate in an output layer")


class SpecConvertedSNN(nn.Module):
    """QCFS source with every activation replaced by an IF neuron.

    THRESHOLD FOLDING, generalised to N layers
    ------------------------------------------
    Identical in principle to PathB_ConvertedSNN in models.py, which explains
    why it matters: a converted IF neuron with threshold lam emits a binary
    spike whose effective postsynaptic amplitude is lam, so lam is folded into
    the NEXT weight layer and the spikes stay binary. Keeping them binary is
    what keeps the next layer's operations accumulates rather than multiplies,
    and therefore what keeps the energy argument true.

    Biases are not scaled: a bias is re-added at every timestep and so survives
    the 1/T average unchanged.

    Here the fold runs down the whole stack -- lam from neuron i folds into
    weight layer i+1 -- so with N convolutions there are N+1 neurons and N+1
    fold targets (conv2..convN, fc1, fc2).
    """
    def __init__(self, source: SpecQCFSSource, cfg):
        super().__init__()
        self.cfg = cfg
        self.spec = source.spec
        self.T = cfg.T
        self.plan = source.plan

        for _, wname, nname, _, _ in self.plan:
            setattr(self, wname, copy.deepcopy(getattr(source, wname)))
            if nname is not None:
                setattr(self, nname, copy.deepcopy(getattr(source, nname)))

        lams = []
        for kind, _, _, snname, _ in self.plan:
            if kind == "out":
                break
            lam = float(F.relu(getattr(source, "act_" + snname).lam).detach().item()) + 1e-6
            lams.append(lam)
            setattr(self, snname, make_if(cfg, lam, cfg.qcfs_shift))
        self.lams = tuple(lams)

        # fold lam_i into the weights of layer i+1
        targets = [wname for _, wname, _, _, _ in self.plan][1:]
        assert len(targets) == len(lams), (
            f"fold mismatch: {len(lams)} thresholds, {len(targets)} following layers")
        with torch.no_grad():
            for lam, wname in zip(lams, targets):
                getattr(self, wname).weight.mul_(lam)

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, spikes):
        acc = 0.0
        for t in range(self.T):
            x = spikes[t]
            for kind, wname, nname, snname, pool in self.plan:
                w = getattr(self, wname)
                if kind == "out":
                    acc = acc + w(x)
                    break
                if kind == "fc":
                    x = torch.flatten(x, 1)
                x = getattr(self, snname)(getattr(self, nname)(w(x)))
                if pool:
                    x = F.max_pool2d(x, 2)
        return acc / self.T


def build_spec_model(cfg, spec):
    if cfg.pathway == "ann":
        return SpecANN(spec)
    if cfg.pathway == "stbp":
        return SpecSTBP(cfg, spec)
    if cfg.pathway == "qcfs":
        return SpecQCFSSource(cfg, spec)
    raise ValueError(f"unknown pathway: {cfg.pathway}")


def monitor_names(spec):
    """(neuron_names, weight_names) for SpikeMonitor on a spec model."""
    weights, neurons = layer_names(spec)
    return tuple(neurons), tuple(weights)
