"""The three instantiations of the shared backbone (Thesis Section 3.2).

  BaselineANN   -- ReLU activations, full precision (or quantised at eval).
  PathA_STBP    -- PLIF neurons + BNTT, trained end-to-end by STBP.
  PathB_QCFS    -- ANN trained with QCFS activations, then converted to IF.

All three share identical layer dimensions and Xavier-uniform initialisation.
Any measured difference is attributable to the paradigm, which is the property
that makes RQ1 answerable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .neurons import make_plif, make_if, NativeIF
from .config import ARCH


def xavier_init(module):
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ==========================================================================
# 1. Baseline ANN
# ==========================================================================
class BaselineANN(nn.Module):
    def __init__(self, a=ARCH):
        super().__init__()
        self.conv1 = nn.Conv2d(a["in_channels"], a["c1"], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(a["c1"])
        self.conv2 = nn.Conv2d(a["c1"], a["c2"], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(a["c2"])
        self.fc1 = nn.Linear(a["c2"] * (a["input_hw"] // 4) ** 2, a["fc_hidden"])
        self.bn3 = nn.BatchNorm1d(a["fc_hidden"])
        self.fc2 = nn.Linear(a["fc_hidden"], a["num_classes"])
        xavier_init(self)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(F.relu(self.bn2(self.conv2(x))), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.bn3(self.fc1(x)))
        return self.fc2(x)


# ==========================================================================
# 2. Path A -- STBP with PLIF + BNTT
# ==========================================================================
class BNTT2d(nn.Module):
    """Batch Normalisation Through Time (Kim & Panda 2021).

    T independent BatchNorm2d modules, indexed by timestep. This is the
    mechanism that restores gradient magnitude across the temporal unrolling
    (Thesis Section 2.4.1).
    """
    def __init__(self, num_features, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features) for _ in range(T)])

    def forward(self, x, t):
        return self.bns[t](x)


class BNTT1d(nn.Module):
    def __init__(self, num_features, T):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_features) for _ in range(T)])

    def forward(self, x, t):
        return self.bns[t](x)


class PathA_STBP(nn.Module):
    def __init__(self, cfg, a=ARCH):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T
        T = cfg.T

        self.conv1 = nn.Conv2d(a["in_channels"], a["c1"], 3, padding=1)
        self.conv2 = nn.Conv2d(a["c1"], a["c2"], 3, padding=1)
        self.fc1 = nn.Linear(a["c2"] * (a["input_hw"] // 4) ** 2, a["fc_hidden"])
        self.fc2 = nn.Linear(a["fc_hidden"], a["num_classes"])

        if cfg.use_bntt:
            self.n1 = BNTT2d(a["c1"], T)
            self.n2 = BNTT2d(a["c2"], T)
            self.n3 = BNTT1d(a["fc_hidden"], T)
        else:
            self.n1 = nn.BatchNorm2d(a["c1"])
            self.n2 = nn.BatchNorm2d(a["c2"])
            self.n3 = nn.BatchNorm1d(a["fc_hidden"])

        self.sn1 = make_plif(cfg)
        self.sn2 = make_plif(cfg)
        self.sn3 = make_plif(cfg)
        xavier_init(self)

    def _norm(self, mod, x, t):
        return mod(x, t) if self.cfg.use_bntt else mod(x)

    def forward(self, spikes):
        """spikes: [T, B, 1, 28, 28] binary input from the encoder.

        Returns the mean output logits over the simulation window
        (TET-style readout, Thesis Section 3.4).
        """
        out = 0.0
        for t in range(self.T):
            x = spikes[t]
            x = self.sn1(self._norm(self.n1, self.conv1(x), t))
            x = F.max_pool2d(x, 2)
            x = self.sn2(self._norm(self.n2, self.conv2(x), t))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.sn3(self._norm(self.n3, self.fc1(x), t))
            out = out + self.fc2(x)
        return out / self.T


# ==========================================================================
# 3. Path B -- QCFS source ANN and its converted SNN
# ==========================================================================
class QCFS(nn.Module):
    """Quantisation-Clip-Floor-Shift activation (Bu et al., ICLR 2022).

        h(x) = (lam / L) * clip( floor( x * L / lam + shift ), 0, L )

    lam is a learnable per-layer threshold; L is the number of quantisation
    levels, which becomes the natural timestep count after conversion. The
    shift term (0.5) is what halves the expected conversion error -- the
    thesis draft omitted it, and it is the reason T = 4 conversion works.

    The gradient uses a straight-through estimator over the clipped range.
    """
    def __init__(self, L=8, shift=0.5, lam_init=8.0):
        super().__init__()
        self.L = L
        self.shift = shift
        self.lam = nn.Parameter(torch.tensor(float(lam_init)))

    def forward(self, x):
        lam = F.relu(self.lam) + 1e-6
        y = x * self.L / lam + self.shift
        y = torch.clamp(y, 0.0, float(self.L))
        # straight-through floor
        y = y + (torch.floor(y) - y).detach()
        return y * lam / self.L


class QCFSSourceANN(nn.Module):
    """The conventional network Path B trains, before conversion."""
    def __init__(self, cfg, a=ARCH):
        super().__init__()
        L, sh = cfg.qcfs_levels, cfg.qcfs_shift
        self.conv1 = nn.Conv2d(a["in_channels"], a["c1"], 3, padding=1)
        self.bn1 = nn.BatchNorm2d(a["c1"])
        self.act1 = QCFS(L, sh)
        self.conv2 = nn.Conv2d(a["c1"], a["c2"], 3, padding=1)
        self.bn2 = nn.BatchNorm2d(a["c2"])
        self.act2 = QCFS(L, sh)
        self.fc1 = nn.Linear(a["c2"] * (a["input_hw"] // 4) ** 2, a["fc_hidden"])
        self.bn3 = nn.BatchNorm1d(a["fc_hidden"])
        self.act3 = QCFS(L, sh)
        self.fc2 = nn.Linear(a["fc_hidden"], a["num_classes"])
        xavier_init(self)

    def forward(self, x):
        x = F.max_pool2d(self.act1(self.bn1(self.conv1(x))), 2)
        x = F.max_pool2d(self.act2(self.bn2(self.conv2(x))), 2)
        x = torch.flatten(x, 1)
        x = self.act3(self.bn3(self.fc1(x)))
        return self.fc2(x)


class PathB_ConvertedSNN(nn.Module):
    """QCFS source ANN with each activation replaced by an IF neuron.

    THRESHOLD FOLDING -- the step that makes conversion work
    -------------------------------------------------------
    A converted IF neuron with threshold lam emits a *binary* spike whose
    effective postsynaptic amplitude is lam, so the quantity the next layer
    should see is lam * s[t], not s[t]. Two ways to arrange that:

      (a) multiply the spike by lam before the next weight layer -- but then the
          next layer's input is no longer binary and it performs MACs, which
          destroys the entire energy argument; or
      (b) fold lam into the next layer's weights and keep the spikes binary.

    We do (b), which is also what hardware does. W_{l+1} <- lam_l * W_{l+1};
    biases are NOT scaled, because a bias is re-added at every timestep and
    therefore survives the 1/T average unchanged.

    Omitting this fold is not a small error: the converted network collapses to
    chance accuracy. Verified on this code -- without folding, a source ANN at
    100% converts to 11.3%.

    The initial membrane potential is lam/2, which is the "shift" of
    Clip-Floor-Shift carried into the spiking domain (Bu et al., ICLR 2022).
    """
    def __init__(self, source: QCFSSourceANN, cfg):
        super().__init__()
        self.cfg = cfg
        self.T = cfg.T

        lam1 = float(F.relu(source.act1.lam).detach().item()) + 1e-6
        lam2 = float(F.relu(source.act2.lam).detach().item()) + 1e-6
        lam3 = float(F.relu(source.act3.lam).detach().item()) + 1e-6
        self.lams = (lam1, lam2, lam3)

        import copy
        self.conv1, self.bn1 = copy.deepcopy(source.conv1), copy.deepcopy(source.bn1)
        self.conv2, self.bn2 = copy.deepcopy(source.conv2), copy.deepcopy(source.bn2)
        self.fc1, self.bn3 = copy.deepcopy(source.fc1), copy.deepcopy(source.bn3)
        self.fc2 = copy.deepcopy(source.fc2)

        with torch.no_grad():
            self.conv2.weight.mul_(lam1)     # sees lam1 * spikes from sn1
            self.fc1.weight.mul_(lam2)       # sees lam2 * spikes from sn2
            self.fc2.weight.mul_(lam3)       # sees lam3 * spikes from sn3

        self.sn1 = make_if(cfg, lam1, cfg.qcfs_shift)
        self.sn2 = make_if(cfg, lam2, cfg.qcfs_shift)
        self.sn3 = make_if(cfg, lam3, cfg.qcfs_shift)

        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, spikes):
        """spikes: [T, B, 1, 28, 28]. Returns accumulated output logits / T."""
        acc = 0.0
        for t in range(self.T):
            x = spikes[t]
            x = self.sn1(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = self.sn2(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = torch.flatten(x, 1)
            x = self.sn3(self.bn3(self.fc1(x)))
            acc = acc + self.fc2(x)
        return acc / self.T



def build_converted(source, cfg):
    """Convert a trained QCFS source to its spiking form, for either backbone.

    run.py should call this rather than PathB_ConvertedSNN directly, so that a
    CIFAR source is converted by the generalised N-layer threshold fold.
    """
    from .config import arch_for
    from .backbones import is_spec
    if is_spec(arch_for(getattr(cfg, "dataset", "mnist"))):
        from .specnets import SpecConvertedSNN
        return SpecConvertedSNN(source, cfg)
    return PathB_ConvertedSNN(source, cfg)


def build_model(cfg):
    from .config import arch_for
    from .backbones import is_spec                  # SPEC_MODEL_DISPATCH
    a = arch_for(getattr(cfg, "dataset", "mnist"))
    if is_spec(a):
        from .specnets import build_spec_model
        return build_spec_model(cfg, a)
    if cfg.pathway == "ann":
        return BaselineANN(a)
    if cfg.pathway == "stbp":
        return PathA_STBP(cfg, a)
    if cfg.pathway == "qcfs":
        return QCFSSourceANN(cfg, a)
    raise ValueError(f"unknown pathway: {cfg.pathway}")
