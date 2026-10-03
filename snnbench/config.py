"""Central configuration for the MR604 four-metric SNN benchmark.

Every number here corresponds to a value stated in the thesis. If you change
one, change it in the thesis too -- that is the whole point of the exercise.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional

# --------------------------------------------------------------------------
# Energy constants (Thesis Section 2.6, Table 2.3)
# --------------------------------------------------------------------------
E_MAC_PJ = 4.6      # Horowitz 2014, 45 nm CMOS, 32-bit MAC
E_AC_TH_PJ = 0.9    # Horowitz 2014, 45 nm CMOS, accumulate
E_AC_SI_PJ = 1.89   # Tan & Wu 2023, measured on fabricated 40 nm SNN core

# Break-even ratios
RATIO_TH = E_MAC_PJ / E_AC_TH_PJ    # 5.1111
RATIO_SI = E_MAC_PJ / E_AC_SI_PJ    # 2.4339


def breakeven_spike_rate(T: int, e_ac_pj: float) -> float:
    """Gate G1 threshold: s* = (E_MAC / E_AC) / T   (Thesis Appendix B.5)."""
    return (E_MAC_PJ / e_ac_pj) / T


# --------------------------------------------------------------------------
# Architecture (Thesis Section 3.3)
# --------------------------------------------------------------------------
ARCH = dict(
    in_channels=1,
    c1=32,
    c2=64,
    fc_hidden=128,
    num_classes=10,
    input_hw=28,
)

# N-MNIST is NATIVE EVENT DATA: 2 polarity channels, 34x34, and it bypasses the
# encoders (the integrated event frames ARE the spikes). Different input geometry
# => its operation counts and energy baseline are re-derived from this arch.
ARCH_NMNIST = dict(
    in_channels=2,
    c1=32,
    c2=64,
    fc_hidden=128,
    num_classes=10,
    input_hw=34,
)



def _cifar_arch(dataset: str):
    from .backbones import ARCH_CIFAR10, ARCH_CIFAR100
    return {"cifar10": ARCH_CIFAR10, "cifar100": ARCH_CIFAR100}.get(dataset)

def arch_for(dataset: str) -> dict:
    """Architecture dict for a dataset. Static 28x28x1 sets share ARCH;
    N-MNIST uses ARCH_NMNIST (2x34x34). Same c1/c2/fc_hidden throughout, so only
    the input geometry differs -- the network topology is unchanged."""
    c = _cifar_arch(dataset)          # CIFAR_ARCH_DISPATCH
    if c is not None:
        return c
    return ARCH_NMNIST if dataset == "nmist" else ARCH

# Datasets. All three are 1x28x28, 10 classes, so the architecture, the operation
# counts and the whole energy model are IDENTICAL across them -- every difference
# between datasets is attributable to the data, not the network.
DATASETS = {
    "mnist":  "MNIST",
    "fmnist": "FashionMNIST",
    "kmnist": "KMNIST",
}

# CIFAR-10/100 are 3x32x32 and need a deeper backbone; a two-convolution network
# reaches roughly 75 percent on CIFAR-10, which would confound the paradigm
# comparison with simple underfitting. They use the VGG-11 spec in backbones.py,
# shared by all three pathways so the comparison stays matched. See Thesis
# Section 5.17 (external validity) and Section 6.3, Gap 4.
CIFAR_DATASETS = {          # CIFAR_ARCH_BLOCK
    "cifar10":  "CIFAR10",
    "cifar100": "CIFAR100",
}
DATASETS.update(CIFAR_DATASETS)

# Event datasets bypass the encoders (frames are already spikes) and are STBP-only
# (ANN/QCFS need a static image). N-MNIST is loaded via SpikingJelly, not torchvision.
EVENT_DATASETS = {"nmist"}

# Verified analytical counts (Thesis Table 5.1 / Appendix B).
# run.py re-derives these at runtime and asserts agreement.
EXPECTED_MACS = 4_241_152
EXPECTED_PARAMS = 422_090
EXPECTED_SPIKING_SITES = 37_770
EXPECTED_E_ANN_UJ = 19.509_299     # microjoules per inference, fp32

# Per-dataset expected counts. The MNIST-family entry is the published one;
# the CIFAR entries are derived by backbones.spec_counts(), whose agreement with
# these MNIST figures is asserted by backbones.mnist_spec_matches_legacy().
# run.py verify re-derives all of them at runtime and asserts agreement.
EXPECTED_COUNTS = {
    "mnist":    dict(macs=4_241_152,   params=422_090,   sites=37_770),
    "fmnist":   dict(macs=4_241_152,   params=422_090,   sites=37_770),
    "kmnist":   dict(macs=4_241_152,   params=422_090,   sites=37_770),
    "nmist":    None,                  # 2x34x34: derived at runtime, not pinned
    "cifar10":  dict(macs=153_031_680, params=9_494_794, sites=152_074),
    "cifar100": dict(macs=153_077_760, params=9_540_964, sites=152_164),
}

# ANN energy baselines, microjoules per inference at 45 nm fp32 (N_MAC * 4.6 pJ).
EXPECTED_E_ANN_UJ_BY_DATASET = {
    "mnist": 19.509_299, "fmnist": 19.509_299, "kmnist": 19.509_299,
    "cifar10": 703.945_728, "cifar100": 704.157_696,
}


# --------------------------------------------------------------------------
# Training (Thesis Section 3.3 / Appendix A)
# --------------------------------------------------------------------------
@dataclass
class RunConfig:
    # identity
    pathway: str = "ann"            # "ann" | "stbp" | "qcfs"
    T: int = 0                      # 0 for ANN
    seed: int = 42
    dataset: str = "mnist"          # "mnist" | "fmnist" | "kmnist" | "nmist" (STBP-only)

    # optimisation
    epochs: int = 64
    batch_size: int = 256
    eval_batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 0.0
    cosine_t_max: Optional[int] = None   # defaults to epochs
    # Gradient-norm clip. 0 = off. CIFAR STBP defaults to 5.0: without it the
    # T = 16 run's input-layer gradient exploded (~5e14), went non-finite under
    # AMP and froze training from about epoch 6. Healthy T = 8 norms are ~1, so
    # the clip is meant to be inactive except on a blow-up; clip_frac is logged.
    grad_clip: float = 0.0

    # Path A (STBP)
    plif_tau_init: float = 0.25
    v_threshold: float = 1.0
    atan_alpha: float = 2.0
    use_bntt: bool = True
    encoding: str = "rate"          # "rate" | "direct" | "ttfs"

    # Path B (QCFS)
    qcfs_levels: int = 8            # L in Bu et al.; also the natural T
    qcfs_shift: float = 0.5

    # infrastructure
    backend: str = "spikingjelly"   # "spikingjelly" | "native"
    amp: bool = True
    num_workers: int = 2
    data_root: str = "./data"
    out_dir: str = "./results"
    ckpt_dir: str = "./checkpoints"
    device: str = "cuda"
    cpu_latency_reps: int = 200     # thesis specifies 1000; lower for Colab
    gpu_latency_reps: int = 1000
    latency_warmup: int = 100

    def __post_init__(self):
        if (self.grad_clip == 0.0 and self.pathway == 'stbp'
                and self.dataset in ('cifar10', 'cifar100')):
            self.grad_clip = 5.0
        if self.cosine_t_max is None:
            self.cosine_t_max = self.epochs

    @property
    def tag(self) -> str:
        if self.pathway == "ann":
            return f"ann_s{self.seed}"
        return f"{self.pathway}_T{self.T}_s{self.seed}"

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# The grid (Thesis Sections 3.4, 3.5)
# --------------------------------------------------------------------------
SEEDS = [42, 123, 789]
SEEDS_ESCALATED = [42, 123, 789, 456, 1337]   # used if gate G2 fails
STBP_TIMESTEPS = [4, 8, 16]
QCFS_TIMESTEPS = [2, 4, 8]


def full_grid(epochs: int = 64, seeds: List[int] = None, **overrides) -> List[RunConfig]:
    """The 21-run grid: 3 ANN + 9 Path A + 9 Path B."""
    seeds = seeds or SEEDS
    runs = []
    for s in seeds:
        runs.append(RunConfig(pathway="ann", T=0, seed=s, epochs=epochs, **overrides))
    for T in STBP_TIMESTEPS:
        for s in seeds:
            runs.append(RunConfig(pathway="stbp", T=T, seed=s, epochs=epochs, **overrides))
    for T in QCFS_TIMESTEPS:
        for s in seeds:
            runs.append(RunConfig(pathway="qcfs", T=T, seed=s, epochs=epochs,
                                  qcfs_levels=T, **overrides))
    return runs


def smoke_grid(**overrides) -> List[RunConfig]:
    """Stage 1: every configuration, 2 epochs, one seed. Proves the pipeline."""
    ov = dict(epochs=2, **overrides)
    if ov.get("dataset") in EVENT_DATASETS:
        # Event data is STBP-only.
        return [RunConfig(pathway="stbp", T=T, seed=42, **ov) for T in STBP_TIMESTEPS]
    runs = [RunConfig(pathway="ann", T=0, seed=42, **ov)]
    for T in STBP_TIMESTEPS:
        runs.append(RunConfig(pathway="stbp", T=T, seed=42, **ov))
    for T in QCFS_TIMESTEPS:
        runs.append(RunConfig(pathway="qcfs", T=T, seed=42, qcfs_levels=T, **ov))
    return runs


def nmist_grid(epochs: int = 64, seeds: List[int] = None, **overrides) -> List[RunConfig]:
    """N-MNIST grid: STBP direct-training on native event frames only, at
    T in {4, 8, 16}. ANN/QCFS need a static image and do not apply to event data,
    so N-MNIST is reported as a complementary event-data result (15 runs at 5 seeds).
    The ANN energy baseline is still available analytically (2x34x34 MAC count),
    so the four metrics and the validity gates are all computed."""
    seeds = seeds or SEEDS
    overrides.setdefault("dataset", "nmist")
    runs = []
    for T in STBP_TIMESTEPS:
        for s in seeds:
            runs.append(RunConfig(pathway="stbp", T=T, seed=s, epochs=epochs, **overrides))
    return runs



# CIFAR defaults. 64 epochs is enough for MNIST and is not enough for CIFAR;
# 120 is the smallest budget at which VGG-11 results are worth reporting.
# Batch size is held at 128 for EVERY CIFAR run, including T = 16: varying it
# with T would change the effective learning rate and stop the timestep sweep
# being a controlled comparison. If T = 16 will not fit, lower it for the whole
# grid and re-run, do not lower it for one arm.
CIFAR_EPOCHS = 120
CIFAR_BATCH = 128


def cifar_grid(dataset: str = "cifar10", epochs: int = CIFAR_EPOCHS,
               seeds: List[int] = None, **overrides) -> List[RunConfig]:
    """The 21-run grid on CIFAR: 3 ANN + 9 Path A + 9 Path B, VGG-11 backbone."""
    if dataset not in CIFAR_DATASETS:
        raise ValueError(f"{dataset} is not a CIFAR dataset")
    overrides.setdefault("dataset", dataset)
    overrides.setdefault("batch_size", CIFAR_BATCH)
    overrides.setdefault("eval_batch_size", CIFAR_BATCH)
    return full_grid(epochs=epochs, seeds=seeds, **overrides)


def cifar_smoke_grid(dataset: str = "cifar10", **overrides) -> List[RunConfig]:
    """Two epochs of every CIFAR configuration. Proves the pipeline before the
    grid is allowed to spend GPU hours."""
    overrides.setdefault("dataset", dataset)
    overrides.setdefault("batch_size", CIFAR_BATCH)
    overrides.setdefault("eval_batch_size", CIFAR_BATCH)
    return smoke_grid(**overrides)

def priority_grid(epochs: int = 64, **overrides) -> List[RunConfig]:
    """A defensible three-way comparison, run first if time is short."""
    runs = []
    for s in SEEDS:
        runs.append(RunConfig(pathway="ann", T=0, seed=s, epochs=epochs, **overrides))
        runs.append(RunConfig(pathway="stbp", T=8, seed=s, epochs=epochs, **overrides))
        runs.append(RunConfig(pathway="qcfs", T=4, seed=s, epochs=epochs,
                              qcfs_levels=4, **overrides))
    return runs


# --------------------------------------------------------------------------
# Deployment profiles (Thesis Section 6.4, RQ3)
# --------------------------------------------------------------------------
DEPLOYMENT_PROFILES = [
    dict(name="ultra_low_power", power_mw=10,  acc_floor=95.0),
    dict(name="balanced",        power_mw=50,  acc_floor=98.0),
    dict(name="high_accuracy",   power_mw=100, acc_floor=99.0),
]
ASSUMED_INFERENCE_RATE_HZ = 10   # stated explicitly in Thesis Section 6.4
