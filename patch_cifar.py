"""Surgical, idempotent patches that wire the CIFAR spec path into snnbench.

Every edit is additive: the MNIST-family code paths are not altered, so every
figure already in the thesis still comes from the code that produced it.
Run from the repo root. Safe to run twice.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "snnbench"
changed, skipped = [], []


def patch(path, anchor, addition, marker, before=False):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    if marker in s:
        skipped.append(f"{p.name}: {marker}")
        return
    if anchor not in s:
        raise SystemExit(f"ANCHOR NOT FOUND in {p}: {anchor[:70]!r}")
    s = s.replace(anchor, (addition + anchor) if before else (anchor + addition), 1)
    p.write_text(s, encoding="utf-8")
    changed.append(f"{p.name}: {marker}")


def sub(path, pattern, repl, marker, count=0):
    p = Path(path)
    s = p.read_text(encoding="utf-8")
    if marker in s:
        skipped.append(f"{p.name}: {marker}")
        return
    s2, n = re.subn(pattern, repl, s, count=count)
    if n == 0:
        raise SystemExit(f"PATTERN NOT FOUND in {p}: {pattern[:70]!r}")
    p.write_text(s2, encoding="utf-8")
    changed.append(f"{p.name}: {marker} ({n} site{'s' if n != 1 else ''})")


# ==========================================================================
# config.py -- CIFAR datasets, architectures, expected counts, grid
# ==========================================================================
patch(PKG / "config.py",
      'DATASETS = {\n    "mnist":  "MNIST",\n    "fmnist": "FashionMNIST",\n    "kmnist": "KMNIST",\n}',
      '''

# CIFAR-10/100 are 3x32x32 and need a deeper backbone; a two-convolution network
# reaches roughly 75 percent on CIFAR-10, which would confound the paradigm
# comparison with simple underfitting. They use the VGG-11 spec in backbones.py,
# shared by all three pathways so the comparison stays matched. See Thesis
# Section 5.17 (external validity) and Section 6.3, Gap 4.
CIFAR_DATASETS = {          # CIFAR_ARCH_BLOCK
    "cifar10":  "CIFAR10",
    "cifar100": "CIFAR100",
}
DATASETS.update(CIFAR_DATASETS)''',
      "# CIFAR_ARCH_BLOCK")

patch(PKG / "config.py",
      'def arch_for(dataset: str) -> dict:',
      '''
def _cifar_arch(dataset: str):
    from .backbones import ARCH_CIFAR10, ARCH_CIFAR100
    return {"cifar10": ARCH_CIFAR10, "cifar100": ARCH_CIFAR100}.get(dataset)

''', "def _cifar_arch", before=True)

sub(PKG / "config.py",
    r'    return ARCH_NMNIST if dataset == "nmist" else ARCH\n',
    '    c = _cifar_arch(dataset)          # CIFAR_ARCH_DISPATCH\n'
    '    if c is not None:\n'
    '        return c\n'
    '    return ARCH_NMNIST if dataset == "nmist" else ARCH\n',
    "CIFAR_ARCH_DISPATCH")

patch(PKG / "config.py",
      'EXPECTED_E_ANN_UJ = 19.509_299     # microjoules per inference, fp32',
      '''

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
}''',
      "EXPECTED_COUNTS")

patch(PKG / "config.py",
      'def priority_grid(epochs: int = 64, **overrides) -> List[RunConfig]:',
      '''
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

''', "def cifar_grid", before=True)

# ==========================================================================
# data.py -- CIFAR loaders
# ==========================================================================
patch(PKG / "data.py",
      'def _nmist_loaders(cfg):',
      '''
def _cifar_loaders(cfg):
    """CIFAR-10/100 with the standard augmentation and, deliberately, NO
    mean/standard-deviation normalisation.

    READ THIS BEFORE ADDING transforms.Normalize
    -------------------------------------------
    encode_rate() treats each pixel as a Bernoulli probability and calls
    torch.bernoulli on it. Normalising to zero mean puts roughly half the input
    below zero, where bernoulli() is undefined -- it raises, or worse, silently
    produces garbage under AMP. The MNIST pipeline relies on the same property.
    Normalisation is instead done by the first BatchNorm, exactly as it is for
    the MNIST family, which keeps the encoder contract and the energy accounting
    identical across every dataset in the study.

    Random crop with 4-pixel padding and horizontal flip are applied because
    without them VGG-11 overfits CIFAR badly. Both preserve the [0, 1] range.
    Augmentation is training-only; there is no test-time augmentation.
    """
    from .config import DATASETS
    ds = cfg.dataset
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),                       # -> [0, 1], no Normalize
    ])
    test_tf = transforms.Compose([transforms.ToTensor()])

    cls = getattr(datasets, DATASETS[ds])
    train = cls(cfg.data_root, train=True, download=True, transform=train_tf)
    test = cls(cfg.data_root, train=False, download=True, transform=test_tf)

    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=False, generator=g,
    )
    test_loader = DataLoader(
        test, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
    )
    return train_loader, test_loader


''', "def _cifar_loaders", before=True)

sub(PKG / "data.py",
    r'    if ds in EVENT_DATASETS:\n        return _nmist_loaders\(cfg\)\n',
    '    if ds in EVENT_DATASETS:\n        return _nmist_loaders(cfg)\n'
    '    from .config import CIFAR_DATASETS          # CIFAR_LOADER_DISPATCH\n'
    '    if ds in CIFAR_DATASETS:\n'
    '        return _cifar_loaders(cfg)\n',
    "CIFAR_LOADER_DISPATCH")

# ==========================================================================
# models.py -- route spec architectures to the spec models
# ==========================================================================
sub(PKG / "models.py",
    r'def build_model\(cfg\):\n    from \.config import arch_for\n    a = arch_for\(getattr\(cfg, "dataset", "mnist"\)\)\n',
    'def build_model(cfg):\n'
    '    from .config import arch_for\n'
    '    from .backbones import is_spec                  # SPEC_MODEL_DISPATCH\n'
    '    a = arch_for(getattr(cfg, "dataset", "mnist"))\n'
    '    if is_spec(a):\n'
    '        from .specnets import build_spec_model\n'
    '        return build_spec_model(cfg, a)\n',
    "SPEC_MODEL_DISPATCH")

patch(PKG / "models.py",
      'def build_model(cfg):',
      '''
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


''', "def build_converted", before=True)

# ==========================================================================
# energy.py -- spec-aware counting, per-dataset verification, monitor names
# ==========================================================================
sub(PKG / "energy.py",
    r'    from \.config import ARCH\n    a = arch or ARCH\n    H = a\["input_hw"\]\n',
    '    from .config import ARCH\n'
    '    from .backbones import is_spec, spec_counts    # SPEC_COUNT_DISPATCH\n'
    '    a = arch or ARCH\n'
    '    if is_spec(a):\n'
    '        return spec_counts(a)\n'
    '    H = a["input_hw"]\n',
    "SPEC_COUNT_DISPATCH")

sub(PKG / "energy.py",
    r'def verify_counts\(model=None, verbose=True\):',
    'def verify_counts(model=None, verbose=True, dataset="mnist"):  # VERIFY_DATASET_ARG',
    "VERIFY_DATASET_ARG")

_VERIFY_TAIL = '''    exp = EXPECTED_COUNTS.get(dataset)
    if exp is None:
        # Nothing pinned for this dataset (N-MNIST): report the derivation and
        # let the caller judge, rather than asserting against MNIST's figures.
        if verbose:
            print("--- operation counts for %s (derived, not pinned) ---" % dataset)
            for k, r in rows.items():
                print("  %-8s MAC=%12s  params=%11s  sites=%9s"
                      % (k, format(r["macs"], ","), format(r["params"], ","),
                         format(r["sites"], ",")))
            print("  %-8s MAC=%12s  params=%11s  sites=%9s"
                  % ("TOTAL", format(total["macs"], ","), format(total["params"], ","),
                     format(total["sites"], ",")))
        return True, rows, total, ["%s: counts derived at runtime, not pinned" % dataset]
    EXPECTED_MACS = exp["macs"]
    EXPECTED_PARAMS = exp["params"]
    EXPECTED_SPIKING_SITES = exp["sites"]
'''

sub(PKG / "energy.py",
    r'    from \.config import EXPECTED_MACS, EXPECTED_PARAMS, EXPECTED_SPIKING_SITES\n'
    r'    rows, total = analytical_counts\(\)\n',
    '    from .config import EXPECTED_COUNTS, arch_for   # VERIFY_PER_DATASET\n'
    '    arch = arch_for(dataset)\n'
    '    rows, total = analytical_counts(arch)\n'
    + _VERIFY_TAIL,
    "VERIFY_PER_DATASET")

# fvcore cross-check is written for the MNIST backbone only; skip it elsewhere.
sub(PKG / "energy.py",
    r'    try:\n        from fvcore\.nn import FlopCountAnalysis\n',
    '    try:\n'
    '        if dataset not in ("mnist", "fmnist", "kmnist"):   # FVCORE_MNIST_ONLY\n'
    '            raise RuntimeError("fvcore cross-check is defined for the "\n'
    '                               "1x28x28 backbone only")\n'
    '        from fvcore.nn import FlopCountAnalysis\n',
    "FVCORE_MNIST_ONLY")

sub(PKG / "energy.py",
    r'    def __init__\(self, model, neuron_names=\("sn1", "sn2", "sn3"\),\n'
    r'                 weight_names=\("conv1", "conv2", "fc1", "fc2"\)\):\n'
    r'        self\.model = model\n',
    '    def __init__(self, model, neuron_names=None, weight_names=None):\n'
    '        # SPEC_MONITOR_NAMES: a spec-built model knows its own topology, so\n'
    '        # ask it rather than assuming the four-layer MNIST backbone.\n'
    '        if neuron_names is None or weight_names is None:\n'
    '            spec = getattr(model, "spec", None)\n'
    '            from .backbones import is_spec\n'
    '            if is_spec(spec):\n'
    '                from .specnets import monitor_names\n'
    '                n_auto, w_auto = monitor_names(spec)\n'
    '            else:\n'
    '                n_auto = ("sn1", "sn2", "sn3")\n'
    '                w_auto = ("conv1", "conv2", "fc1", "fc2")\n'
    '            neuron_names = neuron_names or n_auto\n'
    '            weight_names = weight_names or w_auto\n'
    '        self.model = model\n',
    "SPEC_MONITOR_NAMES")

# ==========================================================================
# run.py -- conversion factory, per-dataset verify, CIFAR grid selection
# ==========================================================================
sub(ROOT / "run.py",
    r'from snnbench\.models import build_model, PathB_ConvertedSNN\n',
    'from snnbench.models import (build_model, PathB_ConvertedSNN,\n'
    '                             build_converted)   # IMPORT_BUILD_CONVERTED\n',
    "IMPORT_BUILD_CONVERTED")

sub(ROOT / "run.py", r'PathB_ConvertedSNN\((source|model), (c2|cfg)\)',
    r'build_converted(\1, \2)   # USE_BUILD_CONVERTED', "USE_BUILD_CONVERTED")

sub(ROOT / "run.py", r'verify_counts\(\)',
    'verify_counts(dataset=args.dataset)   # VERIFY_WITH_DATASET',
    "VERIFY_WITH_DATASET")

# Batch size: 0 means "use this dataset's default" (256 for the MNIST family,
# CIFAR_BATCH for CIFAR), so the CIFAR grid is not silently run at 256.
sub(ROOT / "run.py",
    r'ap\.add_argument\("--batch-size", type=int, default=256\)',
    'ap.add_argument("--batch-size", type=int, default=0,   # BATCH_DEFAULT_BY_DATASET\n'
    '                    help="0 = the default for the chosen dataset "\n'
    '                         "(256 for the MNIST family, 128 for CIFAR)")',
    "BATCH_DEFAULT_BY_DATASET")

sub(ROOT / "run.py",
    r'                  batch_size=args\.batch_size, amp=not args\.no_amp\)',
    '                  batch_size=(args.batch_size or\n'
    '                              (CIFAR_BATCH if args.dataset in CIFAR_DATASETS else 256)),\n'
    '                  amp=not args.no_amp)   # BATCH_RESOLVED',
    "BATCH_RESOLVED")

sub(ROOT / "run.py",
    r'(\s+)elif args\.dataset in EVENT_DATASETS:',
    r'\1elif args.dataset in CIFAR_DATASETS:          # CIFAR_GRID_DISPATCH'
    r'\1    runs = cifar_grid(dataset=args.dataset, epochs=args.epochs, seeds=args.seeds,'
    r'\1                      **{k: v for k, v in common.items() if k != "dataset"})'
    r'\1elif args.dataset in EVENT_DATASETS:',
    "CIFAR_GRID_DISPATCH")

sub(ROOT / "run.py",
    r'from snnbench\.config import \(RunConfig, full_grid, smoke_grid, priority_grid,\n'
    r'                             nmist_grid, EVENT_DATASETS,',
    'from snnbench.config import (RunConfig, full_grid, smoke_grid, priority_grid,\n'
    '                             nmist_grid, EVENT_DATASETS,\n'
    '                             cifar_grid, CIFAR_DATASETS, CIFAR_BATCH,  # CIFAR_IMPORTS',
    "CIFAR_IMPORTS")

sub(ROOT / "run.py",
    r'choices=\["mnist", "fmnist", "kmnist", "nmist"\],',
    'choices=["mnist", "fmnist", "kmnist", "nmist",   # CIFAR_CHOICES\n'
    '                             "cifar10", "cifar100"],',
    "CIFAR_CHOICES")

print("PATCHED:")
for c in changed:
    print("  +", c)
if skipped:
    print("ALREADY PRESENT (skipped):")
    for s in skipped:
        print("  =", s)
print("\nNow run:  python run.py verify --dataset cifar10")
