"""MNIST loaders and the three spike encoders (Thesis Sections 2.8, 3.4)."""
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_loaders(cfg):
    """Loaders for the static datasets (MNIST / Fashion-MNIST / KMNIST) or, for
    N-MNIST, native event frames. No augmentation, no test-time augmentation."""
    from .config import DATASETS, EVENT_DATASETS
    ds = getattr(cfg, "dataset", "mnist")
    if ds in EVENT_DATASETS:
        return _nmist_loaders(cfg)
    from .config import CIFAR_DATASETS          # CIFAR_LOADER_DISPATCH
    if ds in CIFAR_DATASETS:
        return _cifar_loaders(cfg)

    tf = transforms.Compose([
        transforms.ToTensor(),                       # -> [0, 1]
    ])
    cls = getattr(datasets, DATASETS[ds])
    train = cls(cfg.data_root, train=True, download=True, transform=tf)
    test = cls(cfg.data_root, train=False, download=True, transform=tf)

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


def _nmist_loaders(cfg):
    """N-MNIST as T integrated event frames of shape [T, 2, 34, 34], via SpikingJelly.

    Downloads and caches the dataset under cfg.data_root/nmnist on first use
    (~1 GB; the first run also spends time integrating events into frames, then
    caches the frames). frames_number is tied to cfg.T so each T re-frames the
    same events. The frames are binarised to {0,1} in encode() so conv1 sees
    spikes (ACs), keeping the energy accounting identical in kind to the other
    datasets. num_workers is forced to 0 on Windows to avoid the known
    SpikingJelly frame-cache multiprocessing lock-up."""
    import os
    from spikingjelly.datasets.n_mnist import NMNIST
    root = os.path.join(cfg.data_root, "nmnist")
    os.makedirs(root, exist_ok=True)
    common = dict(root=root, data_type="frame", frames_number=cfg.T, split_by="number")
    train = NMNIST(train=True, **common)
    test = NMNIST(train=False, **common)

    nw = 0  # frame cache + Windows: keep workers off to avoid a rare lock-up
    g = torch.Generator()
    g.manual_seed(cfg.seed)
    train_loader = DataLoader(
        train, batch_size=cfg.batch_size, shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=False, generator=g,
    )
    test_loader = DataLoader(
        test, batch_size=cfg.eval_batch_size, shuffle=False,
        num_workers=nw, pin_memory=True,
    )
    return train_loader, test_loader


# --------------------------------------------------------------------------
# Encoders. All return [T, B, C, H, W].
# --------------------------------------------------------------------------
def encode_rate(x, T, generator=None):
    """Bernoulli (Poisson-approximating) rate coding.

    Each pixel fires independently at each timestep with probability equal to
    its normalised intensity. Highest accuracy, highest spike count.
    """
    p = x.clamp(0.0, 1.0).unsqueeze(0).expand(T, *x.shape)
    return torch.bernoulli(p, generator=generator)


def encode_direct(x, T, generator=None):
    """Direct (constant-current) encoding.

    The analogue pixel value is injected unchanged at every timestep. NOTE:
    the input to the first conv is then NOT binary, so the first layer performs
    MACs rather than ACs. energy.py accounts for this -- do not report direct
    encoding energy using the all-AC formula.
    """
    return x.unsqueeze(0).expand(T, *x.shape).contiguous()


def encode_ttfs(x, T, generator=None):
    """Time-to-first-spike encoding.

    Intensity p maps to a single spike at t = floor(T * (1 - p)), clamped to
    [0, T-1]. Zero-intensity pixels never fire. At most one spike per input
    unit per inference, by construction.
    """
    B = x.shape[0]
    p = x.clamp(0.0, 1.0)
    t_idx = torch.floor(T * (1.0 - p)).clamp(0, T - 1).long()
    out = torch.zeros(T, *x.shape, device=x.device, dtype=x.dtype)
    fires = (p > 0).unsqueeze(0)
    onehot = torch.zeros_like(out)
    onehot.scatter_(0, t_idx.unsqueeze(0), 1.0)
    return onehot * fires


ENCODERS = {"rate": encode_rate, "direct": encode_direct, "ttfs": encode_ttfs}


def encode(x, cfg, generator=None):
    # N-MNIST: x is already [B, T, 2, 34, 34] event frames. Binarise (frames may
    # hold event COUNTS > 1) so conv1 receives spikes, and move T to the front to
    # match the [T, B, C, H, W] contract the spiking models expect.
    if getattr(cfg, "dataset", "mnist") == "nmist":
        return (x > 0).to(x.dtype).movedim(1, 0).contiguous()
    return ENCODERS[cfg.encoding](x, cfg.T, generator)


def encoding_is_binary(cfg) -> bool:
    """Whether the first layer receives spikes (AC) or analogue values (MAC)."""
    if getattr(cfg, "dataset", "mnist") == "nmist":
        return True                       # binarised event frames are spikes
    return cfg.encoding in ("rate", "ttfs")
