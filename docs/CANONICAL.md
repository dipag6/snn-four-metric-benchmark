# SNN-MR604 — CANONICAL code folder (use this one)

Single source of truth for the MR604 four-metric SNN benchmark.
Consolidated 2026-09-03 from `...\MR604\code`, with multi-dataset support added.
Every other copy under `D:\Research\SNN-for-ImageRecognition` is SUPERSEDED —
do not edit or run them. Push THIS folder to GitHub and deploy THIS folder to
Colab and the RTX 5070 laptop.

## What changed vs the old code
- New `--dataset` option:
  - Static (all four metrics, all three paths — ANN / STBP / QCFS):
    `mnist` | `fmnist` (Fashion-MNIST) | `kmnist` (KMNIST).
    All three are 1x28x28, 10 classes, so the network, the operation counts and the
    energy model are IDENTICAL across them — every difference is attributable to the
    data, which is the whole point of the comparison.
  - Event-native (STBP only): `nmist` (N-MNIST). See the N-MNIST section below.
- 5-seed runs use the existing `--seeds` flag (no code change).
- Results/checkpoints/tables/figures are kept per-dataset so they never mix.
- Nothing in the training methods, gates, or energy accounting was touched for the
  static datasets; N-MNIST adds a parallel path, it does not alter the existing one.

## Environment (identical on Colab and the RTX 5070 laptop)
Python 3.10+.
1. Install torch FIRST, matching the machine's CUDA:
   - RTX 5070 is Blackwell — needs CUDA 12.8:
       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
     Verify:
       python -c "import torch;print(torch.cuda.is_available(),torch.cuda.get_device_name(0))"
   - Colab: torch is preinstalled — skip this step.
2. pip install -r requirements.txt
3. python tools/check_install.py
(For N-MNIST only, SpikingJelly may additionally need `pip install scipy`.)

## Run the three static datasets, 5 seeds
Linux / Colab (one dataset per command):
    DATASET=mnist  SEEDS_ENV="42 123 789 456 1337" bash run_gpu.sh
    DATASET=fmnist SEEDS_ENV="42 123 789 456 1337" bash run_gpu.sh
    DATASET=kmnist SEEDS_ENV="42 123 789 456 1337" bash run_gpu.sh

Any platform (Windows/5070 included), per dataset:
    python run.py grid --dataset fmnist --epochs 64 --seeds 42 123 789 456 1337 \
        --out-dir results/fmnist --ckpt-dir checkpoints/fmnist
    python analyse.py --out-dir results/fmnist --tables-dir tables/fmnist --figures-dir figures/fmnist
(repeat with mnist / kmnist)

## N-MNIST (dataset #4) — event-native, STBP-only
N-MNIST is 2x34x34 event data integrated into T frames. It bypasses the spike
encoders (the frames are already spikes) and runs STBP only — conversion (QCFS) and
the ANN baseline are undefined for event streams, so nmist_grid emits STBP T in
{4,8,16} across the 5 seeds and nothing else. The architecture auto-switches to
`ARCH_NMNIST` (in_channels=2, input_hw=34); operation counts and the analytical
energy model are re-derived from that arch inside energy.py, so the four-metric
accounting still holds for the STBP rows.

    python run.py smoke --dataset nmist      # validate first (downloads ~1 GB once, then caches frames)
    python run.py grid --dataset nmist --epochs 64 --seeds 42 123 789 456 1337 \
        --out-dir results/nmist --ckpt-dir checkpoints/nmist

Do NOT run analyse.py on nmist yet — it assumes the static ANN/STBP/QCFS layout.
The N-MNIST result JSONs get their own (STBP-only) tables, built separately.

## Split across two machines WITHOUT muddying latency
Run each dataset ENTIRELY on one machine — never split a single dataset across
Colab and the 5070, or its latency column comes from two GPUs. e.g. Fashion-MNIST
on Colab, KMNIST on the 5070, MNIST on either.

## CIFAR is NOT drop-in
CIFAR changes the input dimensions AND needs a larger network to be meaningful, so it
would force re-deriving the operation counts and the energy baseline (Appendix B) AND
a thesis architecture change. mnist/fmnist/kmnist reuse everything as-is; N-MNIST reuses
the energy machinery via a parameterised arch (STBP-only). CIFAR is a bigger change —
decide before starting, not midway.
