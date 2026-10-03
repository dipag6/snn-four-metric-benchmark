# Running MR604 on the RTX 5070 (Windows) — stable, no Colab

A local RTX 5070 won't recycle or drop a Drive mount mid-run, so this completes
reliably. Results stay on local disk. Datasets ready: mnist, fmnist, kmnist
(static, all four metrics) and nmist (N-MNIST, event-native, STBP-only — see §5).

## 0. Get the code onto the 5070
Copy the whole SNN-MR604 folder to the 5070 (USB, or download from Google Drive).
Put it anywhere, e.g. C:\SNN-MR604. Only a few MB (code only, no results).

## 1. Python + PyTorch — the one gotcha
Install Python 3.10 or 3.11 (not 3.13/3.14). Then in PowerShell, inside the folder:

    cd C:\SNN-MR604
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

The 5070 is Blackwell (sm_120) — it needs a CUDA 12.8 build or it falls back to CPU:

    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
    pip install -r requirements.txt
    python tools\check_install.py

VERIFY the GPU (must print True and NVIDIA GeForce RTX 5070):

    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

If it prints False, fix the torch install before training.

## 2. Run the three static datasets (5 seeds each)
    .\run_gpu.ps1 -Dataset mnist
    .\run_gpu.ps1 -Dataset fmnist
    .\run_gpu.ps1 -Dataset kmnist

Default seeds: 42 123 789 456 1337. Resume-safe: re-run the same line to continue.
If PowerShell blocks scripts:
    powershell -ExecutionPolicy Bypass -File .\run_gpu.ps1 -Dataset mnist
Or direct, per dataset:
    python run.py grid --dataset mnist --epochs 64 --seeds 42 123 789 456 1337 --out-dir results\mnist --ckpt-dir checkpoints\mnist
    python analyse.py --out-dir results\mnist --tables-dir tables\mnist --figures-dir figures\mnist

## 3. What to expect
- Slow part is STBP T=16 (~1 hr/seed). One dataset a few hours; three ~most of a day, unattended.
- Each static dataset = 50 result rows. No ZERO SPIKES / ~10% accuracy warnings = healthy.
- FutureWarnings about torch.cuda.amp are harmless — ignore.

## 4. When done — send results back
    Compress-Archive -Path results,tables,figures -DestinationPath MR604_5070_results.zip
Send me that zip (or drop it in Drive); I'll verify every row, run the stats, and
build the cross-dataset summary + updated thesis tables.

## 5. N-MNIST (dataset #4) — event-native, STBP-only
N-MNIST is real event-camera data: 2 polarity channels, 34x34, integrated into T
temporal frames. It is NOT a drop-in copy of MNIST — it bypasses the rate/direct/ttfs
encoders (the frames ARE the spikes) and runs STBP only (no ANN baseline, no QCFS
conversion — conversion is defined for static-image ANNs, not event streams). So its
result set is smaller and its table has no ANN/QCFS rows.

FIRST — smoke test it (downloads ~1 GB on first use, then integrates events into
frames and caches them; the first run is slow before any training starts):

    python run.py smoke --dataset nmist

Expect a healthy spiking network (non-zero spikes, accuracy climbing above chance).
If SpikingJelly errors on a missing package during the N-MNIST download/extract, add it:

    pip install scipy

THEN the full grid (STBP, T in {4,8,16}, 5 seeds):

    python run.py grid --dataset nmist --epochs 64 --seeds 42 123 789 456 1337 --out-dir results\nmist --ckpt-dir checkpoints\nmist

Notes:
- Keep results\nmist separate from the static datasets (it already is by the --out-dir above).
- The first-use download + frame integration happens once and is cached under the data root;
  re-runs are fast and resume-safe like the others.
- Do NOT run analyse.py on nmist yet — it assumes the static 4-row-per-config layout with an
  ANN baseline. Send me the raw result JSONs from results\nmist and I'll build the N-MNIST
  tables (STBP-only) to match the thesis.

## Notes
- Run each dataset entirely on the 5070 (consistent latency). Ignore the 15 Colab
  fmnist rows — the 5070 regenerates fmnist cleanly.
