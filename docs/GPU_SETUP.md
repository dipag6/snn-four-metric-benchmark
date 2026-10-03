# Running the MR604 benchmark on a GPU machine

No Colab, no Drive, no notebook. `run.py` is plain PyTorch and runs anywhere with a
CUDA GPU. Total cost from scratch is about **7 GPU-hours** on a T4; faster hardware
scales roughly linearly.

---

## 1. What to copy across

Copy this whole `code/` directory to the GPU machine. That's all the benchmark needs:

```
code/
  run.py              entry point
  analyse.py          tables and figures
  snnbench/           models, neurons, energy accounting, gates, engine
  tools/              install and backend consistency checks
  requirements.txt
  run_gpu.sh          <- Linux/macOS runner
  run_gpu.ps1         <- Windows runner
```

## 2. Environment

Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1

# Install torch FIRST, matching the machine's CUDA version.
# Get the exact command from https://pytorch.org/get-started/locally/
# Example for CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
python tools/check_install.py      # verifies versions and the SpikingJelly symbols used
```

`requirements.txt` deliberately does not pin torch — pinning it would fight whatever
CUDA build the machine already has.

## 3. Carry over the runs that are already done (optional but recommended)

Nine of the twenty-one runs completed on 2026-08-04 and are sitting in Google Drive:
`MyDrive/MR604/results/` and `MyDrive/MR604/checkpoints/`.

Download both folders and drop their contents into `code/results/` and
`code/checkpoints/`. `run.py` skips any configuration whose result `.json` already
exists, so those nine are not retrained and you go straight to the remaining twelve —
roughly **4.3 GPU-hours instead of 7**.

Already finished: `ann_s{42,123,789}`, `stbp_T4_s{42,123,789}`, `stbp_T8_s{42,123,789}`.

Skip this step if you would rather regenerate everything on one consistent machine.
That is the cleaner choice scientifically — mixing hardware across a results table is
defensible for accuracy but muddies the latency columns, which are hardware-dependent.
**If you keep the Colab results, the GPU and CPU latency figures in the final table will
come from two different machines. Say so in the thesis, or regenerate all 21 runs.**

## 4. Run it

```bash
bash run_gpu.sh                    # Linux/macOS
```

```powershell
.\run_gpu.ps1                      # Windows
```

Both scripts:

- refuse to start without a CUDA GPU (CPU is ~100× slower — not a fallback here)
- run `verify` first and abort if the operation counts disagree, since every energy
  number in Chapter 5 depends on them
- fetch and validate MNIST
- train all 21 configurations, **ordered cheapest-first by result rows per GPU-hour**
- keep going if one configuration fails, rather than losing the session
- finish with the validity gates, `thesis_tables.md`, and the three figures

Useful overrides:

```bash
EPOCHS=8 bash run_gpu.sh           # quick sanity pass, NOT thesis numbers
FORCE=1  bash run_gpu.sh           # retrain everything, ignoring existing results
RESULTS=/scratch/mr604/results bash run_gpu.sh
```

```powershell
.\run_gpu.ps1 -Epochs 8
.\run_gpu.ps1 -Force
.\run_gpu.ps1 -Results D:\scratch\mr604\results
```

## 5. Why this ordering

The default `run.py grid` trains STBP T=16 early. On an interruptible machine that is
the wrong way round:

| Block | Runs | Approx. GPU time | Result rows |
|---|---|---|---|
| ANN baseline | 3 | ~12 min | 3 |
| QCFS T=2, 4, 8 | 9 | ~45 min | **18** (two encodings per run) |
| STBP T=4 | 3 | ~55 min | 3 |
| STBP T=8 | 3 | ~1 h 45 | 3 |
| STBP T=16 | 3 | ~3 h 30 | 3 |

The nine QCFS runs produce 18 of the 30 rows in about 45 minutes. STBP T=16 produces 3
rows in three and a half hours. The scripts do the cheap, high-yield work first so that
a killed session costs the tail.

Interrupting is safe regardless: checkpoints are written every epoch and re-running the
script resumes from the last one.

## 6. If a job scheduler is involved (Slurm)

```bash
#!/bin/bash
#SBATCH --job-name=mr604
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --output=mr604_%j.log

module load cuda                   # site-specific
source /path/to/.venv/bin/activate
bash /path/to/code/run_gpu.sh
```

Eight hours covers a full run from scratch with margin. If the site caps wall time
lower, submit the same script repeatedly — it resumes.

## 7. Two things to check in the output

**The direct-encoding conversion loss.** The Colab notebook's prose claims 0 pp loss at
T=4 and T=8 under direct encoding. The 2-epoch smoke test gave 19.13 pp and 24.36 pp.
Whether that is just an undertrained QCFS source at 2 epochs is still unverified. Check
it at 64 epochs before repeating the 0 pp claim anywhere in the thesis.

**STBP T=16 against gate G1.** The calibrated break-even threshold at T=16 is 0.152, G1
is applied per layer, and conv2 carries 85.2% of all operations. If conv2 fires above
0.152, T=16 fails G1 at the silicon constant regardless of the network mean. The
candidature pilot's network mean of 0.121 does not settle this.

## 8. Energy-ratio note

If anyone asks why `e_si / e_th` is not exactly 2.10 (= 1.89 / 0.9 pJ per SOP) for the
direct-encoded QCFS rows: `snn_energy_uj` is
`(n_sop * e_ac_pj + n_mac_residue * E_MAC_PJ) / 1e6`, and the MAC residue term uses
`E_MAC_PJ` (4.6 pJ) identically in both variants. The 2.10 identity therefore only holds
when `n_mac_residue == 0`. Direct encoding carries residue by design — conv1 runs MACs at
every timestep. The accounting is correct; a smoke-test assertion that ignored this was
the thing at fault and has been corrected in the notebook.
