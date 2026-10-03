# MR604 — Four-Metric SNN Benchmark

Training and evaluation code for the thesis *A Four-Metric Benchmark of STBP Direct
Training and QCFS Conversion for Energy-Efficient Spiking Neural Networks on MNIST*.

Deepak Gajmer (MIT251589) · Melbourne Institute of Technology · MIT licence.

---

## What this produces

Every number that Chapter 5 of the thesis currently marks `[MEASURED VALUE REQUIRED]`.
Specifically, for each of 21 configurations:

| Metric | Where it comes from |
|---|---|
| Top-1 accuracy | Full 10,000-image MNIST test set, no TTA, no ensembling |
| Latency ×3 platforms | GPU @ batch 256, CPU @ batch 1, projected neuromorphic (labelled) |
| Per-layer spike rate | Forward hooks on every neuron layer |
| Energy ×2 constants | Exact measured SOP count × 0.9 pJ and × 1.89 pJ |

Plus the four validity gates (G1–G4) applied automatically, and CSV/Markdown tables
matching Tables 5.4 – 5.11 and 6.1.

---

## Step-by-step: Google Colab

### Step 0 — before you start

Open `MR604_Colab.ipynb` in Colab. Then **Runtime → Change runtime type → GPU**.
Confirm with the first cell. If it says CPU, stop: STBP at T=16 on CPU is a
multi-day job.

Free Colab disconnects after ~4 hours of activity and reclaims the disk. Everything
below checkpoints to Google Drive every epoch, so a disconnect costs you one epoch,
not one run. **Do not skip the Drive mount.**

### Step 1 — install and verify (5 minutes)

```bash
pip install spikingjelly==0.0.0.0.14 fvcore tqdm pandas matplotlib tabulate
python run.py verify
```

`verify` trains nothing. It re-derives the operation counts from the architecture,
cross-checks them against fvcore, and prints the break-even thresholds. It must
print `verification: PASS` before you train anything — every energy figure in the
thesis depends on these counts being right, and this is exactly the check that
would have caught the candidature-stage error.

Expected output:

```
  conv1    MAC=   225,792  params=      320  sites=  25,088
  conv2    MAC= 3,612,672  params=   18,496  sites=  12,544
  fc1      MAC=   401,408  params=  401,536  sites=     128
  fc2      MAC=     1,280  params=    1,290  sites=      10
  TOTAL    MAC= 4,241,152  params=  422,090  sites=  37,770
  fvcore MACs: 4,241,152 (analytical 4,241,152, delta +0)
  verification: PASS

   T   s* @0.9pJ   s* @1.89pJ                    binding?
   2      2.5556       1.2169             neither binding
   4      1.2778       0.6085             calibrated only
   8      0.6389       0.3042                both binding
  16      0.3194       0.1521                both binding

ANN baseline energy: 19.509 uJ/image
```

Then run the backend consistency check once and keep the output:

```bash
python tools/check_backends.py
```

### Step 2 — smoke test (about 20 minutes on a T4)

```bash
python run.py smoke --out-dir /content/drive/MyDrive/MR604/results_smoke \
                    --ckpt-dir /content/drive/MyDrive/MR604/ckpt_smoke
```

Runs all 7 distinct configurations for 2 epochs at one seed.

Accuracy will be below the final figures — 2 epochs is not 64 — but it should look
*sane*: roughly 97–99% for the ANN, and spiking configurations firing normally. The
smoke test is checking plumbing, not numbers:

1. Every spiking configuration produces a JSON containing a `spike_rate_per_layer`
   dict with one entry per neuron layer (the ANN correctly has `null`).
2. `n_sop > 0` for every spiking run — non-zero SOPs prove the monitor is hooked
   even when the neurons are silent, because the encoder still drives conv1.
3. `e_si_uj / e_th_uj` equals exactly **2.10** for every spiking run.
4. The gate report prints and shows PASS/FAIL without crashing.
5. `latency_gpu_ms`, `latency_cpu_ms` and the projected neuromorphic figure are all
   present and distinct.

If any of those five is wrong, fix it now. Discovering it seven hours into the full
grid is the expensive failure mode.

### If you see `ZERO SPIKES` or ~10% accuracy, stop and read this

The harness prints a loud warning in that case. The cause is almost always
unconverged BatchNorm running statistics, and it is worth understanding because it
is invisible in training-mode metrics.

BatchNorm's running variance is initialised at 1.0 and decays toward the true value
as `0.9^k` over `k` updates. At low `k` the eval-mode activations are divided by a
variance far too large; the network's decision boundaries break and it predicts a
constant class, while *training-mode* accuracy looks perfectly healthy. Measured on
this exact code with the plain ANN baseline:

| BatchNorm updates | eval accuracy | train-mode accuracy |
|---|---|---|
| 16 | 11.7% | 89.7% |
| 51 | 57.1% | 95.6% |
| 96 | **98.3%** | 98.1% |

Full MNIST at batch size 256 is **235 updates per epoch**, so a single epoch clears
it and the 2-epoch smoke test (470 updates) is comfortably past it. This affects
every pathway, not just Path A — the table above is the plain ANN.

So on the real dataset you should **never** see this. If you do, you are training on
a subset, using a very large batch size, or something is genuinely broken. Do not
dismiss it as warm-up and do not report the run.

Then wipe the smoke results so they cannot contaminate the real ones:

```bash
rm -rf /content/drive/MyDrive/MR604/results_smoke /content/drive/MyDrive/MR604/ckpt_smoke
```

### Step 3 — the full grid

```bash
python run.py grid --epochs 64 \
    --out-dir /content/drive/MyDrive/MR604/results \
    --ckpt-dir /content/drive/MyDrive/MR604/checkpoints
```

21 runs: 3 ANN + 9 Path A + 9 Path B. **Rerun the identical command after any
disconnect** — completed runs are skipped and partial runs resume from the last
epoch checkpoint.

Rough T4 budget (an A100 is roughly 3× faster):

| Configuration | Per seed | ×3 seeds |
|---|---|---|
| ANN baseline | ~4 min | 12 min |
| QCFS source ANN (T=2,4,8) | ~5 min each | 45 min |
| STBP T=4 | ~18 min | 55 min |
| STBP T=8 | ~35 min | 1 h 45 |
| STBP T=16 | ~70 min | 3 h 30 |
| **Total** | | **≈ 7 hours** |

On free Colab that is 2–3 sessions. Order matters: the driver runs ANN first, then
Path A ascending in T, then Path B, so if you run out of time you still have a
complete Path A story.

If you are short on time, `python run.py priority --epochs 64` gives you
ANN + STBP T=8 + QCFS T=4 across all three seeds in about 2.5 hours — enough for a
defensible three-way comparison — and the full grid can backfill later.

### Step 4 — gates and tables

```bash
python run.py gates --out-dir /content/drive/MyDrive/MR604/results
python analyse.py  --out-dir /content/drive/MyDrive/MR604/results \
                   --tables-dir /content/drive/MyDrive/MR604/tables \
                   --figures-dir /content/drive/MyDrive/MR604/figures
```

`tables/thesis_tables.md` is paste-ready for Chapter 5. Replace each
`[MEASURED VALUE REQUIRED]` cell with the corresponding value.

---

## Local GPU instead of Colab

```bash
conda create -n mr604 python=3.10 -y && conda activate mr604
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
python run.py verify && python run.py smoke && python run.py grid --epochs 64
```

Everything else is identical. Drop `--amp` (`--no-amp`) if you see NaN losses on
older cards.

---

## What each file does

```
run.py                  driver: verify | smoke | grid | priority | one | gates
analyse.py              results/*.json  ->  thesis tables + figures
snnbench/config.py      all constants and the grid definition
snnbench/models.py      BaselineANN, PathA_STBP (PLIF+BNTT), QCFS source + converted SNN
snnbench/neurons.py     PLIF / IF, ATan surrogate; SpikingJelly and native backends
snnbench/data.py        MNIST loaders + rate / direct / TTFS encoders
snnbench/energy.py      operation counting, spike monitor, energy proxy   <- read this one
snnbench/engine.py      training loop, four-metric evaluation, latency timing
snnbench/gates.py       G1-G4
tools/check_backends.py asserts native == SpikingJelly
```

---

## Four things in the code that differ from the thesis draft

I changed these because the draft was wrong or imprecise. **Update the thesis text
to match, or tell me to change the code back — but do not leave them inconsistent.**

**1. SOP counting is exact, not the closed-form approximation.**
The thesis states `N_SOP ≈ s̄ · T · N_MAC`, which assumes every layer fires at the
same rate. That is fine for the analytical surface in Table 5.3 but it is not what
gets measured. The code computes, per weight layer, `(input spikes over all T) ×
(fan-out per input element)` — exact, because every Conv2d and Linear in the spiking
pathways receives a binary tensor. **Action:** in §5.3, label Table 5.3 as an
illustrative closed form under uniform firing, and state that measured figures use
the exact per-layer count.

**2. The QCFS activation in §3.5 is missing the shift term.**
The draft has `QCFS(x) = clip(floor(x/θ)·θ, 0, a_max)` with θ fixed at 1.0. Bu et
al.'s actual function is `h(x) = (λ/L)·clip(floor(x·L/λ + 0.5), 0, L)` with λ
*learnable* per layer. The `+0.5` is the "shift" in Clip-Floor-Shift — it is what
halves the expected conversion error and it is the reason T=4 works at all. Dropping
it makes Path B look far worse than it is. The code implements the correct form,
including the matching membrane initialisation `v(0) = λ/2` after conversion.
**Action:** fix the equation in §3.5 and Appendix A.

**3. "SpikingJelly SOPMonitor" may not exist.**
The thesis cites `SOPMonitor` as the instrumentation in §3.6, §4.4 and Table 3.1.
SpikingJelly's monitor module ships `OutputMonitor`, `InputMonitor`,
`AttributeMonitor` and the gradient monitors; `SOPMonitor` is not reliably present
across versions. The code therefore hooks `nn.Conv2d`/`nn.Linear` directly, which
works on every version and on both backends. **Action:** describe the actual
instrumentation rather than naming a function you may not have used. It is a claim
about your method.

**4. Quantised-baseline energy scaling is an approximation.**
`ann_energy_uj(..., bits=b)` scales per-operation energy by `(b/32)²`. Reduced-
precision energy does not scale purely quadratically — the multiplier is roughly
quadratic, the adder roughly linear, and the memory term differently again.
**Action:** state in §5.4 that the quantised baselines use a first-order quadratic
scaling and cite it as such, or measure them properly on the target.

---

## The result you should be prepared for

At T=16 the calibrated break-even threshold is **0.152 spikes/neuron/timestep**.
Your candidature pilot measured s̄ = 0.121 — that passes, but only by 21%, and that
was a network-mean figure. Gate G1 here is applied **per layer**, and conv2 carries
85% of the operations. If conv2 fires above 0.152, STBP T=16 fails G1 at the
calibrated constant regardless of what the mean says.

There is also a real chance that no SNN configuration beats a 4-bit quantised ANN at
1.89 pJ/SOP. `analyse.py` prints an explicit note if that happens. **That is a valid
answer to RQ1, not a failure of the experiment.** Report it. The thesis is built to
survive a negative result — Chapter 6.2 already states the decision rule that leads
to it — and a benchmark that can only produce one answer was never a benchmark.
