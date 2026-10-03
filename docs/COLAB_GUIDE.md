# Using Google Colab — walkthrough for the MR604 benchmark

Written for someone who has not used Colab before. Follow it in order.
Total setup time before training starts: about 15 minutes.

---

## What Colab actually is

A Jupyter notebook that runs on a Google server instead of your PC. You get a
temporary Linux machine with a GPU attached, free. You interact with it through
your browser.

Two things about it will surprise you, and both matter for a 7-hour job:

- **The machine is temporary.** When your session ends, everything on its local
  disk is deleted. This is why every step below saves to Google Drive.
- **It disconnects on its own.** Idle timeout is around 90 minutes, and the hard
  session cap is 12 hours. You will be disconnected at least once during the
  full grid. The code is built to survive this — see Step 7.

---

## Step 1 — Put the code on Google Drive

1. Go to <https://drive.google.com> and sign in with the Google account you will
   use for Colab. **It must be the same account.**
2. Click **New → Folder**, name it `MR604`.
3. Open `MR604`, then **New → Folder upload**, and select your local
   `D:\Research\SNN-for-ImageRecognition\MR604\code` folder.

When it finishes you should have this on Drive:

```
MyDrive/
└── MR604/
    └── code/
        ├── MR604_Colab.ipynb
        ├── run.py
        ├── analyse.py
        ├── snnbench/
        └── tools/
```

The notebook expects exactly `MyDrive/MR604/code`. If you put it elsewhere,
change the `CODE = ...` line in the notebook to match.

> If **Folder upload** is missing or fails, upload the files individually, but
> keep the `snnbench/` and `tools/` subfolders intact — the imports depend on them.

---

## Step 2 — Open the notebook

### "It opens in VS Code"

That is Windows doing what you told it to. `.ipynb` is registered to VS Code on your
PC, so clicking the file **anywhere on your machine** — including the link in this
chat — hands it to VS Code. The file is fine; you are just opening it in the wrong
place.

Colab runs in a browser. The notebook has to get to Google first. Two ways:

**Option A — upload straight to Colab (fastest, do this to look around)**

1. Go to <https://colab.research.google.com>
2. **File → Upload notebook → Browse**
3. Pick `D:\Research\SNN-for-ImageRecognition\MR604\code\MR604_Colab.ipynb`

It opens immediately. Note this uploads *only the notebook* — `run.py` and the
`snnbench/` package still have to be on Drive (Step 1), or the cells will fail with
`run.py not found`.

**Option B — open from Drive (do this for the real run)**

After Step 1, in Drive right-click `MR604_Colab.ipynb` →
**Open with → Google Colaboratory**.

If Colaboratory is not in that menu: **Open with → Connect more apps**, search
"Colaboratory", **Install**, then try again.

> Do **not** double-click the `.ipynb` on your Windows drive and expect Colab. If you
> want that to change permanently, right-click the file → **Open with → Choose
> another app → Google Chrome → Always**. Not required for any of this.

---

## Step 3 — Turn the GPU on

This is the step people forget, and without it your job takes days instead of
hours.

**Runtime → Change runtime type → Hardware accelerator: GPU → Save.**

Pick **T4** if you are offered a choice (the free option). The first cell of the
notebook asserts a GPU is present and stops with an error if it is not, so you
cannot silently train on CPU.

---

## Step 4 — Running cells

A notebook is a list of cells. Click a cell and press **Shift+Enter** to run it
and move to the next one. A spinner means it is running; a number in brackets
means it finished.

Run the cells **in order, top to bottom.** Do not skip ahead — later cells depend
on variables defined earlier.

The very first time you run a cell you will see a warning that the notebook was
not authored by Google. Click **Run anyway**. Then a Drive permission popup will
appear — choose your account and click **Allow**. This is Colab asking for access
to your own Drive so it can save checkpoints.

---

## Step 4b — Installing SpikingJelly

Colab already has PyTorch with CUDA. The only thing you install is SpikingJelly and
three small helpers. Run this in a cell (the `!` runs a shell command):

```
!pip install -q spikingjelly==0.0.0.0.14 fvcore tqdm tabulate
```

**Pin the version.** The package name is `spikingjelly`, and the version format is
unusual — four dots, `0.0.0.0.14`, not `0.0.0.14`. Getting that wrong gives
`No matching distribution found`. I checked PyPI: `0.0.0.0.14` is the current
release, and the full list is 0.0.0.0.1 through 0.0.0.0.14.

Confirm it worked:

```
!python tools/check_install.py
```

This prints versions, checks the GPU, and — more useful — confirms the specific
SpikingJelly symbols this code calls actually exist and that `ParametricLIFNode`
accepts `v_reset`. It then fires one neuron to prove the forward pass runs.

**Do not use `spikingjelly.__version__`.** It does not exist. The top-level
`spikingjelly` package is empty (`dir()` returns nothing public), so touching that
attribute raises `AttributeError: module 'spikingjelly' has no attribute
'__version__'`. To read the version, use the installed distribution metadata:

```
import importlib.metadata as md
print(md.version('spikingjelly'))     # -> 0.0.0.0.14
```

or from a shell cell, `!pip show spikingjelly`. This works for any pip-installed
package and does not depend on the author having defined `__version__`.

Notes:

- **Do not install PyTorch.** Colab's build is CUDA-matched; `pip install torch`
  can replace it with something that does not see the GPU.
- **You must reinstall after every disconnect.** Installed packages live on the
  temporary machine, not on Drive. This is a 20-second cell, not a problem.
- Ignore `pip` dependency-resolver warnings about unrelated Colab packages. If a
  cell asks you to restart the runtime, do it, then re-run the setup cells.
- If SpikingJelly fails to install or its API has shifted, add `--backend native`
  to every `run.py` command. The native implementation is a ~60-line reimplementation
  of the same neurons and is fully tested — verified here to produce **byte-identical**
  spike trains, spike rates and SOP counts to SpikingJelly through a complete
  train/eval/energy run.

### One SpikingJelly trap, already fixed in this code

`neuron.ParametricLIFNode` defaults to `v_reset=0.0`, which is a **hard** reset. The
thesis specifies a **soft** reset for Path A (Section 3.4). You have to pass
`v_reset=None` explicitly to get soft reset. Omitting it changes the neuron dynamics
silently — nothing errors, the network still trains, and the spike counts are simply
wrong. `snnbench/neurons.py` passes it, and `tools/check_backends.py` is what caught
it: with the argument missing, SpikingJelly emitted 49 spikes where the native
implementation emitted 61.

This is why Step 5 runs the backend check before you train anything.

---

## Step 5 — Verify before training

The notebook's Step 2 cell runs `python run.py verify`. It trains nothing and
takes seconds.

**It must print `verification: PASS`.** If it does not, stop and fix it —
everything downstream depends on those operation counts.

---

## Step 6 — Smoke test, then the real run

Run the smoke test cell (~20 min). **It will look broken and it is not** — Path A
will report ~10% accuracy and zero spikes, because BatchNorm statistics have not
converged after 2 epochs. The notebook explains this where it happens and the
automated check cell tells you whether the *plumbing* is fine, which is what you
are actually testing.

Then run the full grid cell. Roughly 7 hours on a T4.

---

## Step 7 — Surviving disconnects

You **will** get disconnected. Here is what to do about it.

**Keep the tab open and the machine awake.** The ~90-minute idle timeout is
judged on browser activity, not on whether your job is running. A long training
run in a tab you have walked away from can still be killed. So:

- Leave the Colab tab open in the foreground where practical.
- Disable sleep/hibernate on your PC (Windows: Settings → System → Power).
- Check in on it every hour or so.

**When it does disconnect:** reconnect, re-run the setup cells (Drive mount,
`cd` to the code folder, `pip install`), then re-run **the same grid cell**.
Completed runs are skipped and a partly-trained run resumes from its last epoch
checkpoint. You lose at most one epoch.

This only works because checkpoints go to Drive. Do not change the paths to
local ones to make it faster.

---

## Free or Pro?

I checked the current limits rather than guessing.

**Free tier** gives roughly 15–30 hours per week of T4 time, a 12-hour maximum
session, and a ~90-minute idle timeout. Google does not publish exact figures and
says they vary with demand. GPU access is **not guaranteed** — at busy times you
may be given a CPU-only machine or dropped.

Your job needs about 7 GPU-hours, so free is *feasible*, across probably 2–3
sessions. The risk is not the total hours; it is being handed a CPU machine or
disconnected repeatedly mid-run.

**Colab Pro** is about US$10–12/month and includes 100 compute units. A T4 burns
roughly 1.76 units/hour, so 100 units is around 55 T4-hours — far more than you
need. You also get longer sessions and better GPU priority. Unused units carry
over for 90 days.

**My recommendation: buy one month of Pro.** The full grid is ~7 hours and you
may want to re-run configurations after seeing the gate results. Ten dollars to
remove the risk of losing a run at hour six of a thesis experiment is the correct
trade. Cancel after the month.

You do **not** need Pro+ ($50/month) or an A100. An A100 burns ~15 units/hour and
would exhaust Pro's allowance in under 7 hours for a job that a T4 handles fine.

---

## Step 8 — Getting your results back

Everything lands in your Drive under `MR604/`:

```
MR604/
├── results/    one JSON per run — the raw four-metric records
├── tables/     CSVs + thesis_tables.md (paste-ready for Chapter 5)
├── figures/    training curves, accuracy-energy frontier, spike rates
└── checkpoints/ trained weights (large — you can delete these afterwards)
```

Download `tables/` and `figures/` to your PC. Keep `results/` — it is the
evidence behind every number, and Appendix D commits you to releasing it.

---

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `AssertionError: No GPU` | Runtime is CPU | Step 3, then **Runtime → Restart** |
| `run.py not found` | Code uploaded to the wrong Drive folder | Check the path matches `MyDrive/MR604/code`, or edit `CODE =` |
| Drive mount popup never appears | Popups blocked | Allow popups for `colab.research.google.com` |
| `^C` / cell stops with no error | Disconnected | Re-run setup cells, then the same grid cell — it resumes |
| Everything is very slow | Got a CPU machine despite asking for GPU | Runtime → Disconnect and delete runtime, reconnect, check Step 3 again |
| `ModuleNotFoundError: snnbench` | The `cd` cell was not run | Re-run the Step 1 cell that does `os.chdir(CODE)` |
| Out of memory at T=16 | Batch too large for the card | Add `--batch-size 128` to the grid command |
