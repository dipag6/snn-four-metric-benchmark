# snn-four-metric-benchmark

Code and result files for the MR604 Master of ICT Research thesis
*Spiking Neural Networks for Image Recognition*
(Deepak Gajmer, MIT251589, Melbourne Institute of Technology, 2026).

The benchmark (U4M-T) reports four metrics for every configuration (top-1 accuracy, latency per platform, per-layer spike rate, and energy at 0.9 pJ and 1.89 pJ per synaptic operation) and applies four validity gates before any energy figure is used:

| Gate | Test |
|---|---|
| G1 | every spiking layer satisfies s_l < (E_MAC / E_AC) / T at both constants |
| G2 | seed standard deviation of accuracy < 1 pp (3 seeds, escalate to 5) |
| G3 | GPU and CPU latency reported separately, never aggregated |
| G4 | every SNN energy figure reported at both constants |

## Layout

| Path | Contents |
|---|---|
| `snnbench/` | models, neurons (PLIF, IF), QCFS, encoders, energy model, gates, evaluation engine (incl. gated max pooling) |
| `run.py`, `analyse.py` | training/evaluation grid and table/figure generation |
| `arms/` | diagnostic arms: post-training quantisation, spike-rate regularisation, G2 escalation driver |
| `tools/` | install checks and the instrumentation verification script (predicts the 75,520-MAC fvcore residue) |
| `MR604_CIFAR_Colab.ipynb`, `patch_cifar.py` | CIFAR-10/100 VGG-11 extension |
| `results/` | one JSON per configuration and seed, as used in the thesis |
| `docs/` | step-by-step run guides (Colab, RTX 5070) |

## Result sets and the thesis tables they support

| Folder | Platform | Thesis |
|---|---|---|
| `results/primary_rtx5070` | RTX 5070 Laptop | Tables 4.1–4.4, F.1–F.3, F.14 |
| `results/arms_rtx5070` | RTX 5070 Laptop | Table 4.1 quantised rows, Section 4.3 escalation (`_quant_escalation.json`), Tables F.11, F.12 |
| `results/replication_a100` | A100-SXM4-40GB | Tables F.5–F.8, F.16 |
| `results/multidataset_rtx5070` | RTX 5070 Laptop (corrected pass) | Tables 4.7–4.9 |
| `results/gatedpool_mnistfamily_l4` | L4, inference only | Table 4.5 |
| `results/cifar10_l4`, `results/cifar100_l4` | L4 (Colab) | Tables 4.10, 4.11, F.17 |

The NVIDIA T4 result sets and the quarantined 17 August run listed in the thesis (Appendix E) are not in this repository.

Check value: the mean `edp_th` of `results/primary_rtx5070/stbp_T4_s*.json` is 6.9968 (thesis Table F.3: 6.997).

Trained weights are not in the git tree. The MNIST checkpoints of the primary grid, the A100 replication and the diagnostic arms are assets of the [v1.0 release](https://github.com/dipag6/snn-four-metric-benchmark/releases/tag/v1.0); its `MANIFEST.md` maps each checkpoint to its result file and lists what the release does not contain.

## Install

```
pip install torch torchvision          # pick the CUDA build for your GPU (RTX 50-series needs cu128)
pip install -r requirements.txt
python tools/check_install.py
python tools/verify_instrumentation.py
```

See `docs/RUNNING.md` for the full grid commands.

## Licence

MIT, see `LICENSE`.
