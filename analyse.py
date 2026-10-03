#!/usr/bin/env python
"""Turns results/*.json into the thesis tables and figures.

  python analyse.py --out-dir ./results --tables-dir ./tables

Produces:
  tables/table_5_4_baseline.csv        Conventional baselines
  tables/table_5_5_pathA.csv           Path A by timestep
  tables/table_5_6_pathA_layers.csv    Path A per-layer spike rates + G1
  tables/table_5_7_pathB.csv           Path B by timestep
  tables/table_5_8_pathB_layers.csv    Path B per-layer spike rates + G1
  tables/table_5_9_unified.csv         THE central four-metric table
  tables/table_5_10_gates.csv          Gate outcomes
  tables/table_5_11_composite.csv      EDP / AEP rankings
  tables/table_6_1_deployment.csv      Deployment recommendations
  tables/thesis_tables.md              All of the above, paste-ready
  figures/fig_training_curves.png
  figures/fig_accuracy_energy.png      the accuracy-energy frontier
  figures/fig_spike_rates.png          per-layer rates against G1 thresholds
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from snnbench.config import (E_AC_TH_PJ, E_AC_SI_PJ, breakeven_spike_rate,
                             DEPLOYMENT_PROFILES, ASSUMED_INFERENCE_RATE_HZ)
from snnbench.gates import apply_gates


def load(out_dir):
    res = []
    for f in sorted(Path(out_dir).glob("*.json")):
        if f.name.startswith("_"):
            continue
        res.append(json.load(open(f)))
    if not res:
        raise SystemExit(f"no results found in {out_dir}")
    return res


def cfg_label(r):
    return "ANN fp32" if r["pathway"] == "ann" else f"{r['pathway'].upper()} T={r['T']}"


def agg(results):
    """Mean and s.d. across seeds for each (pathway, T)."""
    df = pd.DataFrame([{k: v for k, v in r.items()
                        if not isinstance(v, (dict, list))} for r in results])
    g = df.groupby(["pathway", "T"], as_index=False).agg(
        n_seeds=("seed", "count"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_sd=("accuracy", "std"),
        spike_rate_mean=("spike_rate_mean", "mean"),
        n_sop=("n_sop", "mean"),
        e_th_uj=("e_th_uj", "mean"),
        e_si_uj=("e_si_uj", "mean"),
        latency_gpu_ms=("latency_gpu_ms", "mean"),
        latency_cpu_ms=("latency_cpu_ms", "mean"),
        edp_th=("edp_th", "mean"),
        edp_si=("edp_si", "mean"),
        aep_th=("aep_th", "mean"),
        aep_si=("aep_si", "mean"),
    )
    g["accuracy_sd"] = g["accuracy_sd"].fillna(0.0)
    g["config"] = g.apply(lambda r: "ANN fp32" if r["pathway"] == "ann"
                          else f"{r['pathway'].upper()} T={int(r['T'])}", axis=1)
    return g


def layer_table(results, pathway):
    rows = []
    for r in results:
        if r["pathway"] != pathway or not r.get("spike_rate_per_layer"):
            continue
        for layer, rate in r["spike_rate_per_layer"].items():
            s_th = breakeven_spike_rate(r["T"], E_AC_TH_PJ)
            s_si = breakeven_spike_rate(r["T"], E_AC_SI_PJ)
            rows.append(dict(
                config=cfg_label(r), seed=r["seed"], layer=layer, spike_rate=rate,
                s_star_th=s_th, G1_th="PASS" if rate < s_th else "FAIL",
                s_star_si=s_si, G1_si="PASS" if rate < s_si else "FAIL"))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return (df.groupby(["config", "layer"], as_index=False)
              .agg(spike_rate=("spike_rate", "mean"),
                   s_star_th=("s_star_th", "first"),
                   s_star_si=("s_star_si", "first"),
                   G1_th=("G1_th", lambda s: "PASS" if all(s == "PASS") else "FAIL"),
                   G1_si=("G1_si", lambda s: "PASS" if all(s == "PASS") else "FAIL")))


def deployment(g, per_config_gates):
    """RQ3: min-EDP configuration meeting each profile's accuracy floor and all gates."""
    out = []
    for prof in DEPLOYMENT_PROFILES:
        row = dict(profile=prof["name"], power_budget_mW=prof["power_mw"],
                   accuracy_floor=prof["acc_floor"])
        for const, edp_col, adm_key in (("theoretical", "edp_th", "G1_theoretical_all_pass"),
                                        ("calibrated", "edp_si", "G1_calibrated_all_pass")):
            cand = g[g["accuracy_mean"] >= prof["acc_floor"]].copy()
            keep = []
            for _, r in cand.iterrows():
                key = f"{r['pathway']}_T{int(r['T'])}"
                s = per_config_gates.get(key)
                if r["pathway"] == "ann":
                    keep.append(True); continue
                keep.append(bool(s and s[adm_key] and s["G2"]["passed"] is not False))
            cand = cand[keep]
            if len(cand) == 0:
                row[f"recommend_{const}"] = "no admissible configuration"
                row[f"edp_{const}"] = None
                row[f"arithmetic_power_uW_{const}"] = None
            else:
                best = cand.loc[cand[edp_col].idxmin()]
                e = best["e_th_uj"] if const == "theoretical" else best["e_si_uj"]
                row[f"recommend_{const}"] = best["config"]
                row[f"edp_{const}"] = float(best[edp_col])
                row[f"arithmetic_power_uW_{const}"] = float(e * ASSUMED_INFERENCE_RATE_HZ)
        row["selection_stable"] = (row["recommend_theoretical"] == row["recommend_calibrated"])
        out.append(row)
    return pd.DataFrame(out)


def figures(results, g, fig_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. training curves
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in results:
        if not r.get("history") or r["seed"] != 42:
            continue
        ep = [h["epoch"] for h in r["history"]]
        ax[0].plot(ep, [h["train_loss"] for h in r["history"]], label=cfg_label(r))
        ax[1].plot(ep, [h["test_acc"] for h in r["history"]], label=cfg_label(r))
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("training loss"); ax[0].set_yscale("log")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("test accuracy (%)")
    ax[1].legend(fontsize=7); ax[0].set_title("Training loss"); ax[1].set_title("Test accuracy")
    fig.tight_layout(); fig.savefig(fig_dir / "fig_training_curves.png", dpi=200); plt.close(fig)

    # 2. accuracy-energy frontier, both constants
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in g.iterrows():
        ax.scatter(r["e_th_uj"], r["accuracy_mean"], marker="o", s=60)
        ax.scatter(r["e_si_uj"], r["accuracy_mean"], marker="x", s=60)
        ax.annotate(r["config"], (r["e_th_uj"], r["accuracy_mean"]),
                    fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_xlabel("energy per inference (uJ, log scale)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy-energy frontier\n(o = 0.9 pJ/SOP, x = 1.89 pJ/SOP)")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(fig_dir / "fig_accuracy_energy.png", dpi=200); plt.close(fig)

    # 3. per-layer spike rates against G1 thresholds
    rows = []
    for r in results:
        if not r.get("spike_rate_per_layer"):
            continue
        for l, v in r["spike_rate_per_layer"].items():
            rows.append((cfg_label(r), l, v, breakeven_spike_rate(r["T"], E_AC_TH_PJ),
                         breakeven_spike_rate(r["T"], E_AC_SI_PJ)))
    if rows:
        df = pd.DataFrame(rows, columns=["config", "layer", "rate", "s_th", "s_si"])
        piv = df.groupby(["config", "layer"])["rate"].mean().unstack()
        fig, ax = plt.subplots(figsize=(9, 5))
        piv.plot(kind="bar", ax=ax)
        for cfgname in piv.index:
            sub = df[df["config"] == cfgname]
            ax.axhline(sub["s_th"].iloc[0], ls="--", lw=0.7, color="green", alpha=0.5)
            ax.axhline(sub["s_si"].iloc[0], ls=":", lw=0.9, color="red", alpha=0.6)
        ax.set_ylabel("spikes / neuron / timestep")
        ax.set_title("Per-layer spike rate vs G1 break-even thresholds\n"
                     "(dashed green = 0.9 pJ, dotted red = 1.89 pJ)")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout(); fig.savefig(fig_dir / "fig_spike_rates.png", dpi=200); plt.close(fig)
    print(f"  figures -> {fig_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--tables-dir", default="./tables")
    ap.add_argument("--figures-dir", default="./figures")
    a = ap.parse_args()

    results = load(a.out_dir)
    per_run, summary = apply_gates(results)
    g = agg(results)
    td = Path(a.tables_dir); td.mkdir(parents=True, exist_ok=True)

    # 5.4 baselines
    t54 = g[g["pathway"] == "ann"][["config", "accuracy_mean", "accuracy_sd",
                                    "latency_gpu_ms", "latency_cpu_ms", "e_th_uj"]]
    t54.to_csv(td / "table_5_4_baseline.csv", index=False)

    # 5.5 / 5.7 pathways
    for pw, name in (("stbp", "table_5_5_pathA"), ("qcfs", "table_5_7_pathB")):
        sub = g[g["pathway"] == pw]
        sub.to_csv(td / f"{name}.csv", index=False)

    layer_table(results, "stbp").to_csv(td / "table_5_6_pathA_layers.csv", index=False)
    layer_table(results, "qcfs").to_csv(td / "table_5_8_pathB_layers.csv", index=False)

    # 5.9 unified
    t59 = g[["config", "accuracy_mean", "accuracy_sd", "spike_rate_mean",
             "latency_gpu_ms", "latency_cpu_ms", "e_th_uj", "e_si_uj"]].copy()
    t59["G1_th"] = [("n/a" if r.pathway == "ann" else
                     ("PASS" if summary[f"{r.pathway}_T{int(r.T)}"]["G1_theoretical_all_pass"]
                      else "FAIL")) for r in g.itertuples()]
    t59["G1_si"] = [("n/a" if r.pathway == "ann" else
                     ("PASS" if summary[f"{r.pathway}_T{int(r.T)}"]["G1_calibrated_all_pass"]
                      else "FAIL")) for r in g.itertuples()]
    t59.to_csv(td / "table_5_9_unified.csv", index=False)

    # 5.10 gates
    pd.DataFrame([dict(config=k, **{kk: vv for kk, vv in v.items() if kk != "G2"},
                       G2_sd=v["G2"]["sd"], G2_pass=v["G2"]["passed"])
                  for k, v in summary.items()]).to_csv(td / "table_5_10_gates.csv", index=False)

    # 5.11 composite
    t511 = g[["config", "edp_th", "edp_si", "aep_th", "aep_si"]].copy()
    for c in ("edp_th", "edp_si", "aep_th", "aep_si"):
        t511[f"rank_{c}"] = t511[c].rank().astype(int)
    t511.sort_values("edp_th").to_csv(td / "table_5_11_composite.csv", index=False)

    # 6.1 deployment
    dep = deployment(g, summary)
    dep.to_csv(td / "table_6_1_deployment.csv", index=False)

    # markdown bundle
    with open(td / "thesis_tables.md", "w") as f:
        for label, df in [("Table 5.4 Baselines", t54),
                          ("Table 5.5 Path A", g[g["pathway"] == "stbp"]),
                          ("Table 5.6 Path A per-layer", layer_table(results, "stbp")),
                          ("Table 5.7 Path B", g[g["pathway"] == "qcfs"]),
                          ("Table 5.8 Path B per-layer", layer_table(results, "qcfs")),
                          ("Table 5.9 Unified four-metric", t59),
                          ("Table 5.11 Composite rankings", t511.sort_values("edp_th")),
                          ("Table 6.1 Deployment", dep)]:
            f.write(f"\n\n## {label}\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".4f") if len(df) else "_(no data)_\n")

    figures(results, g, Path(a.figures_dir))

    print("\n" + "=" * 90)
    print(t59.to_string(index=False))
    print("=" * 90)
    print("\nDeployment (RQ3):")
    print(dep.to_string(index=False))
    print(f"\nAll tables written to {td}/  (thesis_tables.md is paste-ready)")

    # --- a blunt check the analyst should see
    q = g[(g["pathway"] == "ann")]
    best_snn = g[g["pathway"] != "ann"]
    if len(q) and len(best_snn):
        if best_snn["e_si_uj"].min() >= q["e_th_uj"].min():
            print("\n*** NOTE: no SNN configuration beats the ANN baseline at the "
                  "silicon-calibrated constant. That is a legitimate finding for RQ1. "
                  "Report it; do not quietly switch to the theoretical constant. ***")


if __name__ == "__main__":
    main()
