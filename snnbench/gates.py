"""The four validity gates (Thesis Section 3.7).

The gates are applied here, in code, not by the analyst afterwards. A
configuration that fails G1 is recorded as energy-invalid and is excluded from
deployment recommendations; it is NOT deleted from the results. The record of
failures is where the boundary of the energy-favourable regime lives.
"""
import statistics
from collections import defaultdict

from .config import E_AC_TH_PJ, E_AC_SI_PJ, breakeven_spike_rate


def gate_g1(result, e_ac_pj):
    """Break-even check, applied PER LAYER (not to the network mean).

    A network whose mean rate passes can still contain a layer that fails, and
    because energy is additive across layers that layer dominates the total.
    """
    if result.get("spike_rate_per_layer") is None:
        return dict(applicable=False, passed=None, threshold=None, violations=[])
    T = result["T"]
    s_star = breakeven_spike_rate(T, e_ac_pj)
    violations = [dict(layer=k, rate=v, threshold=s_star)
                  for k, v in result["spike_rate_per_layer"].items() if v >= s_star]
    non_binding = s_star >= 1.0        # a rate-coded neuron cannot exceed 1.0

    # If any weight layer received analogue rather than binary input (direct
    # encoding), that layer executes MACs at every timestep. The closed-form
    # break-even threshold assumes an all-AC network and is then NECESSARY but
    # NOT SUFFICIENT: the spike rates can all pass while total energy still
    # exceeds the ANN baseline because of the MAC residue. Flag it and fall back
    # to the direct energy comparison.
    mac_residue = result.get("n_mac_residue", 0) or 0
    energy_dominated_by_mac = mac_residue > 0
    direct_ok = None
    if energy_dominated_by_mac:
        e_key = "e_th_uj" if abs(e_ac_pj - 0.9) < 1e-9 else "e_si_uj"
        direct_ok = result.get(e_key, float("inf")) < result.get("e_ann_uj", float("inf"))

    passed = (len(violations) == 0)
    if energy_dominated_by_mac:
        passed = passed and bool(direct_ok)

    return dict(applicable=True,
                passed=passed,
                threshold=s_star,
                non_binding=non_binding,
                violations=violations,
                mac_residue=mac_residue,
                closed_form_sufficient=(not energy_dominated_by_mac),
                direct_energy_below_ann=direct_ok,
                max_rate=max(result["spike_rate_per_layer"].values()))


def gate_g2(results_same_config, tolerance_pp=1.0):
    """Cross-seed reproducibility: accuracy s.d. across seeds < 1 percentage point."""
    accs = [r["accuracy"] for r in results_same_config]
    if len(accs) < 2:
        return dict(applicable=False, passed=None, sd=None, n_seeds=len(accs),
                    note="need at least two seeds")
    sd = statistics.stdev(accs)
    return dict(applicable=True, passed=sd < tolerance_pp, sd=sd,
                mean=statistics.mean(accs), n_seeds=len(accs),
                escalate=(sd >= tolerance_pp),
                note=("escalate to 5 seeds" if sd >= tolerance_pp else ""))


def gate_g3(result):
    """Separate latency reporting -- satisfied by construction if all three
    figures are present and no aggregate is computed."""
    keys = ["latency_gpu_ms", "latency_cpu_ms", "latency_neuromorphic_projected_ms"]
    have = [k for k in keys if result.get(k) is not None]
    spiking = result["pathway"] in ("stbp", "qcfs")
    need = 3 if spiking else 2
    return dict(applicable=True, passed=len(have) >= need, reported=have)


def gate_g4(result):
    """Dual energy reporting -- both constants present, ratio stated."""
    ok = result.get("e_th_uj") is not None and result.get("e_si_uj") is not None
    ratio = (result["e_si_uj"] / result["e_th_uj"]) if ok and result["e_th_uj"] else None
    return dict(applicable=True, passed=bool(ok), ratio_si_over_th=ratio)


def apply_gates(results):
    """Apply all four gates to a list of per-run results.

    Returns (per_run_gates, per_config_summary). Configuration = (pathway, T).
    """
    by_config = defaultdict(list)
    for r in results:
        by_config[(r["pathway"], r["T"])].append(r)

    per_run = {}
    summary = {}
    for key, runs in by_config.items():
        g2 = gate_g2(runs)
        for r in runs:
            g1_th = gate_g1(r, E_AC_TH_PJ)
            g1_si = gate_g1(r, E_AC_SI_PJ)
            g = dict(G1_theoretical=g1_th, G1_calibrated=g1_si,
                     G2=g2, G3=gate_g3(r), G4=gate_g4(r))
            g["admissible_theoretical"] = bool(
                (g1_th["passed"] is not False) and (g2["passed"] is not False)
                and g["G3"]["passed"] and g["G4"]["passed"])
            g["admissible_calibrated"] = bool(
                (g1_si["passed"] is not False) and (g2["passed"] is not False)
                and g["G3"]["passed"] and g["G4"]["passed"])
            per_run[r["tag"]] = g

        accs = [r["accuracy"] for r in runs]
        rates = [r["spike_rate_mean"] for r in runs if r.get("spike_rate_mean") is not None]
        summary[f"{key[0]}_T{key[1]}"] = dict(
            pathway=key[0], T=key[1], n_seeds=len(runs),
            accuracy_mean=sum(accs) / len(accs),
            accuracy_sd=(statistics.stdev(accs) if len(accs) > 1 else 0.0),
            spike_rate_mean=(sum(rates) / len(rates) if rates else None),
            e_th_uj=sum(r["e_th_uj"] for r in runs) / len(runs),
            e_si_uj=sum(r["e_si_uj"] for r in runs) / len(runs),
            latency_gpu_ms=sum(r["latency_gpu_ms"] for r in runs) / len(runs),
            latency_cpu_ms=sum(r["latency_cpu_ms"] for r in runs) / len(runs),
            G2=g2,
            G1_theoretical_all_pass=all(
                per_run[r["tag"]]["G1_theoretical"]["passed"] is not False for r in runs),
            G1_calibrated_all_pass=all(
                per_run[r["tag"]]["G1_calibrated"]["passed"] is not False for r in runs),
        )
    return per_run, summary


def format_gate_report(per_run, summary):
    lines = ["=" * 96, "VALIDITY GATE REPORT (Thesis Section 3.7 / Table 5.10)", "=" * 96]
    hdr = f"{'config':<14}{'acc mean':>10}{'sd':>7}{'s_bar':>9}{'G1 th':>8}{'G1 si':>8}{'G2':>6}{'admit':>18}"
    lines.append(hdr)
    lines.append("-" * 96)
    for k, s in sorted(summary.items()):
        sb = f"{s['spike_rate_mean']:.4f}" if s["spike_rate_mean"] is not None else "n/a"
        g1t = "PASS" if s["G1_theoretical_all_pass"] else "FAIL"
        g1s = "PASS" if s["G1_calibrated_all_pass"] else "FAIL"
        g2 = ("PASS" if s["G2"]["passed"] else "FAIL") if s["G2"]["applicable"] else "n/a"
        if s["pathway"] == "ann":
            g1t = g1s = "n/a"
        admit = ("th+si" if (s["G1_theoretical_all_pass"] and s["G1_calibrated_all_pass"])
                 else ("theoretical only" if s["G1_theoretical_all_pass"] else "neither"))
        if s["G2"]["applicable"] and not s["G2"]["passed"]:
            admit = "BLOCKED (G2)"          # unstable across seeds: nothing is admissible
        if s["pathway"] == "ann":
            admit = "baseline" if (not s["G2"]["applicable"] or s["G2"]["passed"]) else "baseline (G2 fail)"
        lines.append(f"{k:<14}{s['accuracy_mean']:>10.2f}{s['accuracy_sd']:>7.2f}"
                     f"{sb:>9}{g1t:>8}{g1s:>8}{g2:>6}{admit:>18}")
    lines.append("-" * 96)
    lines.append("G1 is applied PER LAYER. 'theoretical only' means the energy claim holds")
    lines.append("at 0.9 pJ/SOP but not at the 1.89 pJ/SOP measured on fabricated silicon.")
    lines.append("Report both. Do not quote the favourable one alone.")
    return "\n".join(lines)
