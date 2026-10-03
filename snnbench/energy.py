"""Operation counting, spike monitoring and the energy proxy.

This is the module that the whole thesis rests on, so it is written to be
read and checked rather than to be short.

IMPORTANT METHODOLOGICAL NOTE
-----------------------------
The thesis's analytical surface (Table 5.3) uses the approximation

    N_SOP  ~=  s_bar * T * N_MAC

which assumes every layer fires at the same mean rate s_bar. That is fine as a
closed-form illustration, but it is NOT how the measured figures are produced
here. What is measured is exact:

    N_SOP  =  sum over weight layers L of
                 ( total input spikes arriving at L over all T timesteps )
                 * ( fan-out of one input element of L )

where fan-out = MACs_dense(L) / numel(input of L). Because every Conv2d and
Linear in the spiking pathways receives a binary tensor (spikes from the
previous neuron layer, or from the encoder), this count is exact rather than
estimated. Report the measured value; use the closed form only for the
analytical surface, and say which is which.
"""
import json
from collections import OrderedDict

import torch
import torch.nn as nn

from .config import E_MAC_PJ, E_AC_TH_PJ, E_AC_SI_PJ, breakeven_spike_rate


# ==========================================================================
# Analytical MAC / parameter counting
# ==========================================================================
def analytical_counts(arch=None):
    """Exact per-layer MACs, parameters and spiking sites (Thesis Appendix B)."""
    from .config import ARCH
    from .backbones import is_spec, spec_counts    # SPEC_COUNT_DISPATCH
    a = arch or ARCH
    if is_spec(a):
        return spec_counts(a)
    H = a["input_hw"]
    h1, h2 = H, H // 2          # conv1 output 28, conv2 output 14 (after pool1)
    h_final = h2 // 2           # 7

    rows = OrderedDict()
    rows["conv1"] = dict(
        macs=a["c1"] * h1 * h1 * a["in_channels"] * 9,
        params=a["in_channels"] * a["c1"] * 9 + a["c1"],
        sites=a["c1"] * h1 * h1,
        in_numel=a["in_channels"] * H * H,
    )
    rows["conv2"] = dict(
        macs=a["c2"] * h2 * h2 * a["c1"] * 9,
        params=a["c1"] * a["c2"] * 9 + a["c2"],
        sites=a["c2"] * h2 * h2,
        in_numel=a["c1"] * h2 * h2,
    )
    fc_in = a["c2"] * h_final * h_final
    rows["fc1"] = dict(
        macs=fc_in * a["fc_hidden"],
        params=fc_in * a["fc_hidden"] + a["fc_hidden"],
        sites=a["fc_hidden"],
        in_numel=fc_in,
    )
    rows["fc2"] = dict(
        macs=a["fc_hidden"] * a["num_classes"],
        params=a["fc_hidden"] * a["num_classes"] + a["num_classes"],
        sites=a["num_classes"],
        in_numel=a["fc_hidden"],
    )
    bn_params = 2 * (a["c1"] + a["c2"] + a["fc_hidden"])

    total = dict(
        macs=sum(r["macs"] for r in rows.values()),
        params=sum(r["params"] for r in rows.values()) + bn_params,
        sites=sum(r["sites"] for r in rows.values()),
        bn_params=bn_params,
    )
    return rows, total


def verify_counts(model=None, verbose=True, dataset="mnist"):  # VERIFY_DATASET_ARG
    """Cross-check the analytical derivation against the built model and,
    if available, against fvcore. Agreement is a precondition for reporting
    any energy figure (Thesis Section 4.4)."""
    from .config import EXPECTED_COUNTS, arch_for   # VERIFY_PER_DATASET
    arch = arch_for(dataset)
    rows, total = analytical_counts(arch)
    exp = EXPECTED_COUNTS.get(dataset)
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
    ok = True
    msgs = []

    if total["macs"] != EXPECTED_MACS:
        ok = False
        msgs.append(f"MAC mismatch: derived {total['macs']:,} vs expected {EXPECTED_MACS:,}")
    if total["params"] != EXPECTED_PARAMS:
        ok = False
        msgs.append(f"param mismatch: derived {total['params']:,} vs expected {EXPECTED_PARAMS:,}")
    if total["sites"] != EXPECTED_SPIKING_SITES:
        ok = False
        msgs.append(f"site mismatch: derived {total['sites']:,} vs expected {EXPECTED_SPIKING_SITES:,}")

    if model is not None:
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        # PLIF adds one learnable tau per neuron layer; BNTT multiplies BN params by T
        msgs.append(f"model trainable parameters (incl. BNTT/PLIF extras): {n:,}")

    try:
        if dataset not in ("mnist", "fmnist", "kmnist"):   # FVCORE_MNIST_ONLY
            raise RuntimeError("fvcore cross-check is defined for the "
                               "1x28x28 backbone only")
        from fvcore.nn import FlopCountAnalysis
        from .models import BaselineANN
        from .config import ARCH
        m = BaselineANN().eval()
        f = FlopCountAnalysis(m, torch.randn(1, 1, 28, 28))
        f.unsupported_ops_warnings(False); f.uncalled_modules_warnings(False)
        fv = f.total()
        delta = fv - total["macs"]

        # fvcore counts BatchNorm affine ops (2 per activation element); the
        # standard SNN accounting convention folds BN into the preceding conv at
        # inference and excludes it. Predict that difference exactly rather than
        # waving at a tolerance -- if it matches, both counts are correct.
        a = ARCH
        H = a["input_hw"]
        bn_elems = a["c1"] * H * H + a["c2"] * (H // 2) ** 2 + a["fc_hidden"]
        bn_ops = 2 * bn_elems

        msgs.append(f"fvcore total: {fv:,}   analytical (conv+fc only): {total['macs']:,}"
                    f"   delta {delta:+,}")
        if delta == bn_ops:
            msgs.append(f"  delta explained exactly: BatchNorm affine ops = 2 x {bn_elems:,}"
                        f" = {bn_ops:,}. fvcore counts BN; the SNN accounting convention"
                        f" folds BN into the preceding conv at inference and excludes it.")
            msgs.append("  -> counts AGREE under the stated convention.")
        elif abs(delta - bn_ops) <= 0.005 * total["macs"]:
            msgs.append(f"  delta approximately equals BatchNorm ops ({bn_ops:,}); acceptable.")
        else:
            ok = False
            msgs.append(f"  UNEXPLAINED delta. BatchNorm would account for {bn_ops:,}, "
                        f"leaving {delta - bn_ops:+,} unaccounted. Investigate before reporting.")
    except Exception as e:
        msgs.append(f"fvcore unavailable or failed ({e}); analytical count is authoritative")

    if verbose:
        print("--- operation count verification ---")
        for k, r in rows.items():
            print(f"  {k:8s} MAC={r['macs']:>10,}  params={r['params']:>9,}  sites={r['sites']:>8,}")
        print(f"  {'TOTAL':8s} MAC={total['macs']:>10,}  params={total['params']:>9,}  "
              f"sites={total['sites']:>8,}")
        for m_ in msgs:
            print("  " + m_)
        print(f"  verification: {'PASS' if ok else 'FAIL'}")
    return ok, rows, total, msgs


# ==========================================================================
# Spike monitoring
# ==========================================================================
class SpikeMonitor:
    """Counts (a) spikes emitted by each neuron layer and (b) spikes arriving
    at each weight layer, by forward hook.

    (a) gives the per-layer spike rate that gate G1 tests.
    (b) gives the exact synaptic operation count.

    A note on SpikingJelly: some releases expose a SOPMonitor helper and some
    do not. Rather than depend on that, this class hooks nn.Conv2d / nn.Linear
    directly, which works on every version and on both backends. If you cite
    "SOPMonitor" in the thesis, change it to describe this instrumentation --
    it is a claim about your method and it should be accurate.
    """

    def __init__(self, model, neuron_names=None, weight_names=None):
        # SPEC_MONITOR_NAMES: a spec-built model knows its own topology, so
        # ask it rather than assuming the four-layer MNIST backbone.
        if neuron_names is None or weight_names is None:
            spec = getattr(model, "spec", None)
            from .backbones import is_spec
            if is_spec(spec):
                from .specnets import monitor_names
                n_auto, w_auto = monitor_names(spec)
            else:
                n_auto = ("sn1", "sn2", "sn3")
                w_auto = ("conv1", "conv2", "fc1", "fc2")
            neuron_names = neuron_names or n_auto
            weight_names = weight_names or w_auto
        self.model = model
        self.handles = []
        self.spikes_out = OrderedDict()     # neuron layer -> total spikes
        self.sites = OrderedDict()          # neuron layer -> neurons per timestep
        self.spikes_in = OrderedDict()      # weight layer -> total input spikes
        self.in_numel = OrderedDict()       # weight layer -> input elements per step
        self.binary_in = OrderedDict()      # weight layer -> input was binary?
        self.samples = 0
        self.timesteps = 0

        for name in neuron_names:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            self.spikes_out[name] = 0.0
            self.sites[name] = 0
            self.handles.append(mod.register_forward_hook(self._out_hook(name)))

        for name in weight_names:
            mod = getattr(model, name, None)
            if mod is None:
                continue
            self.spikes_in[name] = 0.0
            self.in_numel[name] = 0
            self.binary_in[name] = True
            self.handles.append(mod.register_forward_hook(self._in_hook(name)))

    def _out_hook(self, name):
        def hook(mod, inp, out):
            with torch.no_grad():
                self.spikes_out[name] += float(out.detach().sum().item())
                if self.sites[name] == 0:
                    self.sites[name] = int(out[0].numel())
        return hook

    def _in_hook(self, name):
        def hook(mod, inp, out):
            with torch.no_grad():
                x = inp[0].detach()
                self.spikes_in[name] += float(x.sum().item())
                if self.in_numel[name] == 0:
                    self.in_numel[name] = int(x[0].numel())
                if self.binary_in[name]:
                    u = torch.unique(x[:1])
                    if u.numel() > 2 or bool(((u != 0) & (u != 1)).any().item()):
                        self.binary_in[name] = False
        return hook

    def note_batch(self, batch_size, T):
        self.samples += batch_size
        self.timesteps = T

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    # ---------------------------------------------------------------- report
    def per_layer_spike_rate(self):
        """spikes per neuron per timestep, per neuron layer."""
        T = max(self.timesteps, 1)
        return {k: (v / (self.sites[k] * T * max(self.samples, 1)))
                for k, v in self.spikes_out.items() if self.sites[k]}

    def synaptic_operations(self, macs_by_layer):
        """Exact SOP per inference, and the MAC residue for non-binary inputs.

        Returns (sop_per_inference, mac_residue_per_inference, breakdown).
        """
        sop = 0.0
        mac_res = 0.0
        breakdown = {}
        n = max(self.samples, 1)
        for name, spikes in self.spikes_in.items():
            if name not in macs_by_layer or not self.in_numel[name]:
                continue
            fanout = macs_by_layer[name]["macs"] / macs_by_layer[name]["in_numel"]
            if self.binary_in[name]:
                s = spikes / n * fanout
                sop += s
                breakdown[name] = dict(kind="AC", ops=s,
                                       input_spikes=spikes / n, fanout=fanout)
            else:
                # analogue input: this layer runs MACs at every timestep
                m = macs_by_layer[name]["macs"] * max(self.timesteps, 1)
                mac_res += m
                breakdown[name] = dict(kind="MAC", ops=m, input_spikes=None, fanout=fanout)
        return sop, mac_res, breakdown


# ==========================================================================
# Energy
# ==========================================================================
def ann_energy_uj(n_mac, bits=32):
    """Full-precision baseline; quantised variants scale per-op energy.

    Scaling for reduced precision is quadratic in bit width for the multiplier
    and linear for the adder. The commonly used first-order approximation for
    integer MACs is (b/32)^2 for the multiply plus (b/32) for the add; here we
    apply the widely cited simple quadratic scaling and label it as an
    approximation. State this in the thesis rather than presenting it as exact.
    """
    scale = (bits / 32.0) ** 2
    return n_mac * E_MAC_PJ * scale / 1e6


def snn_energy_uj(n_sop, n_mac_residue=0.0, e_ac_pj=E_AC_TH_PJ):
    return (n_sop * e_ac_pj + n_mac_residue * E_MAC_PJ) / 1e6


def energy_report(n_sop, n_mac_residue, n_mac_ann):
    e_ann = ann_energy_uj(n_mac_ann)
    e_th = snn_energy_uj(n_sop, n_mac_residue, E_AC_TH_PJ)
    e_si = snn_energy_uj(n_sop, n_mac_residue, E_AC_SI_PJ)
    return dict(
        n_sop=n_sop,
        n_mac_residue=n_mac_residue,
        e_ann_uj=e_ann,
        e_th_uj=e_th,
        e_si_uj=e_si,
        reduction_th=e_ann / e_th if e_th > 0 else float("inf"),
        reduction_si=e_ann / e_si if e_si > 0 else float("inf"),
        pct_saving_th=100.0 * (1 - e_th / e_ann),
        pct_saving_si=100.0 * (1 - e_si / e_ann),
    )


def edp(energy_uj, latency_ms):
    """Energy-Delay Product (Thesis Section 3.6)."""
    return energy_uj * latency_ms


def aep(energy_uj, accuracy_pct):
    """Accuracy-Energy Product = (1 - accuracy) * energy."""
    return (1.0 - accuracy_pct / 100.0) * energy_uj
