"""Spec-driven backbones, so one topology description serves all three pathways.

WHY THIS MODULE EXISTS
----------------------
The MNIST backbone in models.py is written out layer by layer: conv1, conv2,
fc1, fc2. That is readable and it is exactly what the thesis describes, so it
is left alone. CIFAR-10/100 need a deeper network -- a two-convolution backbone
reaches roughly 75 percent on CIFAR-10 and would tell us nothing -- and writing
a second, third and fourth deep network by hand invites the three pathways to
drift apart. Drift between pathways is precisely the thing that makes RQ1
unanswerable, so instead every pathway here is built from one spec.

A spec is a VGG-style list, read left to right:

    [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

an integer meaning a 3x3 convolution with that many output channels, and 'M'
meaning a 2x2 max-pool. After the features come flatten, one hidden fully
connected layer of `fc_hidden` units, and the output layer.

The MNIST backbone is expressible in the same form ([32, 'M', 64, 'M'] with
fc_hidden=128), and mnist_spec_matches_legacy() below asserts that the generic
path reproduces the published MNIST counts exactly -- 4,241,152 MACs, 422,090
parameters, 37,770 spiking sites. That assertion is the reason this module can
be trusted on CIFAR: it is checked against numbers that are already in the
thesis.

MAX-POOLING IS KEPT DELIBERATELY
--------------------------------
Section 5.8 of the thesis measures a spike-domain pooling bias: because the
maximum is convex, a converted network max-pooling binary spikes passes more
activity than its source, and that bias grows with T. Average pooling is the
usual remedy and Section 6.3 names it as future work. Switching to average
pooling here would silently change the thing under study, so the spec keeps
max-pooling and `pool` is left as a parameter for the arm that tests it.
"""
from collections import OrderedDict


# --------------------------------------------------------------------------
# Specs
# --------------------------------------------------------------------------
VGG11 = [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M']

# The published MNIST backbone, expressed as a spec. Used only to prove the
# generic counter agrees with the hand-written one; the MNIST runs themselves
# still go through models.py unchanged.
MNIST_LEGACY = [32, 'M', 64, 'M']


def make_spec(features, in_channels, input_hw, num_classes, fc_hidden, name):
    return dict(features=list(features), in_channels=in_channels,
                input_hw=input_hw, num_classes=num_classes,
                fc_hidden=fc_hidden, name=name)


ARCH_CIFAR10 = make_spec(VGG11, 3, 32, 10, 512, "vgg11")
ARCH_CIFAR100 = make_spec(VGG11, 3, 32, 100, 512, "vgg11")


def is_spec(arch) -> bool:
    """True for a spec-style arch dict, False for the legacy MNIST ARCH dict."""
    return isinstance(arch, dict) and "features" in arch


# --------------------------------------------------------------------------
# Analytical counting over a spec
# --------------------------------------------------------------------------
def spec_counts(spec):
    """Exact per-layer MACs, parameters and spiking sites for a spec.

    Mirrors energy.analytical_counts() so the two are directly comparable:
      macs      3x3 conv at HxW with C_in -> C_out costs C_out*H*W*C_in*9
      params    weights + bias, BatchNorm affine counted separately in the total
      sites     one spiking site per output element per timestep
      in_numel  input elements per timestep, used to derive synaptic fan-out

    Returns (rows, total) with the same shape as analytical_counts().
    """
    rows = OrderedDict()
    hw = spec["input_hw"]
    c_in = spec["in_channels"]
    bn_channels = []
    conv_i = 0

    for item in spec["features"]:
        if item == 'M':
            if hw % 2 != 0:
                raise ValueError(
                    f"spec {spec['name']}: pooling a {hw}x{hw} map is not exact; "
                    "every 'M' must halve an even spatial dimension")
            hw //= 2
            continue
        conv_i += 1
        c_out = int(item)
        rows[f"conv{conv_i}"] = dict(
            macs=c_out * hw * hw * c_in * 9,
            params=c_in * c_out * 9 + c_out,
            sites=c_out * hw * hw,
            in_numel=c_in * hw * hw,
        )
        bn_channels.append(c_out)
        c_in = c_out

    fc_in = c_in * hw * hw
    rows["fc1"] = dict(
        macs=fc_in * spec["fc_hidden"],
        params=fc_in * spec["fc_hidden"] + spec["fc_hidden"],
        sites=spec["fc_hidden"],
        in_numel=fc_in,
    )
    bn_channels.append(spec["fc_hidden"])
    rows["fc2"] = dict(
        macs=spec["fc_hidden"] * spec["num_classes"],
        params=spec["fc_hidden"] * spec["num_classes"] + spec["num_classes"],
        sites=spec["num_classes"],
        in_numel=spec["fc_hidden"],
    )

    bn_params = 2 * sum(bn_channels)
    total = dict(
        macs=sum(r["macs"] for r in rows.values()),
        params=sum(r["params"] for r in rows.values()) + bn_params,
        sites=sum(r["sites"] for r in rows.values()),
        bn_params=bn_params,
        final_hw=hw,
        n_conv=conv_i,
    )
    return rows, total


def layer_names(spec):
    """Weight-layer and neuron-layer names, in forward order.

    The spiking models below register their submodules under exactly these
    names so SpikeMonitor can hook them without knowing the topology. Note
    that fc2 is a weight layer but has no neuron after it: the output layer
    accumulates logits rather than spiking, which is why `neurons` is one
    shorter than `weights`.
    """
    _, total = spec_counts(spec)
    weights = [f"conv{i}" for i in range(1, total["n_conv"] + 1)] + ["fc1", "fc2"]
    neurons = [f"sn{i}" for i in range(1, total["n_conv"] + 2)]
    return weights, neurons


# --------------------------------------------------------------------------
# Self-check against the published MNIST numbers
# --------------------------------------------------------------------------
def mnist_spec_matches_legacy(verbose=False):
    """Assert the generic counter reproduces the thesis's MNIST figures.

    If this fails, nothing else in this module should be believed.
    """
    spec = make_spec(MNIST_LEGACY, 1, 28, 10, 128, "mnist_legacy")
    _, total = spec_counts(spec)
    expected = dict(macs=4_241_152, params=422_090, sites=37_770)
    bad = {k: (total[k], v) for k, v in expected.items() if total[k] != v}
    if verbose:
        for k, v in expected.items():
            flag = "ok" if total[k] == v else "MISMATCH"
            print(f"  {k:7s} generic={total[k]:>12,}  thesis={v:>12,}  {flag}")
    if bad:
        raise AssertionError(
            "generic spec counter disagrees with the published MNIST counts: "
            + ", ".join(f"{k}: got {g:,}, thesis says {e:,}" for k, (g, e) in bad.items()))
    return True
