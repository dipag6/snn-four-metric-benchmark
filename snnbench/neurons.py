"""Spiking neuron models.

Two backends are provided:

  backend="spikingjelly"  -- uses SpikingJelly's ParametricLIFNode / IFNode.
                             This is what the thesis specifies and what you
                             should report.
  backend="native"        -- a ~60-line pure-PyTorch reimplementation with the
                             identical update rule and ATan surrogate.

The native backend exists so that a SpikingJelly API change cannot strand you
mid-candidature, and so you can verify that the framework is not doing anything
you did not ask for. tools/check_backends.py asserts that the two agree to
within floating-point tolerance on identical inputs; run it once and cite the
result if a reviewer asks whether the harness is framework-dependent.
"""
import math
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# Native implementation
# --------------------------------------------------------------------------
class ATanSurrogate(torch.autograd.Function):
    """Heaviside forward, arctangent-derivative backward.

    g(x)  = (1/pi) * arctan(pi/2 * alpha * x) + 1/2
    g'(x) = alpha / (2 * (1 + (pi/2 * alpha * x)^2))

    At x = 0 the surrogate derivative is alpha/2. This is the attenuation
    factor discussed in Thesis Section 2.4.1.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0.0).to(x)

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        a = ctx.alpha
        sg = a / 2.0 / (1.0 + (math.pi / 2.0 * a * x).pow(2))
        return grad_out * sg, None


def atan_spike(x, alpha=2.0):
    return ATanSurrogate.apply(x, alpha)


class NativePLIF(nn.Module):
    """Parametric LIF with learnable membrane time constant and soft reset.

    Follows Fang et al. (2021). tau is parameterised through a sigmoid so it
    stays in (1, inf): the learnable scalar w gives decay factor
    sigmoid(w) = 1/tau. w is initialised so that 1/tau = tau_init.

    Update (soft reset):
        H[t] = V[t-1] + (1/tau) * (X[t] - V[t-1])
        S[t] = Theta(H[t] - Vth)
        V[t] = H[t] - S[t] * Vth
    """
    def __init__(self, tau_init=0.25, v_threshold=1.0, alpha=2.0):
        super().__init__()
        # tau_init here is the *decay rate* 1/tau, matching the thesis wording
        # "learnable tau initialised at 0.25" as used by SpikingJelly, where the
        # constructor argument init_tau is the reciprocal-style parameter.
        p = float(tau_init)
        p = min(max(p, 1e-4), 1 - 1e-4)
        w0 = math.log(p / (1 - p))                      # sigmoid(w0) = p
        self.w = nn.Parameter(torch.tensor(w0, dtype=torch.float32))
        self.v_threshold = v_threshold
        self.alpha = alpha
        self.v = None

    def reset(self):
        self.v = None

    @property
    def decay(self):
        return torch.sigmoid(self.w)

    def forward(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.zeros_like(x)
        self.v = self.v + self.decay * (x - self.v)
        s = atan_spike(self.v - self.v_threshold, self.alpha)
        # detach the reset, matching SpikingJelly's detach_reset=True: the reset
        # path is not differentiated, only the forward spike is.
        self.v = self.v - s.detach() * self.v_threshold          # soft reset
        return s


class NativeIF(nn.Module):
    """Integrate-and-fire with hard reset, for Path B (QCFS conversion).

        V[t] = V[t-1] + X[t]
        S[t] = Theta(V[t] - Vth)
        V[t] = V[t] * (1 - S[t])          # hard reset to zero

    v_init allows the QCFS shift term (v(0) = threshold/2), which is what makes
    T = 4 conversion work (Bu et al. 2022).
    """
    def __init__(self, v_threshold=1.0, v_init_frac=0.5, alpha=2.0):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_init_frac = v_init_frac
        self.alpha = alpha
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x):
        if self.v is None or self.v.shape != x.shape:
            self.v = torch.full_like(x, self.v_init_frac * self.v_threshold)
        self.v = self.v + x
        s = (self.v >= self.v_threshold).to(x)          # inference only, no grad
        self.v = self.v - s * self.v_threshold          # soft reset for QCFS parity
        return s


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------
def make_plif(cfg):
    if cfg.backend == "spikingjelly":
        from spikingjelly.activation_based import neuron, surrogate
        # v_reset=None is REQUIRED for soft reset. SpikingJelly defaults to
        # v_reset=0.0, which is a HARD reset -- the wrong dynamics for Path A and
        # a silent departure from the thesis specification (Section 3.4). Omitting
        # this argument makes the SpikingJelly and native backends disagree;
        # tools/check_backends.py catches it.
        return neuron.ParametricLIFNode(
            init_tau=1.0 / cfg.plif_tau_init if cfg.plif_tau_init < 1 else cfg.plif_tau_init,
            v_threshold=cfg.v_threshold,
            v_reset=None,                       # soft reset: V <- V - V_th
            surrogate_function=surrogate.ATan(alpha=cfg.atan_alpha),
            detach_reset=True,
            step_mode="s",
        )
    return NativePLIF(cfg.plif_tau_init, cfg.v_threshold, cfg.atan_alpha)


def make_if(cfg, v_threshold=1.0, v_init_frac=0.5):
    if cfg.backend == "spikingjelly":
        from spikingjelly.activation_based import neuron, surrogate
        n = neuron.IFNode(
            v_threshold=v_threshold,
            v_reset=None,                    # None => soft reset (QCFS semantics)
            surrogate_function=surrogate.ATan(alpha=cfg.atan_alpha),
            step_mode="s",
        )
        n._qcfs_v_init = v_init_frac * v_threshold
        return n
    return NativeIF(v_threshold, v_init_frac, cfg.atan_alpha)


def reset_net(model, cfg=None):
    """Clear all membrane state. Must be called between samples/batches."""
    did = False
    if cfg is None or cfg.backend == "spikingjelly":
        try:
            from spikingjelly.activation_based import functional
            functional.reset_net(model)
            did = True
        except Exception:
            pass
    for m in model.modules():
        if hasattr(m, "reset") and not did:
            m.reset()
        # re-apply the QCFS membrane shift after SpikingJelly's reset
        if hasattr(m, "_qcfs_v_init"):
            m.v = m._qcfs_v_init
