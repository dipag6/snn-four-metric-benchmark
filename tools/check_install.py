#!/usr/bin/env python
"""Verify the environment before training anything.

Checks versions AND the specific API symbols this code depends on. A version
string on its own proves very little -- what matters is whether the functions
actually exist and accept the arguments we pass.

  python tools/check_install.py
"""
import importlib.metadata as md
import sys


def version(pkg):
    """Robust version lookup.

    NOTE: `spikingjelly.__version__` does NOT exist -- the top-level package is
    empty (dir() returns nothing public). Reaching for `__version__` raises
    AttributeError. importlib.metadata reads the installed distribution
    metadata and works for every pip-installed package.
    """
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def main():
    ok = True
    print("=" * 68)
    print("ENVIRONMENT CHECK")
    print("=" * 68)

    # --- versions
    for pkg in ("torch", "torchvision", "spikingjelly", "fvcore", "numpy", "pandas"):
        v = version(pkg)
        if v is None:
            print(f"  {pkg:<14} NOT INSTALLED")
            if pkg in ("torch", "torchvision"):
                ok = False
        else:
            print(f"  {pkg:<14} {v}")

    # --- GPU
    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"\n  CUDA available: {cuda}")
        if cuda:
            print(f"  GPU           : {torch.cuda.get_device_name(0)}")
            print(f"  VRAM          : "
                  f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("  *** No GPU. Runtime -> Change runtime type -> GPU. ***")
            print("      STBP at T=16 on CPU is a multi-day job.")
            ok = False
    except Exception as e:
        print(f"  torch import failed: {e}")
        ok = False

    # --- the API symbols we actually call
    print("\n  SpikingJelly API:")
    if version("spikingjelly") is None:
        print("    not installed -- use --backend native on every run.py command")
    else:
        try:
            from spikingjelly.activation_based import neuron, surrogate, functional
            import inspect
            checks = []
            checks.append(("neuron.ParametricLIFNode", hasattr(neuron, "ParametricLIFNode")))
            checks.append(("neuron.IFNode", hasattr(neuron, "IFNode")))
            checks.append(("surrogate.ATan", hasattr(surrogate, "ATan")))
            checks.append(("functional.reset_net", hasattr(functional, "reset_net")))

            # v_reset=None is what gives SOFT reset. The default is 0.0 (HARD
            # reset), which is the wrong dynamics for Path A and silently changes
            # every spike count. Confirm the argument exists.
            sig = inspect.signature(neuron.ParametricLIFNode.__init__)
            checks.append(("ParametricLIFNode accepts v_reset", "v_reset" in sig.parameters))
            checks.append(("ParametricLIFNode accepts init_tau", "init_tau" in sig.parameters))
            checks.append(("ParametricLIFNode accepts step_mode", "step_mode" in sig.parameters))

            for name, good in checks:
                print(f"    {'OK  ' if good else 'MISS'}  {name}")
                ok &= good

            # smoke-fire one neuron to prove it runs
            import torch
            n = neuron.ParametricLIFNode(init_tau=4.0, v_threshold=1.0, v_reset=None,
                                         surrogate_function=surrogate.ATan(alpha=2.0),
                                         detach_reset=True, step_mode="s")
            with torch.no_grad():
                s = sum(float(n(torch.randn(4, 8) * 3.0).sum()) for _ in range(8))
            print(f"    OK    forward pass runs, emitted {s:.0f} spikes")
            if s == 0:
                print("    *** emitted zero spikes -- unexpected for this input scale ***")
                ok = False
        except Exception as e:
            print(f"    IMPORT/API FAILURE: {type(e).__name__}: {e}")
            print("    -> add --backend native to every run.py command; that path is tested")
            ok = False

    print("=" * 68)
    print("RESULT:", "PASS" if ok else "FAIL -- read the messages above")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
