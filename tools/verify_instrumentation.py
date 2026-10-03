#!/usr/bin/env python
"""Refuse to train unless the spike-monitor fix is present in this copy of the code.

The first execution of the MR604 grid was invalidated by a defect in which the spike
hooks were detached only after the latency measurement, so 28,380 of 38,380 forward
passes were counted into the numerator and none into the denominator. Every spike
rate, synaptic-operation count and energy figure was inflated by roughly a factor of
four, no warning was emitted, and the resulting conclusion pointed the wrong way.

The fixed code is not necessarily the code that runs. A stale copy on Google Drive,
a partially synced folder or an older archive imports cleanly and produces plausible
numbers. This script inspects the source that is about to execute.

Reads the file directly rather than importing it, so it works before torch is
installed and cannot be fooled by a different module shadowing the package.

    python tools/verify_instrumentation.py [--path snnbench/engine.py]
"""
from __future__ import annotations
import argparse, ast, re, sys
from pathlib import Path


def fail(msg):
    print("\n" + "!" * 78)
    print("INSTRUMENTATION CHECK FAILED -- do not train with this copy of the code.")
    print("!" * 78)
    print(msg)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="snnbench/engine.py")
    a = ap.parse_args()

    p = Path(a.path)
    if not p.exists():
        fail(f"{p} not found. Run this from the code/ directory.")

    source = p.read_text(encoding="utf-8")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        fail(f"{p} does not parse: {e}")

    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "evaluate_four_metrics"),
              None)
    if fn is None:
        fail(f"evaluate_four_metrics not found in {p}")

    lines = source.splitlines()
    body = "\n".join(lines[fn.lineno - 1: (fn.end_lineno or len(lines))])

    i_remove = body.find("monitor.remove()")
    i_lat = body.find("measure_latency")
    checks = [
        ("spike hooks detached before latency measurement",
         i_remove != -1 and i_lat != -1 and i_remove < i_lat),
        ("per-layer rate above 1.0 raises rather than warns",
         bool(re.search(r"v\s*>\s*1\.0", body)) and "raise RuntimeError" in body),
        ("sample counter incremented inside the accuracy loop",
         "note_batch" in body),
    ]

    print(f"Instrumentation check on {p.resolve()}")
    for name, ok in checks:
        print(f"  [{'OK ' if ok else 'BAD'}] {name}")

    if not all(ok for _, ok in checks):
        fail(
            "This copy predates the correction described in Section 4.4.1 of the\n"
            "thesis. Training with it will silently reproduce the defect: spike rates\n"
            "and energies inflated roughly fourfold, with no warning in the output.\n\n"
            "Fix: copy the corrected code/ directory over this one, then re-run this\n"
            "check. The corrected snnbench/engine.py detaches the monitor immediately\n"
            "after the accuracy loop and raises on any per-layer rate above 1.0.")

    print("\ninstrumentation check: PASS -- safe to train")
    return 0


if __name__ == "__main__":
    sys.exit(main())
