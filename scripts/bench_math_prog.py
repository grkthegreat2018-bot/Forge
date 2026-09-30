"""Bench math_prog eval throughput — target: >100 tries/sec.

Measures the domain's vectorized batch_evaluate (all candidate programs x
all tasks in one VM pass) and the sequential evaluate() path for reference.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from forge.evolution.domains.math_prog_domain import MathProgDomain


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=200)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--tasks-per-eval", type=int, default=3)
    args = ap.parse_args()

    dev = args.device
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    domain = MathProgDomain(tasks_per_eval=args.tasks_per_eval, device=dev)

    # Random configs (uniform param vectors → decoded programs)
    D = domain.output_dim()
    configs = [domain.decode(torch.rand(D)) for _ in range(args.batch)]

    # Warmup (JIT-free path, but first call builds lazily-imported modules)
    domain.batch_evaluate(configs[:8])

    t0 = time.perf_counter()
    n = 0
    for _ in range(args.iters):
        domain.batch_evaluate(configs)
        n += len(configs)
    dt = time.perf_counter() - t0
    print(f"batch_evaluate: {n} programs / {dt:.2f}s = "
          f"{n / dt:.0f} tries/sec on {dev} "
          f"(batch={args.batch}, tasks_per_eval={args.tasks_per_eval})")

    # Sequential path for reference (single-config evaluate)
    t0 = time.perf_counter()
    for c in configs[:50]:
        domain.evaluate(c)
    dt2 = time.perf_counter() - t0
    print(f"evaluate():     50 programs / {dt2:.2f}s = {50 / dt2:.0f} tries/sec")
    return 0


if __name__ == "__main__":
    sys.exit(main())
