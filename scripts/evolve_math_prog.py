"""Evolve math algorithms — batched program synthesis on ForgeEvolve.

Examples:
    # Randomized tasks (func / goal / goal_io mix), difficulty 1-5
    python scripts/evolve_math_prog.py --steps 30

    # Ask for an outcome: program must return A=794
    python scripts/evolve_math_prog.py --target 794 --steps 50

    # Outcome at pinned inputs: A=794 when x0=12, x1=7, and the program
    # must generalize to held-out pairs of the hidden function
    python scripts/evolve_math_prog.py --target 794 --inputs 12,7 --steps 50
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from forge.evolution.domains.math_prog_domain import MathProgDomain
from forge.evolution.program_vm import OPS, prog_from_config, render
from forge.evolution.engine import ForgeEvolve, ForgeEvolveConfig

import torch

# Op whitelists for --ops: restrict which instructions programs may use.
# "cheap" = deployable in fused kernels (no transcendentals).
_OPIDX = {name: i for i, (name, *_r) in enumerate(OPS)}
_OP_SETS = {
    "all": None,
    "cheap": [_OPIDX[k] for k in ("add", "sub", "mul", "min", "max",
                                  "neg", "abs", "floor", "round")],
    "cheapdiv": [_OPIDX[k] for k in ("add", "sub", "mul", "pdiv", "pmod",
                                     "min", "max", "neg", "abs",
                                     "floor", "round")],
}

# --fit targets: real functions worth approximating with cheap ops.
# Each preset: (callable [N,K]->[N], K, lo, hi)
def _gelu(x):
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 *
                                       (x + 0.044715 * x ** 3)))

FIT_TARGETS = {
    "tanh":    (torch.tanh, 1, -8.0, 8.0),
    "sigmoid": (torch.sigmoid, 1, -8.0, 8.0),
    "silu":    (lambda x: x * torch.sigmoid(x), 1, -8.0, 8.0),
    "gelu":    (_gelu, 1, -8.0, 8.0),
    "expn":    (lambda x: torch.exp(-x.abs()), 1, -8.0, 8.0),
    "log2":    (lambda x: torch.log2(x.abs() + 1e-3), 1, 0.001, 16.0),
    "sin":     (lambda x: torch.sin(x), 1, -3.14159, 3.14159),
    "rsqrt":   (lambda x: torch.rsqrt(x.abs() + 0.01), 1, 0.01, 4.0),
    "recip":   (lambda x: 1.0 / x, 1, 0.5, 8.0),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=float, default=None,
                    help="Pinned outcome: program must return A=target")
    ap.add_argument("--inputs", type=str, default=None,
                    help="Pinned inputs for --target, comma-separated (e.g. 12,7)")
    ap.add_argument("--fit", choices=sorted(FIT_TARGETS.keys()), default=None,
                    help="Fit mode: evolve a program approximating this function")
    ap.add_argument("--range", type=str, default=None, metavar="LO,HI",
                    help="Override the fit input range (e.g. -8,8)")
    ap.add_argument("--ops", choices=sorted(_OP_SETS.keys()), default="all",
                    help="Op whitelist for evolved programs")
    ap.add_argument("--steps", type=int, default=50, help="generations")
    ap.add_argument("--generators", type=int, default=500)
    ap.add_argument("--filter-ratio", type=int, default=20,
                    help="N_generated / N_evaluated per generation")
    ap.add_argument("--max-evaluate", type=int, default=150,
                    help="Hard cap on candidates evaluated per generation")
    ap.add_argument("--elite-mutants", type=int, default=0,
                    help="Per-gen program-space mutations of archive elites "
                         "(GP-style local search; 0=off)")
    ap.add_argument("--tasks-per-eval", type=int, default=3)
    ap.add_argument("--n-steps", type=int, default=12, help="program length")
    ap.add_argument("--difficulty", type=str, default="1-5")
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--db-path", default="forge_evolve.db")
    ap.add_argument("--quiet", "-q", action="store_true")
    args = ap.parse_args()

    lo, hi = (int(x) for x in args.difficulty.split("-"))
    goal_inputs = ([float(v) for v in args.inputs.split(",")]
                   if args.inputs else None)

    target_fn, fit_k, ilo, ihi = None, 1, -8.0, 8.0
    if args.fit:
        target_fn, fit_k, ilo, ihi = FIT_TARGETS[args.fit]
        if args.range:
            ilo, ihi = (float(v) for v in args.range.split(","))

    domain = MathProgDomain(n_steps=args.n_steps,
                            difficulty_range=(lo, hi),
                            tasks_per_eval=args.tasks_per_eval,
                            goal_target=args.target,
                            goal_inputs=goal_inputs,
                            target_fn=target_fn, fit_k=fit_k,
                            input_lo=ilo, input_hi=ihi,
                            op_ids=_OP_SETS[args.ops],
                            device=None if args.device == "auto" else args.device)

    cfg = ForgeEvolveConfig(domain=domain, n_generators=args.generators,
                            filter_ratio=args.filter_ratio,
                            max_evaluate=args.max_evaluate,
                            elite_mutants=args.elite_mutants,
                            generations=args.steps, device=args.device,
                            db_path=args.db_path, verbose=not args.quiet)
    t0 = time.perf_counter()
    results = ForgeEvolve(cfg).run()
    dt = time.perf_counter() - t0

    n_eval = results["total_evaluations"]
    print("=" * 60)
    print(f"math_prog run complete — {n_eval} tries in {dt:.1f}s "
          f"({n_eval / max(dt, 1e-9):.0f} tries/sec)")
    print(f"  best score: {results['best_score']:.4f}")
    best = results.get("best_config") or {}
    if best:
        prog = prog_from_config(best)
        print(f"  best program: {render(prog)}")
        if target_fn is not None:
            # Verify: dense held-out sweep, real torch math.
            x = torch.linspace(ilo, ihi, 4096, dtype=torch.float64)\
                   .unsqueeze(-1).expand(-1, fit_k).contiguous()
            from forge.evolution.program_vm import run_programs, stack_programs
            out = run_programs(stack_programs([prog]), x)["out"][0]
            tgt = target_fn(x).reshape(-1)
            err = (out - tgt).abs()
            print(f"  verify vs {args.fit}: max_err={err.max().item():.6g} "
                  f"mae={err.mean().item():.6g} on 4096-pt dense sweep "
                  f"[{ilo:g},{ihi:g}]")
    if args.target is not None:
        print(f"  pinned goal:  A={args.target:g}"
              + (f" at inputs {args.inputs}" if args.inputs else " (no inputs)"))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
