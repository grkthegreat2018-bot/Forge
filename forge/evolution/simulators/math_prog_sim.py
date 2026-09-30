"""math_prog simulator — live program execution metrics.

Scores a decoded register-machine program (from program_vm) against the
domain's current task set by actually running it — every metric comes from
executed math, not model judgment.

Anti-gaming signals (the "real working math" requirement):
  - heldout_exact: exact-match on input cases never shown to the search
    signal — memorizing shown targets fails held-out generalization.
  - uses_inputs:  fraction of provided input registers read inside the
    output's DEPENDENCY CONE (dead-end reads don't count) — a constant-
    returning program on an input-dependent task scores 0 here.
  - const_cheat:  require_inputs task where the program ignores ≥1 input —
    catches "always return the target constant" degenerate winners.
  - degenerate:   program output selects a raw input/const register with
    zero compute steps on a build-the-target task.
"""
from __future__ import annotations

import math
from typing import Any

import torch

from . import register
from ..program_vm import (dep_steps, prog_from_config, run_programs,
                          stack_programs, steps_used)

_EXACT_TOL = 1e-6


def _tol_exact(out: torch.Tensor, tgt: torch.Tensor) -> float:
    """Continuous proximity for FIT tasks: mean(1 - |err|/tol) clipped ≥0.

    tol = 5% of the task's target span. Smooth triangle kernel — gives
    REINFORCE a gradient everywhere instead of binary 2% hits.
    """
    span = float((tgt.max() - tgt.min()).item()) if tgt.numel() > 1 else 0.0
    tol = 0.05 * max(span, 1.0)
    return (1.0 - (out - tgt).abs() / tol).clamp(min=0.0).double().mean().item()


def case_metrics(out: torch.Tensor, targets: torch.Tensor) -> dict:
    """Exactness + error metrics for one program on one task's case set."""
    err = (out - targets).abs()
    rel = err / targets.abs().clamp(min=1.0)
    rel = torch.nan_to_num(rel, nan=1e9, posinf=1e9, neginf=1e9).clamp(max=1e9)
    exact = (err <= _EXACT_TOL).double().mean().item()
    rel_err = rel.mean().item()
    return {
        "exact": float(exact),
        "rel_err": float(rel_err),
        "out_var": float(out.var(unbiased=False).item()) if out.numel() > 1 else 0.0,
        "tgt_var": float(targets.var(unbiased=False).item()) if targets.numel() > 1 else 0.0,
    }


def program_metrics(prog: dict, tasks: list[dict], time_s: float = 0.0,
                    device=None, K_layout: int = 4, M: int = 8) -> dict[str, float]:
    """Run one program on every task and aggregate raw metrics.

    Args:
        prog: program dict (program_vm format)
        tasks: list of task dicts from MathProgDomain._gen_task
        time_s: wall-clock the candidate took to produce this program
        K_layout: input-register count used at decode/layout time
        M: const-register count
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    S = int(prog["ops"].numel())
    batched = stack_programs([prog], device=dev)

    train_exact_l, ho_exact_l, rel_err_l = [], [], []
    uses_in_l, const_cheat_l, degen_l = [], [], []
    steps_used_l = []
    abs_err_sum, abs_err_max, n_cases = 0.0, 0.0, 0

    for task in tasks:
        K = task["K"]
        inp = torch.tensor(task["inputs"], dtype=torch.float64, device=dev)
        tgt = torch.tensor(task["targets"], dtype=torch.float64, device=dev)
        res = run_programs(batched, inp)
        out = res["out"][0]                                  # [T]
        m = case_metrics(out, tgt)
        # FIT tasks: exact-match is unreachable by design — grade within a
        # 2%-of-span tolerance band instead (still fully executed math).
        train_exact_l.append(_tol_exact(out, tgt) if task["kind"] == 3
                             else m["exact"])
        rel_err_l.append(m["rel_err"])
        ae = (out - tgt).abs().nan_to_num(nan=1e9, posinf=1e9, neginf=1e9)
        abs_err_sum += float(ae.sum().item()); n_cases += ae.numel()
        abs_err_max = max(abs_err_max, float(ae.max().item()))

        # Held-out cases — the honesty check.
        if task["ho_inputs"]:
            ho_inp = torch.tensor(task["ho_inputs"], dtype=torch.float64, device=dev)
            ho_tgt = torch.tensor(task["ho_targets"], dtype=torch.float64, device=dev)
            ho_out = run_programs(batched, ho_inp)["out"][0]
            hm = case_metrics(ho_out, ho_tgt)
            ho_exact_l.append(_tol_exact(ho_out, ho_tgt) if task["kind"] == 3
                              else hm["exact"])
            rel_err_l[-1] = 0.5 * (rel_err_l[-1] + hm["rel_err"])
            hae = (ho_out - ho_tgt).abs().nan_to_num(nan=1e9, posinf=1e9, neginf=1e9)
            abs_err_sum += float(hae.sum().item()); n_cases += hae.numel()
            abs_err_max = max(abs_err_max, float(hae.max().item()))
        else:
            ho_exact_l.append(m["exact"])

        # Input usage counts ONLY reads in the output's dependency cone —
        # dead-end steps that never reach A cannot fake usage.
        K_l = int(prog.get("n_inputs", K_layout))
        cone, cone_ins = dep_steps(prog, K_l, M)
        used = (len([i for i in cone_ins if i < K]) / K) if K else 1.0
        uses_in_l.append(used)
        su = len(cone) / max(S, 1)
        steps_used_l.append(su)

        if task.get("require_inputs") and K > 0:
            const_cheat_l.append(1.0 if used < 1.0 else 0.0)
        else:
            # Constant-target tasks: outputting the right constant is legal.
            const_cheat_l.append(0.0)
        degen_l.append(1.0 if (task["kind"] == 1 and su <= 0.0) else 0.0)

    n = max(len(tasks), 1)
    train_exact = sum(train_exact_l) / n
    heldout_exact = sum(ho_exact_l) / n
    rel_err = sum(rel_err_l) / n
    # Smooth "closer is better" shaping for the REINFORCE gradient.
    err_score = 1.0 / (1.0 + math.log10(1.0 + rel_err))
    mae = abs_err_sum / max(n_cases, 1)
    # fit_score: -log10 of worst-case absolute error — the real signal for
    # approximation tasks where exact-match is impossible.
    fit_score = -math.log10(abs_err_max + 1e-9)
    return {
        "exact": 1.0 if (train_exact >= 1.0 and heldout_exact >= 1.0) else 0.0,
        "train_exact": float(train_exact),
        "heldout_exact": float(heldout_exact),
        "rel_err": float(rel_err),
        "err_score": float(err_score),
        "uses_inputs": float(sum(uses_in_l) / n),
        "const_cheat": float(max(const_cheat_l) if const_cheat_l else 0.0),
        "degenerate": float(max(degen_l) if degen_l else 0.0),
        "parsimony": float(1.0 - sum(steps_used_l) / n),
        "steps_used_frac": float(sum(steps_used_l) / n),
        "mae": float(mae),
        "max_err": float(abs_err_max),
        "fit_score": float(fit_score),
        "task_kind": float(tasks[0]["kind"]) if tasks else 0.0,
        "difficulty": float(tasks[0]["difficulty"]) if tasks else 1.0,
        "time_s": float(time_s),
        "speed_score": float(1.0 / (1.0 + max(time_s, 0.0))),
    }


@register("math_prog_simulate")
def math_prog_simulate(config: dict, domain: Any = None) -> dict:
    """Registry entry — single-program eval against domain.current_tasks."""
    prog = prog_from_config(config)
    tasks = getattr(domain, "current_tasks", None) or []
    K_layout = getattr(getattr(domain, "spec_vm", None), "max_inputs", 4)
    M = getattr(getattr(domain, "spec_vm", None), "n_consts", 8)
    return program_metrics(prog, tasks,
                           time_s=float(config.get("time_s", 0.0)),
                           device=getattr(domain, "_device", None),
                           K_layout=K_layout, M=M)
