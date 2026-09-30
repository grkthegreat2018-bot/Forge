"""MathProgDomain — evolve small math ALGORITHMS, not just configs.

The searched object is a closed-ISA register-machine program (program_vm).
The batched generator population proposes [0,1]^D vectors → decoded to
programs → executed live against freshly randomized tasks → scored by
RewardGuard. The engine's REINFORCE trainer + kill/clone evolution then
optimize which programs get proposed. Nothing is judged by a model: all
math is executed.

Task kinds (randomized complexity via difficulty 1-5):
  0 FUNC     — match a hidden random f over inputs; graded on held-out cases
  1 GOAL     — "must return A=T" with NO inputs: compose the target constant
               from bounded const registers (big T needs ≥2-3 mul/add steps)
  2 GOAL_IO  — "must return A=T_j for inputs u_j": constraint pairs drawn
               from a hidden int function; held-out pairs grade generality.
               With goal_target/goal_inputs set this is the "ask for an
               outcome" mode — e.g. pinned A=794 at u0 while held-out pairs
               still force the program to learn the real mapping.

Anti-gaming (real working math, no forced victory):
  - held-out input cases on FUNC/GOAL_IO (memorized outputs fail)
  - uses_inputs metric + const_cheat penalty (returning a constant while
    ignoring provided inputs is detected and punished)
  - bounded decoded consts (±64) — a 794-style target cannot be embedded
    as a literal; it must be computed
  - degenerate penalty when the output register bypasses all compute

Safety: candidates can only express ops from the VM's closed table — no
eval, no IO, no loops. There is no generated code to scan; the ISA is the
sandbox. Extending to LLM-param-level discovery = adding tensor opcodes
(matmul/softmax/quant) to the table, still closed-world.

VRAM: batched eval touches [B ≤ ~200, T ≤ 8, R ≤ ~30] float64 tensors —
a few hundred KB. Effectively free on the 12GB budget; CPU path identical.
"""
from __future__ import annotations

import math
import random
from typing import Any

import torch

from ..program_vm import (
    CONST_HIGH, CONST_LOW, INT_OPS, N_OPS, OPS, ProgramSpec, config_dict,
    decode_params,
    dep_steps, encode_program, prog_from_config, program_from_lists, render,
    stack_programs, run_programs, steps_used,
)
from ..reward_guard import RewardGuard
from ..simulators import get_simulator
from ..simulators.math_prog_sim import program_metrics
from . import BaseDomain

KIND_FUNC, KIND_GOAL, KIND_GOAL_IO, KIND_FIT = 0, 1, 2, 3

# Int-preserving op ids usable inside hidden goal/goal_io functions so that
# the target-forcing shift stays exactly representable by int composition.
_BIN_INT = [i for i in INT_OPS if OPS[i][1] == 2 and OPS[i][0] != "pmod"]
_BIN_INT_PMOD = [i for i in INT_OPS if OPS[i][1] == 2]
_UN_INT = [i for i in INT_OPS if OPS[i][1] == 1]


class MathProgDomain(BaseDomain):
    """Evolve register-machine math programs against randomized tasks.

    Args:
        n_steps: program length S (registers written)
        n_consts: const pool M (first two pinned to 0/1, rest decode ±64 ints)
        max_inputs: K layout size for decode
        difficulty_range: (lo, hi) difficulty sampled per task
        tasks_per_eval: tasks each candidate is run against per evaluate()
        goal_target: if set (e.g. 794.0), tasks become pinned-outcome
            "must return A=T" — pure const-build when goal_inputs is None,
            else constraint-satisfaction at the pinned inputs + held-out
            generalization from a hidden function through (u0, T)
        goal_inputs: optional pinned input tuple for GOAL_IO mode
        seed/device/seq_len: standard domain plumbing (seq_len unused)
    """

    def __init__(self, n_steps: int = 12, n_consts: int = 8,
                 max_inputs: int = 4, difficulty_range=(1, 5),
                 tasks_per_eval: int = 3, n_shown: tuple[int, int] = (1, 4),
                 n_heldout: int = 4, goal_target: float | None = None,
                 goal_inputs: list[float] | None = None,
                 kind_weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
        target_fn=None, fit_k: int = 1, input_lo: float = -8.0,
        input_hi: float = 8.0, n_fit_cases: int = 16,
        op_ids: list[int] | None = None,
        seed: int = 42, device=None, seq_len: int | None = None):
        super().__init__()
        self.spec_vm = ProgramSpec(n_steps=n_steps, n_consts=n_consts,
                                   max_inputs=max_inputs, op_ids=tuple(op_ids) if op_ids else None)
        self.difficulty_range = tuple(difficulty_range)
        self.tasks_per_eval = max(1, int(tasks_per_eval))
        self.n_shown = tuple(n_shown)
        self.n_heldout = int(n_heldout)
        self.goal_target = goal_target
        self.goal_inputs = list(goal_inputs) if goal_inputs is not None else None
        self.kind_weights = kind_weights
        # FIT mode: evolve a program approximating target_fn over sampled
        # inputs. target_fn: torch callable [N,K] -> [N]. fit_name for display.
        self.target_fn = target_fn
        self.fit_k = int(fit_k)
        self.input_lo = float(input_lo)
        self.input_hi = float(input_hi)
        self.n_fit_cases = int(n_fit_cases)
        self.fit_name = getattr(target_fn, "__name__", "target") if target_fn else ""
        self._rng = random.Random(seed)
        self._seed = seed
        self.current_tasks: list[dict] = []
        if device is not None:
            self._device = torch.device(device) if not isinstance(device, torch.device) else device

        # Scoring spec (components/penalties only — params list unused here)
        from ..domain_spec import load_spec
        self.spec = load_spec("math_prog")
        self._guard = RewardGuard(self.spec.scoring)
        self._guard._behavioral_names = [b[0] for b in self.spec.behavioral_dims]

    # ------------------------------------------------------------------
    # BaseDomain interface
    # ------------------------------------------------------------------

    def name(self) -> str:
        return "math_prog"

    def output_dim(self) -> int:
        return self.spec_vm.output_dim()

    def behavioral_dims(self) -> list[tuple]:
        return [("task_kind", 4, 0, 4), ("difficulty", 5, 1, 5)]

    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        prog = decode_params(params, self.spec_vm, self.spec_vm.max_inputs)
        return config_dict(prog, raw_params=params)

    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        if "_p" in config:
            return torch.tensor(config["_p"], dtype=torch.float32)
        return encode_program(prog_from_config(config), self.spec_vm)

    def seed_configs(self) -> list[dict[str, Any]]:
        """Three trivially-true programs to bootstrap the archive."""
        S, M, K = self.spec_vm.n_steps, self.spec_vm.n_consts, self.spec_vm.max_inputs
        ops = [0] * S
        a = [0] * S
        b = [0] * S
        consts = [0.0] * M
        seeds = []
        for out_sel, label in ((0, "x0"), (K + M + S - 1, "last"), (K + 1, "c1")):
            seeds.append(config_dict(program_from_lists(
                ops, a, b, consts, out_sel, K)))
        if self.target_fn is not None:
            # Bootstrap the archive with generic approximation forms so the
            # search starts from plausible regions, not noise.
            ops_map = {o[0]: i for i, o in enumerate(OPS)}
            R0 = K + M
            ok = lambda n: n in ops_map
            # x / (1 + |x|) — classic squasher (needs abs + pdiv)
            if ok("abs") and ok("pdiv") and ok("add"):
                seeds.append(config_dict(program_from_lists(
                    [ops_map["abs"], ops_map["add"], ops_map["pdiv"]]
                    + [0] * (S - 3),
                    [0, R0, 0] + [0] * (S - 3),
                    [0, K + 1, R0 + 1] + [0] * (S - 3),
                    consts, R0 + 2, K)))
            # x * (3 - x^2) / 2 — cubic odd approx (needs mul + sub)
            if ok("mul") and ok("sub"):
                seeds.append(config_dict(program_from_lists(
                    [ops_map["mul"], ops_map["sub"], ops_map["mul"],
                     ops_map["mul"]] + [0] * (S - 4),
                    [0, K + 3, R0 + 1, R0 + 2] + [0] * (S - 4),
                    [0, R0, K + 2, 0] + [0] * (S - 4),
                    [0.0, 1.0, 0.5, 3.0] + [0.0] * (M - 4), R0 + 3, K)))
        return seeds

    # ------------------------------------------------------------------
    # Program-space local search (opt-in engine hook, elite_mutants > 0)
    # ------------------------------------------------------------------

    def propose_elite_mutants(self, archive, n: int, gen: int = 0):
        """GP-style mutation of archive elites, in program space.

        Generator-weight mutation lands anywhere in [0,1]^D — for a rugged
        discrete decode map that is exploration, not hill-climbing. Mutating
        the decoded program directly (one field at a time) gives the search
        a true local operator: swap an op, rewire one operand, nudge a const.
        """
        S, M, K = self.spec_vm.n_steps, self.spec_vm.n_consts, self.spec_vm.max_inputs
        op_ids = list(self.spec_vm.op_ids) if self.spec_vm.op_ids else list(range(N_OPS))
        rng = self._rng
        entries = sorted(getattr(archive, "history", []),
                         key=lambda e: e.score, reverse=True)
        seen, elites = set(), []
        for e in entries:
            if "ops" not in e.config:
                continue
            key = (tuple(e.config["ops"]), tuple(e.config["a_addr"]),
                   tuple(e.config["b_addr"]), int(e.config.get("out_sel", 0)))
            if key not in seen:
                seen.add(key)
                elites.append(e)
            if len(elites) >= 8:
                break
        if not elites:
            return []

        mutants = []
        for i in range(n):
            parent = elites[i % len(elites)]
            ops = [int(v) for v in parent.config["ops"]]
            a = [int(v) for v in parent.config["a_addr"]]
            b = [int(v) for v in parent.config["b_addr"]]
            consts = [float(v) for v in parent.config["consts"]]
            if M >= 2:
                consts[0], consts[1] = 0.0, 1.0   # ISA pin invariant
            out_sel = int(parent.config.get("out_sel", K + M + S - 1))
            n_edits = 1 if rng.random() < 0.75 else rng.randint(2, 3)
            for _ in range(n_edits):
                kind = rng.randrange(4)
                if kind == 0:                        # op swap
                    ops[rng.randrange(S)] = rng.choice(op_ids)
                elif kind == 1:                      # operand rewire
                    s = rng.randrange(S)
                    visible = K + M + s
                    if rng.random() < 0.5:
                        a[s] = rng.randrange(visible)
                    else:
                        b[s] = rng.randrange(visible)
                elif kind == 2:                      # const nudge
                    if M > 2:
                        m = rng.randrange(2, M)      # c0/c1 stay pinned
                        consts[m] = max(CONST_LOW,
                                        min(CONST_HIGH,
                                            consts[m] + rng.randint(-8, 8)))
                else:                                # output reselect
                    out_sel = rng.randrange(K + M + S)
            mutants.append(config_dict(program_from_lists(
                ops, a, b, consts, out_sel, K)))
        return mutants

    # ------------------------------------------------------------------
    # Evaluation — live execution
    # ------------------------------------------------------------------

    def evaluate(self, config: dict[str, Any]) -> dict:
        tasks = self._sample_tasks()
        self.current_tasks = tasks
        prog = prog_from_config(config)
        metrics = program_metrics(prog, tasks, device=self._device,
                                  K_layout=self.spec_vm.max_inputs,
                                  M=self.spec_vm.n_consts)
        result = self._guard.score(config, metrics)
        md = result["metadata"]
        md["program"] = render(prog)
        md["tasks"] = [t["text"] for t in tasks]
        return result

    def batch_evaluate(self, configs: list[dict]) -> list[dict]:
        """Vectorized eval: ALL programs × all sampled tasks in one pass.

        Same semantics as evaluate() (same task set per call, same metrics),
        but the VM runs the whole candidate batch per task — this is what
        makes >100 tries/sec trivial.
        """
        tasks = self._sample_tasks()
        self.current_tasks = tasks
        progs = [prog_from_config(c) for c in configs]
        batched = stack_programs(progs, device=self._device)
        B = len(progs)
        S = self.spec_vm.n_steps

        acc = {k: torch.zeros(B, device=self._device, dtype=torch.float64)
               for k in ("train_exact", "heldout_exact", "rel_err", "uses_inputs")}
        const_cheat = torch.zeros(B, device=self._device)
        degen = torch.zeros(B, device=self._device)
        abs_err_sum = torch.zeros(B, device=self._device, dtype=torch.float64)
        abs_err_max = torch.zeros(B, device=self._device, dtype=torch.float64)
        n_cases = 0

        for task in tasks:
            K = task["K"]
            inp = torch.tensor(task["inputs"], dtype=torch.float64,
                               device=self._device)                       # [T,K]
            tgt = torch.tensor(task["targets"], dtype=torch.float64,
                               device=self._device)
            res = run_programs(batched, inp)
            out = res["out"]                                              # [B,T]
            if task["kind"] == KIND_FIT:
                span = float((tgt.max() - tgt.min()).item()) if tgt.numel() > 1 else 0.0
                tol = 0.05 * max(span, 1.0)
                acc["train_exact"] += \
                    (1.0 - (out - tgt).abs() / tol).clamp(min=0.0).double().mean(1)
            else:
                acc["train_exact"] += (out - tgt).abs().le(1e-6).double().mean(1)
            acc["rel_err"] += (out - tgt).abs().div(tgt.abs().clamp(min=1.0)) \
                .nan_to_num(nan=1e9, posinf=1e9, neginf=1e9).clamp(max=1e9).mean(1)
            ae = (out - tgt).abs().nan_to_num(nan=1e9, posinf=1e9, neginf=1e9)
            abs_err_sum += ae.sum(1)
            abs_err_max = torch.maximum(abs_err_max, ae.max(1).values)
            n_cases += out.shape[1]
            if K:
                # Usage counts reads inside the output's dependency cone only
                uses = torch.zeros(B, device=self._device, dtype=torch.float64)
                K_l = self.spec_vm.max_inputs
                for b in range(B):
                    _cone, ins = dep_steps(progs[b], K_l, self.spec_vm.n_consts)
                    uses[b] = len([i for i in ins if i < K]) / K
                acc["uses_inputs"] += uses
                if task.get("require_inputs"):
                    const_cheat = torch.maximum(
                        const_cheat, (uses < 1.0).double())
            else:
                acc["uses_inputs"] += 1.0

            if task["ho_inputs"]:
                hi = torch.tensor(task["ho_inputs"], dtype=torch.float64,
                                  device=self._device)
                ht = torch.tensor(task["ho_targets"], dtype=torch.float64,
                                  device=self._device)
                ho = run_programs(batched, hi)["out"]
                if task["kind"] == KIND_FIT:
                    hspan = float((ht.max() - ht.min()).item()) if ht.numel() > 1 else 0.0
                    h_tol = 0.05 * max(hspan, 1.0)
                    acc["heldout_exact"] += \
                        (1.0 - (ho - ht).abs() / h_tol).clamp(min=0.0).double().mean(1)
                else:
                    acc["heldout_exact"] += (ho - ht).abs().le(1e-6).double().mean(1)
                acc["rel_err"] += (ho - ht).abs().div(ht.abs().clamp(min=1.0)) \
                    .nan_to_num(nan=1e9, posinf=1e9, neginf=1e9).clamp(max=1e9).mean(1)
                hae = (ho - ht).abs().nan_to_num(nan=1e9, posinf=1e9, neginf=1e9)
                abs_err_sum += hae.sum(1)
                abs_err_max = torch.maximum(abs_err_max, hae.max(1).values)
                n_cases += ho.shape[1]
            else:
                acc["heldout_exact"] += (out - tgt).abs().le(1e-6).double().mean(1)

            if task["kind"] == KIND_GOAL:
                for b in range(B):
                    if steps_used(progs[b], self.spec_vm.max_inputs,
                                  self.spec_vm.n_consts) <= 0:
                        degen[b] = 1.0

        n = max(len(tasks), 1)
        n_rel = 2 * n  # rel_err accumulated over shown + heldout passes
        results = []
        for b in range(B):
            rel_err = float(acc["rel_err"][b].item()) / n_rel
            te = float(acc["train_exact"][b].item()) / n
            he = float(acc["heldout_exact"][b].item()) / n
            metrics = {
                "exact": 1.0 if (te >= 1.0 and he >= 1.0) else 0.0,
                "train_exact": te,
                "heldout_exact": he,
                "rel_err": rel_err,
                "err_score": 1.0 / (1.0 + math.log10(1.0 + rel_err)),
                "uses_inputs": float(acc["uses_inputs"][b].item()) / n,
                "const_cheat": float(const_cheat[b].item()),
                "degenerate": float(degen[b].item()),
                "parsimony": 1.0 - steps_used(progs[b], self.spec_vm.max_inputs,
                                              self.spec_vm.n_consts) / S,
                "mae": float(abs_err_sum[b].item()) / max(n_cases, 1),
                "max_err": float(abs_err_max[b].item()),
                "fit_score": -math.log10(float(abs_err_max[b].item()) + 1e-9),
                "task_kind": float(tasks[0]["kind"]),
                "difficulty": float(tasks[0]["difficulty"]),
                "time_s": 0.0,
                "speed_score": 1.0,
            }
            r = self._guard.score(configs[b], metrics)
            r["metadata"]["program"] = render(progs[b])
            r["metadata"]["tasks"] = [t["text"] for t in tasks]
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Task generation — randomized equations, randomized complexity
    # ------------------------------------------------------------------

    def _sample_tasks(self) -> list[dict]:
        return [self._gen_task() for _ in range(self.tasks_per_eval)]

    def _gen_task(self) -> dict:
        lo, hi = self.difficulty_range
        difficulty = self._rng.randint(lo, hi)
        if self.target_fn is not None:
            kind = KIND_FIT
        elif self.goal_target is not None:
            kind = KIND_GOAL_IO if self.goal_inputs is not None else KIND_GOAL
        else:
            kind = self._rng.choices(
                [KIND_FUNC, KIND_GOAL, KIND_GOAL_IO],
                weights=self.kind_weights, k=1)[0]
        for _ in range(8):
            try:
                task = (self._gen_func(difficulty) if kind == KIND_FUNC
                        else self._gen_goal(difficulty) if kind == KIND_GOAL
                        else self._gen_goal_io(difficulty) if kind == KIND_GOAL_IO
                        else self._gen_fit(difficulty))
                if task is not None and self._targets_ok(task):
                    return task
            except Exception:
                continue
        # Degenerate fallback — still a valid task.
        if self.target_fn is not None:
            return self._gen_fit(1, _allow_fallback=True)
        return {"kind": KIND_GOAL, "K": 0, "inputs": [[ ]],
                "targets": [float(self.goal_target or 100.0)],
                "ho_inputs": [], "ho_targets": [], "require_inputs": False,
                "difficulty": 1, "text": f"A must equal {self.goal_target or 100.0:g}.",
                "prog": None}

    # ---- fit target ---------------------------------------------------

    def _gen_fit(self, difficulty: int, _allow_fallback: bool = False) -> dict | None:
        """FIT task: approximate a pinned target function.

        Inputs are freshly sampled each call (uniform [lo,hi] per dim);
        targets are computed live by target_fn — real math, and held-out
        samples grade generalization. Not guaranteed solvable (that's the
        point of approximation): score is driven by mae/max_err/fit_score.
        """
        if self.target_fn is None:
            return None
        K = self.fit_k
        rng = self._rng
        lo, hi = self.input_lo, self.input_hi
        n_show = self.n_fit_cases
        inputs = [[rng.uniform(lo, hi) for _ in range(K)] for _ in range(n_show)]
        ho_inputs = [[rng.uniform(lo, hi) for _ in range(K)]
                     for _ in range(self.n_heldout * 2)]
        x = torch.tensor(inputs + ho_inputs, dtype=torch.float64)
        y = self.target_fn(x)
        y = y.reshape(x.shape[0], -1).squeeze(-1)  # [N] (K=1 → squeeze)
        if not torch.isfinite(y).all():
            if _allow_fallback:
                y = torch.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                return None
        targets = y[:n_show].tolist()
        ho_targets = y[n_show:].tolist()
        return {"kind": KIND_FIT, "K": K, "inputs": inputs,
                "targets": [float(v) for v in targets],
                "ho_inputs": ho_inputs,
                "ho_targets": [float(v) for v in ho_targets],
                "require_inputs": True, "difficulty": difficulty,
                "text": (f"Approximate {self.fit_name}(x) on "
                         f"[{lo:g},{hi:g}] — {n_show} sampled points "
                         f"+ held-out."),
                "prog": None}

    @staticmethod
    def _targets_ok(task: dict) -> bool:
        vals = task["targets"] + task["ho_targets"]
        return all(abs(v) < 1e8 and v == v and abs(v) != float("inf") for v in vals)

    def _rand_inputs(self, K: int, n: int, difficulty: int,
                     ints: bool) -> list[list[float]]:
        rng = self._rng
        span = 8 + 4 * difficulty
        if ints:
            return [[float(rng.randint(-span, span)) for _ in range(K)]
                    for _ in range(n)]
        return [[rng.uniform(-span, span) for _ in range(K)] for _ in range(n)]

    def _n_shown(self, difficulty: int) -> int:
        lo, hi = self.n_shown
        return min(max(lo, 1 + difficulty // 2), hi)

    # ---- hidden oracle program ----------------------------------------

    def _rand_hidden_prog(self, K: int, difficulty: int, int_only: bool,
                          oracle_consts: list[float] | None = None,
                          ) -> dict | None:
        """Random program guaranteed to read every input (when K>0).

        Returns a program dict; its value at the inputs defines the task's
        hidden function / target. Construction guarantees solvability: the
        oracle lives in the same ISA the searcher decodes into.
        """
        rng = self._rng
        S = self.spec_vm.n_steps
        depth = min(2 + difficulty, S - 1)
        M_h = 4
        consts = [float(rng.randint(-9, 9)) for _ in range(M_h)]
        if oracle_consts:
            consts += [float(c) for c in oracle_consts]
            M_h += len(oracle_consts)
        ops, aa, bb = [], [], []
        nxt = K + M_h  # next writable register

        # Consume every input once up front → uses_inputs is reachable.
        for i in range(K):
            op = rng.choice(_BIN_INT if int_only else [0, 1, 2])
            ops.append(op)
            aa.append(i)
            bb.append(rng.randrange(K, K + M_h) if rng.random() < 0.5
                      else rng.randrange(K))
            nxt += 1
        for s in range(K, depth):
            op = rng.choice(_BIN_INT_PMOD if int_only
                            else list(range(N_OPS)))
            vis = nxt
            ops.append(op)
            aa.append(rng.randrange(vis))
            bb.append(rng.randrange(vis) if OPS[op][1] == 2 else 0)
            nxt += 1
        if not ops or nxt - (K + M_h) > S:
            return None
        return program_from_lists(ops, aa, bb, consts, nxt - 1, K)

    def _eval_oracle(self, prog: dict, inputs: list[list[float]]) -> list[float]:
        batched = stack_programs([prog], device="cpu")
        out = run_programs(batched, torch.tensor(inputs, dtype=torch.float64))
        return out["out"][0].tolist()

    # ---- kind generators ----------------------------------------------

    def _gen_func(self, difficulty: int) -> dict | None:
        K = self._rng.randint(2, self.spec_vm.max_inputs)
        prog = self._rand_hidden_prog(K, difficulty, int_only=False)
        if prog is None:
            return None
        n_show = self._n_shown(difficulty)
        inputs = self._rand_inputs(K, n_show, difficulty, ints=False)
        ho_inputs = self._rand_inputs(K, self.n_heldout, difficulty, ints=False)
        targets = self._eval_oracle(prog, inputs)
        ho_targets = self._eval_oracle(prog, ho_inputs)
        expr = render(prog, K)
        return {"kind": KIND_FUNC, "K": K, "inputs": inputs,
                "targets": targets, "ho_inputs": ho_inputs,
                "ho_targets": ho_targets, "require_inputs": True,
                "difficulty": difficulty,
                "text": f"Find f reproducing {n_show} input->output pairs "
                        f"(difficulty {difficulty}). Hidden: {expr}",
                "prog": prog}

    def _gen_goal(self, difficulty: int) -> dict | None:
        if self.goal_target is not None:
            T = float(self.goal_target)
        else:
            T = float(self._rng.randint(65, min(10 ** (1 + difficulty), 10**6)))
        return {"kind": KIND_GOAL, "K": 0, "inputs": [[]],
                "targets": [T], "ho_inputs": [], "ho_targets": [],
                "require_inputs": False, "difficulty": difficulty,
                "text": f"A must equal {T:g}. No inputs — compose it.",
                "prog": None}

    def _gen_goal_io(self, difficulty: int) -> dict | None:
        if self.goal_inputs is not None:
            K = len(self.goal_inputs)
            u0 = [float(v) for v in self.goal_inputs]
        else:
            K = self._rng.randint(1, self.spec_vm.max_inputs)
            u0 = None
        # Reserve one oracle const slot (appended last so no register
        # collision) — used for the target-forcing shift when pinned.
        h = self._rand_hidden_prog(K, difficulty, int_only=True,
                                   oracle_consts=[0.0])
        if h is None:
            return None
        n_show = self._n_shown(difficulty)
        inputs = ([u0] if u0 is not None else []) + \
            self._rand_inputs(K, n_show - (1 if u0 else 0), difficulty, ints=True)
        ho_inputs = self._rand_inputs(K, self.n_heldout, difficulty, ints=True)

        if self.goal_target is not None and u0 is not None:
            # Target-forcing wrapper: f(u) = h(u) + (T - h(u0)). The oracle
            # embeds C2 exactly; the searcher must compose it from bounded
            # consts — reachable because everything stays integer.
            T = float(self.goal_target)
            C2 = T - self._eval_oracle(h, [u0])[0]
            M_h = h["consts"].numel()
            c_addr = K + M_h - 1                       # the reserved const slot
            consts = h["consts"].tolist()
            consts[-1] = C2
            ops = h["ops"].tolist() + [0]              # add
            aa = h["a_addr"].tolist() + [int(h["out_sel"].item())]
            bb = h["b_addr"].tolist() + [c_addr]
            prog = program_from_lists(ops, aa, bb, consts, len(ops) - 1, K)
        else:
            prog = h

        targets = self._eval_oracle(prog, inputs)
        ho_targets = self._eval_oracle(prog, ho_inputs)
        if self.goal_target is not None and u0 is not None:
            targets[0] = float(self.goal_target)  # pinned pair is exact by spec
        pairs = "; ".join(f"A={t:g} at {u}" for u, t in zip(inputs, targets))
        return {"kind": KIND_GOAL_IO, "K": K, "inputs": inputs,
                "targets": targets, "ho_inputs": ho_inputs,
                "ho_targets": ho_targets, "require_inputs": True,
                "difficulty": difficulty,
                "text": f"A must equal: {pairs} (+ held-out pairs).",
                "prog": prog}

    # ------------------------------------------------------------------

    def to_cpu(self) -> "MathProgDomain":
        return MathProgDomain(
            n_steps=self.spec_vm.n_steps, n_consts=self.spec_vm.n_consts,
            max_inputs=self.spec_vm.max_inputs,
            difficulty_range=self.difficulty_range,
            tasks_per_eval=self.tasks_per_eval, n_shown=self.n_shown,
            n_heldout=self.n_heldout, goal_target=self.goal_target,
            goal_inputs=self.goal_inputs, kind_weights=self.kind_weights,
            target_fn=self.target_fn, fit_k=self.fit_k,
            input_lo=self.input_lo, input_hi=self.input_hi,
            n_fit_cases=self.n_fit_cases,
            op_ids=list(self.spec_vm.op_ids) if self.spec_vm.op_ids else None,
            seed=self._rng.randint(0, 1_000_000), device=torch.device("cpu"))
