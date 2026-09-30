"""Tests for the math_prog program-synthesis domain + closed-ISA VM."""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, r"D:\windsurf\ForgeAI")

import pytest
import torch

from forge.evolution.program_vm import (
    OPS, ProgramSpec, decode_params, encode_program, program_from_lists,
    prog_from_config, render, run_programs, stack_programs, steps_used,
)
from forge.evolution.domains.math_prog_domain import (
    MathProgDomain, KIND_FUNC, KIND_GOAL, KIND_GOAL_IO,
)
from forge.evolution.simulators.math_prog_sim import program_metrics

SPEC = ProgramSpec(n_steps=12, n_consts=8, max_inputs=4)
CPU = torch.device("cpu")


def _prog(ops, a, b, consts, out_sel, K, S=12, M=8):
    """Pad a short program to the standard layout."""
    ops = list(ops) + [0] * (S - len(ops))
    a = list(a) + [0] * (S - len(a))
    b = list(b) + [0] * (S - len(b))
    consts = list(consts) + [0.0] * (M - len(consts))
    return program_from_lists(ops, a, b, consts, out_sel, K)


def _task(kind=KIND_FUNC, K=2, inputs=None, targets=None,
          ho_inputs=None, ho_targets=None, require_inputs=True,
          difficulty=3):
    inputs = inputs if inputs is not None else [[2.0, 3.0]]
    targets = targets if targets is not None else [5.0]
    return {"kind": kind, "K": K, "inputs": inputs, "targets": targets,
            "ho_inputs": ho_inputs or [], "ho_targets": ho_targets or [],
            "require_inputs": require_inputs, "difficulty": difficulty,
            "text": "test", "prog": None}


# ---------------------------------------------------------------------------
# VM semantics
# ---------------------------------------------------------------------------

# Layout for hand-built programs: K=4 inputs (0-3), M=8 consts (4-11),
# step registers r_s at 12+s — matching spec_vm.max_inputs decode layout.
R0 = 12

class TestVM:
    def test_add(self):
        # r0 = x0 + x1; A = r0.
        prog = _prog([0], [0], [1], [0.0] * 8, R0, K=4)
        out = run_programs(stack_programs([prog]),
                           torch.tensor([[2.0, 3.0, 0.0, 0.0]]))["out"]
        assert out[0, 0].item() == pytest.approx(5.0)

    def test_compose_constant(self):
        # 794 = 64*12 + 26: r0 = c2*c3, r1 = r0 + c4, A = r1
        prog = _prog([2, 0], [6, R0], [7, 8],
                     [0.0, 1.0, 64.0, 12.0, 26.0, 0, 0, 0], R0 + 1, K=4)
        out = run_programs(stack_programs([prog]),
                           torch.zeros(1, 0))["out"]
        assert out[0, 0].item() == pytest.approx(794.0)

    def test_protected_div(self):
        # r0 = x0 / x1 with x1=0 → finite fallback, no crash
        pdiv = next(i for i, o in enumerate(OPS) if o[0] == "pdiv")
        prog = _prog([pdiv], [0], [1], [0.0] * 8, R0, K=4)
        out = run_programs(stack_programs([prog]),
                           torch.tensor([[5.0, 0.0, 0.0, 0.0]]))["out"]
        assert torch.isfinite(out).all()

    def test_batched_different_programs(self):
        p_add = _prog([0], [0], [1], [0.0] * 8, R0, K=4)
        p_mul = _prog([2], [0], [1], [0.0] * 8, R0, K=4)
        progs = stack_programs([p_add, p_mul])
        out = run_programs(progs, torch.tensor([[2.0, 3.0, 0.0, 0.0]]))["out"]
        assert out[0, 0].item() == pytest.approx(5.0)
        assert out[1, 0].item() == pytest.approx(6.0)

    def test_used_in(self):
        prog = _prog([0], [0], [1], [0.0] * 8, R0, K=4)
        res = run_programs(stack_programs([prog]),
                           torch.tensor([[1.0, 2.0, 9.0]]))  # task K=3
        assert res["used_in"][0].tolist() == [True, True, False]

    def test_decode_bounds(self):
        """Every decoded program is well-formed: addrs < visible prefix."""
        rng = torch.Generator().manual_seed(0)
        for _ in range(50):
            p = torch.rand(SPEC.output_dim(), generator=rng)
            prog = decode_params(p, SPEC, SPEC.max_inputs)
            for s in range(SPEC.n_steps):
                visible = SPEC.max_inputs + SPEC.n_consts + s
                assert 0 <= prog["a_addr"][s].item() < visible
                assert 0 <= prog["b_addr"][s].item() < visible
            assert prog["consts"][0].item() == 0.0
            assert prog["consts"][1].item() == 1.0
            assert prog["consts"][2:].abs().max().item() <= 64.0

    def test_steps_used(self):
        # A = r1 = r0 * x1; r0 = x0 + x1 → 2 steps used
        prog = _prog([0, 2], [0, R0], [1, 1], [0.0] * 8, R0 + 1, K=4)
        assert steps_used(prog, 4, 8) == 2

    def test_render(self):
        prog = _prog([0], [0], [1], [0.0] * 8, R0, K=4)
        assert "A = r0" in render(prog, K=4)


# ---------------------------------------------------------------------------
# Metrics — honesty checks
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_correct_program_scores_exact(self):
        # f(x0,x1) = x0 + x1; task encodes that mapping
        prog = _prog([0], [0], [1], [0.0] * 8, R0, K=4)
        task = _task(K=2, inputs=[[2.0, 3.0], [0.0, 1.0]],
                     targets=[5.0, 1.0],
                     ho_inputs=[[4.0, 4.0], [1.0, -1.0]],
                     ho_targets=[8.0, 0.0])
        m = program_metrics(prog, [task], K_layout=4, M=8)
        assert m["exact"] == 1.0 and m["heldout_exact"] == 1.0
        assert m["const_cheat"] == 0.0

    def test_const_program_fails_heldout(self):
        # Program returns constant 5 — nails shown targets, fails held-out.
        # Operands point at const regs (4,5) so no input is ever read.
        prog = _prog([0], [4], [5], [5.0, 0, 0, 0, 0, 0, 0, 0], 4, K=4)  # A = c0=5
        task = _task(K=2, inputs=[[1.0, 1.0]], targets=[5.0],
                     ho_inputs=[[3.0, 3.0], [0.0, 0.0]],
                     ho_targets=[6.0, 0.0])
        m = program_metrics(prog, [task], K_layout=4, M=8)
        assert m["train_exact"] == 1.0
        assert m["heldout_exact"] < 1.0
        assert m["const_cheat"] == 1.0   # ignores inputs on input-dependent task

    def test_degenerate_goal_flagged(self):
        # Goal task: A must equal 794, but program outputs raw const register.
        prog = _prog([], [], [], [0.0] * 8, 5, K=4)  # A = c1 = 1.0, no steps
        task = _task(kind=KIND_GOAL, K=0, inputs=[[]], targets=[794.0],
                     require_inputs=False)
        m = program_metrics(prog, [task], K_layout=4, M=8)
        assert m["degenerate"] == 1.0

    def test_goal_solver_exact(self):
        prog = _prog([2, 0], [6, R0], [7, 8],
                     [0.0, 1.0, 64.0, 12.0, 26.0, 0, 0, 0], R0 + 1, K=4)
        task = _task(kind=KIND_GOAL, K=0, inputs=[[]], targets=[794.0],
                     require_inputs=False)
        m = program_metrics(prog, [task], K_layout=4, M=8)
        assert m["exact"] == 1.0 and m["degenerate"] == 0.0


# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

class TestDomain:
    def test_registry(self):
        from forge.evolution.domains import DOMAINS
        assert "MathProgDomain" in DOMAINS

    def test_output_dim(self):
        d = MathProgDomain()
        assert d.output_dim() == 3 * 12 + 8 + 1

    def test_decode_encode_roundtrip(self):
        d = MathProgDomain()
        p = torch.rand(d.output_dim())
        cfg = d.decode(p)
        p2 = d.encode(cfg)
        assert torch.allclose(p, p2, atol=1e-6)

    def test_tasks_solvable(self):
        """Hidden oracle solves its own FUNC/GOAL_IO tasks (solvable by construction)."""
        d = MathProgDomain(seed=7)
        for _ in range(12):
            task = d._gen_task()
            if task["prog"] is None:  # pure goal — oracle trivially constructible
                continue
            m = program_metrics(task["prog"], [task], K_layout=4, M=8)
            # The oracle may read beyond decode bounds (its own consts) but
            # must hit its own targets.
            assert m["heldout_exact"] >= 0.99 or task["kind"] == KIND_GOAL

    def test_evaluate_metadata_json_safe(self):
        d = MathProgDomain(seed=1)
        cfg = d.decode(torch.rand(d.output_dim()))
        res = d.evaluate(cfg)
        assert isinstance(res["score"], float)
        json.dumps(res["metadata"])
        json.dumps(cfg)

    def test_pinned_goal_794(self):
        """'A must equal 794' — pinned pure-goal mode."""
        d = MathProgDomain(goal_target=794.0, seed=3)
        task = d._gen_task()
        assert task["kind"] == KIND_GOAL
        assert task["targets"] == [794.0]
        solver = _prog([2, 0], [6, R0], [7, 8],
                       [0.0, 1.0, 64.0, 12.0, 26.0, 0, 0, 0], R0 + 1, K=4)
        cfg = {"ops": solver["ops"].tolist(), "a_addr": solver["a_addr"].tolist(),
               "b_addr": solver["b_addr"].tolist(),
               "consts": solver["consts"].tolist(),
               "out_sel": int(solver["out_sel"]), "n_inputs": 4}
        d.current_tasks = [task]
        res = d.evaluate(cfg)
        # Pinned solver must hit the target on every goal task it sees
        assert res["metadata"]["train_exact"] == 1.0

    def test_pinned_goal_io(self):
        """Pinned target + pinned inputs → GOAL_IO with first pair at target."""
        d = MathProgDomain(goal_target=794.0, goal_inputs=[12.0, 7.0], seed=5)
        task = d._gen_task()
        assert task["kind"] == KIND_GOAL_IO
        assert task["K"] == 2
        assert task["inputs"][0] == [12.0, 7.0]
        assert task["targets"][0] == pytest.approx(794.0)
        assert task["ho_targets"]  # held-out pairs exist
        # Oracle program reproduces every shown + heldout pair
        m = program_metrics(task["prog"], [task], K_layout=4, M=8)
        assert m["heldout_exact"] >= 0.99

    def test_batch_matches_sequential_structure(self):
        d = MathProgDomain(seed=11)
        cfgs = [d.decode(torch.rand(d.output_dim())) for _ in range(8)]
        results = d.batch_evaluate(cfgs)
        assert len(results) == 8
        for r in results:
            assert isinstance(r["score"], float)
            assert len(r["behavioral"]) == 2
            json.dumps(r["metadata"])

    def test_to_cpu(self):
        d = MathProgDomain(goal_target=794.0, seed=2)
        c = d.to_cpu()
        assert c.goal_target == 794.0
        assert c._device.type == "cpu"

    def test_fit_mode(self):
        """FIT tasks: fixed target fn, fresh inputs, held-out grading."""
        d = MathProgDomain(target_fn=torch.tanh, fit_k=1,
                           input_lo=-8.0, input_hi=8.0, seed=4)
        task = d._gen_task()
        assert task["kind"] == 3 and task["K"] == 1
        assert len(task["inputs"]) == d.n_fit_cases
        assert len(task["ho_inputs"]) > 0
        # Task targets really are tanh of the sampled inputs
        x = torch.tensor(task["inputs"], dtype=torch.float64)
        assert torch.allclose(torch.tanh(x).flatten(),
                              torch.tensor(task["targets"],
                                           dtype=torch.float64), atol=1e-9)
        # A perfect program (unreachable here — tanh not in cheap ISA) would
        # score; a linear program must produce finite, non-degenerate metrics.
        cfg = d.decode(torch.rand(d.output_dim()))
        res = d.evaluate(cfg)
        assert isinstance(res["score"], float)
        assert res["behavioral"][0] == 3.0  # FIT bin
        json.dumps(res["metadata"])

    def test_fit_known_approximation(self):
        """x/(1+|x|) approximates tanh loosely — verify metrics reflect it."""
        d = MathProgDomain(target_fn=torch.tanh, fit_k=1,
                           input_lo=-2.0, input_hi=2.0, seed=4)
        # r0 = abs(x0); r1 = r0 + 1; A = x0 / r1  →  x/(1+|x|)
        absop = next(i for i, o in enumerate(OPS) if o[0] == "abs")
        pdiv = next(i for i, o in enumerate(OPS) if o[0] == "pdiv")
        prog = _prog([absop, 0, pdiv], [0, R0, 0], [0, 5, R0 + 1],
                     [0.0, 1.0] + [0.0] * 6, R0 + 2, K=4)
        task = d._gen_task()
        m = program_metrics(prog, [task], K_layout=4, M=8)
        assert m["mae"] < 0.3           # x/(1+|x|) hugs tanh within ~0.3
        assert m["max_err"] < 0.45
        assert m["fit_score"] > 0.3
        assert m["uses_inputs"] == 1.0

    def test_op_whitelist(self):
        cheapdiv = [i for i, o in enumerate(OPS)
                    if o[0] in ("add", "sub", "mul", "pdiv", "pmod",
                                "min", "max", "neg", "abs", "floor", "round")]
        d = MathProgDomain(op_ids=cheapdiv)
        rng = torch.Generator().manual_seed(1)
        banned = {i for i in range(len(OPS))} - set(cheapdiv)
        for _ in range(200):
            prog = decode_params(torch.rand(d.output_dim(), generator=rng),
                                 d.spec_vm, 4)
            assert not (set(prog["ops"].tolist()) & banned)

    def test_throughput_smoke(self):
        """Batched eval must beat 100 tries/sec by a wide margin."""
        d = MathProgDomain(seed=0)
        cfgs = [d.decode(torch.rand(d.output_dim())) for _ in range(64)]
        d.batch_evaluate(cfgs[:8])  # warmup
        t0 = time.perf_counter()
        n = 0
        for _ in range(5):
            d.batch_evaluate(cfgs)
            n += len(cfgs)
        rate = n / (time.perf_counter() - t0)
        assert rate > 100, f"only {rate:.0f} tries/sec"

    def test_elite_mutants(self):
        """Program-space mutants of elites: well-formed, whitelisted, differ."""
        from types import SimpleNamespace
        from forge.evolution.program_vm import config_dict
        d = MathProgDomain(target_fn=torch.tanh, fit_k=1, seed=9)
        S, M, K = d.spec_vm.n_steps, d.spec_vm.n_consts, d.spec_vm.max_inputs
        parent = config_dict(_prog([0, 2], [0, R0], [1, 5],
                                   [0.0, 1.0, 7.0, 3.0] + [0.0] * 4,
                                   R0 + 1, K=4))
        archive = SimpleNamespace(history=[
            SimpleNamespace(score=90.0, config=parent),
            SimpleNamespace(score=10.0, config=config_dict(
                _prog([1], [0], [1], [0.0] * 8, R0, K=4))),
        ])
        mutants = d.propose_elite_mutants(archive, 24, gen=1)
        assert len(mutants) == 24
        n_ops = len(OPS) if not d.spec_vm.op_ids else len(d.spec_vm.op_ids)
        for m in mutants:
            prog = prog_from_config(m)
            assert prog["consts"][0].item() == 0.0   # pinned consts survive
            assert prog["consts"][1].item() == 1.0
            for s in range(S):
                assert prog["a_addr"][s].item() < K + M + s   # prefix rule
                assert prog["b_addr"][s].item() < K + M + s
            # executable
            out = run_programs(stack_programs([prog]),
                               torch.zeros(1, K, dtype=torch.float64))["out"]
            assert torch.isfinite(out).all()
        # differ from parent but structurally similar (bounded edit distance)
        diffs = sum(1 for m in mutants
                    if tuple(m["ops"]) != tuple(parent["ops"])
                    or m["out_sel"] != parent["out_sel"])
        assert diffs > 0
        # whitelisted ops only when whitelist set
        cheap = [i for i, o in enumerate(OPS)
                 if o[0] in ("add", "sub", "mul", "pdiv")]
        d2 = MathProgDomain(op_ids=cheap, seed=3)
        muts = d2.propose_elite_mutants(archive, 16, gen=1)
        allowed = set(cheap)
        for m in muts:
            assert set(m["ops"]) <= allowed

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_cuda_exec(self):
        """VM runs batched on CUDA and matches CPU results bit-for-bit."""
        prog = _prog([0, 2], [0, R0], [1, 1], [0.0] * 8, R0 + 1, K=4)
        inp = torch.tensor([[2.0, 3.0, 1.0, 1.0], [4.0, 0.5, 0.0, 0.0]])
        cpu_out = run_programs(stack_programs([prog]), inp)["out"]
        gpu_out = run_programs(stack_programs([prog], device="cuda"),
                               inp.to("cuda"))["out"]
        assert torch.equal(cpu_out, gpu_out.cpu())
        d = MathProgDomain(device="cuda", seed=1)
        cfgs = [d.decode(torch.rand(d.output_dim())) for _ in range(32)]
        results = d.batch_evaluate(cfgs)
        assert len(results) == 32
        assert all(isinstance(r["score"], float) for r in results)
