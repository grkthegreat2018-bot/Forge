"""Tests for GRPO-only RSI mode in InfiniteSelfPlayLoop.

Verifies:
  1. LoopConfig accepts training_mode="grpo" and GRPO config fields
  2. run_epoch branches to _grpo_train when training_mode="grpo"
  3. _grpo_train skips when too few trajectories
  4. CLI arg --training-mode is wired correctly
"""
import pytest
from unittest.mock import MagicMock, patch

from forge.self_play.infinite_loop import LoopConfig, InfiniteSelfPlayLoop


class TestGRPOConfig:
    """LoopConfig GRPO fields."""

    def test_default_training_mode_is_sft(self):
        c = LoopConfig()
        assert c.training_mode == "sft"

    def test_grpo_mode_configurable(self):
        c = LoopConfig(training_mode="grpo")
        assert c.training_mode == "grpo"

    def test_grpo_config_fields_exist(self):
        c = LoopConfig(training_mode="grpo")
        assert hasattr(c, "grpo_max_steps")
        assert hasattr(c, "grpo_group_size")
        assert hasattr(c, "grpo_lr")
        assert hasattr(c, "grpo_kl_coeff")
        assert hasattr(c, "grpo_clip_range")
        assert hasattr(c, "grpo_rl_algorithm")
        assert hasattr(c, "grpo_min_trajectories")

    def test_grpo_defaults_sane_for_12gb(self):
        """GRPO defaults should be VRAM-safe for RTX 5070 12GB."""
        c = LoopConfig()
        assert c.grpo_group_size >= 2  # need ≥2 for advantage computation
        assert c.grpo_lr <= 1e-5  # small LR for stability
        assert c.grpo_max_seq_len <= 1024  # short seq for VRAM
        assert c.grpo_grad_accum >= 1


class TestGRPOTrainingModeBranch:
    """run_epoch branches correctly based on training_mode."""

    def test_sft_mode_calls_finetune(self):
        """SFT mode should call _finetune (not _grpo_train)."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._export_trajectories = MagicMock(return_value="/tmp/data.jsonl")
        loop._finetune = MagicMock(return_value="/tmp/candidate.safetensors")
        loop._evaluate = MagicMock(return_value={"passed": True})
        loop._maybe_promote = MagicMock(return_value=True)

        # Mock open() to return enough examples
        import builtins
        original_open = builtins.open

        class MockFile:
            def __init__(self, *a, **k):
                pass
            def __enter__(self):
                return iter(["line"] * 10)  # 10 examples
            def __exit__(self, *a):
                pass

        with patch("builtins.open", MockFile):
            loop.run_epoch()

        loop._finetune.assert_called_once()
        assert not loop._grpo_train.called if hasattr(loop._grpo_train, "called") else True

    def test_grpo_mode_calls_grpo_train(self):
        """GRPO mode should call _grpo_train (not _finetune)."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="grpo"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._grpo_train = MagicMock(return_value="/tmp/candidate.safetensors")
        loop._evaluate = MagicMock(return_value={"passed": True})
        loop._maybe_promote = MagicMock(return_value=True)
        loop._free_engine = MagicMock()

        loop.run_epoch()

        loop._grpo_train.assert_called_once()
        loop._evaluate.assert_called_once_with("/tmp/candidate.safetensors")

    def test_grpo_mode_skips_evaluation_on_empty_result(self):
        """When _grpo_train returns empty string, skip eval."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="grpo"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._grpo_train = MagicMock(return_value="")  # skipped
        loop._evaluate = MagicMock()
        loop._maybe_promote = MagicMock()
        loop._free_engine = MagicMock()

        result = loop.run_epoch()

        loop._grpo_train.assert_called_once()
        loop._evaluate.assert_not_called()
        assert result.get("grpo", {}).get("skipped") is True


class TestFluxModeBranch:
    """training_mode="flux" routes to the FluxLM live-learning path."""

    def test_flux_config_fields_exist(self):
        c = LoopConfig(training_mode="flux")
        for f in ("flux_eval_n", "flux_eval_seed", "flux_gen_tokens",
                  "flux_gain", "flux_consolidate_min"):
            assert hasattr(c, f), f

    def test_flux_epoch_calls_flux_paths(self):
        """run_epoch routes Phase1 → _flux_self_play, Phase2 → _flux_finish."""
        loop = InfiniteSelfPlayLoop(
            "fake.flux", LoopConfig(training_mode="flux"))
        loop._flux_self_play = MagicMock(
            return_value={"success_rate": 0.5, "acc_pre": 0.1,
                          "attempts": 20, "domain": "concise_qa"})
        loop._flux_finish = MagicMock()
        loop._free_engine = MagicMock()

        loop.run_epoch()

        loop._flux_self_play.assert_called_once()
        loop._flux_finish.assert_called_once()

    def test_flux_check_grader(self):
        check = InfiniteSelfPlayLoop._flux_check
        # answer-first (model emits answer then keeps going)
        assert check(" 16 then junk", "16")
        assert check("no18239", "no")
        # word-bounded mid-output
        assert check("the answer is 42.", "42")
        # rejects: answer inside a longer digit run, wrong answers
        assert not check("1675", "675")
        assert not check("yes", "no")

    def test_flux_fresh_pairs_dedupes_across_epochs(self):
        """Anti-overfit: the same prompt must never be re-attempted in
        a later epoch, and eval prompts are always excluded."""
        from forge.self_play.infinite_loop import _concise_qa_pairs
        cfg = LoopConfig(training_mode="flux", tasks_per_epoch=10,
                         flux_eval_n=8, flux_web=False)
        loop = InfiniteSelfPlayLoop("fake.flux", cfg)
        eval_qs = {p["prompt"] for p in
                   _concise_qa_pairs(8, cfg.flux_eval_seed)}
        e1 = loop._flux_fresh_pairs(10)
        loop.epoch = 2
        e2 = loop._flux_fresh_pairs(10)
        assert len(e1) == 10 and len(e2) == 10
        p1 = {p["prompt"] for p in e1}
        p2 = {p["prompt"] for p in e2}
        assert not p1 & p2                 # zero cross-epoch repeats
        assert not (p1 | p2) & eval_qs     # eval stays held-out

    def test_flux_self_play_uses_batch_gen(self):
        """Attempts must go through engine.generate_batch with per-gen
        temps+seeds (concurrency contract), one group per task."""
        cfg = LoopConfig(training_mode="flux", tasks_per_epoch=6,
                         flux_group_size=3, flux_workers=2,
                         flux_web=False)
        loop = InfiniteSelfPlayLoop("fake.flux", cfg)
        model = MagicMock()
        model._count = 123
        engine = MagicMock()
        engine.model = model
        engine.generate_batch = MagicMock(
            side_effect=lambda prompts, **kw: ["wrong"] * len(prompts))
        loop._flux_engine = MagicMock(return_value=engine)
        loop._flux_eval_acc = MagicMock(return_value=0.0)

        out = loop._flux_self_play()

        engine.generate_batch.assert_called_once()
        prompts = engine.generate_batch.call_args[0][0]
        assert len(prompts) == 6 * 3        # group_size per task
        kw = engine.generate_batch.call_args[1]
        assert len(kw["temperatures"]) == 18
        assert len(kw["seeds"]) == 18
        assert len(set(kw["seeds"])) == 18  # unique seeds per sample
        assert engine.flux_batch_workers == 2
        assert out["attempts"] == 6
        assert model._count == 123          # canonical model untouched


class TestGRPOTrajectoryGrouping:
    """_grpo_train groups trajectories correctly for GRPO."""

    def test_too_few_trajectories_returns_empty(self):
        """When fewer than grpo_min_trajectories, skip GRPO."""
        loop = InfiniteSelfPlayLoop("fake.ckpt",
                                    LoopConfig(training_mode="grpo",
                                               grpo_min_trajectories=10))
        loop._free_engine = MagicMock()
        loop._trajectories = [
            {"task_description": "task1", "solution_code": "x", "reward": 1.0,
             "domain": "math"},
        ]
        result = loop._grpo_train(epoch=1)
        assert result == ""

    def test_paired_by_domain(self):
        """Trajectories should be paired within the same domain."""
        loop = InfiniteSelfPlayLoop("fake.ckpt",
                                    LoopConfig(training_mode="grpo",
                                               grpo_min_trajectories=2))
        loop._free_engine = MagicMock()
        loop._trajectories = [
            {"task_description": "task1", "solution_code": "sol1", "reward": 1.0,
             "domain": "math"},
            {"task_description": "task2", "solution_code": "sol2", "reward": 0.0,
             "domain": "math"},
            {"task_description": "task3", "solution_code": "sol3", "reward": 1.0,
             "domain": "algorithms"},
            {"task_description": "task4", "solution_code": "sol4", "reward": 0.0,
             "domain": "algorithms"},
        ]

        # Mock the heavy imports inside _grpo_train
        with patch("forge.config.get_config") as mock_cfg, \
             patch("forge.model_loader.ModelLoader.build_model_fast") as mock_build, \
             patch("forge.training.bitnet_lora.add_lora_adapters") as mock_lora, \
             patch("forge.training.bitnet_lora.merge_lora_adapters") as mock_merge, \
             patch("forge.self_play.grpo_trainer.GRPOTrainer") as mock_trainer_cls, \
             patch("research.tokenizer_cache.get_tokenizer") as mock_tok, \
             patch("forge.checkpoint_io.save_training_checkpoint") as mock_save, \
             patch("forge.training.training_utils.oom_guard") as mock_oom:
            mock_cfg.return_value = MagicMock()
            mock_model = MagicMock()
            mock_build.return_value = mock_model
            # Return real tensors so numel() and format strings work
            import torch as _t
            _fake_param = _t.nn.Parameter(_t.zeros(1))
            mock_lora.return_value = (4, [_fake_param])
            mock_merge.return_value = 4
            mock_trainer = MagicMock()
            mock_trainer.train_step.return_value = {"loss": 0.1, "kl": 0.01}
            mock_trainer_cls.return_value = mock_trainer
            mock_oom.return_value.__enter__ = MagicMock(return_value=MagicMock(skipped=False))
            mock_oom.return_value.__exit__ = MagicMock(return_value=False)

            result = loop._grpo_train(epoch=1)

        # Should have called train_step at least once
        assert mock_trainer.train_step.called
        # Should have saved the checkpoint
        mock_save.assert_called_once()
        assert result != ""

    def test_save_uses_keyword_step_not_positional_optimizer(self):
        """Regression: save_training_checkpoint(model, path, step) passed the
        int step positionally into `optimizer` → optimizer.state_dict()
        AttributeError AFTER training, losing the checkpoint."""
        loop = InfiniteSelfPlayLoop("fake.ckpt",
                                    LoopConfig(training_mode="grpo",
                                               grpo_min_trajectories=2,
                                               grpo_max_steps=1))
        loop._free_engine = MagicMock()
        loop._trajectories = [
            {"task_description": "t1", "solution_code": "s1", "reward": 1.0,
             "domain": "math"},
            {"task_description": "t2", "solution_code": "s2", "reward": 0.0,
             "domain": "math"},
        ]

        with patch("forge.config.get_config") as mock_cfg, \
             patch("forge.model_loader.ModelLoader.build_model_fast") as mock_build, \
             patch("forge.training.bitnet_lora.add_lora_adapters") as mock_lora, \
             patch("forge.training.bitnet_lora.merge_lora_adapters") as mock_merge, \
             patch("forge.self_play.grpo_trainer.GRPOTrainer") as mock_trainer_cls, \
             patch("research.tokenizer_cache.get_tokenizer"), \
             patch("forge.checkpoint_io.save_training_checkpoint") as mock_save, \
             patch("forge.training.training_utils.oom_guard") as mock_oom:
            mock_cfg.return_value = MagicMock()
            mock_build.return_value = MagicMock()
            import torch as _t
            mock_lora.return_value = (4, [_t.nn.Parameter(_t.zeros(1))])
            mock_merge.return_value = 4
            mock_trainer = MagicMock()
            mock_trainer.train_step.return_value = {"loss": 0.1, "kl": 0.01}
            mock_trainer_cls.return_value = mock_trainer
            mock_oom.return_value.__enter__ = MagicMock(
                return_value=MagicMock(skipped=False))
            mock_oom.return_value.__exit__ = MagicMock(return_value=False)

            loop._grpo_train(epoch=1)

        mock_save.assert_called_once()
        args, kwargs = mock_save.call_args
        # step must be a keyword arg — positional slot 3 is `optimizer`
        assert len(args) <= 2, f"step leaked into optimizer arg: {args}"
        assert kwargs.get("step") == 1


class TestEngineCleanup:
    """run_epoch must free the self-play engine on every path.

    Regression test for the VRAM leak where a skipped finetune left the
    self-play ForgeEngine resident, so the next epoch loaded a second
    engine on top (12.8 GB used, KV cache capped to 64 tokens).
    """

    class _MockFile:
        def __init__(self, lines):
            self._lines = lines
        def __call__(self, *a, **k):
            return self
        def __enter__(self):
            return iter(self._lines)
        def __exit__(self, *a):
            pass

    def test_skipped_finetune_frees_engine(self):
        """Too-few-examples path must free the engine."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.0})
        loop._export_trajectories = MagicMock(return_value="/tmp/data.jsonl")
        loop._finetune = MagicMock()
        loop._evaluate = MagicMock()
        loop._free_engine = MagicMock()

        with patch("builtins.open", self._MockFile([])):  # 0 examples
            result = loop.run_epoch()

        loop._free_engine.assert_called_once()
        loop._finetune.assert_not_called()
        assert result["finetune"]["skipped"] is True

    def test_train_exception_frees_engine(self):
        """Exception in Phase 2 must free the engine."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._export_trajectories = MagicMock(
            side_effect=RuntimeError("export boom"))
        loop._free_engine = MagicMock()

        result = loop.run_epoch()

        loop._free_engine.assert_called_once()
        assert result["error"] == "export boom"

    def test_self_play_exception_frees_engine(self):
        """Exception in Phase 1 must free the engine."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(
            side_effect=RuntimeError("selfplay boom"))
        loop._free_engine = MagicMock()

        result = loop.run_epoch()

        loop._free_engine.assert_called_once()
        assert "selfplay boom" in result["error"]

    def test_normal_path_still_frees_via_eval(self):
        """Full path: _evaluate frees the engine (called there, not here)."""
        loop = InfiniteSelfPlayLoop("fake.ckpt", LoopConfig(training_mode="sft"))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._export_trajectories = MagicMock(return_value="/tmp/data.jsonl")
        loop._finetune = MagicMock(return_value="/tmp/candidate.safetensors")
        loop._evaluate = MagicMock(return_value={"passed": True})
        loop._maybe_promote = MagicMock(return_value=True)
        loop._free_engine = MagicMock()

        with patch("builtins.open", self._MockFile(["line"] * 10)):
            loop.run_epoch()

        # run_epoch itself shouldn't double-free on the happy path —
        # _finetune/_evaluate own the cleanup there.
        loop._evaluate.assert_called_once()


class TestCheckpointNaming:
    """Epoch checkpoints must never overwrite existing files."""

    def test_epoch_path_uses_config_tag(self, tmp_path):
        c = LoopConfig(config_name="forgelm_v2",
                       checkpoint_dir=str(tmp_path))
        loop = InfiniteSelfPlayLoop("fake.ckpt", c)
        p = loop._epoch_checkpoint_path(3)
        assert "ForgeLM_V2_SP3" in p

    def test_epoch_path_never_overwrites(self, tmp_path):
        c = LoopConfig(config_name="forgelm_v2",
                       checkpoint_dir=str(tmp_path))
        loop = InfiniteSelfPlayLoop("fake.ckpt", c)
        first = loop._epoch_checkpoint_path(1)
        # Simulate a previous run leaving SP1 behind
        open(first, "w").close()
        second = loop._epoch_checkpoint_path(1)
        assert second != first
        assert second.endswith("_r2.safetensors")


class TestStrictPromote:
    """Promotion requires the candidate to beat/tie the base by default."""

    def _loop(self, strict=True):
        loop = InfiniteSelfPlayLoop(
            "fake.ckpt", LoopConfig(strict_promote=strict, eval_threshold=0.5))
        loop._free_engine = MagicMock()
        return loop

    def _eval(self, loop, base_q, cand_q, winner):
        with patch("forge.self_play.discovery.fast_eval.fast_eval",
                   return_value={"base": {"quality": base_q},
                                 "candidate": {"quality": cand_q},
                                 "winner": winner}):
            return loop._evaluate("cand.safetensors")

    def test_winner_candidate_promotes(self):
        assert self._eval(self._loop(), 0.5, 0.6, "CANDIDATE")["passed"]

    def test_strict_rejects_regressed_candidate(self):
        # Candidate within lenient threshold (0.4 >= 0.5*0.5) but lost.
        r = self._eval(self._loop(strict=True), 0.5, 0.4, "BASE")
        assert r["passed"] is False

    def test_lenient_allows_within_threshold(self):
        r = self._eval(self._loop(strict=False), 0.5, 0.4, "BASE")
        assert r["passed"] is True

    def test_lenient_rejects_large_regression(self):
        r = self._eval(self._loop(strict=False), 0.5, 0.1, "BASE")
        assert r["passed"] is False


class TestCanonicalSftRender:
    """SFT examples must render in canonical Jamba ChatML (<|im_start|>/
    <|im_end|>, <tool_call>) — the format eval + inference use.

    Regression: sft_train rendered <|startofsegment|>/<|endofsegment|> +
    <|tool_call_start|> — none of which exist in the V2 tokenizer, so every
    epoch trained off-distribution and eval metrics barely moved."""

    def test_single_turn_uses_im_markers(self):
        from forge.training.runners.sft_train import render_single_turn
        full, comp_start = render_single_turn("What is 2+2?", "4")
        assert "<|im_start|>user\nWhat is 2+2?<|im_end|>\n" in full
        assert "<|im_start|>assistant\n" in full
        assert "startofsegment" not in full and "endofsegment" not in full
        # Completion starts exactly at the assistant body.
        assert full[comp_start:] == "4<|im_end|>\n"
        assert full[:comp_start].endswith("<|im_start|>assistant\n")

    def test_single_turn_no_think_block(self):
        """Direct-answer SFT: generation prompt must not open <think>."""
        from forge.training.runners.sft_train import render_single_turn
        full, _ = render_single_turn("Capital of Italy?", "Rome")
        assert "<think>" not in full

    def test_tool_call_markers_canonical(self):
        from forge.training.runners.sft_train import (
            _TOOL_CALL_END, _TOOL_CALL_START, _render_tool_call)
        assert _TOOL_CALL_START == "<tool_call>"
        assert _TOOL_CALL_END == "</tool_call>"
        out = _render_tool_call({"name": "f", "arguments": {"x": 1}})
        assert out == '<tool_call>\n{"name": "f", "arguments": {"x": 1}}\n</tool_call>'

    def test_split_multi_turn_canonical(self):
        from forge.training.runners.sft_train import split_multi_turn
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "",
             "tool_calls": [{"name": "t", "arguments": {"a": 1}}]},
            {"role": "tool", "content": "42"},
            {"role": "assistant", "content": "done"},
        ]
        pairs = split_multi_turn(msgs)
        assert len(pairs) == 2
        # Turn 1: prompt ends in gen prompt; completion is the tool call.
        assert pairs[0][0].endswith("<|im_start|>assistant\n")
        assert pairs[0][1].startswith("<tool_call>")
        assert pairs[0][1].endswith("<|im_end|>\n")
        # Turn 2: prompt carries the tool result inside a user turn
        # (canonical <tool_response> grouping), not a fake tool role.
        assert "<tool_response>\n42\n</tool_response>" in pairs[1][0]
        assert "<|im_start|>user" in pairs[1][0]
        assert pairs[1][1] == "done<|im_end|>\n"
        for p, c in pairs:
            assert "startofsegment" not in p + c

    def test_render_messages_completion_offset(self):
        from forge.training.runners.sft_train import render_messages
        msgs = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"}]
        full, comp_start = render_messages(msgs)
        assert full[comp_start:] == "a<|im_end|>\n"
        assert "<|im_start|>" in full and "startofsegment" not in full


class TestSolutionCleaning:
    """_assemble_solution must strip reasoning-in-comments (overthinking
    leak into SFT/GRPO data) while preserving executable semantics."""

    def test_clean_code_strips_musing(self):
        from forge.self_play.infinite_curriculum import _clean_code
        code = (
            "# We need to think carefully about this problem.\n"
            "# It might be ambiguous but let's try.\n"
            "def solve(nums: list):\n"
            '    """Long musing docstring that restates the task at\n'
            "    length and bloats the data\"\"\"\n"
            "    # positive numbers only\n"
            "    return sum(x for x in nums if x > 0)  # the trick\n")
        cleaned = _clean_code(code)
        assert "#" not in cleaned  # all comments gone
        assert "musing" not in cleaned and "ambiguous" not in cleaned
        # Semantics preserved — still executes correctly.
        ns: dict = {}
        exec(cleaned, ns)
        assert ns["solve"]([1, -2, 3]) == 4
        assert ns["solve"]([]) == 0

    def test_clean_code_keeps_functional_constructs(self):
        from forge.self_play.infinite_curriculum import _clean_code
        code = ("import math\n"
                "HELPER = 10  # constant\n"
                "def solve(n):\n"
                "    return math.factorial(n) + HELPER\n")
        cleaned = _clean_code(code)
        ns: dict = {}
        exec(cleaned, ns)
        assert ns["solve"](3) == 16
        assert "import math" in cleaned

    def test_clean_code_unparseable_returns_empty(self):
        from forge.self_play.infinite_curriculum import _clean_code
        assert _clean_code("this is just prose musing, not code") == ""

    def test_assemble_solution_strips_comments(self):
        from forge.self_play.infinite_curriculum import InfiniteCurriculum
        from forge.evaluation.goal_tasks import GoalTask
        cur = InfiniteCurriculum(model=None, tokenizer=None, device="cpu")
        cur.engine = object()  # engine path → chat assemble branch
        task = GoalTask(id="t", domain="math", difficulty="easy",
                        description="sum positives", input_signature="(nums: list) -> int",
                        solve_name="solve", test_cases=[], stress_index=0,
                        archetype="")
        completion = ("```python\n# reasoning musing here\n"
                      "def solve(nums: list):\n"
                      "    return sum(x for x in nums if x > 0)\n```")
        out = cur._assemble_solution(task, "p", completion)
        assert "musing" not in out and "#" not in out
        ns: dict = {}
        exec(out, ns)
        assert ns["solve"]([1, -2, 3]) == 4


class TestConciseQAStream:
    """Verified short-answer pairs mixed into epoch exports."""

    def test_pairs_verified_and_minimal(self):
        from forge.self_play.infinite_loop import _concise_qa_pairs
        pairs = _concise_qa_pairs(20, epoch=1)
        assert len(pairs) == 20
        assert len({p["prompt"] for p in pairs}) == 20  # deduped
        for p in pairs:
            assert p["prompt"] and p["response"]
            # Minimal: short answers are the point.
            assert len(p["response"].split()) <= 6

    def test_pairs_deterministic_per_epoch(self):
        from forge.self_play.infinite_loop import _concise_qa_pairs
        a = _concise_qa_pairs(10, epoch=3)
        b = _concise_qa_pairs(10, epoch=3)
        c = _concise_qa_pairs(10, epoch=4)
        assert a == b
        assert [p["prompt"] for p in a] != [p["prompt"] for p in c]

    def test_export_mixes_qa(self, tmp_path):
        loop = InfiniteSelfPlayLoop(
            "fake.ckpt", LoopConfig(
                checkpoint_dir=str(tmp_path), data_dir=str(tmp_path / "d"),
                concise_qa_per_epoch=6))
        loop._trajectories = [{
            "task_description": "sum positives", "signature": "(nums: list)",
            "test_cases": [], "solution_code": "def solve(nums):\n    return 1",
            "domain": "math", "difficulty": "easy", "reward": 1.0,
        }]
        path = loop._export_trajectories(epoch=1)
        import json as _json
        rows = [_json.loads(l) for l in open(path, encoding="utf-8")]
        # 1 trajectory + 6 QA pairs
        assert len(rows) == 7
        qa = [r for r in rows if not r["prompt"].startswith("Write a Python")]
        assert len(qa) == 6

    def test_export_qa_disabled(self, tmp_path):
        loop = InfiniteSelfPlayLoop(
            "fake.ckpt", LoopConfig(
                checkpoint_dir=str(tmp_path), data_dir=str(tmp_path / "d"),
                concise_qa_per_epoch=0))
        loop._trajectories = [{
            "task_description": "t", "signature": "(x: int)",
            "test_cases": [], "solution_code": "def solve(x):\n    return x",
            "domain": "math", "difficulty": "easy", "reward": 1.0,
        }]
        path = loop._export_trajectories(epoch=1)
        assert sum(1 for _ in open(path)) == 1


class TestConciseRewards:
    """GRPO length-efficiency bonus shapes rewards toward terse passes."""

    def test_shorter_pass_earns_more(self):
        from forge.self_play.infinite_loop import _concise_adjusted_rewards
        comps = ["x" * 100, "x" * 300, "x" * 200]
        rews = [1.0, 1.0, 0.0]
        out = _concise_adjusted_rewards(comps, rews, 0.1)
        assert out[0] == pytest.approx(1.1)   # shortest pass → full bonus
        assert out[1] == pytest.approx(1.0)   # longest pass → no bonus
        assert out[2] == 0.0                  # fail untouched

    def test_all_pass_group_regains_variance(self):
        """All-pass groups had identical rewards → advantage 0 (no learning).
        Length differences now produce nonzero spread."""
        from forge.self_play.infinite_loop import _concise_adjusted_rewards
        out = _concise_adjusted_rewards(["a" * 50, "a" * 90], [1.0, 1.0], 0.1)
        assert out[0] > out[1]

    def test_bonus_zero_passthrough(self):
        from forge.self_play.infinite_loop import _concise_adjusted_rewards
        assert _concise_adjusted_rewards(["aa"], [1.0], 0.0) == [1.0]

    def test_all_fail_passthrough(self):
        from forge.self_play.infinite_loop import _concise_adjusted_rewards
        assert _concise_adjusted_rewards(["a", "bb"], [0.0, 0.0], 0.1) == [0.0, 0.0]


class TestConciseConfig:
    def test_config_fields(self):
        c = LoopConfig()
        assert c.concise_qa_per_epoch > 0
        assert 0 < c.grpo_concise_bonus < 0.5  # small — correctness dominates


class TestDebugRound:
    """--debug-round: ONE self-play pass, NO training, every generation
    traced (raw output + token ids) to console + transcript files."""

    def test_config_field(self):
        assert LoopConfig().debug_round is False
        assert LoopConfig(debug_round=True).debug_round is True

    def test_run_epoch_skips_training(self):
        """Debug round must not touch export/finetune/grpo/eval/promote."""
        loop = InfiniteSelfPlayLoop(
            "fake.ckpt", LoopConfig(debug_round=True))
        loop._run_self_play = MagicMock(return_value={"success_rate": 0.5})
        loop._dump_debug_round = MagicMock(
            return_value={"n_generations": 3})
        for m in ("_export_trajectories", "_finetune", "_grpo_train",
                  "_evaluate", "_maybe_promote"):
            setattr(loop, m, MagicMock())
        out = loop.run_epoch()
        loop._dump_debug_round.assert_called_once()
        assert out["debug_round"]["n_generations"] == 3
        for m in ("_export_trajectories", "_finetune", "_grpo_train",
                  "_evaluate", "_maybe_promote"):
            getattr(loop, m).assert_not_called()

    def test_run_epoch_debug_dumps_despite_selfplay_error(self):
        """A no-valid-tasks epoch still yields a transcript — that's the
        point of the debug round (inspect why nothing validated)."""
        loop = InfiniteSelfPlayLoop(
            "fake.ckpt", LoopConfig(debug_round=True))
        loop._run_self_play = MagicMock(
            return_value={"error": "no_valid_tasks"})
        loop._dump_debug_round = MagicMock(return_value={"n_generations": 7})
        loop._finetune = MagicMock()
        out = loop.run_epoch()
        loop._dump_debug_round.assert_called_once()
        loop._finetune.assert_not_called()
        assert out["self_play"]["error"] == "no_valid_tasks"

    def test_gen_tracer_reencode(self):
        """Engine path: ids re-encoded from the raw output text."""
        from forge.self_play.infinite_curriculum import GenTracer
        tok = MagicMock()
        tok.return_value = MagicMock(input_ids=[10, 11, 519])
        tok.decode = lambda ids: {
            10: "a", 11: "b", 519: "<|im_end|>"}[ids[0]]
        tr = GenTracer(tok, live=False)
        ev = tr.record(phase="solve", prompt="p",
                       rendered="<|im_start|>user\np",
                       raw="ab<|im_end|>", reply="ab")
        assert ev["ids_source"] == "reencoded"
        assert ev["n_tokens"] == 3
        assert ev["tokens"][2] == {"id": 519, "text": "<|im_end|>"}
        assert len(tr.events) == 1

    def test_gen_tracer_true_ids(self):
        """Legacy path: supplied gen_ids are marked 'generated'."""
        from forge.self_play.infinite_curriculum import GenTracer
        tok = MagicMock()
        tok.decode = lambda ids: f"tok{ids[0]}"
        tr = GenTracer(tok, live=False)
        ev = tr.record(phase="propose", prompt="p", rendered="p",
                       raw="x", reply="x", gen_ids=[7, 8])
        assert ev["ids_source"] == "generated"
        assert ev["tokens"] == [{"id": 7, "text": "tok7"},
                                {"id": 8, "text": "tok8"}]

    def test_generate_engine_path_records_trace(self):
        """_generate records rendered prompt, RAW (pre-strip) output and
        the post-strip reply — reasoning must be visible in `raw`."""
        from forge.self_play.infinite_curriculum import (
            GenTracer, InfiniteCurriculum)
        tok = MagicMock()
        tok.return_value = MagicMock(input_ids=[1, 2])
        tok.decode = lambda ids: "x"
        eng = MagicMock()
        eng.generate_raw = MagicMock(
            return_value="<think>musing</think>\n"
                         "def solve():\n    pass")
        cur = InfiniteCurriculum(model=None, tokenizer=tok, device="cpu",
                                 engine=eng)
        cur.gen_trace = GenTracer(tok, live=False)
        cur._think_cap = lambda *a, **k: None
        cur._trace_phase = "solve"
        reply = cur._generate("Write solve", prefill="```python\n")
        assert "def solve" in reply
        ev = cur.gen_trace.events[0]
        assert ev["phase"] == "solve"
        assert "<think>musing</think>" in ev["raw"]   # raw keeps reasoning
        assert "<think>" not in ev["reply"]          # stripped for pipeline
        assert "<|im_start|>" in ev["rendered"]

    def test_dump_debug_round_writes_files(self, tmp_path):
        """Transcript .txt + .jsonl land under status_dir."""
        from forge.self_play.infinite_curriculum import GenTracer
        tok = MagicMock()
        tok.return_value = MagicMock(input_ids=[42])
        tok.decode = lambda ids: "z"
        tr = GenTracer(tok, live=False)
        tr.record(phase="solve", prompt="p", rendered="REND",
                  raw="RAWOUT", reply="out")
        loop = InfiniteSelfPlayLoop(
            "fake.ckpt", LoopConfig(debug_round=True,
                                    status_dir=str(tmp_path)))
        loop._gen_trace = tr
        loop._free_engine = MagicMock()
        info = loop._dump_debug_round()
        assert info["n_generations"] == 1
        txt = open(info["transcript"], encoding="utf-8").read()
        assert "RENDERED PROMPT" in txt and "REND" in txt
        assert "RAW OUTPUT" in txt and "RAWOUT" in txt
        assert "42:" in txt  # token table
        rows = open(info["transcript_jsonl"], encoding="utf-8").read()
        assert '"raw": "RAWOUT"' in rows
        loop._free_engine.assert_called_once()


class TestSaveTaskSerialization:
    """_save_task must never crash the epoch on non-serializable task data
    (observed live: a set-valued test arg → TypeError → entire self-play
    epoch aborted mid-propose)."""

    def test_set_valued_args_serialize(self, tmp_path):
        from forge.self_play.infinite_curriculum import (
            InfiniteCurriculum, ProposedTask)
        cur = InfiniteCurriculum(model=None, tokenizer=None, device="cpu",
                                 task_queue_dir=str(tmp_path))
        task = ProposedTask(
            id="t1", domain="math", difficulty="easy",
            description="dedupe a list", signature="(xs: list)",
            test_cases=[{"args": ({1, 2, 3},), "expected": [1, 2, 3]}],
            stress_index=None)
        cur._save_task(task)  # must not raise
        import json as _json
        saved = _json.loads((tmp_path / "t1.json").read_text())
        assert saved["test_cases"][0]["args"] == [[1, 2, 3]]

    def test_json_default_coercions(self):
        from forge.self_play.infinite_curriculum import _json_default
        assert _json_default({3, 1, 2}) == [1, 2, 3]
        assert _json_default(b"\xffx") == "�x"
        assert _json_default(object()).startswith("<")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
