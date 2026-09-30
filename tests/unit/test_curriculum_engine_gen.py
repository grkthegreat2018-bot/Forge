"""Tests for engine-routed generation in InfiniteCurriculum.

Covers:
  - BatchedDecoding honors a caller-supplied Jamba EOS set ({2, 519}) —
    regression test for the stale {7, 151643, 151645} LFM/Qwen default
    that never fired on ForgeLM V2 (batched runs went to max_tokens).
  - Curriculum._generate / _generate_batch route through ForgeEngine with
    canonical Jamba ChatML rendering (GUI direct-mode suffix included).
  - solve_tasks_batch issues ONE batched generation for all first attempts
    and assembles chat-mode replies into runnable solutions.
  - Prompt selection: chat instruction prompts with engine, raw completion
    prompts without.

All CPU-only — engine is a MagicMock, the model is a tiny stub.
"""
import torch
from unittest.mock import MagicMock

from forge.engine.batched_decoding import BatchedDecoding
from forge.self_play.infinite_curriculum import (
    CHAT_PROPOSE_PROMPT,
    InfiniteCurriculum,
    ProposedTask,
    _strip_reply,
)

# Think-mode render tail: generation prompt opens the assistant <think>
# block — the trained format (chat_template thinking=True).
_THINK_TAIL = "<|im_start|>assistant\n<think>\n"


# ── BatchedDecoding EOS ─────────────────────────────────────────────

class _EOSModel:
    """Fake model whose logits always argmax to <|im_end|> (519)."""

    def __call__(self, ids, attention_mask=None, past_key_values=None,
                 use_cache=False, **_kw):
        B, L = ids.shape
        logits = torch.zeros(B, L, 600)
        logits[..., 519] = 10.0
        return (logits, None, object())  # (logits, loss, past) tuple shape


class TestBatchedDecodingEOS:
    def test_stops_on_jamba_eos(self):
        """Engine-supplied {2, 519} EOS set must terminate generation."""
        bd = BatchedDecoding(eos_token_ids={2, 519})
        prompts = [torch.tensor([[1, 100, 200]]), torch.tensor([[1, 300]])]
        out = bd.generate_batch(
            _EOSModel(), prompts,
            max_tokens_list=[50, 50],
            temperatures=[0.0, 0.0],
            top_ps=[1.0, 1.0])
        # prompt + exactly one EOS token each — early stop fired
        assert out[0].shape[1] == 4
        assert out[1].shape[1] == 3
        assert out[0][0, -1].item() == 519
        assert out[1][0, -1].item() == 519

    def test_engine_eos_set_overrides_legacy_default(self):
        bd_default = BatchedDecoding()
        assert bd_default.eos_set == {7, 151643, 151645}
        bd = BatchedDecoding(eos_token_ids={2, 519})
        assert 519 in bd.eos_set
        assert 2 in bd.eos_set


# ── Curriculum engine routing ───────────────────────────────────────

def _make_curriculum(tmp_path, engine=None):
    import types
    tok = types.SimpleNamespace(
        encode=lambda s, add_special_tokens=False: [10, 542, 11, 12])
    return InfiniteCurriculum(
        model=None, tokenizer=tok, device="cpu",
        max_gen_tokens=32, task_queue_dir=str(tmp_path), engine=engine)


class TestEngineGeneration:
    def test_generate_routes_through_engine_chatml(self, tmp_path):
        eng = MagicMock()
        eng.generate_raw.return_value = "reply text"
        cur = _make_curriculum(tmp_path, engine=eng)

        out = cur._generate("do a thing")

        assert out == "reply text"
        eng.generate_raw.assert_called_once()
        prompt = eng.generate_raw.call_args[0][0]
        # Canonical Jamba ChatML: user turn + assistant generation prompt
        # + GUI direct-mode closed-think Answer anchor.
        assert "<|im_start|>user\n" in prompt
        assert "do a thing" in prompt
        assert prompt.endswith(_THINK_TAIL)
        kw = eng.generate_raw.call_args.kwargs
        assert kw["eos_token_ids"] == [2, 519]
        assert kw["logits_processor"] is not None
        # </think> must survive decode for the reasoning split
        assert kw["skip_special_tokens"] is False
        # Think-cap: forces </think> close after budget think tokens
        proc = kw["logits_processor"]
        logits = torch.zeros(1, 600)
        forced = proc(logits, list(range(192)))  # at solve budget
        assert forced[0, 10].item() != float("-inf")  # forced suffix token
        assert forced.argmax(-1).item() == 10

    def test_generate_batch_single_engine_call(self, tmp_path):
        eng = MagicMock()
        eng.generate_batch.return_value = ["a", "b", "c"]
        cur = _make_curriculum(tmp_path, engine=eng)

        out = cur._generate_batch(["p1", "p2", "p3"])

        assert out == ["a", "b", "c"]
        eng.generate_batch.assert_called_once()
        prompts = eng.generate_batch.call_args[0][0]
        assert len(prompts) == 3
        for p, raw in zip(prompts, ["p1", "p2", "p3"]):
            assert "<|im_start|>user\n" in p
            assert raw in p
            assert p.endswith(_THINK_TAIL)

    def test_propose_prompt_selects_chat_format(self, tmp_path):
        eng = MagicMock()
        cur = _make_curriculum(tmp_path, engine=eng)
        p = cur._propose_prompt("math", "easy", "(n: int) -> int", "double it")
        assert "DIFFERENT easy task in the math domain" in p
        assert "double it" in p
        # Raw fallback stays the legacy completion prompt
        cur_raw = _make_curriculum(tmp_path, engine=None)
        p_raw = cur_raw._propose_prompt("math", "easy", "(n: int) -> int",
                                        "double it")
        assert "def solve" in p_raw
        assert p_raw != p

    def test_think_cap_masks_structural_ids(self, tmp_path):
        """The think-cap processor bans the same structural ids the GUI
        bans in self-play: im_start, tool_call pair, tool_response pair,
        and <think> re-open (518, 531, 532, 539, 540, 541)."""
        cur = _make_curriculum(tmp_path, engine=MagicMock())
        proc = cur._think_cap(192)
        logits = torch.zeros(1, 600)
        out = proc(logits.clone(), [])
        for tid in (518, 531, 532, 539, 540, 541):
            assert out[0, tid].item() == float("-inf"), tid
        assert out[0, 100].item() == 0.0  # untouched elsewhere
        assert out[0, 519].item() == 0.0  # eos NOT banned


class TestSolveTasksBatch:
    def _tasks(self):
        return [
            ProposedTask(id="t1", domain="math", difficulty="easy",
                         description="add one",
                         signature="(n: int) -> int",
                         test_cases=[{"args": (1,), "expected": 2},
                                     {"args": (5,), "expected": 6}]),
            ProposedTask(id="t2", domain="math", difficulty="easy",
                         description="double it",
                         signature="(n: int) -> int",
                         test_cases=[{"args": (3,), "expected": 6}]),
        ]

    def test_one_batch_call_and_correct_solutions(self, tmp_path):
        eng = MagicMock()
        eng.generate_batch.return_value = [
            "```python\ndef solve(n):\n    return n + 1\n```",
            "```python\ndef solve(n):\n    return n * 2\n```",
        ]
        cur = _make_curriculum(tmp_path, engine=eng)

        results = cur.solve_tasks_batch(self._tasks())

        assert len(results) == 2
        # All first attempts went through ONE batched engine call
        eng.generate_batch.assert_called_once()
        assert eng.generate_raw.call_count == 0  # no retries needed
        for task, result, elapsed_ms in results:
            assert result["final_success"] is True
            assert result["rounds_used"] == 1
            assert elapsed_ms > 0

    def test_solve_prompt_is_instruction_in_engine_mode(self, tmp_path):
        eng = MagicMock()
        eng.generate_batch.return_value = [
            "```python\ndef solve(n):\n    return n + 1\n```"]
        eng.generate_raw.return_value = (
            "```python\ndef solve(n):\n    return n + 1\n```")
        cur = _make_curriculum(tmp_path, engine=eng)
        cur.solve_tasks_batch(self._tasks()[:1])
        prompt = eng.generate_batch.call_args[0][0][0]
        # Instruction-style prompt rendered as ChatML (not raw stub) —
        # direct turn prefilled with the code-fence opener (solve prefill).
        assert "add one" in prompt
        assert "def solve(n: int)" in prompt
        assert prompt.endswith("<|im_start|>assistant\n```python\n")

    def test_empty_task_list(self, tmp_path):
        cur = _make_curriculum(tmp_path, engine=MagicMock())
        assert cur.solve_tasks_batch([]) == []


class TestStripReply:
    def test_drops_think_block(self):
        reply = "musing about the problem\n</think>\n```python\npass\n```"
        assert _strip_reply(reply) == "```python\npass\n```"

    def test_strips_answer_anchor(self):
        assert _strip_reply("Answer:\ncode") == "code"
        assert _strip_reply("  Final Answer: code") == "code"

    def test_plain_reply_untouched(self):
        assert _strip_reply("just text") == "just text"

    def test_strips_turn_markers(self):
        reply = "</think>\n```python\npass\n```<|im_end|>"
        assert _strip_reply(reply) == "```python\npass\n```"


class TestExecutorDerivedOutputs:
    """AZR semantics — the model proposes (program, inputs); the executor
    derives o = solve(*args). Declared expectations are never trusted."""

    SPEC = """```python
# Task: count even numbers
def solve(n: int) -> int:
    count = 0
    for x in n:
        if x % 2 == 0:
            count += 1
    return count

# solve([])
# solve([2, 4])
# solve([1, 3])
# solve([2, 4, 6, 8])
```"""

    def test_parse_collects_pending_inputs(self, tmp_path):
        cur = _make_curriculum(tmp_path)
        task = cur._parse_proposal(self.SPEC, "math", "easy", "induction")
        assert task is not None
        assert len(task.pending_inputs) == 4
        assert task.test_cases == []

    def test_validate_derives_expected_outputs(self, tmp_path):
        cur = _make_curriculum(tmp_path)
        task = cur._parse_proposal(self.SPEC, "math", "easy", "induction")
        assert cur._validate_task(task) is True
        assert len(task.test_cases) == 4
        assert task.pending_inputs == []
        assert {"args": ([2, 4],), "expected": 2} in task.test_cases
        assert {"args": ([],), "expected": 0} in task.test_cases
        assert task.proposer_confidence == 1.0

    def test_wrong_declared_outputs_overwritten_by_executor(self, tmp_path):
        """A proposal that declares WRONG expectations is still valid —
        the executor's outputs replace them (was: rejected as broken)."""
        cur = _make_curriculum(tmp_path)
        raw = self.SPEC.replace(
            "# solve([])", "# solve([]) == 999")
        task = cur._parse_proposal(raw, "math", "easy", "induction")
        assert task is not None
        # declared pair parsed, remaining 3 lines pending
        assert len(task.pending_inputs) == 3
        assert cur._validate_task(task) is True
        # executor truth wins: solve([]) == 0, not 999
        assert {"args": ([],), "expected": 0} in task.test_cases


class TestCloneGuards:
    """Anti-overfit dedup: an epoch of near-identical tasks is a narrow,
    degenerate GRPO signal. Descriptions get a Jaccard semantic check;
    validated impls get a probe-input functional fingerprint."""

    def test_semantic_clone_rejected(self, tmp_path):
        cur = _make_curriculum(tmp_path)
        cur._mark_description_seen("Return the number of even digits in n.")
        # Different wording, same content words → clone
        assert cur._is_seen_description(
            "Return the sum of even digits in n") is True
        # Genuinely different task → not a clone
        assert cur._is_seen_description(
            "Check if a binary tree is balanced") is False

    def test_functional_fingerprint_clone(self, tmp_path):
        cur = _make_curriculum(tmp_path)
        task = ProposedTask(
            id="t", domain="math", difficulty="easy", description="d",
            signature="(n: int) -> int",
            test_cases=[{"args": (1,), "expected": 0},
                        {"args": (2,), "expected": 1}])
        impl_a = ("def solve(n):\n    if n == 0:\n        return 1\n"
                  "    c = 0\n    n = abs(n)\n    while n:\n"
                  "        d = n % 10\n        if d % 2 == 0:\n"
                  "            c += 1\n        n //= 10\n    return c")
        impl_b = "def solve(n):\n    return sum(1 for ch in str(abs(n)) if int(ch) % 2 == 0)"
        fa = cur._functional_fingerprint(impl_a, task)
        fb = cur._functional_fingerprint(impl_b, task)
        assert fa is not None and fa == fb  # same function → same fp

    def test_distinct_functions_differ(self, tmp_path):
        cur = _make_curriculum(tmp_path)
        task = ProposedTask(
            id="t", domain="math", difficulty="easy", description="d",
            signature="(n: int) -> int",
            test_cases=[{"args": (1,), "expected": 0},
                        {"args": (2,), "expected": 1}])
        fp1 = cur._functional_fingerprint(
            "def solve(n):\n    return n * 2", task)
        fp2 = cur._functional_fingerprint(
            "def solve(n):\n    return n + 2", task)
        assert fp1 != fp2

    def test_think_cap_safe_on_inference_tensors(self, tmp_path):
        """Think-cap processor must clone before masking — decode runs
        under torch.inference_mode where in-place updates raise."""
        cur = _make_curriculum(tmp_path, engine=MagicMock())
        proc = cur._think_cap(192)
        with torch.inference_mode():
            logits = torch.zeros(1, 600)
            out = proc(logits, [])
        assert out[0, 518].item() == float("-inf")  # im_start banned
        assert out[0, 519].item() == 0.0            # eos NOT banned
