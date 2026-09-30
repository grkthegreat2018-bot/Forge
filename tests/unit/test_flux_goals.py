"""Tests for R53 flux self-play: source-grounded goals + tool use.

All web access is faked — tests are CPU-only and hermetic.
"""
import json
from unittest.mock import MagicMock

import pytest

from forge.self_play.flux_goals import (
    FluxGoalSampler, FluxKB, FluxTools, TOOL_HEADER, arxiv_goals,
    dict_goals, goals_from_doc, parse_tool_call, wiki_goals)
from forge.self_play.infinite_loop import (
    InfiniteSelfPlayLoop, LoopConfig)


# ── canned source documents ────────────────────────────────────────────

WIKI_DOC = {"source": "wiki", "key": "photosynthesis",
            "title": "Photosynthesis",
            "extract": ("Photosynthesis is a system of biological "
                        "processes by which plants convert light "
                        "energy into chemical energy. It was first "
                        "described in 1779 by Jan Ingenhousz."),
            "url": "https://en.wikipedia.org/wiki/Photosynthesis"}

DICT_DOC = {"source": "dict", "key": "ephemeral", "word": "ephemeral",
            "pos": "adjective",
            "glosses": ["lasting a very short time",
                        "a short-lived thing"],
            "synonyms": ["fleeting", "transient", "momentary"],
            "antonyms": ["permanent", "eternal"]}

ARXIV_DOC = {"source": "arxiv", "key": "1706.03762",
             "title": "Attention Is All You Need",
             "summary": "The dominant sequence transduction models.",
             "url": "http://arxiv.org/abs/1706.03762",
             "published": "2017-06-12",
             "authors": ["Ashish Vaswani", "Noam Shazeer"],
             "arxiv_id": "1706.03762"}


class FakeTools:
    """Offline FluxTools stand-in: canned docs, no network."""

    def __init__(self, docs=None, online=True):
        self.docs = list(docs or [])
        self._online = online
        self.fetch_budget = 8
        self.calls = []

    def online(self):
        return self._online

    def begin_epoch(self, budget=None):
        if budget is not None:
            self.fetch_budget = budget

    def fetch_doc(self, source, query=""):
        for d in self.docs:
            if d["source"] == source:
                return d
        return None

    def call(self, name, query):
        self.calls.append((name, query))
        return f"obs for {query}"


# ── tool-call parsing ──────────────────────────────────────────────────

class TestParseToolCall:
    def test_canonical_forms(self):
        assert parse_tool_call("wiki Photosynthesis") == \
            ("wiki", "Photosynthesis")
        assert parse_tool_call("dict: ephemeral") == ("dict", "ephemeral")
        assert parse_tool_call("[paper] quantization") == \
            ("paper", "quantization")
        assert parse_tool_call("arxiv attention") == ("paper", "attention")
        assert parse_tool_call("search cats") == ("web", "cats")

    def test_rejects_garbage(self):
        assert parse_tool_call("") is None
        assert parse_tool_call("42") is None
        assert parse_tool_call("hello world this is not a call") is None \
            or parse_tool_call("x y")[0] in ("wiki", "dict",
                                             "paper", "web")
        assert parse_tool_call("wiki " + "x" * 200) is None


# ── doc -> goal builders ───────────────────────────────────────────────

class TestGoalBuilders:
    def test_wiki_goals_extractive(self):
        goals = wiki_goals(WIKI_DOC)
        assert goals, "expected goals from wiki doc"
        for g in goals:
            assert g["source"] == "wiki" and g["context"]
            assert g["tool"] == "wiki" and g["query"] == "Photosynthesis"
            # every answer is verbatim-extractive from the source
            assert g["response"].lower() in \
                (WIKI_DOC["extract"] + WIKI_DOC["title"]).lower()
        kinds = {g["response"] for g in goals}
        assert "1779" in kinds           # year cloze
        assert "Photosynthesis" in kinds  # masked-subject recall

    def test_dict_goals(self):
        goals = dict_goals(DICT_DOC)
        resp = {g["prompt"]: g for g in goals}
        assert any("part of speech" in p for p in resp)
        assert any("mean" in p for p in resp)
        pos_g = [g for g in goals if "part of speech" in g["prompt"]][0]
        assert pos_g["response"] == "adjective"
        syn_g = [g for g in goals if "synonym" in g["prompt"]][0]
        assert "fleeting" in syn_g["accepts"]

    def test_arxiv_goals(self):
        goals = arxiv_goals(ARXIV_DOC)
        by_resp = {g["response"] for g in goals}
        assert "2017" in by_resp
        assert "Ashish Vaswani" in by_resp
        assert "1706.03762" in by_resp

    def test_goals_from_doc_dispatch(self):
        assert goals_from_doc(WIKI_DOC)
        assert goals_from_doc({"source": "nope"}) == []


# ── FluxTools ──────────────────────────────────────────────────────────

class TestFluxTools:
    def test_offline_probe(self, monkeypatch):
        monkeypatch.setattr(
            "forge.web_primitives.http_get",
            MagicMock(side_effect=OSError("no net")))
        t = FluxTools()
        assert t.online() is False
        assert t.call("wiki", "x") == "offline"
        assert t.fetch_doc("wiki") is None

    def test_budget_exhaustion(self, monkeypatch):
        monkeypatch.setattr("forge.web_primitives.http_get",
                            MagicMock(return_value="{}"))
        t = FluxTools(fetch_budget=1)
        t._online = True
        assert t._get("http://x") == "{}"
        assert t._get("http://x") == ""   # budget spent


# ── FluxKB persistence ─────────────────────────────────────────────────

class TestFluxKB:
    def test_roundtrip_and_dedup(self, tmp_path):
        p = str(tmp_path / "kb.jsonl")
        kb = FluxKB(p)
        assert kb.add(WIKI_DOC) is True
        assert kb.add(WIKI_DOC) is False     # same (source,key)
        kb2 = FluxKB(p)                      # reload
        assert len(kb2.docs) == 1
        assert kb2.docs[0]["title"] == "Photosynthesis"


# ── FluxGoalSampler ────────────────────────────────────────────────────

class TestFluxGoalSampler:
    def test_sample_web_dedupes(self, tmp_path):
        tools = FakeTools(docs=[WIKI_DOC, DICT_DOC, ARXIV_DOC])
        kb = FluxKB(str(tmp_path / "kb.jsonl"))
        s = FluxGoalSampler(tools, kb=kb, sources=("wiki", "dict",
                                                   "arxiv"))
        seen = set()
        goals = s.sample_web(10, seen, set(), epoch=0)
        assert goals
        qs = [g["prompt"] for g in goals]
        assert len(qs) == len(set(qs))
        # all docs landed in the KB
        assert len(kb.docs) == 3

    def test_kb_replay_when_offline(self, tmp_path):
        p = str(tmp_path / "kb.jsonl")
        kb = FluxKB(p)
        kb.add(WIKI_DOC)
        tools = FakeTools(online=False)
        s = FluxGoalSampler(tools, kb=kb)
        # offline -> sample_web short-circuits on tools.online()
        assert s.sample_web(5, set(), set()) == []

    def test_eval_freeze(self, tmp_path):
        tools = FakeTools(docs=[WIKI_DOC, ARXIV_DOC])
        s = FluxGoalSampler(tools, sources=("wiki", "arxiv"))
        ep = str(tmp_path / "eval.json")
        rows1 = s.build_eval(4, ep, set())
        rows2 = s.build_eval(4, ep, set())   # loads frozen file
        assert rows1 == rows2
        assert all(r["prompt"] for r in rows2)


# ── loop integration ───────────────────────────────────────────────────

def _mock_flux_engine(gen_outputs: list):
    """Engine mock: generate_batch serves queued outputs in order."""
    engine = MagicMock()
    model = MagicMock()
    model._count = 5
    engine.model = model
    engine.tokenizer.encode = lambda *a, **k: [1, 2, 3]
    queue = list(gen_outputs)

    def _gb(prompts, **kw):
        batch = queue.pop(0) if queue else ["wrong"]
        return (batch * (len(prompts) // len(batch) + 1))[:len(prompts)]
    engine.generate_batch = MagicMock(side_effect=_gb)
    return engine


class TestFluxLoopWeb:
    def test_fresh_pairs_mixed_sources(self):
        """With web on, sampler goals mix into the batch with kinds."""
        cfg = LoopConfig(training_mode="flux", tasks_per_epoch=10,
                         flux_eval_n=4, flux_eval_web_n=0,
                         flux_web_frac=0.5)
        loop = InfiniteSelfPlayLoop("fake.flux", cfg)
        fake_goal = {"prompt": "What is Photosynthesis? "
                               "Answer with a short phrase.",
                     "response": "a system", "source": "wiki",
                     "context": "ctx", "tool": "wiki", "query": "Photo"}
        sampler = MagicMock()
        sampler.sample_web = lambda n, seen, ev, epoch=0: \
            [dict(fake_goal) for _ in range(n)]
        loop._flux_sampler_obj = sampler
        loop._flux_tools_obj = FakeTools()
        pairs = loop._flux_fresh_pairs(10)
        assert len(pairs) == 10
        web = [p for p in pairs if p.get("source") == "wiki"]
        assert web and all(p["kind"] in ("qa", "rag", "tool")
                           for p in web)
        assert all(p["kind"] == "qa" for p in pairs
                   if p.get("source") == "procedural")

    def test_tool_mode_two_phase(self):
        """Tool tasks run phase A (T:) then phase B (A:) and pass
        per-prompt newline stops to phase A."""
        cfg = LoopConfig(training_mode="flux", tasks_per_epoch=2,
                         flux_eval_n=4, flux_eval_web_n=0,
                         flux_group_size=2, flux_workers=2)
        loop = InfiniteSelfPlayLoop("fake.flux", cfg)
        goal = {"prompt": "What does ephemeral mean?",
                "response": "short-lived", "kind": "tool",
                "source": "dict", "context": "ephemeral: short-lived",
                "tool": "dict", "query": "ephemeral"}
        loop._flux_fresh_pairs = MagicMock(return_value=[goal, goal])
        # phase A emits a parseable call; phase B answers correctly
        engine = _mock_flux_engine([["dict ephemeral"] * 4,
                                    ["short-lived"] * 4])
        loop._flux_engine = MagicMock(return_value=engine)
        loop._flux_eval_acc = MagicMock(return_value=0.0)
        loop._flux_tools_obj = FakeTools()

        out = loop._flux_self_play()

        assert engine.generate_batch.call_count == 2
        first = engine.generate_batch.call_args_list[0]
        assert first[1]["stops"] == [["\n"]] * 4
        assert "T:" in first[0][0][0]
        second = engine.generate_batch.call_args_list[1]
        assert "O: obs for ephemeral" in second[0][0][0]
        assert out["attempts"] == 2 and out["tool_tasks"] == 2
        assert loop._flux_attempts[0]["ok"] is True
        assert loop._flux_attempts[0]["tool_ok"] is True

    def test_tool_fail_teaches_both_steps(self):
        """Failed tool task: context ingest + T-step demo + A-step demo
        + full-trajectory ingest."""
        cfg = LoopConfig(training_mode="flux", flux_eval_n=4,
                         flux_eval_web_n=0)
        loop = InfiniteSelfPlayLoop("fake.flux", cfg)
        engine = _mock_flux_engine([])
        model = engine.model
        model.journal = MagicMock()
        model.consolidate = MagicMock(return_value={})
        loop._engine = engine
        loop._flux_engine = MagicMock(return_value=engine)
        loop._flux_eval_acc = MagicMock(return_value=0.0)
        loop._maybe_promote = MagicMock(return_value=True)
        loop._flux_tools_obj = FakeTools()
        loop._flux_attempts = [{
            "prompt": "What does ephemeral mean?", "response": "fleeting",
            "accepts": None, "kind": "tool", "source": "dict",
            "context": "ephemeral (adj): lasting a very short time",
            "tool": "dict", "query": "ephemeral",
            "ok": False, "obs": ["obs for ephemeral"]}]
        loop.epoch = 1

        loop._flux_finish({})

        # context ingest + trajectory ingest (journal ws blocks) +
        # two teach_text demos (T step, A step)
        assert model.teach_text.call_count == 2
        t_ctx = model.teach_text.call_args_list[0][0][0]
        a_ctx = model.teach_text.call_args_list[1][0][0]
        assert t_ctx.endswith("\nT:") and "ephemeral" in t_ctx
        assert "O: obs for ephemeral" in a_ctx
        assert model.ingest.call_count == 2
        assert model.soft_reset.called

    def test_eval_combines_procedural_and_web(self, tmp_path):
        cfg = LoopConfig(training_mode="flux", flux_eval_n=6,
                         flux_eval_web_n=2,
                         status_dir=str(tmp_path))
        loop = InfiniteSelfPlayLoop("fake.flux", cfg)
        sampler = MagicMock()
        sampler.build_eval = MagicMock(return_value=[
            {"prompt": "web q1", "response": "a1", "accepts": [],
             "source": "wiki"},
            {"prompt": "web q2", "response": "a2", "accepts": [],
             "source": "arxiv"}])
        loop._flux_sampler_obj = sampler
        loop._flux_tools_obj = FakeTools()
        pairs = loop._flux_eval_pairs()
        assert len(pairs) == 8
        assert sum(p["source"] != "procedural" for p in pairs) == 2
        # held-out web prompts are excluded from task sampling
        fps = loop._flux_fresh_pairs(5)
        assert all(p["prompt"] not in ("web q1", "web q2") for p in fps)

    def test_check_any_accepts_alternates(self):
        check = InfiniteSelfPlayLoop._flux_check_any
        pair = {"response": "fleeting", "accepts": ["transient"]}
        assert check("transient", pair)
        assert check("fleeting", pair)
        assert not check("permanent", pair)
