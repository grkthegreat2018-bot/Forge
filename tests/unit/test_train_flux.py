"""Tests for train_flux.py corpus tooling — topic filter, dict flatten,
jsonl field extraction."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from train_flux import (  # noqa: E402
    _flatten_dict_entry,
    _hf_row_text,
    _iter_texts,
    _resolve_topics,
    _topic_scorer,
)


class TestTopicScorer:
    def test_title_match_accepts(self):
        sc = _topic_scorer(_resolve_topics("ai"))
        assert sc("Transformer (deep learning)", "anything")

    def test_multiword_title_keywords(self):
        sc = _topic_scorer(_resolve_topics("ai"))
        assert sc("Large language model", "x")
        assert sc("History of machine learning", "x")

    def test_off_topic_rejected(self):
        sc = _topic_scorer(_resolve_topics("ai,code"))
        assert not sc("Battle of Hastings",
                      "fought in 1066 between the Norman-French army "
                      "and an English army under Harold Godwinson")
        assert not sc("Manchester United",
                      "professional football club based in Old Trafford")

    def test_text_needs_distinct_hits(self):
        sc = _topic_scorer(_resolve_topics("science"))
        # one keyword repeated does not reach min_hits=2
        assert not sc("Untitled", "quantum quantum quantum quantum")
        assert sc("Untitled", "the quantum experiment confirmed "
                              "the hypothesis")

    def test_custom_keywords(self):
        sc = _topic_scorer(_resolve_topics("flux capacitor"))
        assert sc("Flux capacitor", "plain text")
        # single distinct custom keyword in text < min_hits
        assert not sc("Untitled", "flux capacitor flux capacitor")

    def test_preset_merge(self):
        spec = _resolve_topics("ai,code")
        assert "transformer" in spec["title"]
        assert "compiler" in spec["title"]


class TestDictFlatten:
    def test_glosses_render(self):
        row = {"word": "backpropagation", "pos": "noun",
               "senses": [{"glosses": ["the algorithm for training "
                                       "neural networks",
                                       "second gloss"]}]}
        out = _flatten_dict_entry(row)
        assert out.startswith("backpropagation (noun):")
        assert "algorithm" in out

    def test_missing_word_or_gloss(self):
        assert _flatten_dict_entry({"pos": "noun"}) is None
        assert _flatten_dict_entry({"word": "x"}) is None


class TestIterTextsFilter:
    def _write_jsonl(self, tmp_path, rows):
        p = tmp_path / "corpus.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows),
                     encoding="utf-8")
        return p

    def test_jsonl_title_scored(self, tmp_path):
        sc = _topic_scorer(_resolve_topics("ai"))
        rows = [
            {"title": "Neural network",
             "text": "A neural network is a machine learning model."},
            {"title": "Cricket (sport)",
             "text": "Cricket is a bat-and-ball game played between "
                     "two teams."},
        ]
        p = self._write_jsonl(tmp_path, rows)
        got = list(_iter_texts(p, None, False, sc))
        assert len(got) == 1 and "neural network" in got[0]

    def test_jsonl_no_topic_yields_all(self, tmp_path):
        rows = [{"title": "Cricket", "text": "bat and ball game"},
                {"title": "Neural network", "text": "ml model"}]
        p = self._write_jsonl(tmp_path, rows)
        assert len(list(_iter_texts(p, None, False, None))) == 2

    def test_dict_flatten_filtered(self, tmp_path):
        sc = _topic_scorer(_resolve_topics("ai"))
        rows = [
            {"word": "backpropagation", "pos": "noun",
             "senses": [{"glosses": ["algorithm for training "
                                     "neural networks"]}]},
            {"word": "aardvark", "pos": "noun",
             "senses": [{"glosses": ["a nocturnal mammal"]}]},
        ]
        p = self._write_jsonl(tmp_path, rows)
        got = list(_iter_texts(p, None, True, sc))
        assert len(got) == 1 and got[0].startswith("backpropagation")

    def test_txt_unfiltered(self, tmp_path):
        sc = _topic_scorer(_resolve_topics("ai"))
        p = tmp_path / "notes.txt"
        p.write_text("a recipe for chocolate cake", encoding="utf-8")
        assert len(list(_iter_texts(p, None, False, sc))) == 1


class TestHfTemplate:
    def test_qa_template(self):
        row = {"question": "What is entropy?",
               "answer": "a measure of disorder"}
        out = _hf_row_text(row, "Q: {question} A: {answer}")
        assert out == "Q: What is entropy? A: a measure of disorder"

    def test_squad_style_answers_dict(self):
        # squad rows: answers = {"text": [...], "answer_start": [...]}
        row = {"question": "Capital of France?",
               "answers": {"text": ["Paris"], "answer_start": [0]}}
        out = _hf_row_text(row, "Q: {question} A: {answers}")
        assert out == "Q: Capital of France? A: Paris"

    def test_triviaqa_style_value_list(self):
        row = {"question": "H2O is?",
               "answer": {"value": "water", "aliases": ["w"]}}
        out = _hf_row_text(row, "Q: {question} A: {answer}")
        assert "water" in out

    def test_missing_key_blank(self):
        out = _hf_row_text({"a": "1"}, "{a}|{missing}")
        assert out == "1|"
