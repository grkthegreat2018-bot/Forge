"""Source-grounded goal generation + tool use for FluxLM self-play (R53).

Replaces the purely-procedural ``_concise_qa_pairs`` stream for flux-mode
RSI with goals grounded in real reference material:

  * ``wiki``   — Wikipedia REST summaries (search-driven or random)
  * ``dict``   — Wiktionary REST definitions (dictionaryapi.dev
                 fallback), or a local kaikki.org Wiktionary jsonl
                 via ``--flux-dict-file``
  * ``paper``  — arXiv Atom API hits (title/authors/year/abstract)
  * ``web``    — DuckDuckGo/Bing/Google-News snippets (fallback tool)

Every generated goal is EXTRACTIVE: the canonical answer is a verbatim
substring of (or a direct field on) the fetched document, so ``_flux_check``
grading stays exact and the knowledge taught on failure is real.

Tool-call format taught to the model (two-phase generation):

    Tools: wiki <topic> | dict <word> | paper <topic> | web <query>
    Q: {question}
    T: wiki {query}          <- phase A: model emits tool + query
    O: {observation}         <- loop executes the tool, injects result
    A: {answer}              <- phase B: model answers from memory + obs

On failure the loop demonstrates BOTH steps with ``teach_text`` and
ingests the full trajectory — the n-gram/episodic/knn channels learn the
``Q:.. -> T: <tool> <query>`` binding verbatim, which is exactly the kind
of pattern an associative memory can internalize.

Fetched documents persist to ``flux_kb.jsonl`` under the status dir, so
knowledge compounds across epochs and restarts without re-fetching.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import quote_plus

from forge import web_primitives as wp

# ── tool surface ────────────────────────────────────────────────────────

TOOL_NAMES = ("wiki", "dict", "paper", "web")
_TOOL_ALIASES = {
    "wikipedia": "wiki", "wp": "wiki",
    "dictionary": "dict", "define": "dict", "wiktionary": "dict",
    "arxiv": "paper", "paper": "paper", "papers": "paper",
    "search": "web", "ddg": "web", "google": "web",
}

TOOL_HEADER = ("Tools: wiki <topic> | dict <word> | paper <topic> | "
               "web <query>")

_RE_TOOL_CALL = re.compile(
    r"^\s*\[?\s*([A-Za-z]+)\s*[:\]]?\s+(.+?)\s*\]?$")


def parse_tool_call(text: str) -> tuple[str, str] | None:
    """Parse one tool-call line -> (tool, query) or None.

    Accepts ``wiki Photosynthesis``, ``dict: ephemeral``,
    ``[paper] quantization`` — normalized through ``_TOOL_ALIASES``.
    """
    line = (text or "").strip().splitlines()
    if not line:
        return None
    m = _RE_TOOL_CALL.match(line[0].strip())
    if not m:
        return None
    name = _TOOL_ALIASES.get(m.group(1).lower(), m.group(1).lower())
    query = m.group(2).strip().strip('"\'')
    if name not in TOOL_NAMES or not query or len(query) > 120:
        return None
    return name, query


# ── seed query material (search seeds, not training data) ──────────────
# These only decide WHICH verified documents get fetched; all learned
# content comes from the fetched source itself.

_DICT_SEED_WORDS = (
    "neural", "tensor", "entropy", "compiler", "quantum", "algorithm",
    "parliament", "gravity", "mutation", "sonnet", "vector", "fossil",
    "theorem", "catalyst", "harbor", "ephemeral", "luminous", "migrate",
    "resonance", "isotope", "syntax", "equator", "mineral", "rhythm",
)

_ARXIV_SEED_QUERIES = (
    "attention mechanism", "language model", "reinforcement learning",
    "state space model", "quantization", "knowledge distillation",
    "graph neural network", "diffusion model", "machine translation",
    "protein folding", "information retrieval", "computer vision",
)

_WIKI_SEED_QUERIES = (
    "machine learning", "physics", "mathematics", "biology",
    "world history", "chemistry", "computer science", "astronomy",
    "geography", "literature", "economics", "philosophy",
)


# ── tools: cached, budgeted web access ─────────────────────────────────

class FluxTools:
    """Web-search/dictionary/arxiv tool executor for the flux loop.

    One instance per loop: caches every call so repeated queries are
    free, enforces a per-epoch fetch budget, and degrades to offline
    (probe once) when the network is unreachable.
    """

    def __init__(self, dict_file: str = "", obs_chars: int = 240,
                 timeout: float = 6.0, fetch_budget: int = 8):
        self.dict_file = dict_file
        self.obs_chars = obs_chars
        self.timeout = timeout
        self.fetch_budget = fetch_budget
        self._fetches = 0
        self._cache: dict[tuple[str, str], str] = {}
        self._doc_cache: dict[tuple[str, str], dict | None] = {}
        self._online: bool | None = None

    # ── plumbing ──
    def begin_epoch(self, budget: int | None = None) -> None:
        """Reset the per-epoch fetch counter (cache persists)."""
        self._fetches = 0
        if budget is not None:
            self.fetch_budget = budget

    def _get(self, url: str) -> str:
        """Budgeted GET — returns "" on any failure or spent budget."""
        if self._fetches >= self.fetch_budget:
            return ""
        self._fetches += 1
        try:
            return wp.http_get(url, timeout=self.timeout)
        except Exception:
            return ""

    def online(self) -> bool:
        """One-time connectivity probe — the probe itself IS a random
        Wikipedia summary fetch, so a successful probe costs nothing."""
        if self._online is None:
            try:
                body = wp.http_get(
                    "https://en.wikipedia.org/api/rest_v1/page/random"
                    "/summary", timeout=min(self.timeout, 4.0))
                doc = json.loads(body)
                self._online = bool(doc.get("extract"))
                if self._online:
                    self._pending_random = doc
            except Exception:
                self._online = False
        return self._online

    # ── document fetchers (goal-generation path — rich payloads) ──

    def _wiki_summary(self, title: str) -> dict | None:
        # REST path segments treat '+' literally — MediaWiki wants '_'
        body = self._get("https://en.wikipedia.org/api/rest_v1/page/"
                         f"summary/{quote_plus(title.replace(' ', '_'))}")
        if not body:
            return None
        try:
            d = json.loads(body)
        except Exception:
            return None
        extract = (d.get("extract") or "").strip()
        if d.get("type") != "standard" or len(extract) < 160:
            return None                     # disambig/stub/empty
        return {"source": "wiki", "key": d.get("title", title).lower(),
                "title": d.get("title", title), "extract": extract,
                "url": (d.get("content_urls", {}).get("desktop", {})
                        .get("page") or "")}

    def _wiki_doc(self, query: str) -> dict | None:
        if query:
            res = wp.wikipedia_search(query, n=3)
            for r in res.get("results", []):
                doc = self._wiki_summary(r["title"])
                if doc:
                    return doc
            return None
        pend = getattr(self, "_pending_random", None)
        if pend is not None:
            self._pending_random = None
            d = pend
        else:
            body = self._get("https://en.wikipedia.org/api/rest_v1/page/"
                             "random/summary")
            if not body:
                return None
            try:
                d = json.loads(body)
            except Exception:
                return None
        extract = (d.get("extract") or "").strip()
        if d.get("type") != "standard" or len(extract) < 160:
            return None
        return {"source": "wiki", "key": d.get("title", "").lower(),
                "title": d.get("title", ""), "extract": extract,
                "url": (d.get("content_urls", {}).get("desktop", {})
                        .get("page") or "")}

    def _kaikki_lookup(self, word: str) -> dict | None:
        """Stream-scan a local kaikki.org Wiktionary jsonl for ``word``."""
        try:
            with open(self.dict_file, "r", encoding="utf-8",
                      errors="replace") as f:
                for line in f:
                    if f'"word":"{word}"' not in line and \
                            f'"word": "{word}"' not in line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    if row.get("word") == word:
                        return self._dict_entry_from_kaikki(row)
        except OSError:
            return None
        return None

    @staticmethod
    def _dict_entry_from_kaikki(row: dict) -> dict | None:
        glosses, syns, ants = [], [], []
        for s in row.get("senses") or []:
            for g in s.get("glosses") or []:
                if isinstance(g, str) and g:
                    glosses.append(g)
            for r in s.get("synonyms") or []:
                w = r.get("word") if isinstance(r, dict) else r
                if w:
                    syns.append(str(w))
            for r in s.get("antonyms") or []:
                w = r.get("word") if isinstance(r, dict) else r
                if w:
                    ants.append(str(w))
        if not glosses:
            return None
        return {"source": "dict", "key": str(row["word"]).lower(),
                "word": str(row["word"]), "pos": row.get("pos", ""),
                "glosses": glosses, "synonyms": syns, "antonyms": ants}

    def _wiktionary_doc(self, word: str) -> dict | None:
        """Wiktionary REST definitions — {lang: [{partOfSpeech,
        definitions: [{definition: <html>}]}]}."""
        body = self._get("https://en.wiktionary.org/api/rest_v1/page/"
                         f"definition/{quote_plus(word)}")
        if not body:
            return None
        try:
            data = json.loads(body)
        except Exception:
            return None
        glosses, pos = [], ""
        for group in data.get("en") or []:
            pos = pos or group.get("partOfSpeech", "")
            for d in group.get("definitions") or []:
                g = wp.RE_TAG.sub("", d.get("definition") or "")
                g = re.sub(r"\s+", " ", g).strip()
                if g:
                    glosses.append(g)
        if not glosses:
            return None
        return {"source": "dict", "key": word, "word": word, "pos": pos,
                "glosses": glosses[:6], "synonyms": [], "antonyms": []}

    def _dict_doc(self, word: str) -> dict | None:
        word = word.strip().lower()
        if not re.fullmatch(r"[a-z][a-z'-]{1,30}", word):
            return None
        if self.dict_file:
            return self._kaikki_lookup(word)
        doc = self._wiktionary_doc(word)       # Wikimedia: most reliable
        if doc is not None:
            return doc
        body = self._get("https://api.dictionaryapi.dev/api/v2/entries/en/"
                         f"{quote_plus(word)}")
        if not body:
            return None
        try:
            rows = json.loads(body)
        except Exception:
            return None
        if not isinstance(rows, list) or not rows:
            return None
        glosses, syns, ants, pos = [], [], [], ""
        for m in rows[0].get("meanings") or []:
            pos = pos or m.get("partOfSpeech", "")
            syns += [str(s) for s in m.get("synonyms") or []]
            ants += [str(s) for s in m.get("antonyms") or []]
            for d in m.get("definitions") or []:
                g = d.get("definition")
                if g:
                    glosses.append(g)
                syns += [str(s) for s in d.get("synonyms") or []]
                ants += [str(s) for s in d.get("antonyms") or []]
        if not glosses:
            return None
        return {"source": "dict", "key": word, "word": word, "pos": pos,
                "glosses": glosses[:6],
                "synonyms": list(dict.fromkeys(syns))[:6],
                "antonyms": list(dict.fromkeys(ants))[:6]}

    def _arxiv_doc(self, query: str) -> dict | None:
        try:
            res = wp.arxiv_search(query, n=3)
        except Exception:
            return None
        for r in res.get("results", []):
            if not r.get("title"):
                continue
            url = r.get("url", "")
            m = re.search(r"(\d{4}\.\d{4,5})", url)
            return {"source": "arxiv", "key": (m.group(1) if m else
                                               r["title"].lower()),
                    "title": r["title"], "summary": r.get("summary", ""),
                    "url": url, "published": r.get("published", ""),
                    "authors": r.get("authors") or [],
                    "arxiv_id": m.group(1) if m else ""}
        return None

    def fetch_doc(self, source: str, query: str = "") -> dict | None:
        """Fetch one source document for goal generation (cached)."""
        if not self.online():
            return None
        key = (source, query.strip().lower())
        if key in self._doc_cache:
            return self._doc_cache[key]
        fn = {"wiki": self._wiki_doc, "dict": self._dict_doc,
              "arxiv": self._arxiv_doc}.get(source)
        doc = None
        if fn is not None:
            try:
                doc = fn(query)
            except Exception:
                doc = None
        self._doc_cache[key] = doc
        return doc

    # ── tool calls (model-facing path — compact obs strings) ──

    def call(self, name: str, query: str) -> str:
        """Execute a parsed tool call -> one-line observation string."""
        name = _TOOL_ALIASES.get(name.lower(), name.lower())
        if name not in TOOL_NAMES or not self.online():
            return "offline"
        key = (name, query.strip().lower())
        if key in self._cache:
            return self._cache[key]
        obs = ""
        try:
            if name == "wiki":
                doc = self.fetch_doc("wiki", query)
                if doc:
                    obs = f"{doc['title']} — {doc['extract']}"
            elif name == "dict":
                doc = self.fetch_doc("dict", query)
                if doc:
                    pos = f" ({doc['pos']})" if doc.get("pos") else ""
                    gl = "; ".join(doc["glosses"][:3])
                    obs = f"{doc['word']}{pos}: {gl}"
            elif name == "paper":
                doc = self.fetch_doc("arxiv", query)
                if doc:
                    yr = (doc.get("published") or "")[:4]
                    au = ", ".join((doc.get("authors") or [])[:2])
                    by = f" by {au}" if au else ""
                    obs = (f"{doc['title']} ({yr}){by} — "
                           f"{doc['summary']}")
            else:
                res = wp.ddg_search(query, n=3)
                bits = [f"{r['title']}: {r['snippet']}"
                        for r in res.get("results", [])]
                obs = " | ".join(bits)
        except Exception:
            obs = ""
        obs = re.sub(r"\s+", " ", obs).strip()[: self.obs_chars] \
            or "no results"
        self._cache[key] = obs
        return obs


# ── verified doc -> goal builders ───────────────────────────────────────

def _clip_words(text: str, n: int) -> str:
    w = text.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


def _sentence_with(text: str, pattern: str) -> tuple[str, str] | None:
    """Find the first regex hit -> (matched_str, full_sentence)."""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        m = re.search(pattern, sent)
        if m:
            return m.group(0), sent.strip()
    return None


def wiki_goals(doc: dict, max_ctx: int = 700) -> list[dict]:
    """Wikipedia summary -> verified extractive goals (1-3 per doc)."""
    title, extract = doc["title"], doc["extract"]
    ctx = extract[:max_ctx]
    base = {"source": "wiki", "context": ctx, "tool": "wiki",
            "query": title}
    out: list[dict] = []

    # is-a gloss: "What is {title}?" -> the defining noun phrase
    # (stop at commas + clause-introducing participles: trailing clauses
    # push the answer past the cap and dilute the definition)
    m = re.search(r"\b(?:is|are|was|were)\s+(?:an?|the)\s+([^.;,]{6,120})",
                  extract)
    if m:
        ans = re.split(
            r"\b(?:that|which|who|whom|whose|concerned|characterized|"
            r"composed|consisting|devoted|designed|focused|based|known|"
            r"used)\b", m.group(1))[0].strip()
        if len(ans) > 60:                 # trim to a word boundary
            ans = ans[:61].rsplit(" ", 1)[0].rstrip(" ,;:of")
        if 4 <= len(ans) <= 60:
            out.append({**base, "prompt":
                        f"What is {title}? Answer with a short phrase.",
                        "response": ans})
    # masked subject -> title recall
    first = re.split(r"(?<=[.!?])\s+", extract)[0]
    masked = re.sub(r"\b" + re.escape(title) + r"\b", "___", first,
                    count=1)
    if masked != first and len(masked) >= 30:
        out.append({**base, "prompt":
                    f'Name the subject: "{masked}"',
                    "response": title})
    # year cloze from the extract
    hit = _sentence_with(extract, r"\b(1[5-9]\d\d|20\d\d)\b")
    if hit:
        year, sent = hit
        blanked = sent.replace(year, "____", 1)
        if 15 <= len(blanked) <= 220:
            out.append({**base, "prompt":
                        f'Fill in the blank with a year: "{blanked}"',
                        "response": year})
    return out


def dict_goals(entry: dict) -> list[dict]:
    """Dictionary entry -> verified goals (definition/POS/syn/reverse)."""
    w, pos = entry["word"], entry.get("pos", "")
    glosses = entry.get("glosses") or []
    ctx = (f"{w} ({pos}): " if pos else f"{w}: ") \
        + "; ".join(glosses[:4]) + "."
    base = {"source": "dict", "context": ctx[:700], "tool": "dict",
            "query": w}
    out: list[dict] = []
    if pos:
        out.append({**base, "prompt": f"What part of speech is '{w}'?",
                    "response": pos})
    short = [g for g in glosses if 4 <= len(g) <= 70]
    if short:
        g = min(short, key=len)
        out.append({**base, "prompt":
                    f"What does '{w}' mean? Be brief.",
                    "response": g})
        if len(g.split()) <= 9:
            out.append({**base, "prompt":
                        f"What word means '{g}'?",
                        "response": w})
    syns = [s for s in entry.get("synonyms") or []
            if re.fullmatch(r"[a-zA-Z][a-zA-Z' -]{1,30}", s)]
    if syns:
        out.append({**base, "prompt": f"Name a synonym of '{w}'.",
                    "response": syns[0], "accepts": syns})
    ants = [s for s in entry.get("antonyms") or []
            if re.fullmatch(r"[a-zA-Z][a-zA-Z' -]{1,30}", s)]
    if ants:
        out.append({**base, "prompt": f"Name an antonym of '{w}'.",
                    "response": ants[0], "accepts": ants})
    return out


def arxiv_goals(doc: dict) -> list[dict]:
    """arXiv hit -> verified goals (year / first author / arXiv id)."""
    title = doc["title"]
    ctx = f"{title}. {doc.get('summary', '')}"[:700]
    base = {"source": "arxiv", "context": ctx, "tool": "paper",
            "query": title}
    out: list[dict] = []
    year = (doc.get("published") or "")[:4]
    if re.fullmatch(r"(19|20)\d\d", year):
        out.append({**base, "prompt":
                    f"In what year was the paper '{title}' published?",
                    "response": year})
    if doc.get("authors"):
        out.append({**base, "prompt":
                    f"Who is the first author of the paper '{title}'?",
                    "response": doc["authors"][0],
                    "accepts": doc["authors"][:4]})
    if doc.get("arxiv_id"):
        out.append({**base, "prompt":
                    f"What is the arXiv id of the paper '{title}'?",
                    "response": doc["arxiv_id"]})
    return out


_GOAL_BUILDERS = {"wiki": wiki_goals, "dict": dict_goals,
                  "arxiv": arxiv_goals}


def goals_from_doc(doc: dict) -> list[dict]:
    fn = _GOAL_BUILDERS.get(doc.get("source", ""))
    return fn(doc) if fn else []


# ── persistent knowledge base ───────────────────────────────────────────

class FluxKB:
    """Append-only jsonl store of fetched source documents.

    One row per unique (source, key) document. Fetched docs accumulate
    across epochs and runs — goal builders re-derive new question forms
    from cached docs without touching the network again.
    """

    def __init__(self, path: str):
        self.path = path
        self.docs: list[dict] = []
        self._keys: set[tuple[str, str]] = set()
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        doc = row.get("doc")
                        if isinstance(doc, dict) and doc.get("source"):
                            k = (doc["source"], str(doc.get("key", "")))
                            if k not in self._keys:
                                self._keys.add(k)
                                self.docs.append(doc)
            except OSError:
                pass

    def add(self, doc: dict) -> bool:
        k = (doc.get("source", ""), str(doc.get("key", "")))
        if not k[0] or not k[1] or k in self._keys:
            return False
        self._keys.add(k)
        self.docs.append(doc)
        if self.path:
            try:
                Path(self.path).parent.mkdir(parents=True,
                                             exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"doc": doc},
                                       ensure_ascii=False) + "\n")
            except OSError:
                pass
        return True


# ── goal sampler ───────────────────────────────────────────────────────

class FluxGoalSampler:
    """Draws verified goals from web sources, seeded by topic.

    ``sample_web(n)`` mixes (a) freshly fetched documents — bounded by
    the tools' per-epoch fetch budget — and (b) KB replay: cached docs
    re-derived through the goal builders so coverage compounds even when
    the budget is spent or the network is down.
    """

    def __init__(self, tools: FluxTools, kb: FluxKB | None = None,
                 sources=("wiki", "dict", "arxiv"), seed: int = 0):
        self.tools = tools
        self.kb = kb or FluxKB("")
        self.sources = tuple(s for s in sources if s in _GOAL_BUILDERS)
        self.rng = random.Random(0xF105 + seed)
        self._dict_words: deque = deque(_DICT_SEED_WORDS)
        self._arxiv_queries: deque = deque(_ARXIV_SEED_QUERIES)
        self._wiki_queries: deque = deque(_WIKI_SEED_QUERIES)
        self.rng.shuffle(self._dict_words)
        self.rng.shuffle(self._arxiv_queries)
        self.rng.shuffle(self._wiki_queries)

    # queue management: rotate seed terms; chain-feed new words found
    # in fetched wiki titles back into the dict/arxiv queues
    def _next_query(self, source: str) -> str:
        q = {"wiki": self._wiki_queries, "dict": self._dict_words,
             "arxiv": self._arxiv_queries}[source]
        if not q:
            return ""
        term = q.popleft()
        q.append(term)                       # round-robin, never empty
        return term

    def _chain_feed(self, doc: dict) -> None:
        for w in re.findall(r"[A-Za-z]{5,}",
                            str(doc.get("title", ""))):
            lw = w.lower()
            if lw not in self._dict_words and \
                    lw not in _DICT_SEED_WORDS:
                self._dict_words.append(lw)

    def _emit(self, doc: dict, seen: set, eval_prompts: set,
              out: list, n: int) -> None:
        if self.kb.add(doc):
            self._chain_feed(doc)
        for g in goals_from_doc(doc):
            if len(out) >= n:
                break
            if g["prompt"] in seen or g["prompt"] in eval_prompts:
                continue
            out.append(g)

    def sample_web(self, n: int, seen: set, eval_prompts: set,
                   epoch: int = 0) -> list[dict]:
        """Up to ``n`` web-sourced goals, deduped against ``seen`` and
        the held-out eval prompts. Never raises — offline -> []."""
        out: list[dict] = []
        if n <= 0 or not self.sources or not self.tools.online():
            return out
        # fresh docs, round-robin across enabled sources (budget-bound)
        tries = 0
        max_tries = max(4, self.tools.fetch_budget * 2)
        while len(out) < n and tries < max_tries:
            src = self.sources[tries % len(self.sources)]
            tries += 1
            doc = self.tools.fetch_doc(src, self._next_query(src))
            if doc:
                self._emit(doc, seen, eval_prompts, out, n)
        # KB replay — cached docs produce NEW question variants
        if len(out) < n and self.kb.docs:
            docs = self.kb.docs
            start = epoch % len(docs)
            order = docs[start:] + docs[:start]
            for doc in order:
                if len(out) >= n:
                    break
                for g in goals_from_doc(doc):
                    if len(out) >= n:
                        break
                    if g["prompt"] in seen or g["prompt"] in eval_prompts:
                        continue
                    out.append(g)
        return out

    def build_eval(self, n: int, eval_path: str,
                   eval_prompts: set) -> list[dict]:
        """Frozen held-out web eval set — built once, persisted to
        ``eval_path``, reused verbatim on later epochs/restarts."""
        if n <= 0 or not eval_path:
            return []
        if os.path.exists(eval_path):
            try:
                rows = json.loads(Path(eval_path).read_text(
                    encoding="utf-8"))
                return [r for r in rows
                        if isinstance(r, dict) and r.get("prompt")]
            except Exception:
                pass
        # fresh budget for eval construction (independent of epochs)
        saved_budget = self.tools.fetch_budget
        self.tools.begin_epoch(budget=max(6, n // 2))
        try:
            goals = self.sample_web(n, seen=set(),
                                    eval_prompts=eval_prompts)
        finally:
            self.tools.begin_epoch(budget=saved_budget)
        rows = [{"prompt": g["prompt"], "response": g["response"],
                 "accepts": g.get("accepts") or [],
                 "source": g["source"]} for g in goals[:n]]
        if rows:
            try:
                Path(eval_path).parent.mkdir(parents=True,
                                             exist_ok=True)
                Path(eval_path).write_text(
                    json.dumps(rows, ensure_ascii=False, indent=1),
                    encoding="utf-8")
            except OSError:
                pass
        return rows
