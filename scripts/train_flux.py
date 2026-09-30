"""Train a FluxLM on a text/jsonl corpus or HF dataset — streaming,
instant, hot.

Every token updates the sparse memory cells immediately; there is no
separate "training phase" — ingest IS the weights.  Snapshots preserve
the learned state; the journal gives per-tag revert.

Usage:
    python scripts/train_flux.py corpus.txt --out research/checkpoints/flux/base.flux
    python scripts/train_flux.py data/sft/nontool_general.jsonl \
        --field text --out research/checkpoints/flux/sft.flux --tag sft
    python scripts/train_flux.py corpus.txt --resume research/checkpoints/flux/base.flux

    # straight from HuggingFace — Wikipedia + Wiktionary live-learn:
    python scripts/train_flux.py --hf wikimedia/wikipedia \
        --hf-config 20231101.en --wiki-filter --topic ai \
        --device cuda --cuda-primary \
        --max-tokens 50000000 --ckpt-every 2000000 \
        --out research/checkpoints/flux/wiki_ai.flux
    python scripts/train_flux.py kaikki.org-dictionary-English.jsonl \
        --dict-flatten --topic ai --device cuda --cuda-primary \
        --out research/checkpoints/flux/wiktionary_ai.flux
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

from forge.model.flux import FluxConfig, FluxLM  # noqa: E402
from research.tokenizer_cache import get_tokenizer  # noqa: E402
from forge.engine.engine_common import _tokenizer_for_vocab  # noqa: E402


def _iter_texts(path: Path, field: str | None, dict_flatten: bool = False,
                topic=None):
    """Yield raw text chunks from .txt/.md or .jsonl (field or common keys)."""
    if path.suffix.lower() in (".jsonl", ".ndjson"):
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if dict_flatten:
                    t = _flatten_dict_entry(row)
                    if t and (topic is None or topic(
                            str(row.get("word") or ""), t)):
                        yield t
                    continue
                if field and field in row:
                    v = row[field]
                else:
                    v = next((row[k] for k in
                              ("text", "content", "output", "response",
                               "answer", "prompt")
                              if isinstance(row.get(k), str)
                              and row[k]), None)
                if v is not None:
                    if topic is None or topic(
                            str(row.get("title") or ""), str(v)):
                        yield str(v)
                    continue
                else:
                    # chat transcripts: flatten role-tagged turns
                    turns = row.get("messages") or row.get("conversations")
                    if isinstance(turns, list):
                        yield "".join(
                            f"<|im_start|>{t.get('role', 'user')}\n"
                            f"{t.get('content', t.get('value', ''))}"
                            f"<|im_end|>\n" for t in turns
                            if isinstance(t, dict))
    elif path.suffix.lower() == ".json":
        rows = json.loads(path.read_text(encoding="utf-8",
                                         errors="replace"))
        if isinstance(rows, dict):
            rows = rows.get("conversations", rows.get("data", []))
        for row in rows if isinstance(rows, list) else []:
            if isinstance(row, str):
                yield row
            elif isinstance(row, dict):
                # chat transcripts: flatten turns in order
                turns = row.get("conversations") or row.get("messages")
                if isinstance(turns, list):
                    for t in turns:
                        v = t.get("value") or t.get("content") or ""
                        if v:
                            yield str(v)
                elif field and field in row:
                    yield str(row[field])
    else:
        yield path.read_text(encoding="utf-8", errors="replace")


def _flatten_dict_entry(row: dict) -> str | None:
    """kaikki.org Wiktionary jsonl row -> readable definition text.

    {"word": ..., "pos": ..., "senses": [{"glosses": [...]}]} renders as
    "word (noun): gloss1. gloss2." — one line per entry, exactly the
    fact-shape a memory model recalls best."""
    word = row.get("word")
    if not word:
        return None
    pos = row.get("pos", "")
    glosses: list[str] = []
    for s in row.get("senses") or []:
        for g in s.get("glosses") or []:
            if isinstance(g, str) and g:
                glosses.append(g)
    if not glosses:
        return None
    head = f"{word} ({pos})" if pos else str(word)
    return head + ": " + "; ".join(glosses[:4]) + "."


# ── topic filters ───────────────────────────────────────────────────────
# High-precision keyword packs — a false negative costs one skipped page,
# a false positive pollutes memory with off-domain noise.  Title match
# accepts outright; otherwise the intro needs >=MIN_HITS distinct hits.
TOPICS: dict[str, dict] = {
    "ai": {
        "title": [
            "machine learning", "artificial intelligence", "neural net",
            "deep learning", "language model", "large language model",
            "transformer", "natural language", "computer vision",
            "reinforcement learning", "generative", "chatbot", "gpt",
            "bert", "diffusion model", "knowledge graph", "tensorflow",
            "pytorch", "openai", "anthropic", "llm", "tokenizer",
            "attention mechanism", "backpropagation", "gradient descent",
            "supervised learning", "unsupervised", "embedding",
            "computer science", "algorithm", "programming",
            "software", "compiler", "operating system", "database",
            "python", "javascript", "linux", "gpu", "cuda", "api",
            "data structure", "distributed", "cryptograph", "internet",
            "world wide web", "cloud comput", "open-source",
        ],
        "text": [
            "machine learning", "artificial intelligence", "neural network",
            "deep learning", "language model", "natural language",
            "reinforcement learning", "training data", "computer program",
            "programming language", "software", "algorithm", "computer",
            "source code", "compiler", "data structure", "open-source",
            "artificial neural", "token", "inference", "parameter",
        ],
        "min_hits": 2,
    },
    "code": {
        "title": [
            "programming", "software", "compiler", "python", "javascript",
            "rust", "c++", "java ", "linux", "operating system",
            "algorithm", "data structure", "api", "database", "sql",
            "version control", "git", "debugg", "interpreter",
        ],
        "text": [
            "programming language", "source code", "software", "compiler",
            "interpreter", "algorithm", "data structure", "command-line",
            "library", "runtime", "debugg",
        ],
        "min_hits": 2,
    },
    "science": {
        "title": [
            "physics", "quantum", "mathematics", "theorem", "biology",
            "chemistry", "genetics", "neuroscience", "astronomy",
            "particle", "relativity", "evolution",
        ],
        "text": [
            "physicist", "mathematical", "theorem", "quantum", "species",
            "protein", "experiment", "hypothesis",
        ],
        "min_hits": 2,
    },
}


def _topic_scorer(spec: dict):
    """Compile a topic spec into a (title, text) -> bool scorer."""
    tp = re.compile("|".join(re.escape(k) for k in spec["title"]), re.I)
    xp = re.compile("|".join(re.escape(k) for k in spec["text"]), re.I)
    min_hits = int(spec.get("min_hits", 2))

    def accept(title: str, text: str) -> bool:
        if tp.search(title):
            return True
        head = text[:3000]
        return len(set(m.group(0).lower() for m in xp.finditer(head))) \
            >= min_hits
    return accept


def _resolve_topics(arg: str) -> dict:
    """'ai,code' -> merged spec; unknown names = literal keywords."""
    merged: dict[str, list[str] | int] = {"title": [], "text": [],
                                         "min_hits": 2}
    custom = []
    for name in arg.split(","):
        name = name.strip()
        if name in TOPICS:
            s = TOPICS[name]
            merged["title"] += s["title"]
            merged["text"] += s["text"]
            merged["min_hits"] = min(merged["min_hits"],
                                     s.get("min_hits", 2))
        elif name:
            custom.append(name)
    merged["title"] += custom
    merged["text"] += custom
    return merged


def _iter_hf(args):
    """Stream rows from an HF dataset — yields (title, text) pairs."""
    from datasets import load_dataset  # late import — heavy dep
    ds = load_dataset(args.hf, args.hf_config, split=args.hf_split,
                      streaming=True)
    should_skip = None
    if args.wiki_filter:
        try:
            from download_wikipedia import should_skip
        except ImportError:
            print("warn: download_wikipedia.py not importable — "
                  "--wiki-filter ignored")
    topic = _topic_scorer(_resolve_topics(args.topic)) \
        if args.topic else None
    n_scan = n_skip = 0
    for row in ds:
        title = str(row.get("title") or "")
        text = row.get(args.hf_field)
        if not isinstance(text, str) or not text:
            continue
        n_scan += 1
        if should_skip is not None and should_skip(title, text):
            n_skip += 1
            continue
        if topic is not None and not topic(title, text):
            n_skip += 1
            continue
        if n_scan % 5000 == 0:
            print(f"    [filter] {n_scan:,} scanned | "
                  f"{n_skip:,} skipped "
                  f"({100 * n_skip / n_scan:.1f}% off-topic)")
        yield title, text
    print(f"    [filter] done: {n_scan:,} scanned, {n_skip:,} skipped, "
          f"{n_scan - n_skip:,} ingested")


def _learn_docs(model, tok, docs, tag, args, t_start, state):
    """Ingest (label, text) docs; soft_reset between them (independent
    episodes); periodic snapshots for crash safety.  With --pack-docs,
    consecutive short docs are joined into one chunk_chars blob before
    ingest - amortizes the per-ingest block snapshot over thousands of
    tiny records (QA/dictionary rows) instead of paying it per row."""
    pack = getattr(args, "pack_docs", 0)
    pending, last_label = "", ""

    def emit(text, label):
        """One episode: chunked ingest + soft_reset."""
        for i in range(0, len(text), args.chunk_chars):
            ids = tok.encode(text[i:i + args.chunk_chars],
                             add_special_tokens=False)
            state["tokens"] += model.ingest(ids, tag=tag)
            if (args.ckpt_every and
                    state["tokens"] - state["last_ckpt"] >= args.ckpt_every):
                model.snapshot(args.out)
                state["last_ckpt"] = state["tokens"]
                print(f"    [ckpt] {state['tokens']:,} tok -> {args.out}")
        model.soft_reset()
        if state["docs"] % 500 == 0:
            el = time.perf_counter() - t_start
            print(f"    {state['docs']:,} docs | {state['tokens']:,} tok "
                  f"({state['tokens'] / max(el, 1e-9):,.0f} tok/s) "
                  f"last: {label[:60]}")
        return (args.max_tokens and state["tokens"] >= args.max_tokens) \
            or (args.max_docs and state["docs"] >= args.max_docs)

    for label, text in docs:
        buffered = False
        if pack and len(text) < args.chunk_chars:
            pending += text + "\n\n"
            last_label = label
            buffered = True
            if len(pending) < pack:
                state["docs"] += 1
                continue
            text, pending = pending, ""
            label = last_label
        if emit(text, label):
            return True
        if not buffered:
            state["docs"] += 1
        if args.max_docs and state["docs"] >= args.max_docs:
            return True
    if pending:
        emit(pending, last_label)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", nargs="*", help="text/jsonl/json files")
    ap.add_argument("--out",
                    default="research/checkpoints/flux/model.flux",
                    help="snapshot path — under research/checkpoints/ "
                         "it appears in the GUI model index")
    ap.add_argument("--resume", help="snapshot to continue training")
    ap.add_argument("--field", help="jsonl field name")
    ap.add_argument("--dict-flatten", action="store_true",
                    help="kaikki wiktionary jsonl -> 'word (pos): gloss'")
    ap.add_argument("--tag", default=None,
                    help="journal tag (default: filename)")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cpu",
                    help="cpu or cuda (dense readout device; memory is "
                         "host-resident either way)")
    ap.add_argument("--cuda-primary", action="store_true",
                    help="GPU-resident memory tables + bulk ingest "
                         "(~10x faster; use for large corpora)")
    ap.add_argument("--graft", default=None,
                    help="teacher checkpoint (safetensors/.pt or dir) — "
                         "graft its embedding table as the feature "
                         "matrix BEFORE training (e.g. "
                         "research/checkpoints/ForgeLM_V2.safetensors)")
    ap.add_argument("--graft-method", default="randproj",
                    choices=["pca", "randproj"])
    ap.add_argument("--chunk-chars", type=int, default=1 << 18)
    ap.add_argument("--pack-docs", type=int, default=0, metavar="CHARS",
                    help="pack consecutive short docs into CHARS-sized "
                         "blobs before ingest (e.g. 262144) — amortizes "
                         "per-ingest block overhead for corpora of tiny "
                         "records; docs share stream context within a "
                         "pack, soft_reset happens per pack")
    # streaming / corpus-scale options
    ap.add_argument("--hf", help="HF dataset, streamed "
                               "(e.g. wikimedia/wikipedia)")
    ap.add_argument("--hf-config", default=None)
    ap.add_argument("--hf-split", default="train")
    ap.add_argument("--hf-field", default="text")
    ap.add_argument("--wiki-filter", action="store_true",
                    help="skip disambiguation/list/stub wiki pages")
    ap.add_argument("--topic", default=None,
                    help="domain filter — preset names (ai, code, "
                         "science) or comma-separated keywords; "
                         "applies to --hf rows and jsonl/json records "
                         "(title+intro scored; .txt files unfiltered)")
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=0,
                    help="snapshot every N tokens (crash-safe long runs)")
    ap.add_argument("--gcell", action="store_true",
                    help="enable the learned-key gcell channel (SimHash "
                         "SDM, fuzzy read) — ignored when --resume "
                         "(config comes from the snapshot)")
    ap.add_argument("--gsyn", action="store_true",
                    help="enable the dense delta-rule gsyn channel — "
                         "ignored when --resume")
    args = ap.parse_args()

    if not args.corpus and not args.hf:
        ap.error("need corpus files or --hf")

    cfg = FluxConfig(vocab_size=args.vocab, device=args.device,
                     cuda_primary=args.cuda_primary,
                     gcell=args.gcell, gsyn=args.gsyn)
    model = (FluxLM.load(args.resume, device=args.device,
                         cuda_primary=args.cuda_primary or None)
             if args.resume else FluxLM(cfg))
    if args.graft:
        info = model.graft_teacher(args.graft, method=args.graft_method)
        print(f"graft: {info['method']} {info['teacher_dim']}->"
              f"{info['topic_dim']} from {info['src']}")
    tok = get_tokenizer(args.tokenizer
                        or _tokenizer_for_vocab(model.vocab_size))

    t_start = time.perf_counter()
    state = {"tokens": 0, "docs": 0, "last_ckpt": 0}
    done = False

    if args.hf:
        tag = args.tag or args.hf.split("/")[-1]
        done = _learn_docs(model, tok, _iter_hf(args), tag,
                           args, t_start, state)
        print(f"  {args.hf}: {state['docs']:,} docs streamed")

    if not done:
        for corpus in args.corpus:
            path = Path(corpus)
            tag = args.tag or path.stem
            topic = _topic_scorer(_resolve_topics(args.topic)) \
                if args.topic and path.suffix.lower() in \
                (".jsonl", ".ndjson", ".json") else None
            docs = ((path.name, t)
                    for t in _iter_texts(path, args.field,
                                         args.dict_flatten, topic))
            done = _learn_docs(model, tok, docs, tag, args,
                               t_start, state)
            el = time.perf_counter() - t_start
            print(f"  {path.name}: +{state['tokens']:,} tok "
                  f"({state['tokens'] / max(el, 1e-9):,.0f} tok/s)")
            if done:
                break

    out = model.snapshot(args.out)
    rep = model.memory_report()
    cells = rep.get("tables_cells", rep.get("gpu_rows_used", 0))
    print(f"saved {out}")
    print(f"memory: {rep['total_MB_est']:.0f} MB est | "
          f"cells={cells:,} | epi={rep['epi_positions']:,} | "
          f"gcell={rep.get('gcell_cells', 0):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
