"""Process all ForgeAI datasets into a unified tokenized training set for V7.

Scans all data sources (pretrain, SFT, self-play, finetune), normalizes them
into the standard training format, samples a representative subset, tokenizes
with the LFM2.5 tokenizer, and saves packed sequences ready for training.

Output: research/data/v7_train/
  - train.bin   (packed token IDs, ready for CPT or SFT)
  - val.bin     (validation split)
  - manifest.json (dataset stats, source counts, token counts)

Usage:
  python -m research.sandbox.process_v7_data --max-per-source 200 --seq-len 2048
  python -m research.sandbox.process_v7_data --max-per-source 5000 --seq-len 4096
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

if os.name != "nt":  # expandable_segments unsupported on Windows
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "research" / "data"
OUT = DATA / "v7_train"


# ── Source discovery ────────────────────────────────────────────────────────

def find_jsonl_sources():
    """Find all JSONL files across data directories."""
    sources = {}
    # Pretrain (annealing)
    for d in ["synthetic_qa", "synthetic_code_snippet", "algorithmic_corpus"]:
        files = sorted(glob.glob(str(DATA / "pretrain" / "annealing" / d / "*.jsonl")))
        if files:
            sources[f"pretrain/{d}"] = files
    # Pretrain: Wikipedia EN filtered to STEM/coding (R&D round 15)
    # Use wikipedia_filtered if available, fall back to raw wikipedia
    wiki_dir = DATA / "pretrain" / "wikipedia_filtered"
    if not wiki_dir.exists() or not list(wiki_dir.glob("*.jsonl")):
        wiki_dir = DATA / "pretrain" / "wikipedia"
    files = sorted(glob.glob(str(wiki_dir / "*.jsonl")))
    if files:
        sources["pretrain/wikipedia"] = files
    # Pretrain: OpenWebMath (R&D round 15)
    files = sorted(glob.glob(str(DATA / "pretrain" / "openwebmath" / "*.jsonl")))
    if files:
        sources["pretrain/openwebmath"] = files
    # Pretrain: FineWeb-Edu additional (R&D round 15)
    files = sorted(glob.glob(str(DATA / "pretrain" / "fineweb_edu_additional" / "*.jsonl")))
    if files:
        sources["pretrain/fineweb_edu"] = files
    # Pretrain fineweb-math
    files = sorted(glob.glob(str(DATA / "pretrain" / "fineweb-math" / "data" / "*.parquet")))
    if files:
        sources["pretrain/fineweb-math"] = files
    # SFT coding
    for d in ["evol-codealpaca", "magicoder-evol", "magicoder-oss"]:
        files = sorted(glob.glob(str(DATA / "sft" / "coding" / d / "*.jsonl")))
        if files:
            sources[f"sft/coding/{d}"] = files
    # SFT coding parquet
    for d in ["opc-sft-stage1", "opc-sft-stage2"]:
        files = sorted(glob.glob(str(DATA / "sft" / "coding" / d / "data" / "*.parquet")))
        if not files:
            files = sorted(glob.glob(str(DATA / "sft" / "coding" / d / "*" / "*.parquet")))
        if files:
            sources[f"sft/coding/{d}"] = files
    # SFT reasoning
    for d in ["am-r1-distill-1.4M", "metamath", "numina-math-1.5", "openthoughts-114k",
              "mixture-of-thoughts"]:
        files = sorted(glob.glob(str(DATA / "sft" / "reasoning" / d / "data" / "*.parquet")))
        if not files:
            files = sorted(glob.glob(str(DATA / "sft" / "reasoning" / d / "*" / "*.parquet")))
        if not files:
            files = sorted(glob.glob(str(DATA / "sft" / "reasoning" / d / "*.json")))
        if files:
            sources[f"sft/reasoning/{d}"] = files
    # SFT tool_use — EXCLUDE nemotron-agentic-v2 (customer service, not coding)
    for d in ["glaive-fc-v2", "hermes-reasoning-tool", "toolace"]:
        files = sorted(glob.glob(str(DATA / "sft" / "tool_use" / d / "data" / "*.parquet")))
        if not files:
            files = sorted(glob.glob(str(DATA / "sft" / "tool_use" / d / "*.json")))
        if not files:
            files = sorted(glob.glob(str(DATA / "sft" / "tool_use" / d / "*.jsonl")))
        if not files:
            files = sorted(glob.glob(str(DATA / "sft" / "tool_use" / d / "data" / "*.jsonl")))
        if files:
            sources[f"sft/tool_use/{d}"] = files
    # SFT agentic
    for d in ["agent-instruct", "agent-trek", "coderforge", "open-swe-traces",
              "opencode-agentic-mini"]:
        files = sorted(glob.glob(str(DATA / "sft" / "agentic" / d / "data" / "*.parquet")))
        if not files:
            files = sorted(glob.glob(str(DATA / "sft" / "agentic" / d / "trajectories" / "*.parquet")))
        if not files:
            files = sorted(glob.glob(str(DATA / "sft" / "agentic" / d / "*.json")))
        if files:
            sources[f"sft/agentic/{d}"] = files
    # Finetune (self-play outputs)
    files = sorted(glob.glob(str(DATA / "finetune" / "*.jsonl")))
    if files:
        sources["finetune/self_play"] = files
    return sources


# ── Loaders per format ──────────────────────────────────────────────────────

def load_jsonl(path: str, max_n: int) -> list[dict]:
    """Load JSONL with prompt/response or messages format."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if len(examples) >= max_n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if "messages" in obj:
                examples.append({"type": "multi_turn", "messages": obj["messages"]})
            elif "prompt" in obj and ("response" in obj or "solution" in obj):
                resp = obj.get("response", obj.get("solution", ""))
                examples.append({"type": "single_turn",
                                 "prompt": obj["prompt"], "response": resp})
            elif "text" in obj:
                examples.append({"type": "text", "text": obj["text"]})
    return examples


def load_json(path: str, max_n: int) -> list[dict]:
    """Load a JSON file (array of examples or MetaMath format)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    examples = []
    if isinstance(data, list):
        for obj in data[:max_n]:
            if "messages" in obj:
                examples.append({"type": "multi_turn", "messages": obj["messages"]})
            elif "prompt" in obj and "response" in obj:
                examples.append({"type": "single_turn",
                                 "prompt": obj["prompt"], "response": obj["response"]})
            elif "query" in obj and "response" in obj:
                examples.append({"type": "single_turn",
                                 "prompt": obj["query"], "response": obj["response"]})
            elif "text" in obj:
                examples.append({"type": "text", "text": obj["text"]})
    elif isinstance(data, dict):
        # MetaMath format: {"query": ..., "response": ...}
        if "query" in data and "response" in data:
            examples.append({"type": "single_turn",
                             "prompt": data["query"], "response": data["response"]})
    return examples


def load_parquet(path: str, max_n: int) -> list[dict]:
    """Load examples from a parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("  pyarrow not installed, skipping parquet")
        return []
    try:
        table = pq.read_table(path, columns=None)
        rows = table.to_pylist()
    except Exception as e:
        print(f"  Parquet read error: {e}")
        return []
    examples = []
    for row in rows[:max_n]:
        if "messages" in row and row["messages"]:
            msgs = row["messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            # Normalize: handle list of strings or list of dicts
            normalized = []
            for m in msgs:
                if isinstance(m, str):
                    m = json.loads(m)
                if isinstance(m, dict):
                    normalized.append(m)
            if normalized:
                examples.append({"type": "multi_turn", "messages": normalized})
        elif "prompt" in row and ("response" in row or "completion" in row or "solution" in row):
            resp = row.get("response", row.get("completion", row.get("solution", "")))
            examples.append({"type": "single_turn",
                             "prompt": row["prompt"], "response": resp})
        elif "query" in row and "response" in row:
            examples.append({"type": "single_turn",
                             "prompt": row["query"], "response": row["response"]})
        elif "text" in row:
            examples.append({"type": "text", "text": row["text"]})
        elif "conversations" in row:
            # OpenHermes format: [{"from": "human", "value": ...}, ...]
            convs = row["conversations"]
            messages = []
            for c in convs:
                role = "user" if c.get("from") == "human" else "assistant"
                messages.append({"role": role, "content": c.get("value", "")})
            if messages:
                examples.append({"type": "multi_turn", "messages": messages})
    return examples


def load_source(name: str, files: list[str], max_per_source: int) -> list[dict]:
    """Load up to max_per_source examples from a source."""
    examples = []
    per_file = max(1, max_per_source // max(1, len(files)))
    for f in files:
        if len(examples) >= max_per_source:
            break
        if f.endswith(".parquet"):
            exs = load_parquet(f, per_file)
        elif f.endswith(".json"):
            exs = load_json(f, per_file)
        else:
            exs = load_jsonl(f, per_file)
        examples.extend(exs)
    return examples[:max_per_source]


# ── Rendering to training text ──────────────────────────────────────────────

def render_example(ex: dict) -> str:
    """Render an example to text for tokenization."""
    if ex["type"] == "multi_turn":
        parts = []
        for m in ex["messages"]:
            if isinstance(m, str):
                m = json.loads(m)
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "user":
                parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
            elif role == "assistant":
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
            elif role == "tool":
                parts.append(f"<|im_start|>tool\n{content}<|im_end|>\n")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)
    elif ex["type"] == "single_turn":
        return (f"<|im_start|>user\n{ex['prompt']}<|im_end|>\n"
                f"<|im_start|>assistant\n{ex['response']}<|im_end|>\n")
    elif ex["type"] == "text":
        return ex["text"] + "\n\n"
    return ""


# ── Main processing ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Process all datasets for V7 training")
    parser.add_argument("--max-per-source", type=int, default=500,
                        help="Max examples to load per source (default 500 for smoke)")
    parser.add_argument("--seq-len", type=int, default=2048,
                        help="Packed sequence length")
    parser.add_argument("--val-ratio", type=float, default=0.05,
                        help="Fraction of data for validation")
    parser.add_argument("--output-dir", default=str(OUT),
                        help="Output directory")
    args = parser.parse_args()

    random.seed(42)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  V7 DATA PROCESSING")
    print(f"{'='*70}")
    print(f"Max per source: {args.max_per_source}")
    print(f"Seq len: {args.seq_len}")
    print(f"Output: {out_dir}")

    # ── Discover sources ──
    sources = find_jsonl_sources()
    print(f"\nDiscovered {len(sources)} data sources:")
    for name, files in sorted(sources.items()):
        print(f"  {name}: {len(files)} files")

    # ── Load examples from each source ──
    print(f"\nLoading examples (max {args.max_per_source} per source)...")
    all_examples = []
    source_stats = {}
    for name, files in sorted(sources.items()):
        exs = load_source(name, files, args.max_per_source)
        source_stats[name] = len(exs)
        all_examples.extend(exs)
        print(f"  {name}: {len(exs)} examples")
    print(f"\nTotal examples: {len(all_examples)}")

    if not all_examples:
        print("No examples loaded. Exiting.")
        return

    # ── Load tokenizer (fast = Rust-backed, has tokenizer.json) ──
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "research" / "checkpoints" / "lfm25_tokenizer"),
        use_fast=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    is_fast = getattr(tokenizer, "is_fast", False)
    print(f"Tokenizer: vocab={tokenizer.vocab_size}, pad_id={pad_id}, fast={is_fast}")
    if not is_fast:
        print("  WARNING: slow tokenizer loaded — tokenization will be CPU-bound")

    # ── Render + tokenize (batched fast tokenizer → CUDA tensor ops) ──
    print(f"\nRendering {len(all_examples)} examples...")
    texts = []
    for ex in all_examples:
        text = render_example(ex)
        if text and len(text) >= 20:
            texts.append(text)
    print(f"Rendered {len(texts)} valid texts (dropped {len(all_examples) - len(texts)})")

    # Phase 1: Batch tokenize on CPU (Rust parallel if fast tokenizer)
    # The tokenizer releases the GIL during batch encoding, so Python threads
    # can overlap I/O with tokenization.
    import time as _time
    print(f"Batch tokenizing {len(texts)} texts...")
    BATCH = 1000
    all_tokens = []
    t0 = _time.perf_counter()
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i+BATCH]
        enc = tokenizer(batch, add_special_tokens=False, return_tensors=None,
                        padding=False, truncation=False)
        for ids in enc["input_ids"]:
            all_tokens.extend(ids)
        if (i // BATCH) % 2 == 0 or i + BATCH >= len(texts):
            elapsed = _time.perf_counter() - t0
            rate = len(all_tokens) / max(elapsed, 0.001) / 1e6
            print(f"  {i+len(batch)}/{len(texts)} texts, "
                  f"{len(all_tokens)/1e6:.2f}M tokens, "
                  f"{rate:.1f}M tok/s", flush=True)
    tok_elapsed = _time.perf_counter() - t0
    print(f"Tokenized: {len(texts)} texts, {len(all_tokens)} tokens "
          f"({len(all_tokens)/1e6:.2f}M) in {tok_elapsed:.1f}s")

    # Phase 2: CUDA tensor operations (packing, shuffle, split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nMoving {len(all_tokens)} tokens to {device} for packing...")
    t1 = _time.perf_counter()

    # Transfer to GPU as a single tensor (fast via pinned memory)
    token_tensor = torch.tensor(all_tokens, dtype=torch.int32, device=device)
    print(f"  GPU tensor: {token_tensor.shape}, {token_tensor.numel()*4/1e6:.1f} MB")

    # Pack into sequences on GPU (zero-copy view + slice)
    seq_len = args.seq_len
    n_seqs = token_tensor.numel() // seq_len
    if n_seqs == 0:
        # Pad on GPU
        pad_tensor = torch.full((seq_len - token_tensor.numel(),), pad_id,
                                dtype=torch.int32, device=device)
        token_tensor = torch.cat([token_tensor, pad_tensor])
        n_seqs = 1
    packed = token_tensor[:n_seqs * seq_len].view(n_seqs, seq_len)
    print(f"  Packed: {packed.shape} on {device}")

    # Shuffle on GPU (torch.randperm on CUDA = fast)
    perm = torch.randperm(n_seqs, device=device)
    packed = packed[perm]
    print(f"  Shuffled on GPU")

    # Train/val split on GPU
    n_val = max(1, int(n_seqs * args.val_ratio))
    n_train = n_seqs - n_val
    train_seqs = packed[:n_train]
    val_seqs = packed[n_train:]
    print(f"  Split on GPU: train={train_seqs.shape}, val={val_seqs.shape}")
    print(f"  Train: {n_train} seqs ({n_train * seq_len / 1e6:.2f}M tokens)")
    print(f"  Val:   {n_val} seqs ({n_val * seq_len / 1e6:.2f}M tokens)")

    # Transfer back to CPU for saving (contiguous + .cpu() = async + sync)
    train_cpu = train_seqs.contiguous().cpu()
    val_cpu = val_seqs.contiguous().cpu()
    cuda_elapsed = _time.perf_counter() - t1
    print(f"  CUDA ops + D2H transfer: {cuda_elapsed:.2f}s")
    if torch.cuda.is_available():
        print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")

    # ── Save (from CPU tensors transferred from GPU) ──
    train_path = out_dir / "train.bin"
    val_path = out_dir / "val.bin"
    manifest_path = out_dir / "manifest.json"

    train_cpu.numpy().tofile(str(train_path))
    val_cpu.numpy().tofile(str(val_path))

    manifest = {
        "seq_len": seq_len,
        "n_train": n_train,
        "n_val": n_val,
        "total_tokens": len(all_tokens),
        "n_texts_processed": len(texts),
        "n_examples_loaded": len(all_examples),
        "source_stats": source_stats,
        "tokenizer_vocab": tokenizer.vocab_size,
        "pad_id": pad_id,
        "tokenization_time_s": round(tok_elapsed, 2),
        "cuda_ops_time_s": round(cuda_elapsed, 2),
        "device": str(device),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    train_mb = train_path.stat().st_size / 1e6
    val_mb = val_path.stat().st_size / 1e6
    print(f"\nSaved:")
    print(f"  {train_path} ({train_mb:.1f} MB)")
    print(f"  {val_path} ({val_mb:.1f} MB)")
    print(f"  {manifest_path}")
    print(f"\n{'='*70}")
    print(f"  DATA PROCESSING COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
