"""Supervised fine-tuning (SFT) for ForgeLM V10-1.2B.

Trains the model on teacher-generated data (function-calling trajectories,
chain-of-thought, code) using completion-only cross-entropy loss — only
assistant tokens contribute to the loss, prompt/tool/user tokens are masked.

Data format:
  - tool_use_fc: JSONL with {"messages": [...]} — multi-turn OpenAI-style
    function-calling conversations (user / assistant w/ tool_calls / tool /
    final assistant). Rendered into Qwen chat format with tool-call markers.
  - short_cot / code / tool_use: JSONL with {"prompt": ..., "response": ...}
    — single-turn Q&A. Rendered as user→assistant chat.

The ForgeLM V2 (Jamba) tokenizer uses Qwen-style markers
(<|im_start|>/<|im_end|>, ids 518/519) but ships without a chat_template,
so we apply the Qwen chat format manually:

    <|im_start|>user\n{content}<|im_end|>\n
    <|im_start|>assistant\n{content}<|im_end|>\n
    <|im_start|>tool\n{content}<|im_end|>\n

For assistant tool_calls messages, we serialize them using the tokenizer's
native tool-call special tokens (ids 531/532):

    <|im_start|>assistant
    {start_token}
    {"name": "...", "arguments": {...}}
    {end_token}
    <|im_end|>

Usage:
    python -m research.training.runners.sft_train \\
        --data research/data/finetune/tool_use_fc_70.jsonl \\
        --data research/data/finetune/short_cot_70.jsonl \\
        --data research/data/finetune/code_70.jsonl \\
        --config forgelm_v2 \\
        --checkpoint research/checkpoints/ForgeLM_V2.safetensors \\
        --save research/checkpoints/ForgeLM_V2.sft.safetensors \\
        --max-steps 500 --lr 5e-5 --batch-size 2 --seq-len 1024
"""
import argparse
import json
import logging
import math
import os
import random
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger(__name__)

import torch
import torch.nn.functional as F

from forge.checkpoint_io import (
    cleanup_step_checkpoints,
    emergency_save,
    save_training_checkpoint,
    step_checkpoint_path,
)
from forge.config import get_config
from forge.model_loader import ModelLoader
from forge.runtime.task_logger import task_scope
from forge.training.data.efficient_pipeline import (
    AsyncPrefetcher,
    DiskTokenCache,
    PackedSequenceDataset,
)
from forge.training.training_utils import (
    add_safeguard_args,
    configure_optimizer,
    get_lr,
    has_nan_params,
    init_ema,
    load_anchor_cached,
    oom_guard,
    patch_triton_cache_for_windows,
    ram_exceeded,
    ram_usage,
    update_ema,
    vram_exceeded,
    write_heartbeat,
    write_status_json,
)
from research.tokenizer_cache import get_tokenizer

# Tool call markers — the tokenizer's native special tokens (ids 531/532).
# The runtime parser (qwen_parse_tool_calls) expects JSON inside these tags:
#   {"name": "func", "arguments": {"key": "val"}}
# Canonical Jamba markers (chat_template.TOOL_CALL_START/END) — the earlier
# <|tool_call_start|>/<|tool_call_end|> pair was LFM2.5 legacy and does not
# exist in the V2 tokenizer (rendered as literal text → off-format training).
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"

# Canonical Jamba ChatML message markers (ids 518/519). The previous
# <|startofsegment|>/<|endofsegment|> markers are NOT in the V2 tokenizer —
# every SFT example was rendered in a format inference never sees.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"


# ── Chat-format rendering ────────────────────────────────────────────────────

def _render_tool_call(tc: dict) -> str:
    """Serialize one tool call as JSON inside native special token tags.

    Format matches the runtime parser (qwen_parse_tool_calls):
      {"name": "func_name", "arguments": {"key": "val"}}
    """
    name = tc.get("name", "")
    args = tc.get("args", tc.get("arguments", {}))
    if not isinstance(args, dict):
        args = {}
    call_obj = {"name": name, "arguments": args}
    call_json = json.dumps(call_obj, ensure_ascii=False)
    return _TOOL_CALL_START + "\n" + call_json + "\n" + _TOOL_CALL_END


def _chat_render(messages: list[dict], add_generation_prompt: bool) -> str:
    """Render via the canonical Jamba ChatML port (single source of truth —
    forge/self_play/discovery/chat_template.py, a faithful port of the
    tokenizer's built-in Jinja template). thinking=False: SFT data carries
    direct answers; eval + the self-play solve path also render direct.
    bos=False: the tokenizer adds BOS itself at encode time."""
    from forge.self_play.discovery.chat_template import apply_chat_template
    return apply_chat_template(
        messages, add_generation_prompt=add_generation_prompt,
        bos=False, thinking=False)


def render_messages(messages: list[dict]) -> tuple[str, int]:
    """Render a multi-turn message list into Jamba chat format.

    Uses the canonical ChatML render (<|im_start|>/<|im_end|> markers,
    <tool_response> results inside a user turn) — the format the ForgeLM V2
    checkpoint was actually trained on and that eval/inference use.

    Returns (text, completion_start_char). completion_start_char is the
    character offset where the FIRST assistant response begins — everything
    before it is the prompt (loss-masked), everything from there on is the
    completion (loss-bearing).

    For multi-turn conversations (tool_use_fc), the entire conversation after
    the first user message is the "completion" — the model learns to produce
    the full assistant + tool + final-assistant sequence given the user prompt.
    """
    first_asst = next(
        (i for i, m in enumerate(messages) if m.get("role") == "assistant"),
        len(messages))
    # Prefix = conversation up to the first assistant turn + generation prompt
    # ("<|im_start|>assistant\n") — a strict prefix of the full render, so its
    # length is exactly the completion start.
    prefix = _chat_render(messages[:first_asst], add_generation_prompt=True)
    # No assistant turn → render ends in the generation prompt (empty
    # completion), matching the original contract.
    full = _chat_render(
        messages, add_generation_prompt=(first_asst == len(messages)))
    return full, len(prefix)


def render_single_turn(prompt: str, response: str) -> tuple[str, int]:
    """Render a single-turn Q&A into Jamba chat format.

    Returns (text, completion_start_char).
    """
    prompt_text = _chat_render(
        [{"role": "user", "content": prompt}], add_generation_prompt=True)
    full = prompt_text + response + f"{_IM_END}\n"
    return full, len(prompt_text)


def _assistant_body(m: dict) -> str:
    """Assistant turn body per the canonical template: content, then one
    <tool_call> block per call (newline-separated)."""
    content = m.get("content", "") or ""
    calls = [_render_tool_call(tc) for tc in (m.get("tool_calls") or [])]
    return "\n".join([content] + calls) if content else "\n".join(calls)


def split_multi_turn(messages: list[dict]) -> list[tuple[str, str]]:
    """Split a multi-turn conversation into per-turn training examples.

    Each assistant turn becomes a separate (prompt, completion) pair where:
      - prompt = all messages up to (but not including) this assistant turn,
        rendered in Jamba ChatML + a generation prompt
        ("<|im_start|>assistant\n")
      - completion = the assistant turn body + "<|im_end|>\n"

    This teaches the model to STOP after each assistant turn (emit tool calls
    and stop, or emit a final answer and stop) — critical for agent loops
    where the orchestrator intercepts tool calls, executes them, and feeds
    results back.

    For a conversation:
      user → assistant(tool_calls) → tool → tool → assistant(final)
    We produce 2 examples:
      1. prompt=user, completion=tool_calls+<|im_end|>
      2. prompt=user+tool_calls+tool+tool, completion=final+<|im_end|>
    """
    examples = []
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        # Render the prefix through the canonical template so tool messages
        # group into user turns exactly as the model sees them at inference.
        prompt_text = _chat_render(messages[:i], add_generation_prompt=True)
        completion_text = _assistant_body(m) + f"{_IM_END}\n"
        examples.append((prompt_text, completion_text))
    return examples


# ── Dataset loading + tokenization ───────────────────────────────────────────

def load_examples(paths: list[str], strict_data: bool = False) -> list[dict]:
    """Load and deduplicate examples from one or more JSONL files.

    Each line should be a JSON object. For tool_use_fc, expects {"messages": [...]}.
    For short_cot/code/tool_use, expects {"prompt": ..., "response": ...}.

    Args:
        strict_data: If True, raise on the first malformed row instead of
            skipping it. Useful for catching data corruption during dev.
    """
    examples = []
    seen = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            logger.warning("Training data file not found, skipping: %s", path)
            continue
        n_loaded = 0
        n_skipped = 0
        skip_by_class: dict[str, int] = defaultdict(int)
        with open(p, encoding="utf-8") as f:
            content = f.read()
        for line_no, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as exc:
                n_skipped += 1
                skip_by_class[type(exc).__name__] += 1
                preview = line[:120]
                logger.warning(
                    "JSON parse error in %s:%d [%s]: %s | preview: %r",
                    path, line_no, type(exc).__name__, exc, preview,
                )
                if strict_data:
                    raise
                continue
            # Normalize: either messages list or prompt/response.
            if "messages" in obj and isinstance(obj["messages"], list):
                key = json.dumps(obj["messages"], ensure_ascii=False, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                ex = {"type": "multi_turn", "messages": obj["messages"]}
                # Carry reward for reward-weighted SFT (if present)
                if "reward" in obj:
                    ex["reward"] = float(obj["reward"])
                examples.append(ex)
                n_loaded += 1
            elif "prompt" in obj and ("response" in obj or "solution" in obj):
                key = obj["prompt"][:200]
                if key in seen:
                    continue
                seen.add(key)
                resp = obj.get("response", obj.get("solution", ""))
                examples.append({"type": "single_turn",
                                 "prompt": obj["prompt"],
                                 "response": resp})
                n_loaded += 1
            else:
                n_skipped += 1
                skip_by_class["SchemaError"] += 1
                logger.warning(
                    "Schema mismatch in %s:%d: missing 'messages' or "
                    "'prompt'/'response' keys | preview: %r",
                    path, line_no, line[:120],
                )
                if strict_data:
                    raise ValueError(
                        f"Schema mismatch in {path}:{line_no}: row has "
                        f"neither 'messages' nor 'prompt'/'response'"
                    )
        if n_skipped:
            class_summary = ", ".join(
                f"{cls}={n}" for cls, n in sorted(skip_by_class.items())
            )
            logger.warning(
                "  %s: %d loaded, %d skipped (%s)", path, n_loaded,
                n_skipped, class_summary,
            )
        else:
            logger.info("  %s: %d loaded, 0 skipped", path, n_loaded)
    return examples


def load_examples_parquet(
    paths: list[str], strict_data: bool = False
) -> tuple[list[dict], list[dict] | None]:
    """Load examples from one or more Parquet files.

    Returns ``(examples, pre_tokenized)`` where:
      - If the parquet contains pre-tokenized columns (``input_ids`` +
        ``labels``), ``examples`` is empty and ``pre_tokenized`` is a list of
        ``{"input_ids": [...], "labels": [...]}`` dicts ready for training.
      - If the parquet contains raw data (``messages`` or ``prompt``/
        ``response``), ``examples`` is a list of normalised dicts (same format
        as :func:`load_examples`) and ``pre_tokenized`` is ``None``.

    Args:
        strict_data: If True, raise on the first malformed row instead of
            skipping it.
    """
    from forge.training.data.parquet_dataset import ParquetDataset

    examples: list[dict] = []
    seen: set = set()
    pre_tokenized: list[dict] | None = None

    for path in paths:
        p = Path(path)
        if not p.exists():
            logger.warning("Training data file not found, skipping: %s", path)
            continue
        ds = ParquetDataset(str(p))
        col_names = set(ds.schema.names)

        # ── Pre-tokenized path: input_ids + labels columns ──
        if "input_ids" in col_names and "labels" in col_names:
            if pre_tokenized is None:
                pre_tokenized = []
            n_loaded = 0
            for i in range(len(ds)):
                row = ds[i]
                ids = row.get("input_ids")
                labs = row.get("labels")
                if ids is None or labs is None:
                    logger.warning(
                        "Null input_ids/labels in %s row %d, skipping", path, i
                    )
                    if strict_data:
                        raise ValueError(
                            f"Null input_ids/labels in {path} row {i}"
                        )
                    continue
                pre_tokenized.append({
                    "input_ids": list(ids),
                    "labels": list(labs),
                    "n_comp": sum(1 for x in labs if x != -100),
                })
                n_loaded += 1
            logger.info("  %s: %d pre-tokenized rows loaded", path, n_loaded)
            continue

        # ── Raw data path: messages or prompt/response ──
        n_loaded = 0
        n_skipped = 0
        for i in range(len(ds)):
            row = ds[i]
            if "messages" in row and isinstance(row["messages"], list):
                key = json.dumps(row["messages"], ensure_ascii=False, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                examples.append({"type": "multi_turn", "messages": row["messages"]})
                n_loaded += 1
            elif "prompt" in row and ("response" in row or "solution" in row):
                key = str(row["prompt"])[:200]
                if key in seen:
                    continue
                seen.add(key)
                resp = row.get("response", row.get("solution", ""))
                examples.append({"type": "single_turn",
                                 "prompt": row["prompt"],
                                 "response": resp})
                n_loaded += 1
            else:
                n_skipped += 1
                logger.warning(
                    "Schema mismatch in %s row %d: missing 'messages' or "
                    "'prompt'/'response' keys", path, i,
                )
                if strict_data:
                    raise ValueError(
                        f"Schema mismatch in {path} row {i}: row has "
                        f"neither 'messages' nor 'prompt'/'response'"
                    )
        if n_skipped:
            logger.warning("  %s: %d loaded, %d skipped", path, n_loaded, n_skipped)
        else:
            logger.info("  %s: %d loaded, 0 skipped", path, n_loaded)

    return examples, pre_tokenized


# Tokenization cache — avoids re-tokenizing repeated prompts (2-3x speedup)
# Uses a tiered cache: in-memory LRU + disk-backed cache (DiskTokenCache).
# The disk cache persists across runs, eliminating re-tokenization on restart.
#
# NOTE: This cache is distinct from research/tokenizer_cache.py.
#   - tokenizer_cache.py caches the *tokenizer object* (singleton, lru_cache
#     maxsize=8) to avoid the 4.3s AutoTokenizer load overhead across call sites.
#   - This cache (_tokenize_cache) caches *tokenization results* (text→token IDs)
#     to avoid re-encoding repeated training texts, with disk persistence.
_tokenize_cache: OrderedDict[str, list[int]] = OrderedDict()
_CACHE_MAX = 10000  # max 10K entries in memory (~50MB typical)
_disk_tok_cache: DiskTokenCache | None = None

# ── Parallel tokenization globals (set in main(), used by _parallel_tok_worker) ──
_parallel_tokenizer = None
_parallel_seq_len = 1024


def _parallel_tok_worker(ex: dict) -> list[dict]:
    """Module-level worker for multiprocessing.Pool (must be picklable).

    Uses fork-inherited globals (_parallel_tokenizer, _parallel_seq_len)
    set by the parent process before creating the Pool.
    """
    return tokenize_example(ex, _parallel_tokenizer, _parallel_seq_len)


def _get_disk_tok_cache() -> DiskTokenCache:
    """Get the global disk tokenization cache (lazy init)."""
    global _disk_tok_cache
    if _disk_tok_cache is None:
        _disk_tok_cache = DiskTokenCache()
    return _disk_tok_cache


def _tokenize_cached(tokenizer, text: str, max_seq_len: int) -> list[int] | None:
    """Tokenize text with tiered caching (memory LRU + disk). Returns None on error.

    Cache hierarchy:
      1. In-memory LRU (fastest, bounded to _CACHE_MAX entries)
      2. Disk-backed cache (persists across runs, memory-mapped reads)
      3. Tokenizer (slowest, result cached in both layers)

    When disk cache is disabled (None), skip the LRU overhead too —
    direct tokenization is 70x faster than the LRU wrapper for bulk
    tokenization of unique texts (the LRU dict ops dominate).
    """
    # Fast path: no disk cache → direct tokenization (70x faster for bulk)
    if _disk_tok_cache is None:
        try:
            enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
            ids = enc["input_ids"] if isinstance(enc, dict) else enc
            if not isinstance(ids, list):
                ids = list(ids)
            return ids
        except Exception:
            return None

    # Layer 1: in-memory LRU
    if text in _tokenize_cache:
        _tokenize_cache.move_to_end(text)
        return _tokenize_cache[text]

    # Layer 2: disk cache
    disk = _get_disk_tok_cache()
    cached = disk.get(text, getattr(tokenizer, "_tokenizer_hash", "default"))
    if cached is not None:
        # Promote to in-memory cache
        if len(text) < 10000:
            while len(_tokenize_cache) >= _CACHE_MAX:
                _tokenize_cache.popitem(last=False)
            _tokenize_cache[text] = cached
        return cached

    # Layer 3: tokenize and cache in both layers
    try:
        enc = tokenizer(text, add_special_tokens=False, return_tensors=None)
        ids = enc["input_ids"] if isinstance(enc, dict) else enc
        if not isinstance(ids, list):
            ids = list(ids)
        if len(text) < 50000:
            # In-memory cache
            if len(text) < 10000:
                while len(_tokenize_cache) >= _CACHE_MAX:
                    _tokenize_cache.popitem(last=False)
                _tokenize_cache[text] = ids
            # Disk cache (persists across runs)
            disk.put(text, ids, getattr(tokenizer, "_tokenizer_hash", "default"))
        return ids
    except Exception:
        return None


def tokenize_example(ex: dict, tokenizer, max_seq_len: int) -> list[dict]:
    """Tokenize an example into one or more (input_ids, labels) pairs with
    completion-only masking.

    For single_turn: returns one example.
    For multi_turn: splits into per-assistant-turn examples (see split_multi_turn),
    each ending at <|im_end|> so the model learns to STOP after each turn.

    Returns a list (possibly empty if all splits are too long/short).
    """
    if ex["type"] == "multi_turn":
        pairs = split_multi_turn(ex["messages"])
    else:
        full_text, comp_start = render_single_turn(ex["prompt"], ex["response"])
        pairs = [(full_text[:comp_start], full_text[comp_start:])]

    results = []
    for prompt_text, completion_text in pairs:
        full_text = prompt_text + completion_text
        # Tokenize with cache — repeated prompts skip re-tokenization (2-3x faster)
        input_ids = _tokenize_cached(tokenizer, full_text, max_seq_len)
        if input_ids is None:
            continue
        if len(input_ids) < 4 or len(input_ids) > max_seq_len:
            continue

        # Find completion start in token space (also cached).
        prompt_ids = _tokenize_cached(tokenizer, prompt_text, max_seq_len)
        if prompt_ids is None:
            continue
        comp_start_tok = len(prompt_ids)

        # Build labels: -100 for prompt, real id for completion.
        labels = [-100] * comp_start_tok + input_ids[comp_start_tok:]
        while len(labels) < len(input_ids):
            labels.append(-100)

        results.append({"input_ids": input_ids, "labels": labels,
                        "n_comp": len(input_ids) - comp_start_tok,
                        "reward": ex.get("reward", 1.0)})
    return results


def collate_batch(batch: list[dict], pad_id: int, device: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Left-pad a batch of variable-length examples.

    Returns (input_ids, labels, attention_mask, reward_weights) all [B, T] or [B].
    Left-padding keeps the completion at the end so causal LM predicts correctly.
    Uses pin_memory when transferring to GPU for faster H2D copy.

    R&D round 25: when device="cpu" and CUDA is available, tensors are
    pinned anyway — this supports the transfer-on-consume prefetcher path
    where the background thread produces pinned CPU tensors and the
    consumer does the async H2D transfer.
    """
    max_len = max(len(ex["input_ids"]) for ex in batch)
    b = len(batch)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.long)
    labels = torch.full((b, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((b, max_len), dtype=torch.long)
    reward_weights = torch.ones(b, dtype=torch.float32)
    for i, ex in enumerate(batch):
        ids = ex["input_ids"]
        labs = ex["labels"]
        n = len(ids)
        offset = max_len - n  # left-pad
        # R&D round 25: avoid torch.tensor() copy when input is already a tensor
        if isinstance(ids, torch.Tensor):
            input_ids[i, offset:] = ids
            labels[i, offset:] = labs
        else:
            input_ids[i, offset:] = torch.tensor(ids, dtype=torch.long)
            labels[i, offset:] = torch.tensor(labs, dtype=torch.long)
        attn_mask[i, offset:] = 1
        reward_weights[i] = ex.get("reward", 1.0)
    # Use pin_memory + non_blocking for faster CPU→GPU transfer.
    # R&D round 25: pin even when device="cpu" if CUDA is available —
    # supports the transfer-on-consume prefetcher path.
    use_pin = torch.cuda.is_available()
    if use_pin:
        input_ids = input_ids.pin_memory()
        labels = labels.pin_memory()
        attn_mask = attn_mask.pin_memory()
        reward_weights = reward_weights.pin_memory()
    return (input_ids.to(device, non_blocking=use_pin),
            labels.to(device, non_blocking=use_pin),
            attn_mask.to(device, non_blocking=use_pin),
            reward_weights.to(device, non_blocking=use_pin))


# ── Training loop ────────────────────────────────────────────────────────────

def compute_loss(model, input_ids, labels, attn_mask,
                 entropy_alpha: float = 0.0,
                 sample_weights: torch.Tensor | None = None,
                 sigreg_weight: float = 0.0,
                 sigreg_threshold: float = 1.0,
                 is_hf_model: bool = False) -> torch.Tensor:
    """Completion-only cross-entropy loss with optional anti-regression techniques.

    The model's forward(targets=...) computes mean CE over ALL non-ignored
    positions. We pass labels as targets with -100 for masked positions, so
    only completion tokens contribute.

    When is_hf_model=True, uses the manual logits path (HF models don't
    support targets= kwarg in forward).

    Memory-optimized fast paths (no full logits materialization):
      - entropy_alpha=0, no sample_weights → chunked_linear_cross_entropy
      - entropy_alpha>0, no sample_weights  → chunked_entropy_weighted_ce
      Both avoid materializing the [B, T, V] logits tensor — 20x+ faster on
      large vocabularies (vocab=65536).

    Anti-regression extensions (require the manual logits path when
    sample_weights is provided):
      - Token Entropy Weighting (WeFT/VCORE 2025): when entropy_alpha > 0,
        per-token CE is scaled by (1 + alpha * normalized_entropy), giving
        MORE weight to high-entropy (reasoning/uncertain) tokens and LESS
        to low-entropy (boilerplate) tokens.
      - Easy Sample Upweighting (ICML 2025): when sample_weights is provided
        (pre-computed from base model loss), per-example loss is scaled by
        weight = 1/(1+base_loss), so easy samples get higher weight.

    SIGReg (LeWM/LeCun et al.): when sigreg_weight > 0, collects per-layer
    hidden states and adds a spectral norm regularization penalty to prevent
    hidden-state collapse.
    """
    from forge.training.losses.sigreg import SIGRegLoss
    _sigreg = SIGRegLoss(threshold=sigreg_threshold) if sigreg_weight > 0.0 else None
    _need_hidden = _sigreg is not None

    # ── HuggingFace model path: manual logits + CE (no targets= kwarg) ──
    if is_hf_model:
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=input_ids.is_cuda,
        ):
            out = model(input_ids, attention_mask=attn_mask)
        logits = out.logits if hasattr(out, "logits") else out[0]
        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1), ignore_index=-100)
        if not torch.isfinite(ce_loss):
            raise RuntimeError(f"Non-finite training loss ({ce_loss.item()}).")
        return ce_loss

    # ── Fast path: chunked CE (no logits materialization) ──
    # Used when no per-example sample weighting is needed. The model's
    # forward(targets=labels) computes the loss internally via
    # chunked_linear_cross_entropy or chunked_entropy_weighted_ce,
    # never materializing [B, T, V] logits.
    if sample_weights is None:
        # Shift labels for causal LM: hidden[i] predicts token i+1.
        # Pad with -100 to match hidden [B, T] (last position has no target).
        shift_labels = F.pad(labels[:, 1:], (0, 1), value=-100)  # [B, T]
        # Set entropy_alpha on the model config so the forward uses the
        # correct chunked CE variant.
        if hasattr(model, 'config') and hasattr(model.config, 'entropy_alpha'):
            model.config.entropy_alpha = entropy_alpha
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=input_ids.is_cuda,
        ):
            out = model(input_ids, attention_mask=attn_mask, targets=shift_labels,
                        return_hidden_states=_need_hidden)
        # Unpack output — may include hidden_states_list when SIGReg is active.
        hidden_states_list = None
        if _need_hidden and isinstance(out, tuple):
            # (logits, loss, hidden_states_list)
            *rest, hidden_states_list = out
            loss = rest[1] if len(rest) > 1 and rest[1] is not None else rest[0]
        elif isinstance(out, tuple):
            loss = out[1] if out[1] is not None else out[0]
        else:
            loss = out
        if loss is None:
            # Model didn't use chunked CE (use_chunked_ce=False); fall through
            # to the manual path below.
            pass
        else:
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite training loss ({loss.item()}). "
                    f"Check for NaN in inputs, exploding gradients, or lr too high."
                )
            if _sigreg is not None and hidden_states_list is not None:
                sigreg_loss = _sigreg(hidden_states_list, threshold=sigreg_threshold)
                loss = loss + sigreg_weight * sigreg_loss
            return loss

    # ── Manual path: full logits for sample weighting ──
    # Only used when sample_weights is provided (requires per-example control).
    # Forward — compute logits manually for per-position control.
    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16,
        enabled=input_ids.is_cuda,
    ):
        out = model(input_ids, attention_mask=attn_mask,
                    return_hidden_states=_need_hidden)
    # Unpack — may include hidden_states_list when SIGReg is active.
    hidden_states_list = None
    if _need_hidden and isinstance(out, tuple):
        *rest_m, hidden_states_list = out
        logits = rest_m[0]
    else:
        logits = out[0] if isinstance(out, tuple) else out  # [B, T, V]
    if logits is None:
        raise RuntimeError("model returned None logits; pass targets manually")
    # Shift: predict token t+1 from position t.
    shift_logits = logits[:, :-1, :].contiguous()  # [B, T-1, V]
    shift_labels = labels[:, 1:].contiguous()       # [B, T-1]

    # Per-token CE (reduction="none") for weighting control.
    ce_per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.size())  # [B, T-1]

    # Mask for non-ignored (completion) tokens.
    mask = (shift_labels != -100).float()  # [B, T-1]

    if entropy_alpha > 0.0:
        # ── Token Entropy Weighting (WeFT/VCORE 2025) ──
        # High-entropy tokens (uncertain/reasoning) get MORE weight;
        # low-entropy tokens (boilerplate) get LESS.
        with torch.no_grad():
            probs = F.softmax(shift_logits.float(), dim=-1)  # [B, T-1, V]
            # Per-token entropy: H = -sum(p * log(p))
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # [B, T-1]
            # Normalize to [0, 2] range. Max entropy = ln(vocab_size).
            # For vocab=65536, ln(65536) ≈ 11.09.
            vocab_size = float(shift_logits.size(-1))
            max_entropy = torch.log(torch.tensor(vocab_size,
                                                 device=shift_logits.device,
                                                 dtype=torch.float32))
            normalized_entropy = (entropy / max_entropy.clamp(min=1e-8)) * 2.0  # [0, 2]
            # Weight: w = 1.0 + alpha * normalized_entropy
            entropy_weights = 1.0 + entropy_alpha * normalized_entropy  # [B, T-1]
        # Apply entropy weights to per-token CE.
        ce_per_token = ce_per_token * entropy_weights

    # Per-example loss: mean over non-ignored tokens.
    per_example_loss = (ce_per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)  # [B]

    if sample_weights is not None:
        # ── Easy Sample Upweighting (ICML 2025) ──
        # weight = 1/(1+base_loss); easy samples (low base loss) get higher weight.
        per_example_loss = per_example_loss * sample_weights

    loss = per_example_loss.mean()
    if not torch.isfinite(loss):
        raise RuntimeError(
            f"Non-finite training loss ({loss.item()}). "
            f"Check for NaN in inputs, exploding gradients, or lr too high."
        )
    if _sigreg is not None and hidden_states_list is not None:
        sigreg_loss = _sigreg(hidden_states_list, threshold=sigreg_threshold)
        loss = loss + sigreg_weight * sigreg_loss
    return loss


def compute_l2_sp_loss(model, anchor_named_params: dict, base_lambda: float) -> torch.Tensor:
    """L2-SP Anchor Regularization (NeurIPS 2024).

    Penalizes L2 distance between trainable parameters and anchor (pre-SFT)
    weights to prevent catastrophic forgetting / regression on base capabilities.

    Layer-wise lambda: lower layers (0-5) get 10x higher lambda than upper
    layers (6-15) to preserve foundational representations (embedding, early
    conv/attention) while allowing higher layers to adapt to the new task.

    Returns the weighted L2-SP loss (already scaled by per-layer lambda).
    The caller adds this to CE loss: total = ce_loss + l2_sp_loss.
    """
    import re
    l2_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name not in anchor_named_params:
            continue
        anchor_p = anchor_named_params[name]
        # Determine layer-wise lambda from parameter name.
        lam = base_lambda
        m = re.search(r'layers?\.?(\d+)', name)
        if m:
            layer_idx = int(m.group(1))
            if layer_idx <= 5:
                lam = base_lambda * 10.0  # Lower layers: 10x higher penalty
            # else: upper layers use base_lambda
        l2_loss = l2_loss + lam * (p.float() - anchor_p.float()).pow(2).sum()
    return l2_loss


def compute_sample_weights(model, dataset, pad_id, device, batch_size=1):
    """Pre-compute per-example sample weights from base model loss (ICML 2025).

    Runs one forward pass per example (in eval mode, no grad) to get the base
    model's completion-only CE loss. Then weight = 1/(1+base_loss) — easy
    samples (low loss) get higher weight, hard samples get lower weight.

    Returns a tensor of shape [len(dataset)] with per-example weights.
    """
    weights = torch.zeros(len(dataset), dtype=torch.float32, device=device)
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(dataset), batch_size):
            batch = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
            input_ids, labels, attn_mask, _rw = collate_batch(batch, pad_id, device)
            loss = compute_loss(model, input_ids, labels, attn_mask)
            # Per-example loss is the mean CE for this batch (batch_size=1 typically).
            w = 1.0 / (1.0 + loss.item())
            for j in range(len(batch)):
                weights[i + j] = w
    model.train()
    # Normalize weights so mean = 1.0 (keeps loss scale stable).
    weights = weights / weights.mean().clamp(min=1e-8)
    return weights


def main():
    # Opt-in runtime configuration (import of `forge` is side-effect-free).
    from forge.runtime.configure import configure
    configure()

    p = argparse.ArgumentParser(description="SFT for ForgeLM V10-1.2B")
    p.add_argument("--data", nargs="+", required=True,
                   help="Training data file(s) (JSONL or Parquet)")
    p.add_argument("--data-format", default="jsonl", choices=["jsonl", "parquet"],
                   help="Data format: 'jsonl' (default) or 'parquet'. "
                        "When 'parquet', files are read via ParquetDataset "
                        "(memory-mapped, ZSTD-compressed).")
    # ── Data pipeline: disk cache, packed sequences, async prefetch ──
    p.add_argument("--disk-cache", action=argparse.BooleanOptionalAction, default=True,
                   help="Use disk-backed tokenization cache (persists across runs, "
                        "eliminates re-tokenization on restart). Default: True.")
    p.add_argument("--tokenize-workers", type=int, default=0,
                   help="Number of parallel workers for tokenization (0=auto, "
                        "uses min(cpu_count, 32). On Vast.ai with 128 cores, "
                        "this gives ~30x speedup for 150K+ examples).")
    p.add_argument("--pack-sequences", action=argparse.BooleanOptionalAction, default=True,
                   help="Pack variable-length examples into fixed-length sequences "
                        "(Llama-3 style, eliminates 30-50% padding waste). Default: True.")
    p.add_argument("--async-prefetch", action=argparse.BooleanOptionalAction, default=True,
                   help="Use async background prefetcher for data loading (2-3x throughput "
                        "on I/O-bound workloads). Default: True.")
    p.add_argument("--prefetch-count", type=int, default=4,
                   help="Number of batches to prefetch ahead (default 4).")
    p.add_argument("--strict-data", action="store_true",
                   help="Raise on the first malformed training row instead of "
                        "skipping it. Use during dev to catch data corruption "
                        "early; leave off for production runs where a few bad "
                        "rows should not abort the job.")
    p.add_argument("--config", default="forgelm_v2",
                   help="Model config name (default: forgelm_v2)")
    p.add_argument("--checkpoint", default="research/checkpoints/ForgeLM_V2.safetensors",
                   help="Base checkpoint to fine-tune from")
    p.add_argument("--config-overrides", default=None,
                   help="JSON dict of architecture overrides applied before "
                        "model construction (e.g. '{\"use_forge_hybrid\":true,"
                        "\"use_outro\":true}')")
    p.add_argument("--hf-model", default=None,
                   help="Load a HuggingFace model directly (e.g. Qwen/Qwen2.5-0.5B). "
                        "Bypasses ForgeEngine/ModelLoader — loads via transformers. "
                        "Use with --nanoquant-qat or --no-bitnet-everywhere for QAT "
                        "on non-ForgeAI models. The tokenizer is auto-loaded from "
                        "the same HF repo.")
    p.add_argument("--save", default="research/checkpoints/ForgeLM_V2.sft.safetensors",
                   help="Output checkpoint path")
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min-lr", type=float, default=5e-6)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size (NeurIPS 2025: batch size 1 is stable, "
                        "no grad_accum needed for single-GPU)")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--warmup-steps", type=int, default=20)  # warmup needed for stability (evolution 0 was synthetic-only)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--optimizer", default="muon_sf", choices=["fused", "bnb", "lion", "muon", "muon_sf", "muon_sf_plain", "flash_adamw", "flash_lion", "forge", "sf_normuon", "amuse", "mona", "cpu_offload", "badam", "fira_nlrq", "apollo", "flashoptim", "galore", "nvme_muon_4bit"])
    p.add_argument("--qk-clip-tau", type=float, default=0.0,
                   help="QK-Clip tau (Kimi K2 MuonClip): cap per-head max attention "
                        "logit after each optimizer step. 0 = disabled (default). "
                        "30 or 100 recommended when training destabilizes.")
    # ── FreeToken-inspired training enhancements (R&D round 14) ──
    p.add_argument("--freetoken", action="store_true",
                   help="Enable FreeToken-inspired training pipeline (arXiv:2608.16157): "
                        "double-buffered gradient pipeline, bandwidth-adaptive chunked "
                        "transfers, predictive offload. Only applies with --optimizer cpu_offload.")
    p.add_argument("--double-buffer", action="store_true",
                   help="Enable ping-pong grad buffers for transfer/compute overlap. "
                        "Subset of --freetoken. Only applies with --optimizer cpu_offload.")
    p.add_argument("--bandwidth-adaptive", action="store_true",
                   help="Profile PCIe bandwidth and auto-set chunk size for transfers. "
                        "Subset of --freetoken. Only applies with --optimizer cpu_offload.")
    p.add_argument("--chunk-size-mb", type=int, default=None,
                   help="Override auto-computed chunk size for chunked grad transfers. "
                        "Only applies with --bandwidth-adaptive or --freetoken.")
    p.add_argument("--elastic-grad-accum", action="store_true",
                   help="Dynamically increase grad_accum when VRAM pressure is high, "
                        "decrease when low. Prevents OOM without manual tuning.")
    p.add_argument("--grad-mixup", type=int, default=1,
                   help="Grad mixup: average gradients from N batches before optimizer step "
                        "(1=disabled, 2=two-batch, 3=three-batch). "
                        "Tested: 3-way mixup + muon_sf = 1.25x better convergence vs AdamW. "
                        "Cost: N× forward+backward per step, but fewer steps needed.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable gradient checkpointing to save VRAM (default: True, "
                        "use --no-grad-checkpoint to disable)")
    p.add_argument("--checkpoint-strategy", default="all",
                   choices=["all", "ffn", "attn", "none", "lazy", "optimal"],
                   help="Selective gradient checkpointing: 'ffn' recomputes only "
                        "FFN (biggest activation, minimal compute penalty), 'attn' "
                        "recomputes only attention, 'all' recomputes full block, "
                        "'lazy' uses LazyTrain mixed-integer scheduling, "
                        "'optimal' uses Hirschberg knapsack for optimal selection. "
                        "Only applies when --grad-checkpoint is active.")
    p.add_argument("--lazy-train", action="store_true",
                   help="Enable LazyTrain scheduler: mixed-integer checkpoint "
                        "selection + Hybrid 8-bit operator (fused clip + 8-bit state). "
                        "1.24x sustained TFLOPS improvement on limited VRAM.")
    p.add_argument("--oomb", action="store_true",
                   help="Enable OOMB chunk-recurrent training for long contexts. "
                        "Processes sequences in chunks with on-the-fly recomputation. "
                        "O(1) activation memory regardless of sequence length. "
                        "Enables 128K+ context training on 12GB GPU.")
    p.add_argument("--hybrid-clip", action="store_true",
                   help="Use Hybrid8Bit fast gradient clipping (running EMA norm, "
                        "exact every 50 steps). Avoids global norm computation when "
                        "optimizer states are offloaded to CPU.")
    p.add_argument("--curriculum", default="none",
                   choices=["none", "vanilla", "pacing", "interleaved", "warmup"],
                   help="Curriculum learning strategy: order training data easy→hard. "
                        "18-45% fewer steps to reach baseline performance. "
                        "'pacing' gradually increases difficulty, 'warmup' does easy "
                        "warmup then random, 'interleaved' mixes with increasing hard ratio.")
    p.add_argument("--augment", action="store_true",
                   help="Enable training-time data augmentation (token noise, FIM, "
                        "target offset). Regularizes against overfitting in "
                        "multi-epoch training (data-constrained regime).")
    p.add_argument("--synpro", action="store_true",
                   help="Enable SYNPRO synthetic data generation: rephrase + reformat "
                        "training data for 3.7-5.2x effective tokens. "
                        "Useful in data-bound regime.")
    p.add_argument("--norm-type", default="rmsnorm",
                   choices=["rmsnorm", "seednorm", "dyt"],
                   help="Normalization type: 'seednorm' = SeeDNorm (self-rescaled "
                        "dynamic, better zero-shot), 'dyt' = Dynamic Tanh (bounded, "
                        "compatible with Muon optimizers). Default 'rmsnorm'.")
    p.add_argument("--distill", action="store_true",
                   help="Enable knowledge distillation from a teacher model. "
                        "Uses offline top-K logits caching + chunked KL loss.")
    p.add_argument("--teacher-checkpoint", type=str, default=None,
                   help="Teacher model checkpoint for distillation.")
    p.add_argument("--distill-topk", type=int, default=50,
                   help="Top-K logits to cache from teacher (distillation).")
    p.add_argument("--distill-truncate", type=float, default=1.0,
                   help="Sequence truncation ratio for distillation (0.5 = first 50%% "
                        "of tokens, 91%% performance at 50%% FLOPs).")
    p.add_argument("--distill-prefix", action="store_true",
                   help="On-policy prefix distillation: distill only reasoning "
                        "prefixes (2-40x FLOP reduction).")
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                   help="Model dtype (bf16 recommended for 12GB GPU)")
    # NeurIPS 2025: batch size 1 is stable, no grad_accum needed for single-GPU
    # Evolution-discovered: accum_steps=5 + sync_freq=15 gives effective batch=75
    p.add_argument("--grad-accum", type=int, default=5,  # evolution: 5 (was 1), effective batch=75
                   help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--sync-freq", type=int, default=15,
                   help="Sync gradients every N steps (evolution-discovered: 15)")
    p.add_argument("--grad-compression", default="int4", choices=["none", "int4", "int8"],  # evolution: int4
                   help="Gradient compression for CPU offload (evolution: int4 best)")
    p.add_argument("--ema", action="store_true",
                   help="Enable EMA (Exponential Moving Average) shadow weights")
    p.add_argument("--ema-decay", type=float, default=0.999,
                   help="EMA decay rate (default 0.999)")
    p.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True,
                   help="Use LoRA (PEFT) instead of full fine-tuning. Faster, less VRAM. "
                        "(default: True, use --no-lora to disable)")
    p.add_argument("--lora-r", type=int, default=32, help="LoRA rank (default 32)")
    p.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha (default 64)")
    p.add_argument("--bitnet-everywhere", action=argparse.BooleanOptionalAction, default=True,
                   help="Convert ALL nn.Linear to BitNetLinear (ternary b1.58 QAT). "
                        "No NF4/bnb needed — BitNet IS the quantization (1.58 bits). "
                        "Validated on V3 1.2B: 2.39x vs AdamW, 6.32GB VRAM. "
                        "(default: True, use --no-bitnet-everywhere to disable)")
    p.add_argument("--nanoquant-qat", action="store_true", default=False,
                   help="Convert all nn.Linear to NanoQuantQATLinear (sub-1-bit "
                        "binary factorization QAT with STE). Uses SVD-based "
                        "fast init, then STE refines binary factors during "
                        "training. Achieves <1 bit/weight with QAT. "
                        "Requires --no-bitnet-everywhere. Use --nanoquant-rank "
                        "to control compression ratio.")
    p.add_argument("--nanoquant-rank", type=int, default=128,
                   help="NanoQuant factorization rank (default 128). "
                        "Higher rank = better quality, more memory. "
                        "rank=128 → ~0.22 bits/w, rank=512 → ~0.81 bits/w.")
    p.add_argument("--nanoquant-quick-init", type=int, default=1,
                   help="1=SVD fast init (default, instant), 0=ADMM init (slow)")
    p.add_argument("--manual-lora", action="store_true",
                   help="Use manual LoRA adapters (BitNet-compatible, unlike PEFT). "
                        "Works with BitNetLinear. Auto-enabled with --bitnet-everywhere.")
    p.add_argument("--qlora-nf4", action="store_true",
                   help="R32-1: NF4 QLoRA — quantize base weights to NF4 (4-bit normal float) "
                        "and train LoRA adapters on top. Works on standard nn.Linear models "
                        "(no BitNet needed). Industry-standard QLoRA path (Dettmers et al. 2023). "
                        "Use with --no-bitnet-everywhere. LoRA is auto-enabled.")
    p.add_argument("--nf4-group-size", type=int, default=64,
                   help="NF4 group size for per-group scaling (default 64)")
    p.add_argument("--forge-quant", action="store_true",
                   help="Quantize base weights with ForgeQuant (INT4 dense + "
                        "INT8 sparse outliers) before training. Faster than "
                        "NF4 and matches inference-time quantization. LoRA "
                        "adapters train on top of ForgeQuantLinear layers "
                        "(which have built-in lora_adapter support).")
    p.add_argument("--save-lora-adapter", action="store_true",
                   help="Additionally save ONLY the trained LoRA tensors to "
                        "'<save-stem>.lora.safetensors' (small file, hot-loadable "
                   "via ForgeEngine.load_lora / the GUI LoRA Manager). "
                   "The merged standalone checkpoint is still saved normally.")
    p.add_argument("--sequential-freeze", type=int, default=0,
                   help="Sequential freeze/unfreeze: train N layers at a time in phases. "
                        "0=disabled. 4=4 phases (4 layers each for 16-layer model). "
                        "Full forward pass preserves MHC/AttnRes; only gradients are scoped. "
                        "Best with 100+ steps/phase.")
    p.add_argument("--final-finetune-steps", type=int, default=0,
                   help="When using --sequential-freeze, reserve N steps at the end "
                        "to fine-tune ALL layers together. 0=disabled.")
    p.add_argument("--compile", action="store_true",
                   help="Use torch.compile on the model (experimental on Windows)")
    p.add_argument("--use-forge-engine", action=argparse.BooleanOptionalAction, default=True,
                   help="Use ForgeEngine for model loading (provides unified inference engine "
                        "with all activation features, crash recovery, KeyStack auto-detection). "
                        "The engine.model is used for training; engine is used for validation. "
                        "Default: True (use --no-use-forge-engine for legacy ModelLoader path).")
    # ── Anti-regression techniques ──
    p.add_argument("--entropy-alpha", type=float, default=0.5,
                   help="Token entropy weighting alpha (WeFT/VCORE 2025). "
                        "High-entropy tokens get more weight. 0 disables. Default 0.5")
    p.add_argument("--loss-function", default="ce",
                   choices=["ce", "focal", "label_smoothing", "lovasz", "dynamic_focal", "mixture"],
                   help="Loss function: 'ce' (standard), 'focal' (hard token focus), "
                        "'label_smoothing' (anti-overconfidence), 'lovasz' (exact-match), "
                        "'dynamic_focal' (curriculum focal), 'mixture' (combined). "
                        "Focal/Lovász give +36% exact match on math/QA (EMNLP 2024).")
    p.add_argument("--focal-gamma", type=float, default=4.93,
                   help="Focal loss gamma (focusing parameter). 0=CE, 2=typical, "
                        "4.93=evolution-discovered optimum for SFT.")
    p.add_argument("--label-smoothing-eps", type=float, default=0.1,
                   help="Label smoothing epsilon (0=CE, 0.1=typical).")
    p.add_argument("--anchor", default=None,
                   help="Path to anchor checkpoint for L2-SP regularization "
                        "(NeurIPS 2024). If provided, penalizes drift from anchor weights.")
    p.add_argument("--l2-lambda", type=float, default=0.01,
                   help="L2-SP regularization lambda (NeurIPS 2024). "
                        "Lower layers get 10x this value. Default 0.01")
    p.add_argument("--sigreg-weight", type=float, default=0.0,
                   help="SIGReg spectral regularization weight (LeWM/LeCun et al.). "
                        "Penalizes hidden-state spectral norms below threshold to prevent "
                        "collapse. 0 = disabled (default). Typical: 0.01-0.1.")
    p.add_argument("--sigreg-threshold", type=float, default=1.0,
                   help="SIGReg spectral norm threshold. Hidden states with largest "
                        "singular value below this are penalized. Default 1.0.")
    p.add_argument("--sample-weighting", action="store_true", default=False,
                   help="Enable easy sample upweighting (ICML 2025). "
                        "Pre-computes base model loss per example; easy samples get higher weight.")
    # ── MTP auxiliary loss (Nemotron Lightning) ──
    p.add_argument("--mtp-weight", type=float, default=0.0,
                   help="Multi-Token Prediction auxiliary loss weight (Nemotron Lightning). "
                        "When > 0, enables MTP heads for auxiliary next-next-token prediction. "
                        "Improves representation quality and enables speculative decoding at inference. "
                        "0 = disabled (default).")
    p.add_argument("--mtp-n-heads", type=int, default=None,
                   help="Override MTP n_heads (must match checkpoint when warm-starting). "
                        "Default: use config value.")
    # ── Validation during training ──
    p.add_argument("--val-every", type=int, default=0,
                   help="Run validation every N steps (0=disabled). "
                        "Uses a held-out portion of the dataset to compute eval loss. "
                        "Detects regression during training.")
    p.add_argument("--val-size", type=float, default=0.05,
                   help="Fraction of dataset to hold out for validation (default 0.05 = 5%%). "
                        "Only used when --val-every > 0.")
    p.add_argument("--early-stop-patience", type=int, default=0,
                   help="Stop training after N validation checks with no improvement "
                        "(0=disabled). Requires --val-every > 0.")
    p.add_argument("--min-train-loss", type=float, default=0.0,
                   help="Stop training when train loss drops below this threshold "
                        "(0=disabled). Useful for tool-use SFT: stop as soon as the "
                        "format is learned to prevent overfitting/memorization.")
    add_safeguard_args(p)

    # ── Remote Vast.ai cloud backend (offload full fine-tune to rented GPU) ──
    p.add_argument("--remote-vast", action="store_true", default=False,
                   help="Offload training to a rented Vast.ai GPU instance.")
    p.add_argument("--gpu-filter", default="",
                   help="Vast.ai offer query string.")
    p.add_argument("--max-price", type=float, default=0.0)
    p.add_argument("--min-vram-gb", type=float, default=24.0)
    p.add_argument("--min-reliability", type=float, default=0.9)
    p.add_argument("--vast-budget", type=float, default=10.0)
    p.add_argument("--vast-est-sec-per-step", type=float, default=5.0)
    p.add_argument("--vast-disk-gb", type=int, default=100)
    p.add_argument("--vast-image", default=None)
    p.add_argument("--vast-on-demand", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vast-ssh-key", default="")
    p.add_argument("--vast-auto-destroy", action=argparse.BooleanOptionalAction, default=False,
                   help="Destroy instance after run (default False = stop+preserve disk for reuse).")
    p.add_argument("--vast-reuse-instance", action=argparse.BooleanOptionalAction, default=True,
                   help="Find+reuse a stopped instance by label (default True).")
    p.add_argument("--vast-use-volume", action=argparse.BooleanOptionalAction, default=False,
                   help="Attach a persistent Vast.ai volume (default False).")
    p.add_argument("--vast-volume-size-gb", type=int, default=200)
    p.add_argument("--vast-stream-logs", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vast-download-output", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vast-poll-interval", type=float, default=10.0)
    p.add_argument("--vast-startup-timeout", type=float, default=600.0)
    p.add_argument("--vast-maximize-throughput", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--vast-log-filter", default=None,
                   help="Grep filter for live log stream (e.g. 'loss' or 'error').")
    p.add_argument("--from-scratch", action="store_true", default=False,
                   help="Train from random init (no checkpoint). Requires "
                        "--optimizer badam for BitNet int8 configs (auto-set).")

    args = p.parse_args()

    # Evolution-discovered overrides: if --apply-best has been run, the
    # exported best_configs.json overrides argparse *defaults* for the
    # allowlisted params (explicit user flags always win). This makes the
    # "evolution-discovered" comments a live link, not frozen snapshots.
    try:
        from forge.evolution.best_configs import apply_to_namespace
        _evo_defaults = {a.dest: a.default for a in p._actions}
        for _msg in apply_to_namespace(args, _evo_defaults):
            print(f"[evolution] applied {_msg}")
    except Exception as _evo_err:
        print(f"[evolution] best_configs override skipped: {_evo_err}")

    # ── Remote Vast.ai short-circuit: don't train locally ──
    if args.remote_vast:
        from research.cloud.vast_connector import (
            VastConnector,
            build_spec_from_args,
        )
        vast_control_dests = {
            "remote_vast", "gpu_filter", "max_price", "min_vram_gb",
            "min_reliability", "vast_budget", "vast_est_sec_per_step",
            "vast_disk_gb", "vast_image", "vast_on_demand",
            "vast_ssh_key", "vast_auto_destroy", "vast_reuse_instance",
            "vast_use_volume", "vast_volume_size_gb",
            "vast_stream_logs", "vast_download_output",
            "vast_poll_interval", "vast_startup_timeout",
            "vast_maximize_throughput", "vast_log_filter", "from_scratch",
        }
        all_dests = {a.dest for a in p._actions}
        train_arg_dests = all_dests - vast_control_dests
        spec = build_spec_from_args(args, train_arg_dests)
        spec.maximize_throughput = getattr(args, "vast_maximize_throughput", True)
        spec.from_scratch = getattr(args, "from_scratch", False)
        conn = VastConnector(ssh_key=args.vast_ssh_key)
        rc = conn.run_remote_training(spec)
        print(f"\nRemote training finished with exit code {rc}.")
        return rc

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    patch_triton_cache_for_windows()

    # Reduce CUDA memory fragmentation (critical for 12GB VRAM);
    # expandable_segments unsupported on Windows — torch warns per-alloc
    if os.name != "nt":
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bf16" and "cuda" in device else torch.float32

    # ── Load data ──
    print(f"Loading data from {len(args.data)} file(s) "
          f"[format={args.data_format}]...")
    pre_tokenized = None
    if args.data_format == "parquet":
        examples, pre_tokenized = load_examples_parquet(
            args.data, strict_data=args.strict_data
        )
    else:
        examples = load_examples(args.data, strict_data=args.strict_data)

    # ── Curriculum learning: order examples easy→hard ──
    if args.curriculum != "none" and examples:
        from forge.training.data.curriculum_augment import CurriculumScheduler
        scheduler = CurriculumScheduler(strategy=args.curriculum)
        texts = [ex.get('prompt', '') + ' ' + ex.get('response', '') for ex in examples]
        order = scheduler.build_curriculum(texts)
        examples = [examples[i] for i in order]
        print(f"Curriculum ({args.curriculum}): {scheduler.stats()}")

    # ── SYNPRO synthetic data augmentation ──
    if args.synpro and examples:
        from forge.training.data.curriculum_augment import SYNPROGenerator
        gen = SYNPROGenerator()
        texts = [ex.get('prompt', '') + ' ' + ex.get('response', '') for ex in examples[:100]]
        synthetic = gen.generate_batch(texts, n_per_text=2)
        print(f"SYNPRO: generated {len(synthetic)} synthetic variants from {len(texts)} source texts")

    # ── Tokenize (or use pre-tokenized parquet data directly) ──
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # ── Disk tokenization cache info ──
    if args.disk_cache:
        cache = _get_disk_tok_cache()
        print(f"Disk tokenization cache: {cache.stats()}")

    if pre_tokenized is not None:
        # Pre-tokenized parquet: skip tokenisation entirely.
        dataset = pre_tokenized
        print(f"Loaded {len(dataset)} pre-tokenized examples from parquet")
        if not dataset:
            print("No usable examples. Exiting.")
            return
    else:
        if not examples:
            print("No usable examples. Exiting.")
            return
        print(f"Total examples: {len(examples)}")
        n_workers = args.tokenize_workers
        if n_workers <= 0:
            n_workers = min(os.cpu_count() or 1, 32)
        print(f"Tokenizing (with disk cache, {n_workers} workers)...")
        dataset = []
        n_dropped = 0
        if n_workers > 1 and len(examples) > 1000 and sys.platform != "win32":
            # ── Parallel tokenization (multiprocessing.Pool) ──
            # 30x speedup on 128-core Vast.ai instances for 150K+ examples.
            # Linux uses fork() so the tokenizer + disk cache are inherited.
            # Windows can't fork, so falls through to single-threaded below.
            from multiprocessing import get_context
            global _parallel_tokenizer, _parallel_seq_len
            _parallel_tokenizer = tokenizer
            _parallel_seq_len = args.seq_len
            ctx = get_context("fork")
            with ctx.Pool(n_workers) as pool:
                results = pool.map(_parallel_tok_worker, examples, chunksize=256)
            for toks in results:
                if not toks:
                    n_dropped += 1
                    continue
                dataset.extend(toks)
        else:
            # ── Single-threaded (small datasets or 1 worker) ──
            for ex in examples:
                toks = tokenize_example(ex, tokenizer, args.seq_len)
                if not toks:
                    n_dropped += 1
                    continue
                dataset.extend(toks)
        print(f"Tokenized: {len(dataset)} usable examples "
              f"(from {len(examples)} conversations, {n_dropped} dropped)")
        if args.disk_cache:
            print(f"  Disk cache after tokenization: {_get_disk_tok_cache().stats()}")
    if not dataset:
        print("No examples after tokenization. Exiting.")
        return

    # ── Packed sequences (Llama-3 style, eliminates 30-50% padding) ──
    if args.pack_sequences:
        packed_ds = PackedSequenceDataset(dataset, seq_len=args.seq_len,
                                          pack_examples=True, pad_id=pad_id)
        print(f"Packed sequences: {packed_ds.stats()}")
        # R&D round 25: keep packed sequences as tensors — avoid the
        # tensor→list→tensor round-trip that was happening here. The
        # PackedSequenceDataset.__getitem__ returns tensors; collate_batch
        # handles both tensor and list inputs. This eliminates ~5K
        # .tolist() + torch.tensor() conversions per epoch.
        dataset = [{"input_ids": packed_ds[i]["input_ids"],
                     "labels": packed_ds[i]["labels"],
                     "n_comp": packed_ds[i]["n_comp"],
                     "reward": 1.0} for i in range(len(packed_ds))]

    # ── Validation split ──
    val_dataset = None
    if args.val_every > 0:
        n_val = max(1, int(len(dataset) * args.val_size))
        random.shuffle(dataset)
        val_dataset = dataset[:n_val]
        dataset = dataset[n_val:]
        print(f"Validation: {len(val_dataset)} examples held out "
              f"(train: {len(dataset)}, val_every={args.val_every})")

    # ── Build model ──
    config_overrides = None
    if args.config_overrides:
        import json as _json
        config_overrides = _json.loads(args.config_overrides)
    cfg = get_config(args.config, device=device,
                     **(config_overrides or {}))
    cfg.grad_clip = args.grad_clip
    if args.grad_checkpoint:
        cfg.use_gradient_checkpointing = True
        cfg.selective_gradient_checkpointing = args.checkpoint_strategy
        if args.checkpoint_strategy != "all":
            print(f"Selective gradient checkpointing: '{args.checkpoint_strategy}' strategy")
    elif args.batch_size > 8 or args.seq_len > 4096:
        # Auto-enable gradient checkpointing only for very large configs
        cfg.use_gradient_checkpointing = True
        print("Auto-enabled gradient checkpointing (batch>8 or seq>4096)")
    # Use chunked CE to save VRAM on the 65K vocab.
    cfg.use_chunked_ce = True
    cfg.ce_chunk_size = 128
    # MTP auxiliary loss (Nemotron Lightning)
    if args.mtp_weight > 0.0:
        cfg.use_mtp = True
        cfg.mtp_loss_weight = args.mtp_weight
        print(f"MTP auxiliary loss enabled (weight={args.mtp_weight})")
    if args.mtp_n_heads is not None:
        cfg.mtp_n_heads = args.mtp_n_heads
        print(f"MTP n_heads overridden to {args.mtp_n_heads} (must match checkpoint)")

    print(f"Building model ({args.config}) from {args.checkpoint}...")
    _from_scratch = args.checkpoint in ("scratch", "none", "") or args.from_scratch
    # ── From-scratch training: "scratch"/"none" = random init, no checkpoint ──
    if _from_scratch:
        print("  Random initialization (from-scratch training, no checkpoint)")
        args.checkpoint = None
        # BitNet int8 training requires BAdam optimizer (Muon's Newton-Schulz
        # conflicts with BitNet weight normalization). Warn if mismatched.
        if cfg.use_bitnet and getattr(cfg, 'bitnet_int8_training', False) \
                and args.optimizer not in ("badam", "fira_nlrq"):
            print(f"  WARNING: BitNet int8 training with --optimizer {args.optimizer} "
                  f"is not supported. Switching to --optimizer badam.")
            args.optimizer = "badam"
    # Use ForgeEngine for model loading (unified inference engine with all
    # activation features, crash recovery, KeyStack auto-detection).
    # The engine.model is used for training; engine is used for validation/eval.
    forge_engine = None
    # Set device on config so build_model_fast loads weights directly to GPU,
    # avoiding a CPU RAM copy that can exceed 32GB on 8B+ models.
    cfg.device = str(device)

    # ── HuggingFace model loading (for QAT on non-ForgeAI models) ──
    if args.hf_model is not None:
        from transformers import AutoModelForCausalLM as _HFModel
        print(f"  Loading HuggingFace model: {args.hf_model}")
        model = _HFModel.from_pretrained(
            args.hf_model, dtype=dtype, trust_remote_code=True).to(device)
        # Override tokenizer with HF model's tokenizer
        from transformers import AutoTokenizer as _HFTok
        tokenizer = _HFTok.from_pretrained(
            args.hf_model, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        pad_id = tokenizer.pad_token_id
        print(f"  HF model loaded: {type(model).__name__}, "
              f"vocab={tokenizer.vocab_size}, pad_id={pad_id}")
    elif args.checkpoint is not None and args.use_forge_engine:
        try:
            from forge.engine.forge_engine import ForgeEngine
            print("  Loading via ForgeEngine (auto_activate=False for training)...")
            forge_engine = ForgeEngine.from_checkpoint(
                args.checkpoint, config_name=args.config,
                device=device, auto_activate=False,
                config_overrides=config_overrides)
            model = forge_engine.model
            print(f"  ForgeEngine loaded: {type(model).__name__}, "
                  f"KeyStack features: {forge_engine.keystack_features}")
        except Exception as e:
            print(f"  ForgeEngine load failed ({e}), falling back to ModelLoader")
            forge_engine = None
            model = ModelLoader.build_model_fast(cfg, checkpoint_path=args.checkpoint, dtype=dtype)
    else:
        model = ModelLoader.build_model_fast(cfg, checkpoint_path=args.checkpoint, dtype=dtype)
    # Move to target device. When build_model_fast used meta init + assign=True,
    # the weights are already on the target device but some non-persistent
    # buffers may still be on meta — use to_empty() as a fallback.
    try:
        model.to(device)
    except NotImplementedError:
        model = model.to_empty(device=device)
    model.train()

    # ── From-scratch 8B init: NLRQ reset, BitNet QAT disable, kaiming init,
    #    logit scale normalization (mirrors train_8b_all.build_model) ──
    if _from_scratch:
        from types import SimpleNamespace

        from forge.training.runners.train_8b_all import (
            disable_bitnet_qat_,
            initialize_weights_,
            normalize_logit_scale_,
            reset_nlrq_layers_,
        )
        reset_nlrq_layers_(model)
        disable_bitnet_qat_(model)
        initialize_weights_(model, cfg.n_layers)
        scale = normalize_logit_scale_(
            model, torch.device(device),
            SimpleNamespace(vocab_size=cfg.vocab_size))
        if scale != 1.0:
            print(f"  [FromScratch] Logit scale normalized by {scale:.3f} "
                  f"(init CE ≈ ln(vocab) = {math.log(cfg.vocab_size):.2f})")

    # ── Advanced normalization: SeeDNorm or DyT ──
    if args.norm_type == "seednorm":
        from forge.training.optim.advanced_norm import replace_rmsnorm_with_seednorm
        replace_rmsnorm_with_seednorm(model)
    elif args.norm_type == "dyt":
        from forge.training.optim.advanced_norm import replace_rmsnorm_with_dyt
        replace_rmsnorm_with_dyt(model)

    # ── BitNet-everywhere: convert all Linear → BitNetLinear ──
    if args.bitnet_everywhere and not args.nanoquant_qat:
        from forge.training.bitnet_lora import convert_to_bitnet_everywhere
        n_conv, n_already = convert_to_bitnet_everywhere(model)
        print(f"BitNet-everywhere: {n_conv} Linear → BitNetLinear, {n_already} already BitNet")

    # ── NanoQuant QAT: convert all Linear → NanoQuantQATLinear (sub-1-bit) ──
    if args.nanoquant_qat:
        from forge.engine.quant.novel_quant_r48 import (
            convert_model_to_nanoquant_qat,
            estimate_r48_memory,
        )
        n_conv = convert_model_to_nanoquant_qat(
            model, rank=args.nanoquant_rank,
            quick_init=bool(args.nanoquant_quick_init), verbose=True)
        mem = estimate_r48_memory(model)
        print(f"NanoQuant QAT: {n_conv} Linear → NanoQuantQATLinear "
              f"(rank={args.nanoquant_rank}, ~{mem['total_mb']:.1f} MB, "
              f"{mem['avg_eff_bits']:.3f} bits/w)")
        # Disable LoRA — NanoQuant QAT trains the binary factors directly
        args.lora = False
        args.manual_lora = False

    # ── R32-1: NF4 QLoRA — quantize base to NF4, then add LoRA adapters ──
    if args.qlora_nf4:
        from forge.training.bitnet_lora import convert_to_nf4_qlora
        target_mods = ["q_proj", "k_proj", "v_proj", "out_proj",
                       "w_gate", "w_up", "w_down"]
        n_conv, n_skip = convert_to_nf4_qlora(
            model, group_size=args.nf4_group_size, target_modules=target_mods)
        print(f"NF4 QLoRA: {n_conv} Linear → NF4Linear ({n_skip} skipped), "
              f"group_size={args.nf4_group_size}")
        # Force manual LoRA (NF4Linear needs manual adapters, PEFT can't handle it)
        args.lora = True
        args.manual_lora = True

    # ── ForgeQuant training — quantize base to INT4+sparse, then add LoRA ──
    if getattr(args, 'forge_quant', False):
        from forge.engine.quant.forge_quant import quantize_model_forge_quant
        n_q = quantize_model_forge_quant(
            model, group_size=128, sparse_ratio=0.10)
        print(f"ForgeQuant: {n_q} layers quantized (INT4 dense + INT8 sparse, "
              f"group=128, sparse_ratio=0.10). Training on quantized base.")
        # ForgeQuantLinear has built-in lora_adapter support — use manual LoRA
        args.lora = True
        args.manual_lora = True

    # ── LoRA ──
    # Use manual LoRA if --bitnet-everywhere or --manual-lora or --qlora-nf4
    # (PEFT can't handle BitNetLinear or NF4Linear).
    # Use PEFT LoRA only for --lora on non-BitNet, non-NF4 models.
    use_manual_lora = args.manual_lora or (args.bitnet_everywhere and args.lora)

    if use_manual_lora:
        from forge.training.bitnet_lora import add_lora_adapters
        target_mods = ["q_proj", "k_proj", "v_proj", "out_proj",
                       "w_gate", "w_up", "w_down"] if not args.bitnet_everywhere else None
        n_adapters, lora_params = add_lora_adapters(
            model, rank=args.lora_r, alpha=args.lora_alpha,
            target_modules=target_mods,
        )
        print(f"Manual LoRA: {n_adapters} adapters (rank={args.lora_r}), "
              f"{sum(p.numel() for p in lora_params)/1e6:.2f}M trainable params")
        # Freeze ALL non-LoRA params — only LoRA adapter weights should train.
        # Without this, embeddings/norms/MTP/NLRQ factors all keep requires_grad=True,
        # causing CPUAdamW to allocate optimizer states for 1.9B+ params → OOM.
        lora_param_ids = {id(p) for p in lora_params}
        n_frozen = 0
        for param in model.parameters():
            if id(param) not in lora_param_ids:
                param.requires_grad_(False)
                n_frozen += 1
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Frozen {n_frozen} non-LoRA params | "
              f"{n_trainable/1e6:.2f}M trainable total")
    elif args.lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            # LFM2.5 module names: attn uses out_proj (not o_proj),
            # ffn uses w_gate/w_up/w_down (not gate_proj/up_proj/down_proj).
            # No task_type — ConfigurableResearchLLM is not an HF model;
            # we only train + merge, no generation through PEFT wrapper.
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj",
                            "w_gate", "w_up", "w_down"],
            lora_dropout=0.0,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        for param in model.parameters():
            param.requires_grad_(True)
        # With tied embeddings + BAdam: the head.weight IS embed.weight (same
        # tensor). BAdam's block partitioning sees both and toggles
        # requires_grad inconsistently → "does not require grad" error on the
        # embeddings_head block. Mark head.weight as frozen so BAdam only
        # manages it via embed.weight (one reference, no conflict).
        if getattr(cfg, 'tie_word_embeddings', True) and args.optimizer == "badam":
            if hasattr(model, 'head') and hasattr(model, 'embed'):
                if model.head.weight is model.embed.weight:
                    # Mark the head's reference as frozen — embed.weight still
                    # trains normally in the embeddings block.
                    model.head.weight._forge_frozen = True
                    print("  [BAdam] Marked tied head.weight as frozen (embed.weight still trains)")

    # ── R&D round 15: BitNet int8 trainable storage ──
    # For 8B dense models on 12GB VRAM: int8 ternary weights on GPU (8GB),
    # bf16 master weights on CPU (16GB RAM). BAdam updates CPU master,
    # then re-quantizes active block to int8 on GPU.
    if getattr(cfg, 'bitnet_int8_training', False):
        # Check if model was already loaded with int8 weights (prequantized).
        # If so, skip convert_model_to_int8 / enable_int8_training — the
        # weights are already int8 buffers, no bf16→int8 re-quantization needed.
        from forge.keys.quantization.bitnet_b158_key import BitNetLinear
        n_already_int8 = sum(1 for m in model.modules()
                             if isinstance(m, BitNetLinear) and m._prequantized)
        if use_manual_lora:
            if n_already_int8 > 0:
                gpu_w = sum(m.weight_int8.numel() for m in model.modules()
                            if hasattr(m, 'weight_int8'))
                print(f"  [Int8Storage] {n_already_int8} BitNetLinear layers already int8 (prequantized load)")
                print(f"    GPU int8 weights: {gpu_w/1e9:.2f}B params ({gpu_w/1e9:.2f} GB)")
            else:
                # LoRA path: freeze base as int8 inference storage (no CPU master,
                # no trainable int8). Only LoRA adapters train.
                from forge.keys.quantization.bitnet_b158_key import convert_model_to_int8
                n_converted = convert_model_to_int8(model)
                if n_converted > 0:
                    gpu_w = sum(m.weight_int8.numel() for m in model.modules()
                                if hasattr(m, 'weight_int8'))
                    print(f"  [Int8Storage] {n_converted} BitNetLinear layers → int8 inference storage (LoRA base frozen)")
                    print(f"    GPU int8 weights: {gpu_w/1e9:.2f}B params ({gpu_w/1e9:.2f} GB)")
        else:
            if n_already_int8 > 0:
                print(f"  [Int8Train] {n_already_int8} layers already int8 — skipping enable_int8_training (no bf16 re-quant)")
            else:
                from forge.keys.quantization.bitnet_b158_key import enable_int8_training
                n_converted = enable_int8_training(model)
                if n_converted > 0:
                    gpu_w = sum(m.weight_int8.numel() for m in model.modules()
                                if hasattr(m, '_int8_trainable') and m._int8_trainable)
                    cpu_w = sum(m.weight.numel() for m in model.modules()
                                if hasattr(m, '_int8_trainable') and m._int8_trainable)
                    print(f"  [Int8Train] {n_converted} BitNetLinear layers converted to int8 trainable storage")
                    print(f"    GPU int8 weights: {gpu_w/1e9:.2f}B params ({gpu_w/1e9:.2f} GB)")
                    print(f"    CPU bf16 master:  {cpu_w/1e9:.2f}B params ({cpu_w*2/1e9:.2f} GB RAM)")

    # ── NLRQ factor training (STE masters) for NLRQ-compressed configs ──
    # 8B-D uses NLRQ FFN compression. Without STE factor training, only the
    # singular values S train — U/V factors stay frozen at their init.
    # enable_factor_training_all_ creates float master factors (U_m, V_m)
    # that train via straight-through estimator around the INT8 quantizer.
    # Checkpoints stay pure-INT8 (masters are stripped on save via export).
    if getattr(cfg, 'ffn_compression', 'none') == "nlrq" and not use_manual_lora:
        from forge.training.runners.train_8b_all import enable_factor_training_all_
        n_nlrq = enable_factor_training_all_(model)
        if n_nlrq > 0:
            print(f"  [NLRQ-STE] {n_nlrq} layers with factor training (U_m/V_m masters)")

    # ── MTP freezing when MTP loss is not active ──
    # When use_mtp=True but mtp_weight=0 (default), the MTP draft head gets
    # no gradients (the CE loss pathway doesn't pass targets to MTP). These
    # are dead params that BAdam would partition into their own block →
    # crash when that block activates ("element 0 does not require grad").
    # Freeze explicitly before the dead-param probe to skip the probe for MTP.
    if getattr(cfg, 'use_mtp', False) and args.mtp_weight <= 0.0:
        mtp = getattr(model, 'mtp_module', None)
        if mtp is not None:
            n_mtp_params = 0
            for p in mtp.parameters():
                p.requires_grad_(False)
                p._forge_frozen = True  # BAdam excludes _forge_frozen params
                n_mtp_params += 1
            print(f"  [MTP] Frozen {n_mtp_params} MTP params (mtp_weight=0, no CE pathway)")

    # ── Dead-param freeze probe (critical for BAdam + 8B) ──
    # Models carry architecture-conditional modules that never receive
    # gradients: MTP draft head (when mtp_weight=0), loop_block (hyperloop),
    # gated-off AttnRes/MHC paths, unused heads. BAdam partitions ALL params
    # into blocks; when a block contains ONLY dead params, backward() crashes
    # with "element 0 of tensors does not require grad".
    # The probe does one tiny forward+backward with all params trainable,
    # then freezes any param where p.grad is None (autograd's reachability
    # truth). This mirrors train_8b_all.freeze_dead_params_.
    if args.optimizer in ("badam", "fira_nlrq"):
        # Skip the dead-param probe when int8 training is active — the probe
        # does a full forward+backward with ALL params trainable, which creates
        # bf16 grads for the entire model on GPU. With int8 training, the
        # masters are on CPU and the probe would OOM trying to materialize
        # 9.7GB of grads on GPU. The MTP freeze above already handles the
        # known dead params; other dead params (loop_block, gated modules)
        # are not present in V10 configs.
        if not getattr(cfg, 'bitnet_int8_training', False):
            from forge.training.runners.train_8b_all import freeze_dead_params_
            use_flce = args.batch_size * args.seq_len >= 1024
            n_dead = freeze_dead_params_(model, torch.device(device), use_flce=use_flce)
            if n_dead > 0:
                print(f"  [DeadParams] Froze {n_dead} dead param tensors "
                      f"(no grad path — MTP/loop_block/gated modules); "
                      f"BAdam blocks now contain only live params")
        else:
            print("  [DeadParams] Skipped probe (int8 training — MTP freeze sufficient)")

    # ── Post-loading memory cleanup ──
    # The model loading path (build_model_fast + convert_model_to_int8) creates
    # large intermediate tensors (int8→bf16 cast = 10.5 GB for 5B params) that
    # may not be returned to the OS on Windows. Force GC + empty_cache after
    # all model modifications are done, before the training loop starts.
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if device == "cuda":
        vram_after = torch.cuda.memory_allocated() / 1e9
        import psutil
        ram_pct = psutil.virtual_memory().percent
        print(f"  [MemCleanup] After loading: VRAM {vram_after:.2f} GB, "
              f"RAM {ram_pct:.0f}%")

    if args.grad_checkpoint:
        # ConfigurableResearchLLM uses enable_gradient_checkpointing (not HF's
        # gradient_checkpointing_enable). The config flag path above already
        # enabled it via cfg.use_gradient_checkpointing, but call the method
        # explicitly too in case the model was built before the flag was set.
        if hasattr(model, "enable_gradient_checkpointing"):
            model.enable_gradient_checkpointing()

        # LazyTrain: mixed-integer checkpoint scheduling (1.24x TFLOPS)
        if args.lazy_train or args.checkpoint_strategy == "lazy":
            try:
                from forge.training.runners.lazy_train import LazyTrainScheduler, TrainingBudget
                budget = TrainingBudget(gpu_memory_bytes=(12 * 1024**3))
                scheduler = LazyTrainScheduler(model, budget)
                scheduler.analyze()
                scheduler.apply()
                print(f"  [LazyTrain] {scheduler.stats()}")
            except Exception as e:
                print(f"  [LazyTrain] unavailable ({e}), using standard checkpointing")

        # Optimal checkpointing: Hirschberg knapsack (25-28% faster than dp_knapsack)
        if args.checkpoint_strategy == "optimal":
            try:
                from forge.training.runners.optimal_checkpoint import OptimalCheckpointPlanner
                planner = OptimalCheckpointPlanner(
                    model, memory_budget_bytes=(6 * 1024**3))
                plan = planner.plan()
                planner.apply(plan)
                print(f"  [OptimalCheckpoint] {planner.memory_estimate()}")
            except Exception as e:
                print(f"  [OptimalCheckpoint] unavailable ({e})")

    # ── OOMB chunk-recurrent training (long context, O(1) activation memory) ──
    if args.oomb:
        try:
            from forge.training.runners.oomb_trainer import ChunkRecurrentTrainer
            ChunkRecurrentTrainer(
                model, chunk_size=512, max_seq_len=args.seq_len * 4,
                device=str(device), dtype=dtype)
            print(f"  [OOMB] Chunk-recurrent training: active "
                  f"(chunk=512, max_seq={args.seq_len * 4})")
        except Exception as e:
            print(f"  [OOMB] unavailable ({e})")

    # ── Hybrid 8-bit fast gradient clipping ──
    hybrid_clipper = None
    if args.hybrid_clip:
        try:
            from forge.training.runners.lazy_train import Hybrid8BitOperator
            hybrid_clipper = Hybrid8BitOperator(max_norm=args.grad_clip)
            print("  [Hybrid8Bit] Fast gradient clipping: active (EMA norm, exact/50)")
        except Exception as e:
            print(f"  [Hybrid8Bit] unavailable ({e})")

    # ── torch.compile (experimental) ──
    if args.compile:
        try:
            from research.paths import TORCH_CACHE_DIR
            model = torch.compile(model, mode="reduce-overhead", dynamic=True,
                                  cache_dir=str(TORCH_CACHE_DIR / "inductor"))
            print("torch.compile enabled (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile failed ({e}), continuing without")

    # Auto-default to CPUAdamW on small GPUs (<15GB) if user didn't override
    if args.optimizer == "adamw" and torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if total_vram_gb < 15.0:
            print(f"  [Auto] GPU has {total_vram_gb:.1f}GB VRAM — switching to CPUAdamW optimizer")
            args.optimizer = "cpu_offload"

    optimizer = configure_optimizer(model, args.lr, args.weight_decay,
                                     optimizer_name=args.optimizer,
                                     freetoken=args.freetoken,
                                     double_buffer=args.double_buffer,
                                     bandwidth_adaptive=args.bandwidth_adaptive,
                                     chunk_size_mb=args.chunk_size_mb)

    qk_clip_monitor = None
    if getattr(args, "qk_clip_tau", 0) > 0:
        from forge.training.optim.qk_clip import QKClipMonitor
        qk_clip_monitor = QKClipMonitor.attach(model, tau=args.qk_clip_tau)
        print(f"  [QK-Clip] monitoring {len(qk_clip_monitor.attention_modules)} attention layers at tau={args.qk_clip_tau}")

    # ── FORGE optimizer: register gradient hooks (must run AFTER optimizer
    # creation — the optimizer object didn't exist at the previous site) ──
    if args.optimizer == "forge":
        if hasattr(optimizer, 'register_hooks'):
            optimizer.register_hooks(model)

    # ── L2-SP Anchor Regularization (NeurIPS 2024) ──
    # Load anchor checkpoint and build a name→tensor map for L2-SP loss.
    # Only activates when --anchor is provided AND l2_lambda > 0.
    anchor_named_params = None
    if args.anchor and getattr(args, 'l2_lambda', 0) > 0:
        print(f"Loading anchor checkpoint for L2-SP: {args.anchor}")
        anchor_sd = load_anchor_cached(args.anchor)
        # Map anchor tensors to model device; match by parameter name.
        anchor_named_params = {}
        model_named = dict(model.named_parameters())
        for name, p in model_named.items():
            if name in anchor_sd:
                anchor_named_params[name] = anchor_sd[name].to(device).to(p.dtype)
        print(f"  L2-SP active: {len(anchor_named_params)} matched params, "
              f"lambda={args.l2_lambda} (lower layers 10x)")

    # ── Easy Sample Upweighting (ICML 2025) ──
    # Pre-compute per-example weights from base model loss.
    # Only activates when --sample-weighting flag is set.
    sample_weights = None
    if args.sample_weighting:
        print("Computing sample weights from base model loss (ICML 2025)...")
        sample_weights = compute_sample_weights(
            model, dataset, pad_id, device, batch_size=1)
        print(f"  Sample weights: min={sample_weights.min().item():.4f}, "
              f"max={sample_weights.max().item():.4f}, "
              f"mean={sample_weights.mean().item():.4f}")

    # ── EMA ──
    ema_state = None
    if args.ema:
        ema_state = init_ema(model)
        print(f"EMA enabled (decay={args.ema_decay})")

    # ── Grad accumulation ──
    grad_accum = max(1, args.grad_accum)
    eff_batch = args.batch_size * grad_accum
    print(f"\nTraining {args.max_steps} steps | batch {args.batch_size} | "
          f"grad_accum {grad_accum} | eff_batch {eff_batch} | "
          f"lr {args.lr} | seq_len {args.seq_len} | {len(dataset)} examples")
    if args.early_stop_patience > 0:
        print(f"  Early stopping: patience={args.early_stop_patience} val checks")
    if args.min_train_loss > 0:
        print(f"  Min train loss threshold: {args.min_train_loss} (stops when format learned)")
    # Early stopping tracking
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopped = False
    if args.grad_mixup > 1:
        print(f"  Grad mixup: {args.grad_mixup}-way (averaging {args.grad_mixup} batches' gradients per step)")
    if args.entropy_alpha > 0:
        print(f"  Token entropy weighting: alpha={args.entropy_alpha} (WeFT/VCORE 2025)")

    # ── Sequential freeze/unfreeze schedule ──
    phase_schedule = None
    if args.sequential_freeze > 0:
        from forge.training.bitnet_lora import compute_phase_schedule
        n_layers = len(model.blocks) if hasattr(model, 'blocks') else 16
        phase_schedule = compute_phase_schedule(
            n_layers=n_layers,
            n_phases=args.sequential_freeze,
            total_steps=args.max_steps,
            final_finetune_steps=args.final_finetune_steps,
        )
        print(f"  Sequential freeze: {args.sequential_freeze} phases, "
              f"{n_layers} layers, "
              f"{len(phase_schedule)} schedule entries")
        for i, (s, e, active) in enumerate(phase_schedule):
            layer_str = "ALL" if active is None else f"{active[0]}-{active[-1]}"
            print(f"    Phase {i+1}: steps {s}-{e}, layers {layer_str}")

    print(f"Save: {args.save}")

    # ── Async prefetcher info ──
    if args.async_prefetch:
        print(f"Async prefetcher: enabled (prefetch_count={args.prefetch_count})")

    aborted = False
    with task_scope("sft") as log:
        step = 0
        accum_count = 0
        indices = list(range(len(dataset)))
        last_loss = 0.0  # initialized; updated at log intervals (R&D round 25)

        # ── Async prefetcher: pre-collate batches in background ──
        # The prefetcher runs a background thread that pre-loads and collates
        # the next N batches while the GPU processes the current one.
        # We use it to pre-collate batches, then consume them in the main loop.
        prefetcher = None
        if args.async_prefetch and not args.augment:
            prefetcher = AsyncPrefetcher(
                dataset, batch_size=args.batch_size, device=device,
                collate_fn=collate_batch, prefetch_count=args.prefetch_count,
                shuffle=True, pad_id=pad_id)
            # R&D round 25: CPU-first collate + consumer-side H2D transfer.
            # The background thread produces pinned CPU tensors; the main
            # thread does .to(device, non_blocking=True) which overlaps the
            # H2D copy with the previous batch's GPU compute.
            if "cuda" in device:
                prefetcher.set_transfer_on_consume(True)

        for epoch in range(100):
            random.shuffle(indices)
            # If using prefetcher, get batches from it; otherwise use indices
            if prefetcher is not None:
                batch_iter = iter(prefetcher)
            for batch_start in range(0, len(indices), args.batch_size):
                if step >= args.max_steps:
                    break
                # R&D round 25: run VRAM/RAM safeguards every 10 steps instead
                # of every step. torch.cuda.memory_allocated() is a sync point
                # and psutil.virtual_memory() is a syscall — both add per-step
                # latency that starves the GPU pipeline. 10-step granularity is
                # sufficient: OOM develops over multiple steps, not instantly.
                if step % 10 == 0:
                    if vram_exceeded(args.vram_limit_gb, device):
                        print("VRAM limit exceeded; emergency save + abort.")
                        emergency_save(model, args.save, "emergency", step, optimizer=optimizer)
                        aborted = True
                        break

                    # ── RAM safeguard (psutil) ──
                    # Prevents OOMKilled / freezing / stuttering from host memory pressure.
                    # Throttle at threshold%, emergency save at threshold+10%.
                    if ram_exceeded(args.ram_limit_percent):
                        ru = ram_usage()
                        emergency_pct = args.ram_limit_percent + 10
                        if ru.get("percent", 0) > emergency_pct:
                            print(f"RAM critical ({ru.get('percent', 0):.0f}%); emergency save + abort.")
                            emergency_save(model, args.save, "emergency", step, optimizer=optimizer)
                            aborted = True
                            break
                        else:
                            print(f"RAM high ({ru.get('percent', 0):.0f}%); gc.collect() + skip batch.")
                            import gc
                            gc.collect()
                            continue  # skip this batch, try next

                # ── Periodic gc.collect() to prevent memory fragmentation ──
                # R&D round 25: reduced from every 100 to every 200 steps.
                # gc.collect() is expensive (can take 100ms+ on large heaps)
                # and the BAdam optimizer now does its own periodic cleanup.
                if step > 0 and step % 200 == 0:
                    import gc
                    gc.collect()

                batch_idx = indices[batch_start:batch_start + args.batch_size]
                if prefetcher is not None:
                    # Async prefetcher: batch is already collated and on device
                    try:
                        input_ids, labels, attn_mask, reward_weights = next(batch_iter)
                    except StopIteration:
                        break  # epoch exhausted
                else:
                    # Manual batching fallback (no DataLoader).
                    # collate_batch already uses pin_memory for GPU transfers.
                    # Future improvement: wrap dataset in torch.utils.data.DataLoader
                    # with num_workers=4, pin_memory=True, prefetch_factor=4 for
                    # multi-process data loading. The AsyncPrefetcher (enabled by
                    # default via --async-prefetch) provides similar throughput
                    # using a single background thread.
                    batch = [dataset[i] for i in batch_idx]
                    with log.time_phase("data_load"):
                        input_ids, labels, attn_mask, reward_weights = collate_batch(batch, pad_id, device)

                # ── Training-time data augmentation ──
                if args.augment:
                    from forge.training.data.curriculum_augment import DataAugmentor
                    if not hasattr(main, '_augmentor'):
                        main._augmentor = DataAugmentor()
                    aug_input, aug_target = main._augmentor.augment(input_ids, vocab_size=cfg.vocab_size)
                    input_ids = aug_input
                    labels = aug_target

                # ── Sequential freeze/unfreeze ──
                # At phase boundaries, freeze/unfreeze LoRA params and rebuild optimizer.
                if phase_schedule is not None:
                    from forge.training.bitnet_lora import (
                        freeze_unfreeze_lora,
                        get_active_layers_for_step,
                        get_active_lora_params,
                    )
                    active_layers = get_active_layers_for_step(step, phase_schedule)
                    # Check if we just entered a new phase (compare to previous step)
                    if step == 0 or get_active_layers_for_step(step - 1, phase_schedule) != active_layers:
                        freeze_unfreeze_lora(model, active_layers=active_layers)
                        active_p = get_active_lora_params(model)
                        layer_str = "ALL" if active_layers is None else f"{active_layers[0]}-{active_layers[-1]}"
                        print(f"  [Phase] step {step}: training layers {layer_str} ({len(active_p)} params)")
                        # Rebuild optimizer with only active params
                        if active_p:
                            if args.optimizer in ("muon_sf", "muon_sf_plain"):
                                from forge.training.bitnet_lora import build_muon_sf_lora_opt
                                optimizer = build_muon_sf_lora_opt(active_p, lr_muon=5e-3, lr_adam=args.lr)
                            else:
                                optimizer = configure_optimizer(model, args.lr, args.weight_decay,
                                                                optimizer_name=args.optimizer)
                            if hasattr(optimizer, 'train'):
                                optimizer.train()

                # ── Easy Sample Upweighting (ICML 2025) ──
                # Get pre-computed sample weights for this batch (if enabled).
                sw = None
                if sample_weights is not None:
                    sw = sample_weights[batch_idx].to(device)

                # ── Reward-Weighted SFT ──
                # Weight per-example loss by trajectory reward (if present).
                # High-reward trajectories get more weight, low-reward get less.
                # reward_weights is [B] with values in [0, 1]; normalize to mean=1.
                rw = reward_weights
                if rw.max() > 0:
                    rw = rw / rw.mean().clamp(min=1e-8)  # normalize mean=1
                    if sw is not None:
                        sw = sw * rw  # combine with easy-sample weights
                    else:
                        sw = rw

                # ── FreeToken 3-stage pipeline (R&D round 14) ──
                # With overlap optimizer (cpu_offload + double_buffer/freetoken):
                # The previous step's CPU AdamW runs in a background thread,
                # overlapping with this forward pass. We must wait() before
                # backward to ensure params are synced and grads are consumed.
                # This creates the 3-stage pipeline:
                #   Stage 1: GPU forward+backward (current batch)
                #   Stage 2: GPU→CPU grad transfer (overlapped with stage 3)
                #   Stage 3: CPU AdamW update (overlapped with stage 1 of next batch)
                _has_overlap = hasattr(optimizer, 'overlap') and optimizer.overlap
                if _has_overlap and hasattr(optimizer, 'wait'):
                    optimizer.wait()

                # Forward + backward (accumulate gradients).
                # Token entropy weighting (WeFT/VCORE 2025) applied inside compute_loss.
                with oom_guard(str(device), label="sft_fwd") as safe:
                    with log.time_phase("forward"):
                        ce_loss = compute_loss(model, input_ids, labels, attn_mask,
                                               entropy_alpha=args.entropy_alpha,
                                               sample_weights=sw,
                                               sigreg_weight=args.sigreg_weight,
                                               sigreg_threshold=args.sigreg_threshold,
                                               is_hf_model=args.hf_model is not None)

                    # ── L2-SP Anchor Regularization (NeurIPS 2024) ──
                    # Total loss = CE_loss + l2_lambda * l2_sp_loss (layer-wise lambda inside).
                    if anchor_named_params is not None:
                        l2_sp = compute_l2_sp_loss(model, anchor_named_params, args.l2_lambda)
                        loss = ce_loss + l2_sp
                    else:
                        loss = ce_loss

                    with log.time_phase("backward"):
                        (loss / grad_accum).backward()
                if safe.skipped:
                    optimizer.zero_grad()
                    accum_count = 0
                    continue
                accum_count += 1
                # R&D round 25: defer .item() to log intervals to avoid
                # GPU→CPU sync every step. Keep the loss tensor detached on
                # GPU; only materialize the scalar when logging (every 5
                # steps). This eliminates a per-step sync point that was
                # blocking the GPU pipeline.
                last_loss_tensor = ce_loss.detach()

                # ── Grad Mixup (novel: average gradients from N batches) ──
                # Tested in .devin/test_stack_winners.py: 3-way mixup + muon_sf
                # = 1.25x better convergence vs AdamW. Averages gradients from
                # N-1 additional batches before the optimizer step.
                if args.grad_mixup > 1 and accum_count >= grad_accum:
                    # Save current grads from first batch
                    saved_grads = {}
                    for n, p in model.named_parameters():
                        if p.grad is not None:
                            saved_grads[n] = p.grad.clone()

                    # Fetch N-1 more batches and accumulate their grads
                    mixup_batch_start = batch_start + args.batch_size
                    for mixup_i in range(args.grad_mixup - 1):
                        mixup_bs = mixup_batch_start + mixup_i * args.batch_size
                        if mixup_bs + args.batch_size > len(indices):
                            break  # not enough data for full mixup, use what we have
                        mixup_idx = indices[mixup_bs:mixup_bs + args.batch_size]
                        mixup_batch = [dataset[i] for i in mixup_idx]
                        mi, ml, mm, _ = collate_batch(mixup_batch, pad_id, device)
                        msw = sample_weights[mixup_idx].to(device) if sample_weights is not None else None
                        mce = compute_loss(model, mi, ml, mm,
                                           entropy_alpha=args.entropy_alpha,
                                           sample_weights=msw,
                                           sigreg_weight=args.sigreg_weight,
                                           sigreg_threshold=args.sigreg_threshold)
                        if anchor_named_params is not None:
                            mloss = mce + compute_l2_sp_loss(model, anchor_named_params, args.l2_lambda)
                        else:
                            mloss = mce
                        optimizer.zero_grad()
                        (mloss / grad_accum).backward()
                        # Accumulate: running average
                        for n, p in model.named_parameters():
                            if p.grad is not None and n in saved_grads:
                                saved_grads[n] = (saved_grads[n] * (mixup_i + 1) + p.grad) / (mixup_i + 2)

                    # Restore averaged grads
                    optimizer.zero_grad()
                    for n, p in model.named_parameters():
                        if n in saved_grads:
                            p.grad = saved_grads[n]

                # Only step optimizer when we've accumulated enough gradients.
                if accum_count < grad_accum:
                    continue

                # Optimizer step.
                with log.time_phase("optimizer"):
                    if hybrid_clipper is not None:
                        hybrid_clipper.clip(model.parameters())
                    else:
                        # R&D round 25: for BAdam, only the active block has
                        # gradients (all other params have grad=None). Filter
                        # to only params with grads to avoid iterating the
                        # full parameter list every step.
                        grad_params = [p for p in model.parameters()
                                       if p.grad is not None]
                        if grad_params:
                            torch.nn.utils.clip_grad_norm_(grad_params, args.grad_clip)
                    lr = get_lr(step, args.max_steps, args.lr, args.min_lr, args.warmup_steps)
                    for g in optimizer.param_groups:
                        g["lr"] = lr
                    optimizer.step()
                    optimizer.zero_grad()
                    # QK-Clip (Kimi K2 MuonClip): cap per-head attention logits.
                    if qk_clip_monitor is not None:
                        n_clip = qk_clip_monitor.clip(model)
                        if n_clip:
                            print(f"  [QK-Clip] step {step}: clipped {n_clip} head(s) at tau={qk_clip_monitor.tau}")
                    # BitNet int8: re-quantize CPU master → GPU int8 buffer
                    if getattr(cfg, 'bitnet_int8_training', False):
                        from forge.keys.quantization.bitnet_b158_key import BitNetLinear
                        for m in model.modules():
                            if isinstance(m, BitNetLinear) and getattr(m, '_int8_trainable', False):
                                m.requantize_from_master()
                    # DeepSeek-V3 aux-loss-free: update expert bias after step.
                    try:
                        from forge.moe.moe import disable_dense_bypass, update_moe_biases
                        update_moe_biases(model)
                        # Disable dense_bypass after warmup so router activates.
                        warmup_steps = getattr(cfg, 'moe_dense_bypass_warmup_steps', 0)
                        if warmup_steps > 0 and step + 1 == warmup_steps:
                            disable_dense_bypass(model)
                    except Exception:
                        logger.debug("MoE dense_bypass disable failed (no-op for dense models)", exc_info=True)  # no-op for dense models
                accum_count = 0

                # ── FreeToken bandwidth recording + predictive offload (R&D 14) ──
                # Record bandwidth/VRAM samples for the predictive offload policy.
                # Every 10 steps, check if we should pre-emptively increase offload.
                if hasattr(optimizer, 'record_bandwidth_sample'):
                    vram_gb = torch.cuda.memory_allocated() / 1e9 if "cuda" in device else 0.0
                    optimizer.record_bandwidth_sample(vram_gb=vram_gb)
                    if step > 0 and step % 10 == 0 and hasattr(optimizer, 'should_preempt_offload'):
                        if optimizer.should_preempt_offload():
                            print("  [FreeToken] Predictive offload: VRAM trend suggests "
                                  "approaching limit, increasing offload")

                # ── Elastic grad_accum (R&D 14) ──
                # Dynamically adjust grad_accum based on VRAM pressure.
                # When VRAM is high, increase grad_accum (smaller effective batch
                # per step, but avoids OOM). When low, decrease for faster convergence.
                if args.elastic_grad_accum and "cuda" in device:
                    vram_gb = torch.cuda.memory_allocated() / 1e9
                    vram_limit = args.vram_limit_gb
                    vram_ratio = vram_gb / vram_limit if vram_limit > 0 else 0
                    if vram_ratio > 0.85 and grad_accum < args.grad_accum * 4:
                        grad_accum = min(grad_accum + 1, args.grad_accum * 4)
                        print(f"  [Elastic] VRAM {vram_gb:.2f}/{vram_limit}GB — "
                              f"increased grad_accum to {grad_accum}")
                    elif vram_ratio < 0.60 and grad_accum > args.grad_accum:
                        grad_accum = max(grad_accum - 1, args.grad_accum)
                        print(f"  [Elastic] VRAM {vram_gb:.2f}/{vram_limit}GB — "
                              f"decreased grad_accum to {grad_accum}")

                # EMA update.
                if ema_state is not None:
                    update_ema(ema_state, model, args.ema_decay)

                # Periodic save.
                if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                    # NLRQ: export STE masters to INT8 before snapshotting
                    if getattr(cfg, 'ffn_compression', 'none') == "nlrq":
                        from forge.training.runners.train_8b_all import export_nlrq_
                        export_nlrq_(model)
                    ckpt = step_checkpoint_path(args.save, step)
                    save_training_checkpoint(model, ckpt, optimizer=optimizer,
                                             ema_state=ema_state, step=step)
                    cleanup_step_checkpoints(args.save, args.keep_checkpoints)

                # NaN check.
                if step > 0 and step % 50 == 0 and has_nan_params(model):
                    print("NaN/Inf in params; emergency save + abort.")
                    emergency_save(model, args.save, "nan", step, optimizer=optimizer)
                    aborted = True
                    break

                # Logging.
                if step % 5 == 0 or step == args.max_steps - 1:
                    # R&D round 25: materialize the deferred loss scalar only
                    # at log intervals. This is the only point where we sync
                    # for loss reporting — the GPU pipeline runs uninterrupted
                    # between log intervals.
                    last_loss = last_loss_tensor.item()
                    vram_gb = torch.cuda.memory_allocated() / 1e9 if "cuda" in device else 0.0
                    msg = (f"Step {step+1}/{args.max_steps} | epoch {epoch} | "
                           f"loss {last_loss:.4f} | lr {lr:.2e} | "
                           f"vram {vram_gb:.2f} GB")
                    # FreeToken bandwidth stats
                    if hasattr(optimizer, 'bandwidth_stats') and optimizer.bandwidth_adaptive:
                        bw = optimizer.bandwidth_stats()
                        if 'b_p' in bw:
                            msg += f" | B_P={bw['b_p']}"
                    print(msg)
                    log.log(msg)
                    write_status_json(args.status_file, {
                        "step": step + 1, "max_steps": args.max_steps,
                        "loss": last_loss, "lr": lr, "epoch": epoch,
                    })
                    write_heartbeat(args.heartbeat_file)

                # ── Validation ──
                if val_dataset is not None and step > 0 and step % args.val_every == 0:
                    # R&D round 25: ensure last_loss is materialized for val
                    # logging (may not be set if val_every isn't a multiple of 5).
                    if step % 5 != 0:
                        last_loss = last_loss_tensor.item()
                    # Note: don't call model.eval() — BitNet eval mode uses fp32
                    # master weights which causes dtype mismatch with bf16 inputs.
                    # torch.inference_mode() is sufficient for validation.
                    val_losses = []
                    with torch.inference_mode():
                        n_val_batches = min(10, len(val_dataset) // args.batch_size)
                        for vi in range(n_val_batches):
                            vbatch = val_dataset[vi * args.batch_size:(vi + 1) * args.batch_size]
                            if not vbatch:
                                continue
                            vi_ids, vi_labels, vi_mask, _ = collate_batch(vbatch, pad_id, device)
                            vloss = compute_loss(model, vi_ids, vi_labels, vi_mask,
                                                 entropy_alpha=0.0,  # plain CE for eval
                                                 is_hf_model=args.hf_model is not None)
                            val_losses.append(vloss.item())
                    val_loss = sum(val_losses) / max(len(val_losses), 1)
                    print(f"  [Val] step {step+1}: val_loss={val_loss:.4f} "
                          f"(train_loss={last_loss:.4f})")
                    write_status_json(args.status_file, {
                        "step": step + 1, "max_steps": args.max_steps,
                        "loss": last_loss, "val_loss": val_loss,
                        "lr": lr, "epoch": epoch,
                    })
                    # Early stopping: patience-based (val loss not improving)
                    if args.early_stop_patience > 0:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            print(f"  [EarlyStop] val_loss not improving "
                                  f"({patience_counter}/{args.early_stop_patience})")
                            if patience_counter >= args.early_stop_patience:
                                print(f"  [EarlyStop] Stopping: val loss hasn't improved "
                                      f"for {args.early_stop_patience} checks.")
                                early_stopped = True
                                break
                # Early stopping: min train loss (format learned, stop before overfit)
                # Checked at every logging interval (every 5 steps), not just validation.
                if args.min_train_loss > 0 and step % 5 == 0:
                    if last_loss is not None and last_loss < args.min_train_loss:
                        print(f"  [EarlyStop] Train loss {last_loss:.4f} < "
                              f"threshold {args.min_train_loss} — format learned, stopping.")
                        early_stopped = True
                        break
                step += 1
            if aborted or early_stopped or step >= args.max_steps:
                break

        if not aborted:
            # If EMA enabled, restore EMA weights before saving (smoother model).
            if ema_state is not None:
                from forge.training.training_utils import restore_ema
                restore_ema(ema_state, model)
                print("Restored EMA weights for final save.")
            # If LoRA enabled, merge adapter weights into base model so the
            # saved checkpoint is a standalone full model (no LoRA dependency
            # needed for inference). This is the standard approach for
            # self-play loops where the next epoch loads from a plain checkpoint.
            # (--save-lora-adapter snapshots the un-merged LoRA tensors first
            #  so the adapter can be hot-loaded / swapped via ForgeEngine.)
            if args.save_lora_adapter and (use_manual_lora or args.lora):
                from safetensors.torch import save_file as _save_lora_file
                lora_state = {n: p.detach().cpu().contiguous()
                              for n, p in model.named_parameters()
                              if "lora_" in n}
                if lora_state:
                    lora_path = str(Path(args.save).with_suffix(".lora.safetensors"))
                    _save_lora_file(lora_state, lora_path)
                    print(f"Saved LoRA adapter ({len(lora_state)} tensors) "
                          f"to {lora_path}")
                else:
                    print("No LoRA tensors found — adapter save skipped.")
            if use_manual_lora:
                from forge.training.bitnet_lora import merge_lora_adapters
                n_merged = merge_lora_adapters(model)
                print(f"Merged {n_merged} manual LoRA adapters into base model for standalone save.")
            elif args.lora and hasattr(model, "merge_and_unload"):
                model = model.merge_and_unload()
                print("Merged LoRA adapters into base model for standalone save.")
            # NLRQ factor training: export STE masters to INT8 buffers before
            # saving so the checkpoint is pure-INT8 (loadable by ForgeEngine
            # without training-mode flags). Masters are stripped from state_dict.
            if getattr(cfg, 'ffn_compression', 'none') == "nlrq":
                from forge.training.runners.train_8b_all import export_nlrq_
                export_nlrq_(model)
                print("Exported NLRQ STE masters to INT8 buffers for standalone save.")
            save_training_checkpoint(model, args.save, optimizer=optimizer,
                                     ema_state=ema_state, step=step,
                                     meta={"config": cfg.__dict__, "sft": True,
                                           "n_examples": len(dataset)})
            print(f"\nSaved SFT model to {args.save}")
            print(f"  Trained {step} steps on {len(dataset)} examples")


if __name__ == "__main__":
    main()
