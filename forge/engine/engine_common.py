"""Shared module-level helpers for the ForgeEngine split modules.

Kept import-light (stdlib + torch only). forge_engine.py re-exports these
names so ``from forge.engine.forge_engine import _min_k_filter`` etc. keep
working.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch

logger = logging.getLogger(__name__)

_DEFAULT_CPU_MEMORY_BYTES = 32 * 1024**3

# Free-VRAM headroom required by the fast (fully in-VRAM) checkpoint load
# path: weights at ckpt size + ~30% for CUDA context, activations, and
# warmup. Shared by ForgeEngine.from_checkpoint (load-path selection) and
# the GUI pre-flight so the two checks can never drift apart again — a
# stale 2.5x multiplier in the GUI demanded ~16GB for the 6.4GB ForgeLM V2
# checkpoint and made every boot fail "not enough VRAM" on a 12GB card.
_FAST_LOAD_VRAM_HEADROOM = 1.3


def _fast_load_vram_required(ckpt_size: int) -> int:
    """Free-VRAM bytes the fast in-VRAM load path needs for a checkpoint of
    ``ckpt_size`` bytes."""
    return int(ckpt_size * _FAST_LOAD_VRAM_HEADROOM)


# EOS token IDs for all supported parent models:
#   LFM2.5: 7 (<|im_end|>), 151643/151645 (Qwen-style)
#   Jamba Reasoning 3B: 2 (<|endoftext|>), 519 (<|im_end|>)
_DEFAULT_EOS_TOKEN_IDS = frozenset({2, 7, 519, 151643, 151645})


def _is_memory_model(model) -> bool:
    """True for models whose state IS the memory (FluxLM) — no KV-cache
    forward semantics.  Replaces scattered
    ``model.__class__.__name__ == "FluxLM"`` sniffing: models declare the
    contract via ``supports_kv_cache = False`` and engine code checks the
    capability, so future memory-model architectures route correctly
    without another name check."""
    return not getattr(model, "supports_kv_cache", True)


@dataclass
class SamplingParams:
    """Bundle of per-request sampling/decoding settings.

    Collapses the ~15-argument parameter lists duplicated across
    ``generate``/``generate_stream``/``generate_raw``/``generate_batch``/
    ``_decode_tokens``/``_decode_with_kv``/``_finish_to_stop``.  Public
    methods keep accepting individual kwargs (call-site compat) and build
    one of these; internals pass the object so new sampling options touch
    one class instead of six signatures.
    """
    max_new_tokens: int = 100
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 80
    repetition_penalty: float = 1.05
    min_p: float = 0.0
    min_k: float = 0.0
    dry_multiplier: float = 0.0
    dry_base: float = 1.75
    dry_allowed_length: int = 2
    dry_penalty_last_n: int = 512
    stop: list[str] | None = None
    skip_special_tokens: bool = True
    learn: bool | None = None
    # recall<->generation dial (memory models only): scales the gcell/
    # gsyn channel lift weights.  None = model default (1.0); <1 leans
    # on verbatim recall, >1 leans on generalized generation.
    mem_bias: float | None = None

    @classmethod
    def from_kwargs(cls, **kw) -> "SamplingParams":
        """Build from a kwargs dict, ignoring keys that aren't fields."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in kw.items() if k in names})


def _min_k_filter(logits: torch.Tensor, sensitivity: float) -> torch.Tensor:
    """Min-k semantic-cliff sampling filter (ACL 2026, Ding et al.).

    Analyzes the local shape of the sorted logit distribution to identify
    "semantic cliffs" — sharp transitions from high-confidence core tokens
    to uncertain long-tail tokens. Computes a position-weighted relative
    decay rate to dynamically determine truncation boundaries at each step.

    Temperature-invariant: operates on relative logit dynamics, not absolute
    probabilities. Low sensitivity to hyperparameter choices.

    Args:
        logits: (batch, vocab) logit tensor.
        sensitivity: cliff detection sensitivity in [0, 1]. Higher = more
            aggressive truncation. 0 = no filtering.

    Returns:
        Filtered logits with long-tail tokens masked to -inf.
    """
    if sensitivity <= 0:
        return logits
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1)
    # Compute relative decay rate between consecutive sorted logits.
    # Avoid division by zero by adding small epsilon.
    eps = 1e-8
    diffs = sorted_logits[..., :-1] - sorted_logits[..., 1:]
    # Position weights: emphasize earlier positions (high-confidence core).
    # Weight decays linearly so cliffs near the top are more significant.
    n = diffs.shape[-1]
    weights = torch.linspace(1.0, 0.1, n, device=logits.device, dtype=logits.dtype)
    weighted_diffs = diffs * weights
    # Find the sharpest transition (semantic cliff).
    # The cliff is where the weighted decay rate exceeds sensitivity * max_decay.
    max_decay = weighted_diffs.max(dim=-1, keepdim=True).values.clamp(min=eps)
    cliff_mask = weighted_diffs > sensitivity * max_decay  # (batch, n)
    # Find the first cliff position (rightmost cliff that separates core from tail).
    # We want to keep all tokens up to and including the last cliff.
    # cliff_mask is True at positions where there's a sharp transition.
    # The truncation point is the position of the sharpest cliff.
    idx = torch.arange(n, device=logits.device).expand_as(cliff_mask)
    last_cliff = torch.where(
        cliff_mask, idx, torch.full_like(idx, -1)).amax(dim=-1, keepdim=True)
    # Fallback: no position exceeds the threshold (e.g. sensitivity=1)
    # — truncate at the single sharpest transition.
    cliff_pos = torch.where(
        last_cliff >= 0, last_cliff,
        weighted_diffs.argmax(dim=-1, keepdim=True))  # (batch, 1)
    # Keep tokens [0, cliff_pos], mask the rest.
    positions = torch.arange(sorted_logits.shape[-1], device=logits.device)
    keep = positions <= cliff_pos  # (batch, vocab)
    # Apply mask: set logits below threshold to -inf.
    # We need to map back from sorted to original indices.
    sorted_indices = torch.sort(logits, descending=True, dim=-1).indices
    # Create a mask in original index space.
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, ~keep)
    return logits.masked_fill(mask, float("-inf"))


def _dry_penalties(
    context_ids: list[int],
    last_n: int,
    allowed_length: int,
    multiplier: float,
    base: float,
) -> dict[int, float]:
    """DRY repetition penalty (llama.cpp ``sampler_dry``, p-e-w).

    Finds the longest suffix of ``context_ids`` that also ends at each
    earlier position ``i``; the token that would continue such a repeat
    (``context_ids[i+1]``) receives penalty
    ``multiplier * base ** (match_len - allowed_length)`` (logit-space
    subtraction, i.e. multiplicative in probability space).

    Args:
        context_ids: full token sequence so far (prompt + generated).
        last_n: only the trailing ``last_n`` positions are scanned as
            candidate match end-points (llama.cpp ``dry_penalty_last_n``).
        allowed_length: repeats of at most this length are free
            (llama.cpp default 2).
        multiplier: penalty scale; 0 disables the check.
        base: exponential growth per excess repeated token
            (llama.cpp default 1.75).

    Returns:
        ``{token_id: penalty}`` — apply ``logits[..., tok] -= penalty``.
    """
    n = len(context_ids)
    if multiplier <= 0 or n < 2 or last_n <= 0:
        return {}
    last_idx = n - 1
    start = max(0, last_idx - last_n)
    match_len: dict[int, int] = {}
    for i in range(start, last_idx):
        if context_ids[i] != context_ids[last_idx]:
            continue
        r = 1
        while i - r >= 0 and context_ids[i - r] == context_ids[last_idx - r]:
            r += 1
        if r > allowed_length:
            nxt = context_ids[i + 1]
            if r > match_len.get(nxt, 0):
                match_len[nxt] = r
    return {
        tok: multiplier * (base ** (r - allowed_length))
        for tok, r in match_len.items()
    }


def _apply_dry(
    next_logits: torch.Tensor, penalties: dict[int, float]
) -> torch.Tensor:
    """Subtract DRY penalties from a (vocab,) or (B, vocab) logit slice."""
    if not penalties:
        return next_logits
    toks = torch.tensor(list(penalties.keys()), device=next_logits.device)
    pens = torch.tensor(
        list(penalties.values()), device=next_logits.device, dtype=next_logits.dtype
    )
    next_logits.index_copy_(-1, toks, next_logits.index_select(-1, toks) - pens)
    return next_logits


# Module-level caches to avoid repeated disk I/O for checkpoint metadata and sizes.
# Keyed by (path, mtime) so stale entries are invalidated when the file changes.
# Bounded with LRU eviction to prevent unbounded growth in long-running servers.
# Thread-safe: guarded by _ckpt_cache_lock (forge_server runs generate() from
# multiple worker threads via SessionManager/BatchQueue).
from collections import OrderedDict as _OrderedDict

_CKPT_CACHE_MAX = 64
_checkpoint_metadata_cache: _OrderedDict[tuple[str, float], dict] = _OrderedDict()
_checkpoint_size_cache: _OrderedDict[str, int] = _OrderedDict()
_ckpt_cache_lock = threading.Lock()

# Tokenizer dispatch by config vocab size. The canonical LFM tokenizer
# (vocab 65536) is wrong for HF-family checkpoints; Qwen-family checkpoints
# (vocab 151936) need the Qwen tokenizer for correct encode/decode.
_QWEN_VOCAB = 151936
_QWEN_TOKENIZER_PATH = "Qwen/Qwen2.5-0.5B"


def _tokenizer_for_vocab(vocab_size: int | None) -> str:
    """Return the tokenizer path matching a config's vocabulary size."""
    if vocab_size == _QWEN_VOCAB:
        return _QWEN_TOKENIZER_PATH
    return "research/checkpoints/forgelm_v2_tokenizer"


if TYPE_CHECKING:
    from forge.engine.forge_engine import ForgeEngine


def _map_gguf_to_forge(name: str) -> str | None:
    """Map GGUF tensor names to Forge model state-dict keys.

    GGUF: blk.N.attn_q.weight, blk.N.ffn_gate.weight, token_embd.weight, etc.
    Forge: blocks.N.attn.q_proj.weight, blocks.N.ffn.w_gate.weight, embed.weight, etc.
    """
    # Token embeddings
    if name == "token_embd.weight":
        return "embed.weight"
    # Output head
    if name == "output.weight":
        return "head.weight"
    # Final norm
    if name == "output_norm.weight":
        return "ln_f.weight"
    # Per-block mappings
    if name.startswith("blk."):
        parts = name.split(".")
        layer_idx = parts[1]
        rest = ".".join(parts[2:])
        mapping = {
            "attn_norm.weight": f"blocks.{layer_idx}.ln1.weight",
            "attn_q.weight": f"blocks.{layer_idx}.attn.q_proj.weight",
            "attn_k.weight": f"blocks.{layer_idx}.attn.k_proj.weight",
            "attn_v.weight": f"blocks.{layer_idx}.attn.v_proj.weight",
            "attn_output.weight": f"blocks.{layer_idx}.attn.out_proj.weight",
            "ffn_norm.weight": f"blocks.{layer_idx}.ln2.weight",
            "ffn_gate.weight": f"blocks.{layer_idx}.ffn.w_gate.weight",
            "ffn_up.weight": f"blocks.{layer_idx}.ffn.w_up.weight",
            "ffn_down.weight": f"blocks.{layer_idx}.ffn.w_down.weight",
            # Gated attention variants
            "attn_q_norm.weight": f"blocks.{layer_idx}.attn.q_norm.weight",
            "attn_k_norm.weight": f"blocks.{layer_idx}.attn.k_norm.weight",
            "ffn_gate_inp.weight": f"blocks.{layer_idx}.ffn.gate.weight",
        }
        return mapping.get(rest)
    return None



class _ScalingModelAdapter:
    """Adapter wrapping a ``ForgeEngine`` in the string-based
    ``generate(prompt, **kwargs) -> str`` interface that the test-time
    scaling strategies (:class:`FirstFinishSearch`, :class:`BeamSearch`,
    :class:`MCTSDecoder`) and :class:`ModelCascade` expect.

    The scaling/cascade classes call ``model.generate(prompt,
    max_new_tokens=..., temperature=..., top_p=...)``.  ``ForgeEngine``
    already accepts those kwargs, but this adapter also:

    * forwards any default kwargs captured at construction time (so the
      caller can pin ``temperature`` / ``top_p`` once for all sub-calls),
    * exposes ``self.tokenizer`` so :class:`BeamSearch` can decode token
      ids when real log-probs are available,
    * exposes a no-op ``seed(s)`` so :class:`FirstFinishSearch` can
      attempt seeding without crashing.
    """

    def __init__(self, engine: "ForgeEngine", **default_kwargs):
        self._engine = engine
        self._default_kwargs = default_kwargs
        # Expose tokenizer for BeamSearch._token_to_str.
        self.tokenizer = getattr(engine, "tokenizer", None)

    def seed(self, s: int) -> None:
        """No-op seed hook (FFS calls this if present)."""
        pass

    def generate(self, prompt: str, **kwargs: object) -> str:
        merged = {**self._default_kwargs, **kwargs}
        return self._engine.generate(prompt, **merged)

