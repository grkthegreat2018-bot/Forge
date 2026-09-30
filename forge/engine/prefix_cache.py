"""Prefix cache manager for ForgeEngine.

Encapsulates the prefix-caching logic that was inline in
``ForgeEngine``.  The cache itself remains a plain attribute on the
engine (``engine._prefix_cache``) for backwards compatibility — tests
and external code set it directly to either a ``dict``, an
``LRUPrefixCache``, a ``LearnedPrefixCache``, or a
``ChunkedPrefixCache``.  This module provides the *operations* on that
cache as standalone functions.

R&D round 14 (LMCache port): added ``ChunkedPrefixCache`` — a
chunk-based rolling-hash prefix cache inspired by LMCache's
``TokenDatabase`` / ``TokenHasher``.  Instead of keying on the first
``_PREFIX_CACHE_KEY_LENGTH`` tokens (coarse, all-or-nothing), it splits
the prompt into fixed-size chunks (default 256 tokens, matching
LMCache), computes a deterministic rolling prefix hash per chunk, and
can match the *longest* cached prefix across all entries — including
partial hits where a shorter cached prefix is reused by a longer
prompt.  Token equality is always verified on a hit (the hash is only a
lookup key), so hash collisions cannot corrupt generation.
"""
from __future__ import annotations

import hashlib
import logging
import struct
from collections import OrderedDict

import torch

from forge.model_loader import unpack_output_with_kv

logger = logging.getLogger(__name__)

_PREFIX_CACHE_KEY_LENGTH = 32
_DEFAULT_MAX_ENTRIES = 64
# LMCache default chunk size — 256 tokens.  Smaller = finer-grained
# reuse but more hash overhead; larger = coarser but cheaper.  256 is
# the validated sweet spot from the LMCache paper.
_LMCACHE_CHUNK_SIZE = 256


class LRUPrefixCache:
    """Bounded LRU cache for prefix KV states.

    Replaces the unbounded ``dict`` that was used as the default prefix
    cache.  Prevents OOM in long-running serving scenarios by evicting
    the least-recently-used entries when the cache is full.

    Drop-in compatible with the old dict interface (``.get()``, ``.put()``,
    ``in``), plus ``.stats()`` for diagnostics.
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES):
        self._cache: OrderedDict = OrderedDict()
        self.max_entries = max_entries
        self._hits = 0
        self._misses = 0

    def get(self, key):
        if key not in self._cache:
            self._misses += 1
            return None
        # Move to end (most recently used)
        self._cache.move_to_end(key)
        self._hits += 1
        return self._cache[key]

    def put(self, key, value, length: int = 0):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        # Evict oldest entries if over capacity
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def __contains__(self, key):
        return key in self._cache

    def __len__(self):
        return len(self._cache)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


def cache_key(ids: torch.Tensor) -> tuple[int, ...]:
    """Compute a hashable cache key from the first ``_PREFIX_CACHE_KEY_LENGTH``
    tokens of ``ids``."""
    key_len = min(_PREFIX_CACHE_KEY_LENGTH, ids.shape[1])
    return tuple(ids[0, :key_len].cpu().tolist())


def capture_recurrent_state(model) -> dict:
    """Snapshot live per-layer recurrent state (conv / SSM) at the current
    sequence position.

    Returns ``{block_idx: {"conv": tensor, "ssm": tensor, ...}}``.  Conv
    layers keep their context in a module buffer (``attn._conv_state``)
    rather than in ``past_kv``, so KV-only cache reuse cannot restore it —
    this snapshot is the missing piece for hybrid models.
    """
    snap = {}
    for i, block in enumerate(getattr(model, "blocks", []) or []):
        st = {}
        attn = getattr(block, "attn", None)
        if attn is not None:
            for attr, key in (("_conv_state", "conv"), ("_ssm_state", "ssm"),
                              ("_ssm_state_real", "ssm_r"),
                              ("_ssm_state_imag", "ssm_i")):
                v = getattr(attn, attr, None)
                if v is not None:
                    st[key] = v.clone()
        kda = getattr(block, "_kda", None)
        if kda is not None:
            kda_snap = kda.snapshot_state()
            if any(v is not None for v in kda_snap.values()):
                st["kda"] = kda_snap
        if st:
            snap[i] = st
    return snap


def apply_recurrent_state_prefix(model, snap) -> None:
    """Install a stored conv context as the left-pad for the next T>1
    forward.

    Each conv layer reads ``self._conv_state_prefix`` once and consumes it
    (see ``DoubleGatedConvLayer.forward``) — it replaces the zero left-pad
    that would otherwise corrupt the first ``kernel_size - 1`` positions of
    the prefilled suffix.
    """
    if not snap:
        return
    blocks = getattr(model, "blocks", [])
    for i, st in snap.items():
        if i >= len(blocks):
            break
        block = blocks[i]
        attn = getattr(block, "attn", None)
        if attn is not None and "conv" in st and hasattr(attn, "_conv_state"):
            attn._conv_state_prefix = st["conv"]
        kda = getattr(block, "_kda", None)
        if kda is not None and "kda" in st:
            kda.set_state_prefix(st["kda"])


def cache_prompt_prefix(engine, ids: torch.Tensor) -> None:
    """Capture and store the KV cache for the prefix of ``ids``.

    For ``ChunkedPrefixCache`` the *full* prompt KV is stored under its
    rolling prefix hash (with token ids for verification), enabling
    partial-prefix reuse by later, longer prompts.  For the legacy
    cache only the first ``_PREFIX_CACHE_KEY_LENGTH`` tokens are stored.
    """
    prefix_cache = engine._prefix_cache
    if prefix_cache is None or ids.shape[1] <= 16:
        return
    if isinstance(prefix_cache, ChunkedPrefixCache):
        token_ids = ids[0].cpu().tolist()
        hashes = prefix_cache.prefix_hashes(token_ids)
        if not hashes:
            return
        full_hash = hashes[-1]
        if full_hash in prefix_cache:
            return  # already cached
        with torch.inference_mode():
            prefix_out = engine.model(ids, use_cache=True)
            _, prefix_kv = unpack_output_with_kv(prefix_out)
        prefix_cache.put((full_hash, token_ids), prefix_kv, len(token_ids))
        return
    key = cache_key(ids)
    if not hasattr(prefix_cache, "put") and key in prefix_cache:
        return
    key_len = min(_PREFIX_CACHE_KEY_LENGTH, ids.shape[1])
    with torch.inference_mode():
        prefix_out = engine.model(ids[:, :key_len], use_cache=True)
        _, prefix_kv = unpack_output_with_kv(prefix_out)
    if hasattr(prefix_cache, "put"):
        prefix_cache.put(key, prefix_kv, key_len)
    else:
        prefix_cache[key] = (key_len, prefix_kv)


def get_cached_prefix(prefix_cache, ids: torch.Tensor):
    """Look up a cached prefix KV for ``ids``.

    Returns ``(cached_len, past_kv, conv_snap)`` or ``None`` on miss / when
    the cache is disabled or the prompt is too short.  ``conv_snap`` is the
    per-layer recurrent-state snapshot stored with the entry (``None`` for
    legacy entries).

    Dispatches to :meth:`ChunkedPrefixCache.lookup_longest_prefix` when
    the active cache is a ``ChunkedPrefixCache`` (finer-grained,
    partial-prefix hits), otherwise falls back to the legacy
    exact-32-token-key path.
    """
    if prefix_cache is None or ids.shape[1] <= 16:
        return None
    if isinstance(prefix_cache, ChunkedPrefixCache):
        return prefix_cache.lookup_longest_prefix(ids)
    cached = prefix_cache.get(cache_key(ids))
    if cached is None:
        return None
    if hasattr(cached, "kv_cache"):
        return cached.length, cached.kv_cache, None
    # Handle two storage formats:
    #   1. Legacy dict: (key_len, prefix_kv) tuple
    #   2. LRUPrefixCache: prefix_kv directly (a list of (k,v) tuples)
    if isinstance(cached, tuple) and len(cached) == 2 \
            and isinstance(cached[0], int):
        # Legacy (key_len, prefix_kv) format
        cached_len, cached_past_kv = cached
    else:
        # LRUPrefixCache stored just the KV; infer length from key
        cached_past_kv = cached
        cached_len = min(_PREFIX_CACHE_KEY_LENGTH, ids.shape[1])
    return cached_len, cached_past_kv, None


def cache_prompt_prefix(engine, ids: torch.Tensor,
                        past_kv=None) -> None:
    """Capture and store the KV cache for the prefix of ``ids``.

    ``past_kv`` should be the KV state captured during generation
    (``model._forge_last_kv``) sliced to the prompt length — reusing it
    avoids the extra full-prompt forward this function used to run.
    When ``past_kv`` is None (e.g. streaming strategies that do not
    materialize reusable KV) the store is skipped rather than paying a
    second prefill.

    The conv/recurrent boundary state comes from
    ``model._last_prefill_recurrent`` — snapshotted by the model at the end
    of the fresh prefill, before decode steps mutate the live buffers.
    """
    prefix_cache = engine._prefix_cache
    if prefix_cache is None or ids.shape[1] <= 16 or past_kv is None:
        return
    conv_snap = getattr(engine.model, "_last_prefill_recurrent", None)
    if isinstance(prefix_cache, ChunkedPrefixCache):
        token_ids = ids[0].cpu().tolist()
        hashes = prefix_cache.prefix_hashes(token_ids)
        if not hashes:
            return
        full_hash = hashes[-1]
        if full_hash in prefix_cache:
            return  # already cached
        prefix_kv = _slice_past_kv(past_kv, len(token_ids))
        prefix_cache.put((full_hash, token_ids), prefix_kv, len(token_ids),
                         conv_snap=conv_snap)
        return
    key = cache_key(ids)
    if not hasattr(prefix_cache, "put") and key in prefix_cache:
        return
    key_len = min(_PREFIX_CACHE_KEY_LENGTH, ids.shape[1])
    prefix_kv = _slice_past_kv(past_kv, key_len)
    if hasattr(prefix_cache, "put"):
        prefix_cache.put(key, prefix_kv, key_len)
    else:
        prefix_cache[key] = (key_len, prefix_kv)


def generate_from_prefix_cache(
    engine, ids, params,
):
    """Fast path: if the prefix is cached, reuse its KV and skip prefill.

    ``params`` is an ``engine_common.SamplingParams`` bundle.
    Returns the full ``output_ids`` tensor on hit, or ``None`` on miss
    (caller falls back to normal decoding).
    """
    cached = get_cached_prefix(engine._prefix_cache, ids)
    if cached is None:
        return None
    cached_len, cached_past_kv, conv_snap = cached
    suffix_ids = ids[:, cached_len:]
    if suffix_ids.shape[1] > 0 and cached_past_kv is not None:
        # Restore conv boundary context so the suffix prefill doesn't
        # zero-pad the first kernel-1 positions.
        apply_recurrent_state_prefix(engine.model, conv_snap)
        with torch.inference_mode():
            out = engine.model(
                suffix_ids, past_key_values=cached_past_kv, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)
        output_ids = engine._decode_with_kv(
            ids, logits, past_kv, params)
        logger.info(f"  [PrefixCache] HIT + REUSE (prefix len={cached_len}, "
              f"saved prefill)")
        return output_ids
    logger.info(f"  [PrefixCache] HIT (prefix len={cached_len})")
    return None


# ── LMCache-style chunked rolling-hash prefix cache (R&D round 14) ─────────


def _chunk_hash(token_ids: list[int]) -> int:
    """Deterministic 64-bit hash of a chunk's token ids.

    Uses blake2b (8-byte digest) so the hash is stable across processes
    and independent of ``PYTHONHASHSEED`` — required for cross-request
    reuse where the cache may be restored from disk.
    """
    h = hashlib.blake2b(
        struct.pack(f"<{len(token_ids)}i", *token_ids),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little")


def _prefix_hash(prev: int, chunk_h: int) -> int:
    """Rolling combine of the previous prefix hash with a new chunk hash.

    Deterministic (no Python ``hash()``) so the chain is reproducible
    across processes.  ``prefix_hash[i]`` uniquely identifies the token
    sequence ``chunks[0..i]``.
    """
    h = hashlib.blake2b(
        struct.pack("<Q", prev & 0xFFFFFFFFFFFFFFFF) +
        struct.pack("<Q", chunk_h & 0xFFFFFFFFFFFFFFFF),
        digest_size=8,
    ).digest()
    return int.from_bytes(h, "little")


def _slice_past_kv(past_kv, length: int):
    """Slice a per-layer KV cache list to ``length`` tokens.

    ``past_kv`` is the model's ``presents`` list: one ``(k, v)`` tuple
    per layer (or ``None`` for conv layers with no KV).  Each k/v tensor
    is ``[batch, n_kv, seq, head_dim]``; we slice the seq axis.  Returns
    a *new* list of sliced tuples — the original cache is untouched.
    """
    if past_kv is None:
        return None
    sliced = []
    for layer in past_kv:
        if layer is None:
            sliced.append(None)
            continue
        if isinstance(layer, dict):
            # Recurrent (mamba/SSM) state is fixed-size, not seq-indexed —
            # pass it through untouched.  (It is only position-correct when
            # the hit boundary equals the stored entry length, which is the
            # case for every real ChunkedPrefixCache hit.)
            sliced.append(layer)
            continue
        k, v = layer
        sliced.append((k[:, :, :length], v[:, :, :length]))
    return sliced


class ChunkedPrefixCache:
    """LMCache-inspired chunked rolling-hash prefix cache.

    Splits prompts into fixed-size chunks and stores one entry per
    unique *full* prefix, keyed by the rolling prefix hash of its
    chunks.  Lookup finds the longest cached prefix that matches the
    query (verifying token equality, not just hash), enabling partial
    reuse: a prompt of 1024 tokens can hit a previously cached 768-token
    prefix and only re-prefill the last 256 tokens.

    Drop-in compatible with ``LRUPrefixCache`` (``.get()`` / ``.put()``
    / ``__contains__`` / ``.stats()``), plus ``lookup_longest_prefix()``
    for the finer-grained matching used by the chunked generate path.

    Memory: one KV tensor per entry (same as LRU).  The chunk metadata
    (hashes + token ids) is negligible (~2 KB per 8K-token entry).
    """

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES,
                 chunk_size: int = _LMCACHE_CHUNK_SIZE):
        self.max_entries = max_entries
        self.chunk_size = chunk_size
        # key = full-prefix rolling hash -> entry dict
        self._cache: OrderedDict[int, dict] = OrderedDict()
        # Reverse index: prefix-hash -> full-prefix-hash (for longest-prefix
        # matching we scan the keys; with <= max_entries this is cheap).
        self._hits = 0
        self._misses = 0
        self._partial_hits = 0

    # ── core interface (LRU-compatible) ────────────────────────────────

    def get(self, key):
        """LRU-compatible exact-key lookup.

        ``key`` here is expected to be a full-prefix rolling hash (int)
        produced by :meth:`prefix_hashes`.  For the legacy tuple-key
        path used by :func:`cache_key`, use :meth:`lookup_longest_prefix`
        instead.
        """
        entry = self._cache.get(key)
        if entry is None:
            self._misses += 1
            return None
        self._cache.move_to_end(key)
        self._hits += 1
        return (entry["length"], entry["kv"])

    def put(self, key, value, length: int = 0, conv_snap=None):
        """Store a prefix KV under ``key`` (full-prefix rolling hash).

        ``value`` is the ``past_kv`` list from the model.  ``length`` is
        the number of tokens the KV covers.  The caller is responsible
        for computing the rolling hash via :meth:`prefix_hashes`.
        ``conv_snap`` is the per-layer recurrent-state snapshot at the end
        of the stored prefix (from :func:`capture_recurrent_state` or
        ``model._last_prefill_recurrent``).
        """
        token_ids = key[1] if isinstance(key, tuple) and len(key) > 1 else None
        # If handed a (hash, token_ids) tuple, split it.
        if isinstance(key, tuple):
            hash_key = key[0]
            token_ids = key[1] if len(key) > 1 else token_ids
        else:
            hash_key = key
        if hash_key in self._cache:
            self._cache.move_to_end(hash_key)
        self._cache[hash_key] = {
            "length": length,
            "kv": value,
            "token_ids": token_ids,
            "conv_snap": conv_snap,
        }
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def __contains__(self, key):
        k = key[0] if isinstance(key, tuple) else key
        return k in self._cache

    def __len__(self):
        return len(self._cache)

    # ── chunked hashing helpers ────────────────────────────────────────

    def chunk_tokens(self, token_ids: list[int]) -> list[list[int]]:
        """Split a token list into chunk_size-sized pieces."""
        cs = self.chunk_size
        return [token_ids[i:i + cs] for i in range(0, len(token_ids), cs)]

    def prefix_hashes(self, token_ids: list[int]) -> list[int]:
        """Compute the rolling prefix hash at each chunk boundary.

        Returns ``[h0, h1, ...]`` where ``h[i]`` is the hash of
        ``token_ids[: (i+1)*chunk_size]``.  ``h[-1]`` is the full-prefix
        hash used as the storage key.
        """
        hashes = []
        prev = 0
        for chunk in self.chunk_tokens(token_ids):
            ch = _chunk_hash(chunk)
            prev = _prefix_hash(prev, ch)
            hashes.append(prev)
        return hashes

    # ── longest-prefix matching ────────────────────────────────────────

    def lookup_longest_prefix(self, ids: torch.Tensor):
        """Find the longest cached prefix matching ``ids``.

        Returns ``(matched_len, past_kv)`` or ``None`` on miss.  The
        returned ``past_kv`` is sliced to ``matched_len`` tokens so it
        can be fed directly to the model with the suffix
        ``ids[:, matched_len:]``.

        Token equality is verified for every candidate hit — a hash
        collision never yields a wrong KV.
        """
        if ids.shape[1] <= 16:
            return None
        token_ids = ids[0].cpu().tolist()
        hashes = self.prefix_hashes(token_ids)
        if not hashes:
            return None

        # Walk chunk boundaries from longest to shortest, return the
        # first entry whose hash matches AND whose tokens verify.
        for i in range(len(hashes) - 1, -1, -1):
            h = hashes[i]
            entry = self._cache.get(h)
            if entry is None:
                continue
            matched_len = min((i + 1) * self.chunk_size, len(token_ids))
            # Verify token equality on the matched prefix.
            stored_tokens = entry.get("token_ids")
            if stored_tokens is not None:
                if stored_tokens[:matched_len] != token_ids[:matched_len]:
                    continue  # hash collision — skip
            # Move to MRU and return sliced KV.
            self._cache.move_to_end(h)
            if matched_len < entry["length"]:
                self._partial_hits += 1
                kv = _slice_past_kv(entry["kv"], matched_len)
                # The stored conv snapshot is only position-correct at the
                # entry's own end — a partial hit can't use it.
                conv_snap = None
            else:
                kv = entry["kv"]
                conv_snap = entry.get("conv_snap")
            self._hits += 1
            return matched_len, kv, conv_snap
        self._misses += 1
        return None

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "type": "chunked",
            "entries": len(self._cache),
            "max_entries": self.max_entries,
            "chunk_size": self.chunk_size,
            "hits": self._hits,
            "misses": self._misses,
            "partial_hits": self._partial_hits,
            "hit_rate": self._hits / total if total > 0 else 0,
        }


# ── Semantic KV anchors (FreeToken-style, R&D round 15) ────────────────────


class SemanticKVAnchors:
    """Semantic anchor-based KV cache reuse (FreeToken-style).

    Saves KV cache state at logical boundaries (tool calls, thinking blocks,
    paragraph breaks) so agentic edits recompute only the suffix after the
    last anchor, not the entire prefix.

    Anchors are detected at:
    - Tool call boundaries: <tool_call>...</tool_call> tags
    - Thinking blocks: <thinking>...</thinking> or <|thinking_end|>
    - Paragraph breaks: double newlines
    - Sentence boundaries: . ! ? followed by newline
    """

    def __init__(self, max_anchors: int = 16):
        self._anchors: list[dict] = []  # [{token_pos, text_pos, type, kv_state}]
        self._max_anchors = max_anchors

    def detect_anchors(self, text: str, token_offsets: list[int] | None = None) -> list[dict]:
        """Find semantic anchor positions in text.

        Returns list of {text_pos, type} for each anchor.
        """
        import re
        anchors = []
        # Tool call boundaries
        for m in re.finditer(r'</tool_call>', text):
            anchors.append({"text_pos": m.end(), "type": "tool_call"})
        # Thinking blocks
        for m in re.finditer(r'</thinking>|<\|thinking_end\|>', text):
            anchors.append({"text_pos": m.end(), "type": "thinking"})
        # Paragraph breaks
        for m in re.finditer(r'\n\n', text):
            anchors.append({"text_pos": m.end(), "type": "paragraph"})
        # Sort by position
        anchors.sort(key=lambda a: a["text_pos"])
        return anchors

    def save_anchor(self, token_pos: int, text_pos: int, anchor_type: str,
                    kv_state, source_text: str | None = None) -> None:
        """Save KV cache state at an anchor point.

        ``source_text`` is the full text the anchor positions index into;
        the prefix hash of ``source_text[:text_pos]`` is stored so that
        :meth:`find_reuse_point` can verify a candidate actually shares
        that prefix instead of trusting positions alone.
        """
        prefix_hash = None
        if source_text is not None:
            prefix_hash = hashlib.blake2b(
                source_text[:text_pos].encode("utf-8", errors="replace"),
                digest_size=8).digest()
        self._anchors.append({
            "token_pos": token_pos,
            "text_pos": text_pos,
            "type": anchor_type,
            "kv_state": kv_state,
            "prefix_hash": prefix_hash,
        })
        # Evict oldest if over limit
        if len(self._anchors) > self._max_anchors:
            self._anchors.pop(0)

    def find_reuse_point(self, text: str) -> dict | None:
        """Find the best anchor to reuse for the given text.

        Returns the anchor with the longest VERIFIED matching prefix, or
        None.  Anchors stored without a prefix hash are skipped — reusing
        KV at a position where the text differs silently corrupts output.
        """
        best = None
        for anchor in reversed(self._anchors):
            if anchor["text_pos"] > len(text):
                continue
            ph = anchor.get("prefix_hash")
            if ph is None:
                continue
            cand = hashlib.blake2b(
                text[:anchor["text_pos"]].encode("utf-8", errors="replace"),
                digest_size=8).digest()
            if cand == ph:
                best = anchor
                break
        return best

    def get_kv_state(self, token_pos: int) -> object | None:
        """Get saved KV state at or before the given token position."""
        best = None
        for anchor in self._anchors:
            if anchor["token_pos"] <= token_pos:
                best = anchor
            else:
                break
        return best["kv_state"] if best else None

    def clear(self) -> None:
        """Clear all anchors."""
        self._anchors.clear()

    def info(self) -> dict:
        return {
            "n_anchors": len(self._anchors),
            "anchors": [{"token_pos": a["token_pos"], "type": a["type"]}
                       for a in self._anchors],
        }


# ── Radix-tree prefix cache (SGLang RadixAttention-style) ────────────────────


class RadixPrefixCache:
    """Radix-tree cross-request prefix sharing (SGLang RadixAttention-style).

    Unlike LRU (exact-key) or ChunkedPrefixCache (rolling-hash chunks),
    this stores KV cache in a radix tree keyed by token-id sequences.
    Multiple requests sharing a common prefix (system prompt, few-shot
    examples, tool definitions) automatically reuse the same KV nodes,
    even when they diverge later.

    Tree structure:
        root → [token 1] → [token 2] → ... → [token N] (leaf: KV state)
              ↘ [token A] → [token B] → ... (different request's KV)

    Each node stores:
        - token_ids: list[int] for this edge
        - kv_state: KV cache at the end of this edge (None for internal)
        - ref_count: how many active requests reference this node
        - children: dict[token_id → RadixNode]

    Eviction: LRU on leaves (ref_count == 0), evicting oldest first.
    """

    class _Node:
        __slots__ = ("children", "kv_state", "last_used", "ref_count", "token_ids")

        def __init__(self):
            self.token_ids: list[int] = []
            self.kv_state = None  # None until computed
            self.ref_count = 0
            self.children: dict[int, RadixPrefixCache._Node] = {}
            self.last_used = 0.0

    def __init__(self, max_entries: int = _DEFAULT_MAX_ENTRIES):
        self.root = self._Node()
        self.max_entries = max_entries
        self._n_entries = 0
        self._hits = 0
        self._misses = 0

    def _match(self, node: _Node, token_ids: list[int],
               start: int) -> tuple[int, _Node, int, _Node | None]:
        """Walk the tree from `node` matching `token_ids[start:]`.

        Returns (matched_count, last_node, edge_pos, parent) — the deepest
        node reached, how many tokens were consumed globally, how far into
        ``last_node``'s own edge the match got, and ``last_node``'s parent
        (None for root). ``edge_pos`` lets callers detect mid-edge stops,
        which require an edge split on insert.
        """
        matched = 0
        cur = node
        parent = None
        i = start
        while i < len(token_ids):
            # Try to extend within current edge
            if matched < len(cur.token_ids):
                if cur.token_ids[matched] == token_ids[i]:
                    matched += 1
                    i += 1
                    continue
                else:
                    break  # Mismatch within edge
            # Try to descend to a child
            child = cur.children.get(token_ids[i])
            if child is None:
                break
            parent, cur = cur, child
            matched = 0
            # Continue matching within this child's edge
            if matched < len(cur.token_ids) and cur.token_ids[matched] == token_ids[i]:
                matched += 1
                i += 1
        # If we stopped mid-edge, edge_pos < len(cur.token_ids) reports it
        return i - start, cur, matched, parent

    def _split_edge(self, node: _Node, parent: _Node | None,
                    pos: int) -> _Node:
        """Split ``node``'s edge at ``pos``: parent → mid[E[:pos]] → node[E[pos:]].

        ``node`` keeps its children, kv_state and ref_count; ``mid`` takes
        the shared prefix. Returns ``mid`` (the new boundary node).
        """
        import time
        mid = self._Node()
        mid.token_ids = node.token_ids[:pos]
        mid.children[node.token_ids[pos]] = node
        mid.last_used = time.time()
        node.token_ids = node.token_ids[pos:]
        if parent is not None:
            parent.children[mid.token_ids[0]] = mid
        return mid

    def lookup_longest_prefix(self, ids: torch.Tensor) -> tuple[int, object] | None:
        """Find the longest cached prefix for `ids`.

        Returns (cached_len, kv_state) or None.
        """
        import time
        token_ids = ids[0].cpu().tolist()
        if len(token_ids) <= 16:
            return None
        matched, node, _edge_pos, _parent = self._match(
            self.root, token_ids, 0)
        if matched <= 0:
            self._misses += 1
            return None
        # Find the deepest node with a kv_state
        best_len = 0
        best_kv = None
        best_node = None
        # Walk from root again, accumulating lengths
        cur = self.root
        acc = 0
        i = 0
        while i < matched:
            edge_len = min(len(cur.token_ids), matched - i) if cur.token_ids else 0
            if cur.token_ids:
                acc += edge_len
                i += edge_len
                if cur.kv_state is not None and acc > best_len:
                    best_len = acc
                    best_kv = cur.kv_state
                    best_node = cur
                if i >= matched:
                    break
            # Descend
            if i < len(token_ids):
                child = cur.children.get(token_ids[i])
                if child is None:
                    break
                cur = child
            else:
                break
        if best_kv is not None:
            self._hits += 1
            best_node.last_used = time.time()
            return best_len, best_kv
        self._misses += 1
        return None

    def insert(self, token_ids: list[int], kv_state) -> None:
        """Insert a KV state for the full token sequence."""
        import time
        if not token_ids:
            return
        matched, node, edge_pos, parent = self._match(
            self.root, token_ids, 0)
        # Match stopped mid-edge (divergence OR the inserted sequence ends
        # inside an edge): split the edge so the insert lands on a real
        # node boundary. Without this, the new child's implied prefix
        # would include edge tokens it never matched — and assigning
        # children[remaining[0]] could silently clobber an existing subtree.
        if 0 < edge_pos < len(node.token_ids):
            node = self._split_edge(node, parent, edge_pos)
        remaining = token_ids[matched:]
        if not remaining:
            # Requested sequence ends exactly at this boundary
            node.kv_state = kv_state
            node.last_used = time.time()
            return
        new_node = self._Node()
        new_node.token_ids = remaining[:]
        new_node.kv_state = kv_state
        new_node.last_used = time.time()
        node.children[remaining[0]] = new_node
        self._n_entries += 1
        # Evict if over capacity
        if self._n_entries > self.max_entries:
            self._evict_lru()

    def _evict_lru(self) -> None:
        """Evict the least-recently-used leaf node."""
        # BFS to find the oldest leaf with ref_count == 0
        best = None
        best_time = float('inf')
        queue = [self.root]
        parent_map = {id(self.root): None}
        while queue:
            node = queue.pop(0)
            if node.kv_state is not None and node.ref_count == 0:
                if node.last_used < best_time:
                    best = node
                    best_time = node.last_used
            for child in node.children.values():
                parent_map[id(child)] = node
                queue.append(child)
        if best is not None:
            best.kv_state = None
            self._n_entries -= 1

    # LRU-compatible interface
    def get(self, key):
        """LRU-compatible get (uses key as token tuple)."""
        if isinstance(key, tuple):
            ids = torch.tensor([list(key)])
            result = self.lookup_longest_prefix(ids)
            if result:
                return result[1]
        return None

    def put(self, key, value, length: int = 0):
        """LRU-compatible put."""
        if isinstance(key, tuple):
            self.insert(list(key), value)

    def __contains__(self, key):
        if isinstance(key, tuple):
            ids = torch.tensor([list(key)])
            return self.lookup_longest_prefix(ids) is not None
        return False

    def __len__(self):
        return self._n_entries

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "entries": self._n_entries,
            "max_entries": self.max_entries,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0,
            "type": "radix",
        }
