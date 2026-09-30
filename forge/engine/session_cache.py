"""Session-aware KV cache management for ForgeEngine.

Layer map (three similarly-named modules):
  - session_cache.py  — per-session KV reuse *inside* the engine
    (SessionCacheManager, radix prefix match + TTL). This file.
  - engine_sessions.py — the engine-facing mixin that exposes it as
    begin_session()/continue_session().
  - session_manager.py — server-side conversation history + request
    batching queue (BatchQueue/BatchedDecoding driver). Different layer.

Provides:
  - SessionCacheManager: per-session KV cache with radix tree prefix matching
  - KV cache TTL: pin KV cache during tool calls / pauses, auto-evict on expiry
  - Multi-turn optimization: O(Δt) per turn instead of O(n) re-prefill

Based on SGLang's Unified Radix Cache + Continuum's KV TTL + AgServe's
session-aware eviction.

Usage (automatic via ForgeEngine):
    engine = ForgeEngine.from_checkpoint(...)

    # Session-aware multi-turn generation:
    session = engine.begin_session("chat_001")
    engine.continue_session("chat_001", "Hello, how are you?")
    engine.continue_session("chat_001", "Tell me about Python.")
    engine.end_session("chat_001")

    # Or use generate() which auto-uses radix tree prefix matching:
    engine.generate("Hello, how are you?")
    engine.generate("Hello, how are you? Tell me about Python.")  # prefix hit!
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from forge.engine.forge_engine import ForgeEngine


def _kv_seq_len(past_kv) -> int | None:
    """Sequence length covered by a per-layer KV cache, or None if the
    cache has no length-bearing (attention) layers."""
    if past_kv is None:
        return None
    for layer in past_kv:
        if layer is None or isinstance(layer, dict):
            continue
        try:
            return layer[0].shape[2]  # (B, n_kv, seq, head_dim)
        except (TypeError, IndexError):
            continue
    return None


@dataclass
class SessionState:
    """Per-session KV cache state."""
    session_id: str
    token_ids: list[int] = field(default_factory=list)
    past_kv: Any = None  # KV cache tensor from model
    recurrent_state: Any = None  # conv/SSM snapshot (prefix_cache.capture_recurrent_state)
    created_at: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    ttl: float | None = None  # seconds until auto-eviction (None = no TTL)
    ttl_expires_at: float | None = None
    pinned: bool = False  # if True, won't be evicted

    @property
    def is_expired(self) -> bool:
        if self.ttl_expires_at is None:
            return False
        return time.time() > self.ttl_expires_at

    def refresh_ttl(self, ttl: float | None = None):
        """Reset the TTL countdown."""
        if ttl is not None:
            self.ttl = ttl
        if self.ttl is not None:
            self.ttl_expires_at = time.time() + self.ttl
        else:
            self.ttl_expires_at = None

    def touch(self):
        self.last_access = time.time()


class SessionCacheManager:
    """Manages per-session KV cache with radix tree prefix matching + TTL.

    Features:
      - Session-aware: each session maintains its own KV + recurrent state.
        When a session continues, only the delta (new tokens) needs
        prefilling.  (Cross-session prefix sharing lives in the separate
        ``ChunkedPrefixCache``/``RadixPrefixCache`` — this manager is
        strictly per-session.)
      - TTL: when a session pauses (e.g. tool call), pin its KV cache with
        a time-to-live. Auto-evict when TTL expires.
      - LRU eviction: when GPU memory is tight, evict least-recently-used
        sessions first (unless pinned or within TTL).
    """

    def __init__(self, engine: ForgeEngine,
                 max_sessions: int = 32,
                 default_ttl: float | None = None,
                 eviction_check_interval: int = 8):
        self.engine = engine
        self.max_sessions = max_sessions
        self.default_ttl = default_ttl
        self.eviction_check_interval = eviction_check_interval
        self._sessions: dict[str, SessionState] = {}
        self._access_count = 0
        self._prefix_hits = 0
        self._prefix_misses = 0

    def begin_session(self, session_id: str,
                      ttl: float | None = None) -> SessionState:
        """Start a new session with optional TTL.

        Args:
            session_id: unique session identifier.
            ttl: time-to-live in seconds. If set, the session's KV cache
                will be auto-evicted after this many seconds of inactivity.
                None means no TTL (persists until explicitly ended or LRU-evicted).
        """
        if session_id in self._sessions:
            # Session already exists — refresh it
            session = self._sessions[session_id]
            session.touch()
            session.refresh_ttl(ttl or self.default_ttl)
            return session

        # Evict expired sessions if needed
        self._maybe_evict()

        session = SessionState(
            session_id=session_id,
            ttl=ttl or self.default_ttl,
        )
        session.refresh_ttl()
        self._sessions[session_id] = session

        self.engine._log(f"Session started: {session_id}",
                         source="session", session_id=session_id)
        return session

    def continue_session(self, session_id: str,
                         prompt: str) -> tuple[torch.Tensor, Any, int]:
        """Continue a session with a new prompt.

        Returns:
            token_ids: full token IDs (session history + new prompt)
            past_kv: cached KV state (from previous turns + prefix match)
            cached_len: number of tokens that hit the cache (skip prefill)
            recurrent_state: conv/SSM snapshot for boundary restoration
        """
        if session_id not in self._sessions:
            # Auto-create session if it doesn't exist
            self.begin_session(session_id)

        session = self._sessions[session_id]
        session.touch()
        session.refresh_ttl()

        # Tokenize the new prompt
        new_ids = self.engine.tokenizer(
            prompt, return_tensors="pt",
            add_special_tokens=False).input_ids[0].tolist()

        # Build full token sequence: previous session tokens + new tokens
        if session.token_ids:
            full_ids = session.token_ids + new_ids
        else:
            full_ids = new_ids

        # Check for prefix cache hit
        cached_len = 0
        past_kv = session.past_kv
        recurrent = session.recurrent_state

        if past_kv is not None and session.token_ids:
            # Verify the stored KV actually covers all prior tokens — an
            # evicting KV strategy (paged/snapkv) may have shrunk it.
            kv_len = _kv_seq_len(past_kv)
            if kv_len is not None and kv_len != len(session.token_ids):
                self._prefix_misses += 1
                self.engine._log(
                    f"Session KV length mismatch: {kv_len} != "
                    f"{len(session.token_ids)} tokens — full prefill",
                    source="session", level="warn")
                past_kv = None
                recurrent = None
            else:
                # We have cached KV from previous turns — only prefill the delta
                cached_len = len(session.token_ids)
                self._prefix_hits += 1
                self.engine._log(
                    f"Session prefix hit: {cached_len} tokens cached, "
                    f"{len(new_ids)} new tokens to prefill",
                    source="session", level="profile")
        else:
            self._prefix_misses += 1

        # Update session state
        session.token_ids = full_ids

        ids_tensor = torch.tensor([full_ids], device=self.engine.device)
        return ids_tensor, past_kv, cached_len, recurrent

    def update_session_kv(self, session_id: str, past_kv: Any,
                          recurrent_state: Any = None):
        """Update the session's cached KV + recurrent state after generation."""
        if session_id in self._sessions:
            self._sessions[session_id].past_kv = past_kv
            self._sessions[session_id].recurrent_state = recurrent_state
            self._sessions[session_id].touch()

    def pin_session(self, session_id: str, ttl: float | None = None):
        """Pin a session's KV cache (prevent eviction).

        Used when a session pauses for a tool call — pin with a TTL so
        the KV cache survives the pause but auto-evicts if the tool
        takes too long.
        """
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.pinned = True
            session.refresh_ttl(ttl or 30.0)  # default 30s TTL for tool calls
            self.engine._log(f"Session pinned: {session_id} (TTL={session.ttl}s)",
                             source="session", session_id=session_id)

    def unpin_session(self, session_id: str):
        """Unpin a session (allow eviction again)."""
        if session_id in self._sessions:
            self._sessions[session_id].pinned = False
            self._sessions[session_id].refresh_ttl(self.default_ttl)

    def end_session(self, session_id: str):
        """End a session and release its KV cache."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self.engine._log(f"Session ended: {session_id}",
                             source="session", session_id=session_id)

    def _maybe_evict(self):
        """Evict expired and LRU sessions."""
        self._access_count += 1
        if self._access_count % self.eviction_check_interval != 0:
            return

        # Evict expired sessions
        expired = [sid for sid, s in self._sessions.items()
                   if s.is_expired and not s.pinned]
        for sid in expired:
            del self._sessions[sid]
            self.engine._log(f"Session expired (TTL): {sid}",
                             source="session", level="warn")

        # Evict LRU if over capacity
        if len(self._sessions) > self.max_sessions:
            sorted_sessions = sorted(
                self._sessions.items(),
                key=lambda x: x[1].last_access)
            while len(self._sessions) > self.max_sessions:
                sid, session = sorted_sessions.pop(0)
                if not session.pinned:
                    del self._sessions[sid]
                    self.engine._log(f"Session evicted (LRU): {sid}",
                                     source="session", level="warn")

    def get_session(self, session_id: str) -> SessionState | None:
        return self._sessions.get(session_id)

    def stats(self) -> dict:
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "prefix_hits": self._prefix_hits,
            "prefix_misses": self._prefix_misses,
            "hit_rate": (self._prefix_hits /
                         max(self._prefix_hits + self._prefix_misses, 1)),
            "sessions": {
                sid: {
                    "tokens": len(s.token_ids),
                    "has_kv": s.past_kv is not None,
                    "pinned": s.pinned,
                    "ttl": s.ttl,
                    "age_s": time.time() - s.created_at,
                }
                for sid, s in self._sessions.items()
            },
        }

    def clear(self):
        """Clear all sessions and release KV cache tensors."""
        for session in self._sessions.values():
            session.past_kv = None
            session.recurrent_state = None
        self._sessions.clear()
        self._prefix_hits = 0
        self._prefix_misses = 0

    def __del__(self):
        try:
            self.clear()
        except Exception:
            logger.debug("Error during SessionCacheManager cleanup", exc_info=True)
