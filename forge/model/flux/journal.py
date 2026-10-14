"""Append-only provenance journal for FluxLM writes."""
from __future__ import annotations

import numpy as np

class FluxJournal:
    """Append-only provenance log for every weight mutation.

    Entry kinds (all tuples starting with a 1-char op):
      ("t", tok, tag)                 token marker (position in stream)
      ("b", ep_start, c_fp16)         episode boundary (soft_reset)
      ("u", tok)                      unigram count +1
      ("c", order, key, tok, delta)   table cell write
      ("e", key, pos)                 episodic index insert
      ("r", tok, prev_pos_or_-1)      recency last_seen write
      ("a", tok, vec_slot_or_-1)      topic row += delta (fp16 in vec ring)
      ("h", w_slot_or_-1)             hedge update - pre-update w vector
                                      snapshotted fp64 in w_ring (clip
                                      makes a multiplicative inverse
                                      lossy, so we store the vector)
      ("h2", regime, w_slot_or_-1)    cond_hedge variant - the snapshot
                                      targets _wR[regime]/_hmxR[regime]
    """

    __slots__ = ("entries", "cap", "dropped", "vec_ring", "vec_cursor",
                 "vec_cap", "w_ring", "w_cursor", "w_cap",
                 "block_bytes", "block_cap", "dropped_blocks")

    def __init__(self, cap: int, vec_cap: int, topic_dim: int,
                 n_chan: int, block_cap: int = 1 << 30):
        self.entries: list[tuple] = []
        self.cap = cap
        self.dropped = 0
        # bulk-block payload budget: "bc" entries carry per-block undo
        # tensors + sub-journals (~50-200MB each) - counting them as
        # one entry made `cap` blind to bytes.  `block_cap` bounds the
        # retained block payload; eviction is prefix-order (oldest
        # first), the same semantics revert already tolerates.
        self.block_bytes = 0
        self.block_cap = block_cap
        self.dropped_blocks = 0
        self.vec_ring = np.zeros((vec_cap, topic_dim), dtype=np.float16)
        self.vec_cursor = 0
        self.vec_cap = vec_cap
        # +1 column: AdaHedge cumulative mix-gap rides alongside w so
        # undo restores the whole hedge state exactly
        self.w_ring = np.zeros((vec_cap, n_chan + 1), dtype=np.float64)
        self.w_cursor = 0
        self.w_cap = vec_cap

    def append(self, entry: tuple) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.cap:
            # drop oldest chunk (amortized) - revert horizon shrinks
            for e in self.entries[: self.cap // 10]:
                if e[0] == "bc":
                    self.block_bytes -= _block_bytes(e[1])
                    self.dropped_blocks += 1
            del self.entries[: self.cap // 10]
            self.dropped += self.cap // 10

    def seal_bc(self, blk: dict) -> int:
        """A bulk-ingest block finished collecting: account its payload
        and prefix-evict oldest entries until the block byte budget
        holds.  Returns the block's estimated bytes."""
        n = _block_bytes(blk)
        self.block_bytes += n
        while self.entries and self.block_bytes > self.block_cap:
            e = self.entries[0]
            if e[0] == "bc":
                self.block_bytes -= _block_bytes(e[1])
                self.dropped_blocks += 1
            del self.entries[0]
            self.dropped += 1
        return n

    def pop_tail(self) -> tuple:
        """entries.pop() with block accounting (undo paths)."""
        e = self.entries.pop()
        if e[0] == "bc":
            self.block_bytes -= _block_bytes(e[1])
        return e

    def drop_range(self, lo: int, hi: int) -> None:
        """del entries[lo:hi] with block accounting (undo paths)."""
        for e in self.entries[lo:hi]:
            if e[0] == "bc":
                self.block_bytes -= _block_bytes(e[1])
        del self.entries[lo:hi]

    def recompute(self) -> None:
        """Rebuild block_bytes after load() swapped in entries."""
        self.block_bytes = sum(
            _block_bytes(e[1]) for e in self.entries if e[0] == "bc")

    def append_vec(self, tok: int, delta: np.ndarray) -> tuple:
        slot = -1
        if self.vec_cap:
            slot = self.vec_cursor
            self.vec_ring[slot] = delta.astype(np.float16)
            self.vec_cursor = (self.vec_cursor + 1) % self.vec_cap
        return ("a", tok, slot)

    def append_w(self, w: np.ndarray, mix: float = 0.0,
                 regime: int = -1) -> tuple:
        slot = -1
        if self.w_cap:
            slot = self.w_cursor
            self.w_ring[slot, :w.size] = w
            self.w_ring[slot, w.size] = mix
            self.w_cursor = (self.w_cursor + 1) % self.w_cap
        return ("h2", regime, slot) if regime >= 0 else ("h", slot)

    def __len__(self) -> int:
        return len(self.entries)


def _block_bytes(blk: dict) -> int:
    """Rough retained-payload estimate for a "bc" block: tensor/array
    nbytes + container overhead for sub-journal tuples, epi pairs, and
    dict snapshots.  A heuristic bound, not exact accounting."""
    total = 512
    stack = [blk]
    while stack:
        x = stack.pop()
        nb = getattr(x, "nbytes", None)
        if nb is not None:
            try:
                total += int(nb)
                continue
            except Exception:
                pass
        if isinstance(x, dict):
            total += 232 + 96 * len(x)
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            total += 56 + 64 * len(x)
            stack.extend(x)
        elif isinstance(x, np.ndarray):
            total += x.nbytes
    return total


# -- CUDA-primary sparse backend ----------------------------------------
#
# Pair-key open-addressed tables: (context_key, follower) -> count live
# in flat int64/fp32 GPU tensors.  One V-wide probe per order replaces
# the per-follower dict walk, so prediction is ~30 kernel launches per
# step instead of hundreds of Python dict ops.  Sentinel 0 = empty slot;
# pair keys are forced nonzero.  Slots are never freed in place - decay
# zeroes small counts and `rehash` rebuilds a table dropping dead keys.
