"""Labeled cell blocks - the STRATA info substrate.

``CellBlock`` = one bounded open-addressed 2-choice hash table of
packed cells (``cells.py``).  Blocks are the unit of labeling,
sealing, reverting and I/O:

  * ACTIVE  - mutable; every write pushes an undo snapshot into a
    per-block arena so ``revert_tag`` can restore bytes exactly.
  * SEALED  - immutable + checksummed; keeps only a tag manifest.
    Revert drops blocks wholly owned by the tag; partially-tagged
    sealed blocks are reported as tainted (bytes frozen by design).

``BlockStore`` = registry: kind ("o8", "epi", ...) -> [active, sealed,
..., cold(host)].  Reads probe newest->oldest; writes land in the
active block only (auto-seal+roll past ``fill_cap``).

Deterministic: probes are ~9 fixed tensor ops; writes pre-resolve
intra-batch slot collisions by sorting, and losers of the same-slot
race re-insert sequentially on the live table.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import torch

from forge.model.flux.helpers import (_MIX1, _MIX2, _s64, _mix64)
from .cells import (
    EMPTY_FOLL, count_lut, decode_counts, quantize_counts, pack_meta,
    meta_nfoll, meta_wtag, foll_tok, foll_cntq, foll_aux, foll_valid,
    pack_follower,
)


def _kind_salt(kind: str, block_id: int) -> int:
    """Per-(kind, block) i64 salt so identical ctx keys never collide
    the same way in two blocks."""
    h = zlib.crc32(kind.encode()) & 0xFFFFFFFF
    return _s64((h * 0x9E3779B97F4A7C15 + block_id * 0xBF58476D1CE4E5B9)
                % (1 << 64))


@dataclass
class BlockMeta:
    """Label header - the unit of debugging/QA tracking."""
    id: int
    kind: str
    born_seq: int = 0
    sealed: bool = False
    cold: bool = False
    checksum: int = 0
    tags: dict = field(default_factory=dict)   # tag_id -> write count
    writes: int = 0
    evictions: int = 0
    tainted: bool = False                      # sealed + partially owned
                                               # by a reverted tag


class CellBlock:
    """Fixed-stride packed cell table.  See module docstring for the
    active/sealed lifecycle."""

    def __init__(self, kind: str, block_id: int, slots: int, M: int,
                 dev: torch.device, store_raw: bool = True,
                 undo_cap: int = 200_000, vocab: int = 65536):
        self.kind = kind
        self.id = block_id
        self.S = slots
        self.M = M
        self.V = vocab
        self.dev = dev
        self.salt = _kind_salt(kind, block_id)
        self.meta = BlockMeta(id=block_id, kind=kind)
        self.tag = torch.zeros(slots, dtype=torch.int32, device=dev)
        self.metaw = torch.zeros(slots, dtype=torch.int32, device=dev)
        self.foll = torch.full((slots, M), EMPTY_FOLL,
                               dtype=torch.int32, device=dev)
        self.raw = (torch.zeros(slots, dtype=torch.int64, device=dev)
                    if store_raw else None)
        self.lut = count_lut(dev)
        self._undo: list[dict] = []      # batched undo snapshots
        self._undo_len = 0
        self.undo_cap = undo_cap
        self._hits = torch.zeros((), dtype=torch.int64, device=dev)
        self._wr = torch.zeros((), dtype=torch.int64, device=dev)
        self._ev = torch.zeros((), dtype=torch.int64, device=dev)
        self._ck = torch.zeros((), dtype=torch.int64, device=dev)
        self._sealed = False

    # ---- probe -----------------------------------------------------------

    def pk_of(self, keys: torch.Tensor) -> torch.Tensor:
        return _mix64(keys.to(torch.int64) * _MIX1
                      + torch.tensor(self.salt, dtype=torch.int64,
                                     device=keys.device))

    def tag32(self, pk: torch.Tensor) -> torch.Tensor:
        t = (pk >> 32).to(torch.int32)
        return torch.where(t == 0, torch.ones_like(t), t)

    def _slots2(self, pk: torch.Tensor):
        # s2 MUST be decorrelated from s1: pk*MIX2+MIX1 & (S-1) keeps
        # both slots functions of pk mod S, so a mod-S key collision
        # forces a double-slot collision (measured: ~2.4x expected
        # losers at 20% load).  A second _mix64 mixes in the high bits.
        s1 = pk & (self.S - 1)
        s2 = _mix64(pk + _MIX2) & (self.S - 1)
        return s1, s2

    def probe(self, keys: torch.Tensor):
        """(slot, found) - 2-choice read probe."""
        pk = self.pk_of(keys)
        t = self.tag32(pk)
        s1, s2 = self._slots2(pk)
        f1 = self.tag[s1] == t
        f2 = self.tag[s2] == t
        slot = torch.where((~f1) & f2, s2, s1)
        return slot, f1 | f2

    def _row_tot(self, slots: torch.Tensor) -> torch.Tensor:
        """Derived row total = sum of decoded follower counts."""
        f = self.foll[slots]                                   # [B,M]
        q = foll_cntq(f)
        c = decode_counts(q, self.lut)
        return (c * foll_valid(f).to(c.dtype)).sum(1)

    def read(self, keys: torch.Tensor):
        """(toks[B,M], cnts[B,M], tot[B], found[B]) - counts decoded,
        invalid columns zeroed."""
        slot, found = self.probe(keys)
        f = self.foll[slot]
        v = foll_valid(f) & found[:, None]
        toks = torch.where(v, foll_tok(f),
                           torch.full_like(f, -1))
        cnts = decode_counts(foll_cntq(f), self.lut) * v
        self._hits += found.sum()
        return toks, cnts, cnts.sum(1), found

    # ---- write -------------------------------------------------------------

    def _snapshot(self, slots: torch.Tensor, wtag: int,
                  seq: int, primary: bool = False) -> None:
        """Pre-write undo snapshot of every touched cell.  ``primary``
        marks the batch's own key slots - only those count toward the
        ``revert_*`` "writes undone" tally; side-effect snapshots
        (cuckoo dsts, repair-path probes) restore bytes silently."""
        if self._sealed or slots.numel() == 0:
            return
        ent = {"seq": seq, "wtag": wtag, "primary": primary,
               "slot": slots.clone(),
               "tag": self.tag[slots].clone(),
               "meta": self.metaw[slots].clone(),
               "foll": self.foll[slots].clone(),
               "raw": (self.raw[slots].clone()
                       if self.raw is not None else None)}
        self._undo.append(ent)
        self._undo_len += int(slots.numel())
        while self._undo_len > self.undo_cap and self._undo:
            self._undo_len -= int(self._undo.pop(0)["slot"].numel())

    def _dedup(self, keys: torch.Tensor, toks: torch.Tensor,
               delta: torch.Tensor):
        """Fold exact (key, tok) duplicates into one delta-summed write
        before the probe pass - otherwise two identical pairs in a batch
        read the same pre-write count and only +d lands instead of +2d.
        Stable tok-then-pk sort makes duplicates adjacent; group sums
        via _group_totals (deterministic)."""
        pk = self.pk_of(keys)
        o1 = torch.argsort(toks, stable=True)
        o2 = torch.argsort(pk[o1], stable=True)
        order = o1[o2]
        pk_s, t_s, d_s, k_s = pk[order], toks[order], delta[order], \
            keys[order]
        dup = (pk_s[1:] == pk_s[:-1]) & (t_s[1:] == t_s[:-1])
        head = torch.cat([torch.ones(1, dtype=torch.bool,
                                     device=keys.device), ~dup])
        # group sums in fp64: fp32 cumsum-diff picks up cancellation
        # error that differs between CPU sequential and GPU parallel
        # scan order (~3e-5 at k=2000) - enough to flip a log-quantized
        # count at a rounding boundary, breaking bit-exact parity.
        # fp64 error (~1e-16) cannot reach a 4.4% q-step boundary.
        starts = torch.nonzero(head).squeeze(1)
        ends = torch.cat([starts[1:], starts.new_full((1,), d_s.numel())])
        cum = torch.cat([torch.zeros(1, dtype=torch.float64,
                                     device=keys.device),
                         torch.cumsum(d_s.double(), 0)])
        dsum = cum[ends] - cum[starts]
        return k_s[head], t_s[head], dsum

    def add(self, keys: torch.Tensor, toks: torch.Tensor,
            delta: torch.Tensor, wtag: int = 0,
            seq: int = 0) -> torch.Tensor:
        """Batched follower writes.  Returns # of NEW cells inserted
        (device scalar).  All shapes [B]."""
        assert not self._sealed
        dev = self.dev
        keys, toks, delta = self._dedup(keys, toks, delta)
        B = keys.numel()
        pk = self.pk_of(keys)
        t = self.tag32(pk)
        s1, s2 = self._slots2(pk)
        f1 = self.tag[s1] == t
        f2 = self.tag[s2] == t
        found = f1 | f2
        e1 = self.tag[s1] == 0
        e2 = self.tag[s2] == 0
        evict = ~found & ~e1 & ~e2
        if bool(evict.any()):
            tot1 = self._row_tot(s1[evict])
            tot2 = self._row_tot(s2[evict])
            pick2_ev = tot2 < tot1
        else:
            pick2_ev = torch.zeros(0, dtype=torch.bool, device=dev)
        pick2 = ((~f1) & f2) | ((~found) & (~e1) & e2)
        pick2 = pick2.clone()
        pick2[evict] = pick2_ev
        slot = torch.where(pick2, s2, s1)
        new = ~found
        # cuckoo BEFORE mutations: victim raws must be the OLD keys
        moved = self._cuckoo_batch(slot[evict], wtag, seq,
                                   blocked=slot[new]) \
            if bool(evict.any()) else None
        self._snapshot(slot, wtag, seq, primary=True)

        # insert new keys / wipe victims (moved rows still get wiped
        # here - their copy already landed in the new slot).  Dup-slot
        # scatter is nondeterministic on CUDA ("which duplicate wins"
        # is undefined) - pick the winner explicitly as the LAST batch
        # position per slot, matching CPU sequential semantics.
        if bool(new.any()):
            sn, tn = slot[new], t[new]
            o = torch.argsort(sn, stable=True)
            sn_o = sn[o]
            last = torch.ones(sn.numel(), dtype=torch.bool, device=dev)
            last[:-1] = sn_o[1:] != sn_o[:-1]
            w = o[last]
            self.tag[sn[w]] = tn[w]
            if self.raw is not None:
                self.raw[sn[w]] = keys[new][w]
        if bool(evict.any()):
            ev = slot[evict]
            self.foll[ev] = EMPTY_FOLL
            self._ev += ev.numel() - (int(moved.sum())
                                      if moved is not None else 0)
            self._ck += (int(moved.sum())
                         if moved is not None else 0)

        # loser = new key whose slot was won by a later same-slot entry.
        # Losers never write followers (the winner owns the row) - the
        # old path wrote them at rank>0 then wiped, leaving sparse rows.
        ni = torch.nonzero(new).squeeze(1)
        win = torch.zeros(B, dtype=torch.bool, device=dev)
        if ni.numel():
            win[ni[w]] = True
        loser = new & ~win

        # follower column selection: rank = intra-slot order among LIVE
        # writers only (losers excluded), so a fresh row's first
        # follower always lands at first_empty (left-packed invariant:
        # valid follower cols are exactly [0, nfoll)).
        rows = self.foll[slot]                              # [B,M]
        match = (foll_tok(rows) == (toks[:, None].long()
                                    & 0xFFFF).to(torch.int32))
        match &= foll_valid(rows)
        has = match.any(1)
        empty = ~foll_valid(rows)
        has_empty = empty.any(1)
        # argmax/argmin tie-breaks are not guaranteed identical across
        # CPU and CUDA - resolve first-index ties explicitly with amin
        # so both devices pick the same column bit-for-bit.
        ar = torch.arange(self.M, device=dev, dtype=torch.int64)
        ar_b = ar.unsqueeze(0).expand_as(empty)
        big = torch.full_like(ar_b, self.M)
        first_empty = torch.where(empty, ar_b, big).amin(1)
        rows_c = decode_counts(foll_cntq(rows), self.lut)
        rows_c = torch.where(foll_valid(rows), rows_c,
                             torch.full_like(rows_c, 1e9))
        is_min = rows_c == rows_c.amin(1, keepdim=True)
        min_col = torch.where(is_min, ar_b, big).amin(1)
        rank = torch.zeros(B, dtype=torch.int64, device=dev)
        li = torch.nonzero(~loser).squeeze(1)
        if li.numel():
            lo = torch.argsort(slot[li], stable=True)
            lsi = li[lo]
            gs = slot[lsi]
            head = torch.ones(lsi.numel(), dtype=torch.bool, device=dev)
            head[1:] = gs[1:] != gs[:-1]
            r = head.long().cumsum(0) - 1
            rf = torch.searchsorted(gs, gs)
            rank[lsi] = r - r[rf]
        # <=1 matching column per row by construction (followers hold
        # distinct toks), so this argmax is tie-free.
        col = torch.where(
            has, match.float().argmax(1),
            torch.where(has_empty,
                        (first_empty + rank).clamp(max=self.M - 1),
                        min_col))

        # write follower words: merge-bump for existing toks, full word
        # for new followers.  One deterministic scatter (last batch
        # position wins a cell_flat collision - same rule as tags).
        aux = torch.full_like(toks, wtag & 0xF)
        dq = quantize_counts(delta)
        new_word = pack_follower(toks, dq, aux)
        cell_flat = slot * self.M + col
        vf = self.foll.view(-1)
        old_w = vf[cell_flat]
        old_c = decode_counts(foll_cntq(old_w), self.lut)
        upd = torch.where(has, quantize_counts(old_c + delta), dq)
        upd_w = ((old_w.long() & ~0xFF0000)
                 | (upd.long() << 16)).to(torch.int32)
        w_final = torch.where(has, upd_w, new_word)
        wi = torch.nonzero(~loser).squeeze(1)
        if wi.numel():
            cf = cell_flat[wi]
            co = torch.argsort(cf, stable=True)
            cf_o = cf[co]
            last = torch.ones(cf.numel(), dtype=torch.bool, device=dev)
            last[:-1] = cf_o[1:] != cf_o[:-1]
            sel = wi[co[last]]
            vf[cell_flat[sel]] = w_final[sel]

        stuck = loser

        # meta: nfoll recompute + wtag stamp, preserve flags halfword
        touched = slot.unique()
        fr = self.foll[touched]
        nf = foll_valid(fr).sum(1)
        self.metaw[touched] = (
            (nf & 0xFF).to(torch.int32)
            | torch.tensor((wtag & 0xFF) << 8, dtype=torch.int32,
                           device=dev)
            | (self.metaw[touched] & ~0xFFFF))
        self._wr += B
        self.meta.writes += B
        self.meta.tags[wtag] = self.meta.tags.get(wtag, 0) + B

        sidx = torch.nonzero(stuck).squeeze(1)
        for j in sidx.tolist():
            self._add_one(int(keys[j]), int(toks[j]),
                          float(delta[j]), wtag, seq)

        # verify pass: every key in this batch must be readable after
        # the write.  Collision cascades (dst-vs-new races, repair-path
        # evictions wiping a row written moments earlier) can orphan a
        # fresh key; re-insert stragglers via the sequential path until
        # clean.  Only runs when collisions occurred - at sane load a
        # straggler's candidates are free or evict-only-old-rows, so it
        # converges in 1 round; the bound is for dense/adversarial
        # tables where eviction is the designed outcome.
        if bool(stuck.any()) or bool(evict.any()):
            for _ in range(4):
                _, fnd = self.probe(keys)
                if bool(fnd.all()):
                    break
                for j in torch.nonzero(~fnd).squeeze(1).tolist():
                    self._add_one(int(keys[j]), int(toks[j]),
                                  float(delta[j]), wtag, seq)
        return (new & ~stuck).sum()

    def _cuckoo_batch(self, slots: torch.Tensor, wtag: int,
                      seq: int,
                      blocked: torch.Tensor | None = None
                      ) -> torch.Tensor:
        """Relocate each victim row to ITS owner's other candidate slot
        when that slot is empty (raw keys required to re-derive pk).
        Returns the moved mask.  Rows not moved keep FLUX evict
        semantics.  ``blocked`` = slot indices already claimed by this
        batch's new keys - a dst colliding with one is skipped, else
        the moved row's tag would be overwritten mid-write and its
        followers silently orphaned under the new key's tag."""
        moved = torch.zeros(slots.numel(), dtype=torch.bool,
                            device=self.dev)
        if self.raw is None or slots.numel() == 0:
            return moved
        vpk = self.pk_of(self.raw[slots])
        vs1, vs2 = self._slots2(vpk)
        dst = torch.where(vs1 == slots, vs2, vs1)
        free = self.tag[dst] == 0
        if blocked is not None and blocked.numel():
            free &= ~torch.isin(dst, blocked.to(dst.device))
        # two victims can target the same dst - first one wins
        seen = torch.zeros(self.S, dtype=torch.bool, device=self.dev)
        for i in torch.nonzero(free).squeeze(1).tolist():
            d = int(dst[i])
            if not bool(seen[d]):                   # rare path only
                seen[d] = True
                moved[i] = True
        if bool(moved.any()):
            src, d = slots[moved], dst[moved]
            self._snapshot(d, wtag, seq)            # dst cells (empty)
            self.tag[d] = self.tag[src]
            self.metaw[d] = self.metaw[src]
            self.foll[d] = self.foll[src]
            self.raw[d] = self.raw[src]
        return moved

    def _displace(self, s: int, wtag: int, seq: int,
                  hops: int = 8) -> bool:
        """Cuckoo-walk: free slot ``s`` by shifting its occupant to the
        occupant's other candidate slot, kicking further rows along the
        walk until an empty slot absorbs the chain.  Sequential single-
        row ops - repair path only.  Requires ``raw`` (victim
        alternatives are re-derived from stored keys).  If the walk
        exhausts ``hops``, the last displaced row drops (counted as an
        eviction - ``s`` is still freed).  A walk that circles back
        into ``s`` returns False: the caller keeps evict semantics."""
        if self.raw is None:
            return False
        dev = self.dev
        h_tag = int(self.tag[s]); h_meta = int(self.metaw[s])
        h_foll = self.foll[s].clone(); h_raw = int(self.raw[s])
        self._snapshot(torch.tensor([s], dtype=torch.int64, device=dev),
                       wtag, seq)
        cur = s
        for _ in range(hops):
            vpk = self.pk_of(torch.tensor([h_raw], dtype=torch.int64,
                                          device=dev))
            vs1, vs2 = self._slots2(vpk)
            o = int(vs2[0]) if int(vs1[0]) == cur else int(vs1[0])
            if o == s:
                return False
            self._snapshot(torch.tensor([o], dtype=torch.int64,
                                        device=dev), wtag, seq)
            n_tag = int(self.tag[o]); n_meta = int(self.metaw[o])
            n_foll = self.foll[o].clone(); n_raw = int(self.raw[o])
            self.tag[o] = h_tag
            self.metaw[o] = h_meta
            self.foll[o] = h_foll
            self.raw[o] = h_raw
            if n_tag == 0:
                return True
            h_tag, h_meta, h_foll, h_raw = n_tag, n_meta, n_foll, n_raw
            cur = o
        self._ev += 1
        return True

    def _add_one(self, key: int, tok: int, d: float, wtag: int,
                 seq: int) -> None:
        """Single-key live insert (collision-repair path)."""
        dev = self.dev
        k = torch.tensor([_s64(key % (1 << 64))], dtype=torch.int64,
                         device=dev)
        pk = self.pk_of(k)
        t = self.tag32(pk)
        s1, s2 = self._slots2(pk)
        for s in (int(s1[0]), int(s2[0])):
            if int(self.tag[s]) == int(t[0]):
                self._merge_one(s, tok, d, wtag, seq)
                return
        # prefer an empty candidate, else cuckoo the smaller row to its
        # other candidate (or evict it when raw keys are off).  Each
        # path snapshots exactly the slot it mutates - no defensive
        # both-candidates snapshot (they inflated revert tallies and
        # burned undo-arena capacity on unmodified cells).
        for s in (int(s1[0]), int(s2[0])):
            if int(self.tag[s]) == 0:
                self._snapshot(torch.tensor([s], dtype=torch.int64,
                                            device=dev), wtag, seq)
                self._fresh_row(s, int(t[0]), key, tok, d, wtag)
                return
        # cuckoo-walk the cheaper-movable victim: try each candidate's
        # row in ascending-tot order; evict only when no chain reaches
        # an empty slot within the hop bound (or lean mode has no raw
        # keys to re-derive victim alternatives)
        t1 = float(self._row_tot(
            torch.tensor([int(s1[0])], device=dev))[0])
        t2 = float(self._row_tot(
            torch.tensor([int(s2[0])], device=dev))[0])
        cand = sorted((int(s1[0]), int(s2[0])),
                      key=lambda s: t1 if s == int(s1[0]) else t2)
        s = None
        for c in cand:
            if self._displace(c, wtag, seq):
                self._ck += 1
                s = c
                break
        if s is None:
            s = cand[0]
            self._ev += 1
            self._snapshot(torch.tensor([s], dtype=torch.int64,
                                        device=dev), wtag, seq)
            self.foll[s] = EMPTY_FOLL
        self._fresh_row(s, int(t[0]), key, tok, d, wtag)

    def _fresh_row(self, s: int, t: int, key: int, tok: int, d: float,
                   wtag: int) -> None:
        self.tag[s] = t
        if self.raw is not None:
            self.raw[s] = key
        self.foll[s] = EMPTY_FOLL
        self.foll[s, 0] = int(pack_follower(
            torch.tensor(tok), quantize_counts(torch.tensor(d)),
            torch.tensor(wtag & 0xF)))
        self.metaw[s] = int(pack_meta(torch.tensor(1), wtag & 0xFF, 0))

    def _merge_one(self, s: int, tok: int, d: float, wtag: int,
                   seq: int) -> None:
        self._snapshot(torch.tensor([s], dtype=torch.int64,
                                    device=self.dev), wtag, seq)
        row = self.foll[s]
        toks = foll_tok(row)
        m = (toks == (tok & 0xFFFF)) & foll_valid(row)
        if bool(m.any()):
            c = int(m.to(torch.int8).argmax())
            old = decode_counts(foll_cntq(row[c:c + 1]), self.lut)
            q = quantize_counts(old + d)
            self.foll[s, c] = int(
                ((int(row[c]) & ~0xFF0000) | (int(q[0]) << 16)))
        else:
            v = foll_valid(row)
            ar = torch.arange(self.M, device=dev)
            if bool((~v).any()):
                c = int(torch.where(~v, ar, self.M).amin())
            else:
                rc = decode_counts(foll_cntq(row), self.lut)
                c = int(torch.where(rc == rc.amin(), ar,
                                    self.M).amin())
            self.foll[s, c] = int(pack_follower(
                torch.tensor(tok), quantize_counts(torch.tensor(d)),
                torch.tensor(wtag & 0xF)))
        self.metaw[s] = int(pack_meta(
            foll_valid(self.foll[s]).sum(), wtag & 0xFF, 0))

    # ---- revert / seal ----------------------------------------------------

    def revert_tag(self, wtag: int) -> int:
        """Undo every active-arena write made under ``wtag``; restores
        bytes exactly (later non-tag writes to the same cells are lost
        - documented destructive-pre-state semantics).  Returns the
        count of PRIMARY writes undone - side-effect snapshots (cuckoo
        relocations, repair probes) restore bytes but don't count."""
        n = 0
        keep = []
        for ent in reversed(self._undo):
            if ent["wtag"] == wtag:
                s = ent["slot"]
                self.tag[s] = ent["tag"]
                self.metaw[s] = ent["meta"]
                self.foll[s] = ent["foll"]
                if self.raw is not None and ent["raw"] is not None:
                    self.raw[s] = ent["raw"]
                if ent.get("primary", True):
                    n += int(s.numel())
                self._undo_len -= int(s.numel())
            else:
                keep.append(ent)
        self._undo = list(reversed(keep))
        self.meta.tags.pop(wtag, None)
        return n

    def revert_since(self, seq: int) -> int:
        """Undo all writes with entry seq > ``seq`` (point-in-time)."""
        n = 0
        while self._undo and self._undo[-1]["seq"] > seq:
            ent = self._undo.pop()
            s = ent["slot"]
            self.tag[s] = ent["tag"]
            self.metaw[s] = ent["meta"]
            self.foll[s] = ent["foll"]
            if self.raw is not None and ent["raw"] is not None:
                self.raw[s] = ent["raw"]
            n += int(s.numel())
            self._undo_len -= int(s.numel())
        return n

    def checksum(self) -> int:
        """Cheap content checksum over the three slabs."""
        def h(t):
            b = t.detach().to(torch.int64).cpu().numpy().tobytes()
            return zlib.crc32(b)
        v = h(self.tag) ^ (h(self.metaw) << 1) ^ (h(self.foll) << 2)
        return v & 0xFFFFFFFF

    def seal(self) -> None:
        self._sealed = True
        self.meta.sealed = True
        self.meta.checksum = self.checksum()

    def occupied(self) -> torch.Tensor:
        return (self.tag != 0).sum()

    def load(self) -> float:
        return float(self.occupied()) / self.S

    # ---- io ---------------------------------------------------------------

    def tensors(self) -> dict:
        t = {f"b{self.id}.tag": self.tag,
             f"b{self.id}.meta": self.metaw,
             f"b{self.id}.foll": self.foll}
        if self.raw is not None:
            t[f"b{self.id}.raw"] = self.raw
        return t

    def to_dict(self) -> dict:
        return {"id": self.id, "kind": self.kind, "slots": self.S,
                "M": self.M, "V": self.V, "sealed": self._sealed,
                "has_raw": self.raw is not None,
                "meta": {"born_seq": self.meta.born_seq,
                         "sealed": self.meta.sealed,
                         "cold": self.meta.cold,
                         "checksum": self.meta.checksum,
                         "tags": {str(k): v for k, v
                                  in self.meta.tags.items()},
                         "writes": self.meta.writes,
                         "evictions": int(self._ev),
                         "tainted": self.meta.tainted}}

    @classmethod
    def from_dict(cls, d: dict, tensors: dict, dev: torch.device,
                  undo_cap: int = 200_000) -> "CellBlock":
        b = cls(d["kind"], d["id"], d["slots"], d["M"], dev,
                store_raw=d["has_raw"], undo_cap=undo_cap,
                vocab=d.get("V", 65536))
        p = f"b{d['id']}."
        b.tag = tensors[p + "tag"].to(dev)
        b.metaw = tensors[p + "meta"].to(dev)
        b.foll = tensors[p + "foll"].to(dev)
        if d["has_raw"]:
            b.raw = tensors[p + "raw"].to(dev)
        m = d["meta"]
        b.meta.born_seq = m["born_seq"]
        b.meta.sealed = m["sealed"]
        b._sealed = m["sealed"]
        b.meta.cold = m["cold"]
        b.meta.checksum = m["checksum"]
        b.meta.tags = {int(k): v for k, v in m["tags"].items()}
        b.meta.writes = m["writes"]
        b.meta.tainted = m["tainted"]
        return b


class BlockStore:
    """kind -> [active, sealed-newest ... sealed-oldest ... cold-host]."""

    def __init__(self, slots: int = 1 << 18, M: int = 8,
                 dev: str = "cpu", store_raw: bool = True,
                 fill_cap: float = 0.85, max_blocks: int = 8,
                 cold_blocks: int = 16, undo_cap: int = 200_000,
                 vocab: int = 65536):
        assert slots & (slots - 1) == 0
        self.slots, self.M, self.V = slots, M, vocab
        self.dev = torch.device(dev)
        self.store_raw = store_raw
        self.fill_cap = fill_cap
        self.max_blocks = max_blocks
        self.cold_blocks = cold_blocks
        self.undo_cap = undo_cap
        self.kinds: dict[str, list[CellBlock]] = {}
        self.tag_ids: dict[str, int] = {"": 0}
        self._next_tag = 1
        self._next_bid = 0
        self.seq = 0                     # global write clock

    def tag_id(self, name: str) -> int:
        if name not in self.tag_ids:
            self.tag_ids[name] = self._next_tag
            self._next_tag += 1
        return self.tag_ids[name]

    def _new_block(self, kind: str, dev=None) -> CellBlock:
        b = CellBlock(kind, self._next_bid, self.slots, self.M,
                      dev or self.dev, self.store_raw, self.undo_cap,
                      self.V)
        self._next_bid += 1
        return b

    def _active(self, kind: str) -> CellBlock:
        bl = self.kinds.setdefault(kind, [])
        if not bl or bl[0]._sealed:
            bl.insert(0, self._new_block(kind))
        return bl[0]

    def write(self, kind: str, keys: torch.Tensor, toks: torch.Tensor,
              delta: torch.Tensor, tag: str = "") -> torch.Tensor:
        b = self._active(kind)
        if b.load() >= self.fill_cap:
            self.seal(kind)
            b = self._active(kind)
        tid = self.tag_id(tag)
        b.meta.born_seq = b.meta.born_seq or self.seq
        n = b.add(keys.to(self.dev), toks.to(self.dev),
                  delta.to(self.dev), wtag=tid & 0xFF, seq=self.seq)
        self.seq += int(keys.numel())
        if b.load() >= self.fill_cap:
            self.seal(kind)
        return n

    def read(self, kind: str, keys: torch.Tensor):
        """Merge across blocks, newest first.  Returns
        (toks[B,M], cnts[B,M], tot[B], found[B])."""
        bl = self.kinds.get(kind, [])
        keys = keys.to(self.dev)
        out_t = torch.full((keys.numel(), self.M), -1,
                           dtype=torch.int32, device=self.dev)
        out_c = torch.zeros((keys.numel(), self.M),
                            dtype=torch.float32, device=self.dev)
        got = torch.zeros(keys.numel(), dtype=torch.bool,
                          device=self.dev)
        for b in bl:
            if not bool((~got).any()):
                break
            kk = keys[~got]
            toks, cnts, tot, fnd = b.read(kk.to(b.dev))
            idx = torch.nonzero(~got).squeeze(1)
            hit = idx[fnd.to(idx.device)]
            out_t[hit] = toks[fnd].to(self.dev)
            out_c[hit] = cnts[fnd].to(self.dev)
            got[hit] = True
        return out_t, out_c, out_c.sum(1), got

    def seal(self, kind: str) -> None:
        """Seal the active block, enforce the hot/cold budget, then roll
        a fresh active on top - a kind ALWAYS ends seal() with an active
        head so ``kinds[kind][0]`` is writable and revert-drop arithmetic
        keeps ``len`` stable."""
        bl = self.kinds.get(kind)
        if bl and not bl[0]._sealed:
            bl[0].seal()
            self._enforce_budget(kind)
            self._active(kind)

    def _enforce_budget(self, kind: str) -> None:
        """Oldest sealed blocks demote to host past ``max_blocks``;
        coldest dropped entirely past ``cold_blocks``."""
        bl = self.kinds[kind]
        hot = [b for b in bl if not b.meta.cold]
        while len(hot) > self.max_blocks:
            victim = hot.pop()
            victim.meta.cold = True
            victim.tag = victim.tag.cpu()
            victim.metaw = victim.metaw.cpu()
            victim.foll = victim.foll.cpu()
            if victim.raw is not None:
                victim.raw = victim.raw.cpu()
            victim.lut = victim.lut.cpu()
            victim.dev = torch.device("cpu")
        colds = [b for b in bl if b.meta.cold]
        while len(colds) > self.cold_blocks:
            bl.remove(colds.pop())

    def revert_tag(self, tag: str) -> dict:
        """Active blocks: restore bytes via arena.  Sealed blocks fully
        owned by the tag: dropped.  Partially-tagged sealed: kept,
        flagged tainted (frozen bytes win over destructive undo)."""
        tid = self.tag_ids.get(tag)
        rep = {"undone": 0, "dropped": [], "tainted": []}
        if tid is None:
            return rep
        for kind, bl in self.kinds.items():
            for b in list(bl):
                if not b._sealed:
                    rep["undone"] += b.revert_tag(tid)
                else:
                    tot = sum(b.meta.tags.values())
                    own = b.meta.tags.get(tid, 0)
                    if own and own == tot:
                        bl.remove(b)
                        rep["dropped"].append(b.id)
                    elif own:
                        b.meta.tainted = True
                        rep["tainted"].append(b.id)
        return rep

    def revert_since(self, seq: int) -> int:
        return sum(b.revert_since(seq) for bl in self.kinds.values()
                   for b in bl if not b._sealed)

    def stats(self) -> dict:
        out = {}
        for kind, bl in self.kinds.items():
            out[kind] = [{
                "id": b.id, "sealed": b._sealed, "cold": b.meta.cold,
                "load": round(b.load(), 4),
                "writes": b.meta.writes,
                "hits": int(b._hits),
                "evictions": int(b._ev),
                "tags": dict(b.meta.tags),
                "tainted": b.meta.tainted} for b in bl]
        return out

    # ---- io ---------------------------------------------------------------

    def tensors(self) -> dict:
        t = {}
        for bl in self.kinds.values():
            for b in bl:
                t.update(b.tensors())
        return t

    def manifest(self) -> dict:
        return {"version": 1, "slots": self.slots, "M": self.M,
                "store_raw": self.store_raw, "seq": self.seq,
                "next_bid": self._next_bid,
                "tag_ids": dict(self.tag_ids),
                "kinds": {k: [b.to_dict() for b in bl]
                          for k, bl in self.kinds.items()}}

    @classmethod
    def from_manifest(cls, m: dict, tensors: dict,
                      dev: str = "cpu") -> "BlockStore":
        st = cls(slots=m["slots"], M=m["M"], dev=dev,
                 store_raw=m["store_raw"])
        st.seq = m["seq"]
        st._next_bid = m["next_bid"]
        st.tag_ids = {k: int(v) for k, v in m["tag_ids"].items()}
        st._next_tag = max(st.tag_ids.values()) + 1
        for kind, bds in m["kinds"].items():
            bl = []
            for bd in bds:
                b = CellBlock.from_dict(
                    bd, tensors, torch.device(dev), st.undo_cap)
                bl.append(b)
            st.kinds[kind] = bl
        return st
