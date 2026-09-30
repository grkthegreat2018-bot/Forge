"""CUDA-primary open-addressed sparse tables for FluxLM."""
from __future__ import annotations

import numpy as np
import torch

from .helpers import _MIX1, _MIX2, _group_add, _mix64, _s64

class _GPUCellTables:
    """orders -> open-addressed context-key -> follower-row tables,
    stacked: all per-order tensors live in [O, S] / [O, S, M] buffers
    so ONE probe loop serves every order (fewer kernel launches).

    Each occupied slot holds a fixed-width row of ``M`` (tok, count)
    pairs.  Rows that overflow keep their top-M followers (smallest
    count evicted), matching the heavy-tailed follower distribution.
    ``raw`` stores the pre-hash context key so rows export without a
    host reverse map.  Sentinel 0 = empty slot.
    """

    # extra shared-key-matrix rows: row O = csem, row O+1 = epi
    ROW_CSEM_SALT = _s64(0xC5E400C5E400C5E4 % (1 << 64))
    ROW_EPI_SALT = _s64(0xE91A55E91A55E91A % (1 << 64))

    def __init__(self, orders, slots: int, dev: torch.device,
                 probe_max: int, vocab: int, M: int = 16):
        self.orders = list(orders)
        self.kidx = {k: i for i, k in enumerate(orders)}
        self.O = len(self.orders)
        self.R = self.O + 2                 # + csem row + epi row
        self.row_csem = self.O
        self.row_epi = self.O + 1
        self.S = slots
        self.V = vocab
        self.M = M
        self.dev = dev
        self.probe_max = probe_max
        # ONE key matrix shared by order/csem/epi tables - a single
        # probe loop resolves every table's context per step
        self.tkeys = torch.zeros((self.R, slots), dtype=torch.int64,
                                 device=dev)
        self.traw = torch.zeros((self.R, slots), dtype=torch.int64,
                                device=dev)
        self.ttoks = torch.full((self.O, slots, M), -1,
                                dtype=torch.int32, device=dev)
        self.tcnt = torch.zeros((self.O, slots, M), dtype=torch.float32,
                                device=dev)
        self.ttot = torch.zeros((self.O, slots), dtype=torch.float32,
                                device=dev)
        # per-row load for 2-choice eviction: orders -> follower total,
        # epi -> position count, csem -> write count
        self.tocc = torch.zeros((self.R, slots), dtype=torch.float32,
                                device=dev)
        self.used = [0] * self.R
        self._ord_t = torch.arange(self.R, dtype=torch.int64,
                                   device=dev)
        self._ordx = torch.arange(self.O, dtype=torch.int64,
                                  device=dev)
        salts = [_s64((k * 0x9E3779B97F4A7C15) % (1 << 64))
                 for k in self.orders]
        salts += [self.ROW_CSEM_SALT, self.ROW_EPI_SALT]
        self._salt = torch.tensor(salts, dtype=torch.int64, device=dev)

    def ctx_hash(self, k: int, ctx_keys: torch.Tensor) -> torch.Tensor:
        return _mix64(ctx_keys * _MIX1 + _s64(
            (k * 0x9E3779B97F4A7C15) % (1 << 64)))

    def ctx_hash_all(self, ctx_keys: torch.Tensor) -> torch.Tensor:
        """pk for an [R] vector of context keys (orders+csem+epi)."""
        return _mix64(ctx_keys * _MIX1 + self._salt[:ctx_keys.numel()])

    def _slots2(self, pk: torch.Tensor) -> tuple[torch.Tensor,
                                                torch.Tensor]:
        """Two candidate slots per key (d-left / 2-choice hashing).
        Deterministic: writes and reads probe the same pair, so a key
        is always found at one of its two slots - no probe chain."""
        s1 = pk & (self.S - 1)
        s2 = (pk * _MIX2 + _MIX1) & (self.S - 1)
        return s1, s2

    def probe_fixed(self, pk: torch.Tensor,
                    iters: int = 0,
                    rows: torch.Tensor | None = None
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read probe over the shared [R,S] key matrix: each key
        checks its two candidate slots.  ~9 ops, zero host syncs -
        graph-safe.  ``rows`` overrides the default first-n rows so
        arbitrary (row, key) batches probe in one call."""
        n = pk.numel()
        if rows is None:
            rows = self._ord_t[:n]
        s1, s2 = self._slots2(pk)
        kr = self.tkeys
        f1 = kr[rows, s1] == pk
        f2 = kr[rows, s2] == pk
        found = f1 | f2
        slot = torch.where((~f1) & f2, s2, s1)
        return slot, found

    def probe_write(self, pk: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """Write probe over [R,S]: (slot, found, evict).  Insert
        prefers the empty candidate; when both are occupied the row
        with the smaller load (``tocc``) is evicted - bounded-memory
        LRU-ish forgetting."""
        n = pk.numel()
        rows = self._ord_t[:n]
        s1, s2 = self._slots2(pk)
        kr, oc = self.tkeys, self.tocc
        f1 = kr[rows, s1] == pk
        f2 = kr[rows, s2] == pk
        found = f1 | f2
        e1 = kr[rows, s1] == 0
        e2 = kr[rows, s2] == 0
        evict = ~found & ~e1 & ~e2
        pick2 = ((~f1) & f2) | ((~found) & (~e1) & e2) \
            | (evict & (oc[rows, s2] < oc[rows, s1]))
        return torch.where(pick2, s2, s1), found, evict

    def _probe(self, tab_keys: torch.Tensor,
               pk: torch.Tensor,
               check_every: int = 0) -> tuple[torch.Tensor,
                                              torch.Tensor]:
        """1-D read probe on a single table row (``tab_keys`` [S]).
        Two-choice lookup - no chain, no syncs."""
        s1, s2 = self._slots2(pk)
        f1 = tab_keys[s1] == pk
        f2 = tab_keys[s2] == pk
        return torch.where((~f1) & f2, s2, s1), f1 | f2

    def _probe_row_w(self, row: int, pk: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor]:
        """1-D write probe on ``tkeys[row]`` - same policy as
        ``probe_write``."""
        s1, s2 = self._slots2(pk)
        kr, oc = self.tkeys[row], self.tocc[row]
        k1, k2 = kr[s1], kr[s2]
        f1, f2 = k1 == pk, k2 == pk
        found = f1 | f2
        e1, e2 = k1 == 0, k2 == 0
        evict = ~found & ~e1 & ~e2
        pick2 = ((~f1) & f2) | ((~found) & (~e1) & e2) \
            | (evict & (oc[s2] < oc[s1]))
        return torch.where(pick2, s2, s1), found, evict

    def read_row(self, k: int, ctx_key: int):
        """(toks[M], cnts[M], tot) for one context - predict path."""
        oi = self.kidx[k]
        pk = self.ctx_hash(k, torch.tensor(
            [_s64(ctx_key)], dtype=torch.int64, device=self.dev))
        slot, found = self._probe(self.tkeys[oi], pk)
        s = int(slot[0])
        if not bool(found[0]):
            z = torch.zeros(self.M, dtype=torch.float32,
                            device=self.dev)
            return torch.full((self.M,), -1, dtype=torch.int32,
                              device=self.dev), z, 0.0
        return (self.ttoks[oi][s], self.tcnt[oi][s],
                float(self.ttot[oi][s]))

    def read_tot(self, k: int, ctx_keys: torch.Tensor) -> torch.Tensor:
        oi = self.kidx[k]
        pk = self.ctx_hash(k, ctx_keys)
        slot, found = self._probe(self.tkeys[oi], pk)
        return torch.where(found, self.ttot[oi][slot],
                           torch.zeros((), device=self.dev))

    def add(self, k: int, ctx_keys: torch.Tensor, toks: torch.Tensor,
            d: torch.Tensor) -> torch.Tensor:
        """Batched follower-count writes into one order's rows.
        Fully sync-free - masked writes run unconditionally (empty
        index sets are legal); returns the device-side count of new
        EMPTY-slot inserts for deferred ``used`` bookkeeping."""
        oi = self.kidx[k]
        keys, raw, rt, rc, rt_tot = (self.tkeys[oi], self.traw[oi],
                                     self.ttoks[oi], self.tcnt[oi],
                                     self.ttot[oi])
        pk = self.ctx_hash(k, ctx_keys)
        slot, found, evict = self._probe_row_w(oi, pk)
        new = ~found
        keys[slot[new]] = pk[new]
        raw[slot[new]] = ctx_keys[new]
        ev = new & evict                        # victim rows: wipe
        rt[slot[ev]] = -1
        rc[slot[ev]] = 0.0
        rt_tot[slot[ev]] = 0.0
        rows_t = rt[slot]                          # [N, M]
        match = rows_t == toks[:, None].to(torch.int32)
        has = match.any(dim=1)
        empty = rows_t == -1
        has_empty = empty.any(dim=1)
        first_empty = empty.float().argmax(dim=1)
        rows_c = rc[slot]
        min_col = rows_c.argmin(dim=1)
        # per-slot rank among distinct tokens so same-row writes in
        # this batch land on distinct columns (not all on first_empty)
        ord_ = torch.argsort(slot * (self.V + 1) + toks)
        g, tk_s = slot[ord_], toks[ord_]
        is_new = ~((g[1:] == g[:-1]) & (tk_s[1:] == tk_s[:-1]))
        rank_g = torch.cat([torch.ones(1, dtype=torch.int64,
                                       device=self.dev),
                            is_new.long()]).cumsum(0) - 1
        row_first = torch.searchsorted(g, g)
        inv = torch.empty_like(ord_)
        inv[ord_] = torch.arange(ord_.numel(), device=self.dev)
        rank = (rank_g - rank_g[row_first])[inv]
        col = torch.where(
            has, match.float().argmax(dim=1),
            torch.where(has_empty,
                        (first_empty + rank).clamp(max=self.M - 1),
                        min_col))
        # non-existing followers need their token id written into the
        # chosen column (empty slot AND evict-min cases)
        write_tok = ~has
        vic = rows_c.gather(1, col[:, None]).squeeze(1)
        rt[slot[write_tok], col[write_tok]] = \
            toks[write_tok].to(torch.int32)
        ev_c = write_tok & ~has_empty
        rc[slot[ev_c], col[ev_c]] = 0.0
        flat = slot * self.M + col
        # deterministic accumulation - duplicate (slot,col) cells or
        # repeated slots inside the batch would otherwise sum through
        # float atomics in kernel-arrival order
        _group_add(rc.view(-1), flat, d)
        _group_add(rt_tot, slot[ev_c], -vic[ev_c])
        _group_add(rt_tot, slot, d)
        self.tocc[oi][slot] = rt_tot[slot]
        return (new & ~evict).sum()

    def add_multi(self, ctx_keys: torch.Tensor, toks: torch.Tensor,
                  d: torch.Tensor, valid: torch.Tensor
                  ) -> torch.Tensor:
        """All orders in ONE call: ``ctx_keys``/``toks``/``d``/``valid``
        are [O, N].  ~40 tensor ops total instead of ~20 per order.
        Same (row, slot) collisions inside the batch get distinct
        follower columns via intra-group rank, so results match
        sequential ``add`` exactly."""
        O, N = ctx_keys.shape
        dev = self.dev
        pk = _mix64(ctx_keys * _MIX1 + self._salt[:O, None])
        rf = self._ordx[:, None].expand(O, N).reshape(-1)
        pf = pk.reshape(-1)
        vf = valid.reshape(-1)
        tf = toks.reshape(-1)
        df = d.reshape(-1)
        s1, s2 = self._slots2(pf)
        k1 = self.tkeys[rf, s1]
        k2 = self.tkeys[rf, s2]
        f1 = k1 == pf
        f2 = k2 == pf
        found = f1 | f2
        e1, e2 = k1 == 0, k2 == 0
        evict = ~found & ~e1 & ~e2 & vf
        pick2 = ((~f1) & f2) | ((~found) & (~e1) & e2) \
            | (evict & (self.tocc[rf, s2] < self.tocc[rf, s1]))
        slot = torch.where(pick2, s2, s1)
        row_ix = rf * self.S + slot                       # [B]
        new = ~found & vf
        # capture victim rows BEFORE any write touches their slots
        # (exact undo needs the displaced key + followers + total)
        evr = row_ix[evict]                             # evict &= vf
        evict_snap = (
            evr,
            self.traw.view(-1).index_select(0, evr).clone(),
            self.ttoks.view(-1, self.M).index_select(0, evr).clone(),
            self.tcnt.view(-1, self.M).index_select(0, evr).clone(),
            self.ttot.view(-1).index_select(0, evr).clone())
        tk_v = self.tkeys.view(-1)
        tr_v = self.traw.view(-1)
        tk_v.index_copy_(0, row_ix[new], pf[new])
        tr_v.index_copy_(0, row_ix[new],
                         ctx_keys.reshape(-1)[new])
        self.ttoks.view(-1, self.M).index_fill_(0, evr, -1)
        self.tcnt.view(-1, self.M).index_fill_(0, evr, 0.0)
        self.ttot.view(-1).index_fill_(0, evr, 0.0)
        # follower columns - gather post-wipe rows
        rows_t = self.ttoks.view(-1, self.M)[row_ix]    # [B, M]
        rows_c = self.tcnt.view(-1, self.M)[row_ix]
        match = rows_t == tf[:, None].to(torch.int32)
        has = match.any(dim=1)
        empty = rows_t == -1
        has_empty = empty.any(dim=1)
        first_empty = empty.float().argmax(dim=1)
        min_col = rows_c.argmin(dim=1)
        # intra-(row,slot) rank over distinct tokens: positions sharing
        # a row with different followers must take different columns.
        # Rank must be PER-ROW - a global pair-rank offset makes
        # first_empty+rank overflow into col M-1 collisions.
        ord_ = torch.argsort(row_ix * (self.V + 1) + tf)
        g = row_ix[ord_]
        tk_s = tf[ord_]
        is_new_tok = ~((g[1:] == g[:-1]) & (tk_s[1:] == tk_s[:-1]))
        rank_g = torch.cat([torch.ones(1, dtype=torch.int64,
                                       device=dev),
                            is_new_tok.long()]).cumsum(0) - 1
        row_first = torch.searchsorted(g, g)
        rank_row = rank_g - rank_g[row_first]
        inv = torch.empty_like(ord_)
        inv[ord_] = torch.arange(ord_.numel(), device=dev)
        rank = rank_row[inv]
        col = torch.where(
            has, match.float().argmax(dim=1),
            torch.where(has_empty,
                        (first_empty + rank).clamp(max=self.M - 1),
                        min_col))
        write_tok = ~has & vf
        cell = row_ix * self.M + col
        # column-evict victims (full row -> min column displaced):
        # capture (cell, tok, cnt) BEFORE the write so undo restores
        # exactly; ttot must also drop the displaced mass
        rep = write_tok & ~has_empty
        ttfl = self.ttoks.view(-1)
        tcfl = self.tcnt.view(-1)
        col_snap = (cell[rep],
                    ttfl.index_select(0, cell[rep]).clone(),
                    tcfl.index_select(0, cell[rep]).clone())
        ttv = self.ttoks.view(-1)
        tcv = self.tcnt.view(-1)
        ttv.index_copy_(
            0, cell[write_tok], tf[write_tok].to(torch.int32))
        tcv[cell[write_tok & ~has_empty]] = 0.0
        zv = torch.zeros((), device=dev)
        _group_add(tcv, cell, torch.where(vf, df, zv))
        _group_add(self.ttot.view(-1), row_ix[rep], -col_snap[2])
        _group_add(self.ttot.view(-1), row_ix,
                   torch.where(vf, df, zv))
        oc = self.tocc.view(-1)
        oc.index_copy_(
            0, row_ix,
            torch.where(vf,
                        self.ttot.view(-1).index_select(0, row_ix),
                        oc.index_select(0, row_ix)))
        # intra-batch slot collisions: two distinct new keys (or a new
        # key evicting a found row) can pick the same slot - the losing
        # key's follower landed in the winner's row.  Detect elements
        # whose key didn't stick, undo their misplaced cell exactly,
        # and re-insert them sequentially (rare: ~birthday collisions).
        # Probe + undo are BATCHED - with near-full tables the loser
        # count is ~10% of a block, and per-key probes/syncs were the
        # dominant bulk-ingest cost.
        stuck = vf & (tk_v.index_select(0, row_ix) != pf)
        sidx = torch.nonzero(stuck).squeeze(1)
        extra_victims = []
        ins_list = []
        if sidx.numel():
            # batched undo of misplaced follower writes
            cell_s = cell[sidx]
            row_s = row_ix[sidx]
            df_s = df[sidx]
            _group_add(tcv, cell_s, -df_s)
            under = tcv.index_select(0, cell_s) <= 1e-9
            if bool(under.any()):
                ttv.index_fill_(0, cell_s[under], -1)
            _group_add(self.ttot.view(-1), row_s, -df_s)
            # batched 2-choice probe on the LIVE table (same pick
            # rule as the main pass) - the loser count is ~10% of a
            # block at high load, so per-key probes were the dominant
            # bulk-ingest cost; one D2H pulls every scalar needed
            oi_s = rf[sidx]
            pk_s = pf[sidx]
            s1s, s2s = self._slots2(pk_s)
            k1 = self.tkeys[oi_s, s1s]
            k2 = self.tkeys[oi_s, s2s]
            f1 = k1 == pk_s
            f2 = k2 == pk_s
            found_s = f1 | f2
            e1 = k1 == 0
            e2 = k2 == 0
            pick2 = ((~f1) & f2) | ((~found_s) & (~e1) & e2) \
                | ((~found_s) & (~e1) & (~e2)
                   & (self.tocc[oi_s, s2s] < self.tocc[oi_s, s1s]))
            slot_s = torch.where(pick2, s2s, s1s)
            oi_l = oi_s.tolist()
            sj_l = slot_s.tolist()
            fj_l = found_s.tolist()
            pk_l = pk_s.tolist()
            tf_l = tf[sidx].tolist()
            df_l = df_s.tolist()
            jl = sidx.tolist()
            ck_flat = ctx_keys.reshape(-1)
            # repair-vs-repair conflicts: the batched probe saw
            # pre-repair state, so two losers can target the same slot
            # (or share a key) - flag repeated (oi,slot) targets and
            # re-probe them live inside the loop, preserving the
            # original sequential semantics
            flat_l = (oi_s * self.S + slot_s).tolist()
            _seen: set[int] = set()
            dup_l = []
            for _f in flat_l:
                dup_l.append(_f in _seen)
                _seen.add(_f)
            # occupancy of the PICKED slot at batch-probe time - safe
            # for non-dup elements (a same-flat earlier repair is
            # exactly what dup_l catches)
            emp_l = torch.where(pick2, e2, e1).tolist()
            # follower-column choice for the whole set in one pass;
            # dup elements re-resolve below (their row may have been
            # written by an earlier repair)
            rt_s = self.ttoks[oi_s, slot_s]                # [K, M]
            rc_s = self.tcnt[oi_s, slot_s]
            match_s = rt_s == tf[sidx][:, None]
            em_s = rt_s == -1
            has_s = match_s.any(1)
            col_s = torch.where(
                has_s, match_s.float().argmax(1),
                torch.where(em_s.any(1), em_s.float().argmax(1),
                            rc_s.argmin(1)))
            col_l = col_s.tolist()
            has_l = has_s.tolist()
            # flats actually written by earlier repairs - includes
            # re-probed destinations, so a later element re-checks if
            # an earlier repair landed in its pre-computed slot
            written: set[int] = set()
            for i in range(len(jl)):
                j = jl[i]
                oi, sj, fj = int(oi_l[i]), int(sj_l[i]), bool(fj_l[i])
                pkj = int(pk_l[i])
                tfj = int(tf_l[i])
                dfj = float(df_l[i])
                live = dup_l[i] or (oi * self.S + sj) in written
                if live:
                    # live re-probe - an earlier repair may have taken
                    # this slot after the batch probe was computed
                    s_r, f_r, e_r = self._probe_row_w(
                        oi, torch.tensor([pkj], dtype=torch.int64,
                                         device=dev))
                    sj, fj = int(s_r[0]), bool(f_r[0])
                    ej = bool(e_r[0])
                    has = False
                else:
                    ej = not bool(emp_l[i])
                    # batched rt_s was gathered pre-wipe: for evicting
                    # inserts the row is zeroed below, so the post-
                    # wipe first-empty is always column 0
                    has = bool(has_l[i]) and not ej
                flat_j = oi * self.S + sj
                if not fj:
                    if ej:   # evict victim at the new slot
                        extra_victims.append((
                            torch.tensor([flat_j], dtype=torch.int64,
                                         device=dev),
                            self.traw.view(-1)[flat_j]
                                .reshape(1).clone(),
                            self.ttoks.view(-1, self.M)[flat_j]
                                .reshape(1, -1).clone(),
                            self.tcnt.view(-1, self.M)[flat_j]
                                .reshape(1, -1).clone(),
                            self.ttot.view(-1)[flat_j]
                                .reshape(1).clone()))
                        self.ttoks[oi, sj] = -1
                        self.tcnt[oi, sj] = 0.0
                        self.ttot[oi, sj] = 0.0
                    self.tkeys[oi, sj] = pkj
                    self.traw[oi, sj] = ck_flat[j]
                    if not ej:
                        ins_list.append(oi)
                if live:
                    # column resolve on the live row - one host pull
                    # beats 4-5 kernel launches + syncs of GPU indexing
                    row_l = self.ttoks[oi, sj].tolist()
                    if tfj in row_l:
                        has = True
                        cj = row_l.index(tfj)
                    elif -1 in row_l:
                        has = False
                        cj = row_l.index(-1)
                    else:
                        has = False
                        cnt_l = self.tcnt[oi, sj].tolist()
                        cj = cnt_l.index(min(cnt_l))
                elif ej:
                    cj = 0          # wiped row -> first empty
                else:
                    cj = int(col_l[i])
                if not has:
                    self.ttoks[oi, sj, cj] = tfj
                self.tcnt[oi, sj, cj] += dfj
                self.ttot[oi, sj] += dfj
                self.tocc[oi, sj] = self.ttot[oi, sj]
                written.add(flat_j)
        if extra_victims:
            evict_snap = tuple(
                torch.cat([evict_snap[i]] +
                          [v[i] for v in extra_victims])
                for i in range(5))
        # per-row distinct inserts (for `used` bookkeeping) - one
        # bincount; caller does a single .tolist() D2H
        ins = torch.unique(row_ix[new & ~evict])
        if ins_list:
            ins = torch.unique(torch.cat(
                [ins, torch.tensor(ins_list, dtype=torch.int64,
                                   device=dev)]))
        return torch.bincount(ins // self.S, minlength=self.R), \
            evict_snap, col_snap

    def add_rows(self, ctx_keys: torch.Tensor, toks: torch.Tensor,
                 d: torch.Tensor,
                 valid: torch.Tensor) -> torch.Tensor:
        """Sync-free write across ALL orders at once: ctx_keys/toks/d
        are [O]-shaped (one write per order), ``valid`` masks dead
        orders.  Returns the [O] new-key mask for host bookkeeping."""
        pk = self.ctx_hash_all(ctx_keys)
        slot, found, evict = self.probe_write(pk)
        new = (~found) & valid
        self.tkeys[self._ordx, slot] = torch.where(
            new, pk, self.tkeys[self._ordx, slot])
        self.traw[self._ordx, slot] = torch.where(
            new, ctx_keys, self.traw[self._ordx, slot])
        wipe = (evict & valid)[:, None]           # victim row reset
        self.ttoks[self._ordx, slot] = torch.where(
            wipe, torch.full((), -1, dtype=torch.int32,
                             device=self.dev),
            self.ttoks[self._ordx, slot])
        self.tcnt[self._ordx, slot] = torch.where(
            wipe, torch.zeros((), device=self.dev),
            self.tcnt[self._ordx, slot])
        self.ttot[self._ordx, slot] = torch.where(
            evict & valid, torch.zeros((), device=self.dev),
            self.ttot[self._ordx, slot])
        rows_t = self.ttoks[self._ordx, slot]       # [O, M]
        rows_c = self.tcnt[self._ordx, slot]        # [O, M]
        match = rows_t == toks[:, None].to(torch.int32)
        has = match.any(dim=1)
        empty = rows_t == -1
        has_empty = empty.any(dim=1)
        first_empty = empty.float().argmax(dim=1)
        min_col = rows_c.argmin(dim=1)
        col = torch.where(has, match.float().argmax(dim=1),
                          torch.where(has_empty, first_empty, min_col))
        replace = (~has) & (~has_empty) & valid
        vic_c = rows_c.gather(1, col[:, None]).squeeze(1)
        rt_flat = self.ttoks.view(self.O, -1)
        rc_flat = self.tcnt.view(self.O, -1)
        flat = slot * self.M + col
        # evicted-min replacement: overwrite token, zero its count
        rep_tok = torch.where(replace, toks.to(torch.int32),
                              torch.zeros(self.O, dtype=torch.int32,
                                          device=self.dev))
        rt_flat[self._ordx, flat] = torch.where(
            replace, rep_tok,
            rt_flat[self._ordx, flat])
        rc_flat[self._ordx, flat] = torch.where(
            replace, torch.zeros((), device=self.dev),
            rc_flat[self._ordx, flat])
        rc_flat[self._ordx, flat] += torch.where(
            valid, d, torch.zeros((), device=self.dev))
        self.ttot[self._ordx, slot] += torch.where(
            replace, -vic_c, torch.zeros((), device=self.dev))
        self.ttot[self._ordx, slot] += torch.where(
            valid, d, torch.zeros((), device=self.dev))
        self.tocc[self._ordx, slot] = self.ttot[self._ordx, slot]
        return new & ~evict

    def read_pk(self, k: int, ctx_keys: torch.Tensor,
                toks: torch.Tensor) -> torch.Tensor:
        """Count of (key,tok) for a batch - reward/gap probes."""
        oi = self.kidx[k]
        pk = self.ctx_hash(k, ctx_keys)
        slot, found = self._probe(self.tkeys[oi], pk)
        rows_t = self.ttoks[oi][slot]
        match = rows_t == toks[:, None].to(torch.int32)
        c = torch.where(match, self.tcnt[oi][slot],
                        torch.zeros((), device=self.dev)).sum(dim=1)
        return torch.where(found, c, torch.zeros((), device=self.dev))

    def subtract(self, k: int, ctx_keys: torch.Tensor,
                 toks: torch.Tensor, d: torch.Tensor) -> int:
        """Decrement existing (key,tok) counts only - undo path; never
        inserts.  Returns #cells actually decremented."""
        oi = self.kidx[k]
        pk = self.ctx_hash(k, ctx_keys)
        slot, found = self._probe(self.tkeys[oi], pk)
        if not bool(found.any()):
            return 0
        rows_t = self.ttoks[oi][slot]
        match = rows_t == toks[:, None].to(torch.int32)
        msum = match.any(dim=1) & found
        if not bool(msum.any()):
            return 0
        cols = match.float().argmax(dim=1)
        flat = (slot * self.M + cols)[msum]
        _group_add(self.tcnt[oi].view(-1), flat, -d[msum])
        self.tcnt[oi].clamp_(min=0.0)
        _group_add(self.ttot[oi], slot[msum], -d[msum])
        self.ttot[oi].clamp_(min=0.0)
        # hygiene: columns emptied by the subtract revert to -1; a row
        # that loses all mass is cleared wholesale (CPU dicts pop empty
        # cells - the GPU table must match or undo leaves dead keys).
        # Thresholds are above fp32 index_add residue (~1e-7) but far
        # below the minimum real write (surprise_floor ~0.05).
        dead = self.tcnt[oi] <= 1e-6
        self.ttoks[oi][dead] = -1
        rows_s = slot[msum].unique()
        empt = rows_s[self.ttot[oi][rows_s] <= 1e-5]
        if empt.numel():
            self.tkeys[oi, empt] = 0
            self.traw[oi, empt] = 0
            self.ttoks[oi, empt] = -1
            self.tcnt[oi, empt] = 0.0
            self.ttot[oi, empt] = 0.0
            self.used[oi] -= int(empt.numel())
        self.tocc[oi].copy_(self.ttot[oi])
        return int(msum.sum())

    def _pick_slot(self, oi: int, pk: torch.Tensor) -> tuple[
            int, bool]:
        """Undo helper: slot for a re-insert of a displaced row.
        Prefers the slot still holding the key, then an empty slot,
        then the lower-load candidate (exact only if it's dead)."""
        s1, s2 = self._slots2(pk)
        s1i, s2i = int(s1[0]), int(s2[0])
        pkv = int(pk[0])
        k1 = int(self.tkeys[oi, s1i])
        k2 = int(self.tkeys[oi, s2i])
        if k1 == pkv or k1 == 0:
            return s1i, True
        if k2 == pkv or k2 == 0:
            return s2i, True
        t1 = float(self.tocc[oi, s1i])
        t2 = float(self.tocc[oi, s2i])
        s = s1i if t1 <= t2 else s2i
        return s, float(self.tocc[oi, s]) == 0.0

    def restore_row(self, flat: int, raw: int, toks: torch.Tensor,
                    cnts: torch.Tensor, tot: float) -> bool:
        """Re-insert an evicted follower row (undo path).  Exact when
        the victim's slot is free or holds a dead (subtracted) row."""
        oi = int(flat) // self.S
        dev = self.dev
        rt = torch.tensor([_s64(raw)], dtype=torch.int64, device=dev)
        pk = _mix64(rt * _MIX1 + self._salt[oi])
        s, exact = self._pick_slot(oi, pk)
        if int(self.tkeys[oi, s]) == 0:
            self.used[oi] += 1
        self.tkeys[oi, s] = pk[0]
        self.traw[oi, s] = _s64(raw)
        self.ttoks[oi, s] = toks
        self.tcnt[oi, s] = cnts
        self.ttot[oi, s] = tot
        self.tocc[oi, s] = tot
        return exact

    def decay(self, k: int, floor: float = 0.25) -> None:
        oi = self.kidx[k]
        self.tcnt[oi].mul_(0.5)
        self.tcnt[oi][self.tcnt[oi] < floor] = 0.0
        self.ttot[oi].mul_(0.5)
        self.tocc[oi].copy_(self.ttot[oi])

    def rehash(self, k: int) -> None:
        """Rebuild dropping rows whose counts all decayed to zero."""
        oi = self.kidx[k]
        keys = self.tkeys[oi]
        live = (keys != 0) & (self.ttot[oi] > 0)
        pk = keys[live].clone()
        rw = self.traw[oi][live].clone()
        rows_t = self.ttoks[oi][live].clone()
        rows_c = self.tcnt[oi][live].clone()
        tots = self.ttot[oi][live].clone()
        self.tkeys[oi].zero_()
        self.traw[oi].zero_()
        self.ttoks[oi].fill_(-1)
        self.tcnt[oi].zero_()
        self.ttot[oi].zero_()
        self.used[oi] = 0
        if pk.numel():
            slot, _, _ = self._probe_row_w(oi, pk)
            keys[slot] = pk
            self.traw[oi][slot] = rw
            self.ttoks[oi][slot] = rows_t
            self.tcnt[oi][slot] = rows_c
            self.ttot[oi][slot] = tots
            self.tocc[oi][slot] = tots
            self.used[oi] = int(pk.numel())

    def triplets(self) -> dict:
        """{k: (ctx_keys_u64, toks_i64, cnts_f32)} decoded rows.
        Vectorized row dump - per-element Python loops over ~100M
        follower cells made snapshot saves minutes-long."""
        out = {}
        for k in self.orders:
            oi = self.kidx[k]
            nz = (self.tkeys[oi] != 0) & (self.ttot[oi] > 0)
            rks = self.traw[oi][nz].cpu().numpy()
            rt = self.ttoks[oi][nz].cpu().numpy()
            rc = self.tcnt[oi][nz].cpu().numpy()
            kk = np.repeat(
                np.asarray(rks, dtype=np.uint64), self.M)
            tt = np.asarray(rt, dtype=np.int64).reshape(-1)
            cc = np.asarray(rc, dtype=np.float32).reshape(-1)
            m = (tt >= 0) & (cc != 0)
            out[k] = (kk[m], tt[m], cc[m])
        return out

    def load_triplets(self, tri: dict,
                      chunk: int = 2_000_000) -> None:
        dev = self.dev
        for k in self.orders:
            ent = tri.get(k, tri.get(str(k)))
            if not ent:
                continue
            kk_np, tt_np, cc_np = ent
            for s in range(0, len(kk_np), chunk):
                kk = torch.from_numpy(
                    np.ascontiguousarray(
                        kk_np[s:s + chunk], dtype=np.int64)).to(dev)
                tt = torch.from_numpy(
                    np.ascontiguousarray(
                        tt_np[s:s + chunk], dtype=np.int64)).to(dev)
                dd = torch.from_numpy(
                    np.ascontiguousarray(
                        cc_np[s:s + chunk], dtype=np.float32)).to(dev)
                self.used[self.kidx[k]] += int(
                    self.add(k, kk, tt, dd))

