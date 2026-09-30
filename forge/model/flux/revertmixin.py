"""Journal undo / audit for FluxLM."""
from __future__ import annotations

import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .helpers import (_HASH_P, _MASK64, _MIX1, _MIX2,
                      _group_add, _load_teacher_embedding, _mix64, _s64,
                      _scatter_sum)
from .config import FluxConfig
from .journal import FluxJournal
from .tables import _GPUCellTables
from .kernels import HAS_TRITON

if HAS_TRITON:
    from .kernels import (_flux_deep_votes_kernel,
                          _flux_logit_tail_kernel)


class _RevertMixin:

    def _undo_block(self, blk: dict) -> None:
        """Exact inverse of one bulk-ingest block (state snapshot)."""
        c = self.config
        dev = self._dev
        base = blk["count"]
        # subtract table writes (rows found by deterministic re-probe);
        # deltas are the surprise-scaled magnitudes applied forward.
        # Block tensors are stored host-side - re-upload on demand.
        kk_all, tt_all, valid_o = (t.to(dev)
                                   for t in blk["writes_all"])
        s_v = blk.get("surprise")
        if s_v is not None:
            s_v = s_v.to(dev)
        for oi, k in enumerate(self.config.orders):
            idx = torch.nonzero(valid_o[oi]).squeeze(1)
            if not idx.numel():
                continue
            kk = kk_all[oi, idx]
            d = (s_v[idx] if s_v is not None
                 else torch.ones(kk.numel(), dtype=torch.float32,
                                 device=dev))
            self._gt.subtract(k, kk, tt_all[idx], d)
        # restore follower columns displaced by evict-min writes
        cv = blk.get("col_evicted")
        if cv is not None and cv[0].numel():
            cells, toks_v, cnts_v = (t.to(dev) for t in cv)
            gt = self._gt
            gt.ttoks.view(-1).index_copy_(
                0, cells, toks_v.to(torch.int32))
            gt.tcnt.view(-1).index_copy_(0, cells, cnts_v)
            _group_add(gt.ttot.view(-1), cells // gt.M, cnts_v)
            gt.tocc.copy_(gt.ttot)
        # csem rows
        rcs = self._gt.row_csem
        ckr = self._gt.tkeys[rcs]
        crr = self._gt.traw[rcs]
        for s_, prev in blk["csem"]:
            if prev is None:
                ckr[s_] = 0
                crr[s_] = 0
                self._csem_rows[s_].zero_()
                self._gt.tocc[rcs, s_] = 0
                self._gt.used[rcs] -= 1
            else:
                self._csem_rows[s_] = torch.from_numpy(
                    prev.astype(np.float16)).to(dev)
        # epi positions (deferred entries: remove by ctx_end)
        for key, p in reversed(blk["epi"]):
            self._epi_remove(key, p)
        # knn bank rows: restore overwritten rows, clear fresh ones
        knn_blk = blk.get("knn")
        if knn_blk is not None:
            kslots, kalive, kppos, kpsucc, kpfp = (
                t.to(dev) for t in knn_blk)
            self._knn_pos.index_copy_(0, kslots, torch.where(
                kalive, kppos, torch.full((), -1, dtype=torch.int64,
                                          device=dev)))
            self._knn_succ.index_copy_(0, kslots, torch.where(
                kalive, kpsucc, torch.full((), -1, dtype=torch.int32,
                                           device=dev)))
            self._knn_fp.index_copy_(0, kslots, torch.where(
                kalive[:, None], kpfp,
                torch.zeros((), dtype=self._knn_fp.dtype,
                            device=dev)))
            self._knn_cur -= int(kslots.numel())
            if self._gpu:
                self._knncur_g.fill_(self._knn_cur)
        # topic rows: subtract the exact deltas applied forward
        # (stored fp32 in the block - no re-derivation, bit-exact)
        ids_np = blk["ids"]
        ids_t = torch.from_numpy(ids_np.astype(np.int64)).to(dev)
        deltas = blk.get("deltas")
        if deltas and isinstance(deltas[0], tuple):
            # grouped format: (uniq_key, damped_row_delta) per chunk -
            # subtract the exact damped sums that were applied
            for u_c, d_c in deltas:
                u = u_c.to(dev)
                dd = d_c.to(dev)
                _group_add(self._A, u, -dd)
                _group_add(self._Ac, u, -(dd @ self._Pcsem))
                self._An.index_copy_(
                    0, u, self._A.index_select(0, u).norm(dim=1))
                self._Acn.index_copy_(
                    0, u, self._Ac.index_select(0, u).norm(dim=1))
        elif deltas:
            dall = torch.cat([d.to(dev) for d in deltas])
            _group_add(self._A, ids_t, -dall)
            _group_add(self._Ac, ids_t, -(dall @ self._Pcsem))
            uniq = torch.unique(ids_t)
            self._An.index_copy_(
                0, uniq, self._A.index_select(0, uniq).norm(dim=1))
            self._Acn.index_copy_(
                0, uniq,
                self._Ac.index_select(0, uniq).norm(dim=1))
        else:  # legacy blocks: replay chunk c-vectors, subtract adds
            CH = c.gpu_bulk_chunk
            c_save = self._c
            for ci0, c_start in enumerate(blk["c_starts"]):
                xchunk = ids_t[ci0 * CH:(ci0 + 1) * CH]
                self._c = torch.from_numpy(c_start).to(dev)
                cprev = self._chunk_cprev(xchunk)
                dc_ = (-cprev * c.hebb_lr).float()
                _group_add(self._A, xchunk, dc_)
                _group_add(self._Ac, xchunk, dc_ @ self._Pcsem)
                axn = self._A.index_select(0, xchunk)
                self._An.index_copy_(0, xchunk, axn.norm(dim=1))
                self._Acn.index_copy_(
                    0, xchunk,
                    self._Ac.index_select(0, xchunk).norm(dim=1))
            self._c = c_save
        # scalar/vector state - tensors may be CPU-resident after a
        # snapshot/load (torch.save drops the device), so re-home them
        self._c = blk["c"].to(dev)
        c2 = blk.get("c2")
        self._c2 = (c2.to(dev) if c2 is not None
                    else torch.zeros_like(self._c2))
        self._proto = blk["proto"].to(dev)
        self._w = blk["w"]
        self._hedge_mix = blk.get("hedge_mix", 0.0)
        self._wR = blk.get("wR", np.tile(blk["w"], (2, 1)))
        self._hmxR = blk.get("hmxR", np.zeros(2))
        self._regime = blk.get("regime", 0)
        self._uni = blk["uni"]
        self._unig = blk["unig"].to(dev)
        self._log_uni_num = blk["log_uni_num"].to(dev)
        self._total -= blk["n"]
        self._count = base
        self._ep_start = blk["ep_start"]
        self._last_seen = blk["last_seen"]
        self._seen_g = blk["seen_g"].to(dev)
        if self._gpu:
            self._wg = torch.from_numpy(
                blk["w"].astype(np.float64)).to(dev)
            self._hmg.fill_(self._hedge_mix)
            self._wgR = torch.from_numpy(
                self._wR.astype(np.float64)).to(dev)
            self._hmxRg = torch.from_numpy(
                self._hmxR.astype(np.float64)).to(dev)
            self._regime_g.fill_(self._regime)
            self._reg_used_g.fill_(self._regime)
            self._pos_g.fill_(base)
            self._ep_g.fill_(blk["ep_start"])
            self._tot_g.fill_(float(self._total))
            eg = blk.get("epi_gpu")
            if eg is not None:
                if len(eg) == 9 and eg[5].dtype != torch.bool:
                    # candidate-slot format: wholesale restore of
                    # (key, raw, pos, cnt, succ, cert) for every slot
                    # either probe could have touched - subsumes
                    # victim-row restore for evictions
                    _dk, _dp, us, pos_r, cnt_r, pre_k, pre_raw, \
                        succ_r, cert_r = eg
                    re_ = self._gt.row_epi
                    us_t = us.to(dev)
                    pk0 = pre_k.to(dev)
                    cur = self._gt.tkeys[re_, us_t]
                    newly = (pk0 == 0) & (cur != 0)
                    self._gt.used[re_] -= int(newly.sum())
                    self._gt.tkeys[re_, us_t] = pk0
                    self._gt.traw[re_, us_t] = pre_raw.to(dev)
                    self._epi_pos[us_t] = pos_r.to(dev)
                    self._epi_cnt[us_t] = cnt_r.to(dev)
                    if succ_r is not None:
                        self._epi_succ[us_t] = succ_r.to(dev)
                    if cert_r is not None:
                        self._epi_cert[us_t] = cert_r.to(dev)
                    self._gt.tocc[re_, us_t] = \
                        cnt_r.to(dev).float()
                elif len(eg) >= 7:
                    # per-slot pre-write snapshot - restore wholesale
                    # (exact incl. ring cursor); newer blocks also
                    # carry the succ/cert payload columns
                    _dk, _dp, us, pos_r, cnt_r, new_u, inc_u = eg[:7]
                    succ_r = eg[7] if len(eg) > 7 else None
                    cert_r = eg[8] if len(eg) > 8 else None
                    re_ = self._gt.row_epi
                    us_t = us.to(dev)
                    nw = new_u.to(dev)
                    if bool(nw.any()):
                        clr = us_t[nw]
                        self._gt.tkeys[re_, clr] = 0
                        self._gt.traw[re_, clr] = 0
                    inc = inc_u.to(dev)
                    self._gt.used[re_] -= int(inc.sum())
                    self._epi_pos[us_t] = pos_r.to(dev)
                    self._epi_cnt[us_t] = cnt_r.to(dev)
                    if succ_r is not None:
                        self._epi_succ[us_t] = succ_r.to(dev)
                    if cert_r is not None:
                        self._epi_cert[us_t] = cert_r.to(dev)
                    self._gt.tocc[re_, us_t] = \
                        cnt_r.to(dev).float()
                else:
                    # legacy (dk, dp) format - clear matching positions
                    dk_np, dp_np = eg
                    pk_e = self._epi_hash(torch.from_numpy(
                        dk_np.astype(np.int64)).to(dev))
                    se, fe = self._epi_probe(pk_e)
                    if bool(fe.any()):
                        rows = self._epi_pos[se[fe]]
                        m_ = rows == torch.from_numpy(
                            dp_np.astype(np.int64)).to(dev)[fe][:, None]
                        rows[m_] = -1
                        self._epi_pos[se[fe]] = rows
            # restore rows this block's inserts evicted (victims were
            # displaced first in forward order -> restored last here)
            evx = blk.get("evicted")
            if evx is not None and evx[0].numel():
                row_ix, raw, toks, cnts, tot = (
                    t.to(dev) for t in evx)
                for j in range(row_ix.numel()):
                    self._gt.restore_row(
                        int(row_ix[j]),
                        int(raw[j]) & 0xFFFFFFFFFFFFFFFF,
                        toks[j], cnts[j], float(tot[j]))
            eev = blk.get("epi_ev")
            if eev is not None and eev[0].numel():
                sv, raw, pos_r, cnt_r = (t.to(dev)
                                         for t in eev[:4])
                succ_r = eev[4].to(dev) if len(eev) > 4 else None
                cert_r = (eev[5].to(dev)
                          if len(eev) > 5 and eev[5] is not None
                          else None)
                re_ = self._gt.row_epi
                for j in range(sv.numel()):
                    # cells written by this block (pos >= base) are
                    # being undone - keep only pre-block positions
                    keep = (pos_r[j] >= 0) & (pos_r[j] < base)
                    cnt_j = int(keep.sum())   # cursor-equivalent mod ER
                    if cnt_j == 0:
                        # victim held only block-era cells - it did
                        # not exist pre-block; restoring it would
                        # create a zombie row (and could displace a
                        # live row via _pick_slot's 2-choice fallback)
                        continue
                    pk = self._epi_hash(raw[j].reshape(1))
                    s, _ = self._gt._pick_slot(re_, pk)
                    if int(self._gt.tkeys[re_, s]) == 0:
                        self._gt.used[re_] += 1
                    self._gt.tkeys[re_, s] = pk[0]
                    self._gt.traw[re_, s] = raw[j]
                    self._epi_pos[s] = torch.where(
                        keep, pos_r[j],
                        torch.full((), -1, dtype=torch.int64,
                                   device=dev))
                    if succ_r is not None:
                        self._epi_succ[s] = torch.where(
                            keep, succ_r[j],
                            torch.full((), -1, dtype=torch.int32,
                                       device=dev))
                    if cert_r is not None:
                        self._epi_cert[s] = torch.where(
                            keep[:, None], cert_r[j],
                            torch.zeros((), dtype=torch.int64,
                                        device=dev))
                    self._epi_cnt[s] = cnt_j
                    self._gt.tocc[re_, s] = float(cnt_j)
            # chunkwise gsyn updates journal a pre-state snapshot -
            # wholesale restore, exact regardless of the frozen-S
            # linearization the forward pass uses
            gs = blk.get("gsyn")
            if gs is not None:
                self._S.copy_(gs[0].to(dev))
                if gs[1] is not None and c.gsyn_slow_decay > 0:
                    self._Ss.copy_(gs[1].to(dev))
            # block-local journal: per-write entries (G/gE/C/K/W/H)
            # collected on the block instead of journal.entries so bulk
            # ingests can't evict this block's own "bc" header - replay
            # in reverse, same handlers as the global journal
            for se in reversed(blk.get("sub", ())):
                self._undo_entry(se)
            cev = blk.get("csem_ev")
            if cev is not None and cev[0].numel():
                sv, raw, vec_r, oc_r = cev
                rc_ = self._gt.row_csem
                for j in range(sv.numel()):
                    pk = self._csem_hash(raw[j].reshape(1))
                    s, _ = self._gt._pick_slot(rc_, pk)
                    if int(self._gt.tkeys[rc_, s]) == 0:
                        self._gt.used[rc_] += 1
                    self._gt.tkeys[rc_, s] = pk[0]
                    self._gt.traw[rc_, s] = raw[j]
                    self._csem_rows[s] = vec_r[j]
                    self._gt.tocc[rc_, s] = oc_r[j]
        self._rebuild_recent()
        while len(self._gaps) > blk["gaps"]:
            self._gaps.pop()

    def _undo_entry(self, e: tuple) -> bool:
        """Apply the inverse of one journal entry. Returns exactness."""
        op = e[0]
        if op == "t":
            # rewind stream head; ring bytes stay (harmless once head
            # moved) and invert the topic EMA exactly:
            # c_i = lam*c_{i-1} + (1-lam)*r_x  =>  c_{i-1} = (c_i -
            # (1-lam)*r_x)/lam
            _, tok, _tag = e
            lam = self.config.topic_decay
            if lam > 1e-9:
                self._c.sub_(self._R[tok].float() * (1.0 - lam)).div_(lam)
                if self.config.bigram_sketch:
                    # same EMA inverse on the sketch: the pair that was
                    # bound at push time is bind(ring[p], tok) with
                    # p = count-2 - zero when p was pre-episode
                    p = self._count - 2
                    if p >= self._ep_start:
                        b2 = self._bind_pair(
                            int(self._ring[
                                p % self.config.ring_capacity]), tok)
                    else:
                        b2 = torch.zeros_like(self._c2)
                    self._c2.sub_(b2 * (1.0 - lam)).div_(lam)
            # proto EMA inverse is approximate (A[tok] may have drifted)
            pd = self.config.proto_decay
            if pd > 1e-9:
                self._proto.sub_(self._A[tok] * (1.0 - pd)).div_(pd)
            self._count -= 1
            if self._gpu:
                self._pos_g -= 1
            return True
        if op == "b":
            self._ep_start = e[1]
            self._c.copy_(torch.from_numpy(
                e[2].astype(np.float32)).to(self._dev))
            if len(e) > 3:
                self._c2.copy_(torch.from_numpy(
                    e[3].astype(np.float32)).to(self._dev))
            else:
                self._c2.zero_()   # pre-sketch journal
            if self._gpu:
                self._ep_g.fill_(e[1])
            return True
        if op == "G":
            # gcell write: restore the pre-write successor score and
            # cell total exactly; drop the cell if this write created
            # it (fifo tail-pop only if the cell was created here -
            # LIFO undo keeps insertion order exact)
            _, code, tok, old, old_tot = e
            cell = self._gc.get(code)
            if cell is None:
                return False
            if old:
                cell[tok] = old
            else:
                cell.pop(tok, None)
            if old_tot:
                self._gct[code] = old_tot
            else:
                self._gct.pop(code, None)
            if not cell:
                del self._gc[code]
                self._gct.pop(code, None)
                self._gc_bucket_remove(code)
                if self._gc_fifo and self._gc_fifo[-1] == code:
                    self._gc_fifo.pop()
            return True
        if op == "gE":
            _, code, cell, tot = e
            self._gc[code] = cell
            self._gct[code] = tot
            self._gc_fifo.appendleft(code)
            self._gc_buckets.setdefault(
                code >> self.config.gcell_bucket_shift,
                set()).add(code)
            return True
        if op == "GS":
            # gsyn delta write: forward was S = g*S_pre + c*(v_err x k)
            # -> S_pre = (S - c*v_err x k)/g.  Factors ride an fp16
            # ring - fp16-precision inverse, same class as 'a'.
            _, slot = e
            if slot < 0:
                return False
            d = self.config.topic_dim
            row = self._gsyn_ring[slot]
            g = self.config.gsyn_decay
            gs = self.config.gsyn_slow_decay
            if abs(g) < 1e-9 or (gs > 0 and abs(gs) < 1e-9):
                return False          # decay 0 update isn't invertible
            ve = torch.from_numpy(
                np.asarray(row[2:2 + d], dtype=np.float32)).to(
                    self._dev)
            k = torch.from_numpy(
                np.asarray(row[2 + d:], dtype=np.float32)).to(
                    self._dev)
            outer = torch.outer(ve, k)
            self._S.sub_(outer, alpha=float(row[0])).div_(g)
            if gs > 0:
                self._Ss.sub_(outer, alpha=float(row[1])).div_(gs)
            return True
        if op == "u":
            if self._uni[e[1]] >= 1.0:
                self._uni[e[1]] -= 1.0
                self._total -= 1
                self._log_uni_num[e[1]] = math.log(
                    self._uni[e[1]] + self.config.uni_alpha)
                if self._gpu:
                    self._unig[e[1]] -= 1.0
                    self._tot_g -= 1.0
                return True
            return False
        if op == "c":
            _, k, key, tok, delta = e
            if self._gpu:
                ctx_t = torch.tensor([_s64(key)], dtype=torch.int64,
                                     device=self._dev)
                tok_t = torch.tensor([tok], dtype=torch.int64,
                                     device=self._dev)
                d_t = torch.tensor([delta], dtype=torch.float32,
                                   device=self._dev)
                return self._gt.subtract(k, ctx_t, tok_t, d_t) > 0
            cell = self._tables[k].get(key)
            if cell is None or tok not in cell:
                return False
            cell[tok] -= delta
            tots = self._totals[k]
            tots[key] = tots.get(key, 0.0) - delta
            # epsilon, not zero: subtracting N surprise-scaled deltas
            # leaves ~1e-16 fp residue that would ghost the cell
            if cell[tok] <= 1e-9:
                del cell[tok]
                if not cell:
                    del self._tables[k][key]
                    tots.pop(key, None)
            return True
        if op == "WH":
            # distill_head: restore pre-train Wh/Wh1 (or absence).
            # Entry is ("WH", blob, blob1); 2-tuples predate Wh1.
            # Wh is [V, *]; Wh1 is [*, head_in_dim].
            def _unblob(b, cols_first):
                if b is None:
                    return None
                a = np.frombuffer(b, dtype=np.float16).copy()
                a = (a.reshape(self.vocab_size, -1) if cols_first
                     else a.reshape(-1, self._head_in_dim()))
                return torch.from_numpy(a).float().to(self._dev)
            self._Wh = _unblob(e[1], True)
            self._Wh1 = (_unblob(e[2], False) if len(e) > 2
                         else None)
            # captured CUDA graphs bake the old _Wh pointer (or its
            # absence) - drop them so the next predict recaptures
            self._graphs = {}
            self._pgraph = None
            return True
        if op == "Ab":
            # distill_hebbian: restore a snapshotted _A row (taken on
            # first touch) and recompute its maintained projections
            _, tok, prev = e
            row = torch.as_tensor(
                np.asarray(prev), dtype=torch.float32).to(self._dev)
            self._A[tok] = row
            if self._gpu:
                self._An[tok] = row.norm()
                self._Ac[tok] = row @ self._Pcsem
                self._Acn[tok] = self._Ac[tok].norm()
            return True
        if op == "C":
            # sparse-tap cell write - host tables only (GPU paths take
            # the uni fallback and never emit "C")
            _, j, key, tok, delta = e
            taps = self.config.sparse_taps[j]
            cell = self._ptabs[taps].get(key)
            if cell is None or tok not in cell:
                return False
            cell[tok] -= delta
            tots = self._ptots[taps]
            tots[key] = tots.get(key, 0.0) - delta
            if cell[tok] <= 1e-9:
                del cell[tok]
                if not cell:
                    del self._ptabs[taps][key]
                    tots.pop(key, None)
            return True
        if op == "K":
            # class-order cell write - host tables, same undo shape
            _, k, key, tok, delta = e
            cell = self._gtabs[k].get(key)
            if cell is None or tok not in cell:
                return False
            cell[tok] -= delta
            tots = self._gtots[k]
            tots[key] = tots.get(key, 0.0) - delta
            if cell[tok] <= 1e-9:
                del cell[tok]
                if not cell:
                    del self._gtabs[k][key]
                    tots.pop(key, None)
            return True
        if op == "W":
            # word-order cell write - host tables, same undo shape
            _, j, key, tok, delta = e
            cell = self._wtabs[j].get(key)
            if cell is None or tok not in cell:
                return False
            cell[tok] -= delta
            tots = self._wtots[j]
            tots[key] = tots.get(key, 0.0) - delta
            if cell[tok] <= 1e-9:
                del cell[tok]
                if not cell:
                    del self._wtabs[j][key]
                    tots.pop(key, None)
            return True
        if op == "H":
            # hybrid class+raw cell write - host tables, same undo
            _, j, key, tok, delta = e
            cell = self._htabs[j].get(key)
            if cell is None or tok not in cell:
                return False
            cell[tok] -= delta
            tots = self._htots[j]
            tots[key] = tots.get(key, 0.0) - delta
            if cell[tok] <= 1e-9:
                del cell[tok]
                if not cell:
                    del self._htabs[j][key]
                    tots.pop(key, None)
            return True
        if op in ("Cb", "Kb", "Wb", "Hb"):
            # packed block write: (key, succ, delta) arrays from one
            # dedup-grouped table pass - replay in reverse, same undo
            # arithmetic as the per-write entries
            _, jj, kk_, xx_, dd_ = e
            c = self.config
            if op == "Cb":
                taps = c.sparse_taps[jj]
                tab, tots = self._ptabs[taps], self._ptots[taps]
            elif op == "Kb":
                tab, tots = self._gtabs[jj], self._gtots[jj]
            elif op == "Wb":
                tab, tots = self._wtabs[jj], self._wtots[jj]
            else:
                tab, tots = self._htabs[jj], self._htots[jj]
            ok = True
            for key, tok, delta in zip(kk_[::-1].tolist(),
                                       xx_[::-1].tolist(),
                                       dd_[::-1].tolist()):
                cell = tab.get(key)
                if cell is None or tok not in cell:
                    ok = False
                    continue
                cell[tok] -= delta
                tots[key] = tots.get(key, 0.0) - delta
                if cell[tok] <= 1e-9:
                    del cell[tok]
                    if not cell:
                        del tab[key]
                        tots.pop(key, None)
            return ok
        if op == "eE":
            # restore an entry evicted by deque-overflow/FIFO-cap -
            # journaled BEFORE the displacing write, so on undo it is
            # re-inserted as the oldest row (appendleft)
            _, key, pos, succ, certs = e
            dq = self._epi.get(key)
            if dq is None:
                dq = self._epi[key] = deque(
                    maxlen=self.config.epi_max_per_key)
            if len(dq) < self.config.epi_max_per_key:
                self._epi_n += 1
            else:
                dq.pop()              # bound: drop newest to fit
            dq.appendleft((pos, succ, tuple(certs)))
            self._epi_fifo.appendleft((key, pos))
            self._epi_sync_row(key)     # GPU row tracks host dict
            return True
        if op == "e":
            _, key, pos = e
            if not self._epi_remove(key, pos):
                return False
            if self._gpu:
                pk = self._epi_hash(torch.tensor(
                    [_s64(key)], dtype=torch.int64, device=self._dev))
                se, fe = self._epi_probe(pk)
                if bool(fe[0]):
                    row = self._epi_pos[se[0]]
                    m = row == pos
                    if bool(m.any()):
                        col = int(m.float().argmax())
                        row[col] = -1
                        self._epi_succ[se[0], col] = -1
                        if col == int(
                                (self._epi_cnt[se[0]] - 1)
                                % self._epi_pos.shape[1]):
                            self._epi_cnt[se[0]] -= 1
            return True
        if op == "k":
            # knn bank row restore.  CPU entries carry the previous row
            # inline: ("k", slot, None|(pos, succ, fp16[d])); GPU
            # entries carry a slot into _kprev_g:
            # ("k", slot, prev_pos, prev_succ, prev_ring_slot|-1)
            slot = e[1]
            if len(e) == 3:
                prev = e[2]
                if prev is None:
                    self._knn_pos[slot] = -1
                    self._knn_succ[slot] = -1
                    self._knn_fp[slot] = 0
                else:
                    ppos, psucc, pfp = prev
                    self._knn_pos[slot] = ppos
                    self._knn_succ[slot] = psucc
                    self._knn_fp[slot] = torch.from_numpy(
                        np.asarray(pfp)).to(self._dev).to(
                            self._knn_fp.dtype)
            else:
                _, slot, ppos, psucc, rslot = e
                if rslot < 0:
                    self._knn_pos[slot] = -1
                    self._knn_succ[slot] = -1
                    self._knn_fp[slot] = 0
                else:
                    self._knn_pos[slot] = ppos
                    self._knn_succ[slot] = psucc
                    self._knn_fp[slot] = self._kprev_g[rslot]
            # every journaled bank write bumped the cursor once -
            # LIFO undo steps it back (exact for sequential writes)
            self._knn_cur = max(0, self._knn_cur - 1)
            if self._gpu:
                self._knncur_g.fill_(self._knn_cur)
            return True
        if op == "r":
            _, tok, prev = e
            if prev < 0:
                self._last_seen.pop(tok, None)
            else:
                self._last_seen[tok] = prev
            if self._gpu:
                self._seen_g[tok] = prev if prev >= 0 else -10 ** 9
            return True
        if op == "s":
            # csem row restore: (pk_or_key, slot_or_-1, prev_row|None)
            _, key_or_pk, slot, prev = e
            if self._gpu:
                if prev is None:
                    self._gt.tkeys[self._gt.row_csem, slot] = 0
                    self._gt.traw[self._gt.row_csem, slot] = 0
                    self._csem_rows[slot].zero_()
                    self._gt.tocc[self._gt.row_csem, slot] = 0
                    self._gt.used[self._gt.row_csem] -= 1
                else:
                    self._csem_rows[slot] = torch.from_numpy(
                        np.asarray(prev, dtype=np.float16)).to(
                            self._dev)
            else:
                if prev is None:
                    self._csem_cpu.pop(key_or_pk, None)
                else:
                    self._csem_cpu[key_or_pk] = np.asarray(
                        prev, dtype=np.float32)
            return True
        if op == "bc":
            self._undo_block(e[1])
            return True
        if op == "sg":
            # gpu csem write: (slot, jcur_ring_slot_or_-1)
            _, slot, rslot = e
            if rslot < 0:
                self._gt.tkeys[self._gt.row_csem, slot] = 0
                self._gt.traw[self._gt.row_csem, slot] = 0
                self._csem_rows[slot].zero_()
                self._gt.tocc[self._gt.row_csem, slot] = 0
                self._gt.used[self._gt.row_csem] -= 1
            else:
                self._csem_rows[slot] = self._csem_prev[rslot]
            return True
        if op == "a":
            _, tok, slot = e
            if slot < 0:
                return False            # vec delta already recycled
            if self._gpu:
                dv = self._avec_g[slot]
                self._A[tok].sub_(dv)
                self._Ac[tok].sub_(dv @ self._Pcsem)
                self._An[tok] = self._A[tok].norm()
                self._Acn[tok] = self._Ac[tok].norm()
            else:
                self._A[tok].sub_(torch.from_numpy(
                    self.journal.vec_ring[slot].astype(np.float32)
                ).to(self._dev))
            return True
        if op in ("h", "h2"):
            if op == "h2":
                _, regime, slot = e
            else:
                _, slot = e
                regime = -1
            if slot < 0:
                return False            # w snapshot already recycled
            if self._gpu:
                w_np = self._w_ring_g[slot, :-1].cpu().numpy() \
                    .astype(np.float64)
                mx = float(self._w_ring_g[slot, -1])
                if regime >= 0:
                    self._wgR[regime] = self._w_ring_g[slot, :-1]
                    self._hmxRg[regime] = mx
                    self._wR[regime] = w_np
                    self._hmxR[regime] = mx
                    self._w = w_np
                    self._hedge_mix = mx
                else:
                    self._wg.copy_(self._w_ring_g[slot, :-1])
                    self._hmg.fill_(self._w_ring_g[slot, -1])
                    self._w = w_np
                    self._hedge_mix = mx
            else:
                w_np = self.journal.w_ring[slot, :-1]
                mx = float(self.journal.w_ring[slot, -1])
                if regime >= 0:
                    self._wR[regime] = w_np
                    self._hmxR[regime] = mx
                    self._w[:] = w_np
                    self._hedge_mix = mx
                else:
                    self._w[:] = w_np
                    self._hedge_mix = mx
            return True
        if op in ("ws", "we"):
            return True                 # structural markers - no state
        if op == "cv":
            # restore a follower column displaced by evict-min write
            _, flat, col, tok, cnt = e
            if not self._gpu:
                return False
            oi = int(flat) // self._gt.S
            s = int(flat) % self._gt.S
            if int(self._gt.tkeys[oi, s]) == 0:
                return False        # host row was fully cleared
            self._gt.ttoks[oi, s, col] = int(tok)
            self._gt.tcnt[oi, s, col] = float(cnt)
            self._gt.ttot[oi, s] += float(cnt)
            self._gt.tocc[oi, s] = float(self._gt.ttot[oi, s])
            return True
        if op == "ev":
            # restore an order-row evicted by a fused-step insert
            _, flat, raw, tot, toks, cnts = e
            if not self._gpu:
                return False
            return self._gt.restore_row(
                flat, raw,
                torch.from_numpy(
                    np.asarray(toks, dtype=np.int32)).to(self._dev),
                torch.from_numpy(
                    np.asarray(cnts, dtype=np.float32)).to(self._dev),
                tot)
        if op == "eve":
            # restore an evicted episodic row (pos + succ + certs)
            _, flat, raw, cnt, positions = e[:5]
            if not self._gpu:
                return False
            oi = self._gt.row_epi
            pk = self._epi_hash(torch.tensor(
                [_s64(raw)], dtype=torch.int64, device=self._dev))
            s, exact = self._gt._pick_slot(oi, pk)
            if int(self._gt.tkeys[oi, s]) == 0:
                self._gt.used[oi] += 1
            self._gt.tkeys[oi, s] = pk[0]
            self._gt.traw[oi, s] = _s64(raw)
            self._epi_pos[s] = torch.from_numpy(
                np.asarray(positions, dtype=np.int64)).to(self._dev)
            if len(e) > 5:
                self._epi_succ[s] = torch.from_numpy(
                    np.asarray(e[5], dtype=np.int32)).to(self._dev)
            if len(e) > 6:
                self._epi_cert[s] = torch.from_numpy(
                    np.asarray(e[6]).reshape(
                        self._epi_cert.shape[1:])) .to(self._dev)
            self._epi_cnt[s] = cnt
            self._gt.tocc[oi, s] = float(cnt)
            return exact
        if op == "evc":
            # restore an evicted csem row
            _, flat, raw, oc, vec = e
            if not self._gpu:
                return False
            oi = self._gt.row_csem
            pk = self._csem_hash(torch.tensor(
                [_s64(raw)], dtype=torch.int64, device=self._dev))
            s, exact = self._gt._pick_slot(oi, pk)
            if int(self._gt.tkeys[oi, s]) == 0:
                self._gt.used[oi] += 1
            self._gt.tkeys[oi, s] = pk[0]
            self._gt.traw[oi, s] = _s64(raw)
            self._csem_rows[s] = torch.from_numpy(
                np.asarray(vec, dtype=np.float16)).to(self._dev)
            self._gt.tocc[oi, s] = oc
            return exact
        if op == "g":
            # graft provenance marker - init-time state change, not a
            # journaled write; nothing to undo
            return True
        return False

    def revert_since(self, n_entries: int | None = None,
                     seq_pos: int | None = None) -> dict:
        """Undo journaled writes.

        ``n_entries``: revert at least N journal entries, rounded up to
            a token boundary (a token's writes stay atomic).
        ``seq_pos``: revert until the stream length equals this position
            (i.e. un-learn the last ``count - seq_pos`` tokens).
        Returns a report {reverted, inexact}.
        """
        entries = self.journal.entries
        reverted = inexact = popped = 0
        # Boundaries: "t"/"bc" (token writes), "b" (episode reset), and
        # "ws"/"we"/"g" (position-free write-block markers).  Write-block
        # markers count as boundaries so a seq_pos revert can never
        # descend into a teach/consolidate block below the floor - those
        # ops belong to revert_writes()/revert_tag(), not a position
        # floor.
        while entries:
            at_boundary = entries[-1][0] in ("t", "b", "bc",
                                             "ws", "we", "g")
            if seq_pos is not None and self._count <= seq_pos \
                    and at_boundary:
                break
            if n_entries is not None and popped >= n_entries \
                    and at_boundary:
                break
            if seq_pos is None and n_entries is None:
                break
            e = entries.pop()
            popped += 1
            if self._undo_entry(e):
                reverted += 1
            else:
                inexact += 1
            if e[0] == "bc" and seq_pos is not None \
                    and self._count <= seq_pos:
                # a bulk block undoes all-or-nothing (snapshot restore);
                # once it lands the count at/below the floor, every
                # remaining entry belongs to an earlier block - stop
                # even though the next entry isn't a boundary marker
                break
        self._rebuild_recent()
        self._next_lite = False       # rewind - lite flag is stale
        return {"reverted": reverted, "inexact": inexact,
                "stream_len": self._count}

    def _rebuild_recent(self) -> None:
        """Rebuild the in-span recency tail from the ring.  Entries for
        reverted tokens are stale (pos >= count) and a bounded deque may
        have evicted entries the rewind just brought back into span -
        ring bytes before the head are untouched, so this is exact."""
        span = self.config.recency_span
        cap = self.config.ring_capacity
        self._recent = deque(
            ((p, int(self._ring[p % cap]))
             for p in range(max(0, self._count - span), self._count)),
            maxlen=self._recent.maxlen)

    def revert_writes(self, tag: str) -> dict:
        """Undo every ``write_counts`` block journaled under ``tag``
        (distill/manual injections - position-independent, no stream
        rewind needed)."""
        entries = self.journal.entries
        reverted = inexact = 0
        i = len(entries) - 1
        while i >= 0:
            e = entries[i]
            if e[0] == "we" and e[1] == tag:
                # undo back to the matching "ws"
                j = i - 1
                depth = 1
                while j >= 0:
                    op = entries[j][0]
                    if op == "we":
                        depth += 1
                    elif op == "ws":
                        depth -= 1
                        if depth == 0:
                            break
                    else:
                        ok = self._undo_entry(entries[j])
                        reverted += ok
                        inexact += not ok
                    j -= 1
                del entries[j:i + 1]          # drop the whole block
                i = j - 1
            else:
                i -= 1
        return {"reverted": reverted, "inexact": inexact,
                "stream_len": self._count}

    def revert_tag(self, tag: str) -> dict:
        """Un-learn every token journaled under ``tag``, plus any
        ``write_counts`` blocks carrying it.

        Token revert is suffix-based for causal consistency: the stream
        is rewound to just before the earliest journaled token carrying
        ``tag``, which also un-learns later tokens.  Use ``snapshot()``
        first if you need finer-grained rollback.
        """
        rep_w = self.revert_writes(tag)
        # walk token markers backward, tracking a virtual stream length:
        # a "bc" block covers positions blk.count..blk.count+n-1, so the
        # virtual head jumps to blk.count when we pass it
        target = None
        v = self._count
        for e in reversed(self.journal.entries):
            if e[0] == "bc":
                blk = e[1]
                if blk["tag"] == tag:
                    target = blk["count"] if target is None \
                        else min(target, blk["count"])
                v = blk["count"]
            elif e[0] == "t":
                v -= 1
                if e[2] == tag:
                    target = v if target is None else min(target, v)
        if target is None:
            rep_w["found"] = rep_w["reverted"] > 0
            rep_w["tag"] = tag
            return rep_w
        rep = self.revert_since(seq_pos=target)
        rep["reverted"] += rep_w["reverted"]
        rep["inexact"] += rep_w["inexact"]
        rep["found"] = True
        rep["tag"] = tag
        return rep

    def audit(self, n: int = 50, tokenizer=None) -> list[dict]:
        """Decode the last ``n`` learned tokens into readable records:
        what token, under which tag, and the text context tail."""
        out = []
        tail_tokens: deque = deque(maxlen=8)
        for e in self.journal.entries[-(n * 16):]:
            if e[0] == "t":
                rec = {"pos": None, "tok": e[1], "tag": e[2]}
                if tokenizer is not None:
                    rec["token"] = tokenizer.decode([e[1]])
                    rec["context"] = tokenizer.decode(list(tail_tokens))
                tail_tokens.append(e[1])
                out.append(rec)
        return out[-n:]

    def writes_since(self, n_entries: int) -> dict:
        """Summary of the last N journal writes, grouped by op+tag."""
        from collections import Counter
        ops: Counter = Counter()
        tags: Counter = Counter()
        for e in self.journal.entries[-n_entries:]:
            ops[e[0]] += 1
            if e[0] == "t":
                tags[e[2]] += 1
        return {"ops": dict(ops), "tags": dict(tags)}

    # -- persistence -----------------------------------------------------
