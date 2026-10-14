"""Write paths for FluxLM (ingest/teach/fused steps)."""
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

import os

_FLUX_DEBUG_NAN = bool(os.environ.get("FLUX_DEBUG_NAN"))

from .helpers import (_HASH_P, _MASK64, _MIX1, _MIX2,
                      _MIX1U, _MIX2U,
                      _group_add, _group_apply, _group_prefix,
                      _group_sort, _group_totals,
                      _load_teacher_embedding, _mix64, _mix64_np,
                      _mix64_py, _s64, _scatter_sum)
from .config import FluxConfig
from .journal import FluxJournal
from .tables import _GPUCellTables
from .kernels import HAS_TRITON

if HAS_TRITON:
    from .kernels import (_flux_deep_votes_kernel,
                          _flux_logit_tail_kernel)


class _LearnMixin:

    def _step_dev(self, knn_rows: int | None = None,
                  lite: bool = False) -> None:
        """Fused observe+predict for token ``self._bx``: all live
        writes + next-token logits in one device-side pass.  Zero host
        reads - journal metadata goes out through ``self._bout``.

        ``knn_rows`` overrides the scanned-bank bucket for batched
        callers where the host ``_knn_cur`` mirror lags the device
        cursor (the value is a python int baked into the graph slice,
        so it must be projected, not read).  ``lite`` is the adaptive-
        read variant: deep/knn scans skipped (uni fallback rewards +
        logits), so confident steps replay a smaller graph."""
        c = self.config
        dev = self._dev
        gt = self._gt
        O, C = gt.O, len(self.channels)
        V = self.vocab_size
        cap = c.ring_capacity
        pos = self._pos_g                        # x lands at pos
        end = pos - 1                            # context end
        x = self._bx
        jcur = self._jcur_t                      # ring slot scalar

        # -- phase-1 probe at context end (pre-write state): orders +
        # csem + epi all resolved by ONE probe loop -----------------
        keys, live_k = self._keys_dev(end)
        ck_e = self._suffix_dev(c.csem_order, end)
        dk_e = self._suffix_dev(c.deep_order, end)
        pk1 = gt.ctx_hash_all(torch.cat(
            [keys, ck_e.reshape(1), dk_e.reshape(1)]))
        slot1, found1, evict1 = gt.probe_write(pk1)
        slot, found = slot1[:O], found1[:O]
        hit = found & live_k
        rows_t = gt.ttoks[gt._ordx, slot]
        rows_c = gt.tcnt[gt._ordx, slot]
        tots = gt.ttot[gt._ordx, slot]
        cs, cf = slot1[gt.row_csem].reshape(1), \
            found1[gt.row_csem].reshape(1)

        denom = self._tot_g + c.uni_alpha * V
        uni_x = (self._unig.index_select(0, x.reshape(1))
                 .squeeze(0) + c.uni_alpha) / denom

        # -- hedge rewards (pre-write channel probs of x) -----------
        rw = torch.empty(C, dtype=torch.float64, device=dev)
        # sleeping-experts awake mask: True = channel had real evidence
        # (not a unigram fallback).  Built alongside rw; consumed by
        # _hedge_dev's abstention rewrite.
        awake = (torch.ones(C, dtype=torch.bool, device=dev)
                 if c.sleeping_experts else None)
        m_x = (rows_t == x.to(torch.int32))
        cnt_x = (rows_c.double() * m_x).sum(dim=1)
        if c.ppm_escape_c:
            # method-C: escape mass = distinct followers per cell
            a_o = ((((rows_t >= 0) & (rows_t < V)) & hit[:, None])
                   .sum(dim=1).double().clamp(min=1e-9))
        else:
            a_o = float(c.alpha)         # python float - no H2D copy
                                         # inside the captured graph
        p_o = torch.where(hit & (tots > 0),
                          (cnt_x + a_o * uni_x)
                          / (tots.double() + a_o),
                          uni_x.expand(O))
        rw[:O] = torch.where(live_k, p_o, uni_x.expand(O))
        if awake is not None:
            awake[:O] = hit & (tots > 0)
        if lite:
            rw[self._chix["deep"]] = uni_x     # scan skipped -> uni
            if awake is not None:
                awake[self._chix["deep"]] = False
        else:
            floor = torch.clamp(pos - cap, min=0)
            votes = self._deep_votes_dev(
                end, floor, se=slot1[gt.row_epi].reshape(1),
                fe=found1[gt.row_epi].reshape(1), dk=dk_e)
            vt = votes.sum().double()
            vx = votes.index_select(0, x.reshape(1)).squeeze(0).double()
            a_d = (votes.gt(0).sum().double().clamp(min=1e-9)
                   if c.ppm_escape_c
                   else float(c.alpha))   # python float - graph-safe
            rw[self._chix["deep"]] = torch.where(
                vt > 0, (vx + a_d * uni_x)
                / (vt + a_d), uni_x)
            if awake is not None:
                awake[self._chix["deep"]] = vt > 0
        a_x = self._A.index_select(0, x.reshape(1)).squeeze(0)
        cn = self._c.norm()
        cos_t = (a_x @ self._c) \
            / (a_x.norm() * cn + 1e-8)
        if c.calib_rewards:
            # implied channel dist p(v) âˆ p_uni(v)Â·exp(Î²Â·cos_v) - the
            # additive-lift channels get a NORMALIZED reward, not the
            # raw sigmoid (~0.5-for-everything) that lets them hoard
            # weight.  Z = Î£_v p_uniÂ·exp(Î²Â·cos_v) costs one [V] matvec.
            uni_pv = (self._unig + c.uni_alpha) / denom
            chat = self._c / cn.clamp(min=1e-8)
            sims_t = (self._A @ chat) / self._An.clamp(min=1e-8)
            z_t = (uni_pv * torch.exp(c.topic_beta * sims_t)).sum()
            rw[self._chix["topic"]] = torch.where(
                (cn > 1e-8) & (a_x.norm() > 1e-8),
                uni_x * torch.exp(c.topic_beta * cos_t)
                / z_t.clamp(min=1e-12), uni_x)
        else:
            rw[self._chix["topic"]] = torch.sigmoid(
                c.topic_beta * cos_t).double()
        if awake is not None:
            awake[self._chix["topic"]] = (cn > 1e-8) & (a_x.norm() > 1e-8)
        pn = self._proto.norm()
        cos_s = (a_x @ self._proto) \
            / (a_x.norm() * pn + 1e-8)
        if c.calib_rewards:
            phat = self._proto / pn.clamp(min=1e-8)
            sims_s = (self._A @ phat) / self._An.clamp(min=1e-8)
            z_s = (uni_pv * torch.exp(c.sem_beta * sims_s)).sum()
            rw[self._chix["sem"]] = torch.where(
                (pn > 1e-8) & (a_x.norm() > 1e-8),
                uni_x * torch.exp(c.sem_beta * cos_s)
                / z_s.clamp(min=1e-12), uni_x)
        else:
            rw[self._chix["sem"]] = torch.sigmoid(
                c.sem_beta * cos_s).double()
        if awake is not None:
            awake[self._chix["sem"]] = (pn > 1e-8) & (a_x.norm() > 1e-8)
        # csem reward: cosine vs this context's fingerprint row
        cok = (end - c.csem_order + 1 >= self._ep_g) \
            & (end >= c.csem_order - 1) & (end >= 0)
        ax_c = (a_x @ self._Pcsem).double()
        row = self._csem_rows.index_select(0, cs).squeeze(0).double()
        rn = row.norm()
        cos_c = (ax_c @ row) / (ax_c.norm() * rn + 1e-8)
        r_ok = (rn > 1e-8) & torch.isfinite(rn)
        if c.calib_rewards:
            rhat = row / rn.clamp(min=1e-8)
            csims = (self._Ac @ rhat.float()) / self._Acn.clamp(min=1e-8)
            z_c = (uni_pv * torch.exp(c.csem_beta * csims)).sum()
            rw[self._chix["csem"]] = torch.where(
                cf.squeeze() & cok & r_ok,
                (uni_x * torch.exp(c.csem_beta * cos_c)
                 / z_c.clamp(min=1e-12)).double(), uni_x)
        else:
            rw[self._chix["csem"]] = torch.where(
                cf.squeeze() & cok & r_ok,
                torch.sigmoid(c.csem_beta * cos_c).double(), uni_x)
        if awake is not None:
            awake[self._chix["csem"]] = cf.squeeze() & cok & r_ok
        # knn reward: top-k fingerprint neighbours' successor vote on x.
        # fp_store reuses the query fingerprint for the bank write below
        # - compute it even when lite (the write still needs it)
        if c.use_knn:
            fp_store = self._knn_fp_ctx(end).to(self._knn_fp.dtype)
            if lite:
                rw[self._chix["knn"]] = uni_x   # scan skipped -> uni
                if awake is not None:
                    awake[self._chix["knn"]] = False
            else:
                # live ring prefix only - bucketed so the slice stays
                # shape-static inside the captured graph
                kb = self._knn_rows() if knn_rows is None else knn_rows
                kvt, kvx = self._knn_reward_dev(kb, fp_store, x,
                                                end=end)
                # weighted-vote cell: distinct-count has no cheap
                # graph-safe analog (unique breaks capture) - keeps
                # the configured alpha under ppm_escape_c
                rw[self._chix["knn"]] = torch.where(
                    kvt > 0, (kvx + c.alpha * uni_x)
                    / (kvt + c.alpha), uni_x)
                if awake is not None:
                    awake[self._chix["knn"]] = kvt > 0
        else:
            fp_store = torch.zeros(c.csem_dim, device=dev,
                                   dtype=self._knn_fp.dtype)
            rw[self._chix["knn"]] = uni_x
            if awake is not None:
                awake[self._chix["knn"]] = False
        if awake is not None:
            # graph-safe: tensor ops only - a python bool() on a device
            # tensor would force a D2H sync per step and break capture
            if c.use_backoff:
                awake[self._chix["backoff"]] = \
                    (hit & (tots >= c.backoff_min_tot)).any()
            else:
                awake[self._chix["backoff"]] = False
        sv, epi_gate, knn_gate, gap = self._hedge_dev(
            end, x, rw, hit, tots, cnt_x, uni_x, jcur, live_k,
            awake=awake, rows_t=rows_t, rows_c=rows_c)

        # -- writes -------------------------------------------------
        # order rows (branchless; keys written only on new+valid)
        new = (~found) & live_k
        ev = evict1[:O] & live_k               # victim rows get wiped
        ins = new & ~ev                        # empty-slot inserts
        flat_key = gt._ordx * gt.S + slot
        # snapshot victim rows for exact undo (fixed slots in _bev_*)
        EVI, EVF = self._bev_i.shape[1], self._bev_f.shape[1]
        pad_i = EVI - 3 - gt.M
        pad_f = EVF - 1 - gt.M
        zOi = torch.zeros(O, pad_i, dtype=torch.int64, device=dev)
        bev_i_o = torch.cat([
            flat_key[:, None],
            gt.traw.view(-1).index_select(0, flat_key)[:, None],
            torch.zeros(O, 1, dtype=torch.int64, device=dev),
            gt.ttoks.view(-1, gt.M).index_select(
                0, flat_key).long(), zOi], dim=1)
        bev_f_o = torch.cat([
            gt.ttot.view(-1).index_select(0, flat_key)[:, None],
            gt.tcnt.view(-1, gt.M).index_select(0, flat_key),
            torch.zeros(O, pad_f, device=dev)], dim=1)
        self._bev_i[:O] = torch.where(
            ev[:, None].expand(O, EVI), bev_i_o, self._bev_i[:O])
        self._bev_f[:O] = torch.where(
            ev[:, None].expand(O, EVF), bev_f_o, self._bev_f[:O])
        self._w_order_wipe_dev(flat_key, pk1, keys, new, ev, slot)
        # re-gather post-wipe rows for the follower write
        rows_t = gt.ttoks[gt._ordx, slot]
        rows_c = gt.tcnt[gt._ordx, slot]
        m_x = (rows_t == x.to(torch.int32))
        tots = gt.ttot[gt._ordx, slot]
        has = m_x.any(dim=1)
        empty = rows_t == -1
        has_empty = empty.any(dim=1)
        col = torch.where(has, m_x.float().argmax(dim=1),
                          torch.where(has_empty,
                                      empty.float().argmax(dim=1),
                                      rows_c.argmin(dim=1)))
        write_tok = (~has) & live_k                  # empty OR evict
        replace = write_tok & (~has_empty)
        # column-evict victims: capture (col, tok, cnt) pre-write into
        # the bev pad slots, and drop the displaced mass from ttot -
        # otherwise row totals inflate and undo can't balance
        vic_c = rows_c.gather(1, col[:, None]).squeeze(1)
        vic_t = rows_t.gather(1, col[:, None]).squeeze(1)
        if pad_i >= 3 and pad_f >= 1:
            rep3 = replace[:, None].expand(O, 3)
            cv_i = torch.cat([
                flat_key[:, None], col[:, None],
                vic_t[:, None].long()], dim=1)
            self._bev_i[:O, EVI - 3:EVI] = torch.where(
                rep3, cv_i, self._bev_i[:O, EVI - 3:EVI])
            self._bev_f[:O, EVF - 1] = torch.where(
                replace, vic_c, self._bev_f[:O, EVF - 1])
        flat = gt._ordx * (gt.S * gt.M) + slot * gt.M + col
        self._w_order_cells_dev(flat, flat_key, write_tok, replace,
                                live_k, sv, x, vic_c)

        # unigram
        self._unig.index_add_(
            0, x.reshape(1), torch.ones(1, device=dev))
        self._tot_g += 1.0
        self._log_uni_num.index_copy_(
            0, x.reshape(1),
            torch.log(self._unig.index_select(0, x.reshape(1))
                      + c.uni_alpha))

        # push x into ring + prefix hash (needed before epi write)
        h_prev = self._Hg.index_select(
            0, ((pos - 1) % cap).clamp(min=0).reshape(1)).squeeze(0)
        h_new = torch.where(pos > 0, h_prev * 1099511628211 + x + 1,
                            x + 1)
        self._Hg.index_copy_(0, (pos % cap).reshape(1),
                             h_new.reshape(1))
        self._ringg.index_copy_(0, (pos % cap).reshape(1),
                                x.to(torch.int32).reshape(1))

        # -- deferred episodic write: ``x`` is the successor of the
        # context ending at ``end`` - the entry (ctx_end, succ, certs)
        # rides the phase-1 probe slot, so it votes even after the ring
        # wraps.  Predict reads still come from the phase-2 probe.
        slot2, found2, evict2, _live2, cok2, dok2, dk2, pk2 = \
            self._probes_fast(pos)
        dok1 = (end - c.deep_order + 1 >= self._ep_g) \
            & (end >= c.deep_order - 1) & (end >= 0) & epi_gate
        se = slot1[gt.row_epi].reshape(1)
        fe = found1[gt.row_epi].reshape(1)
        ee = (evict1[gt.row_epi] & dok1).reshape(1)
        e_new = (~fe) & dok1
        ER = self._epi_pos.shape[1]
        LC = len(c.cert_orders)
        certs_e = self._w_epi_dev(se, ee, e_new, dok1, pk1, dk_e,
                                  end, x)

        # -- knn bank write: fp of the pre-x context -> successor x --
        # gated writes go to the discard row (index P, never scanned,
        # pos=-1) and do not advance the cursor - no journal, and a
        # revert of the NEXT real write still restores its slot
        ksl, kprev_pos, kprev_succ = self._w_knn_dev(
            knn_gate, jcur, fp_store, x, end)

        # topic write: delta rule - subtract the current readout along
        # the context direction before committing (error-corrected
        # associative memory; repeated contexts stop re-writing)
        self._w_topic_dev(x, jcur, a_x, cn, sv)

        # csem write: EMA this context's row toward follower feature
        if c.use_csem:
            self._w_csem_dev(cs, cf, cok, jcur, pk1, ck_e, ax_c, sv,
                             evict1)

        # recency + context EMA + counters
        self._bout.index_copy_(
            0, self._boff_prevseen,
            self._seen_g.index_select(0, x.reshape(1)))
        self._seen_g.index_copy_(0, x.reshape(1), end.reshape(1))
        self._c.mul_(c.topic_decay).add_(
            self._R.index_select(0, x.reshape(1)).squeeze(0).float()
            * (1.0 - c.topic_decay))
        if c.bigram_sketch:
            # same sketch update as _push, device-side: the pair term
            # is masked to zero at episode heads (pos-1 pre-episode)
            self._c2.mul_(c.topic_decay)
            pv = self._ringg.index_select(
                0, ((pos - 1) % cap).clamp(min=0).reshape(1)).long()
            b2 = (self._R.index_select(0, pv)[:, self._bp1].float()
                  * self._R.index_select(0, x.reshape(1))
                  [:, self._bp2].float()).squeeze(0)
            self._c2.add_(b2 * ((1.0 - c.topic_decay)
                                * (pos - 1 >= self._ep_g).float()))
        self._pos_g += 1

        # -- predict at the new context end (reuses phase-2 probe) --
        # kv/kvt are always tensors - a None-vs-tensor flip is a dynamo
        # specialization boundary, and a recompile during capture kills
        # the graph (cudaErrorStreamCaptureInvalidated)
        kb = self._knn_rows() if knn_rows is None else knn_rows
        if c.use_knn and kb > 0 and not lite:
            kv, kvt = self._knn_votes_dev(kb, end=pos - 1)
        else:
            kv = torch.zeros(V, dtype=torch.float32, device=dev)
            kvt = torch.zeros((), dtype=torch.float32, device=dev)
        self._blogits.copy_(self._logits_fast(
            pos, slot2, found2, _live2, cok2, dok2, dk2,
            kv=kv, kvt=kvt, lite=lite))

        # -- readback bundle (one D2H per token, host journals) -----
        bout = self._bout
        bout[:O] = keys
        # deferred epi insert: host mirrors (key, ctx_end, certs) and
        # `used` bookkeeping come from the PHASE-1 probe
        # NB dk_e is a raw u64 in int64 bit pattern - often negative -
        # so validity must ride a flag bit, not the key's sign
        bout[O] = torch.where(dok1, dk_e, -1)
        bout[O + 1] = torch.where(dok1, se.squeeze(0), -1)
        # bit0 = found, bit1 = evicted, bit2 = dok1 (entry written)
        bout[O + 2] = fe.squeeze(0).long() \
            + ee.squeeze(0).long() * 2 + dok1.long() * 4
        bout[O + 3] = torch.where(cok, cs.squeeze(0), -1)
        bout[O + 4] = cf.squeeze(0).long() \
            + (evict1[gt.row_csem] & cok).long() * 2
        # bit-pack the four [O] flag vectors as vectorized dot products
        # (a per-order `|=` loop costs ~4*O tiny kernels per token)
        bw = self._bitw
        bits = (ins.long() * bw).sum()
        vbits = (live_k.long() * bw).sum()
        ebits = (ev.long() * bw).sum()
        cbits = (replace.long() * bw).sum()
        bout[O + 5] = bits | (vbits << 16) | (ebits << 32) \
            | (cbits << 48)
        bout[O + 6] = gap
        if LC:
            bout[self._boff_cert:self._boff_cert + LC] = certs_e
        bout[self._boff_knn] = torch.where(
            knn_gate, ksl[0],
            torch.full((), -1, dtype=torch.int64, device=dev))
        bout[self._boff_knn + 1] = kprev_pos[0]
        bout[self._boff_knn + 2] = kprev_succ[0]
        bout[self._boff_w:self._boff_w + C] = (
            self._wgR.index_select(0, self._reg_used_g.view(1))
            .squeeze(0) if c.cond_hedge else self._wg
        ).view(torch.int64)
        bout[self._boff_w + C] = sv.double().view(torch.int64)
        bout[self._boff_w + C + 1] = (
            self._hmxRg.index_select(0, self._reg_used_g.view(1))
            .squeeze(0) if c.cond_hedge else self._hmg
        ).view(torch.int64)
        bout[self._boff_regime] = self._reg_used_g
        # predict-time deep key + retrieval-hit flag (utility eviction)
        bout[self._boff_hit] = torch.where(dok2, dk2,
                                           torch.full((), -1,
                                                      dtype=torch.int64,
                                                      device=dev))
        bout[self._boff_hit + 1] = self._dhit_g[0]
        # next-step adaptive-read flag (computed inside _logits_dev
        # from the order cells it already gathered)
        bout[self._boff_next] = self._dconf_g[0]
        # predict-context warm flag for the host sparse-tap lift
        bout[self._boff_thin] = self._dthin_g[0]

    # ---- dynamo-disabled write helpers ------------------------------
    # Inductor lowers index_copy_/index_add_ on a big flat view to a
    # WHOLE-TENSOR read-modify-write (arange + clone + where): a real
    # 8-cell write on the order tables cost ~1ms per fused kernel
    # (measured 27.9us compiled vs ~2us scatter on a 2M int64 tensor).
    # Running the writes eagerly inside disabled helpers keeps them as
    # true scatter kernels while the pointwise producers between them
    # stay inductor-fused.  Order is preserved exactly - each helper is
    # a sequential island.

    @torch._dynamo.disable
    def _w_order_wipe_dev(self, flat_key, pk1, keys, new, ev, slot):
        """Key/raw writes + victim-row wipe on the order tables."""
        dev = self._dev
        gt = self._gt
        O, M = gt.O, gt.M
        gt.tkeys.view(-1).index_copy_(
            0, flat_key,
            torch.where(new, pk1[:O],
                        gt.tkeys.view(-1).index_select(0, flat_key)))
        gt.traw.view(-1).index_copy_(
            0, flat_key,
            torch.where(new, keys,
                        gt.traw.view(-1).index_select(0, flat_key)))
        allm = torch.arange(M, dtype=torch.int64, device=dev)
        flat_row = (gt._ordx[:, None] * (gt.S * M)
                    + slot[:, None] * M + allm[None, :])
        neg1 = torch.full((), -1, dtype=torch.int32, device=dev)
        zO = torch.zeros((), device=dev)
        tt_v = gt.ttoks.view(-1)
        tc_v = gt.tcnt.view(-1)
        fr = flat_row.reshape(-1)
        tt_v.index_copy_(
            0, fr, torch.where(ev[:, None].expand(O, M)
                               .reshape(-1), neg1,
                               tt_v.index_select(0, fr)))
        tc_v.index_copy_(
            0, fr, torch.where(ev[:, None].expand(O, M)
                               .reshape(-1), zO,
                               tc_v.index_select(0, fr)))
        gt.ttot.view(-1).index_copy_(
            0, flat_key, torch.where(
                ev, zO,
                gt.ttot.view(-1).index_select(0, flat_key)))

    @torch._dynamo.disable
    def _w_order_cells_dev(self, flat, flat_key, write_tok, replace,
                           live_k, sv, x, vic_c):
        """Follower writes: cell token/count, row total, occupancy."""
        dev = self._dev
        gt = self._gt
        O = gt.O
        rt_flat = gt.ttoks.view(-1)
        rc_flat = gt.tcnt.view(-1)
        rt_flat.index_copy_(
            0, flat,
            torch.where(write_tok, x.to(torch.int32).expand(O),
                        rt_flat.index_select(0, flat)))
        zeroO = torch.zeros(O, dtype=torch.float32, device=dev)
        sO = sv.float().expand(O)      # surprise-scaled write mass
        rc_flat.index_copy_(
            0, flat,
            torch.where(replace, zeroO,
                        rc_flat.index_select(0, flat)))
        rc_flat.index_add_(
            0, flat, torch.where(live_k, sO, zeroO))
        gt.ttot.view(-1).index_add_(
            0, flat_key, torch.where(replace, -vic_c, zeroO))
        gt.ttot.view(-1).index_add_(
            0, flat_key,
            torch.where(live_k, sO, zeroO))
        gt.tocc.view(-1).index_copy_(
            0, flat_key,
            gt.ttot.view(-1).index_select(0, flat_key))

    @torch._dynamo.disable
    def _w_epi_dev(self, se, ee, e_new, dok1, pk1, dk_e, end, x):
        """Deferred episodic insert: victim wipe + entry write."""
        c = self.config
        dev = self._dev
        gt = self._gt
        ekr = gt.tkeys[gt.row_epi]
        err = gt.traw[gt.row_epi]
        ER = self._epi_pos.shape[1]
        LC = len(c.cert_orders)
        EVI = self._bev_i.shape[1]
        # snapshot victim epi row for undo BEFORE keys overwrite it -
        # positions + successors + certs all restore wholesale
        ep_flat = gt.row_epi * gt.S + se                    # [1]
        pad_e = EVI - 3 - ER * (2 + LC)
        ep_i = torch.cat([
            ep_flat[:, None],
            err.index_select(0, se)[:, None],
            self._epi_cnt.index_select(0, se)[:, None],
            self._epi_pos.index_select(0, se),
            self._epi_succ.index_select(0, se).long(),
            self._epi_cert.index_select(0, se).reshape(1, -1)
            if LC else torch.zeros(1, 0, dtype=torch.int64,
                                   device=dev),
            torch.zeros(1, pad_e, dtype=torch.int64,
                        device=dev)], dim=1)
        self._bev_i.index_copy_(
            0, self._brow_epi,
            torch.where(ee[:, None].expand(1, EVI), ep_i,
                        self._bev_i.index_select(
                            0, self._brow_epi)))
        ekr.index_copy_(
            0, se, torch.where(e_new, pk1[gt.row_epi].reshape(1),
                               ekr.index_select(0, se)))
        err.index_copy_(
            0, se, torch.where(e_new, dk_e.reshape(1),
                               err.index_select(0, se)))
        # evicted epi row: positions/succs/certs wiped, count resets
        epv = self._epi_pos
        prow = epv.index_select(0, se)            # [1, ER]
        epv.index_copy_(
            0, se, torch.where(ee[:, None].expand(1, ER),
                               torch.full((1, ER), -1,
                                          dtype=torch.int64,
                                          device=dev), prow))
        svv = self._epi_succ
        svv.index_copy_(
            0, se, torch.where(ee[:, None].expand(1, ER),
                               torch.full((1, ER), -1,
                                          dtype=torch.int32,
                                          device=dev),
                               svv.index_select(0, se)))
        if LC:
            cvv = self._epi_cert
            cvv.index_copy_(
                0, se, torch.where(
                    ee[:, None, None].expand(1, ER, LC),
                    torch.zeros((1, ER, LC), dtype=torch.int64,
                                device=dev),
                    cvv.index_select(0, se)))
        self._epi_cnt.index_copy_(
            0, se, torch.where(ee, torch.zeros(1, dtype=torch.int64,
                                               device=dev),
                               self._epi_cnt.index_select(0, se)))
        col_e = self._epi_cnt.index_select(0, se) % ER
        eflat = se * ER + col_e
        # certificate suffix hashes at `end` for the stored entry -
        # episode-gated: a cert that crosses ep_start stores 0
        certs_e = None
        if LC:
            certs_e = torch.stack(
                [self._suffix_dev(L, end).squeeze(0)
                 for L in c.cert_orders])
            cok_l = torch.stack(
                [(end - L + 1 >= self._ep_g)
                 & (end >= L - 1) & (end >= 0)
                 for L in c.cert_orders])
            certs_e = torch.where(
                cok_l, certs_e,
                torch.zeros((), dtype=torch.int64, device=dev))
        dok1r = dok1.reshape(1)
        self._epi_pos.view(-1).index_copy_(
            0, eflat,
            torch.where(dok1r, end.reshape(1),
                        self._epi_pos.view(-1).index_select(0, eflat)))
        self._epi_succ.view(-1).index_copy_(
            0, eflat,
            torch.where(dok1r, x.to(torch.int32).reshape(1),
                        self._epi_succ.view(-1).index_select(0, eflat)))
        if LC:
            self._epi_cert.view(-1, LC).index_copy_(
                0, eflat,
                torch.where(dok1r[:, None].expand(1, LC),
                            certs_e.reshape(1, LC),
                            self._epi_cert.view(-1, LC)
                            .index_select(0, eflat)))
        self._epi_cnt.index_add_(
            0, se,
            torch.where(dok1r, torch.ones(1, dtype=torch.int64,
                                          device=dev),
                        torch.zeros(1, dtype=torch.int64,
                                    device=dev)))
        gt.tocc[gt.row_epi].index_copy_(
            0, se, self._epi_cnt.index_select(0, se).float())
        return certs_e

    @torch._dynamo.disable
    def _w_knn_dev(self, knn_gate, jcur, fp_store, x, end):
        """KNN bank write (gated -> discard row) + cursor."""
        c = self.config
        dev = self._dev
        if c.use_knn:
            if c.knn_utility_evict:
                # bank-full -> utility eviction: coldest row =
                # hits*hit_w + pos/total (same score as _knn_write);
                # argmin on [P] is one graph-safe reduction
                P = c.knn_capacity
                sc = (self._knn_hits[:P].double() * c.knn_evict_hit_w
                      + self._knn_pos[:P].double()
                      / self._tot_g.clamp(min=1).double())
                ksl = torch.where(
                    knn_gate.reshape(1),
                    torch.where(
                        (self._knncur_g >= P).reshape(1),
                        sc.argmin().reshape(1),
                        (self._knncur_g % P).reshape(1)),
                    self._knn_disc)
            else:
                ksl = torch.where(
                    knn_gate.reshape(1),
                    (self._knncur_g % c.knn_capacity).reshape(1),
                    self._knn_disc)
            self._knn_hits.index_fill_(0, ksl, 0)
            kprev_pos = self._knn_pos.index_select(0, ksl)
            kprev_succ = self._knn_succ.index_select(0, ksl)
            self._kprev_g.index_copy_(
                0, jcur.reshape(1), self._knn_fp.index_select(0, ksl))
            self._knn_fp.index_copy_(
                0, ksl, fp_store.reshape(1, c.csem_dim))
            self._knn_succ.index_copy_(
                0, ksl, x.to(torch.int32).reshape(1))
            self._knn_pos.index_copy_(
                0, ksl, torch.where(
                    knn_gate.reshape(1), end.reshape(1),
                    torch.full((1,), -1, dtype=torch.int64,
                               device=dev)))
            self._knncur_g += knn_gate.long()
        else:
            ksl = torch.full((1,), -1, dtype=torch.int64, device=dev)
            kprev_pos = torch.full((1,), -1, dtype=torch.int64,
                                   device=dev)
            kprev_succ = torch.full((1,), -1, dtype=torch.int64,
                                    device=dev)
        return ksl, kprev_pos, kprev_succ

    @torch._dynamo.disable
    def _w_topic_dev(self, x, jcur, a_x, cn, sv):
        """Hebbian/delta-rule topic write + incremental projections."""
        c = self.config
        if c.delta_rule:
            chat = self._c / (cn + 1e-8)
            pred = (a_x * chat).sum()
            delta = chat * ((self._const_1.float() - pred)
                            * c.hebb_lr * sv)
        else:
            delta = self._c * (c.hebb_lr * sv)
        self._avec_g.index_copy_(0, jcur.reshape(1),
                                 delta.unsqueeze(0))
        self._A.index_add_(0, x.reshape(1), delta.unsqueeze(0))
        # incremental Ac/An/Acn so predict never rescans A
        ax_new = self._A.index_select(0, x.reshape(1)).squeeze(0)
        self._An.index_copy_(0, x.reshape(1),
                             ax_new.norm().reshape(1))
        self._Ac.index_add_(
            0, x.reshape(1), (delta @ self._Pcsem).unsqueeze(0))
        self._Acn.index_copy_(
            0, x.reshape(1),
            self._Ac.index_select(0, x.reshape(1)).norm(dim=1))
        self._proto.mul_(c.proto_decay).add_(
            ax_new * (1.0 - c.proto_decay))

    @torch._dynamo.disable
    def _w_csem_dev(self, cs, cf, cok, jcur, pk1, ck_e, ax_c, sv,
                    evict1):
        """CSEM row insert + EMA update + victim snapshot."""
        c = self.config
        dev = self._dev
        gt = self._gt
        EVI, EVF = self._bev_i.shape[1], self._bev_f.shape[1]
        cur_row = self._csem_rows.index_select(0, cs)
        prev_row = torch.where(
            cf[0], cur_row.squeeze(0),
            torch.zeros(c.csem_dim, device=dev))
        self._csem_prev.index_copy_(0, jcur.reshape(1),
                                    prev_row.half().unsqueeze(0))
        new_key = (~cf[0]) & cok
        ckr = gt.tkeys[gt.row_csem]
        crr = gt.traw[gt.row_csem]
        # snapshot victim csem row for undo BEFORE keys overwrite
        cev = (evict1[gt.row_csem] & cok).reshape(1)
        cs_flat = gt.row_csem * gt.S + cs               # [1]
        pad_ci = EVI - 2
        pad_cf = EVF - 1 - c.csem_dim
        cs_i = torch.cat([
            cs_flat[:, None],
            crr.index_select(0, cs)[:, None],
            torch.zeros(1, pad_ci, dtype=torch.int64,
                        device=dev)], dim=1)
        cs_f = torch.cat([
            gt.tocc[gt.row_csem].index_select(
                0, cs)[:, None],
            cur_row.float(),
            torch.zeros(1, pad_cf, device=dev)], dim=1)
        self._bev_i.index_copy_(
            0, self._brow_csem,
            torch.where(cev[:, None].expand(1, EVI), cs_i,
                        self._bev_i.index_select(
                            0, self._brow_csem)))
        self._bev_f.index_copy_(
            0, self._brow_csem,
            torch.where(cev[:, None].expand(1, EVF), cs_f,
                        self._bev_f.index_select(
                            0, self._brow_csem)))
        ckr.index_copy_(
            0, cs, torch.where(
                new_key, pk1[gt.row_csem].reshape(1),
                ckr.index_select(0, cs)))
        crr.index_copy_(
            0, cs, torch.where(
                new_key, ck_e.reshape(1),
                crr.index_select(0, cs)))
        upd = prev_row * (1.0 - c.csem_lr * sv) \
            + ax_c.float() * (c.csem_lr * sv)
        # fp16 range guard: _A magnitudes can exceed 65504; rescale
        # direction-preservingly so .half() cannot produce inf rows
        upd = torch.nan_to_num(upd)
        upd = upd * (6e4 / upd.abs().max().clamp(min=6e4))
        self._csem_rows.index_copy_(
            0, cs,
            torch.where(cok, upd.half().unsqueeze(0),
                        cur_row))
        gt.tocc[gt.row_csem].index_add_(
            0, cs, torch.where(
                cok, torch.ones(1, device=dev),
                torch.zeros(1, device=dev)))

    def _step_fast(self, knn_rows: int, lite: bool) -> None:
        """``_step_dev`` via torch.compile - inductor fuses the write
        section's ~600 tiny kernels into a handful.  Tie-order hazards
        (topk in the knn reward, custom triton kernels) are already
        dynamo-disabled inside; any compile failure falls back to the
        eager step permanently.  ``knn_rows``/``lite`` are python
        specializations - bounded by the (lite, bucket) graph key and
        warmed per-key before capture."""
        sc = getattr(self, "_step_c", None)
        if sc is None:
            sc = False
            if self._gpu and self.config.compile_dev:
                try:
                    import torch._inductor.config as _icfg
                    _icfg.assert_indirect_indexing = False
                    sc = torch.compile(self._step_dev, dynamic=False)
                except Exception:
                    sc = False
            self._step_c = sc
        if sc is not False:
            try:
                sc(knn_rows, lite=lite)
                return
            except Exception:
                self._step_c = False
        self._step_dev(knn_rows, lite=lite)

    @torch._dynamo.disable
    def _hedge_dev(self, end, x, rw, hit, tots, cnt_x, uni_x, jcur,
                   live_k, awake=None, rows_t=None, rows_c=None):
        """Channel-tail rewards + AdaHedge + surprise/gate flags.

        dynamo-disabled: this block holds every scalar reduction in the
        step (kstar argmax, pmix/rw sums) fused with fp64 pow/exp - the
        inductor-generated persistent-reduction kernel crashes triton's
        TritonGPURemoveLayoutConversions pass on sm_120.  Running it
        natively keeps the ~300-op write section fusable."""
        c = self.config
        dev = self._dev
        gt = self._gt
        age = (end - self._seen_g.index_select(
            0, x.reshape(1)).squeeze(0)).double()
        if c.calib_rewards:
            # implied channel dist p(v) âˆ p_uni(v)Â·exp(Â±lift(v)):
            # reward = p_uni(x)Â·exp(f_x)/Z, Z = 1 + Î£_v p_uniÂ·(e^fâˆ’1)
            # over the live span window (unseen tokens contribute 0).
            denom_v = (self._tot_g + c.uni_alpha * self.vocab_size)
            uni_pv = (self._unig + c.uni_alpha) / denom_v
            ages_v = (end - self._seen_g)
            rec_v = torch.where(
                (ages_v >= 0) & (ages_v < c.recency_span)
                & (self._seen_g >= 0),
                c.recency_decay ** ages_v.float().clamp(min=0),
                torch.zeros((), device=dev)).double()
            z_r = 1.0 + (uni_pv * (torch.exp(rec_v) - 1.0)).sum()
            rec_x = torch.where(inrec := (age >= 0) & (age < c.recency_span),
                                c.recency_decay ** age.clamp(min=0),
                                torch.zeros((), dtype=torch.float64,
                                            device=dev))
            rw[self._chix["recency"]] = uni_x * torch.exp(rec_x) \
                / z_r.clamp(min=1e-12)
            fat_v = torch.where(
                (ages_v >= 0) & (ages_v < c.fatigue_span)
                & (self._seen_g >= 0),
                c.fatigue_decay ** ages_v.float().clamp(min=0),
                torch.zeros((), device=dev)).double()
            z_f = 1.0 + (uni_pv
                         * (torch.exp(-c.fatigue_strength * fat_v)
                            - 1.0)).sum()
            fat_x = torch.where(infat := (age >= 0) & (age < c.fatigue_span),
                                c.fatigue_decay ** age.clamp(min=0),
                                torch.zeros((), dtype=torch.float64,
                                            device=dev))
            rw[self._chix["fatigue"]] = uni_x \
                * torch.exp(-c.fatigue_strength * fat_x) \
                / z_f.clamp(min=1e-12)
        else:
            inrec = (age >= 0) & (age < c.recency_span)
            rw[self._chix["recency"]] = torch.where(
                inrec, c.recency_decay ** age.clamp(min=0),
                self._const_1e3)
            infat = (age >= 0) & (age < c.fatigue_span)
            rw[self._chix["fatigue"]] = torch.where(
                infat, 1.0 - c.fatigue_decay ** age.clamp(min=0),
                self._const_1)
        rw[self._chix["uni"]] = uni_x
        # host-resident channels (sparse-tap p<j>, class-order g<k>) -
        # the GPU step can't probe them, so they take the unigram
        # fallback / abstain until a device port lands
        for i in self._pidx:
            rw[i] = uni_x
            if awake is not None:
                awake[i] = False
        # fuzzy lift channels: real probs are computed in the eager and
        # bulk paths; the fused per-token step abstains (neutral uni
        # fallback preserves weight - their arbitration lives where
        # bulk training happens)
        for i in self._fidx:
            rw[i] = uni_x
            if awake is not None:
                awake[i] = False
        # backoff reward: prob of x under the longest evidenced order
        if c.use_backoff:
            ok_bo = hit & (tots >= c.backoff_min_tot)
            kstar = torch.where(
                ok_bo, self._orders_g.double(),
                torch.full((), -1.0, dtype=torch.float64,
                           device=dev)).argmax()
            bo_hit = ok_bo.index_select(
                0, kstar.reshape(1)).squeeze(0)
            tot_k = tots.index_select(
                0, kstar.reshape(1)).squeeze(0).double()
            cnt_k = cnt_x.index_select(
                0, kstar.reshape(1)).squeeze(0)
            conf = tot_k / (tot_k + c.backoff_kappa)
            if c.ppm_escape_c:
                rk = rows_t.index_select(0, kstar.reshape(1)).squeeze(0)
                a_bk = ((rk >= 0) & (rk < self.vocab_size)).sum() \
                    .double().clamp(min=1e-9)
            else:
                a_bk = float(c.alpha)    # python float - graph-safe
            rw[self._chix["backoff"]] = torch.where(
                bo_hit,
                conf * (cnt_k + a_bk * uni_x)
                / (tot_k + a_bk), uni_x)
        else:
            rw[self._chix["backoff"]] = uni_x
        # hedge update on device weights (AdaHedge: lr adapts to the
        # accumulated mixability gap - constant regret on easy streams)
        if c.cond_hedge:
            # regime of the context this token was predicted under:
            # warm = some order cell held evidence.  Same probe bits
            # the order channels already computed; stashed on
            # _reg_used_g because _logits_dev overwrites _regime_g.
            # evidence_share>0: warmth requires concentrated mass -
            # must match the predict-side gate or rewards route to
            # the wrong regime row.
            if rows_c is not None and (c.thin_share > 0
                                       or c.evidence_share > 0):
                sh_l = rows_c.max(dim=1).values \
                    / tots.clamp(min=1e-9)
                if c.thin_share > 0:
                    r_now = (hit & (tots > c.thin_tot)
                             & (sh_l >= c.thin_share)).any().long()
                else:
                    r_now = (hit & (tots * sh_l.clamp(min=0.0)
                                   ** c.evidence_share
                                   > c.thin_tot)).any().long()
            else:
                r_now = (hit & (tots > c.thin_tot)).any().long()
            self._reg_used_g.copy_(r_now)
            r1 = r_now.view(1)
            wrow = self._wgR.index_select(0, r1).squeeze(0)
            hmx = self._hmxRg.index_select(0, r1).squeeze(0)
        else:
            wrow, hmx = self._wg, self._hmg
        self._w_ring_g.index_copy_(
            0, jcur.reshape(1),
            torch.cat([wrow, hmx.reshape(1)]).unsqueeze(0))
        if self._pidx and not c.sleeping_experts:
            # host-resident channels get no device reward - permanent
            # abstention (neutral pmix), same closed form as sleeping-
            # experts mode 2 restricted to the p<j>/g<k> arms
            aw = (~self._pmask_g).double()
            wsum = (wrow * aw).sum().clamp(min=1e-12)
            pmix = (wrow * rw * aw).sum() / wsum
            rw = torch.where(self._pmask_g, pmix.expand_as(rw), rw)
        elif awake is not None and c.sleeping_experts == 2:
            # p_mix over awake channels only - abstentions take the
            # mixture's own prob as a neutral, weight-preserving reward
            aw = awake.double()
            wsum = (wrow * aw).sum().clamp(min=1e-12)
            pmix = (wrow * rw * aw).sum() / wsum
            rw = torch.where(awake, rw, pmix.expand_as(rw))
        elif awake is not None and c.sleeping_experts == 1:
            pmix = (wrow * rw).sum()
            rw = torch.where(awake, rw, pmix.expand_as(rw))
        else:
            pmix = (wrow * rw).sum()
        if c.adahedge:
            if c.hedge_discount < 1.0:
                hmx.mul_(c.hedge_discount)
            hmx.add_(rw.max() - pmix)
            eta = torch.minimum(
                self._const_eta_max,
                self._const_lnC / hmx.clamp(min=1e-9))
        else:
            eta = self._const_hedge_lr
        wrow.mul_(torch.exp(eta * (rw - pmix)))
        wrow.clamp_(c.hedge_min, c.hedge_max)
        wrow.div_(wrow.sum())
        if c.hedge_share > 0:
            # fixed-share diffusion: uniform-share floor so a collapsed
            # channel can re-track after a regime switch
            wrow.mul_(1.0 - c.hedge_share).add_(
                c.hedge_share / len(self.channels))
        if c.cond_hedge:
            self._wgR.index_copy_(0, r1, wrow.unsqueeze(0))
            self._hmxRg.index_copy_(0, r1, hmx.reshape(1))

        # surprise: magnitude of this token's writes (1 - p_mix)
        # nan_to_num: a NaN pmix (poisoned channel) -> maximal, never
        # a NaN write strength
        sv = torch.nan_to_num(
            ((torch.clamp(self._const_1 - pmix,
                          min=c.surprise_floor)
              if c.surprise_gate else self._const_1)).float(),
            nan=1.0)

        # write gate: epi entries only for thin contexts or surprises,
        # knn rows only for surprises - branchless folds (dok1 mask +
        # discard-slot write) so the captured graph stays static
        if c.write_gate:
            pki = gt.kidx.get(c.write_gate_probe_order, -1)
            thin_g = (~(hit[pki]
                        & (tots[pki] >= c.write_gate_thin_tot))
                      if pki >= 0 else self._const_true)
            epi_gate = thin_g | (sv.double() >= c.write_gate_epi_s)
            knn_gate = sv.double() >= c.write_gate_knn_s
        else:
            epi_gate = knn_gate = self._const_true

        # -- gap flag: unseen context OR high surprise --------------
        gk_i = gt.kidx.get(c.gap_probe_order, -1)
        gap = torch.zeros((), dtype=torch.int64, device=dev)
        if gk_i >= 0:
            gap = (live_k[gk_i]
                   & (~hit[gk_i] | (sv.double() > c.gap_surprise))
                   ).long()
        return sv, epi_gate, knn_gate, gap

    def _run_dev_step(self, knn_rows: int | None = None) -> None:
        """Enqueue one ``_step_dev`` - eager warmup, graph capture, or
        replay.  ``knn_rows`` projects the bank bucket for batched
        callers; the same projected value is part of the graph key so
        bucket-crossing checks stay consistent.  The lite variant
        (adaptive read: deep/knn scans skipped when order cells were
        confident last step) is chosen from ``self._next_lite`` - the
        flag is device-computed and rides bout, so graph selection
        costs zero extra syncs."""
        c = self.config
        rows = self._knn_rows() if knn_rows is None else knn_rows
        lite = bool(c.adaptive_read and self._next_lite)
        key = (lite, rows)
        for k in [k for k in self._graphs if k[1] != rows]:
            del self._graphs[k]        # stale-bucket pools freed
        g = self._graphs.get(key)
        if g is not None:
            g.replay()
            return
        # warmup is per-graph-key: a global counter let an unwarmed
        # (lite, bucket) variant reach capture, and torch.compile'd
        # sub-calls compiling inside capture invalidate the graph
        wk = self._warm.get(key, 0)
        if wk < 3 or not getattr(self, "_use_cuda_graph", True):
            self._warm[key] = wk + 1
            self._step_fast(rows, lite)          # warmup steps
            return
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            self._step_fast(rows, lite)          # capture (no exec)
        self._graphs[key] = g
        g.replay()                               # real work for x

    def _bev_needed(self, bi) -> bool:
        """Does this step's ``_bout`` row flag any eviction snapshots?"""
        O = self._gt.O
        return bool((bi[O + 5] >> 32) or (bi[O + 2] & 2)
                    or (bi[O + 4] & 2))

    def _fused_step(self, x: int) -> torch.Tensor:
        """Host wrapper for ``_step_dev``: one replay + one D2H read,
        then host-side journal/mirror bookkeeping."""
        self._bx.fill_(x)
        self._jcur_t.fill_(self._jcur)
        gc_pre = None
        gq_k = None
        if (self.config.gcell or self.config.gsyn) \
                and not self._predict_only:
            # pre-x context: device _c/_c2 are pre-update until
            # _run_dev_step runs - the write keys the context, not x
            gxf = self._gcell_feats(self._c, self._c2,
                                    self._count - 1)
            if self.config.gcell:
                gc_pre = self._gcell_codes_from_feats(
                    gxf[None])[0]
            if (self.config.gsyn or self.config.gsyn_deep
                    or self.config.dnet):
                gq_k = self._gsyn_keys(gxf)
        self._run_dev_step()
        bi = self._bout.tolist()                     # the ONE sync
        bev_i = bev_f = None
        if self._bev_needed(bi):
            bev_i = self._bev_i.cpu().numpy()
            bev_f = self._bev_f.cpu().numpy()
        self._bookkeep_fused(x, bi, self._jcur, bev_i, bev_f,
                             gc_code=gc_pre, gsyn_k=gq_k)
        self._apply_sparse_lift_dev(
            self._blogits, int(bi[self._boff_thin]))
        c = self.config
        if c.auto_episode and x == c.eos_token_id:
            self.soft_reset()
        return self._blogits

    def _fused_seq(self, row: list[int]) -> torch.Tensor:
        """Learn a token row on GPU with a single D2H readback.

        ``_step_dev`` is pure device-side (zero host reads), so T steps
        enqueue back-to-back while ``_bout``/``_blogits``/``_bev_*``
        snapshots accumulate in device rows; the host journal/mirror
        bookkeeping then replays from ONE transfer.  Returns [T,V]
        next-position logits.

        Semantic note: host-side epi eviction takes effect at row
        granularity - a cell that hits the cap mid-row keeps voting
        until the row's device phase completes (<=T steps of stale
        votes, bounded by the batch length).  Same granularity for
        adaptive_read: ``_next_lite`` lives in ``_bout`` which isn't
        read back until the row ends, so mid-row lite/full graph
        flips that _fused_step would see stay at the pre-row value
        here - per-position surprise (and every delta scaled by it,
        gcell included) can differ from the per-token path on streams
        where the flag would toggle.  Codes/keys are unaffected
        (SimHash is computed host-side from closed-form context
        states, bit-stable).
        """
        c = self.config
        dev = self._dev
        if c.auto_episode and 0 <= c.eos_token_id \
                and c.eos_token_id in row:
            # episode boundary mid-row: one fused pass per segment,
            # host-side soft_reset between (outside the device pipeline)
            eos = c.eos_token_id
            outs, last = [], 0
            for i, x in enumerate(row):
                if x == eos:
                    outs.append(self._fused_seq(row[last:i + 1]))
                    self.soft_reset()
                    last = i + 1
            if last < len(row):
                outs.append(self._fused_seq(row[last:]))
            return torch.cat(outs)
        T = len(row)
        bout_seq = torch.empty((T, self._bout.numel()),
                               dtype=torch.int64, device=dev)
        blog_seq = torch.empty((T, self.vocab_size),
                               dtype=torch.float32, device=dev)
        bev_i_seq = torch.empty((T,) + tuple(self._bev_i.shape),
                                dtype=self._bev_i.dtype, device=dev)
        bev_f_seq = torch.empty((T,) + tuple(self._bev_f.shape),
                                dtype=self._bev_f.dtype, device=dev)
        jcur0 = self._jcur
        cur0 = self._knn_cur
        # gen-channel context features per position, computed from the
        # pre-row _c/_c2 (device state is post-row once steps run);
        # host tables/keys mean one pass serves the whole row
        row_t = None
        gcw = gcr = gkw = gkr = None
        gcpre = None
        if (c.gcell or c.gsyn or c.gsyn_deep or c.dnet
                or c.r_live):
            row_t = torch.as_tensor(row, dtype=torch.int64,
                                    device=dev)
            gcpre = self._cprev_seq(row_t)          # [n+1, d]
            gc2pre = (self._c2prev_seq(row_t)
                      if c.bigram_sketch else None)
        if c.gcell or c.gsyn or c.gsyn_deep or c.dnet:
            # write feats: pre-token context (rows 0..n-1);
            # read feats: post-token context (rows 1..n)
            f_w = self._gcell_feats_bulk(
                gcpre[:-1], None if gc2pre is None else gc2pre[:-1],
                row_t, 0, self._count, row_t)
            f_r = self._gcell_feats_bulk(
                gcpre[1:], None if gc2pre is None else gc2pre[1:],
                row_t, 0, self._count, row_t, end_off=1)
            if c.gcell:
                gcw = self._gcell_codes_from_feats(f_w)
                gcr = self._gcell_codes_from_feats(f_r)
            if c.gsyn or c.gsyn_deep or c.dnet:
                gkw = self._gsyn_keys(f_w)
                gkr = self._gsyn_keys(f_r)
        for i, x in enumerate(row):
            self._bx.fill_(int(x))
            self._jcur_t.fill_((jcur0 + i) % c.vec_delta_cap)
            self._run_dev_step(self._knn_rows_at(cur0 + i))
            bout_seq[i].copy_(self._bout)
            blog_seq[i].copy_(self._blogits)
            bev_i_seq[i].copy_(self._bev_i)
            bev_f_seq[i].copy_(self._bev_f)
        # -- the ONE sync for the whole row -------------------------
        bis = bout_seq.tolist()
        if any(self._bev_needed(bi) for bi in bis):
            bev_is = bev_i_seq.cpu().numpy()
            bev_fs = bev_f_seq.cpu().numpy()
        else:
            bev_is = bev_fs = None
        # live generalizer: one batched probe over every position's
        # order keys (post-row cells - the transferred distribution
        # includes the row's own writes, the consolidate_gen-class
        # approximation) - folded into the row's single readback
        gl_rt = gl_rc = gl_tot = gl_found = None
        if c.gen_live > 0 and not self._predict_only:
            O = self._gt.O
            kk = torch.tensor(
                [[bi[oi] for oi in range(O)] for bi in bis],
                dtype=torch.int64, device=dev)
            pk = _mix64(
                kk * _MIX1 + self._gt._salt[:O][None, :])
            rows = self._gt._ordx[None, :].expand(T, O).reshape(-1)
            slot_g, found_g = self._gt.probe_fixed(
                pk.reshape(-1), rows=rows)
            gl_rt = self._gt.ttoks[rows, slot_g].cpu().numpy()
            gl_rc = self._gt.tcnt[rows, slot_g].cpu().numpy()
            gl_tot = self._gt.ttot[rows, slot_g].cpu().numpy()
            gl_found = found_g.cpu().numpy()
        for i, x in enumerate(row):
            glc = (None if gl_rt is None else (
                gl_rt[i * self._gt.O:(i + 1) * self._gt.O],
                gl_rc[i * self._gt.O:(i + 1) * self._gt.O],
                gl_tot[i * self._gt.O:(i + 1) * self._gt.O],
                gl_found[i * self._gt.O:(i + 1) * self._gt.O]))
            self._bookkeep_fused(
                int(x), bis[i], (jcur0 + i) % c.vec_delta_cap,
                None if bev_is is None else bev_is[i],
                None if bev_fs is None else bev_fs[i],
                gc_code=None if gcw is None else int(gcw[i]),
                gsyn_k=None if gkw is None else gkw[i],
                rl_ctx=None if gcpre is None else gcpre[i],
                gl_cells=glc)
            # sparse lift inside the replay loop so step i sees only
            # writes up to i inclusive - CPU per-token timing parity
            self._apply_sparse_lift_dev(
                blog_seq[i], int(bis[i][self._boff_thin]),
                gc_code=None if gcr is None else int(gcr[i]),
                gsyn_q=None if gkr is None else gkr[i])
        return blog_seq

    def _bookkeep_fused(self, x: int, bi, jcur: int,
                        bev_i=None, bev_f=None,
                        gc_code: int | None = None,
                        gsyn_k: torch.Tensor | None = None,
                        rl_ctx: torch.Tensor | None = None,
                        gl_cells: tuple | None = None) -> None:
        """Host mirrors + journal for one completed ``_step_dev``.

        ``bi`` is the step's ``_bout`` row (host list); ``bev_i``/
        ``bev_f`` the eviction snapshot arrays when ``_bev_needed`` -
        per-step rows in the batched path, whole tensors otherwise.
        ``jcur`` is the journal-ring cursor for this step (the batched
        caller projects it; single-step passes ``self._jcur``)."""
        c = self.config
        O, C = self._gt.O, len(self.channels)
        pos = self._count
        end = pos - 1
        keys = bi[:O]
        newmask = bi[O + 5] & 0xFFFF
        valmask = (bi[O + 5] >> 16) & 0xFFFF
        bw = self._boff_w
        s_h = float(np.asarray(bi[bw + C],
                              dtype=np.int64).view(np.float64))
        w_row = np.asarray(
            [int(v) for v in bi[bw:bw + C]],
            dtype=np.int64).view(np.float64).copy()
        hmx_v = float(np.asarray(
            bi[bw + C + 1], dtype=np.int64).view(np.float64))
        if c.cond_hedge:
            r_used = int(bi[self._boff_regime])
            self._wR[r_used] = w_row
            self._hmxR[r_used] = hmx_v
            self._regime = r_used
        self._hedge_mix = hmx_v
        # adaptive-read flag for the next step's graph pick (device-
        # computed from this step's phase-2 order probe)
        self._next_lite = bool(bi[self._boff_next])

        # host mirrors (no GPU involvement)
        self._w = w_row
        self._uni[x] += 1.0
        self._total += 1
        prev_seen = bi[O + 7]
        prev_j = int(prev_seen) if prev_seen >= 0 else -1
        self._last_seen[x] = end
        self._recent.append((end, x))
        cap = c.ring_capacity
        slot = pos % cap
        self._ring[slot] = x
        prev_h = int(self._H[(pos - 1) % cap]) if pos > 0 else 0
        self._H[slot] = (prev_h * 1099511628211 + x + 1) % (1 << 64)
        # deferred epi host entry: (ctx_end, succ, certs) - validity is
        # the dok1 flag bit (raw u64 keys can be negative as int64)
        certs = tuple(bi[self._boff_cert + j] & 0xFFFFFFFFFFFFFFFF
                      for j in range(len(c.cert_orders)))
        epi_ok = bool(bi[O + 2] & 4)
        if epi_ok:
            ekey = bi[O] & 0xFFFFFFFFFFFFFFFF
            dq = self._epi.get(ekey)
            if dq is None:
                dq = self._epi[ekey] = deque(maxlen=c.epi_max_per_key)
            elif len(dq) == c.epi_max_per_key:
                # deque append will drop the oldest - journal it first
                # so the write is exactly revertible (same guard as
                # the CPU _push path)
                old = dq[0]
                self.journal.append(
                    ("eE", ekey, old[0], old[1], old[2]))
                self._epi_n -= 1
            dq.append((end, x, certs))
            self._epi_fifo.append((ekey, end))
            self._epi_n += 1
            self._epi_evict()
        if c.use_knn and bi[self._boff_knn] >= 0:
            self._knn_cur += 1          # host mirror of _knncur_g -
                                        # gated writes don't advance it
        # deep-channel retrieval hit - utility signal for eviction
        if bi[self._boff_hit] >= 0 and bi[self._boff_hit + 1]:
            k_ = bi[self._boff_hit] & 0xFFFFFFFFFFFFFFFF
            self._epi_hits[k_] = self._epi_hits.get(k_, 0) + 1
        if bi[O + 6] and c.track_gaps:
            self._gaps.append(pos)
        for i in range(O):
            if (newmask >> i) & 1:
                self._gt.used[i] += 1
        if bi[O + 3] >= 0 and bi[O + 4] == 0:
            self._gt.used[self._gt.row_csem] += 1
        if epi_ok and (bi[O + 2] & 3) == 0:
            self._gt.used[self._gt.row_epi] += 1

        # journal (same op order as the CPU path).  Eviction entries
        # are appended BEFORE the writes that displaced them so LIFO
        # undo restores victims after subtracting the displacers.
        evmask = (bi[O + 5] >> 32) & 0xFFFF
        cemask = (bi[O + 5] >> 48) & 0xFFFF
        epi_ev = bi[O + 2] & 2
        cs_ev = bi[O + 4] & 2
        if (evmask or epi_ev or cs_ev or cemask) and bev_i is None:
            # single-step caller didn't pre-fetch - fall back to a live
            # D2H (batched callers always pass per-step rows)
            bev_i = self._bev_i.cpu().numpy()
            bev_f = self._bev_f.cpu().numpy()
        self.journal.append(
            ("h2", self._regime, jcur) if c.cond_hedge else ("h", jcur))
        self.journal.append(("u", x))
        if evmask:
            M = self._gt.M
            for i in range(O):
                if (evmask >> i) & 1:
                    r_ = bev_i[i]
                    self.journal.append(
                        ("ev", int(r_[0]),
                         int(r_[1]) & 0xFFFFFFFFFFFFFFFF,
                         float(bev_f[i][0]),
                         r_[3:3 + M].astype(np.int32).copy(),
                         bev_f[i][1:1 + M].copy()))
        if cemask:
            EVI = self._bev_i.shape[1]
            EVF = self._bev_f.shape[1]
            for i in range(O):
                if (cemask >> i) & 1:
                    r_ = bev_i[i]
                    self.journal.append(
                        ("cv", int(r_[EVI - 3]), int(r_[EVI - 2]),
                         int(r_[EVI - 1]), float(bev_f[i][EVF - 1])))
        for i, k in enumerate(c.orders):
            if (valmask >> i) & 1:
                self.journal.append(
                    ("c", k, keys[i] & 0xFFFFFFFFFFFFFFFF, x, s_h))
        # sparse-tap tables: host-resident even on GPU models (mixed
        # split - zero VRAM); same surprise-scaled write + "C" journal
        # order as the CPU path (after order writes, before "a")
        if c.sparse_taps and not self._predict_only:
            for j, taps in enumerate(c.sparse_taps):
                key = self._gap_key(taps, end)
                if key is None:
                    continue
                tab = self._ptabs[taps]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_SPARSE, j), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s_h
                tots = self._ptots[taps]
                tots[key] = tots.get(key, 0.0) + s_h
                self.journal.append(("C", j, key, x, s_h))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
        # class-order tables: same host-resident split, "K" entries
        if c.class_orders and not self._predict_only:
            for k in c.class_orders:
                key = self._class_key(k, end)
                if key is None:
                    continue
                tab = self._gtabs[k]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_CLASS, k), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s_h
                tots = self._gtots[k]
                tots[key] = tots.get(key, 0.0) + s_h
                self.journal.append(("K", k, key, x, s_h))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
        # word-order tables: ring-scan derived keys, "W" entries
        if c.word_orders and not self._predict_only:
            for j, w in enumerate(c.word_orders):
                key = self._word_key(w, end)
                if key is None:
                    continue
                tab = self._wtabs[j]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_WORD, j), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s_h
                tots = self._wtots[j]
                tots[key] = tots.get(key, 0.0) + s_h
                self.journal.append(("W", j, key, x, s_h))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
        # hybrid class+raw tables: "H" entries
        if c.hybrid_taps and not self._predict_only:
            for j, pat in enumerate(c.hybrid_taps):
                key = self._hybrid_key(pat, end)
                if key is None:
                    continue
                tab = self._htabs[j]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_HYBRID, j), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s_h
                tots = self._htots[j]
                tots[key] = tots.get(key, 0.0) + s_h
                self.journal.append(("H", j, key, x, s_h))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
        # learned-key cell channel: gc_code is the SimHash code of the
        # PRE-x context (caller computes it - device state is already
        # post-update by bookkeep time)
        # recall->gen floor: the device thin flag is 1 when an order
        # cell held evidence at this context - recall-proven tokens
        # get the gen_recall write floor instead of their (small)
        # surprise scale
        s_gen = (max(s_h, c.gen_recall)
                 if c.gen_recall > 0 and bi[self._boff_thin]
                 else s_h)
        if gc_code is not None and c.gcell and not self._predict_only:
            self._gcell_write(gc_code, x, s_gen)
        if gsyn_k is not None and c.gsyn and not self._predict_only:
            self._gsyn_write(gsyn_k, x, s_gen)
        if gsyn_k is not None and c.gsyn_deep \
                and not self._predict_only:
            # deep memory rides the same context key/surprise as the
            # linear channel - separate weights, separate undo scope
            self._gsd_write(gsyn_k, x, s_gen)
        if gsyn_k is not None and c.dnet and not self._predict_only:
            self._dnet_step(gsyn_k, x, s_gen)
        if c.r_live and not self._predict_only:
            # the device step already advanced _c - rl_ctx carries the
            # pre-x context for row replays (_fused_seq); the single-
            # step path recovers it by inverting the topic EMA
            xr = (x if torch.is_tensor(x)
                  else torch.tensor(x)).reshape(1).long().to(self._dev)
            if rl_ctx is not None:
                c_pre = rl_ctx
            else:
                lam = float(c.topic_decay)
                rold = self._R.index_select(0, xr).squeeze(0).float()
                c_pre = ((self._c - rold * (1.0 - lam)) / lam
                         if abs(lam) > 1e-6 else self._c)
            self._rl_write(xr, c_pre, s_gen)
        # live generalizer "gen head" (R68): the order cells this
        # step's keys address now hold the post-write state (the
        # realized token's count included); their recalled successor
        # distribution transfers into the gen channels.  ``gl_cells``
        # carries the batched probe rows on the _fused_seq path;
        # single-step callers take the small re-probe branch.
        if c.gen_live > 0 and not self._predict_only \
                and (gc_code is not None or gsyn_k is not None
                     or c.r_live):
            gl_min = (c.gen_live_min_tot
                      if c.gen_live_min_tot > 0 else c.thin_tot)
            if gl_cells is not None:
                rt_h, rc_h, tot_h, found_h = gl_cells
            else:
                gt = self._gt
                kt = torch.tensor(keys, dtype=torch.int64,
                                  device=self._dev)
                pk = _mix64(kt * _MIX1 + gt._salt[:gt.O])
                slot, found = gt.probe_fixed(pk, rows=gt._ordx)
                rt_h = gt.ttoks[gt._ordx, slot].cpu().numpy()
                rc_h = gt.tcnt[gt._ordx, slot].cpu().numpy()
                tot_h = gt.ttot[gt._ordx, slot].cpu().numpy()
                found_h = found.cpu().numpy()
            votes: dict[int, float] = {}
            conf = 0.0
            for oi in range(len(found_h)):
                if not found_h[oi] or tot_h[oi] <= gl_min:
                    continue
                conf = max(conf,
                           tot_h[oi] / (tot_h[oi] + c.backoff_kappa))
                for t, ct in zip(rt_h[oi], rc_h[oi]):
                    if t >= 0 and ct > 0:
                        votes[int(t)] = votes.get(int(t), 0.0) \
                            + float(ct)
            self._genlive_write(
                votes, conf, gc_code, gsyn_k,
                c_pre if c.r_live else None)
        self.journal.append(("a", x, jcur))
        if cs_ev:
            r_ = bev_i[self._gt.row_csem]
            self.journal.append(
                ("evc", int(r_[0]),
                 int(r_[1]) & 0xFFFFFFFFFFFFFFFF,
                 float(bev_f[self._gt.row_csem][0]),
                 bev_f[self._gt.row_csem]
                 [1:1 + c.csem_dim].copy()))
        if bi[O + 3] >= 0:
            self.journal.append(
                ("sg", int(bi[O + 3]),
                 jcur if (bi[O + 4] & 1) else -1))
        self.journal.append(("r", x, prev_j))
        self.journal.append(("t", x, self.tag))
        if epi_ev:
            r_ = bev_i[self._gt.row_epi]
            ER = self._epi_pos.shape[1]
            LC = len(c.cert_orders)
            self.journal.append(
                ("eve", int(r_[0]),
                 int(r_[1]) & 0xFFFFFFFFFFFFFFFF,
                 int(r_[2]), r_[3:3 + ER].copy(),
                 r_[3 + ER:3 + 2 * ER].astype(np.int32).copy(),
                 r_[3 + 2 * ER:3 + ER * (2 + LC)].copy()))
        if epi_ok:
            self.journal.append(
                ("e", bi[O] & 0xFFFFFFFFFFFFFFFF, end))
        if c.use_knn and bi[self._boff_knn] >= 0:
            ppos = bi[self._boff_knn + 1]
            self.journal.append(
                ("k", bi[self._boff_knn], ppos,
                 bi[self._boff_knn + 2],
                 jcur if ppos >= 0 else -1))
        self._count += 1
        self._jcur = (jcur + 1) % c.vec_delta_cap

    def _learn_token_gpu(self, x: int) -> None:
        """Single-token learn - fused step (logits discarded)."""
        self._fused_step(x)

    def _learn_token(self, x: int) -> None:
        """All live updates for observing token ``x`` ( journaled )."""
        if self._gpu:
            return self._learn_token_gpu(x)
        c = self.config
        end = self._count - 1               # context ends just before x
        keys = self._keys_at(end) if end >= self._ep_start else {}
        pkeys = {j: self._gap_key(taps, end)
                 for j, taps in enumerate(c.sparse_taps)} \
            if c.sparse_taps else {}
        gkeys = {k: self._class_key(k, end)
                 for k in c.class_orders} \
            if c.class_orders else {}
        wkeys = {j: self._word_key(w, end)
                 for j, w in enumerate(c.word_orders)} \
            if c.word_orders else {}
        hkeys = {j: self._hybrid_key(pat, end)
                 for j, pat in enumerate(c.hybrid_taps)} \
            if c.hybrid_taps else {}
        fast = c.fast_ingest and self.tag not in ("live", "gen")
        skip_reads = fast or (c.adaptive_read
                              and self._orders_confident(keys))
        votes = ({} if skip_reads
                 else (self._deep_votes(end)
                       if end >= self._ep_start else {}))
        kvotes = {} if skip_reads else self._knn_votes()
        denom = self._total + c.uni_alpha * self.vocab_size

        # Hedge update on channel weights (reward = channel prob of x)
        skip = ({"deep", "topic", "sem", "knn",
                 "gcell", "gsyn", "gsd", "dnet"}
                if fast else ())
        probs = np.empty(len(self.channels))
        awake = np.ones(len(self.channels), dtype=bool)
        for i, ch in enumerate(self.channels):
            if ch in skip:
                probs[i] = self._uni_p(x, denom)
                awake[i] = False        # skipped read -> abstain
            else:
                probs[i], awake[i] = self._channel_prob(
                    ch, x, keys, votes, denom, kvotes, pkeys, gkeys,
                    wkeys, hkeys)
        # cond_hedge: rewards update the weight set of the regime this
        # context belongs to (the same regime _logits predicted under)
        if c.cond_hedge:
            self._regime = self._regime_now(keys)
            wv = self._wR[self._regime]
            hmx_v = float(self._hmxR[self._regime])
        else:
            wv = self._w
            hmx_v = self._hedge_mix
        if c.sleeping_experts == 2:
            # p_mix over awake channels only - asleep channels abstain
            # and take the mixture's own prob as a neutral reward
            wsum = float((wv * awake).sum())
            p_mix = float((wv * probs * awake).sum()) \
                / max(wsum, 1e-12)
            rewards = np.where(awake, probs, p_mix)
        elif c.sleeping_experts == 1:
            p_mix = float((wv * probs).sum())
            rewards = np.where(awake, probs, p_mix)
        else:
            rewards = probs
            p_mix = float((wv * rewards).sum())
        # AdaHedge: lr adapts to accumulated mixability gap
        if c.adahedge:
            if c.hedge_discount < 1.0:
                hmx_v *= c.hedge_discount
            hmx_v += float(rewards.max() - p_mix)
            eta = min(c.hedge_eta_max,
                      math.log(len(self.channels))
                      / max(hmx_v, 1e-9))
        else:
            eta = c.hedge_lr
        factors = np.exp(eta * (rewards - p_mix))
        self.journal.append(
            self.journal.append_w(wv, hmx_v,
                                  regime=self._regime if c.cond_hedge
                                  else -1))
        if c.cond_hedge:
            self._hmxR[self._regime] = hmx_v
        else:
            self._hedge_mix = hmx_v
        wv *= factors
        np.clip(wv, c.hedge_min, c.hedge_max, out=wv)
        wv /= wv.sum()
        if c.hedge_share > 0:
            # fixed-share diffusion: uniform-share floor so a collapsed
            # channel can re-track after a regime switch
            wv *= (1.0 - c.hedge_share)
            wv += c.hedge_share / len(self.channels)
        if c.cond_hedge:
            self._w[:] = wv      # keep the active-row mirror current

        # surprise s: magnitude of this token's writes (Titans-style)
        # non-finite p_mix (a poisoned channel logit) -> maximally
        # surprising (R60 convention); a NaN s must never reach the
        # _Rl/gsyn write paths - it would poison state permanently
        s = (max(1.0 - p_mix, c.surprise_floor)
             if c.surprise_gate else 1.0)
        if not math.isfinite(s):
            s = 1.0

        # write gate: skip redundant episodic/knn writes when the order
        # tables already cover this context (pre-write cell mass) and
        # the token was unsurprising - thin contexts and surprises
        # still commit
        epi_ok = knn_ok = True
        if c.write_gate:
            pk = c.write_gate_probe_order
            key = keys.get(pk)
            thin = key is None or \
                self._totals[pk].get(key, 0.0) < c.write_gate_thin_tot
            epi_ok = thin or s >= c.write_gate_epi_s
            knn_ok = s >= c.write_gate_knn_s

        # gap tracking: model was surprised here -> teacher target
        if c.track_gaps and keys:
            probe_k = c.gap_probe_order
            key = keys.get(probe_k)
            thin = key is None or \
                self._totals[probe_k].get(key, 0.0) < c.gap_min_total
            if thin or (1.0 - p_mix) > c.gap_surprise:
                self._gaps.append(self._count)

        # unigram
        self._uni[x] += 1.0
        self._total += 1
        self._log_uni_num[x] = math.log(self._uni[x] + c.uni_alpha)
        self.journal.append(("u", x))

        # recall->gen floor, PRE-write: an order cell that already
        # covers this context (tot > thin_tot) means recall predicted
        # the token before we observed it - gen channels learn the
        # proven association at the gen_recall floor rather than the
        # (small) surprise scale.  Pre-write matches the device thin
        # flag, which is computed during the predict phase.
        s_gen = s
        if c.gen_recall > 0 and keys:
            tt = c.thin_tot
            if any(self._totals[k].get(key, 0.0) > tt
                   for k, key in keys.items()):
                s_gen = max(s, c.gen_recall)

        # order tables (surprise-scaled fractional counts) - skipped by
        # predict-only streams: their updates travel in the FluxPacket
        # instead and the single learner applies selected ones post-hoc
        if not self._predict_only:
            for k, key in keys.items():
                tab = self._tables[k]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_ORDER, k), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s
                tots = self._totals[k]
                tots[key] = tots.get(key, 0.0) + s
                self.journal.append(("c", k, key, x, s))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
            # sparse-tap tables - same surprise-scaled write, journaled
            # as "C" so revert routes to _ptabs/_ptots
            for j, taps in enumerate(c.sparse_taps):
                key = pkeys.get(j)
                if key is None:
                    continue
                tab = self._ptabs[taps]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_SPARSE, j), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s
                tots = self._ptots[taps]
                tots[key] = tots.get(key, 0.0) + s
                self.journal.append(("C", j, key, x, s))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
            # class-order tables - "K" entries route to _gtabs/_gtots
            for k, key in gkeys.items():
                if key is None:
                    continue
                tab = self._gtabs[k]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_CLASS, k), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s
                tots = self._gtots[k]
                tots[key] = tots.get(key, 0.0) + s
                self.journal.append(("K", k, key, x, s))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
            # word-order tables - "W" entries route to _wtabs/_wtots
            for j, key in wkeys.items():
                if key is None:
                    continue
                tab = self._wtabs[j]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_WORD, j), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s
                tots = self._wtots[j]
                tots[key] = tots.get(key, 0.0) + s
                self.journal.append(("W", j, key, x, s))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
            # hybrid tables - "H" entries route to _htabs/_htots
            for j, key in hkeys.items():
                if key is None:
                    continue
                tab = self._htabs[j]
                if self._bloom_ready() and key not in tab \
                        and not self._bloom_touch(
                            self._bloom_salt(self._BFAM_HYBRID, j), key):
                    continue
                cell = tab.get(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x] = cell.get(x, 0.0) + s
                tots = self._htots[j]
                tots[key] = tots.get(key, 0.0) + s
                self.journal.append(("H", j, key, x, s))
                if len(tab) > c.max_cells_per_order:
                    self._decay_table(tab, tots)
            # learned-key cell channel: SimHash of the PRE-x context
            # (_c/_c2 still pre-update - _push runs at the end)
            code_e = gk_e = None
            if c.gcell or c.gsyn or c.gsyn_deep or c.dnet:
                # one feature build feeds all four gen channels
                fx = self._gcell_feats(self._c, self._c2, end)
                if c.gcell:
                    code_e = self._gcell_code_from_feats(fx)
                    self._gcell_write(code_e, x, s_gen)
                gk_e = (self._gsyn_keys(fx)
                        if (c.gsyn or c.gsyn_deep or c.dnet) else None)
                if c.gsyn:
                    self._gsyn_write(gk_e, x, s_gen)
                if c.gsyn_deep:
                    self._gsd_write(gk_e, x, s_gen)
                if c.dnet:
                    self._dnet_step(gk_e, x, s_gen)
            if c.r_live:
                # _c still holds the pre-x context (_push runs last):
                # the token's feature row drifts toward where it was
                # seen - co-occurrence embedding written by the stream
                self._rl_write(x, self._c, s_gen)
            # live generalizer "gen head": the order cells' recalled
            # successor distribution (post-write - includes this
            # token) transfers into the gen channels at gen_live*conf.
            # Thin contexts contribute nothing: a first-seen cell's
            # post-write mass ~ s stays below min_tot.
            if c.gen_live > 0 and keys:
                gl_min = (c.gen_live_min_tot
                          if c.gen_live_min_tot > 0 else c.thin_tot)
                votes: dict[int, float] = {}
                conf = 0.0
                for k, key in keys.items():
                    cell = self._tables[k].get(key)
                    if not cell:
                        continue
                    tot = self._totals[k].get(key, 0.0)
                    if tot <= gl_min:
                        continue
                    conf = max(conf, tot / (tot + c.backoff_kappa))
                    for t, ct in cell.items():
                        votes[t] = votes.get(t, 0.0) + ct
                self._genlive_write(
                    votes, conf, code_e, gk_e,
                    self._c if c.r_live else None)

        # topic row: delta-rule write - subtract the current readout
        # along the context direction before committing (error-corrected
        # memory, no double-count of repeated contexts)
        if c.delta_rule:
            cn = float(self._c.norm())
            if cn > 1e-8:
                chat = self._c / cn
                pred = float(self._A[x] @ chat)
                delta_t = chat * ((1.0 - pred) * c.hebb_lr * s)
            else:
                delta_t = torch.zeros_like(self._c)
        else:
            delta_t = self._c * (c.hebb_lr * s)
        if not self._predict_only:
            self._A[x].add_(delta_t)
            self.journal.append(
                self.journal.append_vec(x, delta_t.cpu().numpy()))
            self._csem_write(x, scale=s)
        self._proto.mul_(c.proto_decay).add_(
            self._A[x] * (1.0 - c.proto_decay))

        # recency
        prev = self._last_seen.get(x, -1)
        self._last_seen[x] = end
        self._recent.append((end, x))
        self.journal.append(("r", x, prev))

        # token marker + push
        self.journal.append(("t", x, self.tag))
        self._push(x, epi_ok=epi_ok, knn_ok=knn_ok)

    # -- bulk vectorized ingest (CUDA-primary) --------------------------

    def _ingest_bulk(self, ids: list[int]) -> int:
        """Vectorized chunk ingest on the GPU backend.

        One journal ``bc`` block per call holds a pre-state snapshot plus
        the write arrays - bulk revert is EXACT (snapshot restore + count
        subtraction), better fidelity than per-token EMA inversion.
        Proto EMA uses post-chunk A rows (documented approximation).
        """
        c = self.config
        dev = self._dev
        V = self.vocab_size
        n = len(ids)
        if n == 0:
            return 0
        if c.auto_episode and 0 <= c.eos_token_id:
            eos = c.eos_token_id
            if eos in ids:
                # document boundaries: bulk-ingest each [..EOS] segment
                # as its own episode; boundary is a journaled soft_reset
                total, last = 0, 0
                for i, x in enumerate(ids):
                    if x == eos:
                        total += self._ingest_bulk(ids[last:i + 1])
                        self.soft_reset()
                        last = i + 1
                if last < n:
                    total += self._ingest_bulk(ids[last:])
                return total
        base = self._count
        cap = c.ring_capacity
        ids_np = np.asarray(ids, dtype=np.int64)
        ids_t = torch.from_numpy(ids_np).to(dev)
        pos = torch.arange(base, base + n, dtype=torch.int64,
                           device=dev)
        slots = pos % cap

        blk = {"tag": self.tag, "n": n, "count": base,
               "ep_start": self._ep_start,
               "c": self._c.clone(), "proto": self._proto.clone(),
               "c2": self._c2.clone(),
               "w": self._w.copy(), "uni": self._uni.copy(),
               "hedge_mix": self._hedge_mix,
               "wR": self._wR.copy(), "hmxR": self._hmxR.copy(),
               "regime": self._regime,
               "last_seen": dict(self._last_seen),
               "seen_g": self._seen_g.clone(),
               "unig": self._unig.clone(),
               "log_uni_num": self._log_uni_num.clone(),
               "recent_len": len(self._recent),
               "gaps": len(self._gaps),
               "epi": [], "csem": [], "writes": {}}
        self.journal.append(("bc", blk))
        # per-write entries collect on the block instead of the global
        # journal - a bulk block emits ~4 entries/token, which would
        # otherwise overflow journal_cap mid-ingest and evict this
        # block's own "bc" header (silent un-revertable writes).
        # _undo_block replays blk["sub"] in reverse via _undo_entry.
        self._blk_journal = blk["sub"] = []
        self._blk_active = blk

        # -- ring + prefix hashes (closed form, mod 2^64 via int64 wrap)
        # ring/hash writes are chunked to <=cap positions so each
        # fancy-index scatter has UNIQUE slots - duplicate-index
        # assignment order is undefined on CUDA (numpy is last-wins;
        # the device path must match)
        for off in range(0, n, cap):
            mlen_ = min(cap, n - off)
            self._ringg[slots[off:off + mlen_]] = \
                ids_t[off:off + mlen_].to(torch.int32)
        np_idx = (base + np.arange(n)) % cap
        self._ring[np_idx] = ids_np.astype(np.int32)
        P = 1099511628211
        Pinv = pow(P, -1, 1 << 64)
        if Pinv >= 1 << 63:
            Pinv -= 1 << 64          # signed int64 bit pattern
        pinv_i = torch.cumprod(
            torch.full((n,), Pinv, dtype=torch.int64, device=dev), 0)
        pw_i = torch.cumprod(
            torch.full((n,), P, dtype=torch.int64, device=dev),
            0)                                       # P^{i+1}
        s = torch.cumsum((ids_t + 1) * pinv_i, 0)
        h_prev = (torch.zeros((), dtype=torch.int64, device=dev)
                  if base == 0 else self._Hg[(base - 1) % cap])
        # H[base+i] = h_prev*P^{i+1} + sum_j (id_j+1)*P^{i-j}
        #           = P^{i+1} * (s_i + h_prev)   [s_i has P^{-(j+1)}]
        H_blk = pw_i * (s + h_prev)
        # Local absolute-position hash lookup for every suffix window
        # the block needs.  The RING alone is insufficient: reading
        # Hg[(e) % cap] after the write aliases positions older than
        # `cap` inside the block (slot p%cap ends holding the LAST
        # congruent position's hash).  Hfull[i] = H[base - Kmax - 1 + i]
        # - pre-block tail read BEFORE the ring overwrite.
        Kmax = max(int(max(c.orders)), int(c.deep_order),
                   int(max(c.cert_orders, default=0)),
                   int(c.csem_order))
        pre_ix = torch.arange(base - Kmax - 1, base,
                              dtype=torch.int64, device=dev)
        pre_h = torch.where(
            (pre_ix >= 0) & (pre_ix >= base - cap),
            self._Hg[(pre_ix.clamp(min=0)) % cap],
            torch.zeros((), dtype=torch.int64, device=dev))
        Hfull = torch.cat([pre_h, H_blk])   # Hfull[e - base + Kmax + 1]
        for off in range(0, n, cap):
            mlen_ = min(cap, n - off)
            self._Hg[slots[off:off + mlen_]] = H_blk[off:off + mlen_]
        self._H[np_idx] = H_blk.cpu().numpy().astype(np.uint64)

        # -- per-order context keys + table writes ----------------------
        # ONE batched read probe over [O,n] keys (pre-write state for
        # hedge rewards + gap detection), then sync-free per-order adds.
        i64 = torch.arange(n, dtype=torch.int64, device=dev)
        O = len(c.orders)
        kvec = self._orders_g                                     # [O]
        posb = pos[None, :].expand(O, n)
        # context for token at pos ENDS at pos-1 (matches live
        # _keys_dev(end=pos-1)): key = suffix of tokens pos-k..pos-1
        valid_o = (posb - kvec[:, None] >= self._ep_start) \
            & (posb >= kvec[:, None])                             # [O,n]
        kk_all = Hfull[(posb - base + Kmax)] - Hfull[
            (posb - kvec[:, None] - base + Kmax).clamp(min=0)] \
            * self._pows_g[kvec][:, None]                         # [O,n]
        pk_all = _mix64(kk_all * _MIX1
                        + self._gt._salt[:O, None])
        rows_f = self._gt._ordx[:, None].expand(O, n).reshape(-1)
        slot_f, found_f = self._gt.probe_fixed(
            pk_all.reshape(-1), rows=rows_f)
        # bloom doorkeeper: first-sighting context keys set counter
        # bits instead of materializing rows - hapax-heavy streams
        # would otherwise flood the tables with cells that never fire
        # again.  write_o is the WRITE mask only: found_f/p_o/rewards
        # still see true pre-block state.
        write_o = valid_o
        if self._bloom_ready():
            pf = pk_all.reshape(-1)
            vf = valid_o.reshape(-1)
            miss = vf & ~found_f
            BM = self._bloom_g.numel()
            b1 = pf.remainder(BM)
            b2 = _mix64(pf * _MIX2 + _MIX1).remainder(BM)
            thr = c.bloom_admit - 1
            hit = (self._bloom_g.index_select(0, b1) >= thr) \
                & (self._bloom_g.index_select(0, b2) >= thr)
            # intra-block repeats: a key's 2nd+ sighting in this same
            # block promotes - the first sighting just set its bits
            ordb = torch.argsort(pf)
            pfs = pf[ordb]
            dup = torch.zeros_like(miss)
            dup[ordb[1:]] = pfs[1:] == pfs[:-1]
            admit = found_f | hit | (miss & dup)
            drop = vf & ~admit
            # count every miss sighting (saturating int16 counters)
            midx = torch.nonzero(miss).squeeze(1)
            if midx.numel():
                ones = torch.ones(midx.numel(), dtype=torch.int16,
                                  device=dev)
                self._bloom_g.index_add_(0, b1[midx], ones)
                self._bloom_g.index_add_(0, b2[midx], ones)
                self._bloom_g.clamp_(0, 32767)
            write_o = (vf & ~drop).reshape(O, n)
        denom0 = float(self._total) + c.uni_alpha * V
        uni_pre = (self._unig[ids_t] + c.uni_alpha) / denom0  # [n]
        uni_rep = uni_pre[None, :].expand(O, n).reshape(-1)
        ids_f = ids_t[None, :].expand(O, n).reshape(-1)
        rt_f = self._gt.ttoks[rows_f, slot_f]                     # [On,M]
        rc_f = self._gt.tcnt[rows_f, slot_f]
        tots_f = self._gt.ttot[rows_f, slot_f]
        cnt_x = (rc_f * (rt_f == ids_f[:, None].to(torch.int32))
                 ).sum(dim=1)                                     # [On]
        if c.ppm_escape_c:
            a_f = (((rt_f >= 0) & (rt_f < V)) & found_f[:, None]) \
                .sum(dim=1).clamp(min=1e-9)                       # [On]
        else:
            a_f = float(c.alpha)           # python float - H2D-free
        p_o = torch.where(
            found_f & (tots_f > 0),
            (cnt_x + a_f * uni_rep) / (tots_f + a_f),
            uni_rep)
        # per-token surprise: 1 - p_mix under the (constant-w) chunk
        # mixture - order probs are exact; non-order channels fall back
        # to the unigram prior, same as the bulk reward approximation
        w_o = torch.from_numpy(
            self._w[:O].astype(np.float32)).to(dev)[:, None]
        awake_o = valid_o & found_f.reshape(O, n) \
            & (tots_f.reshape(O, n) > 0)               # [O,n]
        if c.sleeping_experts == 2:
            # renormalized pmix: asleep channels abstain entirely, so
            # the mixture prob comes only from channels that responded.
            # uni/recency/fatigue are always awake - their rewards are
            # hoisted here (same math as the tail block below).
            seen_pre = self._seen_g[ids_t].cpu().numpy()
            enc = ids_np * (n + 1) + np.arange(n)
            order = np.argsort(enc, kind="stable")
            i_s = (enc[order] % (n + 1)).astype(np.int64)
            tok_s = (enc[order] // (n + 1)).astype(np.int64)
            same = tok_s[1:] == tok_s[:-1]
            prev_blk = np.full(n, -1, dtype=np.int64)
            prev_blk[i_s[1:][same]] = i_s[:-1][same]
            prev_pos = np.maximum(base + prev_blk, seen_pre)
            ages = (base + np.arange(n) - 1) - prev_pos
            rec_r = np.where(ages < c.recency_span,
                             c.recency_decay ** np.clip(ages, 0, None),
                             1e-3)
            fat_r = np.where(ages < c.fatigue_span,
                             1.0 - c.fatigue_decay
                             ** np.clip(ages, 0, None), 1.0)
            col = {ch: i for i, ch in enumerate(self.channels)}
            waw = float(self._w[col["uni"]] + self._w[col["recency"]]
                        + self._w[col["fatigue"]])
            tail_np = (self._w[col["uni"]] * uni_pre.cpu().numpy()
                       + self._w[col["recency"]] * rec_r
                       + self._w[col["fatigue"]] * fat_r)
            pmix_i = (((w_o * p_o.reshape(O, n) * awake_o).sum(0)
                       + torch.from_numpy(tail_np).to(dev))
                      / ((w_o * awake_o).sum(0) + waw)
                      .clamp(min=1e-12)).float()
        else:
            w_rest = float(1.0 - self._w[:O].sum())
            pmix_i = (w_o * p_o.reshape(O, n)).sum(0) \
                + w_rest * uni_pre
        s_v = (torch.clamp(1.0 - pmix_i, min=c.surprise_floor)
               if c.surprise_gate
               else torch.ones(n, device=dev))          # [n]
        # a NaN mixture (dead tables, fp blowup) must not propagate
        # into every downstream write - treat as fully surprising
        s_v = torch.nan_to_num(s_v, nan=1.0)
        # in-block recurrence correction for the WRITE deltas: the
        # eager path's surprise decays live as a repeated (context,
        # token) sighting lands in the cell, so counts grow
        # sublinearly (~log n).  A flat pre-block s_v writes ~full
        # deltas on every in-block repeat -> counts grow linearly in
        # frequency and a hot low-order cell swamps the deeper orders
        # that carry the actual answer.  Exclusive group prefix sums
        # give each position the mass earlier same-block positions
        # already wrote under its context (m_ctx) and its context+
        # token pair (m_pair); p_post recomputes the mixture under
        # that evidence.  Doorkeeper-dropped positions contribute no
        # mass (write_o mask).
        sv_flat = torch.where(
            write_o.reshape(-1),
            s_v[None, :].expand(O, n).reshape(-1),
            torch.zeros((), device=dev))
        m_ctx = _group_prefix(pk_all.reshape(-1), sv_flat)
        m_pair = _group_prefix(pk_all.reshape(-1), sv_flat, ids_f)
        tot_p = tots_f + m_ctx
        cnt_p = cnt_x + m_pair
        p_post = torch.where(
            tot_p > 0,
            (cnt_p + a_f * uni_rep) / (tot_p + a_f),
            uni_rep)
        # effective surprise for WRITE deltas: awake-renormalized
        # mixture - a channel that fires with real evidence votes at
        # full strength regardless of the (frozen within-block) hedge
        # weight, matching what the converged sequential mixture would
        # predict.  Without this, uniform early weights keep every
        # position maximally surprising -> gen matrices (gsyn/dnet)
        # blow up to 300x their eager norms.
        awake_p = awake_o.reshape(-1) | (
            valid_o.reshape(-1) & (m_ctx > 0))
        _num = (w_o * p_post.reshape(O, n)
                * awake_p.reshape(O, n)).sum(0)
        _den = (w_o * awake_p.reshape(O, n)).sum(0)
        pmix_e = torch.where(_den > 1e-9,
                             _num / _den.clamp(min=1e-9),
                             uni_pre)
        s_eff = torch.clamp(1.0 - pmix_e, min=c.surprise_floor) \
            if c.surprise_gate else s_v
        s_eff = torch.nan_to_num(s_eff, nan=1.0)
        # per-position order rewards kept for the chunked hedge update
        # below (weights evolve within the block, not one flat update).
        # sleeping experts: asleep positions take pmix (neutral reward)
        # instead of the unigram fallback.
        if c.sleeping_experts:
            pmix_o = pmix_i[None, :].expand(O, n).reshape(-1)
            rw_o = (torch.where(awake_o.reshape(-1), p_o, pmix_o)
                    .double().reshape(O, n))
        else:
            rw_o = (torch.where(valid_o.reshape(-1), p_o, uni_rep)
                    .double().reshape(O, n))           # [O, n]

        # gap detect: unseen context OR high surprise -> teacher target
        if c.track_gaps and c.gap_probe_order in self._gt.kidx:
            gk_i = self._gt.kidx[c.gap_probe_order]
            thin = valid_o[gk_i] & (
                ~found_f.reshape(O, n)[gk_i]
                | (s_v > c.gap_surprise))
            gp = pos[thin]
            if gp.numel():
                self._gaps.extend(gp.cpu().tolist())

        # ONE batched write across all orders, surprise-scaled with
        # the in-block-corrected deltas (distinct-column
        # disambiguation makes it equivalent to sequential adds)
        d_all = s_eff[None, :].expand(O, n)
        ins_rows, ev_snap, col_snap = self._gt.add_multi(
            kk_all, ids_t[None, :].expand(O, n), d_all, write_o)
        # reward fix-up: the pre-block probe misses every context on a
        # fresh first pass, which left order channels at uniform reward
        # for entire corpora - hedge could never differentiate.  Re-
        # probe the POST-write table: a key that recurred within the
        # block now sees the real cell.  Each position's own delta is
        # subtracted back out (self-leak); later same-key writes stay
        # in - a mild forward leak vs sequential semantics, bounded.
        slot2, found2 = self._gt.probe_fixed(
            pk_all.reshape(-1), rows=rows_f)
        new_cell = ~found_f & found2
        if bool(new_cell.any()):
            wrote = write_o.reshape(-1)
            rt2 = self._gt.ttoks[rows_f, slot2]
            rc2 = self._gt.tcnt[rows_f, slot2]
            tot2 = self._gt.ttot[rows_f, slot2]
            cnt2 = (rc2 * (rt2 == ids_f[:, None].to(torch.int32))
                    ).sum(dim=1)
            d_flat = s_eff[None, :].expand(O, n).reshape(-1)
            # writers remove their own delta (self-leak); positions
            # gated by the doorkeeper read the sibling-built cell clean
            cnt_eff = torch.where(wrote, (cnt2 - d_flat).clamp(min=0),
                                  cnt2)
            tot_eff = torch.where(wrote, (tot2 - d_flat).clamp(min=0),
                                  tot2)
            p_post = (cnt_eff + a_f * uni_rep) / (tot_eff + a_f)
            p_eff = torch.where(new_cell, p_post, p_o)
            if c.sleeping_experts:
                # under sequential ingest the cell would exist by this
                # position - the channel was effectively awake
                awake_eff = awake_o.reshape(-1) | (
                    valid_o.reshape(-1) & new_cell & (tot_eff > 0))
                rw_o = torch.where(awake_eff, p_eff, pmix_o
                                   ).double().reshape(O, n)
            else:
                rw_o = torch.where(valid_o.reshape(-1), p_eff,
                                   uni_rep).double().reshape(O, n)
        # stash the full key/valid/delta matrices - undo re-derives.
        # HOST copies: keeping GPU tensors here would pin VRAM for the
        # journal's lifetime (blocks accumulate during corpus ingest)
        blk["writes_all"] = (kk_all.cpu(), ids_t.cpu(),
                             write_o.cpu())
        blk["surprise"] = s_v.cpu()
        blk["surprise_eff"] = s_eff.cpu()   # actual written deltas
        blk["evicted"] = tuple(
            t.cpu() for t in ev_snap)      # victim rows (undo)
        blk["col_evicted"] = tuple(
            t.cpu() for t in col_snap)     # displaced follower cols
        for oi, cnt in enumerate(ins_rows.tolist()):
            if cnt:
                self._gt.used[oi] += cnt
                if oi < O and self._gt.used[oi] \
                        > 0.85 * c.gpu_table_slots:
                    self._gt.decay(c.orders[oi])
                    self._gt.rehash(c.orders[oi])

        # -- unigram (host + mirror) ------------------------------------
        uniq, cnts = np.unique(ids_np, return_counts=True)
        self._uni[uniq] += cnts
        self._total += n
        # unique indices - a plain index_add_ is exact (each dst row
        # touched once); scattering ones over raw ids_t would race
        # float atomics and round differently across runs
        self._unig.index_add_(
            0, torch.from_numpy(uniq.astype(np.int64)).to(dev),
            torch.from_numpy(cnts.astype(np.float32)).to(dev))
        self._log_uni_num = torch.log(self._unig + c.uni_alpha)

        # -- deferred episodic index: ids[i] succeeds ctx-end pos-1 ----
        # entries carry succ + cert suffix hashes -> archived (off-ring)
        # positions keep voting; ring wrap no longer kills deep memory
        e_end = pos - 1
        dk_valid = (e_end - c.deep_order + 1 >= self._ep_start) \
            & (e_end >= c.deep_order - 1) & (e_end >= 0)
        if c.write_gate:
            # thin-context gate on the pre-block cell-mass probe -
            # covered contexts that weren't surprising skip the entry
            pki = self._gt.kidx.get(c.write_gate_probe_order, -1)
            thin_i = (~(found_f.reshape(O, n)[pki]
                        & (tots_f.reshape(O, n)[pki]
                           >= c.write_gate_thin_tot))
                      if pki >= 0
                      else torch.ones(n, dtype=torch.bool, device=dev))
            dk_valid = dk_valid & (thin_i
                                   | (s_v >= c.write_gate_epi_s))
        di = torch.nonzero(dk_valid).squeeze(1)
        if di.numel():
            ee_ = e_end[di]                                  # ctx ends
            # Hfull (absolute-position lookup) - the ring aliases mod
            # cap for block-internal positions older than cap
            dkk = (Hfull[ee_ - base + Kmax + 1]
                   - Hfull[(ee_ - c.deep_order - base + Kmax + 1)
                          .clamp(min=0)]
                   * self._pows_g[c.deep_order])
            LC = len(c.cert_orders)
            certs_d = (torch.stack([
                        torch.where(
                            (ee_ - L + 1 >= self._ep_start)
                            & (ee_ >= L - 1) & (ee_ >= 0),
                            Hfull[ee_ - base + Kmax + 1]
                            - Hfull[(ee_ - L - base + Kmax + 1)
                                    .clamp(min=0)]
                            * self._pows_g[L],
                            torch.zeros((), dtype=torch.int64,
                                        device=dev))
                        for L in c.cert_orders], dim=1)
                       if LC else None)
            # one D2H for keys + positions + certs + succs
            pack = [dkk, ee_]
            if LC:
                pack.append(certs_d.reshape(-1))           # [nd*LC]
            pack.append(ids_t[di])
            both = torch.cat(pack).cpu().numpy()
            nd = di.numel()
            dk_cpu = both[:nd].astype(np.uint64)
            dp_cpu = both[nd:2 * nd]
            off = 2 * nd
            if LC:
                cert_cpu = both[off:off + nd * LC] \
                    .reshape(nd, LC).astype(np.uint64)
                off += nd * LC
            else:
                cert_cpu = np.zeros((nd, 0), np.uint64)
            succ_cpu = both[off:off + nd]
            for key, p, s_, cv in zip(dk_cpu.tolist(),
                                      dp_cpu.tolist(),
                                      succ_cpu.tolist(),
                                      cert_cpu.tolist()):
                dq = self._epi.get(key)
                if dq is None:
                    dq = self._epi[key] = deque(
                        maxlen=c.epi_max_per_key)
                dq.append((int(p), int(s_), tuple(int(v) for v in cv)))
                self._epi_fifo.append((int(key), int(p)))
                self._epi_n += 1
                blk["epi"].append((int(key), int(p)))
            self._epi_evict()
            # GPU epi mirror - two passes so same-chunk duplicate keys
            # land on the inserted row.  2-choice slot collisions can
            # make a new key lose its row to a same-block sibling; cell
            # writes are gated on fe2 (key actually resident) so a
            # loser never scatters ghost cells onto a foreign row.
            re_ = self._gt.row_epi
            pk_e = self._epi_hash(dkk)
            ekr = self._gt.tkeys[re_]
            err = self._gt.traw[re_]
            se, fe, ee = self._gt._probe_row_w(re_, pk_e)
            enew = ~fe
            ev = enew & ee
            # pre-block snapshot of EVERY candidate slot (both probes)
            # - superset of all rows the block can touch: keys, raw dk,
            # payloads.  Undo restores wholesale => exact, and it
            # subsumes victim-row restore for evictions.
            s1a, s2a = self._gt._slots2(pk_e)
            ust = torch.unique(torch.cat([s1a, s2a]))
            pre_key = ekr.index_select(0, ust)
            pre_raw = err.index_select(0, ust)
            pos_r = self._epi_pos.index_select(0, ust).clone()
            cnt_r = self._epi_cnt.index_select(0, ust).clone()
            suc_r = self._epi_succ.index_select(0, ust).clone()
            crt_r = (self._epi_cert.index_select(0, ust).clone()
                     if LC else None)
            if bool(enew.any()):
                ekr[se[enew]] = pk_e[enew]
                err[se[enew]] = dkk[enew]
                if bool(ev.any()):
                    self._epi_pos[se[ev]] = -1
                    self._epi_succ[se[ev]] = -1
                    if LC:
                        self._epi_cert[se[ev]] = 0
                    self._epi_cnt[se[ev]] = 0
            se2, fe2 = self._epi_probe(pk_e)
            # pass-1 losers (key not found at either candidate - the
            # slot went to a same-block sibling) retry their ALTERNATE
            # candidate if it's still free
            alt = torch.where(se2 == s1a, s2a, s1a)
            retry = ~fe2 & (ekr.index_select(0, alt) == 0)
            if bool(retry.any()):
                ekr[alt[retry]] = pk_e[retry]
                err[alt[retry]] = dkk[retry]
                se2, fe2 = self._epi_probe(pk_e)
            post_key = ekr.index_select(0, ust)
            inc = int(((pre_key == 0) & (post_key != 0)).sum())
            if inc:
                self._gt.used[re_] += inc
            blk["epi_gpu"] = (dk_cpu, dp_cpu, ust.cpu(),
                              pos_r.cpu(), cnt_r.cpu(),
                              pre_key.cpu(), pre_raw.cpu(),
                              suc_r.cpu(),
                              (crt_r.cpu() if LC else None))
            # cells only where the key actually occupies the row
            se2w = se2[fe2]
            ee_w = ee_[fe2]
            ids_w = ids_t[di][fe2]
            ER = self._epi_pos.shape[1]
            us, inv_e = torch.unique(se2w, return_inverse=True)
            # occurrence rank per slot - positions sharing a key in
            # this block must land on DISTINCT columns (advanced-index
            # writes with duplicate (slot,col) are last-wins) and the
            # count add must accumulate (index_add_, not +=).
            ar_e = torch.arange(se2w.numel(), dtype=torch.int64,
                                device=dev)
            first_i = torch.full((us.numel(),), se2w.numel(),
                                 dtype=torch.int64, device=dev)
            first_i = first_i.scatter_reduce(0, inv_e, ar_e,
                                             reduce="amin")
            rank_e = ar_e - first_i[inv_e]
            col = (self._epi_cnt[se2w] + rank_e) % ER
            self._epi_pos[se2w, col] = ee_w
            self._epi_succ[se2w, col] = ids_w.to(torch.int32)
            if LC:
                self._epi_cert[se2w, col] = certs_d[fe2]
            self._epi_cnt.index_add_(
                0, se2w, torch.ones_like(se2w))
            self._gt.tocc[re_][se2w] = self._epi_cnt[se2w].float()

        # -- EMAs, exact closed form per chunk --------------------------
        # delta-rule writes: each position's applied delta is stored in
        # the block so undo is an exact index_add_ of the negated delta
        # (no re-derivation; fp32 device tensor, no fp16 rounding)
        CH = c.gpu_bulk_chunk
        pd = c.proto_decay
        blk["ids"] = ids_np
        blk["deltas"] = []
        knn_fps = []
        for c0 in range(0, n, CH):
            xchunk = ids_t[c0:c0 + CH]
            s_chunk = s_v[c0:c0 + CH]
            cprev = self._chunk_cprev(xchunk)      # c_{i-1} per token
            c2prev = (self._chunk_c2prev(xchunk, c0, base, ids_t)
                      if c.bigram_sketch else None)
            if c.use_knn:
                if c.knn_window > 0:
                    # windowed fp: bag-of-features over the W tokens
                    # before each position (content-weighted) - the
                    # EMA buries a 15-token question under ~140 tokens
                    # of prose, so retrieval matched format not text
                    W = int(c.knn_window)
                    m_c = xchunk.numel()
                    wi = (torch.arange(c0, c0 + m_c, device=self._dev)
                          [:, None] - W + torch.arange(
                              W, device=self._dev)[None, :]
                          ).clamp(min=0)
                    wt = ids_t.index_select(
                        0, wi.reshape(-1)).reshape(
                            m_c, W).long().clamp(0, V - 1)
                    rw_ = self._R.index_select(
                        0, wt.reshape(-1)).reshape(
                            m_c, W, -1).float()
                    if c.knn_idf:
                        w_ = self._knn_tok_w(wt.reshape(-1)).reshape(
                            m_c, W)
                        if c.knn_idf_pow != 1.0:
                            w_ = w_ ** float(c.knn_idf_pow)
                        xw = (rw_ * w_[..., None]).sum(1) \
                            / w_.sum(1, keepdim=True).clamp(min=1e-8)
                    else:
                        xw = rw_.mean(1)
                    knn_fps.append(self._knn_fp_of(xw))
                else:
                    knn_fps.append(self._knn_fp_of(cprev))
            if c.delta_rule:
                chat = cprev / (cprev.norm(dim=1, keepdim=True)
                                + 1e-8)
                ax_c = self._A.index_select(0, xchunk)
                pred = (ax_c * chat.float()).sum(dim=1)
                dc_ = chat.float() * ((1.0 - pred)
                                      * c.hebb_lr * s_chunk)[:, None]
            else:
                dc_ = (cprev.float()
                       * (c.hebb_lr * s_chunk)[:, None])
            if _FLUX_DEBUG_NAN:
                _chat = chat if c.delta_rule else None
                for nm, tt in (("cprev", cprev), ("chat", _chat),
                               ("pred", pred if c.delta_rule else None),
                               ("s_chunk", s_chunk),
                               ("dc_", dc_), ("A", self._A),
                               ("proto", self._proto), ("c", self._c),
                               ("pmix", pmix_i[c0:c0 + CH])):
                    if torch.is_tensor(tt) \
                            and (~torch.isfinite(tt)).any():
                        n = int((~torch.isfinite(tt)).sum())
                        nn_ = int(torch.isnan(tt).sum())
                        print(f"[NANDBG] c0={c0} first={nm} "
                              f"nonfinite={n} nan={nn_} "
                              f"absmax={float(tt.abs().max()):.3g}")
                        raise RuntimeError(f"nonfinite {nm} c0={c0}")
            # deterministic grouped add - xchunk repeats token ids and
            # index_add_ float atomics made _A/_Ac state timing-dependent.
            # A hot row can receive hundreds of writes in one chunk, all
            # computed against the SAME pre-chunk pred: the raw sum's
            # effective step is m*lr*s which crosses the delta rule's
            # stability bound and runs away (observed: A -> inf -> NaN
            # at ~300K tok of corpus ingest).  The sequential rule
            # saturates geometrically instead - pred -> 1 as writes
            # accumulate.  Damp each row's grouped delta by that
            # saturation factor phi, blended by directional coherence
            # kappa = |sum chat_i|/m: exact when the duplicate writes
            # align (phi -> geometric limit), near-1 when they are
            # orthogonal (no compounding to damp).
            # one sort of xchunk serves all four grouped reductions -
            # the sort was ~1/3 of post-doorkeeper ingest time when
            # each _group_totals re-sorted the same keys
            xsrt, xord = _group_sort(xchunk)
            u, tot = _group_apply(xsrt, xord, xchunk, dc_)  # [u,d] sums
            _, mcnt = _group_apply(xsrt, xord, xchunk, torch.ones(
                xchunk.numel(), 1, device=dev))
            _, sbar = _group_apply(xsrt, xord, xchunk, s_chunk[:, None])
            sbar = sbar / mcnt.clamp(min=1.0)
            if c.delta_rule:
                _, csum = _group_apply(xsrt, xord, xchunk, chat)
                kap = (csum.norm(dim=1) / mcnt[:, 0]).clamp(0.0, 1.0)
                a = (c.hebb_lr * sbar[:, 0]).clamp(0.0, 0.9999)
                # 1-(1-a)^m without the 1-x cancellation: -expm1(m*log1p(-a))
                phi = (-torch.expm1(mcnt[:, 0]
                                    * torch.log1p(-a))) \
                    / (mcnt[:, 0] * a).clamp(min=1e-9)
                phi = (1.0 - kap * (1.0 - phi)).clamp(0.0, 1.0)
            else:
                phi = torch.ones(u.numel(), device=dev)
            # containment: a pre-corrupt row produces nonfinite dc_
            # (pred=inf) - drop those deltas instead of spreading them
            dtot = torch.nan_to_num(
                tot * phi[:, None], nan=0.0, posinf=0.0, neginf=0.0)
            dtot = dtot.to(self._A.dtype)
            _group_add(self._A, u, dtot)
            blk["deltas"].append((u.cpu(), dtot.cpu()))  # grouped, host
            # keep incremental Ac/An/Acn consistent (predict reads them)
            _group_add(self._Ac, u, dtot @ self._Pcsem)
            axn = self._A.index_select(0, u)
            self._An.index_copy_(0, u, axn.norm(dim=1))
            self._Acn.index_copy_(
                0, u,
                self._Ac.index_select(0, u).norm(dim=1))
            # proto over post-chunk A rows (approximation - documented)
            # algebraically identical decayed sum, but as one bounded
            # matvec - the old cumsum form computed aj * pd^-j which
            # overflows to inf once a row drifts large, and pdj * inf
            # = 0 * inf = NaN poisoned _proto (then s_v -> _A) wholesale
            m = xchunk.numel()
            j = torch.arange(m, dtype=torch.float64, device=dev)
            aj = self._A[xchunk].double()
            wj = (1 - pd) * (pd ** (m - 1 - j))            # [m] <= 1-pd
            if not torch.isfinite(self._proto).all():
                # self-heal: a poisoned proto stays NaN forever under
                # the EMA; zero it and let the sum rebuild
                self._proto = torch.zeros_like(self._proto)
            self._proto = (pd ** m * self._proto.double()
                           + wj @ aj).float()

            # gen channels: per-position context features -> SimHash
            # codes (gcell) and projected keys (gsyn); writes journal
            # individually and undo before the "bc" snapshot restores
            if (c.gcell or c.gsyn or c.gsyn_deep or c.dnet or c.r_live) \
                    and not self._predict_only:
                # recall->gen floor: positions whose context was warm
                # at probe time (pre-write cell tot > thin_tot) write
                # gen channels at gen_recall instead of their surprise.
                # Uses s_eff (in-block-aware) like the order deltas -
                # raw s_v overwrites fuzzy matrices on in-block repeats.
                se_chunk = s_eff[c0:c0 + m]
                s_gen_chunk = se_chunk
                if c.gen_recall > 0:
                    warm_c = (found_f.reshape(O, n)
                              & (tots_f.reshape(O, n)
                                 > c.thin_tot)).any(0)[c0:c0 + m]
                    s_gen_chunk = torch.where(
                        warm_c,
                        se_chunk.clamp(min=c.gen_recall),
                        se_chunk)
            fx = gk = gcodes = None
            if (c.gcell or c.gsyn or c.gsyn_deep or c.dnet) \
                    and not self._predict_only:
                fx = self._gcell_feats_bulk(cprev, c2prev, xchunk,
                                            c0, base, ids_t)
                s_host = s_gen_chunk.tolist()
                ids_l = ids_np[c0:c0 + m].tolist()
                gk = (self._gsyn_keys(fx)             # [m,dk]
                      if (c.gsyn or c.gsyn_deep or c.dnet)
                      else None)
                if c.gcell:
                    gcodes = self._gcell_codes_from_feats(fx)
                    for i, cd in enumerate(gcodes):
                        self._gcell_write(int(cd), ids_l[i],
                                          s_host[i])
                if c.gsyn:
                    # chunked delta rule: one triangular solve per
                    # 64-token block replaces the sequential rank-1
                    # loop (WY rep).  Undo is the block's gsyn
                    # pre-state snapshot (captured on first gen write)
                    if "gsyn" not in blk:
                        blk["gsyn"] = (
                            self._S.detach().cpu().clone(),
                            self._Ss.detach().cpu().clone())
                    self._gsyn_chunk(gk, xchunk, s_gen_chunk)
                if c.gsyn_deep:
                    # nonlinear memory: batched TTT-style grad steps
                    # (LaCT large-chunk).  Undo = blk["gsd"] theta
                    # pre-state, journaled lazily inside _gsd_chunk
                    self._gsd_chunk(gk, xchunk, s_gen_chunk)
                if c.dnet:
                    # iterated stack: exact sequential per-position
                    # steps (layer l needs l-1's current state)
                    self._dnet_chunk(gk, xchunk, s_gen_chunk)
            if c.r_live and not self._predict_only:
                # live features: every token's _Rl row drifts toward
                # its pre-token context sketch (cprev row i = c_{i-1})
                self._rl_bulk(xchunk, cprev, s_gen_chunk)
            # live generalizer "gen head" (R68): the order cells'
            # recalled successor distribution - re-read off the
            # POST-write re-probe (slot2/found2 after add_multi, so
            # cells hold this block's writes like the eager path's
            # post-write read; later same-key writes leak in, the
            # documented reward fix-up class) - transfers into the
            # gen channels at gen_live*conf.
            if c.gen_live > 0 and not self._predict_only:
                gl_min = (c.gen_live_min_tot
                          if c.gen_live_min_tot > 0 else c.thin_tot)
                M = self._gt.M
                sl_c = slot2.reshape(O, n)[:, c0:c0 + m].reshape(-1)
                fd_c = found2.reshape(O, n)[:, c0:c0 + m]
                rows_c = self._gt._ordx[:, None].expand(
                    O, m).reshape(-1)
                tH = self._gt.ttot[rows_c, sl_c].reshape(O, m)
                ev = fd_c & (tH > gl_min)                    # [O,m]
                if bool(ev.any()):
                    rtB = self._gt.ttoks[rows_c, sl_c].reshape(O, m, M)
                    rcB = self._gt.tcnt[rows_c, sl_c].reshape(O, m, M)
                    conf_t = torch.where(
                        ev, tH / (tH + float(c.backoff_kappa)),
                        tH.new_zeros(())).amax(0)            # [m]
                    ok = ev[:, :, None] & (rtB >= 0) & (rcB > 0)
                    rcB = torch.where(ok, rcB, rcB.new_zeros(()))
                    rcB = rcB.permute(1, 0, 2).reshape(m, O * M)
                    rtB = rtB.permute(1, 0, 2).reshape(m, O * M)
                    topm = min(int(c.gen_live_topm), O * M)
                    cnts, ix = torch.topk(rcB, topm, dim=1)
                    toks = rtB.gather(1, ix)                 # [m, topm]
                    tsum = cnts.sum(1, keepdim=True)
                    keep = (conf_t > 0) & (tsum[:, 0] > 0)
                    sel = torch.nonzero(keep).squeeze(1)
                    if sel.numel():
                        s_i = conf_t[sel] * float(c.gen_live)
                        w = cnts[sel] / tsum[sel].clamp(min=1e-9)
                        w = torch.where(cnts[sel] > 0, w,
                                        w.new_zeros(()))
                        tk = toks[sel].long()                # [K, topm]
                        if gk is not None and (c.gsyn
                                               or c.gsyn_deep
                                               or c.dnet):
                            Rm = (self._Reff
                                  if self._Reff is not None
                                  else self._R)
                            vt = (Rm[tk].float()
                                  * w[:, :, None]).sum(1)    # [K, d]
                            if c.gsyn:
                                self._gsyn_chunk(
                                    gk[sel], xchunk[sel], s_i, v=vt)
                            if c.gsyn_deep:
                                self._gsd_chunk(
                                    gk[sel], xchunk[sel], s_i, v=vt)
                            if c.dnet:
                                self._dnet_chunk(
                                    gk[sel], xchunk[sel], s_i, v=vt)
                        if c.r_live:
                            # followers share the position's context
                            # sketch - grouped residual _rl_bulk
                            wm = w.reshape(-1) > 0
                            xs_f = tk.reshape(-1)[wm]
                            cv_f = cprev[sel][:, None, :].expand(
                                -1, topm, -1).reshape(
                                    -1, c.topic_dim)[wm]
                            s_f = (s_i[:, None] * w).reshape(-1)[wm]
                            self._rl_bulk(xs_f, cv_f, s_f)
                        if gcodes is not None and c.gcell:
                            sel_h = sel.tolist()
                            for pi, (trow, wrow, si) in enumerate(
                                    zip(tk.tolist(), w.tolist(),
                                        s_i.tolist())):
                                cd = gcodes[sel_h[pi]]
                                for t_, w_ in zip(trow, wrow):
                                    if w_ > 0:
                                        self._gcell_write(
                                            cd, int(t_), si * w_)

        # -- real knn hedge rewards on the PRE-WRITE bank --------------
        # (the write below would let positions match their own rows)
        knn_r = None
        if c.bulk_channel_rewards and c.use_knn and knn_fps \
                and self._knn_cur > 0:
            fpq = torch.cat(knn_fps).float()                 # [n, d]
            kb = min(int(self._knn_cur), c.knn_capacity)
            # query-blocked top-k: a full [n, kb] sim matrix is
            # n*kb*4B - 37GB at kb~0.5M on a 12GB card.  topk only
            # needs per-row maxima, so scan in query blocks sized to
            # ~2GB of transient memory.
            fpb = self._knn_fp[:kb].float().T                # [d, kb]
            live = (self._knn_pos[:kb] >= 0)[None, :]
            kk = int(min(c.knn_k, kb))
            qb = max(256, min(int(fpq.shape[0]),
                              int(2e9 // (4 * max(kb, 1)))))
            tv_l, ti_l = [], []
            for s0 in range(0, int(fpq.shape[0]), qb):
                sims = fpq[s0:s0 + qb] @ fpb
                sims = torch.where(
                    live, sims,
                    torch.full((), -1e9, device=dev))
                v, i = torch.topk(sims, kk, dim=1)
                tv_l.append(v)
                ti_l.append(i)
            topv = torch.cat(tv_l)
            topi = torch.cat(ti_l)
            kwv = (topv.clamp(min=0.0) ** c.knn_gamma) \
                * (topv > c.knn_min_sim)
            ksucc = self._knn_succ.index_select(
                0, topi.reshape(-1)).long().clamp(0, V - 1) \
                .reshape(n, kk)
            kvt = kwv.sum(1).double()
            kvx = (kwv * (ksucc == ids_t[:, None])).sum(1).double()
            knn_r = torch.where(
                kvt > 0,
                (kvx + c.alpha * uni_pre.double()) / (kvt + c.alpha),
                uni_pre.double()).cpu().numpy()
            knn_fire = (kvt > 0).cpu().numpy()

        # -- knn bank bulk write: (fp of ctx end pos-1) -> succ ids[i] --
        if c.use_knn and knn_fps:
            P = c.knn_capacity
            fpall = torch.cat(knn_fps)                       # [n, d]
            # surprise gate: covered/unsurprising tokens don't earn a
            # bank row - keep only the newest P of the REAL writes
            keep = (s_v >= c.write_gate_knn_s if c.write_gate
                    else torch.ones(n, dtype=torch.bool, device=dev))
            fpall = fpall[keep][-P:]
            kk_ids = ids_t[keep][-fpall.shape[0]:]
            kk_pos = (pos - 1)[keep][-fpall.shape[0]:]
            if c.knn_utility_evict \
                    and self._knn_cur >= P:
                # bank-full: this block's rows take the n coldest
                # slots (score = hits*hit_w + pos/count - same as the
                # live path; new rows get pos=base+i so their recency
                # term already protects them inside this block)
                sc = (self._knn_hits[:P].double() * c.knn_evict_hit_w
                      + self._knn_pos[:P].double()
                      / max(self._count, 1))
                kslots = sc.topk(fpall.shape[0],
                                 largest=False).indices.sort().values
            else:
                kslots = (self._knn_cur + torch.arange(
                    fpall.shape[0], dtype=torch.int64,
                    device=dev)) % P
            alive = self._knn_pos.index_select(0, kslots) >= 0
            blk["knn"] = (kslots.cpu(), alive.cpu(),
                          self._knn_pos.index_select(0, kslots).cpu(),
                          self._knn_succ.index_select(0, kslots).cpu(),
                          self._knn_fp.index_select(0, kslots).cpu())
            self._knn_fp.index_copy_(
                0, kslots, fpall.to(self._knn_fp.dtype))
            self._knn_succ.index_copy_(0, kslots,
                                       kk_ids.to(torch.int32))
            self._knn_pos.index_copy_(0, kslots, kk_pos)
            self._knn_hits.index_fill_(0, kslots, 0)
            self._knn_cur += fpall.shape[0]
            if self._gpu:
                self._knncur_g.fill_(self._knn_cur)

        # -- csem bulk: per-key EMA toward mean of follower features ----
        # context key = suffix ending at pos-1 (excluding x, same as
        # the live _suffix_key(csem_order, end=count-1))
        if c.use_csem:
            ck_valid = (pos - c.csem_order >= self._ep_start) \
                & (pos >= c.csem_order)
            ci = torch.nonzero(ck_valid).squeeze(1)
            if ci.numel():
                ck = (Hfull[pos[ci] - base + Kmax]
                      - Hfull[(pos[ci] - c.csem_order - base
                              + Kmax).clamp(min=0)]
                      * self._pows_g[c.csem_order])
                pk = self._csem_hash(ck)
                rcs = self._gt.row_csem
                ckr = self._gt.tkeys[rcs]
                slot, found, evi = self._gt._probe_row_w(rcs, pk)
                newm = ~found
                ev = newm & evi
                if bool(ev.any()):
                    sv = slot[ev]
                    blk["csem_ev"] = (
                        sv,
                        self._gt.traw[rcs].index_select(
                            0, sv).clone(),
                        self._csem_rows.index_select(
                            0, sv).clone(),
                        self._gt.tocc[rcs].index_select(
                            0, sv).clone())
                if bool(newm.any()):
                    ckr[slot[newm]] = pk[newm]
                    self._gt.traw[rcs][slot[newm]] = ck[newm]
                    if bool(ev.any()):
                        self._csem_rows[slot[ev]] = 0.0
                    self._gt.used[rcs] += int((newm & ~evi).sum())
                # per-slot occurrence counts + surprise-weighted
                # follower feature sums (project AFTER gathering -
                # nxdim matvec, not Vxdim).  Effective lr per slot =
                # csem_lr * mean-surprise (chunk approximation).
                Ac = self._A[ids_t[ci]] @ self._Pcsem     # [n, csem_dim]
                s_ci = s_v[ci].float()
                uniq_s, inv = torch.unique(slot, return_inverse=True)
                m_per = torch.bincount(inv).float()
                # deterministic group sums - float-weighted bincount and
                # index_add_ both accumulate through CUDA atomics, whose
                # arrival order varies run to run (inv repeats by design)
                _isrt, _iord = _group_sort(inv)
                _gi, ssum = _group_apply(_isrt, _iord, inv,
                                         s_ci.float())
                _gi, asum = _group_apply(_isrt, _iord, inv,
                                         Ac.float() * s_ci[:, None])
                amean = asum / ssum.clamp(min=1e-9)[:, None]
                rows = self._csem_rows[uniq_s].float()   # pre-block rows
                slot_new = torch.zeros(uniq_s.numel(), dtype=torch.bool,
                                       device=dev)
                slot_new[inv[newm]] = True
                rows_np = rows.cpu().numpy()
                for i_u, s_ in enumerate(uniq_s.tolist()):
                    blk["csem"].append(
                        (int(s_),
                         None if bool(slot_new[i_u]) else rows_np[i_u]))
                lr_eff = c.csem_lr * (ssum / m_per)
                decay_m = (1.0 - lr_eff) ** m_per
                new_rows = rows * decay_m[:, None] \
                    + amean * (1.0 - decay_m)[:, None]
                new_rows = torch.nan_to_num(new_rows)
                mx = new_rows.abs().amax(dim=1, keepdim=True)
                new_rows = new_rows * (6e4 / mx.clamp(min=6e4))
                self._csem_rows[uniq_s] = new_rows.half()
                _group_add(self._gt.tocc[rcs], slot,
                           torch.ones(slot.numel(), device=dev))

        # -- recency / fatigue state ------------------------------------
        # exact pre-block last_seen + within-block previous occurrence
        seen_pre = self._seen_g[ids_t].cpu().numpy()
        enc = ids_np * (n + 1) + np.arange(n)
        order = np.argsort(enc, kind="stable")
        i_s = (enc[order] % (n + 1)).astype(np.int64)
        tok_s = (enc[order] // (n + 1)).astype(np.int64)
        same = tok_s[1:] == tok_s[:-1]
        prev_blk = np.full(n, -1, dtype=np.int64)
        prev_blk[i_s[1:][same]] = i_s[:-1][same]
        self._seen_g.scatter_reduce_(0, ids_t, pos, reduce="amax")
        # one D2H for all touched tokens - not .item() per token
        uniq = np.unique(ids_np)
        vals = self._seen_g[
            torch.from_numpy(uniq).to(dev)].cpu().tolist()
        for t, v in zip(uniq.tolist(), vals):
            self._last_seen[int(t)] = int(v)
        self._recent.extend(
            (base + i - 1, int(x)) for i, x in enumerate(ids_np))

        # -- hedge, chunk-approximated (constant-w per chunk) ----------
        prev_pos = np.maximum(base + prev_blk, seen_pre)
        ages = (base + np.arange(n) - 1) - prev_pos
        rec_r = np.where(ages < c.recency_span,
                         c.recency_decay ** np.clip(ages, 0, None), 1e-3)
        fat_r = np.where(ages < c.fatigue_span,
                         1.0 - c.fatigue_decay
                         ** np.clip(ages, 0, None), 1.0)
        if c.calib_rewards:
            # implied channel dist p(v) âˆ p_uni(v)Â·exp(Â±lift(v)) ->
            # reward = p_uni(x)Â·exp(f_x)/Z.  Z per position = sliding
            # window over the last `span` tokens (dedup approximated -
            # repeated tokens in the window count twice).
            uni_np = uni_pre.cpu().numpy()
            zr = np.ones(n)
            zf = np.ones(n)
            for L in range(min(c.recency_span, n)):
                zr[L + 1:] += uni_np[:n - 1 - L] * (
                    math.exp(c.recency_decay ** L) - 1.0)
            for L in range(min(c.fatigue_span, n)):
                zf[L + 1:] += uni_np[:n - 1 - L] * (
                    math.exp(-c.fatigue_strength
                             * c.fatigue_decay ** L) - 1.0)
            rec_x = np.where(ages < c.recency_span, rec_r, 0.0)
            fat_x = np.where(ages < c.fatigue_span, 1.0 - fat_r, 0.0)
            rec_r = uni_np * np.exp(rec_x) / zr
            fat_r = uni_np * np.exp(
                -c.fatigue_strength * fat_x) / np.maximum(zf, 1e-12)
        col = {ch: i for i, ch in enumerate(self.channels)}
        nonord = [col[ch] for ch in
                  ("deep", "topic", "sem", "csem", "knn", "backoff",
                   "gcell", "gsyn", "gsd", "dnet") if ch in col]
        uni_d = uni_pre.double()
        uni_np = uni_pre.cpu().numpy()

        # real per-position rewards for the bulk-scored channels
        # (bulk_channel_rewards): positions where the channel fired get
        # the live-path prob; cold positions keep the uni/pmix fallback
        real_r: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        if c.bulk_channel_rewards:
            p_o2 = p_o.reshape(O, n)
            tots2 = tots_f.reshape(O, n)
            cnt2 = cnt_x.reshape(O, n)
            found2 = found_f.reshape(O, n)
            if c.use_backoff:
                okm = valid_o & found2 & (tots2 >= c.backoff_min_tot)
                ordv = self._orders_g.reshape(O, 1).expand(O, n)
                kstar = torch.where(
                    okm, ordv,
                    torch.full((), -1, dtype=torch.int64,
                               device=dev)).argmax(0)
                hit_b = okm[kstar, i64]
                tot_b = tots2[kstar, i64].double()
                cnt_b = cnt2[kstar, i64]
                conf_b = tot_b / (tot_b + c.backoff_kappa)
                bo = torch.where(
                    hit_b,
                    conf_b * (cnt_b + c.alpha * uni_pre.double())
                    / (tot_b + c.alpha),
                    uni_pre.double())
                real_r[col["backoff"]] = (
                    bo.cpu().numpy(), hit_b.cpu().numpy())
            # deep: probe the host epi index (pre-block entries only -
            # this block's own writes are already appended by now)
            deep_r = uni_np.copy()
            deep_fire = np.zeros(n, dtype=bool)
            dk_ok = (e_end - c.deep_order + 1 >= self._ep_start) \
                & (e_end >= c.deep_order - 1) & (e_end >= 0)
            if bool(dk_ok.any()):
                dk_all = (Hfull[e_end - base + Kmax + 1]
                          - Hfull[(e_end - c.deep_order - base
                                   + Kmax + 1).clamp(min=0)]
                          * self._pows_g[c.deep_order])
                dk_cpu = dk_all.cpu().numpy().astype(np.uint64)
                dk_np = dk_ok.cpu().numpy()
                for i in np.nonzero(dk_np)[0]:
                    dq = self._epi.get(int(dk_cpu[i]))
                    if not dq:
                        continue
                    vt = vx = 0
                    xi = int(ids_np[i])
                    for ent in dq:
                        if ent[0] < base:
                            vt += 1
                            vx += ent[1] == xi
                    if vt:
                        deep_r[i] = (vx + c.alpha * uni_np[i]) \
                            / (vt + c.alpha)
                        deep_fire[i] = True
            real_r[col["deep"]] = (deep_r, deep_fire)
            if knn_r is not None:
                real_r[col["knn"]] = (knn_r, knn_fire)
            # fuzzy lift channels: prob of the observed token under
            # each channel's own log-softmax readout, evaluated on
            # POST-block state (documented approximation - matching
            # the proto-EMA convention; a strict pre-write pass would
            # need a second feature walk before any writes).  This is
            # what lets hedge actually punish a harmful fuzzy channel
            # during bulk training - the fixed-weight bypass was the
            # measured cause of the seen_qa 0.82->0.28 regression.
            if self._fidx and (c.gcell or c.gsyn or c.gsyn_deep
                               or c.dnet):
                Rm = (self._Reff if self._Reff is not None
                      else self._R).float()
                xcol = ids_t.long()

                def _p_chan(o_mat, temp):  # [n,d] readouts -> p(x_i)
                    on = o_mat / o_mat.norm(dim=1, keepdim=True
                                            ).clamp(min=1e-8)
                    out = torch.empty(n, dtype=torch.float64,
                                      device=dev)
                    for j0 in range(0, n, 1024):
                        a = (on[j0:j0 + 1024] @ Rm.T) * temp
                        a = torch.nan_to_num(a, nan=0.0)
                        p_i = a.gather(1, xcol[j0:j0 + 1024, None]
                                       ).squeeze(1).double()
                        out[j0:j0 + 1024] = torch.exp(
                            p_i - torch.logsumexp(a, dim=1).double())
                    return out.cpu().numpy()

                gk_all = None
                if any(ch in col for ch in ("gsyn", "gsd", "dnet")):
                    _gk = []
                    for c0r in range(0, n, CH):
                        xr = ids_t[c0r:c0r + CH]
                        f_r = self._gcell_feats_bulk(
                            self._chunk_cprev(xr),
                            (self._chunk_c2prev(xr, c0r, base, ids_t)
                             if c.bigram_sketch else None),
                            xr, c0r, base, ids_t)
                        _gk.append(self._gsyn_keys(f_r))
                    gk_all = torch.cat(_gk).float()
                if "gsyn" in col:
                    rr = uni_np.copy()
                    fr = np.zeros(n, dtype=bool)
                    if float(self._S.norm()) > 1e-8:
                        o_all = gk_all @ self._S.T
                        if c.gsyn_slow_decay > 0:
                            o_all = o_all + gk_all @ self._Ss.T
                        rr = _p_chan(o_all, c.gsyn_temp)
                        fr[:] = True
                    real_r[col["gsyn"]] = (rr, fr)
                    # gsd/dnet share the key tensor
                    if "gsd" in col:
                        rr = uni_np.copy()
                        fr = np.zeros(n, dtype=bool)
                        if self._Wd2 is not None \
                                and float(self._Wd2.norm()) > 1e-8:
                            h_r = torch.relu(
                                gk_all @ self._Wd1.T
                                + self._Sb1)
                            rr = _p_chan(h_r @ self._Wd2.T
                                         + self._Sb2,
                                         c.gsyn_deep_temp)
                            fr[:] = True
                        real_r[col["gsd"]] = (rr, fr)
                    if "dnet" in col:
                        rr = uni_np.copy()
                        fr = np.zeros(n, dtype=bool)
                        if any(float(s_.norm()) > 1e-8
                               for s_ in self._DS):
                            h_r = gk_all
                            for l_ in range(len(self._DS)):
                                kk_ = h_r @ self._Dk[l_].float()
                                kk_ = kk_ / kk_.norm(
                                    dim=1, keepdim=True
                                ).clamp(min=1e-8)
                                h_r = kk_ @ self._DS[l_].T
                                if c.dnet_relu:
                                    h_r = torch.relu(h_r)
                            rr = _p_chan(h_r, c.dnet_temp)
                            fr[:] = True
                        real_r[col["dnet"]] = (rr, fr)
                if "gcell" in col:
                    rr = uni_np.copy()
                    fr = np.zeros(n, dtype=bool)
                    if self._gc:
                        for c0r in range(0, n, CH):
                            xr = ids_t[c0r:c0r + CH]
                            f_r = self._gcell_feats_bulk(
                                self._chunk_cprev(xr),
                                (self._chunk_c2prev(xr, c0r, base, ids_t)
                                 if c.bigram_sketch else None),
                                xr, c0r, base, ids_t)
                            cds = self._gcell_codes_from_feats(f_r)
                            if torch.is_tensor(cds):
                                cds = cds.tolist()
                            for i, cd in enumerate(cds):
                                num = den = 0.0
                                for cc, dst in self._gcell_neighbors(
                                        int(cd)):
                                    cell = self._gc.get(cc)
                                    if cell:
                                        w_ = c.gcell_neighbor_decay \
                                            ** dst
                                        num += w_ * cell.get(
                                            int(ids_np[c0r + i]), 0.0)
                                        den += w_ * self._gct.get(
                                            cc, 0.0)
                                if den > 0:
                                    rr[c0r + i] = (
                                        num + c.alpha * uni_np[c0r + i]
                                        * den) / (den + c.alpha)
                                    fr[c0r + i] = True
                    real_r[col["gcell"]] = (rr, fr)

        # per-position evidence regime (cond_hedge): warm = some order
        # cell held real mass for this position's context - rewards
        # then route to that regime's own weight set
        reg_np = ((valid_o & found_f.reshape(O, n)
                   & (tots_f.reshape(O, n) > c.thin_tot)).any(dim=0)
                  .cpu().numpy().astype(np.int64)
                  if c.cond_hedge else np.zeros(n, dtype=np.int64))

        # chunked multiplicative hedge: one update per gpu_bulk_chunk
        # slice so w evolves through the block (constant-w within a
        # chunk only - much closer to the sequential/live path)
        for c0 in range(0, n, CH):
            c1 = min(c0 + CH, n)
            for r_ in (range(2) if c.cond_hedge else (0,)):
                if c.cond_hedge:
                    sel = np.nonzero(reg_np[c0:c1] == r_)[0]
                    if sel.size == 0:
                        continue
                    ii = c0 + sel
                else:
                    ii = np.arange(c0, c1)
                m_ = len(ii)
                rw = np.zeros(len(self.channels))
                rw[:O] = rw_o[:, ii].sum(dim=1).cpu().numpy()
                u_sum = float(uni_d[ii].sum())
                rw[col["uni"]] = u_sum
                rw[col["recency"]] = float(rec_r[ii].sum())
                rw[col["fatigue"]] = float(fat_r[ii].sum())
                pmix_sum = (float(pmix_i[ii].sum())
                            if c.sleeping_experts else 0.0)
                for ci_ in self._pidx:
                    # host-side tables; no device reward - abstain
                    # (neutral pmix) under sleeping_experts, else the
                    # closed-form abstention below rewrites this slot
                    rw[ci_] = (
                        pmix_sum if c.sleeping_experts else u_sum)
                pmix_np = (pmix_i[ii].cpu().numpy()
                           if c.sleeping_experts else None)
                for ci_ in nonord:
                    rr = real_r.get(ci_)
                    if rr is not None:
                        arr, fired = rr
                        seg = np.where(
                            fired[ii], arr[ii],
                            pmix_np if c.sleeping_experts
                            else uni_np[ii])
                        rw[ci_] = float(seg.sum())
                    elif c.sleeping_experts:
                        # asleep non-order channels get the chunk's mean
                        # pmix - neutral reward instead of the unigram
                        # fallback that under-weighted retrieval channels
                        rw[ci_] = pmix_sum
                    else:
                        rw[ci_] = u_sum
                rw /= m_
                wv = self._wR[r_] if c.cond_hedge else self._w
                if self._pidx and not c.sleeping_experts:
                    # host-resident channels abstain in bulk too:
                    # neutral reward = mixture prob over the other
                    # arms (closed form - rw[p] = p_mix is implicit)
                    aw = ~self._pmask
                    wsum = float((wv * aw).sum())
                    rw[self._pmask] = float((wv * rw * aw).sum()) \
                        / max(wsum, 1e-12)
                p_mix = float((wv * rw).sum())
                # no "h" journal entry: blk["w"]/["wR"] snapshots already
                # hold the pre-block weights - a trailing "h" would be
                # popped as an orphan and double-revert the previous
                # block's update
                if c.adahedge:
                    hmx = (self._hmxR[r_] if c.cond_hedge
                           else self._hedge_mix)
                    hmx += float(rw.max() - p_mix) * m_
                    if c.cond_hedge:
                        self._hmxR[r_] = hmx
                    else:
                        self._hedge_mix = hmx
                    eta = min(c.hedge_eta_max,
                              math.log(len(self.channels))
                              / max(hmx, 1e-9))
                else:
                    eta = c.hedge_lr
                expo = np.clip(eta * m_ * (rw - p_mix), -30.0, 30.0)
                wv *= np.exp(expo)
                np.clip(wv, c.hedge_min, c.hedge_max, out=wv)
                wv /= wv.sum()
                if c.cond_hedge:
                    self._w[:] = wv   # mirror = last-updated row

        # sparse-tap tables (host-resident on GPU too - zero VRAM):
        # vectorized gapped keys gathered from the host ring mirror,
        # per-position surprise-scaled writes journaled as "C" so the
        # entries undo individually before the "bc" snapshot restores
        _tabs_needed = ((c.sparse_taps or c.class_orders
                         or c.word_orders or c.hybrid_taps)
                        and not self._predict_only)
        if _tabs_needed:
            P64 = np.uint64(1099511628211)
            ends = base + np.arange(n) - 1      # context end per token
            s_np = s_v.cpu().numpy().astype(np.float64)
            cls_ring = None

        def _tab_write(tab, tots, kk, xx, dd, rec, salt=None):
            """Dedup-grouped write into a {key:{succ:cnt}} table:
            lexsort+reduceat collapses repeated (key,x) pairs in the
            block into one dict update each - adds commute, so the
            packed journal record `rec` undoes bit-exactly."""
            if not kk.size:
                return
            ob = np.lexsort((xx, kk))
            kk_s, xx_s, dd_s = kk[ob], xx[ob], dd[ob]
            hd = np.ones(kk_s.size, bool)
            hd[1:] = (kk_s[1:] != kk_s[:-1]) | (xx_s[1:] != xx_s[:-1])
            st = np.flatnonzero(hd)
            uk, ux = kk_s[st], xx_s[st]
            ud = np.add.reduceat(dd_s, st)
            if salt is not None and self._bloom_ready():
                # doorkeeper at key granularity: unique keys already
                # resident pass; first-sighting keys set bits and are
                # dropped (their deltas never reach the journal); a
                # key sighted >=2x in this block promotes outright.
                hdk = np.ones(kk_s.size, bool)
                hdk[1:] = kk_s[1:] != kk_s[:-1]
                stk = np.flatnonzero(hdk)
                ukk = kk_s[stk]
                # per-key sighting count (positions, not just groups)
                cnts = np.diff(np.append(stk, kk_s.size))
                mk = cnts >= 2
                have = np.fromiter((int(k_) in tab for k_ in ukk),
                                   dtype=bool, count=ukk.size)
                keepk = self._bloom_keep_np(salt, ukk, have, mk)
                if not keepk.all():
                    kidx = np.searchsorted(ukk, uk)
                    keep = keepk[kidx]
                    uk, ux, ud = uk[keep], ux[keep], ud[keep]
                    if not uk.size:
                        return
            rec += (uk, ux, ud)
            self._jadd(rec)
            tabg = tab.get
            for key, x_, d in zip(uk.tolist(), ux.tolist(),
                                  ud.tolist()):
                cell = tabg(key)
                if cell is None:
                    cell = tab[key] = {}
                cell[x_] = cell.get(x_, 0.0) + d
                tots[key] = tots.get(key, 0.0) + d

        if c.sparse_taps and not self._predict_only:
            for j, taps in enumerate(c.sparse_taps):
                span = max(taps) + 1
                valid = (ends - span + 1 >= self._ep_start) \
                    & (ends >= span - 1)
                keys = np.zeros(n, dtype=np.uint64)
                for off in taps:
                    keys = keys * P64 + self._ring[
                        (ends - off) % cap].astype(np.uint64) \
                        + np.uint64(1)
                vi = np.nonzero(valid)[0]
                _tab_write(self._ptabs[taps], self._ptots[taps],
                           keys[vi], ids_np[vi], s_np[vi],
                           ("Cb", j),
                           self._bloom_salt(self._BFAM_SPARSE, j))
                if len(self._ptabs[taps]) > c.max_cells_per_order:
                    self._decay_table(self._ptabs[taps],
                                      self._ptots[taps])

        # class-order tables (host-resident): contiguous suffix keys
        # over the class LUT applied to the host ring - "K" entries
        if c.class_orders and not self._predict_only:
            if cls_ring is None:
                cls_ring = self._cls[self._ring]
            for k in c.class_orders:
                valid = (ends - k + 1 >= self._ep_start) \
                    & (ends >= k - 1)
                keys = np.zeros(n, dtype=np.uint64)
                for i in range(k):
                    keys = keys * P64 + cls_ring[
                        (ends - i) % cap].astype(np.uint64) \
                        + np.uint64(1)
                vi = np.nonzero(valid)[0]
                _tab_write(self._gtabs[k], self._gtots[k],
                           keys[vi], ids_np[vi], s_np[vi],
                           ("Kb", k),
                           self._bloom_salt(self._BFAM_CLASS, k))
                if len(self._gtabs[k]) > c.max_cells_per_order:
                    self._decay_table(self._gtabs[k],
                                      self._gtots[k])

        # word-order tables (host-resident): keys derive from a
        # backward ring scan (variable word lengths - not vectorizable),
        # "W" entries undo individually before the "bc" restore
        if c.word_orders and not self._predict_only:
            for j, w in enumerate(c.word_orders):
                keys = np.zeros(n, dtype=np.uint64)
                okk = np.zeros(n, dtype=bool)
                for i in range(n):
                    kw = self._word_key(w, int(ends[i]))
                    if kw is not None:
                        keys[i] = np.uint64(kw)
                        okk[i] = True
                vi = np.nonzero(okk)[0]
                _tab_write(self._wtabs[j], self._wtots[j],
                           keys[vi], ids_np[vi], s_np[vi],
                           ("Wb", j),
                           self._bloom_salt(self._BFAM_WORD, j))
                if len(self._wtabs[j]) > c.max_cells_per_order:
                    self._decay_table(self._wtabs[j],
                                      self._wtots[j])

        # hybrid class+raw tables (host-resident): fixed offsets -
        # vectorizable like the sparse/class writes; "H" entries
        if c.hybrid_taps and not self._predict_only:
            if cls_ring is None:
                cls_ring = self._cls[self._ring]
            for j, pat in enumerate(c.hybrid_taps):
                span = max(p[0] for p in pat) + 1
                valid = (ends - span + 1 >= self._ep_start) \
                    & (ends >= span - 1)
                keys = np.zeros(n, dtype=np.uint64)
                for off, raw in pat:
                    vals = self._ring[(ends - off) % cap] if raw \
                        else cls_ring[(ends - off) % cap]
                    keys = keys * P64 + vals.astype(np.uint64) \
                        + np.uint64(1)
                vi = np.nonzero(valid)[0]
                _tab_write(self._htabs[j], self._htots[j],
                           keys[vi], ids_np[vi], s_np[vi],
                           ("Hb", j),
                           self._bloom_salt(self._BFAM_HYBRID, j))
                if len(self._htabs[j]) > c.max_cells_per_order:
                    self._decay_table(self._htabs[j],
                                      self._htots[j])

        self._blk_journal = None
        self.journal.seal_bc(blk)         # payload budget + evict
        self._blk_active = None
        self._epi_flush_sync()            # batched GPU row resync
        self._count += n
        # device scalars + hedge weights mirrors
        self._pos_g.fill_(self._count)
        self._tot_g.fill_(float(self._total))
        self._wg.copy_(torch.from_numpy(self._w).to(dev))
        self._hmg.fill_(self._hedge_mix)
        if c.cond_hedge:
            self._wgR.copy_(torch.from_numpy(self._wR).to(dev))
            self._hmxRg.copy_(
                torch.from_numpy(self._hmxR).to(dev))
        self._next_lite = False       # bulk jump - flag is stale
        return n

    def _chunk_cprev(self, xchunk: torch.Tensor) -> torch.Tensor:
        """Context vector before each token of the chunk, closed form:
        c_i = lam^{i+1}*c0 + (1-lam)*lam^i*cumsum(r_j*lam^{-j}).
        Advances self._c to the chunk end (caller commits it)."""
        c = self.config
        dev = self._dev
        lam = c.topic_decay
        m = xchunk.numel()
        j = torch.arange(m, dtype=torch.float64, device=dev)
        rj = self._R[xchunk].double()
        lamj = lam ** j
        csum = torch.cumsum(rj * (lam ** -j)[:, None], 0)
        C = (lam ** (j + 1))[:, None] * self._c.double() \
            + (1 - lam) * lamj[:, None] * csum
        cprev = torch.cat([self._c[None].double(), C[:-1]])
        self._c = C[-1].float()
        return cprev

    def _chunk_c2prev(self, xchunk: torch.Tensor, c0: int, base: int,
                      ids_t: torch.Tensor) -> torch.Tensor:
        """Sketch state before each token of the chunk, closed form
        (same recurrence as _chunk_cprev on _c):
        c2_i = lam*c2_{i-1} + (1-lam)*bind(R[x_{i-1}], R[x_i]) - the
        chunk-head token binds the pre-chunk token only when that
        position is in-episode.  Advances self._c2 to the chunk end;
        returns [m, d] pre-token states."""
        c = self.config
        m = xchunk.numel()
        if m < 1:
            return torch.zeros(0, c.topic_dim, device=self._dev)
        lam = c.topic_decay
        prev = torch.empty(m, dtype=torch.int64, device=self._dev)
        prev[1:] = xchunk[:-1].long()
        b0_ok = 1.0
        if c0 > 0:
            prev[0] = ids_t[c0 - 1]
        elif base > self._ep_start:
            prev[0] = int(self._ring[(base - 1)
                                     % c.ring_capacity])
        else:
            prev[0] = 0
            b0_ok = 0.0
        b = self._bigram_feats(prev, xchunk.long())
        b[0] *= b0_ok
        j = torch.arange(m, dtype=torch.float64, device=self._dev)
        lamj = lam ** j
        csum = torch.cumsum(b.double() * (lam ** -j)[:, None], 0)
        C = (lam ** (j + 1))[:, None] * self._c2.double() \
            + (1 - lam) * lamj[:, None] * csum
        c2prev = torch.cat([self._c2[None].double(), C[:-1]])
        self._c2 = C[-1].float()
        return c2prev

    def _gcell_feats_bulk(self, cprev: torch.Tensor,
                          c2prev: torch.Tensor | None,
                          xchunk: torch.Tensor, c0: int, base: int,
                          ids_t: torch.Tensor,
                          end_off: int = 0) -> torch.Tensor:
        """[m, in_dim] normalized context features - vectorized
        _gcell_feats; shared by gcell codes and gsyn keys."""
        c = self.config
        dev = self._dev
        m = xchunk.numel()
        d = c.topic_dim
        k = c.head_orders
        base_f = cprev.float() / cprev.float().norm(
            dim=1, keepdim=True).clamp(min=1e-8)
        if c.bigram_sketch and c2prev is not None:
            base_f = torch.cat(
                [base_f, c2prev.float() / c2prev.float().norm(
                    dim=1, keepdim=True).clamp(min=1e-8)], 1)
        if k > 0:
            cap = c.ring_capacity
            ends = base + c0 + torch.arange(m, device=dev) - 1 \
                + end_off
            jj = torch.arange(k, device=dev)
            pos_ij = ends[:, None] - jj[None, :]             # [m,k]
            valid = pos_ij >= 0
            in_new = pos_ij >= base
            tok_new = ids_t[(pos_ij - base).clamp(min=0)].long()
            ringg = getattr(self, "_ringg", None)
            if ringg is None:
                # CPU path: host ring is authoritative (consolidate_gen
                # calls this on non-GPU models too)
                ringg = torch.from_numpy(
                    np.asarray(self._ring, dtype=np.int64)).to(dev)
            tok_old = ringg[
                (pos_ij % cap).clamp(min=0)].long()
            tok = torch.where(valid, torch.where(in_new, tok_new,
                                                 tok_old),
                              torch.zeros_like(tok_new))
            rows = self._R[tok].float() \
                * valid[..., None].float() / (d ** 0.5)
            base_f = torch.cat([base_f, rows.reshape(m, -1)], 1)
        return base_f / base_f.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def _gcell_codes_bulk(self, cprev: torch.Tensor,
                          c2prev: torch.Tensor | None,
                          xchunk: torch.Tensor, c0: int, base: int,
                          ids_t: torch.Tensor,
                          end_off: int = 0) -> list[int]:
        """Vectorized SimHash codes for a bulk chunk - one per
        position, same feature layout as _gcell_feats (fp64 matmul =
        exact integer accumulation, bit-identical to the single-row
        path)."""
        c = self.config
        X = self._gcell_feats_bulk(cprev, c2prev, xchunk, c0, base,
                                   ids_t, end_off)
        return self._gcell_codes_from_feats(X)

    def _gcell_codes_from_feats(self, X: torch.Tensor) -> list[int]:
        """SimHash codes for a feature matrix [m, in_dim] (fp64
        matmul = exact integer accumulation, bit-identical to the
        single-row path)."""
        c = self.config
        q = (X.double() * 512.0).round()
        bits = q @ self._Wk.double()                         # [m,nb]
        b = (bits > 0).to(torch.int64)
        codes = (b * (1 << torch.arange(
            c.gcell_bits, dtype=torch.int64,
            device=self._dev))).sum(1)
        return codes.tolist()

    def _cprev_seq(self, xchunk: torch.Tensor,
                   c0: torch.Tensor | None = None) -> torch.Tensor:
        """[n+1, d] pre-token context states for a row - PURE (does
        not advance self._c); index m = post-row context.  For
        fused-seq replay: write-side code for step i uses row i,
        read-side lift for the emitted logits uses row i+1.

        ``c0`` overrides the seed context (consolidation replays the
        ring seeded at zero - episode boundary reset - instead of the
        live ``self._c``)."""
        c = self.config
        c0 = self._c if c0 is None else c0
        m = xchunk.numel()
        j = torch.arange(m, dtype=torch.float64, device=self._dev)
        rj = self._R[xchunk.long()].double()
        lamj = c.topic_decay ** j
        csum = torch.cumsum(rj * (c.topic_decay ** -j)[:, None], 0)
        C = (c.topic_decay ** (j + 1))[:, None] * c0.double() \
            + (1 - c.topic_decay) * lamj[:, None] * csum
        return torch.cat([c0[None].double(), C])

    def _c2prev_seq(self, xchunk: torch.Tensor,
                    c20: torch.Tensor | None = None,
                    prev0: int | None = None) -> torch.Tensor:
        """[n+1, d] pre-token sketch states - PURE counterpart of
        _chunk_c2prev (does not advance self._c2).

        ``c20``/``prev0`` override the seed sketch and the token
        preceding the chunk (prev0=-1 forces an invalid first pair -
        episode start)."""
        c = self.config
        c20 = self._c2 if c20 is None else c20
        m = xchunk.numel()
        prev = torch.empty(m, dtype=torch.int64, device=self._dev)
        prev[1:] = xchunk[:-1].long()
        b0_ok = 1.0
        if prev0 is None:
            if self._count > self._ep_start:
                prev[0] = int(self._ring[(self._count - 1)
                                         % c.ring_capacity])
            else:
                prev[0] = 0
                b0_ok = 0.0
        elif prev0 < 0:
            prev[0] = 0
            b0_ok = 0.0
        else:
            prev[0] = prev0
        b = self._bigram_feats(prev, xchunk.long())
        b[0] *= b0_ok
        j = torch.arange(m, dtype=torch.float64, device=self._dev)
        lamj = c.topic_decay ** j
        csum = torch.cumsum(b.double() * (lamj ** -j)[:, None], 0)
        C = (c.topic_decay ** (j + 1))[:, None] * c20.double() \
            + (1 - c.topic_decay) * lamj[:, None] * csum
        return torch.cat([c20[None].double(), C])

    def ingest(self, ids, tag: str | None = None, learn: bool = True) -> int:
        """Stream-train on token ids (instant, hot). Returns #tokens."""
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        self._prompt_start = self._count
        if self._gpu:
            self._prompt_g.fill_(self._count)
        prev_tag, prev_learn = self.tag, self.learning
        if tag is not None:
            self.tag = tag
        self.learning = learn
        if self.learning:
            self._rl_session = {}       # one "rB" entry per call
            self._gsd_session = {}      # one "gd" entry per call
            self._dnet_session = {}     # one "dn" entry per call
        try:
            if self.learning and self._gpu:
                try:
                    n = self._ingest_bulk(ids)
                finally:
                    # a mid-block exception must not leave later writes
                    # silently diverted into a dead block's sub-journal
                    self._blk_journal = None
                    if self._blk_active is not None:
                        self.journal.seal_bc(self._blk_active)
                        self._blk_active = None
            else:
                n = 0
                for x in ids:
                    x = int(x)
                    if self.learning:
                        self._learn_token(x)
                    else:
                        self._push(x)
                    n += 1
        finally:
            self._rl_flush()
            self._gsd_flush()
            self._dnet_flush()
        self.tag, self.learning = prev_tag, prev_learn
        return n

    def write_counts(self, ctx_end: int, deltas: dict[int, float],
                     tag: str = "distill",
                     families: tuple[str, ...] = ("orders",)) -> int:
        """Write fractional follower counts under the context keys valid
        at absolute position ``ctx_end`` (teacher distillation / manual
        knowledge injection).  Journaled under ``tag`` - revertible via
        ``revert_tag``.  Returns number of cell writes.

        Uses RAW suffix hashes (not episode-gated keys): the stored key
        still matches whenever the same token sequence recurs, which is
        what retrieval needs.  Only writes orders whose tokens are still
        in the ring.

        ``families`` adds the generalizing host-side key families -
        "class" (g<k>), "sparse" (p<j>), "hybrid" (h<j>), "word" (w<j>).
        These keys recur on NOVEL text (class profile / gapped taps),
        so teacher mass written there transfers to contexts never seen
        verbatim - order keys alone only fire on repeated contexts."""
        c = self.config
        keys = {}
        floor = max(0, self._count - c.ring_capacity)
        if "orders" in families:
            for k in c.orders:
                if ctx_end - 1 - k + 1 >= floor:
                    keys[k] = self._raw_hash(k, ctx_end - 1)
        self.journal.append(("ws", tag))      # write-block start marker
        n = self._write_counts_keys(keys, deltas)
        end = ctx_end - 1
        if "class" in families:
            for k in c.class_orders:
                key = self._class_key(k, end)
                if key is None:
                    continue
                n += self._write_host_cell(self._gtabs[k],
                                           self._gtots[k], key,
                                           deltas, "K", k)
        if "sparse" in families:
            for j, taps in enumerate(c.sparse_taps):
                key = self._gap_key(taps, end)
                if key is None:
                    continue
                n += self._write_host_cell(self._ptabs[taps],
                                           self._ptots[taps], key,
                                           deltas, "C", j)
        if "hybrid" in families:
            for j, pat in enumerate(c.hybrid_taps):
                key = self._hybrid_key(pat, end)
                if key is None:
                    continue
                n += self._write_host_cell(self._htabs[j],
                                           self._htots[j], key,
                                           deltas, "H", j)
        if "word" in families:
            for j, w in enumerate(c.word_orders):
                key = self._word_key(w, end)
                if key is None:
                    continue
                n += self._write_host_cell(self._wtabs[j],
                                           self._wtots[j], key,
                                           deltas, "W", j)
        self.journal.append(("we", tag))      # write-block end marker
        return n

    def _write_host_cell(self, tab: dict, tots: dict, key: int,
                         deltas: dict[int, float], op: str,
                         ident: int) -> int:
        """Add fractional counts to one host-side table cell and journal
        each delta (op = "C"/"K"/"H"/"W", ident = tap index or class
        order - must match the live-write journal format exactly so
        revert_tag undoes distillation cleanly)."""
        cell = tab.get(key)
        if cell is None:
            cell = tab[key] = {}
        for t, d in deltas.items():
            cell[t] = cell.get(t, 0.0) + d
            tots[key] = tots.get(key, 0.0) + d
            self.journal.append((op, ident, key, int(t), float(d)))
        return len(deltas)

    # -- bloom doorkeeper ------------------------------------------------
    # Family salts decorrelate the shared filters per table identity:
    # order cells use k*PHI (matching the device probe salt so a key's
    # admission state is consistent if the same key is ever probed on
    # both paths); host families get disjoint bands.
    _BFAM_ORDER = 0
    _BFAM_SPARSE = 1 << 32
    _BFAM_CLASS = 2 << 32
    _BFAM_WORD = 3 << 32
    _BFAM_HYBRID = 4 << 32
    _BPHI = 0x9E3779B97F4A7C15

    def _bloom_salt(self, fam: int, ident: int) -> int:
        return (((fam + ident) * self._BPHI) & 0xFFFFFFFFFFFFFFFF)

    def _bloom_ready(self) -> bool:
        """Lazy allocator: config is user-flippable post-init (eval
        --set), so the filters materialize on first gated write."""
        c = self.config
        if c.bloom_admit < 2:
            return False
        if self._bloom_h is None:
            self._bloom_h = np.zeros(c.bloom_bits, dtype=np.int16)
        if self._gpu and self._bloom_g is None:
            self._bloom_g = torch.zeros(
                c.bloom_bits, dtype=torch.int16, device=self._dev)
        return True

    def _bloom_touch(self, salt: int, key: int) -> bool:
        """Host admission test-and-set for one (family-salt, key):
        False = first sightings (bits now set, no cell yet), True =
        promote (the key is worth a real row)."""
        b = self._bloom_h
        if b is None:
            return True
        M = b.shape[0]
        pk = _mix64_py((key * 0x9E3779B97F4A7C15 + salt)
                       & 0xFFFFFFFFFFFFFFFF)
        b1 = pk % M
        b2 = _mix64_py((pk * 0xBF58476D1CE4E5B9 + 0x9E3779B97F4A7C15)
                       & 0xFFFFFFFFFFFFFFFF) % M
        thr = self.config.bloom_admit - 1
        hit = b[b1] >= thr and b[b2] >= thr
        if b[b1] < 32767:
            b[b1] += 1
        if b[b2] < 32767:
            b[b2] += 1
        return hit

    def _bloom_keep_np(self, salt: int, uk: np.ndarray,
                       have: np.ndarray, multi: np.ndarray
                       ) -> np.ndarray:
        """Vectorized admission for grouped bulk writes: ``uk`` =
        unique candidate keys, ``have`` = already-resident in the
        table, ``multi`` = sighted >1 time in this block (the first
        sighting set its bits, so repeats promote).  Sets bits on all
        absent keys and returns the keep mask."""
        b = self._bloom_h
        if b is None:
            return np.ones(uk.shape, dtype=bool)
        M = np.uint64(b.shape[0])
        thr = np.int16(self.config.bloom_admit - 1)
        pk = _mix64_np(uk.astype(np.uint64)
                       * _MIX1U + np.uint64(salt))
        b1 = (pk % M).astype(np.int64)
        b2 = (_mix64_np(pk * _MIX2U + _MIX1U) % M).astype(np.int64)
        hit = (b[b1] >= thr) & (b[b2] >= thr)
        miss = ~have
        np.add.at(b, b1[miss], 1)
        np.add.at(b, b2[miss], 1)
        # saturating counters: clamp only the touched cells (a full-
        # array np.minimum would be O(bits) per table-write call)
        ti = np.unique(np.concatenate([b1[miss], b2[miss]]))
        b[ti] = np.minimum(b[ti], 32767)
        return have | (miss & (hit | multi))

    def _write_counts_keys(self, keys: dict[int, int],
                           deltas: dict[int, float]) -> int:
        """Shared cell writer for ``write_counts``/``teach``: adds each
        delta under every (order, key) pair and journals the ``c``
        entries.  Returns cells written."""
        n = 0
        if self._gpu:
            toks_l = list(deltas.keys())
            ds_l = [float(deltas[t]) for t in toks_l]
            for k, key in keys.items():
                ctx_t = torch.full((len(toks_l),), _s64(key),
                                   dtype=torch.int64, device=self._dev)
                toks_t = torch.tensor(toks_l, dtype=torch.int64,
                                      device=self._dev)
                d_t = torch.tensor(ds_l, dtype=torch.float32,
                                   device=self._dev)
                self._gt.add(k, ctx_t, toks_t, d_t)
                for t, d in deltas.items():
                    self.journal.append(("c", k, key, t, d))
                    n += 1
            return n
        for k, key in keys.items():
            tab = self._tables[k]
            cell = tab.get(key)
            if cell is None:
                cell = tab[key] = {}
            tots = self._totals[k]
            for t, d in deltas.items():
                cell[t] = cell.get(t, 0.0) + d
                tots[key] = tots.get(key, 0.0) + d
                self.journal.append(("c", k, key, t, d))
                n += 1
        return n

    # -- instant hot-training --------------------------------------------

    @staticmethod
    def _prefix_hashes(ids) -> np.ndarray:
        """Model-compatible rolling prefix hashes over a token list -
        the same recurrence the ring maintains, standalone."""
        h = np.empty(len(ids), dtype=np.uint64)
        prev = 0
        for i, t in enumerate(ids):
            prev = (prev * 1099511628211 + int(t) + 1) % (1 << 64)
            h[i] = prev
        return h

    def _seq_key(self, H: np.ndarray, k: int, e: int) -> int | None:
        """Suffix key of length ``k`` ending at seq index ``e``
        (0-based, seq-local - no ring, no episode gate)."""
        if e - k + 1 < 0 or e < k - 1:
            return None
        return (int(H[e]) - int(H[e - k]) * int(self._pows[k])) \
            % (1 << 64)

    def _context_vector(self, ids: list[int]) -> torch.Tensor:
        """Standalone topic-EMA context vector for arbitrary ids -
        same weighting ``self._c`` accumulates live (the constant
        ``1 - lam`` factor is irrelevant: fingerprints normalize)."""
        c = self.config
        if not ids:
            return torch.zeros(c.topic_dim, device=self._dev)
        idx = torch.tensor(ids, dtype=torch.int64, device=self._dev)
        w = (c.topic_decay ** torch.arange(
            len(ids) - 1, -1, -1, dtype=torch.float32,
            device=self._dev))
        return (self._R[idx].float() * w[:, None]).sum(0)

    def teach(self, context_ids, target_ids, weight: float = 25.0,
              tag: str = "teach", generalize: bool = True) -> dict:
        """Instant hot-training: bind ``context -> target``.

        For every target position, writes ``weight`` follower-count
        under every valid order key of its context (exact recall -
        keys are computed from the token sequence itself, so no ring
        presence is needed).  With ``generalize`` the csem fingerprint
        of each context EMAs toward the target's A-row and one knn bank
        entry binds the context fingerprint to the first target token -
        *similar* contexts then also lift the answer.

        All writes are journaled inside one ``ws``/``we`` block under
        ``tag`` - ``revert_tag(tag)`` un-teaches.  Idempotent-ish:
        repeated calls keep adding count mass.

        Returns {cells, csem, knn} write counts.
        """
        c = self.config
        ctx = [int(t) for t in context_ids]
        tgt = [int(t) for t in target_ids]
        rep = {"cells": 0, "csem": 0, "knn": 0}
        if not tgt:
            return rep
        seq = ctx + tgt
        H = self._prefix_hashes(seq)
        self.journal.append(("ws", tag))
        for i, t in enumerate(tgt):
            e = len(ctx) + i - 1                # context end in seq
            keys = {k: self._seq_key(H, k, e) for k in c.orders}
            keys = {k: v for k, v in keys.items() if v is not None}
            rep["cells"] += self._write_counts_keys(keys, {t: weight})
            if not generalize:
                continue
            ck = self._seq_key(H, c.csem_order, e)
            if ck is not None and c.use_csem:
                self._csem_write_key(ck, t, scale=1.0)
                rep["csem"] += 1
            if i == 0 and c.use_knn and ctx:
                # one bank entry: fp(context) -> first target token
                if c.knn_window > 0:
                    wids = torch.as_tensor(
                        ctx[-int(c.knn_window):], dtype=torch.long,
                        device=self._dev).clamp(0, self.vocab_size - 1)
                    rows = self._R.index_select(0, wids).float()
                    if c.knn_idf:
                        w = self._knn_tok_w(wids)
                        fp = self._knn_fp_of(
                            (rows * w[:, None]).sum(0)
                            / w.sum().clamp(min=1e-8))
                    else:
                        fp = self._knn_fp_of(rows.mean(0))
                else:
                    fp = self._knn_fp_of(self._context_vector(ctx))
                self._knn_write(t, self._count + i, fp=fp)
                if self._gpu:
                    self._jcur = (self._jcur + 1) % c.vec_delta_cap
                rep["knn"] += 1
            if (c.gcell or c.gsyn or c.gsyn_deep or c.dnet) and ctx:
                gxf = self._gcell_feats_ctx(ctx + tgt[:i])
            if c.gcell and ctx:
                # learned-code cell: code(prefix) -> target, repeated
                # delta steps so a strong teach converges the score
                code = self._gcell_codes_from_feats(gxf[None])[0]
                for _ in range(max(1, min(int(round(weight)), 12))):
                    self._gcell_write(code, t, 1.0)
                rep["gcell"] = rep.get("gcell", 0) + 1
            if c.gsyn and ctx:
                # dense delta channel: key(prefix) -> target's _R row,
                # repeated steps so teach mass lands like the cells
                gk = self._gsyn_keys(gxf)
                for _ in range(max(1, min(int(round(weight)), 12))):
                    self._gsyn_write(gk, t, 1.0)
                rep["gsyn"] = rep.get("gsyn", 0) + 1
            if c.gsyn_deep and ctx:
                # deep channel gets the same repeated writes - theta
                # snapshots journal once per write scope
                gkd = self._gsyn_keys(gxf)
                for _ in range(max(1, min(int(round(weight)), 12))):
                    self._gsd_write(gkd, t, 1.0)
                rep["gsyn_deep"] = rep.get("gsyn_deep", 0) + 1
            if c.dnet and ctx:
                # iterated stack: teach mass writes through all layers
                gkn = self._gsyn_keys(gxf)
                for _ in range(max(1, min(int(round(weight)), 12))):
                    self._dnet_step(gkn, t, 1.0)
                rep["dnet"] = rep.get("dnet", 0) + 1
            if c.r_live and ctx:
                # the taught target's feature row drifts toward this
                # context too - instant semantic binding, no teacher;
                # per-token "rL" entries ride the ws/we block and undo
                # through revert_writes
                self._rl_write(
                    t, self._context_vector(ctx + tgt[:i]), 1.0)
        self.journal.append(("we", tag))
        return rep

    def teach_text(self, context: str, target: str, tokenizer,
                   **kw) -> dict:
        """``teach`` on raw text (tokenizes context/target)."""
        ctx = tokenizer(context, add_special_tokens=False)
        tgt = tokenizer(target, add_special_tokens=False)
        ctx_ids = getattr(ctx, "input_ids", getattr(ctx, "ids", ctx))
        tgt_ids = getattr(tgt, "input_ids", getattr(tgt, "ids", tgt))
        return self.teach(list(ctx_ids), list(tgt_ids), **kw)

    # -- self-evolution: reinforce / consolidate -----------------------

    def reinforce(self, tag: str, gain: float = 1.0,
                  span: tuple[int, int] | None = None) -> dict:
        """Outcome-weighted replay of a tag's learned cell writes.

        Re-applies every ``c`` delta journaled under ``tag`` scaled by
        ``gain`` - the self-play reward hook: ``gain>0`` strengthens
        what a successful tool-use trajectory learned, ``gain<0``
        weakens (cells floor at 0).  The replay is journaled under
        ``reinforce:<tag>`` so reinforcement itself is revertible.
        Forgetting entirely is ``revert_tag(tag)``.

        ``span``: optional ``(lo, hi)`` absolute stream positions â€”
        only deltas written for tokens inside the range are replayed
        (fine-grained credit: reinforce just the answer span of a
        partially-good episode).  ``gain`` may be a callable
        ``gain(pos) -> float`` for position-shaped credit.
        """
        # absolute positions: "t" markers carry no position, so count
        # journaled tokens and anchor to _count (correct under journal
        # truncation as long as evictions are prefix drops)
        n_tok = 0
        for e in self.journal.entries:
            if e[0] == "t":
                n_tok += 1
            elif e[0] == "bc":
                n_tok += e[1]["writes_all"][0].shape[1]
        base = self._count - n_tok
        pos = base
        deltas = []           # (k, key, t, d)
        gains = []            # stream position each delta belongs to
        pend = []             # buffered deltas of the open token group
        cur = None
        for e in self.journal.entries:
            if e[0] == "ws":
                cur = e[1]
            elif e[0] == "we":
                # position-free writes (teach/write_counts - no "t"
                # group) flush only in unfiltered mode; a span
                # deliberately excludes them
                if cur == tag and pend and span is None:
                    deltas.extend(pend)
                    gains.extend([-1] * len(pend))
                cur = None
                pend = []
            elif e[0] == "t":
                # "t" CLOSES its token's write group (appended last in
                # _learn_token) - buffered deltas belong to this pos
                p = pos
                pos += 1
                if cur == tag and pend and (
                        span is None or span[0] <= p < span[1]):
                    deltas.extend(pend)
                    gains.extend([p] * len(pend))
                pend = []
            elif e[0] == "c" and cur == tag:
                pend.append((e[1], e[2], e[3], e[4]))
            elif e[0] == "bc" and cur == tag:
                # bulk-ingest block inside a tagged span: its order-cell
                # writes are (key, succ, surprise) per (order, token) -
                # the "c" entries' vectorized form.  Amplifying them
                # keeps reinforce() live on the cuda_primary path.
                if pend and span is None:
                    deltas.extend(pend)
                    gains.extend([-1] * len(pend))
                pend = []
                blk = e[1]
                kk_all, tt_all, valid_o = blk["writes_all"]
                s_v = blk.get("surprise_eff", blk.get("surprise"))
                b0 = blk["count"]
                for oi, k in enumerate(self.config.orders):
                    idx = torch.nonzero(valid_o[oi]).squeeze(1)
                    for j in idx.tolist():
                        p = b0 + j
                        if span is not None and not (
                                span[0] <= p < span[1]):
                            continue
                        deltas.append(
                            (k, int(kk_all[oi, j]),
                             int(tt_all[j]),
                             float(s_v[j]) if s_v is not None
                             else 1.0))
                        gains.append(p)
                pos = b0 + kk_all.shape[1]
        if not deltas:
            return {"found": False, "applied": 0, "tag": tag}
        rtag = f"reinforce:{tag}"
        self.journal.append(("ws", rtag))
        for (k, key, t, d), p in zip(deltas, gains):
            g = float(gain(p)) if callable(gain) else gain
            self._write_counts_keys({k: key}, {t: d * g})
        self.journal.append(("we", rtag))
        if (callable(gain) or gain < 0):
            self._clamp_counts({(k, key) for k, key, _, _ in deltas})
        return {"found": True, "applied": len(deltas), "tag": rtag,
                "gain": gain if not callable(gain) else "callable"}

    def reinforce_alts(self, tag: str, packet, gain: float,
                       scale: float = 0.5,
                       block_idx: int = 0) -> dict:
        """Counterfactual credit: boost the untried tokens the sampler
        recorded in ``packet.alts`` (logged-bandit propensities).

        For each generated position j, the block journaled under
        ``tag`` supplies the order-cell keys the model actually used;
        every alternative token a != sampled gets a write scaled by
        its share of the counterfactual mass pi(a)/(1-pi(sampled)).
        On success the runner-ups share a little of the win; on
        failure (call with gain>0 then revert_writes(tag)) the untried
        actions keep credit for what the tried one lost.  The writes
        are journaled under ``cf:<tag>`` â€” independently revertible.

        Requires the packet to have been applied under ``tag`` first
        (the journal block is the source of the per-position keys â€”
        "c"/"t" groups on the host path, "bc" writes_all on GPU).
        """
        alts = getattr(packet, "alts", None) or []
        if not alts:
            return {"found": False, "applied": 0}
        pl = len(packet.prompt_ids)
        gen = packet.gen_ids
        # collect per-position {order: key} maps per ws/we(tag) block -
        # apply_packets gives each packet its own block, so block order
        # matches apply order; block_idx selects this packet's block
        blocks: list[list[dict]] = []
        cur_keys: dict = {}
        groups: list[dict] = []
        cur = None
        for e in self.journal.entries:
            if e[0] == "ws":
                cur = e[1]
                groups = []
                cur_keys = {}
            elif e[0] == "we":
                if cur == tag:
                    blocks.append(groups)
                cur = None
            elif cur != tag:
                continue
            elif e[0] == "c":
                cur_keys[e[1]] = e[2]
            elif e[0] == "t":
                groups.append(cur_keys)
                cur_keys = {}
            elif e[0] == "bc":
                kk_all = e[1]["writes_all"][0]
                for j in range(kk_all.shape[1]):
                    groups.append(
                        {k: int(kk_all[oi, j])
                         for oi, k in enumerate(self.config.orders)
                         if int(kk_all[oi, j])})
        if block_idx >= len(blocks):
            return {"found": False, "applied": 0}
        groups = blocks[block_idx]
        rtag = f"cf:{tag}"
        writes = 0
        self.journal.append(("ws", rtag))
        for j, alt_list in enumerate(alts):
            gi = pl + j
            if gi >= len(groups) or j >= len(gen) or not alt_list:
                continue
            keys = groups[gi]
            if not keys:
                continue
            sampled = int(gen[j])
            pi_s = next((p for t, p in alt_list if t == sampled),
                        None)
            if pi_s is None or pi_s >= 0.999:
                continue          # degenerate (greedy) - no cf mass
            rem = 1.0 - pi_s
            kw = {int(k): int(kk) for k, kk in keys.items()}
            for tok, pi in alt_list:
                if int(tok) == sampled or pi <= 0:
                    continue
                self._write_counts_keys(
                    kw, {int(tok): gain * scale * (pi / rem)})
                writes += 1
        self.journal.append(("we", rtag))
        return {"found": True, "applied": writes, "tag": rtag}

    def apply_packets(self, packets, accept=None) -> dict:
        """Apply stream update packets through the single learning
        process - serialized ``ingest`` of each accepted packet under
        its tag, wrapped in a ``ws``/``we`` write-block so the applied
        writes are visible to ``reinforce(tag)`` (order-cell deltas) and
        wholesale-revertible via ``revert_writes(tag)``/``revert_tag``.

        ``accept``: None (apply all), a bool sequence aligned with
        ``packets``, or a predicate ``accept(packet) -> bool``.  This is
        the gate that decides which concurrent-generation packets are
        worth learning and which are discarded.
        """
        applied = []
        for i, p in enumerate(packets):
            if callable(accept):
                ok = bool(accept(p))
            elif accept is not None:
                ok = bool(accept[i])
            else:
                ok = True
            if not ok:
                continue
            self.journal.append(("ws", p.tag))
            self.ingest(p.ids, tag=p.tag)
            self.journal.append(("we", p.tag))
            applied.append(i)
        return {"applied": applied}

    def _clamp_counts(self, touched: set) -> None:
        """Floor negative cell counts at 0 after negative-gain writes -
        counts are probability masses; negative cells corrupt the
        channel mixtures.  ``touched`` = (order, key) rows to check."""
        if self._gpu:
            self._gt.tcnt.clamp_(min=0)
            self._gt.ttot.copy_(self._gt.tcnt.sum(-1))
            self._gt.tocc.copy_(self._gt.ttot)
            return
        for k, key in touched:
            cell = self._tables[k].get(key)
            if not cell:
                continue
            for t in list(cell):
                if cell[t] < 0:
                    cell[t] = 0.0
            self._totals[k][key] = sum(cell.values())

    def consolidate(self, min_entries: int = 4, min_agree: float = 0.5,
                    mass: float = 1.0, tag: str = "consol") -> dict:
        """Self-distillation: migrate repeated episodic patterns into
        the fast n-gram tables, then free their episodic cells.

        For each episodic key with >= ``min_entries`` cells whose
        successors agree at fraction >= ``min_agree``, the modal
        successor is written into the order tables at every on-ring
        position (``write_counts`` - journaled under ``tag``), and the
        promoted cells are dropped from the episodic deque (journaled
        ``eE`` - revertible).  Off-ring cells stay archived: their
        order keys can't be recomputed from a raw suffix hash, so their
        only faithful store is the episodic row itself.

        This is the math-driven counterpart of SFT: frequently-
        retrieved context->successor pairs promote from expensive
        episodic storage to O(1) n-gram reads."""
        c = self.config
        from collections import Counter
        floor = max(0, self._count - c.ring_capacity)
        rep = {"keys": 0, "promoted": 0, "freed": 0}
        self.journal.append(("ws", tag))
        for key, dq in list(self._epi.items()):
            if len(dq) < min_entries:
                continue
            cnt = Counter(ent[1] for ent in dq if ent[1] >= 0)
            if not cnt:
                continue
            succ, n_agree = cnt.most_common(1)[0]
            if n_agree / len(dq) < min_agree:
                continue
            on_ring = [e for e in dq
                       if floor <= e[0] < self._count]
            if not on_ring:
                continue                     # nothing promotable
            for ent in on_ring:
                # write_counts(ctx_end) keys end at ctx_end-1; the
                # entry's pos IS the last context token -> pos+1
                rep["promoted"] += self.write_counts(
                    int(ent[0]) + 1, {succ: mass}, tag=tag)
            promoted_pos = {e[0] for e in on_ring}
            keep = [e for e in dq if e[0] not in promoted_pos]
            for ent in on_ring:
                self.journal.append(
                    ("eE", key, ent[0], ent[1], ent[2]))
                self._epi_n -= 1
                rep["freed"] += 1
            if keep:
                self._epi[key] = deque(keep,
                                       maxlen=c.epi_max_per_key)
            else:
                del self._epi[key]
                self._epi_hits.pop(key, None)
            self._epi_sync_row(key)
            rep["keys"] += 1
        self.journal.append(("we", tag))
        return rep

    def consolidate_gen(self, mass: float = 1.0, top_m: int = 4,
                        min_tot: float | None = None,
                        tag: str = "gencon",
                        all_ring: bool = False) -> dict:
        """Recall->gen consolidation: replay on-ring stream positions
        and write the order cells' recalled successor DISTRIBUTION
        (top-m followers, share-weighted) into the gen channels -
        gcell cells, the gsyn delta state (soft-feature target), and
        the live _Rl rows.  The gen side distills what the recall
        side already proved, without re-ingesting the source stream:
        the memory contents update the generator directly.

        Contexts reconstruct exactly inside the current episode:
        _c/_c2 reset to zero at ``_ep_start`` and the ring retains
        every episode token, so the closed-form EMA replay
        (_cprev_seq seeded at zero) reproduces each pre-token context.
        When the ring truncated mid-episode (floor > ep_start) the
        missing pre-floor mass is approximate - it decays as
        lam^age, negligible after ~1/(1-lam) positions.
        ``all_ring=True`` drops the episode floor to the ring floor -
        cross-episode replay carries stale context mass across old
        boundaries (the same decaying-approximation class as ring
        truncation); for post-corpus consolidation (train_flux
        --gen-consolidate after per-pack soft_resets) it is the only
        mode that covers more than the last episode.

        Only positions whose context holds order-cell evidence
        (tot > min_tot) transfer - mass is scaled by the cell's
        confidence tot/(tot+kappa).  All writes are journaled inside
        one ws/we block under ``tag``: gsyn via a single ("gs", S,
        Ss) pre-state snapshot (the factor ring would overflow on a
        big sweep), _Rl via the session mechanism ("rB"), gcell via
        per-cell "G" entries.  Revertible via revert_writes(tag) /
        revert_tag(tag)."""
        c = self.config
        dev = self._dev
        cap = c.ring_capacity
        ring_floor = max(0, self._count - cap)
        floor = (ring_floor if all_ring
                 else max(self._ep_start, ring_floor))
        n = self._count - floor
        rep = {"positions": 0, "gcell": 0, "gsyn": 0, "rl": 0,
               "gsyn_deep": 0, "dnet": 0}
        if n < 1 or not (c.gcell or c.gsyn or c.gsyn_deep or c.dnet
                         or c.r_live):
            return rep
        min_tot = c.thin_tot if min_tot is None else min_tot
        # the whole retained range as a closed-form replay, SLICED so
        # big rings don't materialize [n, in_dim] fp64 all at once
        # (897K positions * ~1728 dims * 8B would need ~12GB);
        # context state carries across slices via c_run/c2_run/p_run
        ids_np = np.array(
            [int(self._ring[p % cap]) for p in range(floor,
                                                     self._count)],
            dtype=np.int64)
        ids_t = torch.from_numpy(ids_np).to(dev)
        zero = torch.zeros(c.topic_dim, dtype=torch.float32,
                           device=dev)
        # token before the replay floor is only valid while it is
        # still on the ring (all_ring drops the episode floor)
        p_run = (int(self._ring[(floor - 1) % cap])
                 if floor - 1 >= ring_floor else -1)
        Rm = self._Reff if self._Reff is not None else self._R
        SLICE = 1 << 16
        self.journal.append(("ws", tag))
        self._rl_session = {}                    # one "rB" undo entry
        self._gsd_session = {}                   # one "gd" undo entry
        self._dnet_session = {}                  # one "dn" undo entry
        if c.gsyn:
            # one pre-state snapshot covers the whole gsyn sweep
            self.journal.append(
                ("gs", self._S.detach().cpu().clone(),
                 self._Ss.detach().cpu().clone()
                 if c.gsyn_slow_decay > 0 else None))
        c_run, c2_run = zero, zero.clone()
        try:
            for i0 in range(0, n, SLICE):
                i1 = min(i0 + SLICE, n)
                m_s = i1 - i0
                ids_s = ids_t[i0:i1]
                cprev = self._cprev_seq(ids_s, c0=c_run)   # [m+1, d]
                c2prev = (self._c2prev_seq(ids_s, c20=c2_run,
                                           prev0=p_run)
                          if c.bigram_sketch else None)
                c_run = cprev[m_s].float()
                if c2prev is not None:
                    c2_run = c2prev[m_s].float()
                feats = self._gcell_feats_bulk(
                    cprev[:m_s],
                    None if c2prev is None else c2prev[:m_s],
                    ids_s, 0, floor + i0, ids_s)        # [m, in_dim]
                gcodes = (self._gcell_codes_from_feats(feats)
                          if c.gcell else None)
                gkeys = (self._gsyn_keys(feats)
                         if (c.gsyn or c.gsyn_deep or c.dnet)
                         else None)
                # per-position raw suffix keys for every order, over
                # the host _H ring (ends[i] = floor+i0+i-1; a k-order
                # key is only well-defined while its whole suffix
                # stays on-ring)
                ends = floor + i0 - 1 + np.arange(m_s,
                                                  dtype=np.int64)
                He = self._H[ends % cap].astype(np.uint64)
                votes_l: list[dict[int, float]] = []
                conf = np.zeros(m_s, dtype=np.float64)
                if self._gpu:
                    # batch-probe all (order, position) keys once:
                    # s64 raw keys -> pk -> slot/found -> row gather
                    kk = np.empty((len(c.orders), m_s),
                                  dtype=np.uint64)
                    for oi, k in enumerate(c.orders):
                        He_k = np.where(
                            ends - k >= 0,
                            self._H[(ends - k) % cap].astype(np.uint64),
                            0)
                        # uint64 arithmetic wraps mod 2^64 already
                        kk[oi] = He - He_k * np.uint64(self._pows[k])
                        kk[oi][ends - k + 1 < ring_floor] = 0
                    kk_t = torch.from_numpy(
                        kk.astype(np.int64)).to(dev)          # [O, m]
                    pk_all = _mix64(kk_t * _MIX1
                                    + self._gt._salt[
                                        :len(c.orders), None])
                    rows_f = self._gt._ordx[
                        :len(c.orders), None].expand(
                            len(c.orders), m_s).reshape(-1)
                    slot_f, found_f = self._gt.probe_fixed(
                        pk_all.reshape(-1), rows=rows_f)
                    tot_f = self._gt.ttot[rows_f, slot_f]
                    rt_f = self._gt.ttoks[rows_f, slot_f]
                    rc_f = self._gt.tcnt[rows_f, slot_f]
                    found_h = found_f.reshape(len(c.orders),
                                              m_s).cpu().numpy()
                    tot_h = tot_f.reshape(len(c.orders),
                                          m_s).cpu().numpy()
                    rt_h = rt_f.reshape(len(c.orders), m_s,
                                        -1).cpu().numpy()
                    rc_h = rc_f.reshape(len(c.orders), m_s,
                                        -1).cpu().numpy()
                    for i in range(m_s):
                        v: dict[int, float] = {}
                        for oi in range(len(c.orders)):
                            if (not found_h[oi, i]
                                    or tot_h[oi, i] <= min_tot):
                                continue
                            conf[i] = max(
                                conf[i], tot_h[oi, i]
                                / (tot_h[oi, i] + c.backoff_kappa))
                            for t, ct in zip(rt_h[oi, i],
                                             rc_h[oi, i]):
                                if t >= 0 and ct > 0:
                                    v[int(t)] = v.get(int(t), 0.0) \
                                        + float(ct)
                        votes_l.append(v)
                else:
                    for i in range(m_s):
                        v = {}
                        for k in c.orders:
                            if ends[i] - k + 1 < ring_floor:
                                continue
                            key = (int(He[i])
                                   - int(self._H[(ends[i] - k)
                                                 % cap])
                                   * int(self._pows[k])) % (1 << 64)
                            cell = self._tables[k].get(key)
                            if not cell:
                                continue
                            tot = self._totals[k].get(key, 0.0)
                            if tot <= min_tot:
                                continue
                            conf[i] = max(
                                conf[i],
                                tot / (tot + c.backoff_kappa))
                            for t, ct in cell.items():
                                v[t] = v.get(t, 0.0) + ct
                        votes_l.append(v)
                p_run = int(ids_np[i1 - 1])
                ks, vs, ss = [], [], []
                for i in range(m_s):
                    v = votes_l[i]
                    if not v:
                        continue
                    top = sorted(v.items(),
                                 key=lambda kv: -kv[1])[:top_m]
                    tot_v = sum(w_ for _, w_ in top)
                    if tot_v <= 0:
                        continue
                    s_i = mass * conf[i]
                    if s_i <= 0:
                        continue
                    rep["positions"] += 1
                    if c.gcell:
                        for t, w_ in top:
                            self._gcell_write(gcodes[i], t,
                                              s_i * w_ / tot_v)
                            rep["gcell"] += 1
                    if c.gsyn or c.gsyn_deep or c.dnet:
                        vt = torch.stack(
                            [Rm[t].float() for t, w_ in top])
                        wv = torch.tensor(
                            [w_ / tot_v for _, w_ in top],
                            dtype=torch.float32, device=dev)
                        ks.append(gkeys[i])
                        vs.append((vt * wv[:, None]).sum(0))
                        ss.append(s_i)
                        if c.gsyn:
                            rep["gsyn"] += 1
                        if c.gsyn_deep:
                            rep["gsyn_deep"] += 1
                        if c.dnet:
                            rep["dnet"] += 1
                    if c.r_live:
                        cv = cprev[i].float()
                        for t, w_ in top:
                            self._rl_write(t, cv, s_i * w_ / tot_v)
                            rep["rl"] += 1
                if ks:
                    kk_t = torch.stack(ks)
                    ss_t = torch.tensor(ss, dtype=torch.float32,
                                        device=dev)
                    vv_t = torch.stack(vs)
                    if c.gsyn:
                        self._gsyn_chunk(
                            kk_t,
                            torch.zeros(len(ks), dtype=torch.long,
                                        device=dev),
                            ss_t, v=vv_t)
                    if c.gsyn_deep:
                        self._gsd_chunk(kk_t, None, ss_t, v=vv_t)
                    if c.dnet:
                        self._dnet_chunk(kk_t, None, ss_t, v=vv_t)
        finally:
            self._rl_flush()
            self._gsd_flush()
            self._dnet_flush()
        self.journal.append(("we", tag))
        return rep

    # -- teacher graft -------------------------------------------------

    def graft_teacher(self, emb, method: str | None = None) -> dict:
        """Architectural graft: replace the random bipolar feature
        matrix ``_R`` with a projection of a pretrained teacher's
        embedding table (e.g. ForgeLM V2 ``d_embed.weight``).

        Rationale: ``_R[tok]`` is the sole feature source for the
        context sketch ``_c``, the Hebbian topic matrix ``_A``, proto,
        csem rows and knn fingerprints - so semantic embeddings upgrade
        every channel at once (contexts become a bag-of-meanings, knn
        retrieves *semantically* similar contexts, not just token-
        overlap ones).  Still an associative memory - the graft gives
        good features, not compositionality.

        ``emb`` = [V_t, d] tensor or a safetensors/.pt path (the
        [vocab, dim] tensor is auto-detected).  Rows are unit-normed
        (embedding norm encodes frequency, not meaning), projected to
        ``topic_dim`` via PCA (dominant semantic axes) or a seeded
        random projection (JL - preserves pairwise cosines), then
        rescaled to norm sqrt(topic_dim) so Hebbian/delta-rule rates
        calibrated on +-1 bipolar rows stay valid.

        Must run before any learning - learned state is tied to the
        feature space it was written in."""
        c = self.config
        if self._count or self._epi_n or self._knn_cur:
            raise RuntimeError(
                "graft before learning - existing memory cells were "
                "written under the old feature space")
        method = method or c.teacher_embed_method
        E = (_load_teacher_embedding(emb)
             if isinstance(emb, (str, Path)) else emb.float())
        if E.dim() != 2:
            raise ValueError(f"teacher embedding must be 2-D, got "
                             f"{tuple(E.shape)}")
        if E.shape[0] < self.vocab_size:
            raise ValueError(
                f"teacher vocab {E.shape[0]} < flux vocab "
                f"{self.vocab_size} - cannot fabricate semantics")
        E = E[: self.vocab_size]
        En = E / E.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # drop the shared 'commonness' direction: embedding means encode
        # frequency/register, and subtracting it collapses the random-
        # pair cosine baseline to ~0 while semantic pairs keep their
        # signal (measured: cat-dog 0.155 vs 0.002 baseline on V2)
        En = En - En.mean(0, keepdim=True)
        # project on GPU when available (65536x2560 SVD is heavy on CPU)
        pca_dev = self._dev if self._dev.type == "cuda" else E.device
        En = En.to(pca_dev)
        if method == "pca":
            _, _, Vh = torch.pca_lowrank(En, q=c.topic_dim)
            R = En @ Vh                             # [V, topic_dim]
        elif method == "randproj":
            g = torch.Generator(device=En.device).manual_seed(
                c.seed ^ 0x6A7A)
            W = torch.randn(E.shape[1], c.topic_dim, generator=g,
                            device=En.device) / math.sqrt(E.shape[1])
            R = En @ W
        else:
            raise ValueError(f"unknown graft method {method!r}")
        R = R / R.norm(dim=1, keepdim=True).clamp(min=1e-8) \
            * math.sqrt(c.topic_dim)
        self._R = R.float().to(self._dev)
        if self._Rl is not None:
            # graft rebases the feature space - rebuild the blended
            # readout matrix over the new rows (Rl is zero anyway:
            # graft is only legal before learning)
            self._Reff = self._R.float() + self._Rl
        self._graft = {"src": str(emb), "method": method,
                       "teacher_dim": int(E.shape[1]),
                       "topic_dim": int(c.topic_dim)}
        self.journal.append(("g", dict(self._graft)))
        return self._graft

    def distill_from(self, score_fn, top_k: int = 5, scale: float = 2.0,
                     max_gaps: int | None = None, batch_size: int = 64,
                     ctx_len: int = 32, tag: str = "distill",
                     families: tuple[str, ...] = ("orders",),
                     progress=None) -> dict:
        """Selective teacher distillation.

        For each gap position, build its ``ctx_len``-token suffix window,
        ask ``score_fn`` for the teacher's top-k next-token
        probabilities, and write them as fractional counts into this
        model's order tables (journaled under ``tag``).

        ``score_fn(contexts: list[list[int]]) -> (top_ids[B,k],
        top_probs[B,k])`` - e.g. a ForgeLM/HF forward returning logits.

        ``families`` picks which key families receive teacher mass -
        "orders" (raw suffix keys: fires only when the context recurs
        verbatim) plus any of "class"/"sparse"/"hybrid"/"word" (coarse
        keys that recur on novel text - the path that actually
        generalizes teacher knowledge)."""
        import torch as _t
        gaps = self.gap_positions()
        if not gaps:
            n = self.rescan_gaps()
            if n:
                gaps = self.gap_positions()
        if max_gaps is not None:
            gaps = gaps[:max_gaps]
        written = scored = 0
        t0 = time.perf_counter()
        for i in range(0, len(gaps), batch_size):
            batch = gaps[i:i + batch_size]
            ctxs = [self.window_tokens(p, ctx_len) for p in batch]
            # left-pad short windows to ctx_len (pad with token 0)
            padded = [[0] * (ctx_len - len(c)) + c for c in ctxs]
            top_ids, top_probs = score_fn(padded)
            top_ids = (top_ids.tolist() if isinstance(top_ids, _t.Tensor)
                       else top_ids)
            top_probs = (top_probs.tolist()
                         if isinstance(top_probs, _t.Tensor)
                         else top_probs)
            for pos, ids_row, probs_row in zip(batch, top_ids, top_probs):
                deltas = {int(t): float(p) * scale
                          for t, p in zip(ids_row, probs_row)}
                written += self.write_counts(pos, deltas, tag=tag,
                                             families=families)
            scored += len(batch)
            if progress:
                progress(scored, len(gaps))
        return {"gaps_scored": scored, "cells_written": written,
                "seconds": time.perf_counter() - t0}

    def distill_hebbian(self, score_fn, top_k: int = 5,
                        eta: float = 1.0,
                        max_gaps: int | None = None, batch_size: int = 64,
                        ctx_len: int = 32, tag: str = "distill",
                        progress=None) -> dict:
        """Parametric teacher distillation through the model's own
        learning rule ("imagined experience").

        Table distillation (``distill_from``) stores teacher judgments
        under suffix keys - it fires only where contexts recur.  This
        variant instead writes teacher top-k probabilities into ``_A``
        via the same delta rule live learning uses:

            delta_t = chat * (p_teacher - A[t].chat) * hebb_lr * eta

        so the topic/sem/csem/knn channels - which read ``_A`` by
        fingerprint cosine - generalize teacher knowledge to contexts
        that were never seen verbatim.  This is the mechanism that can
        make an associative memory behave like the teacher on novel
        input, not just recall its answers.

        Journaled: each touched ``_A`` row's prior state is snapshotted
        once ("Ab" entries) inside a ``ws``/``we`` block - fully
        revertible via ``revert_tag(tag)``."""
        c = self.config
        gaps = self.gap_positions()
        if not gaps:
            n = self.rescan_gaps()
            if n:
                gaps = self.gap_positions()
        if max_gaps is not None:
            gaps = gaps[:max_gaps]
        written = scored = 0
        seen_rows: set[int] = set()
        self.journal.append(("ws", tag))
        t0 = time.perf_counter()
        for i in range(0, len(gaps), batch_size):
            batch = gaps[i:i + batch_size]
            ctxs = [self.window_tokens(p, ctx_len) for p in batch]
            padded = [[0] * (ctx_len - len(cc)) + cc for cc in ctxs]
            top_ids, top_probs = score_fn(padded)
            top_ids = top_ids.tolist()
            top_probs = top_probs.tolist()
            for ctx, ids_row, probs_row in zip(ctxs, top_ids, top_probs):
                cv = self._context_vector(ctx)
                cn = float(cv.norm())
                if cn < 1e-8:
                    continue
                chat = cv / cn
                for t, p in zip(ids_row, probs_row):
                    t = int(t)
                    p = float(p) * eta
                    if t not in seen_rows:
                        seen_rows.add(t)
                        self.journal.append(
                            ("Ab", t,
                             self._A[t].detach().cpu().numpy()
                             .astype(np.float16)))
                    a_t = self._A[t]
                    pred = float(a_t @ chat)
                    delta = chat * ((p - pred) * c.hebb_lr)
                    self._A[t] += delta
                    if self._gpu:
                        self._An[t] = self._A[t].norm()
                        self._Ac[t] += delta @ self._Pcsem
                        self._Acn[t] = self._Ac[t].norm()
                    written += 1
                scored += 1
            if progress:
                progress(scored, len(gaps))
        self.journal.append(("we", tag))
        return {"gaps_scored": scored, "hebb_writes": written,
                "rows_touched": len(seen_rows),
                "seconds": time.perf_counter() - t0}

    def distill_head(self, score_fn, top_k: int = 8,
                     ctx_len: int = 32, epochs: int = 2,
                     lr: float = 3e-3, batch_size: int = 64,
                     train_batch: int = 256,
                     max_gaps: int | None = None,
                     scope: str = "gaps",
                     ids: list[int] | None = None,
                     tag: str = "distill", progress=None) -> dict:
        """Parametric distillation: train a linear readout head
        ``Wh [V, topic_dim]`` mapping context fingerprints to the
        teacher's next-token distribution at gap positions.

        Unlike table/cell distillation (lookup - fires only on repeated
        or class-equal contexts) and Hebbian ``_A`` writes (one-step,
        non-contrastive), this learns a *function*: cosine-similar
        contexts get interpolated teacher predictions.  The head is a
        new additive channel read in ``_predict`` with fixed weight
        ``config.head_w`` - it bypasses the hedge floor like
        ``thin_boost`` does, since it exists precisely to answer where
        memory is thin.

        ``scope="gaps"`` trains only at thin-context positions (the
        head's intended deployment niche); ``scope="all"`` trains at
        every ring-retained position - a general context->token map,
        much more data.

        Journaled as one ``"WH"`` blob (pre-train ``Wh``, fp16) inside
        a ``ws``/``we`` block - ``revert_tag`` restores it."""
        c = self.config
        if scope == "all" and ids is not None:
            # full corpus, no ring horizon - fingerprint for position p
            # is the topic-EMA over ids[:p] (context predicts ids[p]).
            gaps = list(range(1, len(ids)))
            if max_gaps is not None and len(gaps) > max_gaps:
                idx = torch.randperm(len(gaps))[:max_gaps].tolist()
                gaps = sorted(gaps[j] for j in idx)
        elif scope == "all":
            start = max(0, self._count - c.ring_capacity)
            gaps = list(range(start + 1, self._count + 1))
            if max_gaps is not None and len(gaps) > max_gaps:
                idx = torch.randperm(len(gaps))[:max_gaps].tolist()
                gaps = sorted(gaps[j] for j in idx)
        else:
            gaps = self.gap_positions()
            if not gaps:
                n = self.rescan_gaps()
                if n:
                    gaps = self.gap_positions()
            if max_gaps is not None:
                gaps = gaps[:max_gaps]
        dev = self._dev
        V = self.vocab_size

        # collect (order-sensitive head feature, teacher top-k) pairs
        xs, ys_ids, ys_p = [], [], []
        scored = 0
        for i in range(0, len(gaps), batch_size):
            batch = gaps[i:i + batch_size]
            if ids is not None:
                ctxs = [ids[max(0, p - ctx_len):p] for p in batch]
            else:
                ctxs = [self.window_tokens(p, ctx_len)
                        for p in batch]
            padded = [[0] * (ctx_len - len(cc)) + cc for cc in ctxs]
            top_ids, top_probs = score_fn(padded)
            top_ids = top_ids.tolist()
            top_probs = top_probs.tolist()
            fx = self._head_feats(ctxs)                 # [B, in_dim]
            keep = fx.norm(dim=1) > 1e-8
            for j, keep_j in enumerate(keep.tolist()):
                if not keep_j:
                    continue
                xs.append(fx[j])
                ys_ids.append(top_ids[j])
                ys_p.append(top_probs[j])
                scored += 1
            if progress:
                progress(scored, len(gaps))
        if not xs:
            return {"gaps_scored": 0, "head_writes": 0, "seconds": 0.0}

        X = torch.stack(xs)                              # [N, in_dim]
        N = X.shape[0]
        in_dim = self._head_in_dim()
        H = c.head_hidden
        if H > 0:
            # MLP channel: logits = Wh @ relu(Wh1 @ x)
            W1 = (self._Wh1.clone() if (
                    self._Wh1 is not None
                    and tuple(self._Wh1.shape) == (H, in_dim))
                  else torch.randn(H, in_dim, device=dev)
                       / (in_dim ** 0.5))
            W = (self._Wh.clone() if (
                    self._Wh is not None
                    and self._Wh.shape == (V, H))
                 else torch.zeros(V, H, device=dev))
            params = [W1, W]
        else:
            W1 = None
            W = (self._Wh.clone() if (
                    self._Wh is not None
                    and self._Wh.shape == (V, in_dim))
                 else torch.zeros(V, in_dim, device=dev))
            params = [W]
        for p in params:
            p.requires_grad_(True)
        opt = torch.optim.AdamW(params, lr=lr)
        self.journal.append(("ws", tag))

        def _blob(t):
            return (None if t is None else
                    t.detach().cpu().numpy().astype(np.float16)
                    .tobytes())
        self.journal.append(
            ("WH", _blob(self._Wh), _blob(self._Wh1)))
        self.journal.append(("we", tag))

        t0 = time.perf_counter()
        for _ep in range(epochs):
            perm = torch.randperm(N)
            for i in range(0, N, train_batch):
                bidx = perm[i:i + train_batch]
                xb = X[bidx]                             # [b, d]
                if W1 is not None:
                    xb = (xb @ W1.T).clamp(min=0)
                lg = xb @ W.T                            # [b, V]
                # target: teacher top-k probs + uniform remainder
                tgt = torch.full((len(bidx), V), 0.0, device=dev)
                ti = torch.tensor(
                    [ys_ids[j] for j in bidx.tolist()],
                    dtype=torch.long, device=dev)
                tp = torch.tensor(
                    [ys_p[j] for j in bidx.tolist()],
                    dtype=torch.float32, device=dev)
                tgt.scatter_(1, ti, tp)
                rem = (1.0 - tgt.sum(1)).clamp(min=0.0)
                tgt += rem[:, None] / (V - top_k)
                loss = -(tgt * torch.log_softmax(lg, -1)).sum(1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._Wh = W.detach()
        self._Wh1 = (None if W1 is None else W1.detach())
        # captured CUDA graphs baked the previous _Wh (or its absence)
        # - drop them so the next predict recaptures the head branch
        self._graphs = {}
        self._pgraph = None
        return {"gaps_scored": scored, "head_writes": N * epochs,
                "final_loss": float(loss.detach()),
                "seconds": time.perf_counter() - t0}

    def fit_text(self, text: str, tokenizer, tag: str = "text") -> int:
        enc = tokenizer(text, add_special_tokens=False)
        ids = getattr(enc, "input_ids", None)
        if ids is None:
            ids = getattr(enc, "ids", enc)
        return self.ingest(list(ids), tag=tag)

    def soft_reset(self) -> None:
        """New episode: reset context sketch, keep all learned memory."""
        self.journal.append(("b", self._ep_start,
                             self._c.detach().cpu().numpy()
                             .astype(np.float16),
                             self._c2.detach().cpu().numpy()
                             .astype(np.float16)))
        self._ep_start = self._count
        self._c.zero_()
        self._c2.zero_()
        self._last_seen.clear()
        self._recent.clear()
        self._regime = 0               # fresh episode starts thin
        if self._gpu:
            self._seen_g.fill_(-10 ** 9)
            self._ep_g.fill_(self._count)
            self._regime_g.fill_(0)
            self._reg_used_g.fill_(0)

