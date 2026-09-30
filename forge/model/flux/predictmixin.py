"""Prediction paths for FluxLM (CPU + CUDA-primary)."""
from __future__ import annotations

import json
import math
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .helpers import (_HASH_P, _MASK64, _MIX1, _MIX2,
                      _load_teacher_embedding, _mix64, _s64,
                      _scatter_sum)
from .config import FluxConfig
from .journal import FluxJournal
from .tables import _GPUCellTables
from .kernels import HAS_TRITON

if HAS_TRITON:
    import triton
    from .kernels import (_flux_deep_votes_kernel,
                          _flux_logit_tail_kernel)


class _PredictMixin:

    def _channel_prob(self, ch: str, x: int, keys: dict[int, int],
                      votes: dict[int, float] | None,
                      denom: float,
                      kvotes: dict[int, float] | None = None,
                      pkeys: dict[int, int] | None = None,
                      gkeys: dict[int, int] | None = None,
                      wkeys: dict[int, int] | None = None,
                      hkeys: dict[int, int] | None = None
                      ) -> tuple[float, bool]:
        """Channel c's predicted probability of token x (for Hedge) plus
        an awake flag: False = the channel had no evidence and fell back
        to the unigram prior (sleeping-experts abstention)."""
        c = self.config
        if ch == "uni":
            return self._uni_p(x, denom), True
        if ch == "knn":
            if not kvotes:
                return self._uni_p(x, denom), False
            vt = sum(kvotes.values())
            return (kvotes.get(x, 0.0) + c.alpha
                    * self._uni_p(x, denom)) / (vt + c.alpha), True
        if ch == "recency":
            age = self._count - 1 - self._last_seen.get(x, -10 ** 9)
            rec = c.recency_decay ** age \
                if 0 <= age < c.recency_span else 0.0
            if c.calib_rewards:
                # implied dist p(v) ∝ p_uni(v)·exp(rec(v)); Z = 1 +
                # Σ_{recent} p_uni·(exp(rec)-1) over latest-age tokens
                z = 1.0 + self._recent_lift_z(c.recency_span,
                                              c.recency_decay, +1.0)
                return self._uni_p(x, denom) * math.exp(rec) / z, True
            return (rec if rec > 0 else 1e-3), True
        if ch == "fatigue":
            # rewards novelty: ~1 when x wasn't just seen, ~0 when repeated
            age = self._count - 1 - self._last_seen.get(x, -10 ** 9)
            fat = c.fatigue_decay ** age \
                if 0 <= age < c.fatigue_span else 0.0
            if c.calib_rewards:
                # implied dist p(v) ∝ p_uni(v)·exp(-str·fat(v))
                z = 1.0 + self._recent_lift_z(c.fatigue_span,
                                              c.fatigue_decay,
                                              -c.fatigue_strength)
                return self._uni_p(x, denom) \
                    * math.exp(-c.fatigue_strength * fat) / z, True
            return 1.0 - fat, True
        if ch == "topic":
            a = self._A[x]
            cn = float(self._c.norm())
            an = float(a.norm())
            cos = float(a @ self._c) / (an * cn + 1e-8)
            awake = cn > 1e-8 and an > 1e-8
            if c.calib_rewards and awake:
                # p(v) ∝ p_uni(v)·exp(β·cos_v); Z ≈ 1 under 0-mean
                # random-feature cosines (E[exp] ≈ 1 + β²σ²/2 ~ 1.04)
                return min(1.0, self._uni_p(x, denom)
                           * math.exp(c.topic_beta * cos)), awake
            return 1.0 / (1.0 + math.exp(-c.topic_beta * cos)), awake
        if ch == "sem":
            a = self._A[x]
            pn = float(self._proto.norm())
            an = float(a.norm())
            cos = float(a @ self._proto) / (an * pn + 1e-8)
            awake = pn > 1e-8 and an > 1e-8
            if c.calib_rewards and awake:
                return min(1.0, self._uni_p(x, denom)
                           * math.exp(c.sem_beta * cos)), awake
            return 1.0 / (1.0 + math.exp(-c.sem_beta * cos)), awake
        if ch == "csem":
            # contextual fingerprint prob: how similar is x's feature to
            # what THIS context has historically preceded
            row = self._csem_read()
            if row is None:
                return self._uni_p(x, denom), False
            ax = self._A[x] @ self._Pcsem
            rn = float(row.norm())
            an = float(ax.norm())
            cos = float(ax @ row) / (an * rn + 1e-8)
            awake = rn > 1e-8 and an > 1e-8
            if c.calib_rewards and awake:
                return min(1.0, self._uni_p(x, denom)
                           * math.exp(c.csem_beta * cos)), awake
            return 1.0 / (1.0 + math.exp(-c.csem_beta * cos)), awake
        if ch == "deep":
            if not votes:
                return self._uni_p(x, denom), False
            vt = sum(votes.values())
            return (votes.get(x, 0.0) + c.alpha * self._uni_p(x, denom)) \
                / (vt + c.alpha), True
        if ch == "backoff":
            # longest order with enough evidence, confidence-discounted
            best_k, btot = -1, 0.0
            for k, key in keys.items():
                t = self._totals[k].get(key, 0.0)
                if t >= c.backoff_min_tot and k > best_k:
                    best_k, btot = k, t
            if best_k < 0:
                return self._uni_p(x, denom), False
            cell = self._tables[best_k].get(keys[best_k]) or {}
            conf = btot / (btot + c.backoff_kappa)
            return conf * (cell.get(x, 0.0)
                           + c.alpha * self._uni_p(x, denom)) \
                / (btot + c.alpha), True
        if ch.startswith("p"):
            # sparse-tap channel "p<j>" - gapped-context cell
            j = int(ch[1:])
            taps = c.sparse_taps[j]
            key = (pkeys or {}).get(j, -1)
            cell = self._ptabs[taps].get(key)
            if not cell:
                return self._uni_p(x, denom), False
            tot = self._ptots[taps].get(key, 0.0)
            return (cell.get(x, 0.0) + c.alpha
                    * self._uni_p(x, denom)) / (tot + c.alpha), tot > 0
        if ch.startswith("g"):
            # class-order channel "g<k>" - contiguous suffix over byte
            # classes (coarse context - generalizes on novel text)
            k = int(ch[1:])
            key = (gkeys or {}).get(k, -1)
            cell = self._gtabs[k].get(key)
            if not cell:
                return self._uni_p(x, denom), False
            tot = self._gtots[k].get(key, 0.0)
            return (cell.get(x, 0.0) + c.alpha
                    * self._uni_p(x, denom)) / (tot + c.alpha), tot > 0
        if ch.startswith("w"):
            # word-order channel "w<j>" - key over last-w word hashes +
            # partial-word bucket (ring-scan derived, no state)
            j = int(ch[1:])
            key = (wkeys or {}).get(j, -1)
            cell = self._wtabs[j].get(key)
            if not cell:
                return self._uni_p(x, denom), False
            tot = self._wtots[j].get(key, 0.0)
            return (cell.get(x, 0.0) + c.alpha
                    * self._uni_p(x, denom)) / (tot + c.alpha), tot > 0
        if ch.startswith("h"):
            # hybrid channel "h<j>" - key over a (offset, is_raw)
            # pattern mixing byte classes and raw bytes
            j = int(ch[1:])
            key = (hkeys or {}).get(j, -1)
            cell = self._htabs[j].get(key)
            if not cell:
                return self._uni_p(x, denom), False
            tot = self._htots[j].get(key, 0.0)
            return (cell.get(x, 0.0) + c.alpha
                    * self._uni_p(x, denom)) / (tot + c.alpha), tot > 0
        # order channel "o<k>"
        k = int(ch[1:])
        key = keys.get(k, -1)
        cell = self._tables[k].get(key)
        if not cell:
            return self._uni_p(x, denom), False
        tot = self._totals[k].get(key, 0.0)
        return (cell.get(x, 0.0) + c.alpha * self._uni_p(x, denom)) \
            / (tot + c.alpha), tot > 0

    def _deep_votes(self, end: int) -> dict[int, float]:
        """Seed-match + backward extension -> successor votes.

        Entries are (ctx_end, succ, certs) tuples.  Positions still in
        the ring get exact backward extension; positions that scrolled
        out (``pos < floor``) vote by certificate: weight = the deepest
        cert order matching the current context's suffix hash, else the
        bare seed order.  The index key IS the validation for archived
        entries (2^-64 collision) - deep memory is unbounded."""
        c = self.config
        key = self._suffix_key(c.deep_order, end)
        if key is None:
            return {}
        positions = self._epi.get(key)
        if not positions:
            return {}
        cap = c.ring_capacity
        floor = max(0, self._count - cap)
        cur_certs = {L: self._suffix_key(L, end)
                     for L in c.cert_orders} if c.cert_orders else {}
        votes: dict[int, float] = {}
        examined = 0
        for ent in reversed(positions):
            pos, succ = ent[0], ent[1]
            if pos >= end:
                continue              # context end not yet writable
            if pos >= floor:
                # on-ring: recompute seed hash (ring bytes may have
                # been overwritten mid-wrap) then extend backwards
                if self._raw_hash(c.deep_order, pos) != key:
                    continue
                m = c.deep_order
                ext = 0
                while (ext < c.deep_max_extend and pos - m >= floor
                       and end - m >= self._ep_start
                       and self._ring[(pos - m) % cap]
                       == self._ring[(end - m) % cap]):
                    ext += 1
                    m += 1
                w = float(m)
            else:
                # archived: certificate-graded vote
                w = float(c.deep_order)
                certs = ent[2] if len(ent) > 2 else ()
                for L, cv in zip(c.cert_orders, certs):
                    if cv and cur_certs.get(L) == cv and L > w:
                        w = float(L)
            if succ < 0:              # legacy entry - read ring
                if pos + 1 >= self._count or pos + 1 < floor:
                    continue
                succ = int(self._ring[(pos + 1) % cap])
            votes[succ] = votes.get(succ, 0.0) + w
            examined += 1
            if examined >= c.deep_max_hits:
                break
        if examined:
            # retrieval hit - feeds utility-scored eviction: cells that
            # actually get read are worth keeping
            self._epi_hits[key] = self._epi_hits.get(key, 0) + examined
        return votes

    def _knn_votes(self, k: int | None = None) -> dict[int, float]:
        """Top-k fingerprint-similar contexts -> successor votes.

        One bank matvec + topk; weights = relu(cos)^gamma, floored at
        ``knn_min_sim``.  Returns {} when the bank is empty."""
        c = self.config
        if not c.use_knn or self._knn_cur == 0:
            return {}
        q = self._knn_fp_of(self._c).to(self._knn_fp.dtype)
        # scan only the live ring prefix (bucketed) - the padded tail
        # is pos=-1 anyway, so slicing is exact, just cheaper
        b = self._knn_rows()
        if b <= 0:
            return {}
        # matvec in bank dtype ([b,d]@[d] - fp16 halves the read cost),
        # then widen the [b] result for stable top-k
        sims = (self._knn_fp[:b] @ q).float()
        sims = torch.where(self._knn_pos[:b] >= 0, sims,
                           torch.full((), -1e9,
                                      device=self._dev))
        kk = int(min(k or c.knn_k, max(self._knn_cur, 1),
                     c.knn_capacity))
        if kk <= 0:
            return {}
        top = torch.topk(sims, kk)
        votes: dict[int, float] = {}
        hit_rows: list[int] = []
        for s, i in zip(top.values.tolist(), top.indices.tolist()):
            if s <= c.knn_min_sim:
                break
            hit_rows.append(i)
            t = int(self._knn_succ[i])
            if t >= 0:
                votes[t] = votes.get(t, 0.0) + max(s, 0.0) \
                    ** c.knn_gamma
        if hit_rows:
            # retrieval utility (feeds knn_utility_evict)
            self._knn_hits.index_add_(
                0, torch.tensor(hit_rows, dtype=torch.int64,
                                device=self._dev),
                torch.ones(len(hit_rows), dtype=torch.int32,
                           device=self._dev))
        return votes

    # -- csem helpers ----------------------------------------------------

    @torch._dynamo.disable
    def _deep_votes_dev(self, end: torch.Tensor, floor: torch.Tensor,
                        se: torch.Tensor | None = None,
                        fe: torch.Tensor | None = None,
                        dk: torch.Tensor | None = None
                        ) -> torch.Tensor:
        """Deep episodic votes [V] - tensor-only, graph-safe.  ``se``/
        ``fe``/``dk`` may be precomputed by a combined probe."""
        c = self.config
        cap = c.ring_capacity
        if dk is None:
            dk = self._suffix_dev(c.deep_order, end).squeeze(0)
        if se is None:
            pk = self._epi_hash(dk.reshape(1))
            se, fe = self._epi_probe(pk)
            se = se.reshape(1)
        pos_row = self._epi_pos.index_select(0, se).squeeze(0)  # [ER]
        if self._use_triton:
            # one launch evaluates all ER cells - seed check, backward
            # extension, cert grading.  Per-cell mass is staged, then
            # scattered deterministically (no atomic_add ordering).
            mlen_buf = torch.zeros(c.epi_row, dtype=torch.float32,
                                   device=self._dev)
            LC = len(c.cert_orders)
            succ_r = self._epi_succ.index_select(0, se).squeeze(0)
            if LC:
                certs = self._epi_cert.index_select(
                    0, se).squeeze(0)                        # [ER, LC]
                cur = torch.stack(
                    [self._suffix_dev(L, end).squeeze(0)
                     for L in c.cert_orders])                # [LC]
                cw = self._cert_g
            else:
                certs = torch.zeros((c.epi_row, 1),
                                    dtype=torch.int64, device=self._dev)
                cur = torch.zeros(1, dtype=torch.int64,
                                  device=self._dev)
                cw = cur
            _flux_deep_votes_kernel[(c.epi_row,)](
                pos_row, succ_r, certs, self._Hg, self._ringg,
                cur, cw, mlen_buf,
                end, floor, self._ep_g, dk, self._pow_deep, fe,
                CAP=cap, DO=c.deep_order, MX=c.deep_max_extend,
                LC=max(LC, 1), ER=c.epi_row)
            return _scatter_sum(
                succ_r.long().clamp(0, self.vocab_size - 1),
                mlen_buf, self.vocab_size)
        valid = (end - c.deep_order + 1 >= self._ep_g) \
            & (end >= c.deep_order - 1) & (end >= 0)
        # fe matters: on a probe miss ``se`` still points at a slot
        # owned by an unrelated key - its cells must not vote
        live = fe & (pos_row >= 0) & (pos_row < end) & valid
        on_ring = live & (pos_row >= floor)
        # validate seed hash only where ring bytes still exist -
        # overwritten slots mismatch and drop out
        hv = self._Hg[pos_row.clamp(min=0) % cap] \
            - self._Hg[(pos_row - c.deep_order)
                       .clamp(min=0) % cap] \
            * self._pows_g[c.deep_order]
        on_ring &= hv == dk
        # backward extension as ONE vectorized op: eq[m,j] = token at
        # pos_j-m equals token at end-m; cumprod over m = running AND
        # -> sum gives the contiguous match length per candidate.
        mx = c.deep_max_extend
        ms = torch.arange(1, mx + 1, dtype=torch.int64,
                          device=self._dev)              # [mx]
        pi = pos_row[None, :] - ms[:, None]              # [mx, ER]
        ei = end - ms                                    # [mx]
        ok = (pi >= floor) & (pi >= 0) \
            & (ei >= self._ep_g)[:, None] & (ei >= 0)[:, None]
        a = self._ringg[pi.clamp(min=0) % cap]
        b = self._ringg.index_select(
            0, (ei.clamp(min=0) % cap).reshape(-1))      # [mx]
        eq = (a.long() == b[:, None].long()) & ok
        runlen = eq.long().cumprod(0).sum(0)             # [ER]
        mlen_ring = torch.where(on_ring,
                                runlen + c.deep_order,
                                torch.zeros_like(runlen))
        # archived entries (pos < floor): certificate-graded weight -
        # stored suffix hashes at cert_orders vs the current context's;
        # the index key match alone guarantees at least seed-order mass
        arch = live & (pos_row < floor)
        base_w = torch.full((), c.deep_order, dtype=torch.int64,
                            device=self._dev)
        mlen_arch = torch.where(arch, base_w.expand_as(runlen),
                                torch.zeros_like(runlen))
        if c.cert_orders:
            certs = self._epi_cert.index_select(0, se)   # [1, ER, L]
            cur = torch.stack([self._suffix_dev(L, end).squeeze(0)
                               for L in c.cert_orders])  # [L]
            cmatch = (certs == cur[None, None, :]) & (certs != 0)
            cw = (cmatch.long()
                  * self._cert_g[None, None, :]).amax(dim=2).squeeze(0)
            mlen_arch = torch.where(
                arch, torch.maximum(cw, base_w),
                torch.zeros_like(runlen))
        # successor comes from the stored entry - never the ring - so
        # votes survive ring wrap (and empty succ=-1 is masked out)
        succ = self._epi_succ.index_select(0, se).squeeze(0) \
            .long().clamp(0, self.vocab_size - 1)
        have_succ = self._epi_succ.index_select(0, se) \
            .squeeze(0) >= 0
        mlen = torch.where(have_succ,
                           mlen_ring + mlen_arch,
                           torch.zeros_like(runlen))
        # deterministic accumulation - epi rows can share a successor
        return _scatter_sum(succ, mlen.float(), self.vocab_size)

    @torch._dynamo.disable
    def _topk_dev(self, x: torch.Tensor, k: int) -> torch.Tensor:
        """Eager topk - dynamo-disabled because inductor's topk picks
        different tie members than the eager kernel (identical knn
        fingerprints / tied logits are the norm), which changed the
        selected rows and thus the logits under compile."""
        return torch.topk(x, k).indices

    @torch._dynamo.disable
    def _knn_votes_dev(self, kb: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k fingerprint-similar past contexts vote their stored
        successors - approximate-match episodic memory.  Only the live
        ring prefix is scanned (bucketed - static shape inside a
        captured graph; unwritten padding stays pos=-1).
        dynamo-disabled for the same topk-tie reason as _topk_dev."""
        c = self.config
        V = self.vocab_size
        qk = self._c @ self._Pcsem
        qkn = qk.norm()
        qk = (qk / (qkn + 1e-8)).to(self._knn_fp.dtype)
        ksims = (self._knn_fp[:kb] @ qk).float()
        ksims = torch.where(
            self._knn_pos[:kb] >= 0, ksims,
            torch.full((), -1e9, device=self._dev))
        top = torch.topk(ksims, min(c.knn_k, kb))
        keep = top.values > c.knn_min_sim
        # retrieval utility: neighbours that cleared the vote floor
        # earn hits (feeds knn_utility_evict scoring) - branchless,
        # index_add_ of keep-as-int is capture-safe inside graphs
        self._knn_hits.index_add_(
            0, top.indices, keep.to(torch.int32))
        wv = (top.values.clamp(min=0.0) ** c.knn_gamma) * keep
        ksucc = self._knn_succ.index_select(
            0, top.indices).long().clamp(0, V - 1)
        # deterministic accumulation - neighbours can share a
        # successor, and index_add_ atomics made logits
        # timing-dependent
        kv = _scatter_sum(ksucc, wv, V)
        return kv, kv.sum()

    @torch._dynamo.disable
    def _knn_reward_dev(self, kb: int, fp_store: torch.Tensor,
                        x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Step-side knn reward scan: returns (kvt, kvx) fp64 scalars -
        total neighbour weight and the weight voting for token x.
        dynamo-disabled: same topk-tie hazard as _knn_votes_dev - a
        compiled topk would pick different tied neighbours and diverge
        the hedge rewards permanently."""
        c = self.config
        sims = (self._knn_fp[:kb] @ fp_store).float()
        sims = torch.where(
            self._knn_pos[:kb] >= 0, sims,
            torch.full((), -1e9, device=self._dev))
        ktop = torch.topk(sims, min(c.knn_k, kb))
        kkeep = ktop.values > c.knn_min_sim
        kwv = (ktop.values.clamp(min=0.0) ** c.knn_gamma) * kkeep
        ksucc = self._knn_succ.index_select(
            0, ktop.indices).long().clamp(0, self.vocab_size - 1)
        kvt = kwv.sum().double()
        kvx = kwv.mul(ksucc == x).sum().double()
        return kvt, kvx

    @torch._dynamo.disable
    def _logit_tail_dev(self, logits: torch.Tensor,
                        log_uni: torch.Tensor, uni_pv: torch.Tensor,
                        sims: torch.Tensor, csims: torch.Tensor,
                        kv: torch.Tensor, kvt: torch.Tensor,
                        cf: torch.Tensor, cok: torch.Tensor,
                        rn: torch.Tensor, pvn: torch.Tensor,
                        end: torch.Tensor,
                        wg: torch.Tensor) -> None:
        """Fused V-map: sem + csem + knn mixture + recency + fatigue.

        dynamo-disabled: inductor re-lowering of the custom triton
        kernel inside a compiled region produced wrong logits
        (verified 4.2 max-diff on follower tokens); running it natively
        keeps bit-exact semantics while the rest of _logits_dev still
        fuses."""
        c = self.config
        V = self.vocab_size
        _flux_logit_tail_kernel[(triton.cdiv(V, 1024),)](
            logits, log_uni, uni_pv, sims, csims, kv,
            self._seen_g, wg, pvn, rn, cf, cok, kvt, end,
            V,
            SEMX=self._chix["sem"], CSEMX=self._chix["csem"],
            KNNX=self._chix["knn"], RECX=self._chix["recency"],
            FATX=self._chix["fatigue"],
            USE_SEM=bool(c.use_sem), USE_CSEM=bool(c.use_csem),
            BETA_SEM=c.sem_beta, BETA_CSEM=c.csem_beta,
            ALPHA=c.alpha, KAPPA=c.backoff_kappa,
            LIFT=bool(c.evidence_lift),
            SPAN_R=c.recency_span,
            LDEC_R=math.log(c.recency_decay),
            SPAN_F=c.fatigue_span,
            LDEC_F=math.log(c.fatigue_decay),
            FAT_STR=c.fatigue_strength,
            BLOCK=1024)

    def _logits_dev(self, end: torch.Tensor,
                    slot: torch.Tensor, found: torch.Tensor,
                    live_k: torch.Tensor, cok: torch.Tensor,
                    dok: torch.Tensor, dk: torch.Tensor,
                    kv: torch.Tensor | None = None,
                    kvt: torch.Tensor | None = None,
                    lite: bool = False,
                    cap: dict | None = None
                    ) -> torch.Tensor:
        """Assemble [V] logits from a combined ``_probes_at`` result.
        ``kv``/``kvt`` are the caller-computed knn vote map + total
        (``_knn_votes_dev``) - kept outside this function because a
        python-int ``knn_rows`` arg made torch.compile specialize per
        value and recompile during CUDA-graph capture.  ``lite`` is
        the adaptive-read variant: deep+knn scans skipped, their
        contribs fall through the zero-vote masks to the unigram prior
        (CPU parity)."""
        c = self.config
        V = self.vocab_size
        gt = self._gt
        O = gt.O
        denom = (self._tot_g + c.uni_alpha * V).float()
        uni_pv = (self._unig + c.uni_alpha) / denom
        log_uni = self._log_uni_num - torch.log(denom)
        logits = log_uni.clone()
        if cap is not None:
            cap["uni"] = log_uni
        floor = torch.clamp(end + 1 - c.ring_capacity, min=0)

        oslot, ofound = slot[:O], found[:O]
        hit = ofound & live_k
        rows_t = gt.ttoks[gt._ordx, oslot]             # [O, M]
        rows_c = gt.tcnt[gt._ordx, oslot]              # [O, M]
        tots = gt.ttot[gt._ordx, oslot]                # [O]
        # warm flag = any order cell has real evidence this context -
        # feeds cond_hedge regime routing AND the host sparse-tap lift
        # gate (sparse votes are thin-context fallback only)
        warm_any = (hit & (tots > c.thin_tot)).any()
        self._dthin_g.copy_(warm_any.reshape(1).long())
        if c.cond_hedge:
            self._regime_g.fill_(warm_any.long())
            wrow = self._wgR.index_select(
                0, self._regime_g.view(1)).squeeze(0)
        else:
            wrow = self._wg
        # adaptive-read flag for the NEXT step: dominant follower share
        # on any evidenced order cell - same cells the order channel
        # already gathered, so the flag costs ~4 kernels
        share = rows_c.max(dim=1).values / tots.clamp(min=1e-9)
        conf_ok = hit & (tots >= c.adaptive_read_min_tot) \
            & (share >= c.adaptive_read_conf)
        self._dconf_g.copy_(conf_ok.any().reshape(1).long())
        wo = wrow[:O].float()[:, None]             # [O,1]
        if c.evidence_lift:
            # effective weight = hedge prior + evidence confidence -
            # KN-style interpolation; a massive cell speaks for itself
            wo = wo + (tots / (tots + c.backoff_kappa))[:, None]
        tok_l = rows_t.long().clamp(0, V - 1)
        row_live = (rows_t >= 0) & (rows_t < V) & hit[:, None]
        p = (rows_c + c.alpha * uni_pv[tok_l]) \
            / (tots[:, None] + c.alpha)
        contrib = wo * (torch.log(p.clamp(min=1e-12))
                        - log_uni[tok_l])
        contrib = torch.where(row_live, contrib,
                              torch.zeros((), device=self._dev))
        # deterministic accumulation - a token can appear in several
        # order cells, and index_add_ atomics made the sum (and thus
        # the logits) depend on kernel timing
        logits += _scatter_sum(tok_l.reshape(-1), contrib.reshape(-1), V)
        if cap is not None:
            for i, k in enumerate(c.orders):
                cap[f"o{k}"] = _scatter_sum(
                    tok_l[i], contrib[i], V)

        # backoff channel (inf-gram): the longest live order with enough
        # evidence, confidence-discounted - hedged alongside the rest
        if c.use_backoff:
            ok_bo = hit & (tots >= c.backoff_min_tot)
            kstar = torch.where(
                ok_bo, self._orders_g.float(),
                torch.full((), -1.0, device=self._dev)).argmax()
            bo_ok = ok_bo.index_select(
                0, kstar.reshape(1)).squeeze(0)
            frow = (kstar * gt.S
                    + oslot.index_select(
                        0, kstar.reshape(1)).squeeze(0))
            rt_k = gt.ttoks.view(-1, gt.M).index_select(
                0, frow.reshape(1)).squeeze(0)          # [M]
            rc_k = gt.tcnt.view(-1, gt.M).index_select(
                0, frow.reshape(1)).squeeze(0)
            tot_k = gt.ttot.view(-1).index_select(
                0, frow.reshape(1)).squeeze(0)
            conf = tot_k / (tot_k + c.backoff_kappa)
            tl = rt_k.long().clamp(min=0)
            rl = rt_k >= 0
            p_b = conf * (rc_k + c.alpha * uni_pv[tl]) \
                / (tot_k + c.alpha)
            wb = wrow[self._chix["backoff"]].float()
            if c.evidence_lift:
                wb = wb + conf
            # same atomic-order hazard as the order channel - a
            # follower token can repeat across the row's M slots
            d_bo = _scatter_sum(
                tl,
                torch.where(
                    rl & bo_ok,
                    wb * (torch.log(p_b.clamp(min=1e-12))
                          - log_uni[tl]),
                    torch.zeros((), device=self._dev)), V)
            logits += d_bo
            if cap is not None:
                cap["backoff"] = d_bo

        if lite:
            # adaptive-read: skip the retrieval scan - the zero-vote
            # mask below already yields the unigram fallback
            votes = torch.zeros(V, dtype=torch.float32,
                                device=self._dev)
            vt = votes.sum()
            self._dhit_g.zero_()          # no read -> no utility hit
        else:
            votes = self._deep_votes_dev(
                end, floor,
                se=slot[gt.row_epi].reshape(1),
                fe=found[gt.row_epi].reshape(1), dk=dk)
            vt = votes.sum()
            # retrieval-hit flag for host-side utility accounting (
            # eviction keeps cells that actually get read )
            self._dhit_g.copy_(
                ((vt > 0) & found[gt.row_epi]).reshape(1).long())
        # thin-context boost: retrieval/semantic channels get a fixed
        # weight bonus when no order cell holds real evidence - same
        # gate as the host-side lift (warm_any already computed above)
        tb = ((1.0 - warm_any.float()) * c.thin_boost
              if c.thin_boost > 0
              else torch.zeros((), dtype=torch.float64,
                               device=self._dev))
        wrow_b = wrow + tb * self._tb_mask
        wd = wrow_b[self._chix["deep"]].float()
        if c.evidence_lift:
            wd = wd + vt / (vt + c.backoff_kappa)
        p_d = (votes + c.alpha * uni_pv) / (vt + c.alpha)
        d_deep = torch.where(
            vt > 0, wd * (torch.log(p_d.clamp(min=1e-12))
                          - log_uni),
            torch.zeros((), device=self._dev))
        logits += d_deep
        if cap is not None:
            cap["deep"] = d_deep
            cap["_votes_deep"] = votes

        # topic: cosine readout on top candidates (norms from _An)
        cn = self._c.norm()
        wt = wrow_b[self._chix["topic"]].float()
        cand = self._topk_dev(logits, min(256, V))
        a = self._A[cand]
        an = self._An.index_select(0, cand)
        cos = (a @ self._c) / (an * cn + 1e-8)
        d_top = torch.where(
            cn > 1e-8, wt * c.topic_beta * cos,
            torch.zeros((), device=self._dev))
        logits[cand] += d_top
        if cap is not None:
            cap["topic"] = torch.zeros(
                V, device=self._dev).scatter(0, cand, d_top)

        # sem + csem share _Ac = A @ Pcsem (rank-1 maintained): JL
        # projection preserves cosine direction, so both channels score
        # in the 64-dim space instead of a fresh [V,192] matvec
        pv = self._proto @ self._Pcsem                    # [64]
        pvn = pv.norm()
        sims = (self._Ac @ pv) / (self._Acn * pvn + 1e-8)

        # csem: per-context fingerprint over full vocab
        cs = slot[gt.row_csem].reshape(1)
        cf = found[gt.row_csem].reshape(1)
        row = self._csem_rows.index_select(0, cs).squeeze(0).float()
        rn = row.norm()
        csims = (self._Ac @ row) / (self._Acn * rn + 1e-8)

        # knn votes arrive pre-computed (see signature note) - a missing
        # pair means the caller wants no knn contribution this step
        if kv is None:
            kv = torch.zeros(V, dtype=torch.float32, device=self._dev)
            kvt = torch.zeros((), dtype=torch.float32,
                              device=self._dev)

        if self._use_triton and cap is None:
            # one fused V-map: sem + csem + knn mixture + recency +
            # fatigue - replaces ~25 elementwise kernels per predict
            self._logit_tail_dev(logits, log_uni, uni_pv, sims, csims,
                                 kv, kvt, cf, cok, rn, pvn, end, wrow_b)
            if self._head_ready():
                gate = (1.0 - warm_any.float() if c.head_thin
                        else 1.0)
                logits = logits + (c.head_w * gate) * self._head_apply(
                    self._head_input_dev(end))
            if c.uni_escape > 0:
                logits = torch.log(
                    (1.0 - c.uni_escape) * torch.softmax(logits, -1)
                    + c.uni_escape * uni_pv)
            return logits

        ws = wrow_b[self._chix["sem"]].float() * float(c.use_sem)
        d_sem = torch.where(
            pvn > 1e-8, ws * c.sem_beta * sims,
            torch.zeros((), device=self._dev))
        logits += d_sem

        wc = wrow_b[self._chix["csem"]].float() * float(c.use_csem)
        d_csem = torch.where(
            cf.squeeze() & cok & (rn > 1e-8),
            wc * c.csem_beta * csims,
            torch.zeros((), device=self._dev))
        logits += d_csem

        d_knn = torch.zeros((), device=self._dev)
        if c.use_knn:
            p_k = (kv + c.alpha * uni_pv) / (kvt + c.alpha)
            wk = wrow_b[self._chix["knn"]].float()
            if c.evidence_lift:
                wk = wk + kvt / (kvt + c.backoff_kappa)
            d_knn = torch.where(
                kvt > 0, wk * (torch.log(p_k.clamp(min=1e-12))
                               - log_uni),
                torch.zeros((), device=self._dev))
            logits += d_knn
        if cap is not None:
            cap["sem"] = d_sem
            cap["csem"] = d_csem
            cap["knn"] = d_knn

        # recency + fatigue over full V
        ages = (end - self._seen_g).clamp(min=0)
        rec = torch.where(
            (ages < c.recency_span) & (self._seen_g >= 0),
            c.recency_decay ** ages.float(),
            torch.zeros((), device=self._dev))
        fat = torch.where(
            (ages < c.fatigue_span) & (self._seen_g >= 0),
            c.fatigue_decay ** ages.float(),
            torch.zeros((), device=self._dev))
        d_rec = wrow[self._chix["recency"]].float() * rec
        d_fat = -wrow[self._chix["fatigue"]].float() \
            * c.fatigue_strength * fat
        logits += d_rec + d_fat
        if cap is not None:
            cap["recency"] = d_rec
            cap["fatigue"] = d_fat
        # distilled head (parametric teacher channel)
        if self._head_ready():
            gate = (1.0 - warm_any.float() if c.head_thin
                    else 1.0)
            d_head = (c.head_w * gate) * self._head_apply(
                self._head_input_dev(end))
            if cap is not None:
                cap["head"] = d_head
            logits = logits + d_head
        if cap is not None:
            cap["_pre_escape"] = logits.clone()
        if c.uni_escape > 0:
            logits = torch.log(
                (1.0 - c.uni_escape) * torch.softmax(logits, -1)
                + c.uni_escape * uni_pv)
        return logits

    def _logits_fast(self, *a, **kw) -> torch.Tensor:
        """``_logits_dev`` via torch.compile when enabled - inductor
        fuses the ~200-op elementwise chain into a handful of kernels
        (~2.6x inside a captured graph).  Custom triton kernels and
        topk sites stay dynamo-disabled (tie-order parity with eager);
        any compile failure permanently falls back to the eager chain.
        Guard checks are python-side - skipped on graph replay."""
        lc = getattr(self, "_logits_c", None)
        if lc is None:
            lc = False
            if self._gpu and self.config.compile_dev:
                try:
                    # inductor's debug index asserts emit tl.device_assert
                    # traps - firing one inside capture invalidates the
                    # whole CUDA graph (bounds are clamped upstream)
                    import torch._inductor.config as _icfg
                    _icfg.assert_indirect_indexing = False
                    lc = torch.compile(self._logits_dev, dynamic=False)
                except Exception:
                    lc = False
            self._logits_c = lc
        if lc is not False:
            try:
                return lc(*a, **kw)
            except Exception:
                self._logits_c = False
        return self._logits_dev(*a, **kw)

    def _probes_fast(self, end: torch.Tensor):
        """Compiled ``_probes_at`` - same lazy-compile/fallback pattern
        as ``_logits_fast`` (probes are read-only gathers; compiled
        output verified tensor-identical)."""
        pc = getattr(self, "_probes_c", None)
        if pc is None:
            pc = False
            if self._gpu and self.config.compile_dev:
                try:
                    import torch._inductor.config as _icfg
                    _icfg.assert_indirect_indexing = False
                    pc = torch.compile(self._probes_at, dynamic=False)
                except Exception:
                    pc = False
            self._probes_c = pc
        if pc is not False:
            try:
                return pc(end)
            except Exception:
                self._probes_c = False
        return self._probes_at(end)

    def _predict_dev(self, end: torch.Tensor) -> torch.Tensor:
        """Next-token logits [V] - pure tensor ops, no host reads."""
        slot, found, _ev, live_k, cok, dok, dk, _pk = \
            self._probes_fast(end)
        kb = self._knn_rows()
        if self.config.use_knn and kb > 0:
            kv, kvt = self._knn_votes_dev(kb)
        else:
            # uniform tensor signature - see _step_dev note
            kv = torch.zeros(self.vocab_size, dtype=torch.float32,
                             device=self._dev)
            kvt = torch.zeros((), dtype=torch.float32,
                              device=self._dev)
        return self._logits_fast(end, slot, found, live_k,
                                 cok, dok, dk, kv=kv, kvt=kvt)

    def _sparse_lift_channels_dev(self, warm: int,
                                  gc_code: int | None = None
                                  ) -> dict[str, dict[int, float]]:
        """Host-table lifts for the current context, split per channel
        ({'p0': {tok: dv}, 'g1': {...}, ...}).  Same math as the merged
        ``_apply_sparse_lift_dev`` - the split is what explain() reports
        and the wrapper merges back for the hot path."""
        c = self.config
        sparse_ok = c.sparse_taps and not (c.sparse_thin_only and warm)
        class_ok = c.class_orders and not (c.class_thin_only and warm)
        word_ok = c.word_orders and not (c.word_thin_only and warm)
        hybrid_ok = c.hybrid_taps and not (c.hybrid_thin_only and warm)
        if not (sparse_ok or class_ok or word_ok or hybrid_ok):
            return {}
        end = self._count - 1
        denom = self._total + c.uni_alpha * self.vocab_size
        wsrc = self._wR[warm] if c.cond_hedge else self._w
        out: dict[str, dict[int, float]] = {}

        def acc(into: dict, cell, tot, wj):
            if c.evidence_lift:
                wj += tot / (tot + c.backoff_kappa)
            for t, cnt in cell.items():
                p = (cnt + c.alpha * self._uni_p(t, denom)) \
                    / (tot + c.alpha)
                into[t] = into.get(t, 0.0) + wj * (
                    math.log(max(p, 1e-12))
                    - (math.log(self._uni[t] + c.uni_alpha)
                       - math.log(denom)))

        if sparse_ok:
            for j, taps in enumerate(c.sparse_taps):
                wj = float(wsrc[self._chix[f"p{j}"]])
                if wj <= 1e-4:
                    continue
                key = self._gap_key(taps, end)
                if key is None:
                    continue
                cell = self._ptabs[taps].get(key)
                if cell:
                    acc(out.setdefault(f"p{j}", {}), cell,
                        self._ptots[taps].get(key, 0.0), wj)
        if class_ok:
            for k in c.class_orders:
                wk = float(wsrc[self._chix[f"g{k}"]])
                if wk <= 1e-4:
                    continue
                key = self._class_key(k, end)
                if key is None:
                    continue
                cell = self._gtabs[k].get(key)
                if cell:
                    tot = self._gtots[k].get(key, 0.0)
                    if tot >= c.class_min_tot:
                        acc(out.setdefault(f"g{k}", {}), cell, tot, wk)
        if word_ok:
            for j, w in enumerate(c.word_orders):
                wj = float(wsrc[self._chix[f"w{j}"]])
                if wj <= 1e-4:
                    continue
                key = self._word_key(w, end)
                if key is None:
                    continue
                cell = self._wtabs[j].get(key)
                if cell:
                    acc(out.setdefault(f"w{j}", {}), cell,
                        self._wtots[j].get(key, 0.0), wj)
        if hybrid_ok:
            for j, pat in enumerate(c.hybrid_taps):
                wj = float(wsrc[self._chix[f"h{j}"]])
                if wj <= 1e-4:
                    continue
                key = self._hybrid_key(pat, end)
                if key is None:
                    continue
                cell = self._htabs[j].get(key)
                if cell:
                    acc(out.setdefault(f"h{j}", {}), cell,
                        self._htots[j].get(key, 0.0), wj)
        if c.gcell and not (c.gcell_thin and warm):
            # learned-key cell channel: SimHash code of the context
            # fingerprint; gc_code overrides the live-context code in
            # batched replay (device _c is end-of-row by then)
            code = gc_code if gc_code is not None else \
                self._gcell_code(self._c, self._c2, end)
            for cc, dist in self._gcell_neighbors(code):
                cell = self._gc.get(cc)
                if cell:
                    acc(out.setdefault("gcell", {}), cell,
                        self._gct.get(cc, 0.0),
                        float(c.gcell_w) * self._gen_bias
                        * c.gcell_neighbor_decay ** dist)
        return out

    def _gsyn_lift_vec(self, warm: int,
                       q: torch.Tensor | None = None
                       ) -> torch.Tensor | None:
        """[V] dense lift for the gsyn channel, or None when gated
        off.  Same additive-logit form as the host-cell channels:
        w * (log p_gsyn - log p_uni)."""
        c = self.config
        if not c.gsyn or (c.gsyn_thin and warm):
            return None
        if q is None:
            q = self._gsyn_keys(self._gcell_feats(
                self._c, self._c2, self._count - 1))
        lp = self._gsyn_logits(q)
        base = self._log_uni_num - math.log(
            self._total + c.uni_alpha * self.vocab_size)
        return float(c.gsyn_w) * self._gen_bias * (lp - base)

    def _apply_sparse_lift_dev(self, row: torch.Tensor,
                               warm: int,
                               gc_code: int | None = None,
                               gsyn_q: torch.Tensor | None = None
                               ) -> None:
        """Host-computed sparse-tap + class-order lift scatter-added
        onto a device [V] logit row.  Those tables are host-resident on
        GPU models (mixed split - zero VRAM); the predict-time warm flag
        rides ``_bout``/``_dthin_g`` so the host knows when the
        thin-only gates let the host channels vote.  Applied
        post-``uni_escape`` here vs in-mixture on CPU - a <=(1-eps)
        magnitude divergence, logits renormalize downstream anyway."""
        lift: dict[int, float] = {}
        for d in self._sparse_lift_channels_dev(
                warm, gc_code=gc_code).values():
            for t, v in d.items():
                lift[t] = lift.get(t, 0.0) + v
        if lift:
            idx = torch.tensor(list(lift), dtype=torch.int64,
                               device=self._dev)
            row.index_add_(0, idx, torch.tensor(
                list(lift.values()), dtype=torch.float32,
                device=self._dev))
        gl = self._gsyn_lift_vec(warm, q=gsyn_q)
        if gl is not None:
            row.add_(gl)

    def _predict_gpu(self) -> torch.Tensor:
        """Standalone predict (no learn) - CUDA-graph captured like
        ``_step_dev``: the context end rides a device input tensor and
        the knn bucket is baked into the capture (recapture on cross).
        Eager fallback while warming or when graphs are disabled
        (clone workers).  Measured ~16x vs the eager pass (5.9ms ->
        0.37ms at V=512)."""
        self._bend.fill_(self._count - 1)
        if not getattr(self, "_use_cuda_graph", True):
            lg = self._predict_dev(self._bend)
        else:
            rows = self._knn_rows()
            if self._pgraph is not None:
                if rows != self._pgraph_knn_b:
                    self._pgraph = None
                else:
                    self._pgraph.replay()
                    lg = self._plogits
                    if (self.config.sparse_taps or self.config.class_orders
                            or self.config.word_orders
                            or self.config.hybrid_taps
                            or self.config.gcell or self.config.gsyn):
                        self._apply_sparse_lift_dev(
                            lg, int(self._dthin_g.item()))
                    return lg
            if self._pwarm < 3:
                self._pwarm += 1
                lg = self._predict_dev(self._bend)
            else:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    self._plogits.copy_(
                        self._predict_dev(self._bend))
                self._pgraph = g
                self._pgraph_knn_b = rows
                g.replay()
                lg = self._plogits
        if (self.config.sparse_taps or self.config.class_orders
                or self.config.word_orders
                or self.config.hybrid_taps or self.config.gcell
                or self.config.gsyn):
            self._apply_sparse_lift_dev(
                lg, int(self._dthin_g.item()))
        return lg

    def _decay_table(self, tab: dict[int, dict[int, float]],
                     tots: dict[int, float]) -> None:
        """Halve all counts and drop empties (forgetting + capacity)."""
        dead = []
        for key, cell in tab.items():
            for t in list(cell):
                v = cell[t] * 0.5
                if v < 0.25:
                    del cell[t]
                else:
                    cell[t] = v
            if not cell:
                dead.append(key)
            else:
                tots[key] = sum(cell.values())
        for key in dead:
            del tab[key]
            tots.pop(key, None)

    # -- prediction ------------------------------------------------------

    def _head_in_dim(self) -> int:
        c = self.config
        return ((2 if c.bigram_sketch else 1) + c.head_orders) \
            * c.topic_dim

    def _head_ready(self) -> bool:
        """_Wh (+ optional _Wh1) shapes consistent with config."""
        c = self.config
        if c.head_w <= 0 or self._Wh is None:
            return False
        lin = self._head_in_dim()
        if self._Wh1 is None:
            return self._Wh.shape[1] == lin and c.head_hidden <= 0
        return (c.head_hidden > 0
                and self._Wh.shape[1] == c.head_hidden
                and tuple(self._Wh1.shape) == (c.head_hidden, lin))

    def _head_apply(self, x: torch.Tensor) -> torch.Tensor:
        """Wh @ x (linear) or Wh @ relu(Wh1 @ x) (MLP channel)."""
        if self._Wh1 is not None:
            x = (x @ self._Wh1.T).clamp(min=0)
        return self._Wh @ x

    def _head_input_dev(self, end: torch.Tensor) -> torch.Tensor:
        """Order-sensitive distill-head input, device path:
        concat(normalized topic sketch, unit-scaled _R rows of the
        last ``head_orders`` tokens, most-recent-first), zero-padded
        before stream start, L2-normalized.  Reads ``_ringg`` indexed
        by the device ``end`` tensor - capture-safe inside CUDA
        graphs (no host reads)."""
        c = self.config
        d = c.topic_dim
        k = c.head_orders
        base = self._c / self._c.norm().clamp(min=1e-8)
        if c.bigram_sketch:
            base = torch.cat(
                [base, self._c2 / self._c2.norm().clamp(min=1e-8)])
        if k <= 0:
            return base
        j = torch.arange(k, device=self._dev)
        pos = end - j                                   # t-1 .. t-k
        valid = (pos >= 0).float()
        ti = self._ringg[pos.clamp(min=0) % c.ring_capacity].long()
        rows = self._R[ti].float() * valid[:, None] / (d ** 0.5)
        x = torch.cat([base, rows.reshape(-1)])
        return x / x.norm().clamp(min=1e-8)

    def _head_input_host(self) -> torch.Tensor:
        """CPU mirror of ``_head_input_dev`` - tail tokens from the
        host ``_ring`` at ``self._count``."""
        c = self.config
        d = c.topic_dim
        k = c.head_orders
        base = self._c / self._c.norm().clamp(min=1e-8)
        if c.bigram_sketch:
            base = torch.cat(
                [base, self._c2 / self._c2.norm().clamp(min=1e-8)])
        if k <= 0:
            return base
        cap = c.ring_capacity
        tail = [int(self._ring[(self._count - 1 - j) % cap])
                for j in range(k)]
        valid = torch.tensor(
            [self._count - 1 - j >= 0 for j in range(k)],
            dtype=torch.float32, device=self._dev)
        rows = (self._R[torch.as_tensor(tail, dtype=torch.long,
                                        device=self._dev)].float()
                * valid[:, None] / (d ** 0.5))
        x = torch.cat([base, rows.reshape(-1)])
        return x / x.norm().clamp(min=1e-8)

    def _head_feats(self, ctxs: list[list[int]]) -> torch.Tensor:
        """Batched head features for arbitrary token windows
        (distill_head training rows).  [B, (1+head_orders)*topic_dim],
        L2-normalized rows (zero rows for empty contexts)."""
        c = self.config
        d = c.topic_dim
        k = c.head_orders
        B = len(ctxs)
        ema = torch.stack([self._context_vector(cc) for cc in ctxs])
        ema = ema / ema.norm(dim=1, keepdim=True).clamp(min=1e-8)
        if c.bigram_sketch:
            bg = torch.stack([self._bigram_vector(cc) for cc in ctxs])
            bg = bg / bg.norm(dim=1, keepdim=True).clamp(min=1e-8)
            ema = torch.cat([ema, bg], 1)
        if k <= 0:
            return ema / ema.norm(dim=1, keepdim=True).clamp(min=1e-8)
        tail = torch.zeros(B, k, dtype=torch.long, device=self._dev)
        vmask = torch.zeros(B, k, device=self._dev)
        for i, cc in enumerate(ctxs):
            n = min(len(cc), k)
            if n:
                tail[i, :n] = torch.tensor(
                    cc[-n:][::-1], dtype=torch.long, device=self._dev)
                vmask[i, :n] = 1.0
        rows = self._R[tail].float() * vmask[:, :, None] / (d ** 0.5)
        x = torch.cat([ema, rows.reshape(B, -1)], 1)
        return x / x.norm(dim=1, keepdim=True).clamp(min=1e-8)

    def _predict(self, cap: dict | None = None) -> np.ndarray:
        """Next-token logit vector [V] from the current sketch.

        ``cap`` (optional): per-channel contribution capture - each
        channel's additive lift is recorded under its Hedge name
        (dict[int, float] or [V] tensor) for explain()."""
        if self._gpu:
            return self._predict_gpu()
        c = self.config
        end = self._count - 1
        keys = self._keys_at(end) if end >= self._ep_start else {}
        denom = self._total + c.uni_alpha * self.vocab_size
        w_src = self._w
        if c.cond_hedge:
            self._regime = self._regime_now(keys)
            w_src = self._wR[self._regime]
        w = {ch: w_src[i] for i, ch in enumerate(self.channels)}

        # base = log unigram prior = log_numer - log(denom); the numer
        # tensor is maintained per-write so this is one vector subtract.
        log_denom = math.log(denom)
        logits = self._log_uni_num - log_denom      # [V] on self._dev
        if cap is not None:
            cap["uni"] = logits.clone()

        def log_uni(t: int) -> float:
            # scalar read from the host-side count array - no GPU sync
            return math.log(self._uni[t] + c.uni_alpha) - log_denom

        lift: dict[int, float] = {}

        def acc(t: int, dv: float, ch: str) -> None:
            lift[t] = lift.get(t, 0.0) + dv
            if cap is not None:
                cd = cap.setdefault(ch, {})
                cd[t] = cd.get(t, 0.0) + dv

        # order channels
        for k in c.orders:
            key = keys.get(k)
            if key is None:
                continue
            cell = self._tables[k].get(key)
            if not cell:
                continue
            tot = self._totals[k].get(key, 0.0)
            wk = w[f"o{k}"]
            if c.evidence_lift:
                wk += tot / (tot + c.backoff_kappa)
            for t, cnt in cell.items():
                p = (cnt + c.alpha * self._uni_p(t, denom)) / (tot + c.alpha)
                acc(t, wk * (math.log(max(p, 1e-12)) - log_uni(t)),
                    f"o{k}")

        # sparse-tap channels "p<j>": gapped-suffix keys skip single
        # corrupted/novel positions that break every contiguous key.
        # thin-only gating (default): they are fallback experts - when
        # contiguous orders have evidence their votes only dilute.
        sparse_ok = bool(c.sparse_taps) and not (
            c.sparse_thin_only and self._regime_now(keys) == 1)
        for j, taps in enumerate(c.sparse_taps) if sparse_ok else ():
            wj = w[f"p{j}"]
            if wj <= 1e-4:
                continue
            key = self._gap_key(taps, end)
            if key is None:
                continue
            cell = self._ptabs[taps].get(key)
            if not cell:
                continue
            tot = self._ptots[taps].get(key, 0.0)
            if c.evidence_lift:
                wj += tot / (tot + c.backoff_kappa)
            for t, cnt in cell.items():
                p = (cnt + c.alpha * self._uni_p(t, denom)) \
                    / (tot + c.alpha)
                acc(t, wj * (math.log(max(p, 1e-12)) - log_uni(t)),
                    f"p{j}")

        # class-order channels "g<k>": contiguous suffixes over byte
        # classes - coarse keys that fire on novel text where no
        # verbatim byte context exists.  Same thin-only fallback
        # gating as the sparse arms.
        class_ok = bool(c.class_orders) and not (
            c.class_thin_only and self._regime_now(keys) == 1)
        for k in c.class_orders if class_ok else ():
            wk = w[f"g{k}"]
            if wk <= 1e-4:
                continue
            key = self._class_key(k, end)
            if key is None:
                continue
            cell = self._gtabs[k].get(key)
            if not cell:
                continue
            tot = self._gtots[k].get(key, 0.0)
            if tot < c.class_min_tot:
                continue
            if c.evidence_lift:
                wk += tot / (tot + c.backoff_kappa)
            for t, cnt in cell.items():
                p = (cnt + c.alpha * self._uni_p(t, denom)) \
                    / (tot + c.alpha)
                acc(t, wk * (math.log(max(p, 1e-12)) - log_uni(t)),
                    f"g{k}")

        # word-order channels "w<j>": keys fold the last-w completed
        # word hashes + partial-word length bucket - generalize across
        # novel contexts at word granularity (PAQ8 word model).  Same
        # thin-only fallback gating.
        word_ok = bool(c.word_orders) and not (
            c.word_thin_only and self._regime_now(keys) == 1)
        for j, nw in enumerate(c.word_orders) if word_ok else ():
            wj = w[f"w{j}"]
            if wj <= 1e-4:
                continue
            key = self._word_key(nw, end)
            if key is None:
                continue
            cell = self._wtabs[j].get(key)
            if not cell:
                continue
            tot = self._wtots[j].get(key, 0.0)
            if c.evidence_lift:
                wj += tot / (tot + c.backoff_kappa)
            for t, cnt in cell.items():
                p = (cnt + c.alpha * self._uni_p(t, denom)) \
                    / (tot + c.alpha)
                acc(t, wj * (math.log(max(p, 1e-12)) - log_uni(t)),
                    f"w{j}")

        # hybrid class+raw channels "h<j>": (offset, is_raw) patterns -
        # sharper than pure class keys, still generalize where only the
        # newest bytes are verbatim.
        hybrid_ok = bool(c.hybrid_taps) and not (
            c.hybrid_thin_only and self._regime_now(keys) == 1)
        for j, pat in enumerate(c.hybrid_taps) if hybrid_ok else ():
            wj = w[f"h{j}"]
            if wj <= 1e-4:
                continue
            key = self._hybrid_key(pat, end)
            if key is None:
                continue
            cell = self._htabs[j].get(key)
            if not cell:
                continue
            tot = self._htots[j].get(key, 0.0)
            if c.evidence_lift:
                wj += tot / (tot + c.backoff_kappa)
            for t, cnt in cell.items():
                p = (cnt + c.alpha * self._uni_p(t, denom)) \
                    / (tot + c.alpha)
                acc(t, wj * (math.log(max(p, 1e-12)) - log_uni(t)),
                    f"h{j}")

        # learned-key cell channel "gcell": SimHash of the context
        # fingerprint -> sparse succ-score cell - generalizes where no
        # verbatim key exists.  Fixed weight (bypass), thin-gated.
        if c.gcell and (not c.gcell_thin
                        or self._regime_now(keys) == 0):
            code = self._gcell_code(self._c, self._c2, end)
            for cc, dist in self._gcell_neighbors(code):
                cell = self._gc.get(cc)
                if not cell:
                    continue
                tot = self._gct.get(cc, 0.0)
                wg = float(c.gcell_w)
                if c.evidence_lift:
                    wg += tot / (tot + c.backoff_kappa)
                wg *= (self._gen_bias
                       * c.gcell_neighbor_decay ** dist)
                for t, sc in cell.items():
                    p = (sc + c.alpha * self._uni_p(t, denom)) \
                        / (tot + c.alpha)
                    acc(t, wg * (math.log(max(p, 1e-12))
                                 - log_uni(t)), "gcell")

        # backoff channel: longest-evidenced order's distribution,
        # confidence-discounted - one arm instead of O probes
        if c.use_backoff:
            best_k, bkey, btot = -1, -1, 0.0
            for k, key in keys.items():
                t = self._totals[k].get(key, 0.0)
                if t >= c.backoff_min_tot and k > best_k:
                    best_k, bkey, btot = k, key, t
            if best_k >= 0:
                cell = self._tables[best_k].get(bkey) or {}
                conf = btot / (btot + c.backoff_kappa)
                wb = w["backoff"] + (conf if c.evidence_lift else 0.0)
                for t, cnt in cell.items():
                    p = conf * (cnt + c.alpha * self._uni_p(t, denom)) \
                        / (btot + c.alpha)
                    acc(t, wb * (math.log(max(p, 1e-12)) - log_uni(t)),
                        "backoff")

        # thin-context boost: retrieval/semantic channels get a fixed
        # weight bonus when orders are silent - their hedge-learned
        # weights sit near the floor (rewards are mediocre in the warm
        # contexts that dominate training), but they are the only
        # signal left when nothing verbatim matches.
        tb = (c.thin_boost
              if c.thin_boost > 0 and self._regime_now(keys) == 0
              else 0.0)

        # deep episodic channel - skipped when an order cell already
        # dominates (adaptive read gate: saves the scan + knn matvec)
        confident = c.adaptive_read and self._orders_confident(keys)
        votes = {} if confident else self._deep_votes(end)
        if votes:
            vt = sum(votes.values())
            wd = w["deep"] + tb + (vt / (vt + c.backoff_kappa)
                                   if c.evidence_lift else 0.0)
            for t, v in votes.items():
                p = (v + c.alpha * self._uni_p(t, denom)) / (vt + c.alpha)
                acc(t, wd * (math.log(max(p, 1e-12)) - log_uni(t)),
                    "deep")

        # knn channel: approximate-match contexts vote their successors
        if c.use_knn:
            kvotes = {} if confident else self._knn_votes()
            if kvotes:
                vt = sum(kvotes.values())
                wk = w["knn"] + tb
                if c.evidence_lift:
                    wk += vt / (vt + c.backoff_kappa)
                for t, v in kvotes.items():
                    p = (v + c.alpha * self._uni_p(t, denom)) \
                        / (vt + c.alpha)
                    acc(t, wk * (math.log(max(p, 1e-12)) - log_uni(t)),
                        "knn")

        # cosine channels on the candidate set - gather the A-rows once
        # per channel and score with a single matvec, then fold the adds
        # back into ``lift`` as a batch.  The old per-row python loop did
        # N separate ``float(...)`` reads (N GPU syncs when _dev is cuda).
        def _add_cosine(vec: torch.Tensor, vec_norm: float,
                        weight: float,
                        proj: torch.Tensor | None = None,
                        ch: str = "") -> None:
            if not lift:
                return
            ids = list(lift)
            idx = torch.tensor(ids, dtype=torch.long, device=self._dev)
            rows = self._A[idx]
            if proj is not None:
                rows = rows @ proj
            an = rows.norm(dim=1)
            denom = (an * vec_norm).clamp(min=1e-8)
            sims = torch.where(an > 1e-8, (rows @ vec) / denom,
                               torch.zeros_like(denom))
            for t, s in zip(ids, (weight * sims).tolist()):
                lift[t] += s
                if cap is not None:
                    cd = cap.setdefault(ch, {})
                    cd[t] = cd.get(t, 0.0) + s

        # topic channel: cosine readout on candidates only
        if lift:
            cn = float(self._c.norm())
            if cn > 1e-8:
                _add_cosine(self._c, cn,
                            (w["topic"] + tb) * c.topic_beta,
                            ch="topic")

        # semantic-spread channel: score tokens by fingerprint similarity
        # to the expected-next fingerprint - generalizes beyond the exact
        # followers seen in this context.  On CUDA it scores all V (one
        # matvec); on CPU it scores candidates only.
        if c.use_sem:
            pn = float(self._proto.norm())
            if pn > 1e-8:
                ws = (w["sem"] + tb) * c.sem_beta
                if self._dev.type == "cuda":
                    an = self._A.norm(dim=1)
                    sims = (self._A @ self._proto) / (an * pn + 1e-8)
                    if cap is not None:
                        cap["sem"] = ws * sims
                    logits = logits + ws * sims
                else:
                    _add_cosine(self._proto, pn, ws, ch="sem")

        # csem: per-context fingerprint - same trick as sem but keyed by
        # the current context (contextual recombination)
        if c.use_csem and lift:
            row = self._csem_read()
            if row is not None:
                rn = float(row.norm())
                if rn > 1e-8:
                    _add_cosine(row, rn, (w["csem"] + tb) * c.csem_beta,
                                proj=self._Pcsem, ch="csem")

        # distilled head: trained linear readout fingerprint->logits -
        # the only channel that generalizes to novel contexts by
        # construction (parametric, not lookup).  Fixed weight, not
        # hedged: it exists to answer where memory is thin.
        if (self._head_ready()
                and (not c.head_thin
                     or self._regime_now(keys) == 0)):
            hd = (c.head_w * self._head_apply(
                self._head_input_host())).float()
            if cap is not None:
                cap["head"] = hd
            logits = logits + hd

        # dense delta channel (gsyn): S@q predicts the next-token
        # fingerprint; lift = w * (logp_candidate - log_uni).  Smooth
        # generalization complement to gcell's sparse cells - fixed
        # weight, thin-gated, votes only where recall is silent.
        if (c.gsyn
                and (not c.gsyn_thin
                     or self._regime_now(keys) == 0)):
            gq = self._gsyn_keys(
                self._gcell_feats(self._c, self._c2, end))
            gl = (float(c.gsyn_w) * self._gen_bias
                  * (self._gsyn_logits(gq)
                     - (self._log_uni_num - log_denom))).float()
            if cap is not None:
                cap["gsyn"] = gl
            logits = logits + gl

        # recency channel: additive boost for recently seen tokens
        wr = w["recency"]
        if wr > 1e-4:
            for pos, t in reversed(self._recent):
                age = end - pos
                if age >= c.recency_span:
                    break
                if age >= 0:
                    acc(t, wr * c.recency_decay ** age, "recency")

        # fatigue channel: subtractive penalty for just-emitted tokens -
        # the memory model's intrinsic "you already said that"
        wf = w["fatigue"] * c.fatigue_strength
        if wf > 1e-4:
            for pos, t in reversed(self._recent):
                age = end - pos
                if age >= c.fatigue_span:
                    break
                if age >= 0:
                    acc(t, -wf * c.fatigue_decay ** age, "fatigue")

        # virtual rollout (thin contexts only): probe the order tables
        # with suffix keys of the hypothetical context ctx+[argmax] -
        # "what follows the most likely continuation?"  Fixed-weight
        # lookahead channel, read-only (no journal, no ring writes).
        if c.rollout_w > 0 and self._regime_now(keys) == 0:
            yhat = int(logits.argmax())
            k2 = self._keys_extended(yhat)
            votes2: dict[int, float] = {}
            w2 = 0.0
            for k, key2 in k2.items():
                cell = self._tables[k].get(key2)
                if not cell:
                    continue
                tot = self._totals[k].get(key2, 0.0)
                if tot <= 0.0:
                    continue
                wk = w[f"o{k}"]
                w2 += wk
                for t, cnt in cell.items():
                    votes2[t] = votes2.get(t, 0.0) + wk * cnt / tot
            if w2 > 0.0:
                for t, m in votes2.items():
                    p2 = (m / w2 + c.alpha * self._uni_p(t, denom)) \
                        / (1.0 + c.alpha)
                    acc(t, c.rollout_w * (
                        math.log(p2 + 1e-12) - log_uni(t)), "rollout")

        if lift:
            idx_t = torch.tensor(list(lift.keys()), dtype=torch.long,
                                 device=self._dev)
            val_t = torch.tensor(list(lift.values()), dtype=torch.float32,
                                 device=self._dev)
            logits.index_add_(0, idx_t, val_t)
        if cap is not None:
            cap["_pre_escape"] = logits.clone()
        if c.uni_escape > 0:
            uni_pv = (self._log_uni_num - log_denom).exp()
            logits = torch.log(
                (1.0 - c.uni_escape) * torch.softmax(logits, -1)
                + c.uni_escape * uni_pv)
        return logits

    # -- public API ------------------------------------------------------

    def gap_positions(self) -> list[int]:
        """Positions whose contexts were thin when learned (distill
        targets), oldest first.  Positions rewound by revert are
        dropped."""
        return sorted(p for p in self._gaps if p <= self._count)

    def rescan_gaps(self) -> int:
        """Rebuild gap positions by scanning the ring - for snapshots
        taken before gap tracking existed (or after big reverts).
        Measures CURRENT memory confidence for each stored context."""
        self._gaps.clear()
        c = self.config
        k = c.gap_probe_order
        start = max(0, self._count - c.ring_capacity)
        if self._gpu:
            # vectorized: raw probe-order hash per position -> tot probe
            pp = torch.arange(start + 1, self._count, dtype=torch.int64,
                              device=self._dev)
            valid = pp - 1 - k + 1 >= 0
            pp = pp[valid]
            if pp.numel() == 0:
                return 0
            e_ = pp - 1
            keys = (self._Hg[e_ % c.ring_capacity]
                    - self._Hg[(e_ - k) % c.ring_capacity]
                    * self._pows_g[k])
            tots = self._gt.read_tot(k, keys)
            thin = pp[tots < c.gap_min_total]
            self._gaps.extend(thin.cpu().tolist())
            return len(self._gaps)
        tab = self._tables.get(k, {})
        tots = self._totals.get(k, {})
        for pos in range(start + 1, self._count):
            key = self._raw_hash(k, pos - 1)
            if tots.get(key, 0.0) < c.gap_min_total:
                self._gaps.append(pos)
        return len(self._gaps)

    # -- explain: per-channel attribution -------------------------------

    def explain(self, top_k: int = 10) -> dict:
        """Per-channel decomposition of the current next-token
        distribution - which memory cells fired and how much each
        channel moved each top token's logit.  Read-only (no writes,
        no journal); runs the eager path so intermediates are visible
        even where the hot path is a fused CUDA graph."""
        if self._count == 0:
            return {"end": -1, "regime": 0, "top": [], "cells": [],
                    "neighbors": [],
                    "weights": self.channel_weights()}
        if self._gpu:
            return self._explain_gpu(top_k)
        return self._explain_cpu(top_k)

    def _explain_cpu(self, top_k: int) -> dict:
        c = self.config
        end = self._count - 1
        keys = self._keys_at(end) if end >= self._ep_start else {}
        cap: dict = {}
        logits = self._predict(cap=cap)
        cells = self._explain_cells_cpu(keys)
        votes = {} if (c.adaptive_read
                       and self._orders_confident(keys)) \
            else self._deep_votes(end)
        rep = self._explain_report(logits, cap, cells, top_k)
        rep["regime"] = self._regime_now(keys)
        rep["neighbors"] = self._knn_neighbors(c.knn_k)
        rep["deep_votes"] = sorted(votes.items(),
                                   key=lambda kv: -kv[1])[:8]
        return rep

    def _explain_gpu(self, top_k: int) -> dict:
        c = self.config
        V = self.vocab_size
        gt = self._gt
        hits0 = self._knn_hits.clone()      # vote scan counts hits
        end = torch.tensor(self._count - 1, dtype=torch.int64,
                           device=self._dev)
        slot, found, _ev, live_k, cok, dok, dk, _pk = \
            self._probes_at(end)
        kb = self._knn_rows()
        if c.use_knn and kb > 0:
            kv, kvt = self._knn_votes_dev(kb)
        else:
            kv = torch.zeros(V, dtype=torch.float32, device=self._dev)
            kvt = torch.zeros((), dtype=torch.float32,
                              device=self._dev)
        cap: dict = {}
        logits = self._logits_dev(end, slot, found, live_k, cok, dok,
                                  dk, kv=kv, kvt=kvt, cap=cap).clone()
        warm = int(self._dthin_g.item())
        sparse = self._sparse_lift_channels_dev(warm)
        lift: dict[int, float] = {}
        for ch, d in sparse.items():
            cap[ch] = d
            for t, v in d.items():
                lift[t] = lift.get(t, 0.0) + v
        if lift:
            idx = torch.tensor(list(lift), dtype=torch.int64,
                               device=self._dev)
            logits.index_add_(0, idx, torch.tensor(
                list(lift.values()), dtype=torch.float32,
                device=self._dev))
        gl = self._gsyn_lift_vec(warm)
        if gl is not None:
            cap["gsyn"] = gl
            logits = logits + gl
        self._knn_hits.copy_(hits0)
        cells = self._explain_cells_gpu(slot, found, live_k)
        rep = self._explain_report(logits, cap, cells, top_k)
        rep["regime"] = warm
        rep["neighbors"] = self._knn_neighbors(c.knn_k)
        vts = cap.pop("_votes_deep", None)
        if vts is not None:
            nz = vts.nonzero().flatten()
            rep["deep_votes"] = sorted(
                ((int(t), float(vts[t])) for t in nz.tolist()),
                key=lambda kv: -kv[1])[:8]
        return rep

    def _explain_cells_cpu(self, keys: dict) -> list[dict]:
        """Cells that fired this context (host tables): channel name,
        raw key, cell total, top followers."""
        c = self.config
        end = self._count - 1
        out = []

        def emit(ch, key, cell, tot, dist=None):
            fl = sorted(cell.items(), key=lambda kv: -kv[1])[:8]
            e = {"channel": ch, "key": int(key),
                 "total": float(tot), "followers": fl}
            if dist is not None:
                e["dist"] = dist
            out.append(e)

        for k, key in keys.items():
            cell = self._tables[k].get(key)
            if cell:
                emit(f"o{k}", key, cell,
                     self._totals[k].get(key, 0.0))
        for j, taps in enumerate(c.sparse_taps):
            key = self._gap_key(taps, end)
            cell = self._ptabs[taps].get(key) if key else None
            if cell:
                emit(f"p{j}", key, cell,
                     self._ptots[taps].get(key, 0.0))
        for k in c.class_orders:
            key = self._class_key(k, end)
            cell = self._gtabs[k].get(key) if key else None
            if cell:
                emit(f"g{k}", key, cell,
                     self._gtots[k].get(key, 0.0))
        for j, w in enumerate(c.word_orders):
            key = self._word_key(w, end)
            cell = self._wtabs[j].get(key) if key else None
            if cell:
                emit(f"w{j}", key, cell,
                     self._wtots[j].get(key, 0.0))
        for j, pat in enumerate(c.hybrid_taps):
            key = self._hybrid_key(pat, end)
            cell = self._htabs[j].get(key) if key else None
            if cell:
                emit(f"h{j}", key, cell,
                     self._htots[j].get(key, 0.0))
        if c.gcell:
            code = self._gcell_code(self._c, self._c2, end)
            for cc, dist in self._gcell_neighbors(code):
                cell = self._gc.get(cc)
                if cell:
                    emit("gcell", cc, cell, self._gct.get(cc, 0.0),
                         dist)
        return out

    def _explain_cells_gpu(self, slot, found, live_k) -> list[dict]:
        """Order cells that fired this context, read from the GPU
        table (raw key + followers)."""
        gt = self._gt
        out = []
        for i, k in enumerate(self.config.orders):
            if not bool(found[i] and live_k[i]):
                continue
            s = int(slot[i])
            toks = gt.ttoks[i, s].tolist()
            cnts = gt.tcnt[i, s].tolist()
            fl = sorted(((t, v) for t, v in zip(toks, cnts)
                         if t >= 0), key=lambda kv: -kv[1])[:8]
            out.append({"channel": f"o{k}",
                        "key": int(gt.traw[i, s]),
                        "total": float(gt.ttot[i, s]),
                        "followers": fl})
        c = self.config
        if c.gcell:
            code = self._gcell_code(self._c, self._c2,
                                    self._count - 1)
            for cc, dist in self._gcell_neighbors(code):
                cell = self._gc.get(cc)
                if cell:
                    fl = sorted(cell.items(), key=lambda kv: -kv[1])[:8]
                    out.append({"channel": "gcell", "key": int(cc),
                                "dist": dist,
                                "total": float(self._gct.get(cc, 0.0)),
                                "followers": fl})
        return out

    def _explain_report(self, logits: torch.Tensor, cap: dict,
                        cells: list, top_k: int) -> dict:
        """Shared report builder: top-k tokens, each with the
        per-channel logit contributions that produced them."""
        lg = logits.float().reshape(-1)
        probs = torch.softmax(lg, -1)
        k = min(top_k, lg.numel())
        top = torch.topk(lg, k)
        out_top = []
        for t, lv in zip(top.indices.tolist(), top.values.tolist()):
            contrib = {}
            for ch, d in cap.items():
                if ch.startswith("_"):
                    continue
                if isinstance(d, dict):
                    v = d.get(t, 0.0)
                else:
                    v = float(d.reshape(-1)[t])
                if v:
                    contrib[ch] = round(v, 6)
            out_top.append({"tok": int(t),
                            "logit": round(float(lv), 6),
                            "prob": round(float(probs[t]), 6),
                            "channels": contrib})
        return {"end": int(self._count - 1), "regime": 0,
                "top": out_top, "cells": cells,
                "weights": self.channel_weights()}

    def format_explain(self, rep: dict) -> str:
        """Human-readable rendering of an explain() report."""
        lines = [f"ctx_end={rep['end']}  regime="
                 f"{'warm' if rep['regime'] else 'thin'}"]
        for e in rep["top"]:
            lines.append(f"  tok={e['tok']:>6}  p={e['prob']:.4f}  "
                         f"logit={e['logit']:+.3f}")
            for ch, v in sorted(e["channels"].items(),
                                key=lambda kv: -abs(kv[1])):
                w = rep["weights"].get(ch, 0.0)
                lines.append(f"      {ch:<8} {v:+.4f}   (w={w:.3f})")
        if rep.get("cells"):
            lines.append("cells fired:")
            for cl in rep["cells"]:
                fl = ", ".join(f"{t}:{v:.1f}"
                               for t, v in cl["followers"][:5])
                lines.append(f"  {cl['channel']:<6} key="
                             f"{cl['key']}  tot={cl['total']:.1f}  "
                             f"[{fl}]")
        if rep.get("neighbors"):
            lines.append("knn neighbors: " + ", ".join(
                f"r{r}@{p}(sim={s:.3f},succ={su})"
                for r, s, su, p in rep["neighbors"][:5]))
        return "\n".join(lines)

    def inspect_cell(self, channel: str, key: int) -> dict | None:
        """Dump one memory cell: {channel, key, total, followers}.
        ``channel`` is the Hedge name - 'o<k>' order, 'p<j>' sparse
        tap, 'g<k>' class, 'w<j>' word, 'h<j>' hybrid.  Returns None
        when no cell exists at that key."""
        c = self.config
        if channel == "gcell":
            cell = self._gc.get(key)
            if not cell:
                return None
            return {"channel": channel, "key": key,
                    "total": self._gct.get(key, 0.0),
                    "followers": sorted(cell.items(),
                                        key=lambda kv: -kv[1])}
        if len(channel) < 2 or channel[0] not in "opgwh" \
                or not channel[1:].lstrip("-").isdigit():
            return None
        fam = channel[0]
        j = int(channel[1:])
        if fam == "o":
            if j not in c.orders:
                return None
            if self._gpu:
                toks, cnts, tot = self._gt.read_row(j, key)
                fl = sorted(((int(t), float(v))
                             for t, v in zip(toks.tolist(),
                                             cnts.tolist())
                             if t >= 0), key=lambda kv: -kv[1])
                if not fl and tot <= 0:
                    return None
                return {"channel": channel, "key": key,
                        "total": tot, "followers": fl}
            cell = self._tables[j].get(key)
            if not cell:
                return None
            return {"channel": channel, "key": key,
                    "total": self._totals[j].get(key, 0.0),
                    "followers": sorted(
                        cell.items(), key=lambda kv: -kv[1])}
        tabs = {"p": (self._ptabs, self._ptots, c.sparse_taps),
                "g": (self._gtabs, self._gtots, c.class_orders),
                "w": (self._wtabs, self._wtots, c.word_orders),
                "h": (self._htabs, self._htots, c.hybrid_taps)}
        ent = tabs.get(fam)
        if ent is None or j >= len(ent[2]):
            return None
        tabkey = ent[2][j]
        cell = ent[0][tabkey].get(key)
        if not cell:
            return None
        return {"channel": channel, "key": key,
                "total": ent[1][tabkey].get(key, 0.0),
                "followers": sorted(cell.items(),
                                    key=lambda kv: -kv[1])}

    def _knn_neighbors(self, k: int) -> list:
        """Top-k live fingerprint neighbours of the current context:
        (row, cosine, successor, ctx_pos).  Shared by explain on both
        backends - the bank tensors live on self._dev either way."""
        kb = self._knn_rows()
        if not self.config.use_knn or kb <= 0:
            return []
        qk = self._c @ self._Pcsem
        qk = qk / (qk.norm() + 1e-8)
        sims = self._knn_fp[:kb].float() @ qk.float()
        sims = torch.where(self._knn_pos[:kb] >= 0, sims,
                           torch.full_like(sims, -1e9))
        top = torch.topk(sims, min(k, kb))
        return [(int(i), float(s), int(self._knn_succ[i]),
                 int(self._knn_pos[i]))
                for s, i in zip(top.values.tolist(),
                                top.indices.tolist())
                if s > self.config.knn_min_sim]

    def forward(self, idx: torch.Tensor, targets=None,
                past_key_values=None, use_cache: bool = False,
                return_hidden: bool = False, **kw):
        """Engine contract: returns (logits, loss_or_None, None).

        Each input token is observed (live-learned when ``learning`` and
        ``learn_prefill`` allow) then used as context for the
        next-position logits - standard causal alignment.
        """
        if isinstance(idx, torch.Tensor):
            rows = idx.tolist()
        else:
            rows = [list(idx)]
        all_logits = []
        losses = []
        tgt_rows = (targets.tolist() if isinstance(targets, torch.Tensor)
                    else None)
        for b, row in enumerate(rows):
            tgts = tgt_rows[b] if tgt_rows else None
            # learn_prefill=False: bulk forwards are observe-only;
            # single-token forwards (engine decode steps) still learn.
            # Scoped onto self.learning so inner gates (_push's epi/knn
            # writes, _learn_token) see the same effective flag.
            prev_learn = self.learning
            self.learning = bool(
                prev_learn
                and (self.config.learn_prefill or len(row) <= 1))
            try:
                if self._gpu and self.learning and row:
                # one device pass + ONE D2H readback for the whole row
                    if len(row) == 1:
                        # decode-time step: skip the seq-tensor allocs
                        outs_t = self._fused_step(int(row[0]))[None]
                    else:
                        outs_t = self._fused_seq([int(x) for x in row])
                    if tgts is not None:
                        for i in range(len(row) - 1):
                            lp = torch.log_softmax(outs_t[i].float(),
                                                   dim=-1)
                            losses.append(-float(lp[int(tgts[i + 1])]))
                else:
                    outs = []
                    for i, x in enumerate(row):
                        x = int(x)
                        if self.learning:
                            self._learn_token(x)
                        else:
                            self._push(x)
                        outs.append(self._predict())
                        if tgts is not None and i + 1 < len(tgts):
                            lp = torch.log_softmax(outs[-1].float(),
                                                   dim=-1)
                            losses.append(-float(lp[int(tgts[i + 1])]))
                    outs_t = (torch.stack(outs) if outs else
                              torch.zeros((0, self.vocab_size),
                                          device=self._dev))
            finally:
                self.learning = prev_learn
            all_logits.append(outs_t)
        logits = torch.stack(all_logits)
        if isinstance(idx, torch.Tensor) and idx.device != self._dev:
            logits = logits.to(idx.device)
        loss = (torch.tensor(float(np.mean(losses)))
                if losses else None)
        return logits, loss, None

    def generate(self, prompt_ids, max_new: int = 64,
                 temperature: float = 0.0, top_p: float = 1.0,
                 learn: bool = True, tag: str = "gen",
                 generator: torch.Generator | None = None
                 ) -> list[int]:
        """Autoregressive decode with support-restricted sampling.

        Learns ``prompt_ids`` under ``tag`` (``learn=False`` observes
        only), then emits ``max_new`` tokens.  ``temperature=0`` is
        argmax; >0 samples after top-p truncation and the
        ``support_thresh`` mask - a token must beat ``thresh * p_uni``,
        i.e. some channel must have actually voted for it, which keeps
        temperature sampling off the unigram escape floor in thin
        contexts (measured: plain temp .8 derails to char-salad, masked
        stays fluent).  Returns the generated token ids."""
        c = self.config
        prompt = [int(t) for t in prompt_ids]
        if prompt:
            self.ingest(prompt, tag=tag, learn=learn)
        prev_tag = self.tag
        self.tag = tag
        V = self.vocab_size
        out: list[int] = []
        try:
            for _ in range(max_new):
                lg = torch.as_tensor(self._predict()).float().clone()
                if temperature <= 0:
                    x = int(lg.argmax())
                else:
                    lg = lg / max(temperature, 1e-5)
                    if top_p < 1.0:
                        srt, si = lg.sort(descending=True)
                        cum = torch.softmax(srt, -1).cumsum(-1)
                        cut = cum > top_p
                        cut[0] = False
                        srt[cut] = float("-inf")
                        lg = torch.empty_like(lg).scatter_(0, si, srt)
                    if c.support_thresh > 0:
                        denom = self._total + c.uni_alpha * V
                        pu = torch.from_numpy(self._uni + c.uni_alpha) \
                            .float().to(lg.device) / denom
                        supp = torch.softmax(lg, -1) \
                            > c.support_thresh * pu
                        if bool(supp.any()):
                            lg = torch.where(
                                supp, lg,
                                torch.full_like(lg, float("-inf")))
                    # host-side multinomial keeps a user-passed
                    # CPU generator valid on both backends
                    probs = torch.softmax(lg, -1).clamp(min=1e-10)
                    probs = torch.nan_to_num(probs, nan=1.0 / V).cpu()
                    x = int(torch.multinomial(probs, 1,
                                              generator=generator)[0])
                out.append(x)
                # decode-step learn path (same as forward's len-1 rows):
                # fused step on GPU, token learn on CPU - bulk ingest
                # is for multi-token prompts, not per-step decode
                if learn:
                    self._learn_token(x)
                else:
                    self._push(x)
        finally:
            self.tag = prev_tag
        return out

    # -- revert / audit --------------------------------------------------
