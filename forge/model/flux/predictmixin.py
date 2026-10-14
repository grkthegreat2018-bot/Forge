"""Prediction paths for FluxLM (CPU + CUDA-primary)."""
from __future__ import annotations

import json
import math
import os
import re
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

    def _esc_alpha(self, u: float) -> float:
        """Escape pseudo-mass for a count cell.  ``ppm_escape_c``
        (PPM method C) substitutes the cell's distinct-follower count
        ``u`` for the constant ``alpha`` - data-derived escape mass."""
        return float(u) if self.config.ppm_escape_c else \
            self.config.alpha

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
            a_b = self._esc_alpha(len(cell))
            return conf * (cell.get(x, 0.0)
                           + a_b * self._uni_p(x, denom)) \
                / (btot + a_b), True
        if ch in ("gcell", "gsyn", "gsd", "dnet"):
            # fuzzy lift channels: real probs so hedge can punish them.
            # prob = exp(channel log-softmax at x) - one readout row.
            # Context features memoized per position (_pfq) so gcell +
            # gsyn + gsd + dnet share one build.
            memo = getattr(self, "_pfq", None)
            if memo is None or memo[0] != self._count:
                fx = self._gcell_feats(self._c, self._c2,
                                       self._count - 1)
                gk = self._gsyn_keys(fx)
                gc = self._gcell_code_from_feats(fx)
                memo = (self._count, gc, gk)
                self._pfq = memo
            _cnt, gc, gk = memo
            if ch == "gcell":
                num = den = 0.0
                for cc, dist in self._gcell_neighbors(gc):
                    cell = self._gc.get(cc)
                    if cell:
                        w_ = c.gcell_neighbor_decay ** dist
                        num += w_ * cell.get(x, 0.0)
                        den += w_ * self._gct.get(cc, 0.0)
                if den <= 0:
                    return self._uni_p(x, denom), False
                return (num + c.alpha * self._uni_p(x, denom) * den) \
                    / (den + c.alpha), True
            if ch == "gsyn":
                if float(self._S.norm()) < 1e-8:
                    return self._uni_p(x, denom), False
                return float(math.exp(
                    float(self._gsyn_logits(gk)[x]))), True
            if ch == "gsd":
                if self._Wd2 is None or float(self._Wd2.norm()) < 1e-8:
                    return self._uni_p(x, denom), False
                return float(math.exp(
                    float(self._gsd_logits(gk)[x]))), True
            if not any(float(s_.norm()) > 1e-8 for s_ in self._DS):
                return self._uni_p(x, denom), False
            return float(math.exp(
                float(self._dnet_logits(gk)[x]))), True
        if ch.startswith("p"):
            # sparse-tap channel "p<j>" - gapped-context cell
            j = int(ch[1:])
            taps = c.sparse_taps[j]
            key = (pkeys or {}).get(j, -1)
            cell = self._ptabs[taps].get(key)
            if not cell:
                return self._uni_p(x, denom), False
            tot = self._ptots[taps].get(key, 0.0)
            a_c = self._esc_alpha(len(cell))
            return (cell.get(x, 0.0) + a_c
                    * self._uni_p(x, denom)) / (tot + a_c), tot > 0
        if ch.startswith("g"):
            # class-order channel "g<k>" - contiguous suffix over byte
            # classes (coarse context - generalizes on novel text)
            k = int(ch[1:])
            key = (gkeys or {}).get(k, -1)
            cell = self._gtabs[k].get(key)
            if not cell:
                return self._uni_p(x, denom), False
            tot = self._gtots[k].get(key, 0.0)
            a_c = self._esc_alpha(len(cell))
            return (cell.get(x, 0.0) + a_c
                    * self._uni_p(x, denom)) / (tot + a_c), tot > 0
        if ch.startswith("w"):
            # word-order channel "w<j>" - key over last-w word hashes +
            # partial-word bucket (ring-scan derived, no state)
            j = int(ch[1:])
            key = (wkeys or {}).get(j, -1)
            cell = self._wtabs[j].get(key)
            if not cell:
                return self._uni_p(x, denom), False
            tot = self._wtots[j].get(key, 0.0)
            a_c = self._esc_alpha(len(cell))
            return (cell.get(x, 0.0) + a_c
                    * self._uni_p(x, denom)) / (tot + a_c), tot > 0
        if ch.startswith("h"):
            # hybrid channel "h<j>" - key over a (offset, is_raw)
            # pattern mixing byte classes and raw bytes
            j = int(ch[1:])
            key = (hkeys or {}).get(j, -1)
            cell = self._htabs[j].get(key)
            if not cell:
                return self._uni_p(x, denom), False
            tot = self._htots[j].get(key, 0.0)
            a_c = self._esc_alpha(len(cell))
            return (cell.get(x, 0.0) + a_c
                    * self._uni_p(x, denom)) / (tot + a_c), tot > 0
        # order channel "o<k>"
        k = int(ch[1:])
        key = keys.get(k, -1)
        cell = self._tables[k].get(key)
        if not cell:
            return self._uni_p(x, denom), False
        tot = self._totals[k].get(key, 0.0)
        a_c = self._esc_alpha(len(cell))
        return (cell.get(x, 0.0) + a_c * self._uni_p(x, denom)) \
            / (tot + a_c), tot > 0

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
        q = self._knn_fp_ctx(self._count - 1,
                             episode=c.knn_episode
                             ).to(self._knn_fp.dtype)
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
        K = int(c.knn_span)
        for s, i in zip(top.values.tolist(), top.indices.tolist()):
            if s <= c.knn_min_sim:
                break
            hit_rows.append(i)
            w0 = max(s, 0.0) ** c.knn_gamma
            if K > 0:
                # span voting: tokens within +/-K of the matched
                # stream position vote (passage->answer extraction)
                p = int(self._knn_pos[i])
                span_ids, span_w = [], []
                for j in range(-K, K + 1):
                    pj = p + j
                    if pj < 0 or pj >= self._count:
                        continue
                    t = int(self._ring[pj % c.ring_capacity])
                    if 0 <= t < self.vocab_size:
                        span_ids.append(t)
                        span_w.append(c.knn_span_decay ** abs(j))
                if c.knn_subq:
                    Wq = int(c.knn_window) or 16
                    q0 = (self._prompt_start if c.knn_episode
                          else max(0, self._count - Wq))
                    qset = {int(self._ring[q % c.ring_capacity])
                            for q in range(q0,
                                           min(self._count,
                                               q0 + Wq))}
                    span_w = [w if t not in qset else 0.0
                              for t, w in zip(span_ids, span_w)]
                if c.knn_span_idf > 0 and span_ids:
                    iw = self._knn_tok_w(torch.as_tensor(
                        span_ids, device=self._dev))
                    iw = (iw / iw.mean().clamp(min=1e-6)
                          ) ** float(c.knn_span_idf)
                    span_w = [w * float(x)
                              for w, x in zip(span_w, iw.tolist())]
                if c.knn_cover > 0:
                    R = int(c.knn_cover)
                    W = int(c.knn_window) or 16
                    qt = torch.as_tensor(
                        [self._ring[q % c.ring_capacity] for q in
                         range(max(0, self._count - W),
                               self._count)],
                        device=self._dev).long().clamp(
                            0, self.vocab_size - 1)
                    qw = self._knn_tok_w(qt)
                    sel = qw.topk(min(8, int(qw.numel()))).indices
                    ct, cw = qt[sel], qw[sel]
                    cw_l, ct_l = cw.tolist(), ct.tolist()
                    cov = 0.0
                    for qj in range(-R, R + 1):
                        pj2 = p + qj
                        if 0 <= pj2 < self._count:
                            t2 = int(
                                self._ring[pj2 % c.ring_capacity])
                            for ctk, cwk in zip(ct_l, cw_l):
                                if t2 == ctk:
                                    cov += cwk
                    cover = cov / max(sum(cw_l), 1e-8)
                    span_w = [w * (1.0 + float(c.knn_cover_w)
                                   * cover) for w in span_w]
                if c.knn_dedup:
                    # a token counts at most its strongest position in
                    # this row's span - support-counting across rows
                    row_v: dict[int, float] = {}
                    for t, w in zip(span_ids, span_w):
                        ww = w0 * w
                        if ww > row_v.get(t, 0.0):
                            row_v[t] = ww
                    for t, w in row_v.items():
                        votes[t] = votes.get(t, 0.0) + w
                else:
                    for t, w in zip(span_ids, span_w):
                        votes[t] = votes.get(t, 0.0) + w0 * w
            else:
                t = int(self._knn_succ[i])
                if t >= 0:
                    votes[t] = votes.get(t, 0.0) + w0
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
    def _knn_votes_dev(self, kb: int,
                       end: torch.Tensor | None = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k fingerprint-similar past contexts vote their stored
        successors - approximate-match episodic memory.  Only the live
        ring prefix is scanned (bucketed - static shape inside a
        captured graph; unwritten padding stays pos=-1).
        dynamo-disabled for the same topk-tie reason as _topk_dev."""
        c = self.config
        V = self.vocab_size
        qk = self._knn_fp_ctx(
            self._count - 1 if end is None else end,
            episode=c.knn_episode
        ).to(self._knn_fp.dtype)
        ksims = (self._knn_fp[:kb] @ qk).float()
        ksims = torch.where(
            self._knn_pos[:kb] >= 0, ksims,
            torch.full((), -1e9, device=self._dev))
        if (c.knn_drop_qa > 0 and end is not None
                and getattr(self, "_ringg", None) is not None):
            # mask rows sitting inside a "Q:... A:" line: their forward
            # ring window contains the same answer-scaffold pair that
            # ends the current prompt - they retrieve format-twin
            # questions, not evidence
            e = end.reshape(-1)[-1]
            D = int(c.knn_drop_qa)
            kp = self._knn_pos[:kb].long()
            s0 = self._ringg.index_select(
                0, (e - 1).clamp(min=0).reshape(1)
                % c.ring_capacity)[0].long()
            s1 = self._ringg.index_select(
                0, (e % c.ring_capacity).reshape(1))[0].long()
            wi = kp[:, None] + torch.arange(
                1, D + 1, device=self._dev)[None, :]
            wvld = (kp >= 0)[:, None] & (wi <= e - 2) & (wi >= 0)
            wt = self._ringg.index_select(
                0, (wi.clamp(min=0) % c.ring_capacity
                    ).reshape(-1)).reshape(wi.shape).long()
            isqa = ((wt[:, :-1] == s0) & (wt[:, 1:] == s1)
                    & wvld[:, :-1]).any(1)
            ksims = torch.where(isqa, torch.full_like(ksims, -1e9),
                                ksims)
        # two-stage retrieval: when coverage is on, pool 4x the rows by
        # sim, rerank by query-content coverage, keep top knn_k - the
        # answer-bearing passage ranks below format twins on raw sim
        # but separates cleanly once coverage reweights it
        ktop_n = min(c.knn_k, kb)
        cover_on = (c.knn_cover > 0
                    and (c.knn_span > 0 or c.knn_twin > 0
                         or c.knn_spread > 0)
                    and end is not None
                    and getattr(self, "_ringg", None) is not None)
        pool_n = min(max(int(c.knn_pool), ktop_n), kb) if cover_on \
            else ktop_n
        top = torch.topk(ksims, pool_n)
        keep = top.values > c.knn_min_sim
        # retrieval utility: neighbours that cleared the vote floor
        # earn hits (feeds knn_utility_evict scoring) - branchless,
        # index_add_ of keep-as-int is capture-safe inside graphs
        self._knn_hits.index_add_(
            0, top.indices, keep.to(torch.int32))
        wv = (top.values.clamp(min=0.0) ** c.knn_gamma) * keep
        rows_i = top.indices                    # bank-row indices
        pos = self._knn_pos.index_select(0, rows_i).long()
        need_ct = ((cover_on or c.knn_anchor > 0
                    or c.knn_spread > 0)
                   and end is not None
                   and getattr(self, "_ringg", None) is not None)
        ct = cw = None
        if need_ct:
            # the QUERY's high-idf content tokens - shared by coverage
            # reranking and content-anchored span voting.  A row/token
            # matters to the extent that the question's rare words sit
            # near it.
            e = end.reshape(-1)[-1]
            W = int(c.knn_window) or 16
            e0 = (self._prompt_g.reshape(-1)[-1]
                  if c.knn_episode and getattr(
                      self, "_prompt_g", None) is not None
                  else (torch.as_tensor(
                      self._prompt_start, device=self._dev)
                      if c.knn_episode
                      else (e - W + 1).clamp(min=0)))
            qi = torch.minimum(
                (e0 + torch.arange(W, device=self._dev)
                 ).clamp(min=0), e)
            qt = self._ringg.index_select(
                0, qi % c.ring_capacity).long().clamp(0, V - 1)
            qw = self._knn_tok_w(qt)
            qw = torch.where(qi >= e0, qw,
                             torch.zeros((), device=self._dev))
            sel = qw.topk(min(16, int(qw.numel()))).indices
            ct, cw = qt[sel], qw[sel]
            cw = torch.where(cw >= 0.6 * cw.max().clamp(min=1e-8),
                             cw, torch.zeros_like(cw))
        if cover_on:
            # coverage rerank on the pool: fraction of the QUERY's
            # content-token idf present in each row's +/-R ring window.
            # The sentence defining the answer carries the question's
            # rare words; format-twin questions and topically-similar
            # passages carry less.
            R = int(c.knn_cover)
            wi2 = pos[:, None] + torch.arange(
                -R, R + 1, device=self._dev)[None, :]
            live2 = (wi2 >= 0) & (wi2 <= e)
            wt2 = self._ringg.index_select(
                0, (wi2.clamp(min=0) % c.ring_capacity
                    ).reshape(-1)).reshape(wi2.shape
                                          ).long().clamp(0, V - 1)
            hit = (wt2[:, :, None] == ct[None, None, :]
                   ) & live2[:, :, None]
            # per-token presence (dedupe): a content token counts ONCE
            # no matter how many window positions contain it
            cov = (hit.any(dim=1).float() * cw[None, :]).sum(
                dim=1) / cw.sum().clamp(min=1e-8)
            score = wv * (1.0 + float(c.knn_cover_w) * cov)
            pick = torch.topk(score, ktop_n).indices
            wv = wv.index_select(0, pick)
            rows_i = rows_i.index_select(0, pick)
            pos = pos.index_select(0, pick)
        if c.knn_clus > 0 and pos.numel() > 1:
            # the true passage retrieves at MANY adjacent positions;
            # false matches are scattered singletons - boost clustered
            # rows by neighbour count within knn_span
            Kc = max(int(c.knn_span), 1)
            deg = ((pos[:, None] - pos[None, :]).abs()
                   <= Kc).float().sum(1) - 1.0
            wv = wv * (1.0 + float(c.knn_clus) * deg)
        twin_votes = None
        if (c.knn_twin > 0 and end is not None
                and getattr(self, "_ringg", None) is not None):
            # twin-question answer adoption: a pool row sitting inside
            # a trained "Q:.. A:<ans>" line has the prompt's scaffold
            # pair in its forward window - the tokens after it are ITS
            # answer.  Near-duplicate held questions read the matching
            # trained answer directly (generalized recall).  Runs on
            # the coverage-reranked selection so the adopted answer
            # comes from a row that also carries the question's
            # content, not merely the format.
            e = end.reshape(-1)[-1]
            Dt = max(int(c.knn_drop_qa), 6)
            s0 = self._ringg.index_select(
                0, (e - 1).clamp(min=0).reshape(1)
                % c.ring_capacity)[0].long()
            s1 = self._ringg.index_select(
                0, (e % c.ring_capacity).reshape(1))[0].long()
            wi = pos[:, None] + torch.arange(
                1, Dt + 1, device=self._dev)[None, :]
            wvld = (pos >= 0)[:, None] & (wi <= e - 2) & (wi >= 0)
            wt = self._ringg.index_select(
                0, (wi.clamp(min=0) % c.ring_capacity
                    ).reshape(-1)).reshape(wi.shape).long()
            hitq = (wt[:, :-1] == s0) & (wt[:, 1:] == s1) \
                & wvld[:, :-1]
            hasq = hitq.any(1)
            ans0 = pos + hitq.float().argmax(1) + 2
            twin_votes = torch.zeros(V, device=self._dev)
            for d in range(3):
                ap = ans0 + d
                al = (ap <= e - 2) & hasq
                at = self._ringg.index_select(
                    0, (ap.clamp(min=0) % c.ring_capacity)
                    ).long().clamp(0, V - 1)
                twin_votes.index_add_(
                    0, at, wv * float(c.knn_twin)
                    * (0.5 ** d) * al.float())
        if (c.knn_spread > 0 and ct is not None
                and int(ct.numel()) > 0
                and getattr(self, "_ringg", None) is not None):
            # spreading-activation extraction: every stream position
            # gets a field value = idf-weighted proximity to the
            # question's content tokens; a token's vote = sum of field
            # over its occurrences.  Corpus-wide, no top-k dependence.
            cap = c.ring_capacity
            ring_all = self._ringg.long().clamp(0, V - 1)
            act = ((ring_all[:, None] == ct[None, :]).float()
                   * cw[None, :]).sum(1)
            hi_s = (end.reshape(-1)[-1] if end is not None
                    else torch.as_tensor(self._count - 1,
                                         device=self._dev))
            live_ar = torch.arange(cap, device=self._dev)
            pre_q = live_ar < e0
            # the query's own positions carry every content token -
            # leaving them live makes the question self-match and drown
            # the evidence clusters
            act = act * pre_q.float()
            Ws = int(c.knn_spread_w)
            if c.knn_spread_doc:
                # doc-bag readout: every position inherits the max
                # content mass in its +/-Ws neighbourhood - the answer
                # needn't sit near the defining phrase, only inside
                # the content-bearing document
                field = torch.nn.functional.max_pool1d(
                    act.view(1, 1, cap), kernel_size=2 * Ws + 1,
                    stride=1, padding=Ws).view(cap)
            else:
                koffs = torch.arange(-Ws, Ws + 1, device=self._dev)
                kern = float(c.knn_spread) ** koffs.abs().float()
                field = torch.nn.functional.conv1d(
                    act.view(1, 1, cap), kern.view(1, 1, -1),
                    padding=Ws).view(cap)
            field = field * (live_ar <= hi_s).float() * pre_q.float()
            if c.knn_spread_pow != 1.0:
                field = field ** float(c.knn_spread_pow)
            # vote only from the hottest content-cluster positions:
            # a candidate must physically sit where the question's
            # rare words concentrate - same-passage drift can't vote
            ntop = min(int(c.knn_spread_minhit * 128) or 256, cap)
            th = field.topk(ntop).values[-1]
            field = field * (field >= th).float()
            # index_add_ (atomic order) over the full ring - O(N) but
            # float-order nondeterministic; acceptable for a readout
            # channel (_scatter_sum is O(N^2) at cap scale)
            votes = torch.zeros(V, device=self._dev)
            if c.knn_spread_doc:
                # doc mode scores a token by doc-mass x own rarity:
                # boilerplate in the right doc still loses to rare
                # answer entities
                votes.index_add_(
                    0, ring_all, field * self._knn_tok_w(ring_all))
            else:
                votes.index_add_(0, ring_all, field)
            if c.knn_subq and end is not None:
                e = end.reshape(-1)[-1]
                e0 = (self._prompt_g.reshape(-1)[-1]
                      if c.knn_episode and getattr(
                          self, "_prompt_g", None) is not None
                      else (torch.as_tensor(
                          self._prompt_start, device=self._dev)
                          if c.knn_episode
                          else (e - int(c.knn_window or 16) + 1
                                ).clamp(min=0)))
                Wq = int(c.knn_window) or 16
                qids = self._ringg.index_select(
                    0, (torch.minimum(
                        (e0 + torch.arange(Wq, device=self._dev)
                         ).clamp(min=0), e)
                        % c.ring_capacity)).long().clamp(0, V - 1)
                qm = torch.zeros(V, dtype=torch.bool,
                                 device=self._dev)
                qm.index_fill_(0, qids, True)
                votes = votes * (~qm).float()
            uni_t = (self._unig if getattr(
                self, "_unig", None) is not None else
                torch.as_tensor(self._uni, device=self._dev))
            if c.knn_tokmin > 0 or c.knn_tokmax < 1e17:
                votes = votes * ((uni_t >= float(c.knn_tokmin))
                                 & (uni_t <= float(c.knn_tokmax))
                                 ).float()
            if c.knn_spread_norm > 0:
                # mean field per occurrence: 'was' co-locates with
                # everything but dilutes over 10^3 occurrences; the
                # answer's few occurrences all sit near content
                votes = votes / (uni_t + 1.0).clamp(
                    min=1e-6) ** float(c.knn_spread_norm)
            if c.knn_spread_idf > 0:
                wall = self._knn_tok_w(torch.arange(
                    V, device=self._dev))
                votes = votes * wall.clamp(
                    min=1e-8) ** float(c.knn_spread_idf)
            if twin_votes is not None:
                votes = votes + twin_votes
            return votes, votes.sum()
        if c.knn_span > 0 and getattr(self, "_ringg", None) is not None:
            # span voting: neighbours retrieve CONTEXTS - every token
            # within +/-K stream positions of a match votes, so the
            # answer needn't be the direct successor (extractive QA:
            # retrieve the passage, read off the entity).  end bounds
            # the span to observed positions; all device ops.
            # anchored voting widens the gather to the coverage radius
            # so a content hit at R> K still anchors a vote.
            K = max(int(c.knn_span),
                    int(c.knn_cover) if c.knn_anchor > 0 else 0)
            offs = torch.arange(-K, K + 1, device=self._dev)
            pj = pos[:, None] + offs[None, :]
            hi = (end.reshape(-1)[-1] if end is not None
                  else pos.new_full((), self._count - 1))
            live = (pj >= 0) & (pj <= hi)
            pj = pj.clamp(min=0) % c.ring_capacity
            toks = self._ringg.index_select(
                0, pj.reshape(-1)).reshape(pj.shape
                                          ).long().clamp(0, V - 1)
            dw0 = (float(c.knn_span_decay)
                   ** offs.abs().float())[None, :]
            if (c.knn_anchor > 0 and ct is not None
                    and int(ct.numel()) > 0):
                # content-anchored vote: the answer sits ADJACENT to
                # the question's content words inside the passage, not
                # at the matched-row centre.  Weight each span position
                # by decayed proximity to the nearest content-token hit
                # (idf-weighted hits anchor harder).  Center decay is
                # kept as a floor so zero-hit rows still vote weakly.
                hmask = ((toks[:, :, None] == ct[None, None, :]).float()
                         * cw[None, None, :]).sum(-1)
                d2 = (offs[:, None] - offs[None, :]).abs()
                M = float(c.knn_anchor) ** d2.float()
                field = hmask @ M.T
                field = field / field.amax(
                    dim=1, keepdim=True).clamp(min=1e-8)
                dw = torch.maximum(field, 0.05 * dw0)
            else:
                dw = dw0
            wij = wv[:, None] * dw * live
            if c.knn_subq and end is not None:
                # the answer is the token the question does NOT
                # contain - mask every prompt token out of the span
                # vote (without this, votes re-emit the question's
                # own content words)
                e = end.reshape(-1)[-1]
                e0 = (self._prompt_g.reshape(-1)[-1]
                      if c.knn_episode and getattr(
                          self, "_prompt_g", None) is not None
                      else (torch.as_tensor(
                          self._prompt_start, device=self._dev)
                          if c.knn_episode
                          else (e - int(c.knn_window or 16) + 1
                                ).clamp(min=0)))
                Wq = int(c.knn_window) or 16
                qids = self._ringg.index_select(
                    0, (torch.minimum(
                        (e0 + torch.arange(Wq, device=self._dev)
                         ).clamp(min=0), e)
                        % c.ring_capacity)).long().clamp(0, V - 1)
                qm = torch.zeros(V, dtype=torch.bool,
                                 device=self._dev)
                qm.index_fill_(0, qids, True)
                wij = wij * (~qm.index_select(
                    0, toks.reshape(-1))).reshape(toks.shape).float()
            if c.knn_span_idf > 0:
                iw = self._knn_tok_w(toks.reshape(-1))
                iw = (iw / iw.mean().clamp(min=1e-6)
                      ) ** float(c.knn_span_idf)
                wij = wij * iw.reshape(toks.shape)
            if c.knn_tokmin > 0 or c.knn_tokmax < 1e17:
                # entity band: boilerplate words and one-off junk can't
                # be answers - gate candidates by corpus frequency
                uni_t = (self._unig if getattr(
                    self, "_unig", None) is not None else
                    torch.as_tensor(self._uni, device=self._dev))
                cnt = uni_t.index_select(
                    0, toks.reshape(-1)).reshape(toks.shape)
                band = (cnt >= float(c.knn_tokmin)) & (
                    cnt <= float(c.knn_tokmax))
                wij = wij * band.float()
            if c.knn_dedup:
                # a token contributes at most its strongest position per
                # row - support-counting: entities spread across many
                # retrieved rows beat words repeated inside one span
                contrib = torch.zeros(
                    (toks.shape[0], V), device=self._dev)
                contrib.scatter_reduce_(
                    1, toks, wij, reduce="amax", include_self=False)
                kv = contrib.sum(0)
                if c.knn_minsup > 1:
                    sup = (contrib > 0).sum(0)
                    kv = kv * (sup >= int(c.knn_minsup)).float()
            else:
                kv = _scatter_sum(toks.reshape(-1), wij.reshape(-1), V)
            if twin_votes is not None:
                kv = kv + twin_votes
            return kv, kv.sum()
        ksucc = self._knn_succ.index_select(
            0, rows_i).long().clamp(0, V - 1)
        # deterministic accumulation - neighbours can share a
        # successor, and index_add_ atomics made logits
        # timing-dependent
        kv = _scatter_sum(ksucc, wv, V)
        if twin_votes is not None:
            kv = kv + twin_votes
        return kv, kv.sum()

    @torch._dynamo.disable
    def _knn_reward_dev(self, kb: int, fp_store: torch.Tensor,
                        x: torch.Tensor,
                        end: torch.Tensor | None = None
                        ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if c.knn_span > 0 and getattr(self, "_ringg", None) is not None:
            K = int(c.knn_span)
            pos = self._knn_pos.index_select(0, ktop.indices).long()
            offs = torch.arange(-K, K + 1, device=self._dev)
            pj = pos[:, None] + offs[None, :]
            hi = (end.reshape(-1)[-1] if end is not None
                  else pos.new_full((), self._count - 1))
            live = (pj >= 0) & (pj <= hi)
            pj = pj.clamp(min=0) % c.ring_capacity
            toks = self._ringg.index_select(
                0, pj.reshape(-1)).long().reshape(pj.shape)
            dw = (float(c.knn_span_decay)
                  ** offs.abs().float())[None, :]
            wij = kwv[:, None] * dw * live
            if c.knn_span_idf > 0:
                iw = self._knn_tok_w(toks.reshape(-1))
                iw = (iw / iw.mean().clamp(min=1e-6)
                      ) ** float(c.knn_span_idf)
                wij = wij * iw.reshape(toks.shape)
            kvt = wij.sum().double()
            kvx = (wij * (toks == x)).sum().double()
            return kvt, kvx
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
        # adaptive-read flag for the NEXT step: dominant follower share
        # on any evidenced order cell - same cells the order channel
        # already gathered, so the flag costs ~4 kernels
        share = rows_c.max(dim=1).values / tots.clamp(min=1e-9)
        # warm flag = any order cell has real evidence this context -
        # feeds cond_hedge regime routing AND the host sparse-tap lift
        # gate (sparse votes are thin-context fallback only).  With
        # evidence_share>0 the mass must CONCENTRATE: a huge diffuse
        # cell (shared '... A:' trigram holding thousands of answers)
        # is not warmth - thin-context channels stay in charge there.
        if c.thin_share > 0:
            warm_any = (hit & (tots > c.thin_tot)
                        & (share >= c.thin_share)).any()
        elif c.evidence_share > 0:
            warm_any = (hit & (tots * share.clamp(min=0.0)
                              ** c.evidence_share
                              > c.thin_tot)).any()
        else:
            warm_any = (hit & (tots > c.thin_tot)).any()
        self._dthin_g.copy_(warm_any.reshape(1).long())
        if c.cond_hedge:
            self._regime_g.fill_(warm_any.long())
            wrow = self._wgR.index_select(
                0, self._regime_g.view(1)).squeeze(0)
        else:
            wrow = self._wg
        conf_ok = hit & (tots >= c.adaptive_read_min_tot) \
            & (share >= c.adaptive_read_conf)
        self._dconf_g.copy_(conf_ok.any().reshape(1).long())
        wo = wrow[:O].float()[:, None]             # [O,1]
        if c.evidence_lift:
            # effective weight = hedge prior + evidence confidence -
            # KN-style interpolation; a massive cell speaks for
            # itself.  backoff_order_sharp > 0 shrinks the kappa for
            # longer contexts (specific hits need less mass)
            if c.backoff_order_sharp > 0:
                # cached device tensor: torch.as_tensor(list) inside
                # this function does an H2D copy, which is illegal
                # during CUDA-graph capture - the warmup calls build
                # it eagerly so replay/capture always hits the cache
                kkey = (c.backoff_kappa, c.backoff_order_sharp,
                        tuple(c.orders))
                if getattr(self, "_kap_key", None) != kkey:
                    o_min = min(c.orders)
                    self._kap_o = c.backoff_kappa * (
                        torch.as_tensor(
                            [o_min / k for k in c.orders],
                            dtype=torch.float32, device=self._dev)
                        ** c.backoff_order_sharp)
                    self._kap_key = kkey
                kap = self._kap_o
            else:
                kap = c.backoff_kappa
            eff = tots
            if c.evidence_share > 0:
                # effective mass concentrates on the mode: a diffuse
                # mega-cell's mass no longer buys free confidence
                eff = tots * (share.clamp(min=0.0)
                              ** c.evidence_share)
            wo = wo + (eff / (eff + kap))[:, None]
        tok_l = rows_t.long().clamp(0, V - 1)
        row_live = (rows_t >= 0) & (rows_t < V) & hit[:, None]
        if c.ppm_escape_c:
            # method-C escape: pseudo-mass = distinct followers per cell
            a_o = row_live.sum(dim=1).clamp(min=1e-9)[:, None].float()
        else:
            a_o = c.alpha              # python float - no H2D in graph
        p = (rows_c + a_o * uni_pv[tok_l]) \
            / (tots[:, None] + a_o)
        contrib = wo * (torch.log(p.clamp(min=1e-12))
                        - log_uni[tok_l])
        if c.ppm_excl:
            # PPM exclusion: a token counted at a deeper evidenced order
            # is masked out of shallower orders' votes.  Owner index =
            # the deepest live order holding the token - one amax
            # scatter (static shapes; mask-indexing's nonzero() D2H
            # sync is illegal under CUDA-graph capture).
            ordx = torch.arange(O, device=self._dev)[:, None].long()
            src = torch.where(row_live, ordx.expand_as(tok_l),
                              tok_l.new_full(tok_l.shape, -1))
            claimed = tok_l.new_full((V,), -1)
            claimed.scatter_reduce_(0, tok_l.reshape(-1),
                                    src.reshape(-1), reduce="amax")
            row_live = row_live & (claimed[tok_l] <= ordx)
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
            eff_k = tot_k
            if c.evidence_share > 0:
                eff_k = tot_k * (
                    (rc_k.max() / tot_k.clamp(min=1e-9))
                    .clamp(min=0.0) ** c.evidence_share)
            conf = eff_k / (eff_k + c.backoff_kappa)
            tl = rt_k.long().clamp(min=0)
            rl = rt_k >= 0
            a_b = (rl.sum().clamp(min=1e-9).float()
                   if c.ppm_escape_c
                   else c.alpha)         # python float - no H2D in
                                         # graph capture
            p_b = conf * (rc_k + a_b * uni_pv[tl]) \
                / (tot_k + a_b)
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
        if c.topic_full:
            # 2-hop association over ALL of vocab: _A[t] = contexts
            # where t appeared vs the content window query - the
            # semantic bridge (question entity -> answer entity) that
            # successor-only channels can't reach.  One [V,192] matvec.
            xq = self._knn_ctx_feat(end, episode=c.knn_episode)
            xn = xq.norm()
            cos_f = (self._A @ xq) / (self._An * xn + 1e-8)
            cos_f = torch.where(self._An > 1e-8, cos_f,
                                torch.zeros((), device=self._dev))
            d_top = torch.where(xn > 1e-8, wt * c.topic_beta * cos_f,
                                torch.zeros((), device=self._dev))
            logits = logits + d_top
            if cap is not None:
                cap["topic"] = d_top
        else:
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
            cf.squeeze() & cok & (rn > 1e-8) & torch.isfinite(rn),
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
            kv, kvt = self._knn_votes_dev(kb, end=end)
        else:
            # uniform tensor signature - see _step_dev note
            kv = torch.zeros(self.vocab_size, dtype=torch.float32,
                             device=self._dev)
            kvt = torch.zeros((), dtype=torch.float32,
                              device=self._dev)
        return self._logits_fast(end, slot, found, live_k,
                                 cok, dok, dk, kv=kv, kvt=kvt)

    # ---------------- channel-evidence capture (open readout) ---------

    def _cap_cell_np(self, cell, tot, wj, ev_mul, denom):
        """Per-cell sparse lift values (mirrors _sparse_lift_vec_dev's
        acc) -> (idx int64[np], v fp64[np])."""
        c = self.config
        if c.evidence_lift:
            wj += ev_mul * tot / (tot + c.backoff_kappa)
        idx = np.fromiter(cell.keys(), dtype=np.int64,
                          count=len(cell))
        cnt = np.fromiter(cell.values(), dtype=np.float64,
                          count=len(cell))
        uni = self._uni[idx]
        a_c = self._esc_alpha(len(cell))
        p = (cnt + a_c * (uni + c.uni_alpha) / denom) / (tot + a_c)
        v = wj * (np.log(np.maximum(p, 1e-12))
                  - (np.log(uni + c.uni_alpha) - math.log(denom)))
        return idx, v

    def _cap_sparse_split(self, warm: int):
        """Per-channel sparse-lift vectors -> (lv[V], flv[V], per{ch:
        (v,idx)}).  Same math as the production merged apply, split so
        each channel's evidence is inspectable (head/knob readout)."""
        c = self.config
        V = self.vocab_size
        end = self._count - 1
        denom = self._total + c.uni_alpha * self.vocab_size
        wsrc = self._wR[warm] if c.cond_hedge else self._w
        lv = np.zeros(V, dtype=np.float64)
        flv = np.zeros(V, dtype=np.float64)
        per = {}

        def put(ch, cell, tot, wj, fuzzy=False):
            idx, v = self._cap_cell_np(cell, tot, wj, 1.0, denom)
            per[ch] = v, idx
            np.add.at(flv if fuzzy else lv, idx, v)

        if c.sparse_taps and not (c.sparse_thin_only and warm):
            for j, taps in enumerate(c.sparse_taps):
                wj = float(wsrc[self._chix[f"p{j}"]])
                if wj <= 1e-4:
                    continue
                key = self._gap_key(taps, end)
                if key is None:
                    continue
                cell = self._ptabs[taps].get(key)
                if cell:
                    put(f"p{j}", cell, self._ptots[taps].get(key, 0.0),
                        wj)
        if c.class_orders and not (c.class_thin_only and warm):
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
                        put(f"g{k}", cell, tot, wk)
        if c.word_orders and not (c.word_thin_only and warm):
            for j, w in enumerate(c.word_orders):
                wj = float(wsrc[self._chix[f"w{j}"]])
                if wj <= 1e-4:
                    continue
                key = self._word_key(w, end)
                if key is None:
                    continue
                cell = self._wtabs[j].get(key)
                if cell:
                    put(f"w{j}", cell, self._wtots[j].get(key, 0.0),
                        wj)
        if c.hybrid_taps and not (c.hybrid_thin_only and warm):
            for j, pat in enumerate(c.hybrid_taps):
                wj = float(wsrc[self._chix[f"h{j}"]])
                if wj <= 1e-4:
                    continue
                key = self._hybrid_key(pat, end)
                if key is None:
                    continue
                cell = self._htabs[j].get(key)
                if cell:
                    put(f"h{j}", cell, self._htots[j].get(key, 0.0),
                        wj)
        if c.gcell and not (c.gcell_thin and warm):
            code = self._gcell_code(self._c, self._c2, end)
            w_rel = len(self.channels) * float(
                wsrc[self._chix["gcell"]])
            gc_v, gc_i = [], []
            for cc, dist in self._gcell_neighbors(code):
                cell = self._gc.get(cc)
                if cell:
                    db = self._gen_bias * c.gcell_neighbor_decay ** dist
                    idx, v = self._cap_cell_np(
                        cell, self._gct.get(cc, 0.0),
                        float(c.gcell_w) * w_rel * db, db, denom)
                    gc_v.append(v)
                    gc_i.append(idx)
                    np.add.at(flv, idx, v)
            if gc_v:
                per["gcell"] = (np.concatenate(gc_v),
                                np.concatenate(gc_i))
        return lv, flv, per

    def predict_cap(self):
        """(logits[V], cap, warm) — the open channel readout.

        ``cap`` records every channel's additive lift under its name
        (dense [V] tensor or sparse dict via ``cap['_sparse_raw']``),
        plus ``cap['_sparse_stats']`` per-channel summary stats.  This
        is the evidence surface a KnobHead routes — all weights-derived,
        zero learned parameters.  On cuda_primary models this forces
        the eager capture path (CUDA graphs can't emit cap)."""
        V = self.vocab_size
        end = torch.tensor(self._count - 1, dtype=torch.int64,
                           device=self._dev)
        if not self._gpu:
            cap: dict = {}
            lg_np = self._predict(cap)
            lg = torch.as_tensor(np.asarray(lg_np), dtype=torch.float32,
                                 device=self._dev)
            keys = self._keys_at(end.item()) if end.item() >= 0 else {}
            warm = self._regime_now(keys)
            return lg, cap, warm
        slot, found, _ev, live_k, cok, dok, dk, _pk = \
            self._probes_at(end)
        kb = self._knn_rows()
        if self.config.use_knn and kb > 0:
            kv, kvt = self._knn_votes_dev(kb, end=end)
        else:
            kv = torch.zeros(V, dtype=torch.float32, device=self._dev)
            kvt = torch.zeros((), dtype=torch.float32,
                              device=self._dev)
        cap = {}
        lg = self._logits_dev(end, slot, found, live_k, cok, dok, dk,
                              kv=kv, kvt=kvt, cap=cap)
        warm = int(self._dthin_g.item())

        lv, flv, per = self._cap_sparse_split(warm)
        cap["_sparse_stats"] = {
            ch: (float(v.max()) if v.size else 0.0,
                 math.log1p(float((v > 0).sum())),
                 math.log1p(float(np.exp(np.clip(v, None, 20)).sum())))
            for ch, (v, _idx) in per.items()}
        cap["_sparse_raw"] = per
        nz = np.nonzero(lv)[0]
        if nz.size:
            lg.index_add_(
                0, torch.from_numpy(nz).to(self._dev, torch.int64),
                torch.from_numpy(lv[nz]).to(self._dev, torch.float32))
        # production _masked(): fuzzy lifts restricted to topk support
        # union topm
        c = self.config
        fuzzy_on = (c.gcell or c.gsyn or c.gsyn_deep or c.dnet)
        sup_idx = (torch.topk(lg, min(c.lift_topk, lg.numel())).indices
                   if (c.lift_topk > 0 and fuzzy_on) else None)

        def _masked(v):
            if sup_idx is None:
                return v
            keep = torch.zeros(v.numel(), dtype=torch.bool,
                               device=self._dev)
            keep[sup_idx] = True
            if c.lift_topm > 0:
                keep[torch.topk(v, min(c.lift_topm, v.numel())
                                ).indices] = True
            return torch.where(keep, v, v.new_zeros(()))

        nz = np.nonzero(flv)[0]
        if nz.size:
            _ws = self._wR[warm] if c.cond_hedge else self._w
            _fc = c.lift_clip * min(
                len(self.channels) * float(_ws[self._chix["gcell"]]),
                1.0)
            flv = np.clip(flv, -_fc, _fc)
            ft = torch.zeros(lg.numel(), dtype=torch.float32,
                             device=self._dev)
            ft.index_add_(
                0, torch.from_numpy(nz).to(self._dev, torch.int64),
                torch.from_numpy(flv[nz]).to(self._dev, torch.float32))
            lg.add_(_masked(ft))

        for name, fn in (("gsyn", self._gsyn_lift_vec),
                         ("gsyn_deep", self._gsd_lift_vec),
                         ("dnet", self._dnet_lift_vec)):
            vec = fn(warm)
            if vec is not None:
                cap[name] = vec
                lg = lg + _masked(vec)
        return lg, cap, warm

    def attach_head(self, head) -> None:
        """Attach a controller head (e.g. KnobHead).  Afterwards every
        ``_predict`` — engine generate, eval, teach probes — routes
        through ``head.remix(lg, cap, self)``; the head remixes the
        channel distributions read off the weight tables.  Attaching
        disables the CUDA-graph predict path (cap can't cross graph
        boundaries) until ``detach_head``."""
        self._head = head
        self._pgraph = None            # drop any captured predict graph

    def detach_head(self) -> None:
        self._head = None

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
                a_c = self._esc_alpha(len(cell))
                p = (cnt + a_c * self._uni_p(t, denom)) \
                    / (tot + a_c)
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

    def _sparse_lift_vec_dev(self, warm: int,
                             gc_code: int | None = None
                             ) -> np.ndarray | None:
        """Vectorized merge of ``_sparse_lift_channels_dev`` for the hot
        decode path - same per-cell math evaluated with fp64 numpy
        gathers instead of per-follower Python loops (hot cells carry
        thousands of followers; the dict walk dominated decode).
        ``explain()`` keeps the dict version for per-channel splits."""
        c = self.config
        sparse_ok = c.sparse_taps and not (c.sparse_thin_only and warm)
        class_ok = c.class_orders and not (c.class_thin_only and warm)
        word_ok = c.word_orders and not (c.word_thin_only and warm)
        hybrid_ok = c.hybrid_taps and not (c.hybrid_thin_only and warm)
        gcell_ok = c.gcell and not (c.gcell_thin and warm)
        if not (sparse_ok or class_ok or word_ok or hybrid_ok
                or gcell_ok):
            return None
        end = self._count - 1
        denom = self._total + c.uni_alpha * self.vocab_size
        wsrc = self._wR[warm] if c.cond_hedge else self._w
        lift = np.zeros(self.vocab_size, dtype=np.float64)
        # fuzzy channels accumulate separately: candidate-restricted
        # (lift_topk/lift_topm) - they rerank support, not inject mass
        flift = np.zeros(self.vocab_size, dtype=np.float64)

        def acc(cell, tot, wj, into=lift, ev_mul=1.0):
            if c.evidence_lift:
                wj += ev_mul * tot / (tot + c.backoff_kappa)
            idx = np.fromiter(cell.keys(), dtype=np.int64,
                              count=len(cell))
            cnt = np.fromiter(cell.values(), dtype=np.float64,
                              count=len(cell))
            uni = self._uni[idx]
            a_c = self._esc_alpha(len(cell))
            p = (cnt + a_c * (uni + c.uni_alpha) / denom) \
                / (tot + a_c)
            into[idx] += wj * (np.log(np.maximum(p, 1e-12))
                               - (np.log(uni + c.uni_alpha)
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
                    acc(cell, self._ptots[taps].get(key, 0.0), wj)
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
                        acc(cell, tot, wk)
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
                    acc(cell, self._wtots[j].get(key, 0.0), wj)
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
                    acc(cell, self._htots[j].get(key, 0.0), wj)
        if gcell_ok:
            code = gc_code if gc_code is not None else \
                self._gcell_code(self._c, self._c2, end)
            w_rel = len(self.channels) * float(
                wsrc[self._chix["gcell"]])
            for cc, dist in self._gcell_neighbors(code):
                cell = self._gc.get(cc)
                if cell:
                    # CPU parity: ev term scales with gen_bias*decay
                    # too - (w + ev) * gen_bias * decay, not w*g*d + ev
                    db = self._gen_bias * c.gcell_neighbor_decay ** dist
                    acc(cell, self._gct.get(cc, 0.0),
                        float(c.gcell_w) * w_rel * db,
                        into=flift, ev_mul=db)
        return lift, flift

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
        # hedge-arbitrated: w_rel = weight relative to the uniform
        # channel share (1.0 at init, sinks below it when the channel
        # loses) - replaces the fixed-weight bypass that made harmful
        # channels unpunishable
        wsrc = self._wR[warm] if c.cond_hedge else self._w
        w_rel = len(self.channels) * float(wsrc[self._chix["gsyn"]])
        # hard cap scaled by the channel's hedged weight: a muted
        # channel (w_rel<<1) stays a whisper even when its bulk-trained
        # dense state over-shoots (measured lp = -400K on cuda_primary
        # bulk) - +-lift_clip*w_rel caps damage to what the weight
        # actually earned.
        cap_eff = c.lift_clip * min(w_rel, 1.0)
        return (float(c.gsyn_w) * w_rel * self._gen_bias
                * (lp - base)).clamp(-cap_eff, cap_eff)

    def _gsd_lift_vec(self, warm: int,
                      q: torch.Tensor | None = None
                      ) -> torch.Tensor | None:
        """[V] dense lift for the gsyn_deep (TTT-MLP) channel, or None
        when gated off.  Same additive-logit form as _gsyn_lift_vec -
        the two channels share key space but carry separate weights."""
        c = self.config
        if not c.gsyn_deep or (c.gsyn_deep_thin and warm):
            return None
        if q is None:
            q = self._gsyn_keys(self._gcell_feats(
                self._c, self._c2, self._count - 1))
        lp = self._gsd_logits(q)
        base = self._log_uni_num - math.log(
            self._total + c.uni_alpha * self.vocab_size)
        wsrc = self._wR[warm] if c.cond_hedge else self._w
        w_rel = len(self.channels) * float(wsrc[self._chix["gsd"]])
        cap_eff = c.lift_clip * min(w_rel, 1.0)
        return (float(c.gsyn_deep_w) * w_rel * self._gen_bias
                * (lp - base)).clamp(-cap_eff, cap_eff)

    def _dnet_lift_vec(self, warm: int,
                       q: torch.Tensor | None = None
                       ) -> torch.Tensor | None:
        """[V] dense lift for the dnet iterated-memory channel, or
        None when gated off.  Same contract as the gsyn lifts - the
        stack's final-layer readout scores candidate fingerprints."""
        c = self.config
        if not c.dnet or (c.dnet_thin and warm):
            return None
        if q is None:
            q = self._gsyn_keys(self._gcell_feats(
                self._c, self._c2, self._count - 1))
        lp = self._dnet_logits(q)
        base = self._log_uni_num - math.log(
            self._total + c.uni_alpha * self.vocab_size)
        wsrc = self._wR[warm] if c.cond_hedge else self._w
        w_rel = len(self.channels) * float(wsrc[self._chix["dnet"]])
        cap_eff = c.lift_clip * min(w_rel, 1.0)
        return (float(c.dnet_w) * w_rel * self._gen_bias
                * (lp - base)).clamp(-cap_eff, cap_eff)

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
        c = self.config
        fuzzy_on = (c.gcell or c.gsyn or c.gsyn_deep or c.dnet)
        sup_idx = None
        if c.lift_topk > 0 and fuzzy_on:
            # candidate support = top-k of the pre-lift row: fuzzy
            # channels rerank these (plus their own top-m proposals)
            sup_idx = torch.topk(
                row, min(c.lift_topk, row.numel())).indices

        def _masked(v):
            """Restrict a fuzzy [V] lift to support ∪ its top-m."""
            if sup_idx is None:
                return v
            keep = torch.zeros(v.numel(), dtype=torch.bool,
                               device=self._dev)
            keep[sup_idx] = True
            if c.lift_topm > 0:
                keep[torch.topk(v, min(c.lift_topm, v.numel())
                                ).indices] = True
            return torch.where(keep, v, v.new_zeros(()))

        res = self._sparse_lift_vec_dev(warm, gc_code=gc_code)
        if res is not None:
            lv, flv = res
            nz = np.nonzero(lv)[0]
            if nz.size:
                row.index_add_(0, torch.from_numpy(nz).to(
                    self._dev, torch.int64), torch.from_numpy(
                    lv[nz]).to(self._dev, torch.float32))
            nz = np.nonzero(flv)[0]
            if nz.size:
                _ws = self._wR[warm] if c.cond_hedge else self._w
                _fc = c.lift_clip * min(
                    len(self.channels) * float(
                        _ws[self._chix["gcell"]]), 1.0)
                flv = np.clip(flv, -_fc, _fc)
                ft = torch.zeros(row.numel(), dtype=torch.float32,
                                 device=self._dev)
                ft.index_add_(0, torch.from_numpy(nz).to(
                    self._dev, torch.int64), torch.from_numpy(
                    flv[nz]).to(self._dev, torch.float32))
                row.add_(_masked(ft))
        gl = self._gsyn_lift_vec(warm, q=gsyn_q)
        if gl is not None:
            row.add_(_masked(gl))
        gd = self._gsd_lift_vec(warm, q=gsyn_q)
        if gd is not None:
            row.add_(_masked(gd))
        dn = self._dnet_lift_vec(warm, q=gsyn_q)
        if dn is not None:
            row.add_(_masked(dn))

    def _predict_gpu(self) -> torch.Tensor:
        """Standalone predict (no learn) - CUDA-graph captured like
        ``_step_dev``: the context end rides a device input tensor and
        the knn bucket is baked into the capture (recapture on cross).
        Eager fallback while warming or when graphs are disabled
        (clone workers).  Measured ~16x vs the eager pass (5.9ms ->
        0.37ms at V=512)."""
        if self._head is not None:
            # knob head attached: route through the channel-evidence
            # capture path — CUDA graphs can't emit cap
            lg, cap, warm = self.predict_cap()
            return self._head.remix(lg, cap, self, warm)
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
                            or self.config.gcell or self.config.gsyn
                            or self.config.gsyn_deep or self.config.dnet):
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
                or self.config.gsyn or self.config.gsyn_deep
                or self.config.dnet):
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
        if self._head is not None and cap is None:
            cap = {}            # head needs channel evidence
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

        # order channels - with backoff_order_sharp the effective
        # kappa shrinks with order length (long-context hits are more
        # specific -> less evidence needed for confidence)
        o_min = min(c.orders) if c.orders else 1
        # PPM exclusion walks deepest-first so a token already counted
        # at order k' is masked out of every shallower order's votes.
        ord_iter = (sorted(c.orders, reverse=True) if c.ppm_excl
                    else c.orders)
        excluded: set[int] = set()
        for k in ord_iter:
            key = keys.get(k)
            if key is None:
                continue
            cell = self._tables[k].get(key)
            if not cell:
                continue
            tot = self._totals[k].get(key, 0.0)
            wk = w[f"o{k}"]
            if c.evidence_lift:
                kap = (c.backoff_kappa * (o_min / k)
                       ** c.backoff_order_sharp
                       if c.backoff_order_sharp > 0
                       else c.backoff_kappa)
                eff = (tot * (max(cell.values()) / tot)
                       ** c.evidence_share
                       if c.evidence_share > 0 else tot)
                wk += eff / (eff + kap)
            a_cell = float(len(cell)) if c.ppm_escape_c else c.alpha
            for t, cnt in cell.items():
                if c.ppm_excl and t in excluded:
                    continue             # deeper order already
                                         # explains this token
                p = (cnt + a_cell * self._uni_p(t, denom)) \
                    / (tot + a_cell)
                acc(t, wk * (math.log(max(p, 1e-12)) - log_uni(t)),
                    f"o{k}")
            if c.ppm_excl:
                excluded.update(cell)

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
                a_c = self._esc_alpha(len(cell))
                p = (cnt + a_c * self._uni_p(t, denom)) \
                    / (tot + a_c)
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
                a_c = self._esc_alpha(len(cell))
                p = (cnt + a_c * self._uni_p(t, denom)) \
                    / (tot + a_c)
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
                a_c = self._esc_alpha(len(cell))
                p = (cnt + a_c * self._uni_p(t, denom)) \
                    / (tot + a_c)
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
                a_c = self._esc_alpha(len(cell))
                p = (cnt + a_c * self._uni_p(t, denom)) \
                    / (tot + a_c)
                acc(t, wj * (math.log(max(p, 1e-12)) - log_uni(t)),
                    f"h{j}")

        # candidate support for fuzzy lift channels: top-k of the
        # evidence-accumulated logits BEFORE fuzzy votes - they rerank
        # supported candidates plus their own top-m proposals instead
        # of injecting mass across the vocab (the measured cause of
        # the ablation regression)
        sup = None
        if c.lift_topk > 0 and (c.gcell or c.gsyn or c.gsyn_deep
                                or c.dnet):
            base_now = logits.clone()
            for t, v in lift.items():
                base_now[t] += v
            sup = set(torch.topk(
                base_now, min(c.lift_topk, base_now.numel())
            ).indices.tolist())

        def _keep_mask(vec):
            """sup ∪ top-m of this lift vector, as a bool mask."""
            keep = torch.zeros(vec.numel(), dtype=torch.bool,
                               device=vec.device)
            keep[torch.as_tensor(list(sup), dtype=torch.int64,
                                 device=vec.device)] = True
            if c.lift_topm > 0:
                keep[torch.topk(
                    vec, min(c.lift_topm, vec.numel())).indices] = True
            return keep

        # learned-key cell channel "gcell": SimHash of the context
        # fingerprint -> sparse succ-score cell - generalizes where no
        # verbatim key exists.  Hedge-arbitrated + candidate-restricted.
        if c.gcell and (not c.gcell_thin
                        or self._regime_now(keys) == 0):
            code = self._gcell_code(self._c, self._c2, end)
            wg0 = float(c.gcell_w) * (len(self.channels)
                                      * float(w_src[
                                          self._chix["gcell"]]))
            for cc, dist in self._gcell_neighbors(code):
                cell = self._gc.get(cc)
                if not cell:
                    continue
                tot = self._gct.get(cc, 0.0)
                wg = wg0
                if c.evidence_lift:
                    wg += tot / (tot + c.backoff_kappa)
                wg *= (self._gen_bias
                       * c.gcell_neighbor_decay ** dist)
                a_g = self._esc_alpha(len(cell))
                lc = c.lift_clip * min(
                    len(self.channels) * float(w_src[
                        self._chix["gcell"]]), 1.0)
                if sup is None:
                    for t, sc in cell.items():
                        p = (sc + a_g * self._uni_p(t, denom)) \
                            / (tot + a_g)
                        dv = wg * (math.log(max(p, 1e-12))
                                   - log_uni(t))
                        acc(t, min(max(dv, -lc), lc), "gcell")
                else:
                    dvs = []
                    for t, sc in cell.items():
                        p = (sc + a_g * self._uni_p(t, denom)) \
                            / (tot + a_g)
                        dvs.append((t, min(max(
                            wg * (math.log(max(p, 1e-12))
                                  - log_uni(t)), -lc), lc)))
                    keep = sup | {t for t, _ in sorted(
                        dvs, key=lambda e: -e[1])[:c.lift_topm]}
                    for t, dv in dvs:
                        if t in keep:
                            acc(t, dv, "gcell")

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
                eff_b = (btot * (max(cell.values()) / btot)
                         ** c.evidence_share
                         if c.evidence_share > 0 and cell
                         else btot)
                conf = eff_b / (eff_b + c.backoff_kappa)
                wb = w["backoff"] + (conf if c.evidence_lift else 0.0)
                a_b = self._esc_alpha(len(cell))
                for t, cnt in cell.items():
                    p = conf * (cnt + a_b * self._uni_p(t, denom)) \
                        / (btot + a_b)
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
            a_d = self._esc_alpha(len(votes))
            for t, v in votes.items():
                p = (v + a_d * self._uni_p(t, denom)) / (vt + a_d)
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
                a_k = self._esc_alpha(len(kvotes))
                for t, v in kvotes.items():
                    p = (v + a_k * self._uni_p(t, denom)) \
                        / (vt + a_k)
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

        # topic channel: cosine readout on candidates only - or, with
        # topic_full, the 2-hop association over all of vocab
        if c.topic_full:
            xq = self._knn_ctx_feat(end, episode=c.knn_episode)
            xn = float(xq.norm())
            if xn > 1e-8:
                an = (self._An
                      if getattr(self, "_An", None) is not None
                      else self._A.float().norm(dim=1))
                sims = (self._A.float() @ xq) / (an * xn + 1e-8)
                sims = torch.where(an > 1e-8, sims,
                                   torch.zeros_like(sims))
                wt_f = (w["topic"] + tb) * c.topic_beta
                logits += (wt_f * sims).float().cpu().numpy()
                if cap is not None:
                    tk = torch.topk(
                        sims.abs(), min(16, sims.numel())).indices
                    cap["topic"] = {t: wt_f * float(sims[t])
                                    for t in tk.tolist()}
        elif lift:
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
            gl = (float(c.gsyn_w) * (len(self.channels)
                  * float(w_src[self._chix["gsyn"]])) * self._gen_bias
                  * (self._gsyn_logits(gq)
                     - (self._log_uni_num - log_denom))).float()
            if cap is not None:
                cap["gsyn"] = gl
            logits = logits + (gl if sup is None
                               else gl * _keep_mask(gl))

        # nonlinear memory channel (gsyn_deep): the MLP counterpart to
        # gsyn's linear table - same key/readout contract, separate
        # theta, separate gate so ablations isolate it cleanly
        if (c.gsyn_deep
                and (not c.gsyn_deep_thin
                     or self._regime_now(keys) == 0)):
            gdq = self._gsyn_keys(
                self._gcell_feats(self._c, self._c2, end))
            gdl = (float(c.gsyn_deep_w) * (len(self.channels)
                   * float(w_src[self._chix["gsd"]])) * self._gen_bias
                   * (self._gsd_logits(gdq)
                      - (self._log_uni_num - log_denom))).float()
            if cap is not None:
                cap["gsyn_deep"] = gdl
            logits = logits + (gdl if sup is None
                               else gdl * _keep_mask(gdl))

        # iterated-memory channel (dnet): the stacked delta-rule
        # "mini-transformer" - final-layer readout votes, thin-gated
        if (c.dnet
                and (not c.dnet_thin
                     or self._regime_now(keys) == 0)):
            dnq = self._gsyn_keys(
                self._gcell_feats(self._c, self._c2, end))
            dnl = (float(c.dnet_w) * (len(self.channels)
                   * float(w_src[self._chix["dnet"]])) * self._gen_bias
                   * (self._dnet_logits(dnq)
                      - (self._log_uni_num - log_denom))).float()
            if cap is not None:
                cap["dnet"] = dnl
            logits = logits + (dnl if sup is None
                               else dnl * _keep_mask(dnl))

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
                a_2 = self._esc_alpha(len(votes2))
                for t, m in votes2.items():
                    p2 = (m / w2 + a_2 * self._uni_p(t, denom)) \
                        / (1.0 + a_2)
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
        if self._head is not None:
            if cap is None:
                cap = {}
            logits = self._head.remix(
                logits if torch.is_tensor(logits)
                else torch.as_tensor(np.asarray(logits)),
                cap, self, warm=self._regime)
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
            kv, kvt = self._knn_votes_dev(kb, end=end)
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
        gd = self._gsd_lift_vec(warm)
        if gd is not None:
            cap["gsyn_deep"] = gd
            logits = logits + gd
        dn = self._dnet_lift_vec(warm)
        if dn is not None:
            cap["dnet"] = dn
            logits = logits + dn
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
        qk = self._knn_fp_ctx(self._count - 1,
                              episode=self.config.knn_episode)
        sims = self._knn_fp[:kb].float() @ qk.float()
        sims = torch.where(self._knn_pos[:kb] >= 0, sims,
                           torch.full_like(sims, -1e9))
        top = torch.topk(sims, min(k, kb))
        return [(int(i), float(s), int(self._knn_succ[i]),
                 int(self._knn_pos[i]))
                for s, i in zip(top.values.tolist(),
                                top.indices.tolist())
                if s > self.config.knn_min_sim]

    def attach_vocab(self, itos: list) -> None:
        """Attach decoded token strings (one per id) so the answer-type
        prior can classify prompts and build entity/digit masks without
        an external tokenizer reference.  Cheap; safe to call post-load."""
        self._vocab_str = [str(s) for s in itos]
        self._cls_masks = None

    def _atype_mask(self, kind: str):
        """V-bool device mask for an answer class, lazily built from
        _vocab_str: 'ent' = word-initial alpha pieces (starts a word -
        named entities survive regardless of case since QA answers are
        ingested lowercased), 'cap' = capitalized pieces, 'num' = pieces
        containing a digit.  None without an attached vocab."""
        if not getattr(self, "_vocab_str", None):
            return None
        if getattr(self, "_cls_masks", None) is None:
            V = self.vocab_size
            ent = torch.zeros(V, dtype=torch.bool, device=self._dev)
            cap = torch.zeros(V, dtype=torch.bool, device=self._dev)
            num = torch.zeros(V, dtype=torch.bool, device=self._dev)
            ent_l, cap_l, num_l = [], [], []
            for s in self._vocab_str[:V]:
                # word-initial = space-initial alpha piece: BPE
                # continuation pieces ('ered','th') start with a letter
                # but can never begin a word
                word_initial = bool(
                    len(s) > 1 and s[0] == " " and s[1].isalpha())
                ent_l.append(word_initial)
                cap_l.append(bool(
                    len(s) > 1 and s[0] == " " and s[1].isupper()))
                num_l.append(any(ch.isdigit() for ch in s))
            ent[: len(ent_l)] = torch.as_tensor(
                ent_l, dtype=torch.bool, device=self._dev)
            cap[: len(cap_l)] = torch.as_tensor(
                cap_l, dtype=torch.bool, device=self._dev)
            num[: len(num_l)] = torch.as_tensor(
                num_l, dtype=torch.bool, device=self._dev)
            self._cls_masks = {"ent": ent, "cap": cap, "num": num}
        return self._cls_masks.get(kind)

    def _q_atype(self, ids) -> str:
        """Classify a question prompt's answer type from its decoded
        text: numeric-expectation words -> 'num', else 'ent' (trivia
        answers are overwhelmingly named entities)."""
        if not getattr(self, "_vocab_str", None):
            return "any"
        vs = self._vocab_str
        txt = "".join(vs[i] if 0 <= i < len(vs) else " "
                      for i in ids[-200:]).lower()
        return ("num" if re.search(
            r"\b(year|when|date|day|age|old|many|much|number|long|"
            r"far|tall|high|population|score|percent|frequency)\b",
            txt) else "ent")

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
                 generator: torch.Generator | None = None,
                 atype: str | None = None
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
        amask = None
        if c.knn_atype and c.knn_lift > 0:
            kind = atype if atype is not None else self._q_atype(prompt)
            amask = self._atype_mask(kind)
        seen_q = False
        if (c.knn_lift > 0 and c.knn_lift_deep > 0
                and getattr(self, "_gpu", False)
                and prompt):
            # any deep-order cell at the prompt end = the question's
            # exact long suffix was trained - recall owns the answer
            end_t = torch.as_tensor(
                self._count - 1, dtype=torch.int64, device=self._dev)
            _sl, _f, _e, _lk, _cok, _dok, _dk, _pk = \
                self._probes_at(end_t)
            n_ord = _f.numel() - 2
            ord_t = torch.as_tensor(
                list(c.orders), dtype=torch.int64, device=self._dev)
            seen_q = bool((_f[:n_ord].bool()
                           & (ord_t >= int(c.knn_lift_deep))).any())
        try:
            for _step in range(max_new):
                lg = torch.as_tensor(self._predict()).float().clone()
                if (c.knn_lift > 0 and c.use_knn
                        and self._knn_cur > 0
                        and _step < int(c.knn_lift_steps)
                        and not seen_q
                        and (not c.knn_lift_thin
                             or int(getattr(
                                 self, "_dthin_g",
                                 torch.zeros((), dtype=torch.int64,
                                             device=self._dev)
                                 ).item()) == 0)
                        and (c.knn_lift_conf <= 0
                             or float(torch.softmax(lg, -1).max())
                             < float(c.knn_lift_conf))):
                    kb = self._knn_rows()
                    e = torch.as_tensor(self._count - 1,
                                        device=self._dev)
                    if getattr(self, "_ringg", None) is not None:
                        kv, kvt = self._knn_votes_dev(kb, end=e)
                        if amask is not None:
                            kv = kv * amask.float()
                            kvt = kv.sum()
                        share = float(
                            kv.max() / kvt.clamp(min=1e-8)
                        ) if float(kvt) > 0 else 0.0
                        if (float(kvt) > 0
                                and share >= float(
                                    c.knn_lift_topconf)):
                            pk = kv / kvt.clamp(min=1e-8)
                            lg = lg + float(c.knn_lift) * torch.log(
                                pk.clamp(min=1e-12))
                    else:
                        vd = self._knn_votes(c.knn_k)
                        if amask is not None:
                            vd = {t: w for t, w in vd.items()
                                  if bool(amask[t])}
                        if vd:
                            tot = sum(vd.values())
                            top = max(vd.values())
                            if (top / max(tot, 1e-8)
                                    >= float(c.knn_lift_topconf)):
                                for t, w in vd.items():
                                    lg[t] += float(c.knn_lift) * (
                                        float(np.log(
                                            max(w / tot, 1e-12))))
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
