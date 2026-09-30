"""Context/ring/episodic/knn state helpers for FluxLM."""
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
                      _load_teacher_embedding, _mix64, _s64,
                      _scatter_sum)
from .config import FluxConfig
from .journal import FluxJournal
from .tables import _GPUCellTables
from .kernels import HAS_TRITON

if HAS_TRITON:
    from .kernels import (_flux_deep_votes_kernel,
                          _flux_logit_tail_kernel)


class _StateMixin:

    def _suffix_key(self, k: int, end: int) -> int | None:
        """Hash of the last ``k`` tokens ending at absolute pos ``end``,
        or None if fewer than ``k`` tokens exist in the current episode."""
        if end - k + 1 < self._ep_start or end < k - 1:
            return None
        cap = self.config.ring_capacity
        h = (int(self._H[end % cap])
             - int(self._H[(end - k) % cap]) * int(self._pows[k])) \
            % (1 << 64)
        return h

    def _keys_at(self, end: int) -> dict[int, int]:
        keys = {}
        for k in self.config.orders:
            key = self._suffix_key(k, end)
            if key is not None:
                keys[k] = key
        return keys

    def _keys_extended(self, tok: int) -> dict[int, int]:
        """Suffix keys of the HYPOTHETICAL context ctx + [``tok``] -
        virtual-rollout probe: pure register math, no ring writes."""
        c = self.config
        end = self._count                    # hypothetical pos of tok
        cap = c.ring_capacity
        h_e = ((int(self._H[(end - 1) % cap]) * 1099511628211 + tok + 1)
               % (1 << 64)) if end > 0 else (tok + 1)
        keys = {}
        for k in c.orders:
            if end - k + 1 < self._ep_start or end < k - 1:
                continue
            keys[k] = (h_e - int(self._H[(end - k) % cap])
                       * int(self._pows[k])) % (1 << 64)
        return keys

    def _gap_key(self, taps: tuple, end: int) -> int | None:
        """Hash of the gapped suffix (tokens at end - off for off in
        ``taps``) - a noncontiguous context key that survives single-
        token corruption.  Fold = same P-multiply chain as _push."""
        span = max(taps) + 1
        if end - span + 1 < self._ep_start or end < span - 1:
            return None
        cap = self.config.ring_capacity
        h = 0
        for off in taps:
            h = (h * 1099511628211
                 + int(self._ring[(end - off) % cap]) + 1) % (1 << 64)
        return h

    def _class_key(self, k: int, end: int) -> int | None:
        """Hash of the last ``k`` byte-CLASSES ending at ``end`` - the
        coarse-context keys for the g<k> channels.  Same P-fold as
        _gap_key but reads the class LUT through the token ring."""
        if end - k + 1 < self._ep_start or end < k - 1:
            return None
        cap = self.config.ring_capacity
        h = 0
        for i in range(k):
            h = (h * 1099511628211
                 + int(self._cls[int(self._ring[(end - i) % cap])])
                 + 1) % (1 << 64)
        return h

    def _hybrid_key(self, pattern: tuple, end: int) -> int | None:
        """Hash over a (offset, is_raw) pattern - each position folds
        either the raw ring byte or its class.  Generalizes a raw
        suffix where only the newest bytes are verbatim."""
        if not pattern:
            return None
        span = max(p[0] for p in pattern) + 1
        if end - span + 1 < self._ep_start or end < span - 1:
            return None
        cap = self.config.ring_capacity
        h = 0
        for off, raw in pattern:
            tok = int(self._ring[(end - off) % cap])
            h = (h * 1099511628211
                 + (tok if raw else int(self._cls[tok])) + 1) % (1 << 64)
        return h

    def _word_key(self, w: int, end: int) -> int | None:
        """Hash of the last ``w`` completed words + current partial-word
        length bucket, ending the context at ``end``.  Derived on demand
        by scanning the ring backward - no extra state.  Words are
        maximal runs of non-separator bytes (separators = space/tab/
        newline, byte classes 3/4).  None when fewer than ``w`` complete
        words exist in the current episode."""
        if end < self._ep_start or end < 0:
            return None
        cap = self.config.ring_capacity
        cls = self._cls
        ring = self._ring
        sep = (cls == 3) | (cls == 4)          # broadcast LUT -> [V] bool
        P = 1099511628211
        words: list[int] = []
        i = end
        floor = self._ep_start - 1
        # current partial-word length (0 when end is a separator)
        cur = 0
        while i > floor and not sep[ring[i % cap]]:
            cur += 1
            i -= 1
        while len(words) < w and i > floor:
            while i > floor and sep[ring[i % cap]]:   # skip separators
                i -= 1
            if i <= floor:
                break
            h = 0                                     # fold word forward
            j = i
            while j > floor and not sep[ring[j % cap]]:
                j -= 1
            for p in range(j + 1, i + 1):
                h = (h * P + int(ring[p % cap]) + 1) % (1 << 64)
            words.append(h)
            i = j
        if len(words) < w:
            return None
        h = min(cur, 3) + 1
        for wh in words:
            h = (h * P + wh + 1) % (1 << 64)
        return h

    def _raw_hash(self, k: int, end: int) -> int:
        """Suffix hash with no episode check - used to validate ring
        positions that may belong to older episodes."""
        cap = self.config.ring_capacity
        return (int(self._H[end % cap])
                - int(self._H[(end - k) % cap]) * int(self._pows[k])) \
            % (1 << 64)

    # -- live learning ---------------------------------------------------

    def _push(self, tok: int, epi_ok: bool = True,
              knn_ok: bool = True) -> None:
        """Append ``tok`` to the episodic stream (ring + hash + index).
        ``epi_ok``/``knn_ok`` are the write-gate decisions - the caller
        (``_learn_token``/fused path) decides whether the expensive
        episodic structures need this write; observe-only pushes never
        reach the write block anyway."""
        c = self.config
        pos = self._count
        slot = pos % c.ring_capacity
        self._ring[slot] = tok
        prev_h = int(self._H[(pos - 1) % c.ring_capacity]) if pos > 0 else 0
        self._H[slot] = (prev_h * 1099511628211 + tok + 1) % (1 << 64)
        if self._gpu:
            self._ringg[slot] = tok
            if pos == 0:
                self._Hg[0] = tok + 1
            else:
                self._Hg[slot] = (self._Hg[(pos - 1) % c.ring_capacity]
                                  * 1099511628211 + tok + 1)
        # deferred episodic index: token ``tok`` is the SUCCESSOR of the
        # context ending at pos-1 - inserting now means the entry
        # carries its successor + certificate hashes inline, so it still
        # votes after the ring wraps (archived deep memory).  Memory
        # write - gated on learning; skipped for observe-only pushes.
        if self.learning:
            key = self._suffix_key(c.deep_order, pos - 1)
            if key is not None and epi_ok:
                certs = tuple(
                    self._suffix_key(L, pos - 1) or 0
                    for L in c.cert_orders)
                dq = self._epi.get(key)
                if dq is None:
                    dq = self._epi[key] = deque(
                        maxlen=c.epi_max_per_key)
                elif len(dq) == c.epi_max_per_key:
                    # deque append will drop the oldest - journal it
                    # first so the write is exactly revertible
                    old = dq[0]
                    self.journal.append(
                        ("eE", key, old[0], old[1], old[2]))
                    self._epi_n -= 1
                dq.append((pos - 1, tok, certs))
                self._epi_fifo.append((key, pos - 1))
                self._epi_n += 1
                self._epi_evict()
                self.journal.append(("e", key, pos - 1))
            # knn bank write: fingerprint of the pre-``tok`` context
            # (self._c is still the context EMA - updated below)
            if c.use_knn and knn_ok:
                self._knn_write(tok, pos - 1)
                if self._gpu:
                    self._jcur = (self._jcur + 1) % c.vec_delta_cap
        # topic EMA: c <- lambda*c + (1-lambda)*r_tok
        self._c.mul_(c.topic_decay).add_(
            self._R[tok].float() * (1.0 - c.topic_decay))
        if c.bigram_sketch:
            # order-sensitive sketch: _c2 <- lam*_c2 + (1-lam)*
            # bind(prev, tok); the pair term is zero at episode heads
            self._c2.mul_(c.topic_decay)
            if pos > self._ep_start:
                self._c2.add_(self._bind_pair(
                    int(self._ring[(pos - 1) % c.ring_capacity]), tok)
                    * (1.0 - c.topic_decay))
        self._count += 1
        if self._gpu:
            self._pos_g += 1
            self._next_lite = False   # flag refers to a different ctx
        if c.auto_episode and tok == c.eos_token_id:
            # document boundary: next token starts a fresh episode
            # (journaled, so revert restores _c/_ep_start correctly)
            self.soft_reset()

    def _orders_confident(self, keys: dict[int, int]) -> bool:
        """Adaptive-read check: some order cell has a dominant follower
        (share >= adaptive_read_conf over >= adaptive_read_min_tot mass)
        - the deep scan and knn matvec are then skippable reads."""
        c = self.config
        for k, key in keys.items():
            cell = self._tables[k].get(key)
            if not cell:
                continue
            tot = self._totals[k].get(key, 0.0)
            if tot >= c.adaptive_read_min_tot and \
                    max(cell.values()) / tot >= c.adaptive_read_conf:
                return True
        return False

    def _regime_now(self, keys: dict[int, int]) -> int:
        """cond_hedge evidence regime of the current context.

        1 = warm (some order table holds tot > thin_tot for this
        context's suffix key), 0 = thin (no order evidence - retrieval/
        semantic channels are the only signal)."""
        tt = self.config.thin_tot
        for k, key in keys.items():
            if self._totals.get(k, {}).get(key, 0.0) > tt:
                return 1
        return 0

    def _recent_lift_z(self, span: int, decay: float,
                       sign: float) -> float:
        """Normalizer correction for calibrated recency/fatigue rewards:
        Σ_v p_uni(v)·(exp(sign·decay^age(v)) − 1) over the live window
        (tokens not seen in the window contribute exp(0) = 1, cancelling
        their own p_uni mass).  Bounded scan of the ``_recent`` deque;
        latest occurrence per token wins."""
        end = self._count - 1
        seen: dict[int, int] = {}
        for pos, t in reversed(self._recent):
            age = end - pos
            if age >= span:
                break
            if age >= 0 and t not in seen:
                seen[t] = age
        if not seen:
            return 0.0
        denom = self._total + self.config.uni_alpha * self.vocab_size
        return float(sum(
            self._uni_p(t, denom)
            * (math.exp(sign * decay ** a) - 1.0)
            for t, a in seen.items()))

    def _bind_pair(self, prev_t: int, cur_t: int) -> torch.Tensor:
        """Order-sensitive bound pair R[prev][bp1] * R[cur][bp2]
        (permuted elementwise product - bind(a,b) != bind(b,a))."""
        return (self._R[prev_t].float()[self._bp1]
                * self._R[cur_t].float()[self._bp2])

    def _bigram_feats(self, prev_ids: torch.Tensor,
                      cur_ids: torch.Tensor) -> torch.Tensor:
        """[B, topic_dim] bound pairs for paired token-id tensors."""
        a = self._R.index_select(0, prev_ids.long().reshape(-1))
        b = self._R.index_select(0, cur_ids.long().reshape(-1))
        return a[:, self._bp1].float() * b[:, self._bp2].float()

    def _bigram_vector(self, ids) -> torch.Tensor:
        """Standalone bigram-bound sketch for arbitrary ids - same
        weighting _c2 accumulates live (the (1-lam) factor is applied
        by callers that need exact scale parity)."""
        c = self.config
        if len(ids) < 2:
            return torch.zeros(c.topic_dim, device=self._dev)
        prev = torch.tensor(ids[:-1], dtype=torch.int64,
                            device=self._dev)
        cur = torch.tensor(ids[1:], dtype=torch.int64,
                           device=self._dev)
        w = (c.topic_decay ** torch.arange(
            len(ids) - 2, -1, -1, dtype=torch.float32,
            device=self._dev))
        return (self._bigram_feats(prev, cur) * w[:, None]).sum(0)

    def _rebuild_c2(self) -> None:
        """Recompute _c2 from the live ring segment [ep_start, count)
        - exact for the in-ring episode tail; used when a snapshot
        predates the bigram sketch."""
        self._c2.zero_()
        n = self._count - self._ep_start
        if n < 2:
            return
        ids = [int(self._ring[i % self.config.ring_capacity])
               for i in range(self._ep_start, self._count)]
        self._c2.copy_(self._bigram_vector(ids)
                       * (1.0 - self.config.topic_decay))

    def _knn_fp_of(self, cvec: torch.Tensor) -> torch.Tensor:
        """Unit-normalized context fingerprint of a context vector."""
        fp = cvec.float() @ self._Pcsem
        return fp / (fp.norm() + 1e-8)

    # -- gcell: learned-key (SimHash) cell channel -------------------------

    def _gcell_feats(self, cvec: torch.Tensor, c2vec: torch.Tensor,
                     end_pos: int) -> torch.Tensor:
        """Context features for the gcell code - same layout as the
        distill head input ([c_hat, (c2_hat), unit-scaled R tail rows,
        most-recent-first]) but parameterized on an explicit context
        end + vectors so bookkeep can score the pre-write context."""
        c = self.config
        d = c.topic_dim
        base = cvec.float() / cvec.float().norm().clamp(min=1e-8)
        if c.bigram_sketch:
            base = torch.cat(
                [base, c2vec.float()
                 / c2vec.float().norm().clamp(min=1e-8)])
        k = c.head_orders
        if k <= 0:
            return base / base.norm().clamp(min=1e-8)
        cap = c.ring_capacity
        tail = [int(self._ring[(end_pos - j) % cap])
                if end_pos - j >= 0 else 0
                for j in range(k)]
        valid = torch.tensor(
            [float(end_pos - j >= 0) for j in range(k)],
            dtype=torch.float32, device=self._dev)
        rows = (self._R[torch.as_tensor(tail, dtype=torch.long,
                                        device=self._dev)].float()
                * valid[:, None] / (d ** 0.5))
        x = torch.cat([base, rows.reshape(-1)])
        return x / x.norm().clamp(min=1e-8)

    def _gcell_code_from_feats(self, x: torch.Tensor) -> int:
        """SimHash code of a precomputed feature vector (single-row
        form of _gcell_codes_from_feats)."""
        q = (x.double() * 512.0).round()                  # [in]
        bits = q @ self._Wk.double()                     # [nbits]
        b = (bits > 0).to(torch.int64)
        return int((b * (1 << torch.arange(
            self.config.gcell_bits, dtype=torch.int64,
            device=q.device))).sum())

    def _gcell_code(self, cvec: torch.Tensor, c2vec: torch.Tensor,
                    end_pos: int) -> int:
        """SimHash code of the context features.  int32-quantized
        input (x*512) x int8 projection, accumulated in fp64 - exact
        integer math, so CPU and CUDA produce the same bits."""
        return self._gcell_code_from_feats(
            self._gcell_feats(cvec, c2vec, end_pos))

    def _gcell_feats_ctx(self, ids) -> torch.Tensor:
        """Context feature vector for an arbitrary token window -
        standalone _gcell_feats (teach/write paths, no ring)."""
        cvec = self._context_vector(ids)
        c2vec = self._bigram_vector(ids)
        c = self.config
        d = c.topic_dim
        base = cvec / cvec.norm().clamp(min=1e-8)
        if c.bigram_sketch:
            base = torch.cat(
                [base, c2vec / c2vec.norm().clamp(min=1e-8)])
        k = c.head_orders
        if k > 0:
            n = min(len(ids), k)
            tail = list(ids[-n:][::-1]) + [0] * (k - n)
            vmask = torch.tensor(
                [1.0] * n + [0.0] * (k - n), dtype=torch.float32,
                device=self._dev)
            rows = (self._R[torch.as_tensor(
                tail, dtype=torch.long, device=self._dev)].float()
                * vmask[:, None] / (d ** 0.5))
            base = torch.cat([base, rows.reshape(-1)])
        return base / base.norm().clamp(min=1e-8)

    def _gcell_code_ctx(self, ids) -> int:
        """Code for an arbitrary token window (teach/write paths)."""
        x = self._gcell_feats_ctx(ids)
        q = (x.double() * 512.0).round()
        bits = q @ self._Wk.double()
        b = (bits > 0).to(torch.int64)
        return int((b * (1 << torch.arange(
            self.config.gcell_bits, dtype=torch.int64,
            device=q.device))).sum())

    def _jadd(self, e: tuple) -> None:
        """Journal a write - routes to the active bulk block's local
        list when one is open, else the global journal."""
        bj = self._blk_journal
        if bj is not None:
            bj.append(e)
        else:
            self.journal.append(e)

    def _gcell_write(self, code: int, x: int, s: float) -> None:
        """Delta-rule successor-score write into a learned-code cell:
        s_x += lr*s*(1 - s_x).  Journaled 'G' (pre-write score + cell
        total) for bit-exact revert; 'gE' snapshots an evicted cell."""
        c = self.config
        cell = self._gc.get(code)
        if cell is None:
            cell = self._gc[code] = {}
            self._gc_fifo.append(code)
            self._gc_buckets.setdefault(
                code >> c.gcell_bucket_shift, set()).add(code)
            while len(self._gc) > c.gcell_max_cells and self._gc_fifo:
                oc = self._gc_fifo.popleft()
                if oc in self._gc:          # stale fifo entries skip
                    self._jadd(
                        ("gE", oc, self._gc.pop(oc),
                         self._gct.pop(oc, 0.0)))
                    self._gc_bucket_remove(oc)
                    break
        old = cell.get(x, 0.0)
        d = c.gcell_lr * s * (1.0 - old)
        if d == 0.0:
            return
        old_tot = self._gct.get(code, 0.0)
        cell[x] = old + d
        self._gct[code] = old_tot + d
        # journal the pre-write values (not the delta) so undo is
        # bit-exact - a+b-b can differ from a in the last ulp
        self._jadd(("G", code, x, old, old_tot))

    def _gc_bucket_remove(self, code: int) -> None:
        """Drop a code from the fuzzy-read bucket index."""
        b = self._gc_buckets.get(code >> self.config.gcell_bucket_shift)
        if b is not None:
            b.discard(code)
            if not b:
                del self._gc_buckets[
                    code >> self.config.gcell_bucket_shift]

    def _gcell_neighbors(self, code: int) -> list[tuple[int, int]]:
        """[(cell_code, hamming_dist)] for all live cells within
        gcell_read_radius of `code`, exact cell first.  Probes the
        coarse bucket index: codes within distance r can differ in at
        most r bits of the high part, so enumerating high-parts within
        r of the query's high part covers every candidate; members are
        distance-checked exactly."""
        c = self.config
        r = c.gcell_read_radius
        if not self._gc:
            return []
        if r <= 0 or not self._gc_buckets:
            return [(code, 0)] if code in self._gc else []
        from itertools import combinations
        shift = c.gcell_bucket_shift
        hi = code >> shift
        hb = c.gcell_bits - shift
        out: list[tuple[int, int]] = []
        seen: set[int] = set()
        for d_hi in range(r + 1):
            for pos in combinations(range(hb), d_hi):
                bkey = hi
                for p in pos:
                    bkey ^= 1 << p
                bucket = self._gc_buckets.get(bkey)
                if not bucket:
                    continue
                for cc in bucket:
                    if cc in seen:
                        continue
                    d = (cc ^ code).bit_count()
                    if d <= r:
                        seen.add(cc)
                        out.append((cc, d))
        out.sort(key=lambda t: t[1])
        return out

    # -- gsyn: dense delta-rule channel --------------------------------
    # S [topic_dim, gsyn_dim] maps a projected context key k to the
    # expected next-token fingerprint:  S <- g*S + beta*s*(v_x - S@k) x k
    # (the delta rule = error-corrected per-token SGD, same update class
    # as _A).  Read: o = S@q is the predicted fingerprint; tokens score
    # by softmax(_R @ o / temp).  Small enough to stay outside the CUDA
    # graph - host-eager update + post-step dense lift.

    def _gsyn_keys(self, feats: torch.Tensor) -> torch.Tensor:
        """k = normalize(_Wq^T @ feats).  feats [..., in_dim] ->
        keys [..., gsyn_dim]; fp32 (quantize is gcell-only)."""
        k = feats.float() @ self._Wq.float()
        return k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    def _gsyn_write(self, k: torch.Tensor, x: int, s: float,
                    journal: bool = True) -> None:
        """One delta-rule step on S.  Journals ('GS', slot) with the
        (coeff, v_err, k) factors so undo re-applies the exact inverse
        up to fp16 factor precision (same class as 'a' vec_ring).
        journal=False is the bulk path: the "bc" block's gsyn
        pre-state snapshot covers undo, so the factor-ring copies
        (per-token D2H syncs) are skipped."""
        c = self.config
        v = self._R[x].float()
        v_err = v - self._S @ k                       # [d]
        coeff = float(c.gsyn_beta) * float(s)
        coeff_s = float(c.gsyn_slow_beta) * float(s)
        if journal:
            slot = self._gsyn_cur
            row = self._gsyn_ring[slot]
            row[0] = coeff
            row[1] = coeff_s
            row[2:2 + c.topic_dim] = v_err.detach().cpu().numpy()
            row[2 + c.topic_dim:] = k.detach().cpu().numpy()
            self._jadd(("GS", slot))
            self._gsyn_cur = (slot + 1) % c.vec_delta_cap
        outer = torch.outer(v_err, k)
        self._S.mul_(c.gsyn_decay).add_(outer, alpha=coeff)
        if c.gsyn_slow_decay > 0:
            # slow timescale: same error direction, slower write and
            # slower forgetting - accumulates the long-range pattern
            self._Ss.mul_(c.gsyn_slow_decay).add_(outer, alpha=coeff_s)

    def _gsyn_chunk(self, k: torch.Tensor, xs: torch.Tensor,
                    s: torch.Tensor, block: int = 64) -> None:
        """Bulk delta-rule update: exact chunked form of the per-token
        recurrence  S <- g*S + b*s_i*(v_i - S@k_i) x k_i  (WY rep).

        Within a block of B tokens the error terms obey
            e_i = r_i - sum_{j<i} (b_j s_j / g) (k_j . k_i) e_j
        with  r_i = g^-i v_i - S_in@k_i   (S_in = block-entry state),
        i.e.  (I + L) e = r  for strictly-lower L - one triangular
        solve per block replaces B sequential rank-1 updates.  Then
            S_out = g^B S_in + g^{B-1} (b s e)^T K  (e scaled: e_i is
        the NORMALIZED error e_i = ehat_i * g^i).
        The slow state reuses the same e (forward path uses the fast
        state's v_err for both S and Ss).  fp64 block math; undo is
        covered by the caller's "bc" gsyn snapshot."""
        c = self.config
        dev = self._dev
        gam = float(c.gsyn_decay)
        if k.shape[0] == 0:
            return
        if gam <= 0.0:
            # non-decaying-zero edge config: sequential fallback
            for i in range(int(k.shape[0])):
                self._gsyn_write(k[i], int(xs[i]),
                                 float(s[i]), journal=False)
            return
        v_all = self._R[xs.long()].float()                  # [m, d]
        beta = float(c.gsyn_beta)
        # larger blocks amortize the triangular solve (algebraically
        # exact at any size); cap so the normalized-error scaling
        # g^-(bs-1) stays within fp64 comfort even for fast decays
        bs_cap = (1 + int(-6.0 * math.log(10.0) / math.log(gam))
                  if gam < 1.0 else int(k.shape[0]))
        bs = max(1, min(int(getattr(c, "gsyn_chunk", 256)),
                        bs_cap, int(k.shape[0])))
        s_f = s.float()
        for b0 in range(0, int(k.shape[0]), bs):
            kb = k[b0:b0 + bs].double()                     # [nb, dk]
            vb = v_all[b0:b0 + bs].double()                 # [nb, d]
            sb = (beta * s_f[b0:b0 + bs].double())          # [nb]
            nb = kb.shape[0]
            gi = gam ** torch.arange(nb, dtype=torch.float64,
                                     device=dev)            # g^i
            # r_i = g^-i v_i - S@k_i
            r = vb / gi[:, None] - kb @ self._S.double().T
            Lm = torch.tril((kb @ kb.T)
                            * (sb / gam)[None, :], -1)
            e = torch.linalg.solve_triangular(
                torch.eye(nb, dtype=torch.float64, device=dev) + Lm,
                r, upper=False, unitriangular=True)
            # write j's raw error is e_j = g^j * e_j (e solved
            # normalized); it decays under the OWNING state's gamma
            # for the remaining nb-1-j steps:
            #   w_j = coeff_j * g_state^(nb-1-j) * g^j
            jj = torch.arange(nb, dtype=torch.float64, device=dev)
            wS = sb * (gam ** (nb - 1 - jj)) * gi
            dS = (wS[:, None] * e).T @ kb                  # [d, dk]
            self._S.copy_((gam ** nb) * self._S.double() + dS)
            if c.gsyn_slow_decay > 0:
                gams = float(c.gsyn_slow_decay)
                wSs = (float(c.gsyn_slow_beta)
                       * s_f[b0:b0 + bs].double()) \
                    * (gams ** (nb - 1 - jj)) * gi
                dSs = (wSs[:, None] * e).T @ kb
                self._Ss.copy_((gams ** nb) * self._Ss.double()
                               + dSs)

    def _gsyn_logits(self, q: torch.Tensor) -> torch.Tensor:
        """[V] log-softmax-ish affinity vector: softmax(_R @ S@q /
        temp), returned as log-probs (caller subtracts log_uni)."""
        c = self.config
        o = self._S @ q                               # [d]
        if c.gsyn_slow_decay > 0:
            o = o + self._Ss @ q
        o = o / o.norm().clamp(min=1e-8)
        a = (self._R.float() @ o) * c.gsyn_temp       # [V]
        return torch.log_softmax(a.float(), dim=-1)

    def _knn_live(self) -> int:
        """Bank rows actually written - the ring wraps at capacity so
        a hot bank is fully live."""
        return min(int(self._knn_cur), self.config.knn_capacity)

    def _knn_rows(self) -> int:
        """Bank rows worth scanning - the live ring prefix rounded up
        to a power-of-2 bucket (>= ``knn_scan_min``).  Bucketing keeps
        the slice shape static inside a captured CUDA graph between
        recapture boundaries; unwritten padding rows carry pos=-1 and
        are masked out by the caller as usual."""
        return self._knn_rows_at(int(self._knn_cur))

    def _knn_rows_at(self, n: int) -> int:
        """``_knn_rows`` for a projected live count - used by the batched
        forward loop where the host cursor lags the device one."""
        if n <= 0:
            return 0
        cap = self.config.knn_capacity
        if n >= cap:
            return cap
        b = max(1, self.config.knn_scan_min)
        while b < n:
            b <<= 1
        return min(b, cap)

    def _knn_write(self, succ: int, ctx_end: int,
                   fp: torch.Tensor | None = None,
                   journal: bool = True) -> int:
        """Write one fingerprint-bank row: (fp of context ending at
        ``ctx_end``, successor ``succ``).  Returns the slot used."""
        c = self.config
        if c.knn_utility_evict and self._knn_cur >= c.knn_capacity:
            # utility eviction (epi_hit_w-style score): the coldest row
            # = fewest retrieval hits + oldest position; a fresh row is
            # protected by its recency term until it earns hits
            P = c.knn_capacity
            sc = (self._knn_hits[:P].double() * c.knn_evict_hit_w
                  + self._knn_pos[:P].double() / max(self._count, 1))
            slot = int(sc.argmin())
        else:
            slot = self._knn_cur % c.knn_capacity
        if fp is None:
            fp = self._knn_fp_of(self._c)
        prev_pos = int(self._knn_pos[slot])
        if journal:
            if self._gpu:
                prev_succ = int(self._knn_succ[slot])
                self._kprev_g[self._jcur] = self._knn_fp[slot]
                self.journal.append(
                    ("k", slot, prev_pos, prev_succ,
                     self._jcur if prev_pos >= 0 else -1))
            else:
                prev = (None if prev_pos < 0 else
                        (prev_pos, int(self._knn_succ[slot]),
                         self._knn_fp[slot].cpu().numpy().copy()))
                self.journal.append(("k", slot, prev))
        self._knn_fp[slot] = fp.to(self._knn_fp.dtype)
        self._knn_succ[slot] = succ
        self._knn_pos[slot] = ctx_end
        self._knn_hits[slot] = 0         # new row re-earns its utility
        self._knn_cur += 1
        if self._gpu:
            self._knncur_g += 1       # keep device cursor in sync
        return slot

    def _epi_drop(self, key: int, pos: int) -> bool:
        """Drop one (key, ctx_end) entry - journaled (``eE``) so a
        revert restores it, and the GPU row is resynced so evicted
        cells stop voting device-side."""
        dq = self._epi.get(key)
        if not dq:
            return False
        for ent in dq:
            if ent[0] == pos:
                # ride the block journal when bulk-ingesting so a big
                # epi-evict sweep can't flood journal.entries
                self._jadd(("eE", key, ent[0], ent[1], ent[2]))
                dq.remove(ent)
                self._epi_n -= 1
                break
        else:
            return False
        if not dq:
            self._epi.pop(key, None)
            self._epi_hits.pop(key, None)
        # defer the GPU row resync - evict sweeps dirty thousands of
        # keys and one batched probe replaces one probe per drop
        self._epi_dirty.add(key)
        return True

    def _epi_flush_sync(self) -> None:
        """Replay queued GPU-row resyncs as ONE batched probe.  A
        utility-evict sweep used to probe per dropped key (~7k
        _probe_row_w calls per block); the row payload writes still go
        per key but the probe + D2H sync happens once."""
        keys = self._epi_dirty
        if not keys or not self._gpu:
            keys.clear()
            return
        kk_list = list(keys)
        keys.clear()
        c = self.config
        gt = self._gt
        oi = gt.row_epi
        dk_t = torch.tensor([_s64(kk) for kk in kk_list],
                            dtype=torch.int64, device=self._dev)
        se, fe = self._epi_probe(self._epi_hash(dk_t))
        fe_l = fe.tolist()
        se_l = se.tolist()
        ER = self._epi_pos.shape[1]
        LC = len(c.cert_orders)
        for kk_, s, f in zip(kk_list, se_l, fe_l):
            if not f:
                continue
            dq = self._epi.get(kk_)
            if not dq:
                gt.tkeys[oi, s] = 0
                gt.traw[oi, s] = 0
                gt.tocc[oi, s] = 0.0
                self._epi_pos[s].fill_(-1)
                self._epi_succ[s].fill_(-1)
                if LC:
                    self._epi_cert[s].zero_()
                self._epi_cnt[s] = 0
                gt.used[oi] -= 1
                continue
            pos = torch.full((ER,), -1, dtype=torch.int64,
                             device=self._dev)
            succ = torch.full((ER,), -1, dtype=torch.int32,
                              device=self._dev)
            cert = torch.zeros((ER, LC), dtype=torch.int64,
                               device=self._dev) if LC else None
            for j, ent in enumerate(list(dq)[-ER:]):
                pos[j] = int(ent[0])
                succ[j] = int(ent[1])
                if LC:
                    for l, cv in enumerate(ent[2] or ()):
                        cert[j, l] = _s64(cv)
            self._epi_pos[s] = pos
            self._epi_succ[s] = succ
            if LC:
                self._epi_cert[s] = cert
            self._epi_cnt[s] = min(len(dq), ER)
            gt.tocc[oi, s] = float(self._epi_cnt[s])

    def _epi_sync_row(self, key: int) -> None:
        """GPU mirror upkeep: rewrite the episodic row for ``key`` from
        the host deque (post evict/consolidate).  Without this the
        device table kept voting evicted cells - the host dict is the
        authority.  No-op off-GPU or when the row isn't present."""
        if not self._gpu:
            return
        c = self.config
        dk_t = torch.tensor([_s64(key)], dtype=torch.int64,
                            device=self._dev)
        pk = self._epi_hash(dk_t)
        se, fe = self._epi_probe(pk)
        if not bool(fe[0]):
            return
        s = int(se[0])
        gt = self._gt
        oi = gt.row_epi
        dq = self._epi.get(key)
        ER = self._epi_pos.shape[1]
        if not dq:                               # key fully evicted
            gt.tkeys[oi, s] = 0
            gt.traw[oi, s] = 0
            gt.tocc[oi, s] = 0.0
            self._epi_pos[s].fill_(-1)
            self._epi_succ[s].fill_(-1)
            if len(c.cert_orders):
                self._epi_cert[s].zero_()
            self._epi_cnt[s] = 0
            gt.used[oi] -= 1
            return
        pos = torch.full((ER,), -1, dtype=torch.int64,
                         device=self._dev)
        succ = torch.full((ER,), -1, dtype=torch.int32,
                          device=self._dev)
        LC = len(c.cert_orders)
        cert = torch.zeros((ER, LC), dtype=torch.int64,
                           device=self._dev) if LC else None
        for j, ent in enumerate(list(dq)[-ER:]):
            pos[j] = int(ent[0])
            succ[j] = int(ent[1])
            if LC:
                for l, cv in enumerate(ent[2] or ()):
                    cert[j, l] = _s64(cv)
        self._epi_pos[s] = pos
        self._epi_succ[s] = succ
        if LC:
            self._epi_cert[s] = cert
        self._epi_cnt[s] = min(len(dq), ER)
        gt.tocc[oi, s] = float(self._epi_cnt[s])

    def _epi_evict(self) -> None:
        """Utility-bounded episodic memory: when over ``epi_total_cap``,
        scan the oldest ``epi_evict_scan`` fifo candidates and evict the
        lowest-utility entries - score = retrieval hits (cells that
        actually get read dominate) + a small recency term.  FIFO is the
        degenerate case (scan=1).  Every drop is journaled (``eE``)."""
        c = self.config
        n_inv = 1.0 / max(self._count, 1)
        while self._epi_n > c.epi_total_cap and self._epi_fifo:
            over = self._epi_n - c.epi_total_cap
            take = min(len(self._epi_fifo),
                       max(c.epi_evict_scan, over))
            batch = [self._epi_fifo.popleft() for _ in range(take)]
            live = [kp for kp in batch
                    if any(e[0] == kp[1]
                           for e in self._epi.get(kp[0], ()))]
            if len(live) <= over:                # mostly tombstones
                for key, pos in live:
                    self._epi_drop(key, pos)
                continue
            scored = sorted(
                live,
                key=lambda kp: (self._epi_hits.get(kp[0], 0)
                                * c.epi_hit_w + kp[1] * n_inv))
            for key, pos in scored[:over]:       # lowest utility out
                self._epi_drop(key, pos)
            for kp in reversed(scored[over:]):   # survivors stay oldest
                self._epi_fifo.appendleft(kp)
        self._epi_flush_sync()

    def _epi_remove(self, key: int, pos: int) -> bool:
        """Undo helper: drop the entry for context-end ``pos`` under
        ``key`` from the host index."""
        dq = self._epi.get(key)
        if not dq:
            return False
        for ent in dq:
            if ent[0] == pos:
                dq.remove(ent)
                self._epi_n -= 1
                break
        else:
            return False
        if not dq:
            del self._epi[key]
        return True

    def _uni_p(self, t: int, denom: float) -> float:
        return (self._uni[t] + self.config.uni_alpha) / denom

    def _csem_read(self, key: int | None = None) -> torch.Tensor | None:
        """Fingerprint row for the current (or given) context key."""
        c = self.config
        if not c.use_csem:
            return None
        if key is None:
            key = self._suffix_key(c.csem_order, self._count - 1)
            if key is None:
                return None
        if self._gpu:
            pk = _mix64(torch.tensor([_s64(key)], dtype=torch.int64,
                                     device=self._dev) * _MIX1
                        + self._gt.ROW_CSEM_SALT)
            slot, found = self._gt._probe(
                self._gt.tkeys[self._gt.row_csem], pk)
            if not bool(found[0]):
                return None
            return self._csem_rows[slot[0]].float()
        row = self._csem_cpu.get(key)
        return None if row is None else torch.from_numpy(row).to(self._dev)

    def _csem_write(self, x: int, journal: bool = True,
                    scale: float = 1.0) -> None:
        """EMA this context's fingerprint toward follower's feature.
        ``scale`` = surprise gate (lr_eff = csem_lr * scale)."""
        c = self.config
        if not c.use_csem:
            return
        end = self._count - 1
        key = self._suffix_key(c.csem_order, end)
        if key is None:
            return
        self._csem_write_key(key, x, scale=scale, journal=journal)

    def _csem_write_key(self, key: int, x: int, scale: float = 1.0,
                        journal: bool = True) -> None:
        """Key-addressed csem write - same EMA as ``_csem_write`` but
        for an explicit context key (teach()/manual injection)."""
        c = self.config
        ax = (self._A[x] @ self._Pcsem).float()
        if self._gpu:
            gt = self._gt
            rcs = gt.row_csem
            pk = _mix64(torch.tensor([_s64(key)], dtype=torch.int64,
                                     device=self._dev) * _MIX1
                        + gt.ROW_CSEM_SALT)
            slot, found, evi = gt._probe_row_w(rcs, pk)
            s = int(slot[0])
            prev = (self._csem_rows[s].clone()
                    if bool(found[0]) else None)
            if not bool(found[0]):
                gt.tkeys[rcs, s] = pk[0]
                gt.traw[rcs, s] = _s64(key)
                if not bool(evi[0]):
                    gt.used[rcs] += 1
            cur = (self._csem_rows[s].float() if prev is not None
                   else torch.zeros(c.csem_dim, device=self._dev))
            lr_eff = c.csem_lr * scale
            self._csem_rows[s] = (cur * (1.0 - lr_eff)
                                  + ax.half() * lr_eff).half()
            if journal:
                self.journal.append(
                    ("s", int(pk[0]), s,
                     None if prev is None else prev.cpu().numpy()))
            return
        prev = self._csem_cpu.get(key)
        if prev is None and len(self._csem_cpu) >= c.csem_slots:
            # bound the cpu table like the gpu row (csem_slots): evict
            # the oldest inserted row, journaled as an "s" removal so
            # revert restores it
            k_old = next(iter(self._csem_cpu))
            self.journal.append(("s", k_old, -1, self._csem_cpu[k_old]))
            del self._csem_cpu[k_old]
        cur = (torch.from_numpy(prev).to(self._dev) if prev is not None
               else torch.zeros(c.csem_dim, device=self._dev))
        lr_eff = c.csem_lr * scale
        new = (cur * (1.0 - lr_eff) + ax * lr_eff).cpu().numpy()
        self._csem_cpu[key] = new.astype(np.float32)
        if journal:
            self.journal.append(("s", key, -1, prev))

    # -- GPU live paths -------------------------------------------------

    def _keys_dev(self, end: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor]:
        """(keys [O] int64, valid [O] bool) at device scalar ``end``.
        Validity is a separate mask - raw u64 keys can be negative in
        int64 bit pattern, so a sign test would be wrong."""
        c = self.config
        cap = c.ring_capacity
        ks = self._orders_g
        he = self._Hg.index_select(0, (end % cap).reshape(1)).squeeze(0)
        valid = (end - ks + 1 >= self._ep_g) & (end >= ks - 1) \
            & (end >= 0)
        keys = he - self._Hg[(end - ks) % cap] \
            * self._pows_g[ks]
        return keys, valid

    def _suffix_dev(self, k: int, end: torch.Tensor) -> torch.Tensor:
        """Raw-order-k suffix hash at device scalar ``end`` (no episode
        gate - caller masks)."""
        cap = self.config.ring_capacity
        return self._Hg.index_select(0, (end % cap).reshape(1)) \
            .squeeze(0) \
            - self._Hg.index_select(
                0, ((end - k) % cap).reshape(1)).squeeze(0) \
            * self._pows_g[k]

    def _epi_probe(self, pk: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor]:
        """Eager probe on the episodic row of the shared key matrix.
        ``pk`` must already be mixed with ROW_EPI_SALT."""
        return self._gt._probe(self._gt.tkeys[self._gt.row_epi], pk)

    def _epi_hash(self, dk: torch.Tensor) -> torch.Tensor:
        return _mix64(dk * _MIX1 + self._gt.ROW_EPI_SALT)

    def _csem_hash(self, ck: torch.Tensor) -> torch.Tensor:
        return _mix64(ck * _MIX1 + self._gt.ROW_CSEM_SALT)

    def _probes_at(self, end: torch.Tensor) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor]:
        """ONE probe loop resolving every table at context ``end``:
        orders [O], csem row, epi row -> slot/found [R] each, plus the
        validity masks and the raw deep key."""
        c = self.config
        gt = self._gt
        keys, live_k = self._keys_dev(end)                 # [O]
        ck = self._suffix_dev(c.csem_order, end)
        dk = self._suffix_dev(c.deep_order, end)
        allk = torch.cat([keys, ck.reshape(1), dk.reshape(1)])
        pk = gt.ctx_hash_all(allk)                         # [R]
        slot, found, evict = gt.probe_write(pk)
        cok = (end - c.csem_order + 1 >= self._ep_g) \
            & (end >= c.csem_order - 1) & (end >= 0)
        dok = (end - c.deep_order + 1 >= self._ep_g) \
            & (end >= c.deep_order - 1) & (end >= 0)
        return slot, found, evict, live_k, cok, dok, \
            dk.squeeze(0), pk


    def window_tokens(self, end: int, n: int) -> list[int]:
        """The last ``n`` ring tokens ending at absolute pos ``end``
        (end exclusive).  Short/empty near stream start or wrap edge."""
        cap = self.config.ring_capacity
        start = max(0, self._count - cap, end - n)
        return [int(self._ring[i % cap]) for i in range(start, end)]
