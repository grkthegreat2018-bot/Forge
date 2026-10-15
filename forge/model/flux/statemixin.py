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
                      _group_totals, _load_teacher_embedding, _mix64,
                      _s64, _scatter_sum)
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
        semantic channels are the only signal).  With evidence_share>0
        the mass must concentrate - a huge diffuse cell is not warmth."""
        c = self.config
        tt = c.thin_tot
        for k, key in keys.items():
            tot = self._totals.get(k, {}).get(key, 0.0)
            if tot <= tt:
                continue
            cell = self._tables[k].get(key) or {}
            top_sh = (max(cell.values()) / tot) if cell else 0.0
            if c.thin_share > 0:
                if top_sh < c.thin_share:
                    continue
            elif c.evidence_share > 0:
                if cell and (tot * top_sh ** c.evidence_share) <= tt:
                    continue
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

    def _fp_mu(self) -> torch.Tensor:
        """Unit mean-prose direction: the unigram-weighted mean token
        feature, i.e. E[_c] under stationarity.  Uses _unig (device
        counts, CUDA-graph capture safe) or the host _uni array."""
        uni = (self._unig if getattr(self, "_unig", None) is not None
               else torch.as_tensor(self._uni, dtype=torch.float32,
                                    device=self._dev))
        mu = self._R.float().T @ uni
        return mu / mu.norm().clamp(min=1e-8)

    def _knn_fp_of(self, cvec: torch.Tensor) -> torch.Tensor:
        """Unit-normalized context fingerprint of a context vector.
        knn_residual>0 removes the shared mean-prose direction first
        (dim-agnostic: cvec may be [d] or [n,d])."""
        x = cvec.float()
        beta = self.config.knn_residual
        if beta > 0:
            mu = self._fp_mu()
            x = x - beta * (x @ mu)[..., None] * mu
        fp = x @ self._Pcsem
        return fp / (fp.norm(dim=-1, keepdim=True) + 1e-8)

    def _knn_tok_w(self, toks: torch.Tensor) -> torch.Tensor:
        """idf-style rarity weight per token: log(1 + N/(cnt+alpha)).
        Derived from _unig/_tot (device) or _uni (host) - journaled
        counts, no extra state.  Capture-safe on the GPU path."""
        c = self.config
        if getattr(self, "_unig", None) is not None:
            u = self._unig.index_select(0, toks)
            # _tot_g is only refreshed at block end in _ingest_bulk -
            # _unig is updated earlier in the same pass, so its sum is
            # the freshest in-flight total (and == _tot_g elsewhere)
            tot = self._unig.sum()
        else:
            u = torch.as_tensor(self._uni, dtype=torch.float32,
                                device=self._dev)[toks]
            tot = float(self._total)
            return torch.log1p(max(tot, 1.0) / (u + c.uni_alpha))
        return torch.log1p(
            tot.float().clamp(min=1.0) / (u.float() + c.uni_alpha))

    def _knn_ctx_feat(self, end, episode: bool = False
                      ) -> torch.Tensor:
        """Raw (pre-projection) context feature for fingerprinting and
        the topic_full association readout.  knn_window>0: (idf-)bag
        of features over the last W ring tokens, residualized against
        the mean-prose direction.  episode=True: bag over the whole
        current episode (prompt segment) instead - a fixed W window at
        query time reaches past the question into the previous
        generation's output.  0: the _c EMA (legacy)."""
        c = self.config
        W = int(c.knn_window)
        if W <= 0:
            return self._c
        dev = self._dev
        e = (end if torch.is_tensor(end)
             else torch.as_tensor(end, device=dev))
        if episode and getattr(self, "_prompt_g", None) is not None:
            e0 = self._prompt_g.reshape(-1)[-1]
        elif episode:
            e0 = torch.as_tensor(self._prompt_start, device=dev)
        else:
            e0 = (e.reshape(-1)[-1] - W + 1).clamp(min=0)
        n = (e.reshape(-1)[-1] - e0 + 1).clamp(min=1, max=W)
        eiv = e.reshape(-1)[-1]
        idx = eiv - n + 1 + torch.arange(W, device=dev)
        valid = idx <= eiv
        idx = torch.where(valid, idx.clamp(min=0),
                          idx.new_zeros(idx.shape))
        if getattr(self, "_ringg", None) is not None:
            toks = self._ringg.index_select(
                0, idx % c.ring_capacity).long()
        else:
            toks = torch.as_tensor(
                [self._ring[int(i) % c.ring_capacity]
                 for i in idx.tolist()],
                dtype=torch.long, device=dev)
        toks = toks.clamp(0, self.vocab_size - 1)
        rows = self._R.index_select(0, toks).float()
        vf = valid.float()
        if c.knn_idf:
            w = self._knn_tok_w(toks) * vf
            if c.knn_idf_pow != 1.0:
                w = w ** float(c.knn_idf_pow)
            x = (rows * w[:, None]).sum(0) / w.sum().clamp(min=1e-8)
        else:
            x = (rows * vf[:, None]).sum(0) / vf.sum().clamp(min=1e-8)
        beta = c.knn_residual
        if beta > 0:
            mu = self._fp_mu()
            x = x - beta * (x @ mu)[..., None] * mu
        return x

    def _knn_fp_ctx(self, end, episode: bool = False
                    ) -> torch.Tensor:
        """Fingerprint of the context ending at stream index ``end`` -
        the _knn_ctx_feat feature projected to csem_dim.  episode=True
        bags the current-episode tokens (the query prompt) rather than
        a fixed W window."""
        c = self.config
        if int(c.knn_window) <= 0:
            return self._knn_fp_of(self._c)
        x = self._knn_ctx_feat(end, episode=episode)
        fp = x @ self._Pcsem
        return fp / (fp.norm(dim=-1, keepdim=True) + 1e-8)

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
        Rm = self._Reff if self._Reff is not None else self._R
        self._gsyn_write_vec(k, Rm[x].float(), s, journal=journal)

    def _gsyn_write_vec(self, k: torch.Tensor, v: torch.Tensor,
                        s: float, journal: bool = True) -> None:
        """``_gsyn_write`` with an explicit target vector (a blended
        feature mix instead of a single token row) - consolidation and
        recall-weighted writes use this."""
        c = self.config
        v = v.float()
        k = k.float()
        if not (bool(torch.isfinite(v).all())
                and bool(torch.isfinite(k).all())):
            return                     # never write non-finite state
        if not math.isfinite(float(s)):
            s = 1.0                    # non-finite surprise = maximal
        v_err = v - self._S @ k               # [d]
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
                    s: torch.Tensor, block: int = 64,
                    v: torch.Tensor | None = None) -> None:
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
                if v is not None:
                    self._gsyn_write_vec(k[i], v[i],
                                         float(s[i]), journal=False)
                else:
                    self._gsyn_write(k[i], int(xs[i]),
                                     float(s[i]), journal=False)
            return
        if v is not None:
            v_all = v.float()                             # [m, d] soft targets
        else:
            Rm = self._Reff if self._Reff is not None else self._R
            v_all = Rm[xs.long()].float()                 # [m, d]
        beta = float(c.gsyn_beta)
        # larger blocks amortize the triangular solve (algebraically
        # exact at any size); cap so the normalized-error scaling
        # g^-(bs-1) stays within fp64 comfort even for fast decays
        bs_cap = (1 + int(-6.0 * math.log(10.0) / math.log(gam))
                  if gam < 1.0 else int(k.shape[0]))
        bs = max(1, min(int(getattr(c, "gsyn_chunk", 256)),
                        bs_cap, int(k.shape[0])))
        # non-finite guards: a poisoned s/k/v must not write NaN into
        # S - one NaN entry destroys the dense matrix permanently
        # (decay never removes NaN).  s NaN -> maximal (R60
        # convention); non-finite k/v rows zeroed -> their outer
        # products degenerate to no-ops.  Explicit posinf/neginf ->
        # 0/1: the default maps inf to +-3.4e38, still finite, which
        # overflows S to inf on the next write (R69-2).
        s_f = torch.nan_to_num(s.float(), nan=1.0,
                               posinf=1.0, neginf=1.0)
        k = torch.nan_to_num(k.float(), nan=0.0,
                             posinf=0.0, neginf=0.0)
        v_all = torch.nan_to_num(v_all, nan=0.0,
                               posinf=0.0, neginf=0.0)
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
        Rm = self._Reff if self._Reff is not None else self._R
        a = (Rm.float() @ o) * c.gsyn_temp            # [V]
        # a poisoned S/Rl must not NaN the whole mixture - degrade to
        # the uniform prior instead (channel effectively abstains)
        a = torch.nan_to_num(a.float(), nan=0.0)
        return torch.log_softmax(a, dim=-1)

    # -- deep memory (gsyn_deep): TTT-MLP / Titans-style channel -------
    # theta maps the same normalized context key k that linear gsyn
    # uses onto the expected next-token fingerprint, via a 2-layer
    # MLP.  The nonlinearity is the point: a rank-1-updated matrix can
    # only store additive key->value associations; an MLP learns
    # nonlinear maps, which is what lets test-time training generalize
    # past retrieval.  Gradients are computed explicitly (no autograd)
    # so the update is fully inspectable; undo is whole-state snapshots
    # since a gradient step has no closed-form inverse.

    def _gsd_params(self) -> tuple:
        """theta + momentum buffers, in canonical order."""
        return (self._Wd1, self._Sb1, self._Wd2, self._Sb2,
                self._Sm1, self._Smb1, self._Sm2, self._Smb2)

    def _gsd_snap(self) -> tuple:
        """CPU clones of theta + momentum - the undo payload."""
        return tuple(t.detach().cpu().clone()
                     for t in self._gsd_params())

    def _gsd_restore(self, snap: tuple) -> None:
        for t, s in zip(self._gsd_params(), snap):
            t.copy_(s.to(t.device))

    def _gsd_journal(self) -> None:
        """Snapshot theta before the first mutation of the current
        undo scope: "bc" block payload (bulk), session dict (eager
        ingest/consolidate), else a standalone ("gd", snap) entry."""
        blk = self._blk_active
        if blk is not None:
            if "gsd" not in blk:
                blk["gsd"] = self._gsd_snap()
        elif self._gsd_session is not None:
            self._gsd_session.setdefault("snap", self._gsd_snap())
        else:
            self._jadd(("gd", self._gsd_snap()))

    def _gsd_flush(self) -> None:
        """Close a deep-memory undo session: one ("gd", snap) entry
        covering the whole span - mirrors _rl_flush."""
        sess, self._gsd_session = self._gsd_session, None
        snap = sess.get("snap") if sess else None
        if snap is not None:
            self._jadd(("gd", snap))

    def _gsd_forward(self, k: torch.Tensor) -> tuple:
        """Explicit forward: z = Wd1@k + Sb1; h = relu(z); o = Wd2@h
        + Sb2.  Returns (z, h, o) so callers/probes can inspect each
        stage.  k [..., gsyn_dim] -> o [..., topic_dim]."""
        z = k.float() @ self._Wd1.T + self._Sb1
        h = torch.relu(z)
        return z, h, h @ self._Wd2.T + self._Sb2

    def _gsd_write(self, k: torch.Tensor, x: int, s: float,
                   journal: bool = True) -> None:
        """One online step toward the token's feature row."""
        Rm = self._Reff if self._Reff is not None else self._R
        self._gsd_write_vec(k, Rm[x].float(), s, journal=journal)

    def _gsd_write_vec(self, k: torch.Tensor, v: torch.Tensor,
                       s: float, journal: bool = True) -> None:
        """Single TTT-style gradient step on ||f_theta(k) - v||^2:
        theta <- decay*theta - lr*s*(mu*M + grad), M = momentum.
        Explicit grads keep the update inspectable; journal= snapshots
        theta once per undo scope."""
        c = self.config
        if not c.gsyn_deep:
            return
        v = v.float()
        k = k.float()
        if not (bool(torch.isfinite(v).all())
                and bool(torch.isfinite(k).all())):
            return                     # never write non-finite state
        if not math.isfinite(float(s)):
            s = 1.0                    # non-finite surprise = maximal
        if journal:
            self._gsd_journal()
        z, h, o = self._gsd_forward(k)
        err = o - v                                       # [d]
        dW2 = torch.outer(err, h)                         # [d,h]
        dh = (self._Wd2.T @ err) * (z > 0).float()        # [h]
        grads = (torch.outer(dh, k), dh, dW2, err)
        self._gsd_step(grads, float(s), gam=float(c.gsyn_deep_decay))

    def _gsd_step(self, grads: tuple, s: float,
                  gam: float = 1.0) -> None:
        """Shared optimizer update: clip global grad norm, momentum
        accumulate, decayed SGD step.  grads = (Wd1, Sb1, Wd2, Sb2)."""
        c = self.config
        ps = self._gsd_params()
        tot = float(torch.sqrt(sum((g * g).sum() for g in grads)))
        if not math.isfinite(tot):
            return                     # poisoned grad - skip update
        scale = min(1.0, float(c.gsyn_deep_clip) / max(tot, 1e-12))
        mu, lr = float(c.gsyn_deep_momentum), float(c.gsyn_deep_lr)
        # decay applies to theta but not momentum - the forget gate
        # shrinks stored knowledge, not the optimizer's velocity
        for p in ps[:4]:
            p.mul_(gam)
        for m, p, g in zip(ps[4:], ps[:4], grads):
            m.mul_(mu).add_(g, alpha=scale)
            p.add_(m, alpha=-lr * s)

    def _gsd_chunk(self, k: torch.Tensor, xs: torch.Tensor,
                   s: torch.Tensor, v: torch.Tensor | None = None
                   ) -> None:
        """Bulk deep-memory update: one batched gradient step per
        gsyn_deep_chunk positions (LaCT-style large-chunk online
        update).  Not the exact sequential recurrence - per-sample
        errors are computed against the chunk-entry theta, then the
        mean surprise-weighted gradient is applied once; the
        pre-state snapshot covers undo either way."""
        c = self.config
        if not c.gsyn_deep or k.shape[0] == 0:
            return
        if v is not None:
            v_all = v.float()
        else:
            Rm = self._Reff if self._Reff is not None else self._R
            v_all = Rm[xs.long()].float()
        s_f = torch.nan_to_num(s.float(), nan=1.0)
        k = torch.nan_to_num(k.float(), nan=0.0)
        v_all = torch.nan_to_num(v_all, nan=0.0)
        bs = max(1, int(c.gsyn_deep_chunk))
        for b0 in range(0, int(k.shape[0]), bs):
            kb = k[b0:b0 + bs]                              # [nb, dk]
            vb = v_all[b0:b0 + bs]                          # [nb, d]
            sb = s_f[b0:b0 + bs]                            # [nb]
            sw = sb.sum().clamp(min=1e-8)
            if b0 == 0:
                self._gsd_journal()   # covers the whole chunk call
            z, h, o = self._gsd_forward(kb)
            err = (o - vb) * (sb / sw)[:, None]             # [nb, d]
            dW2 = err.T @ h                                 # [d, h]
            db2 = err.sum(0)
            dh = (err @ self._Wd2) * (z > 0).float()        # [nb, h]
            dW1 = dh.T @ kb                                 # [h, dk]
            db1 = dh.sum(0)
            self._gsd_step((dW1, db1, dW2, db2), sw,
                           gam=float(c.gsyn_deep_decay) ** int(kb.shape[0]))

    def _gsd_logits(self, q: torch.Tensor) -> torch.Tensor:
        """[V] channel readout: softmax(_Reff @ normalize(f_theta(q))
        * temp) as log-probs - same contract as _gsyn_logits."""
        c = self.config
        _, _, o = self._gsd_forward(q)
        o = o / o.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        Rm = self._Reff if self._Reff is not None else self._R
        a = (Rm.float() @ o) * c.gsyn_deep_temp             # [V]
        a = torch.nan_to_num(a.float(), nan=0.0)
        return torch.log_softmax(a, dim=-1)

    def probe_gsyn_deep(self, top: int = 8) -> dict:
        """Dissection probe for the deep channel: key vector, hidden
        activations (relu stats), output fingerprint, top predicted
        tokens, param/momentum norms, cumulative state delta from the
        seeded init.  Read-only - journal untouched."""
        c = self.config
        if not c.gsyn_deep:
            return {"enabled": False}
        q = self._gsyn_keys(self._gcell_feats(
            self._c, self._c2, self._count - 1))
        z, h, o = self._gsd_forward(q)
        lp = self._gsd_logits(q)
        topv, topi = lp.topk(min(top, lp.shape[0]))
        ps = self._gsd_params()
        hd = int(c.gsyn_deep_hidden)
        gd = torch.Generator().manual_seed(c.seed ^ 0x6CE13)
        w0 = (torch.randn(hd, c.gsyn_dim, generator=gd)
              / math.sqrt(c.gsyn_dim)).float().to(self._dev)
        return {
            "enabled": True,
            "key_norm": float(q.norm()),
            "hidden": {"mean": float(h.mean()),
                       "max": float(h.max()),
                       "frac_active": float((z > 0).float().mean())},
            "out_norm": float(o.norm()),
            "top_tokens": [(int(t), float(v))
                           for t, v in zip(topi.tolist(),
                                           topv.tolist())],
            "param_norms": {"Wd1": float(self._Wd1.norm()),
                            "Wd2": float(self._Wd2.norm())},
            "mom_norms": {"Sm1": float(self._Sm1.norm()),
                          "Sm2": float(self._Sm2.norm())},
            "theta_drift": float((self._Wd1 - w0).norm()),
        }

    # -- iterated memory (dnet): stacked gated delta-rule layers ------
    # The "transformer that learns online": every layer is the same
    # delta-rule associative memory as gsyn's _S, but STACKED - layer
    # l's input is relu(o_{l-1}), so a layer's readout addresses the
    # next layer's memory.  Content routing through depth is what
    # single-step channels cannot express.  Write gate is content-
    # dependent (sigmoid projection of the layer input) so routing is
    # nonlinear; inter-layer relu breaks odd symmetry.

    def _dnet_snap(self) -> tuple:
        """CPU clones of all layer states - the undo payload."""
        return tuple(s.detach().cpu().clone() for s in self._DS)

    def _dnet_restore(self, snap: tuple) -> None:
        for s, sv in zip(self._DS, snap):
            s.copy_(sv.to(s.device))

    def _dnet_journal(self) -> None:
        """Snapshot layer states before the first mutation of the
        current undo scope - same routing as _gsd_journal."""
        blk = self._blk_active
        if blk is not None:
            if "dnet" not in blk:
                blk["dnet"] = self._dnet_snap()
        elif self._dnet_session is not None:
            self._dnet_session.setdefault("snap", self._dnet_snap())
        else:
            self._jadd(("dn", self._dnet_snap()))

    def _dnet_flush(self) -> None:
        """Close a dnet undo session - one ("dn", snap) entry."""
        sess, self._dnet_session = self._dnet_session, None
        snap = sess.get("snap") if sess else None
        if snap is not None:
            self._jadd(("dn", snap))

    def _dnet_layers_fwd(self, u1: torch.Tensor,
                         write: bool = False,
                         v: torch.Tensor | None = None,
                         s: float = 0.0,
                         journal: bool = True) -> torch.Tensor:
        """Run the stack.  write=False = pure readout (predict-time);
        write=True performs the gated delta update per layer toward
        target ``v`` at strength ``s`` and returns the final readout.
        Every layer predicts the SAME fingerprint target from its own
        input - layer l learns the map from layer l-1's recall."""
        c = self.config
        u = u1.float()
        for l in range(len(self._DS)):
            k = u @ self._Dk[l].float()                      # [dk]
            k = k / k.norm().clamp(min=1e-8)
            o = self._DS[l] @ k                              # [dv]
            if write:
                if journal:
                    self._dnet_journal()
                    journal = False        # once per token
                gate = torch.sigmoid(u @ self._Db[l])        # scalar
                beta = (float(c.dnet_beta) * float(s)
                        * float(gate))
                self._DS[l].mul_(c.dnet_decay).add_(
                    torch.outer(v - o, k), alpha=beta)
                o = self._DS[l] @ k        # post-write readout
            u = torch.relu(o) if c.dnet_relu else o
        return o

    def _dnet_step(self, u1: torch.Tensor, x: int, s: float,
                   journal: bool = True) -> None:
        """One token's online update toward the token fingerprint."""
        Rm = self._Reff if self._Reff is not None else self._R
        self._dnet_write_vec(u1, Rm[x].float(), s,
                             journal=journal)

    def _dnet_write_vec(self, u1: torch.Tensor, v: torch.Tensor,
                        s: float, journal: bool = True) -> None:
        """``_dnet_step`` with an explicit fingerprint target
        (consolidation soft targets)."""
        c = self.config
        if not c.dnet:
            return
        v = v.float()
        u1 = u1.float()
        if not (bool(torch.isfinite(v).all())
                and bool(torch.isfinite(u1).all())):
            return                     # never write non-finite state
        if not math.isfinite(float(s)):
            s = 1.0
        self._dnet_layers_fwd(u1, write=True, v=v, s=s,
                              journal=journal)

    def _dnet_chunk(self, ks: torch.Tensor, xs: torch.Tensor,
                    s: torch.Tensor, v: torch.Tensor | None = None
                    ) -> None:
        """Bulk update.  ``dnet_chunk <= 0`` = exact sequential per-
        position stack steps.  Default = LaCT-style sub-chunks (same
        convention as _gsd_chunk): within each sub-chunk, per-position
        keys/gates/readouts are computed in one batch against the
        chunk-ENTRY states, then each layer takes one aggregated gated
        delta write.  Position i's contribution is pre-weighted by
        decay**(n-1-i) so the aggregate matches the sequential form
        up to the intra-chunk state evolution (the documented
        approximation).  Undo via the caller's block/session snapshot,
        journaled once per call."""
        c = self.config
        if not c.dnet or ks.shape[0] == 0:
            return
        if v is None:
            Rm = self._Reff if self._Reff is not None else self._R
            v = Rm[xs.long()].float()
        # nonfinite guards: default nan_to_num maps +-inf to +-3.4e38 -
        # STILL finite, so a poisoned _Reff target row would write a
        # 1e38 error into _DS and the next readout's inf-inf -> NaN
        # kills the layer permanently (R69-2).  Zero nonfinite instead.
        s_f = torch.nan_to_num(s.float(), nan=1.0,
                               posinf=1.0, neginf=1.0)
        ks = torch.nan_to_num(ks.float(), nan=0.0,
                              posinf=0.0, neginf=0.0)
        v = torch.nan_to_num(v.float(), nan=0.0,
                             posinf=0.0, neginf=0.0)
        bs = int(c.dnet_chunk)
        if bs <= 0:                          # exact sequential path
            for i in range(int(ks.shape[0])):
                self._dnet_write_vec(ks[i], v[i], float(s_f[i]),
                                     journal=(i == 0))
            return
        # Chunkwise delta rule (exact - the WY/triangular form of the
        # sequential recurrence, one solve per layer per sub-chunk).
        # Positions 1..nb in a sub-chunk: write i sees state after
        # writes <i, so with e_i = v_i - S_{i-1}k_i the errors obey
        #   e_i = v_i - d^{i-1}S_0k_i - sum_{j<i} d^{i-1-j} b_j (k_j.k_i) e_j
        # => (I + L)e = v - D where L[i,j] = d^{i-1-j} b_j k_j.k_i (j<i),
        # D_i = d^{i-1} S_0k_i.  Forward-substitute (unit-lower-tri),
        # then S <- d^nb S_0 + sum_i d^{nb-i} b_i e_i k_i^T.  Layer l+1's
        # input needs each position's POST-write readout:
        #   o_post_i = S_i k_i = d(v_i - e_i) + b_i e_i   (k unit-norm).
        # Cross-layer dependency stays exact because o_post is the
        # closed-form per-position output, not a batch approximation.
        self._dnet_journal()                 # covers the whole call
        c = self.config
        decay = float(c.dnet_decay)
        dev, dt = ks.device, ks.dtype
        idx = torch.arange(bs, device=dev)
        for b0 in range(0, int(ks.shape[0]), bs):
            u = ks[b0:b0 + bs]                       # [nb, du]
            vb = v[b0:b0 + bs]                       # [nb, dv]
            sb = s_f[b0:b0 + bs]                     # [nb]
            nb = int(u.shape[0])
            ii = idx[:nb]
            for l in range(len(self._DS)):
                k = u @ self._Dk[l].float()              # [nb, dk]
                k = k / k.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                beta = float(c.dnet_beta) * sb * torch.sigmoid(
                    u @ self._Db[l])                     # [nb]
                kk = k @ k.T                             # [nb, nb]
                diff = (ii[1:, None] - ii[None, :nb - 1] - 1)
                L = torch.zeros(nb, nb, device=dev, dtype=dt)
                L[1:, :nb - 1] = (decay ** diff.clamp(min=0)) \
                    * beta[:nb - 1][None, :] * kk[1:, :nb - 1]
                L = torch.tril(L, diagonal=-1)
                D = (k @ self._DS[l].T) * (
                    decay ** ii.to(dt))[:, None]   # d^{i}, i=0..nb-1
                e = torch.linalg.solve_triangular(
                    L + torch.eye(nb, device=dev, dtype=dt),
                    vb - D, upper=False)                 # [nb, dv]
                wpos = decay ** torch.arange(
                    nb - 1, -1, -1, device=dev, dtype=dt)
                self._DS[l].mul_(decay ** nb).add_(
                    (e * (beta * wpos)[:, None]).T @ k)  # [dv, dk]
                o_post = decay * (vb - e) + beta[:, None] * e
                u = torch.relu(o_post) if c.dnet_relu else o_post

    def _dnet_logits(self, q: torch.Tensor) -> torch.Tensor:
        """[V] channel readout: softmax(_Reff @ normalize(o_L)
        * temp) as log-probs - same contract as _gsyn_logits."""
        c = self.config
        o = self._dnet_layers_fwd(q, write=False)
        o = o / o.norm().clamp(min=1e-8)
        Rm = self._Reff if self._Reff is not None else self._R
        a = (Rm.float() @ o) * c.dnet_temp                  # [V]
        a = torch.nan_to_num(a.float(), nan=0.0)
        return torch.log_softmax(a, dim=-1)

    def probe_dnet(self, top: int = 8) -> dict:
        """Dissection probe: per-layer state norms, input/gate stats,
        final readout, top tokens.  Read-only."""
        c = self.config
        if not c.dnet:
            return {"enabled": False}
        q = self._gsyn_keys(self._gcell_feats(
            self._c, self._c2, self._count - 1))
        u = q.float()
        layer_stats = []
        for l in range(len(self._DS)):
            k = u @ self._Dk[l].float()
            k = k / k.norm().clamp(min=1e-8)
            o = self._DS[l] @ k
            layer_stats.append({
                "S_norm": float(self._DS[l].norm()),
                "key_norm": float(k.norm()),
                "read_norm": float(o.norm()),
                "gate": float(torch.sigmoid(u @ self._Db[l])),
            })
            u = torch.relu(o) if c.dnet_relu else o
        lp = self._dnet_logits(q)
        topv, topi = lp.topk(min(top, lp.shape[0]))
        return {
            "enabled": True,
            "layers": layer_stats,
            "out_norm": float(o.norm()),
            "top_tokens": [(int(t), float(v_))
                           for t, v_ in zip(topi.tolist(),
                                            topv.tolist())],
        }

    def _knn_live(self) -> int:
        """Bank rows actually written - the ring wraps at capacity so
        a hot bank is fully live."""
        return min(int(self._knn_cur), self.config.knn_capacity)

    # -- live features (r_live): the no-SFT feature learner ------------
    # _R rows are the model's entire feature space - a teacher graft or
    # distill_gen would inject lexical structure offline; instead the
    # stream provides it: R[x] EMAs toward the normalized pre-x context
    # sketch, so tokens seen in similar contexts converge to similar
    # rows (a co-occurrence embedding written by the same surprise-
    # gated machinery as every other channel).  Seeded bipolar rows are
    # the init; r_eta is the knob; nothing is trained.

    def _rl_write(self, x, cvec: torch.Tensor, s: float) -> None:
        """One live-feature step: R[x] <- (1-e)*R[x] + e*u where
        u = cvec/|cvec|*sqrt(d) (pre-x context sketch at the _R-row
        norm convention) and e = r_eta*s.  ``cvec`` must be the PRE-x
        context - callers pass self._c before _push/_step_dev advances
        it (the fused path reconstructs it by inverting the EMA).
        First-touch rows snapshot into the active undo sink: the open
        "bc" block's "rl" dict, the ingest/teach session dict, else a
        per-token ("rL", tok, row) entry via _jadd."""
        c = self.config
        eta = float(c.r_eta) * float(s)
        cv = cvec.float()
        if (eta <= 0.0 or self._Rl is None
                or not math.isfinite(eta)
                or not bool(torch.isfinite(cv).all())):
            return
        eta = min(eta, 1.0)      # an EMA rate > 1 is never meaningful
        u = cv / cv.norm().clamp(min=1e-8) * (c.topic_dim ** 0.5)
        xr = (x if torch.is_tensor(x)
              else torch.tensor(x)).reshape(1).long().to(self._dev)
        old = self._Rl.index_select(0, xr).squeeze(0)
        # a pre-corrupt row would re-seed NaN into every future write;
        # heal to zero so the EMA can rebuild from the next observation
        old = torch.nan_to_num(old, nan=0.0, posinf=0.0, neginf=0.0)
        blk = self._blk_active
        if blk is not None:
            blk.setdefault("rl", {}).setdefault(
                int(xr[0]), old.detach().cpu().clone())
        elif self._rl_session is not None:
            self._rl_session.setdefault(
                int(xr[0]), old.detach().cpu().clone())
        else:
            self._jadd(("rL", int(xr[0]),
                        old.detach().cpu().clone()))
        self._Rl.index_copy_(
            0, xr, ((1.0 - eta) * old + eta * u).unsqueeze(0))
        self._Reff.index_copy_(
            0, xr,
            (self._R.index_select(0, xr).float()
             + self._Rl.index_select(0, xr)))

    def _rl_bulk(self, xs: torch.Tensor, cprev: torch.Tensor,
                 s: torch.Tensor) -> None:
        """Chunked live-feature update for _ingest_bulk: grouped
        residual add of eta*s_i*(u_i - R[x_i]) per position -
        approximates the sequential EMA (duplicate rows share the
        pre-chunk row, same class as the blk["deltas"] grouping)."""
        c = self.config
        # NaN s -> 0 (skip): clamp alone propagates NaN into _Rl rows
        w = torch.nan_to_num(
            float(c.r_eta) * s.float(), nan=0.0).clamp(0.0, 1.0)
        if self._Rl is None or not bool((w > 0).any()):
            return
        u = cprev.float() / cprev.float().norm(
            dim=1, keepdim=True).clamp(min=1e-8) * (c.topic_dim ** 0.5)
        # default nan_to_num maps +-inf to +-3.4e38 - still finite, so it
        # silently overflows a stored row to inf and the NEXT touch makes
        # inf-inf -> NaN.  Zero any non-finite component instead.
        u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
        xl = xs.reshape(-1).long()
        blk = self._blk_active
        uq = torch.unique(xl)
        rows = self._Rl.index_select(0, uq).cpu()
        if blk is not None:
            sink = blk.setdefault("rl", {})
            for t, row in zip(uq.tolist(), rows):
                sink.setdefault(t, row.clone())
        elif self._rl_session is not None:
            for t, row in zip(uq.tolist(), rows):
                self._rl_session.setdefault(t, row.clone())
        else:
            for t, row in zip(uq.tolist(), rows):
                self._jadd(("rL", t, row.clone()))
        resid = w[:, None] * (u - self._Rl.index_select(0, xl))
        # already-corrupt _Rl rows produce non-finite residuals; drop
        # them so the corruption can neither spread nor re-seed itself
        resid = torch.nan_to_num(resid, nan=0.0, posinf=0.0,
                                 neginf=0.0)
        u2, tot = _group_totals(xl, resid)          # deterministic add
        new = self._Rl.index_select(0, u2) + tot
        # the grouped resid is computed against the PRE-chunk row: m
        # same-token duplicates in one chunk each contribute the full
        # w*(u - old), so a hot token ('\n' ~120x/chunk) overshoots the
        # sequential EMA's geometric limit and ping-pongs to inf at
        # multi-million-token scale (R69-2: poisoned _Reff -> inf
        # targets -> _S/_DS blowup).  Re-impose the sequential
        # invariant exactly: targets are u = unit*sqrt(d), convex EMA
        # keeps ||row|| <= sqrt(topic_dim); nonfinite heals to 0.
        new = torch.nan_to_num(new, nan=0.0, posinf=0.0, neginf=0.0)
        rn = new.norm(dim=1, keepdim=True)
        cap = float(c.topic_dim) ** 0.5
        new = new * (cap / rn.clamp(min=1e-8)).clamp(max=1.0)
        self._Rl.index_copy_(0, u2, new)
        self._Reff.index_copy_(
            0, u2, self._R.index_select(0, u2).float()
            + self._Rl.index_select(0, u2))

    def _rl_flush(self) -> None:
        """Close a live-feature write session: emit one ("rB", rows)
        entry covering every first-touched row (whole-span undo)."""
        sess, self._rl_session = self._rl_session, None
        if sess:
            self._jadd(("rB", sess))

    # -- live generalizer "gen head" (R68) ------------------------------
    # consolidate_gen's per-position transfer, extracted so the learn
    # paths can run it inline: the order cells' recalled successor
    # multiset -> the gen channels.  The head holds NO weights of its
    # own - it routes mass between existing stores, strength
    # gen_live*conf, conf = the deepest order's evidence confidence.

    def _genlive_write(self, votes: dict[int, float], conf: float,
                       code: int | None, k: torch.Tensor | None,
                       cvec: torch.Tensor | None) -> int:
        """One recall->gen transfer at the current context.

        ``votes`` = {follower: count} aggregated over evidenced order
        cells; ``conf`` = max_k tot/(tot+backoff_kappa) over the
        contributing cells; ``code``/``k``/``cvec`` = the context's
        gcell SimHash code, gsyn key, and raw context sketch (any may
        be None when its channel family is off).  Writes:
          - gcell: per-follower delta writes at s_i*share (host cells)
          - gsyn/gsd/dnet: blended soft target sum(share*Rm[t]) at s_i
          - _Rl: per-follower row EMA at s_i*share
        Journaled by the underlying write paths (G/GS/gs/gd/dn/rL/rB)
        - revert_tag/revert_writes undoes transfers exactly."""
        c = self.config
        if not votes or conf <= 0.0 or c.gen_live <= 0.0:
            return 0
        top = sorted(votes.items(), key=lambda kv: -kv[1])[
            : int(c.gen_live_topm)]
        tot_v = sum(w_ for _, w_ in top)
        if tot_v <= 0:
            return 0
        s_i = float(c.gen_live) * float(conf)
        n = 0
        if code is not None and c.gcell:
            for t, w_ in top:
                self._gcell_write(code, t, s_i * w_ / tot_v)
                n += 1
        if k is not None and (c.gsyn or c.gsyn_deep or c.dnet):
            Rm = self._Reff if self._Reff is not None else self._R
            ids = torch.tensor([t for t, _ in top], dtype=torch.long,
                               device=self._dev)
            wv = torch.tensor([w_ / tot_v for _, w_ in top],
                              dtype=torch.float32, device=self._dev)
            vt = (Rm[ids].float() * wv[:, None]).sum(0)
            if c.gsyn:
                self._gsyn_write_vec(k, vt, s_i)
            if c.gsyn_deep:
                self._gsd_write_vec(k, vt, s_i)
            if c.dnet:
                self._dnet_write_vec(k, vt, s_i)
            n += 1
        if cvec is not None and c.r_live:
            for t, w_ in top:
                self._rl_write(t, cvec, s_i * w_ / tot_v)
                n += 1
        return n

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
            fp = self._knn_fp_ctx(self._count - 1)
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
            upd = cur * (1.0 - lr_eff) + ax * lr_eff
            # fp16 range guard: _A magnitudes can exceed 65504; rescale
            # direction-preservingly so .half() cannot produce inf
            upd = torch.nan_to_num(upd)
            upd = upd * (6e4 / upd.abs().max().clamp(min=6e4))
            self._csem_rows[s] = upd.half()
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
