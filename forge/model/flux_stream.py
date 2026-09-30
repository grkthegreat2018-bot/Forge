"""FluxStreamPool — multi-stream batched prediction over shared FluxLM
memory, with a single serialized learning process (update packets).

A FluxLM's learned state is READ-ONLY at predict time: n-gram cell
tables, the episodic index, the knn bank, topic/semantic matrices and the
unigram prior are shared across any number of concurrent contexts.  What
a sequence actually owns is its CONTEXT — position, suffix-hash tail,
seen-set, topic/proto EMAs, hedge weights, unigram counts — ~1 MB per
stream, the FluxLM analogue of a per-sequence KV cache.

This module exploits that: ``FluxStreamPool`` runs N independent token
streams against the canonical model's shared memory with per-stream
context state — one batched [N, V] logits step per round instead of N
cloned models.  Streams NEVER write shared memory: no table cells, no
episodic entries, no journal, no revert.  What a stream "would have
learned" is exported as a :class:`FluxPacket` (prompt + generated ids +
tag); the caller decides which packets the single learning process
applies via :meth:`FluxLM.apply_packets` — the LLM-batching analogue of
"one writer, many readers".

Semantics vs the old clone-worker path:
  * per-stream soft_reset at fork: ep_start = model._count, cleared
    seen-set / context EMA, canonical hedge/proto/unigram init;
  * stream tokens live in a per-stream ring/hash tail (positions >= C);
    reads below C fall through to the canonical ring — the same bytes a
    clone's ring would hold;
  * a stream's OWN writes do not feed its predictions (no self-boost
    from cells it just wrote — recency/fatigue still capture repeats);
    the packet applies them post-hoc through ``ingest`` instead;
  * hedge, unigram, seen, topic/proto EMAs track per stream — clone
    parity for all context state.

Backends:
  * ``model._gpu`` (cuda_primary): fully batched tensor path — one
    probe/logits pass per step for all streams, rewards for the sampled
    token are gathered from the same pass (predict-only streams need no
    second probe — halving the live step's probe count).
  * host model: per-stream ``_StreamCtx`` facades reuse FluxLM's own
    ``_learn_token``/``_predict`` with ``_predict_only`` gating — exact
    parity, and only ring/H copies + scalar state per stream.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch

from forge.model.flux import _MIX1, _mix64

_P = 1099511628211          # ring prefix-hash multiplier


@dataclass
class FluxPacket:
    """One stream's learnable payload — an update package for the
    canonical model's single learning process.

    ``ids`` = prompt + generated tokens in stream order.  Applying it
    (``FluxLM.apply_packets``) is a journaled ``ingest`` under ``tag`` —
    revertible via ``revert_tag`` and reinforceable via
    ``reinforce(tag, gain)``.
    """

    tag: str
    prompt_ids: list[int]
    gen_ids: list[int]
    meta: dict = field(default_factory=dict)
    # counterfactual propensities: per generated token, the top-k
    # (token_id, prob) alternatives under the actual sampling
    # distribution - lets reinforce_alts credit untried actions
    alts: list | None = None

    @property
    def ids(self) -> list[int]:
        return self.prompt_ids + self.gen_ids


# ── CPU path: per-stream context facade ──────────────────────────────────


class _NullJournal:
    """Sink for journaled writes inside predict-only streams."""

    def __init__(self):
        # instance attr — a class-level list would share state across
        # every stream that ever touches .entries
        self.entries: list = []

    def append(self, e) -> None:
        return None

    def append_w(self, *a, **k):
        return ("w",)

    def append_vec(self, *a, **k):
        return ("v",)


class _StreamCtx:
    """Predict-only view of a FluxLM: shared learned memory is READ
    (tables, epi index, knn bank, A/proto matrices fall through to the
    model); stream-private state is held in ``_own``.

    ``_predict_only`` gates the shared-memory writes inside
    ``_learn_token``; ``learning`` is False so ``_push`` skips epi/knn
    memory writes while still maintaining ring/hash/context state.
    """

    def __init__(self, model):
        c = model.config
        own = {
            "learning": False,
            "_predict_only": True,
            "tag": "stream",
            "journal": _NullJournal(),
            "_ep_start": model._count,
            "_count": model._count,
            "_c": torch.zeros(c.topic_dim, device=model._dev),
            "_proto": model._proto.clone(),
            "_w": model._w.copy(),
            "_hedge_mix": float(model._hedge_mix),
            "_wR": model._wR.copy(),
            "_hmxR": model._hmxR.copy(),
            "_regime": int(model._regime),
            "_uni": model._uni.copy(),
            "_total": float(model._total),
            "_log_uni_num": model._log_uni_num.clone(),
            "_last_seen": {},
            "_recent": deque(),
            "_gaps": [],
            "_ring": model._ring.copy(),
            "_H": model._H.copy(),
        }
        object.__setattr__(self, "_own", own)
        object.__setattr__(self, "_model", model)

    def __getattr__(self, name):
        try:
            return object.__getattribute__(self, "_own")[name]
        except KeyError:
            m = object.__getattribute__(self, "_model")
            v = getattr(m, name)
            # rebind model methods so internals run on THIS context —
            # otherwise self._push()/self._deep_votes() inside
            # _learn_token/_predict would read and write the canonical
            # model's ring/count, not the stream's
            if getattr(v, "__self__", None) is m \
                    and getattr(v, "__func__", None) is not None:
                import types
                return types.MethodType(v.__func__, self)
            return v

    def __setattr__(self, name, value):
        self._own[name] = value


# ── the pool ─────────────────────────────────────────────────────────────


class FluxStreamPool:
    """N predict-only streams sharing one FluxLM's memory.

    ``generate`` is the whole loop: lockstep prompt feed → per-step
    batched logits → per-stream sampling → packets.  Streams start at the
    canonical position with a fresh episode and never mutate the model.
    """

    def __init__(self, model, stream_tail: int = 1024):
        self.model = model
        self.dev = model._dev
        self._gpu = bool(model._gpu)
        # per-stream ring/hash tail: reads only reach ~max(cert_orders,
        # deep_order+deep_max_extend, max(orders)) ≈ 104 back — 1024 is
        # ample headroom; must satisfy cap_s > max read span.
        self.cap_s = stream_tail
        self.N = 0
        self.C = 0

    # ── state init ───────────────────────────────────────────────────

    def reset(self, n: int) -> None:
        m = self.model
        self.N = n
        self.C = m._count
        if self._gpu:
            self._reset_gpu(n)
        else:
            self._ctxs = [_StreamCtx(m) for _ in range(n)]

    def _reset_gpu(self, n: int) -> None:
        m, c, dev = self.model, self.model.config, self.dev
        V = m.vocab_size
        self.sring = torch.zeros(n, self.cap_s, dtype=torch.int32,
                                 device=dev)
        self.shash = torch.zeros(n, self.cap_s, dtype=torch.int64,
                                 device=dev)
        self.sseen = torch.full((n, V), -10 ** 9, dtype=torch.int64,
                                device=dev)
        self.sunig = m._unig.unsqueeze(0).expand(n, -1).contiguous()
        self.sloguni = m._log_uni_num.unsqueeze(0).expand(n, -1) \
            .contiguous()
        self.stot = m._tot_g.reshape(1).expand(n).clone()
        self.sc = torch.zeros(n, c.topic_dim, device=dev)
        self.sproto = m._proto.unsqueeze(0).expand(n, -1).contiguous()
        self.sw = m._wg.unsqueeze(0).expand(n, -1).clone()  # [N,C] f64
        self.shmg = m._hmg.reshape(1).expand(n).clone()
        # per-regime weight sets (cond_hedge): sw stays the active-row
        # mirror for parity with the canonical model's _w
        self.swR = m._wgR.unsqueeze(0).expand(n, -1, -1).clone()
        self.shmgR = m._hmxRg.reshape(1, -1).expand(n, -1).clone()
        self.sreg = torch.zeros(n, dtype=torch.int64, device=dev)
        # sparse-tap abstain mask (host tables - no device reward)
        self._pmask_dev = torch.from_numpy(m._pmask).to(dev)
        self.spos = torch.full((n,), self.C, dtype=torch.int64,
                               device=dev)
        self.sep = torch.full((n,), self.C, dtype=torch.int64,
                              device=dev)
        self._ar = torch.arange(n, dtype=torch.int64, device=dev)

    # ── hybrid reads: pos >= C -> stream tail, pos < C -> canonical ────

    def _bidx(self, pos: torch.Tensor) -> torch.Tensor:
        """Flat index into the [N*cap_s] tail buffer for pos [N,*]."""
        n = pos.shape[0]
        sidx = (pos - self.C) % self.cap_s
        return (self._ar.view(n, *([1] * (pos.dim() - 1)))
                .expand_as(pos) * self.cap_s + sidx)

    def _h_at(self, pos: torch.Tensor) -> torch.Tensor:
        """Prefix hash at absolute ``pos`` [N,*] — stream tail for
        pos >= C, canonical ring for pos < C."""
        m = self.model
        canon = m._Hg[(pos % m.config.ring_capacity).clamp(min=0)]
        stream = self.shash.view(-1).index_select(
            0, self._bidx(pos).reshape(-1)).reshape_as(pos)
        return torch.where(pos >= self.C, stream, canon)

    def _t_at(self, pos: torch.Tensor) -> torch.Tensor:
        """Token at absolute ``pos`` [N,*] — stream tail or canonical
        ring (epi candidates are canonical positions < C)."""
        m = self.model
        canon = m._ringg[(pos % m.config.ring_capacity).clamp(min=0)]
        stream = self.sring.view(-1).index_select(
            0, self._bidx(pos).reshape(-1)).reshape_as(pos)
        return torch.where(pos >= self.C, stream, canon)

    def _keys_b(self, end: torch.Tensor):
        """Order keys [N,O] + validity [N,O] at context ``end`` [N]."""
        m, c = self.model, self.model.config
        ks = m._orders_g                                        # [O]
        he = self._h_at(end)                                    # [N]
        keys = he[:, None] - self._h_at(end[:, None]
                                       - ks[None, :]) \
            * m._pows_g[ks][None, :]
        valid = (end[:, None] - ks[None, :] + 1 >= self.sep[:, None]) \
            & (end[:, None] >= ks[None, :] - 1) & (end[:, None] >= 0)
        return keys, valid

    def _suffix_b(self, k: int, end: torch.Tensor) -> torch.Tensor:
        """Raw order-k suffix hash at ``end`` [N] -> [N]."""
        return self._h_at(end) - self._h_at(end - k) \
            * self.model._pows_g[k]

    # ── batched probe + logits (ports of _probes_at/_logits_dev) ─────

    def _probe_b(self, end: torch.Tensor):
        """Batched ``_probes_at`` -> (slot, found) [N,R], live_k [N,O],
        cok/dok [N], dk [N]."""
        m, c = self.model, self.model.config
        gt = m._gt
        n = end.shape[0]
        keys, live_k = self._keys_b(end)                        # [N,O]
        ck = self._suffix_b(c.csem_order, end)                  # [N]
        dk = self._suffix_b(c.deep_order, end)                  # [N]
        allk = torch.cat([keys, ck[:, None], dk[:, None]], 1)   # [N,R]
        R = gt.R
        pk = _mix64(allk * _MIX1 + gt._salt[:R][None, :])       # [N,R]
        rows = gt._ord_t[:R][None, :].expand(n, R).reshape(-1)
        slot, found = gt.probe_fixed(pk.reshape(-1), rows=rows)
        cok = (end - c.csem_order + 1 >= self.sep) \
            & (end >= c.csem_order - 1) & (end >= 0)
        dok = (end - c.deep_order + 1 >= self.sep) \
            & (end >= c.deep_order - 1) & (end >= 0)
        return (slot.reshape(n, R), found.reshape(n, R), live_k,
                cok, dok, dk)

    def _deep_votes_b(self, end: torch.Tensor, floor: torch.Tensor,
                      se: torch.Tensor, fe: torch.Tensor,
                      dk: torch.Tensor) -> torch.Tensor:
        """Batched episodic votes -> [N,V] (eager port of
        ``_deep_votes_dev``; the triton kernel is single-stream)."""
        m, c, dev = self.model, self.model.config, self.dev
        n = end.shape[0]
        V = m.vocab_size
        DO, mx = c.deep_order, c.deep_max_extend
        cap = c.ring_capacity
        pos_row = m._epi_pos.index_select(0, se)                # [N,ER]
        valid = ((end - DO + 1 >= self.sep)
                 & (end >= DO - 1) & (end >= 0))                # [N]
        live = fe[:, None] & (pos_row >= 0) \
            & (pos_row < end[:, None]) & valid[:, None]
        on_ring = live & (pos_row >= floor[:, None])
        hv = m._Hg[pos_row.clamp(min=0) % cap] \
            - m._Hg[(pos_row - DO).clamp(min=0) % cap] \
            * m._pows_g[DO]
        on_ring = on_ring & (hv == dk[:, None])
        ms = torch.arange(1, mx + 1, dtype=torch.int64, device=dev)
        pi = pos_row[:, None, :] - ms[None, :, None]            # [N,mx,ER]
        ei = end[:, None] - ms[None, :]                         # [N,mx]
        ok = (pi >= floor[:, None, None]) & (pi >= 0) \
            & (ei[:, :, None] >= self.sep[:, None, None]) \
            & (ei[:, :, None] >= 0)
        a = m._ringg[pi.clamp(min=0) % cap]                     # [N,mx,ER]
        b = self._t_at(ei.clamp(min=0))                         # [N,mx]
        eq = (a.long() == b[:, :, None].long()) & ok
        runlen = eq.long().cumprod(1).sum(1)                    # [N,ER]
        mlen_ring = torch.where(on_ring, runlen + DO,
                                torch.zeros_like(runlen))
        arch = live & (pos_row < floor[:, None])
        base_w = torch.full((), DO, dtype=torch.int64,
                            device=dev)
        mlen_arch = torch.where(arch, base_w.expand_as(runlen),
                                torch.zeros_like(runlen))
        LC = len(c.cert_orders)
        if LC:
            certs = m._epi_cert.index_select(0, se)             # [N,ER,LC]
            cur = torch.stack(
                [self._suffix_b(L, end) for L in c.cert_orders],
                dim=1)                                          # [N,LC]
            cmatch = (certs == cur[:, None, :]) & (certs != 0)
            cw = (cmatch.long()
                  * m._cert_g[None, None, :]).amax(dim=2)       # [N,ER]
            mlen_arch = torch.where(
                arch, torch.maximum(cw, base_w),
                torch.zeros_like(runlen))
        succ = m._epi_succ.index_select(0, se).long() \
            .clamp(0, V - 1)                                    # [N,ER]
        have_succ = m._epi_succ.index_select(0, se) >= 0
        mlen = torch.where(have_succ, mlen_ring + mlen_arch,
                           torch.zeros_like(runlen))
        votes = torch.zeros(n, V, dtype=torch.float32, device=dev)
        votes.view(-1).index_add_(
            0, (self._ar[:, None] * V + succ).reshape(-1),
            mlen.reshape(-1).float())
        return votes

    def _knn_votes_b(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k fingerprint neighbours vote their successors —
        shared frozen bank -> (kv [N,V], kvt [N])."""
        m, c, dev = self.model, self.model.config, self.dev
        n = self.N
        V = m.vocab_size
        kv = torch.zeros(n, V, dtype=torch.float32, device=dev)
        kb = m._knn_rows()
        if not (c.use_knn and kb > 0):
            return kv, torch.zeros(n, dtype=torch.float32,
                                   device=dev)
        qk = self.sc @ m._Pcsem                                 # [N,64]
        qkn = qk.norm(dim=1, keepdim=True)
        fp = (qk / (qkn + 1e-8)).to(m._knn_fp.dtype)            # [N,64]
        ksims = (fp @ m._knn_fp[:kb].T).float()                 # [N,kb]
        ksims = torch.where((m._knn_pos[:kb] >= 0)[None, :],
                            ksims, torch.full((), -1e9,
                                              device=dev))
        top = torch.topk(ksims, min(c.knn_k, kb), dim=1)
        keep = top.values > c.knn_min_sim
        wv = (top.values.clamp(min=0.0) ** c.knn_gamma) * keep  # [N,k]
        ksucc = m._knn_succ.index_select(
            0, top.indices.reshape(-1)).reshape(n, -1) \
            .long().clamp(0, V - 1)                             # [N,k]
        kv.view(-1).index_add_(
            0, (self._ar[:, None] * V + ksucc).reshape(-1),
            wv.reshape(-1))
        return kv, kv.sum(dim=1)

    def _logits_b(self, end: torch.Tensor, slot: torch.Tensor,
                  found: torch.Tensor, live_k: torch.Tensor,
                  cok: torch.Tensor, dok: torch.Tensor,
                  dk: torch.Tensor, votes: torch.Tensor,
                  kv: torch.Tensor, kvt: torch.Tensor
                  ) -> torch.Tensor:
        """Batched ``_logits_dev`` -> [N,V].  ``votes``/``kv``/``kvt``
        are passed in so the SAME probe feeds both logits and the
        reward gather (predict-only streams need one probe per token —
        the live step pays two because its writes land between)."""
        m, c, dev = self.model, self.model.config, self.dev
        gt = m._gt
        n = end.shape[0]
        V = m.vocab_size
        O = gt.O
        denom = (self.stot + c.uni_alpha * V).float()           # [N]
        uni_pv = (self.sunig + c.uni_alpha) / denom[:, None]
        log_uni = self.sloguni - torch.log(denom)[:, None]
        logits = log_uni.clone()                                # [N,V]

        oslot, ofound = slot[:, :O], found[:, :O]
        hit = ofound & live_k                                   # [N,O]
        frow = gt._ordx[None, :] * gt.S + oslot                 # [N,O]
        rows_t = gt.ttoks.view(-1, gt.M)[frow]                  # [N,O,M]
        rows_c = gt.tcnt.view(-1, gt.M)[frow]
        tots = gt.ttot.view(-1)[frow]                           # [N,O]
        if c.cond_hedge:
            # per-stream evidence regime; swR is canonical, sw the
            # active-row mirror (same convention as _w/_wR)
            self.sreg = (hit & (tots > 0)).any(dim=1).long()
            sw = self.swR[self._ar, self.sreg]                  # [N,C]
            self.sw = sw
        else:
            sw = self.sw
        wo = sw[:, :O].float()[:, :, None]                      # [N,O,1]
        if c.evidence_lift:
            wo = wo + (tots / (tots + c.backoff_kappa))[:, :, None]
        tok_l = rows_t.long().clamp(0, V - 1)                   # [N,O,M]
        row_live = (rows_t >= 0) & (rows_t < V) & hit[:, :, None]
        flat_tl = tok_l.reshape(n, -1)                          # [N,O*M]
        uni_cell = uni_pv.gather(1, flat_tl).reshape(n, O, gt.M)
        luni_cell = log_uni.gather(1, flat_tl).reshape(n, O, gt.M)
        p = (rows_c + c.alpha * uni_cell) \
            / (tots[:, :, None] + c.alpha)
        contrib = wo * (torch.log(p.clamp(min=1e-12)) - luni_cell)
        contrib = torch.where(row_live, contrib,
                              torch.zeros((), device=dev))
        logits.view(-1).index_add_(
            0, (self._ar[:, None, None] * V + tok_l).reshape(-1),
            contrib.reshape(-1))

        # backoff channel: longest evidenced order per stream
        if c.use_backoff:
            ok_bo = hit & (tots >= c.backoff_min_tot)           # [N,O]
            kscore = torch.where(
                ok_bo, m._orders_g.float()[None, :].expand(n, O),
                torch.full((), -1.0, device=dev))
            kstar = kscore.argmax(dim=1)                        # [N]
            bo_ok = ok_bo.gather(1, kstar[:, None]).squeeze(1)
            fs = frow.gather(1, kstar[:, None]).squeeze(1)      # [N]
            rt_k = gt.ttoks.view(-1, gt.M)[fs]                  # [N,M]
            rc_k = gt.tcnt.view(-1, gt.M)[fs]
            tot_k = gt.ttot.view(-1)[fs]                        # [N]
            conf = tot_k / (tot_k + c.backoff_kappa)
            tl = rt_k.long().clamp(min=0)                       # [N,M]
            rl = rt_k >= 0
            p_b = conf[:, None] * (rc_k + c.alpha
                                   * uni_pv.gather(1, tl)) \
                / (tot_k[:, None] + c.alpha)
            wb = sw[:, m._chix["backoff"]].float()
            if c.evidence_lift:
                wb = wb + conf
            logits.view(-1).index_add_(
                0, (self._ar[:, None] * V + tl).reshape(-1),
                torch.where(
                    rl & bo_ok[:, None],
                    wb[:, None] * (torch.log(p_b.clamp(min=1e-12))
                                   - log_uni.gather(1, tl)),
                    torch.zeros((), device=dev)).reshape(-1))

        # deep episodic channel (votes computed by caller's probe)
        vt = votes.sum(dim=1)                                   # [N]
        wd = sw[:, m._chix["deep"]].float()
        if c.evidence_lift:
            wd = wd + vt / (vt + c.backoff_kappa)
        p_d = (votes + c.alpha * uni_pv) / (vt[:, None] + c.alpha)
        logits += torch.where(
            (vt > 0)[:, None],
            wd[:, None] * (torch.log(p_d.clamp(min=1e-12))
                           - log_uni),
            torch.zeros((), device=dev))

        # topic channel: cosine readout on top candidates per stream
        cn = self.sc.norm(dim=1)                                # [N]
        wt = sw[:, m._chix["topic"]].float()
        cand = torch.topk(logits, min(256, V), dim=1).indices   # [N,T]
        a = m._A[cand]                                          # [N,T,D]
        an = m._An.index_select(0, cand.reshape(-1)) \
            .reshape(n, -1)
        cos = (a * self.sc[:, None, :]).sum(-1) \
            / (an * cn[:, None] + 1e-8)                         # [N,T]
        logits.scatter_add_(
            1, cand,
            torch.where((cn > 1e-8)[:, None],
                        (wt * c.topic_beta)[:, None] * cos,
                        torch.zeros((), device=dev)))

        # sem + csem share _Ac = A @ Pcsem (rank-1 maintained)
        pv = self.sproto @ m._Pcsem                             # [N,64]
        pvn = pv.norm(dim=1)                                    # [N]
        sims = (pv @ m._Ac.T) / (m._Acn[None, :]
                                 * pvn[:, None] + 1e-8)         # [N,V]
        ws = sw[:, m._chix["sem"]].float() * float(c.use_sem)
        logits += torch.where(
            (pvn > 1e-8)[:, None],
            (ws * c.sem_beta)[:, None] * sims,
            torch.zeros((), device=dev))

        cs = slot[:, gt.row_csem]                               # [N]
        cf = found[:, gt.row_csem]
        row = m._csem_rows.index_select(0, cs).float()          # [N,64]
        rn = row.norm(dim=1)
        csims = (row @ m._Ac.T) / (m._Acn[None, :]
                                   * rn[:, None] + 1e-8)
        wc = sw[:, m._chix["csem"]].float() \
            * float(c.use_csem)
        logits += torch.where(
            (cf & cok & (rn > 1e-8))[:, None],
            (wc * c.csem_beta)[:, None] * csims,
            torch.zeros((), device=dev))

        # knn channel (votes passed in — shared with reward gather)
        if c.use_knn:
            p_k = (kv + c.alpha * uni_pv) / (kvt[:, None]
                                             + c.alpha)
            wk = sw[:, m._chix["knn"]].float()
            if c.evidence_lift:
                wk = wk + kvt / (kvt + c.backoff_kappa)
            logits += torch.where(
                (kvt > 0)[:, None],
                wk[:, None] * (torch.log(p_k.clamp(min=1e-12))
                               - log_uni),
                torch.zeros((), device=dev))

        # recency + fatigue over [N,V]
        ages = (end[:, None] - self.sseen).clamp(min=0)         # [N,V]
        rec = torch.where(
            (ages < c.recency_span) & (self.sseen >= 0),
            c.recency_decay ** ages.float(),
            torch.zeros((), device=dev))
        fat = torch.where(
            (ages < c.fatigue_span) & (self.sseen >= 0),
            c.fatigue_decay ** ages.float(),
            torch.zeros((), device=dev))
        logits += sw[:, m._chix["recency"]].float()[:, None] \
            * rec \
            - (sw[:, m._chix["fatigue"]].float()
               * c.fatigue_strength)[:, None] * fat
        if c.uni_escape > 0:
            logits = torch.log(
                (1.0 - c.uni_escape) * torch.softmax(logits, -1)
                + c.uni_escape * uni_pv)
        return logits

    # ── batched reward (port of _step_dev's hedge section) ───────────

    def _rewards_b(self, x: torch.Tensor, slot: torch.Tensor,
                   found: torch.Tensor, live_k: torch.Tensor,
                   cok: torch.Tensor, votes: torch.Tensor,
                   kv: torch.Tensor, kvt: torch.Tensor,
                   end: torch.Tensor) -> torch.Tensor:
        """Per-channel prob of x -> [N,Cch] fp64.  Same math as
        ``_step_dev``'s reward block; channel tensors are the ones the
        predict pass already computed (no second probe)."""
        m, c, dev = self.model, self.model.config, self.dev
        gt = m._gt
        n = x.shape[0]
        V = m.vocab_size
        O = gt.O
        Cch = len(m.channels)
        rw = torch.empty(n, Cch, dtype=torch.float64, device=dev)
        denom = (self.stot + c.uni_alpha * V).double()          # [N]
        uni_x = (self.sunig.gather(1, x[:, None]).squeeze(1)
                 + c.uni_alpha).double() / denom                # [N]

        frow = gt._ordx[None, :] * gt.S + slot[:, :O]
        rows_t = gt.ttoks.view(-1, gt.M)[frow]
        rows_c = gt.tcnt.view(-1, gt.M)[frow]
        tots = gt.ttot.view(-1)[frow]                           # [N,O]
        hit = found[:, :O] & live_k
        cnt_x = (rows_c.double()
                 * (rows_t == x[:, None, None].to(torch.int32))
                 ).sum(dim=2)                                   # [N,O]
        p_o = torch.where(
            hit & (tots > 0),
            (cnt_x + c.alpha * uni_x[:, None])
            / (tots.double() + c.alpha),
            uni_x[:, None].expand(n, O))
        rw[:, :O] = torch.where(live_k, p_o,
                                uni_x[:, None].expand(n, O))

        vt = votes.sum(dim=1).double()
        vx = votes.gather(1, x[:, None]).squeeze(1).double()
        rw[:, m._chix["deep"]] = torch.where(
            vt > 0, (vx + c.alpha * uni_x) / (vt + c.alpha),
            uni_x)

        a_x = m._A.index_select(0, x)                           # [N,D]
        cn = self.sc.norm(dim=1)
        cos_t = (a_x * self.sc).sum(1) \
            / (a_x.norm(dim=1) * cn + 1e-8)
        if c.calib_rewards:
            # normalized implied dist p(v) ∝ p_uni(v)·exp(β·cos_v):
            # reward = p_uni(x)·exp(β·cos_x)/Z per stream.  Z needs the
            # [N,V] sims - one matvec per channel (A @ vec.T).
            uni_pv = (self.sunig + c.uni_alpha) \
                / (self.stot + c.uni_alpha * V)[:, None]        # [N,V]
            chat = self.sc / cn.clamp(min=1e-8)[:, None]
            sims_t = (m._A @ chat.T) \
                / m._An.clamp(min=1e-8)[:, None]                # [V,N]
            z_t = (uni_pv * torch.exp(
                c.topic_beta * sims_t.T)).sum(1)                # [N]
            rw[:, m._chix["topic"]] = torch.where(
                (cn > 1e-8) & (a_x.norm(dim=1) > 1e-8),
                (uni_x * torch.exp(c.topic_beta * cos_t)
                 / z_t.clamp(min=1e-12)), uni_x)
        else:
            rw[:, m._chix["topic"]] = torch.sigmoid(
                c.topic_beta * cos_t).double()
        pn = self.sproto.norm(dim=1)
        cos_s = (a_x * self.sproto).sum(1) \
            / (a_x.norm(dim=1) * pn + 1e-8)
        if c.calib_rewards:
            phat = self.sproto / pn.clamp(min=1e-8)[:, None]
            sims_s = (m._A @ phat.T) \
                / m._An.clamp(min=1e-8)[:, None]
            z_s = (uni_pv * torch.exp(
                c.sem_beta * sims_s.T)).sum(1)
            rw[:, m._chix["sem"]] = torch.where(
                (pn > 1e-8) & (a_x.norm(dim=1) > 1e-8),
                (uni_x * torch.exp(c.sem_beta * cos_s)
                 / z_s.clamp(min=1e-12)), uni_x)
        else:
            rw[:, m._chix["sem"]] = torch.sigmoid(
                c.sem_beta * cos_s).double()

        cs = slot[:, gt.row_csem]
        cf = found[:, gt.row_csem]
        ax_c = (a_x @ m._Pcsem).double()                        # [N,64]
        row = m._csem_rows.index_select(0, cs).double()
        rn = row.norm(dim=1)
        cos_c = (ax_c * row).sum(1) \
            / (ax_c.norm(dim=1) * rn + 1e-8)
        if c.calib_rewards:
            rhat = (row / rn.clamp(min=1e-8)[:, None]).float()
            csims = (m._Ac @ rhat.T) \
                / m._Acn.clamp(min=1e-8)[:, None]               # [V,N]
            z_c = (uni_pv * torch.exp(
                c.csem_beta * csims.T)).sum(1)
            rw[:, m._chix["csem"]] = torch.where(
                cf & cok & (rn > 1e-8),
                (uni_x * torch.exp(c.csem_beta * cos_c)
                 / z_c.clamp(min=1e-12)).double(), uni_x)
        else:
            rw[:, m._chix["csem"]] = torch.where(
                cf & cok & (rn > 1e-8),
                torch.sigmoid(c.csem_beta * cos_c).double(), uni_x)

        if c.use_knn:
            kvd = kv.double()
            kvx = kvd.gather(1, x[:, None]).squeeze(1)
            ktd = kvt.double()
            rw[:, m._chix["knn"]] = torch.where(
                ktd > 0, (kvx + c.alpha * uni_x)
                / (ktd + c.alpha), uni_x)
        else:
            rw[:, m._chix["knn"]] = uni_x

        age = (end - self.sseen.gather(
            1, x[:, None]).squeeze(1)).double()                 # [N]
        if c.calib_rewards:
            # same implied-dist reward as _hedge_dev: p_uni·e^f / Z,
            # Z = 1 + Σ_v p_uni·(e^f - 1) over each stream's window
            ages_v = end[:, None] - self.sseen                  # [N,V]
            rec_v = torch.where(
                (ages_v >= 0) & (ages_v < c.recency_span)
                & (self.sseen >= 0),
                c.recency_decay ** ages_v.float().clamp(min=0),
                torch.zeros((), device=dev)).double()
            z_r = 1.0 + (uni_pv * (torch.exp(rec_v) - 1.0)
                         ).sum(dim=1)                           # [N]
            inrec = (age >= 0) & (age < c.recency_span)
            rec_x = torch.where(
                inrec, c.recency_decay ** age.clamp(min=0),
                torch.zeros((), dtype=torch.float64, device=dev))
            rw[:, m._chix["recency"]] = uni_x * torch.exp(rec_x) \
                / z_r.clamp(min=1e-12)
            fat_v = torch.where(
                (ages_v >= 0) & (ages_v < c.fatigue_span)
                & (self.sseen >= 0),
                c.fatigue_decay ** ages_v.float().clamp(min=0),
                torch.zeros((), device=dev)).double()
            z_f = 1.0 + (uni_pv * (torch.exp(
                -c.fatigue_strength * fat_v) - 1.0)).sum(dim=1)
            infat = (age >= 0) & (age < c.fatigue_span)
            fat_x = torch.where(
                infat, c.fatigue_decay ** age.clamp(min=0),
                torch.zeros((), dtype=torch.float64, device=dev))
            rw[:, m._chix["fatigue"]] = uni_x * torch.exp(
                -c.fatigue_strength * fat_x) \
                / z_f.clamp(min=1e-12)
        else:
            inrec = (age >= 0) & (age < c.recency_span)
            rw[:, m._chix["recency"]] = torch.where(
                inrec, c.recency_decay ** age.clamp(min=0),
                torch.full((), 1e-3, dtype=torch.float64,
                           device=dev))
            infat = (age >= 0) & (age < c.fatigue_span)
            rw[:, m._chix["fatigue"]] = torch.where(
                infat, 1.0 - c.fatigue_decay ** age.clamp(min=0),
                torch.ones((), dtype=torch.float64, device=dev))
        rw[:, m._chix["uni"]] = uni_x

        if c.use_backoff:
            ok_bo = hit & (tots >= c.backoff_min_tot)
            kscore = torch.where(
                ok_bo, m._orders_g.double()[None, :].expand(n, O),
                torch.full((), -1.0, dtype=torch.float64,
                           device=dev))
            kstar = kscore.argmax(dim=1)
            bo_hit = ok_bo.gather(1, kstar[:, None]).squeeze(1)
            tot_k = tots.gather(1, kstar[:, None]) \
                .squeeze(1).double()
            cnt_k = cnt_x.gather(1, kstar[:, None]).squeeze(1)
            conf = tot_k / (tot_k + c.backoff_kappa)
            rw[:, m._chix["backoff"]] = torch.where(
                bo_hit,
                conf * (cnt_k + c.alpha * uni_x)
                / (tot_k + c.alpha), uni_x)
        else:
            ok_bo = None
            rw[:, m._chix["backoff"]] = uni_x

        # sleeping-experts awake mask (abstention): a channel with no
        # evidence this step was scored on the unigram fallback - mark
        # it asleep so observe_gpu can give it the neutral p_mix reward
        awake = None
        if c.sleeping_experts:
            awake = torch.ones(n, Cch, dtype=torch.bool, device=dev)
            awake[:, :O] = hit & (tots > 0)
            awake[:, m._chix["deep"]] = vt > 0
            awake[:, m._chix["topic"]] = (cn > 1e-8) & (
                a_x.norm(dim=1) > 1e-8)
            awake[:, m._chix["sem"]] = (pn > 1e-8) & (
                a_x.norm(dim=1) > 1e-8)
            awake[:, m._chix["csem"]] = cf & cok & (rn > 1e-8)
            awake[:, m._chix["knn"]] = (
                kvt > 0 if c.use_knn
                else torch.zeros(n, dtype=torch.bool, device=dev))
            awake[:, m._chix["backoff"]] = (
                ok_bo.any(dim=1) if ok_bo is not None
                else torch.zeros(n, dtype=torch.bool, device=dev))
        for i in m._pidx:
            # host-resident tables (sparse p<j>, class g<k>) - pool
            # streams take the unigram fallback (matches _step_dev)
            rw[:, i] = uni_x
            if awake is not None:
                awake[:, i] = False
        return rw, awake

    # ── stream steps ─────────────────────────────────────────────────

    def observe_gpu(self, x: torch.Tensor, mask: torch.Tensor,
                    rewards: torch.Tensor | None,
                    awake: torch.Tensor | None = None) -> None:
        """Push x [N] into each stream's context.  ``rewards`` = the
        [N,Cch] channel probs gathered at the pre-x context (None =
        skip hedge update, cheap prompt feed).  ``awake`` = [N,Cch]
        sleeping-experts mask - asleep channels take the mixture's own
        p_mix as a neutral reward instead of their unigram fallback.
        Dead lanes write their own slot back unchanged."""
        m, c = self.model, self.model.config
        dev = self.dev
        end = self.spos - 1                                     # [N]

        if rewards is not None:
            if c.cond_hedge:
                # sreg was set by the predicting _logits_b - rewards
                # update that regime's own weight set, written back
                # into swR under the learn mask
                sw = self.swR[self._ar, self.sreg]              # [N,C]
                shmg = self.shmgR[self._ar, self.sreg]          # [N]
            else:
                sw, shmg = self.sw, self.shmg
            if awake is not None and c.sleeping_experts == 2:
                aw = awake.double()
                wsum = (sw * aw).sum(dim=1).clamp(min=1e-12)
                pmix = (sw * rewards * aw).sum(dim=1) / wsum
                rewards = torch.where(awake, rewards,
                                      pmix[:, None].expand_as(rewards))
            elif awake is not None and c.sleeping_experts == 1:
                pmix = (sw * rewards).sum(dim=1)
                rewards = torch.where(awake, rewards,
                                      pmix[:, None].expand_as(rewards))
            elif m._pidx:
                # host-resident arms abstain on GPU: neutral
                # pmix reward over the other arms (closed form)
                aw = ~self._pmask_dev
                wsum = (sw * aw).sum(dim=1).clamp(min=1e-12)
                pmix = (sw * rewards * aw).sum(dim=1) / wsum
                rewards = torch.where(
                    self._pmask_dev[None, :].expand_as(rewards),
                    pmix[:, None].expand_as(rewards), rewards)
            else:
                pmix = (sw * rewards).sum(dim=1)           # [N]
            if c.adahedge:
                shmg = shmg + torch.where(
                    mask, rewards.max(dim=1).values - pmix,
                    torch.zeros((), dtype=torch.float64,
                              device=dev))
                eta = torch.minimum(
                    torch.full((), c.hedge_eta_max,
                               dtype=torch.float64, device=dev),
                    math.log(len(m.channels))
                    / shmg.clamp(min=1e-9))
            else:
                eta = torch.full((), c.hedge_lr,
                                 dtype=torch.float64,
                                 device=dev)
            factors = torch.exp(
                eta[:, None] * (rewards - pmix[:, None]))
            sw = torch.where(mask[:, None], sw * factors, sw)
            sw = sw.clamp(c.hedge_min, c.hedge_max)
            sw = torch.where(
                mask[:, None],
                sw / sw.sum(dim=1, keepdim=True), sw)
            if c.cond_hedge:
                self.swR[self._ar, self.sreg] = sw
                self.shmgR[self._ar, self.sreg] = shmg
            self.sw = sw
            self.shmg = shmg

        # context push into the per-stream tail ring — the tail is
        # indexed by stream-relative position ((pos - C) % cap_s), the
        # same mapping _bidx uses on reads; absolute pos%cap_s would
        # alias differently whenever C % cap_s != 0.
        idx = self._bidx(self.spos)                             # [N]
        h_prev = self._h_at(end)
        h_new = torch.where(self.spos > 0, h_prev * _P + x + 1,
                            x + 1)
        flat_r, flat_h = self.sring.view(-1), self.shash.view(-1)
        flat_r.index_copy_(
            0, idx,
            torch.where(mask, x.to(torch.int32),
                        flat_r.index_select(0, idx)))
        flat_h.index_copy_(
            0, idx,
            torch.where(mask, h_new,
                        flat_h.index_select(0, idx)))
        self.sseen[self._ar, x] = torch.where(
            mask, end, self.sseen[self._ar, x])
        self.sunig[self._ar, x] += mask.float()
        self.sloguni[self._ar, x] = torch.log(
            self.sunig[self._ar, x] + c.uni_alpha)
        self.stot += mask.float()
        self.sc = torch.where(
            mask[:, None],
            self.sc * c.topic_decay
            + m._R[x].float() * (1.0 - c.topic_decay),
            self.sc)
        self.sproto = torch.where(
            mask[:, None],
            self.sproto * c.proto_decay
            + m._A[x] * (1.0 - c.proto_decay),
            self.sproto)
        self.spos += mask.long()

    def _pass(self, end: torch.Tensor):
        """One batched probe -> (logits, reward-context).  The reward
        for whatever token is sampled is gathered from the same probe —
        predict-only streams never need the live step's second probe."""
        slot, found, live_k, cok, dok, dk = self._probe_b(end)
        floor = torch.clamp(end + 1
                            - self.model.config.ring_capacity,
                            min=0)
        votes = self._deep_votes_b(
            end, floor, slot[:, self.model._gt.row_epi],
            found[:, self.model._gt.row_epi], dk)
        kv, kvt = self._knn_votes_b()
        logits = self._logits_b(end, slot, found, live_k, cok,
                                dok, dk, votes, kv, kvt)
        return logits, (slot, found, live_k, cok, votes, kv,
                        kvt, end)

    # ── generate ─────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, prompt_ids: list[list[int]],
                 max_new_tokens: int,
                 temperatures: list[float] | None = None,
                 top_ps: list[float] | None = None,
                 top_ks: list[int] | None = None,
                 repetition_penalties: list[float] | None = None,
                 seeds: list[int | None] | None = None,
                 stops: list[list[str] | None] | None = None,
                 logits_processors: list | None = None,
                 tokenizer=None,
                 eos_ids: set[int] | None = None,
                 tag_fn=None,
                 prompt_rewards: bool = True,
                 cf_topk: int = 0,
                 ) -> tuple[list[list[int]], list[FluxPacket]]:
        """Batched predict-only generation -> (gen_ids, packets).

        The model is never mutated; each stream emits a FluxPacket for
        the caller to apply or discard via ``model.apply_packets``.
        ``prompt_rewards=False`` skips hedge updates during prompt feed
        (cheaper prefill; small mixture-weight divergence vs clones).
        """
        n = len(prompt_ids)
        if n == 0:
            return [], []
        self.reset(n)
        temperatures = temperatures or [0.0] * n
        top_ps = top_ps or [1.0] * n
        top_ks = top_ks or [0] * n
        repetition_penalties = repetition_penalties or [1.0] * n
        seeds = seeds or [None] * n
        stops = stops or [None] * n
        logits_processors = logits_processors or [None] * n
        eos_ids = eos_ids or set()

        gens = []
        for s in seeds:
            if s is None:
                gens.append(None)
            else:
                g = torch.Generator(device=self.dev)
                g.manual_seed(int(s))
                gens.append(g)

        if not self._gpu:
            outs, alts_out = self._generate_cpu(
                prompt_ids, max_new_tokens, temperatures, top_ps,
                top_ks, repetition_penalties, gens, stops,
                logits_processors, tokenizer, eos_ids, cf_topk)
        else:
            outs, alts_out = self._generate_gpu(
                prompt_ids, max_new_tokens, temperatures, top_ps,
                top_ks, repetition_penalties, gens, stops,
                logits_processors, tokenizer, eos_ids,
                prompt_rewards, cf_topk)

        packets = [FluxPacket(
            tag=(tag_fn(i) if tag_fn else "stream"),
            prompt_ids=list(prompt_ids[i]), gen_ids=outs[i],
            meta={"stream": i},
            alts=alts_out[i] if alts_out else None)
            for i in range(n)]
        return outs, packets

    def _generate_gpu(self, prompt_ids, max_new, temps, top_ps,
                      top_ks, rep_pens, gens, stops, processors,
                      tokenizer, eos_ids, prompt_rewards,
                      cf_topk: int = 0):
        n = len(prompt_ids)
        dev = self.dev
        V = self.model.vocab_size
        outs: list[list[int]] = [[] for _ in range(n)]
        alts_out: list[list] = [[] for _ in range(n)]
        active = torch.ones(n, dtype=torch.bool, device=dev)
        plens = torch.tensor([len(p) for p in prompt_ids],
                             dtype=torch.int64, device=dev)
        maxp = int(plens.max())
        pad = torch.full((n, maxp), 0, dtype=torch.int64,
                         device=dev)
        for i, p in enumerate(prompt_ids):
            if p:
                pad[i, :len(p)] = torch.tensor(
                    p, dtype=torch.int64, device=dev)

        # prompt feed — observe-only steps (rewards keep hedge parity)
        for t in range(maxp):
            fed = plens > t
            if not bool(fed.any()):
                break
            xt = pad[:, t]
            rw = None
            if prompt_rewards:
                end = self.spos - 1
                slot, found, live_k, cok, dok, dk = \
                    self._probe_b(end)
                floor = torch.clamp(
                    end + 1 - self.model.config.ring_capacity,
                    min=0)
                votes = self._deep_votes_b(
                    end, floor, slot[:, self.model._gt.row_epi],
                    found[:, self.model._gt.row_epi], dk)
                kv, kvt = self._knn_votes_b()
                rw, awake = self._rewards_b(xt, slot, found, live_k,
                                            cok, votes, kv, kvt, end)
            else:
                awake = None
            self.observe_gpu(xt, fed, rw, awake)

        # generation loop — one batched pass per step for all streams
        pen_mask = torch.zeros(n, V, dtype=torch.bool, device=dev)
        for i, p in enumerate(prompt_ids):
            if p:
                pen_mask[i, torch.tensor(
                    p, dtype=torch.int64, device=dev)] = True
        for _step in range(max_new):
            if not bool(active.any()):
                break
            logits, (slot, found, live_k, cok, votes, kv, kvt,
                     end) = self._pass(self.spos - 1)

            lg = logits.clone()
            tt = torch.tensor(temps, dtype=lg.dtype, device=dev)
            lg = lg / tt.clamp(min=1e-5)[:, None]
            rp = torch.tensor(rep_pens, dtype=lg.dtype,
                              device=dev)
            has_pen = rp != 1.0
            if bool(has_pen.any()):
                penv = torch.where(
                    pen_mask & has_pen[:, None],
                    rp[:, None].expand_as(lg),
                    torch.ones((), dtype=lg.dtype, device=dev))
                lg = lg / penv
            for i in range(n):
                if not bool(active[i]):
                    continue
                if processors[i] is not None:
                    lg[i] = processors[i](
                        lg[i].unsqueeze(0), outs[i]).squeeze(0)
                if top_ks[i] > 0:
                    k = min(top_ks[i], lg.shape[-1])
                    thr = lg[i].topk(k).values[-1]
                    lg[i] = torch.where(
                        lg[i] < thr,
                        torch.full_like(lg[i], float("-inf")),
                        lg[i])
                if top_ps[i] < 1.0:
                    srt, si = lg[i].sort(descending=True)
                    cum = torch.softmax(srt, -1).cumsum(-1)
                    cut = cum > top_ps[i]
                    cut[0] = False
                    srt[cut] = float("-inf")
                    lg[i] = torch.empty_like(lg[i]).scatter_(
                        0, si, srt)
                # support-restricted sampling: keep only tokens some
                # channel voted for (p_mix > thresh * p_uni) - stops
                # temperature sampling falling onto the unigram
                # escape floor in thin contexts
                st = self.model.config.support_thresh
                if st > 0 and temps[i] > 0:
                    ua = self.model.config.uni_alpha
                    denom = self.stot[i] + ua * V
                    pu = (self.sunig[i] + ua) / denom
                    supp = torch.softmax(logits[i].float(), -1) \
                        > st * pu
                    if bool(supp.any()):
                        lg[i] = torch.where(
                            supp, lg[i],
                            torch.full_like(lg[i], float("-inf")))
                lg = torch.where(active[:, None], lg,
                             torch.full_like(lg, float("-inf")))
            # counterfactual propensity capture: top-k of each active
            # stream's final sampling distribution (post temp/masks)
            if cf_topk > 0:
                for i in range(n):
                    if not bool(active[i]):
                        continue
                    tv, ti = torch.softmax(lg[i].float(), -1).topk(
                        cf_topk)
                    alts_out[i].append(list(zip(
                        ti.tolist(), tv.tolist())))

            if all(t == 0 for t in temps):
                nxt = lg.argmax(-1)
            else:
                probs = torch.softmax(lg, -1).clamp(min=1e-10)
                probs = torch.nan_to_num(
                    probs, nan=1.0 / probs.shape[-1])
                nxt = torch.zeros(n, dtype=torch.int64,
                                  device=dev)
                for i in range(n):
                    if not bool(active[i]):
                        continue
                    nxt[i] = (lg[i].argmax() if temps[i] == 0
                              else torch.multinomial(
                                  probs[i], 1,
                                  generator=gens[i])[0])

            nxt_cpu = nxt.tolist()              # one sync per step
            neweos = torch.zeros(n, dtype=torch.bool, device=dev)
            for i in range(n):
                if not bool(active[i]):
                    continue
                outs[i].append(nxt_cpu[i])
                if nxt_cpu[i] in eos_ids:
                    neweos[i] = True
                elif stops[i] and tokenizer is not None:
                    tail = tokenizer.decode(
                        outs[i][-24:], skip_special_tokens=False)
                    if any(s in tail for s in stops[i]):
                        neweos[i] = True
            pen_mask[self._ar[active], nxt[active]] = True
            obs_mask = active.clone()
            active = active & ~neweos
            rw, awake = self._rewards_b(nxt, slot, found, live_k, cok,
                                        votes, kv, kvt, end)
            self.observe_gpu(nxt, obs_mask, rw, awake)
        return outs, alts_out

    # ── CPU facade path ──────────────────────────────────────────────

    def _generate_cpu(self, prompt_ids, max_new, temps, top_ps,
                      top_ks, rep_pens, gens, stops, processors,
                      tokenizer, eos_ids, cf_topk: int = 0):
        from forge.model.flux import FluxLM
        m = self.model
        n = len(prompt_ids)
        outs: list[list[int]] = [[] for _ in range(n)]
        alts_out: list[list] = [[] for _ in range(n)]
        for i, ids in enumerate(prompt_ids):
            ctx = self._ctxs[i]
            for x in ids:
                FluxLM._learn_token(ctx, int(x))
        active = [True] * n
        for _step in range(max_new):
            if not any(active):
                break
            for i in range(n):
                if not active[i]:
                    continue
                ctx = self._ctxs[i]
                logits = FluxLM._predict(ctx).float().clone()
                if rep_pens[i] != 1.0:
                    for t in set(prompt_ids[i]) | set(outs[i]):
                        logits[t] /= rep_pens[i]
                if processors[i] is not None:
                    logits = processors[i](
                        logits.unsqueeze(0), outs[i]).squeeze(0)
                if temps[i] == 0:
                    x = int(logits.argmax())
                else:
                    lg = logits / max(temps[i], 1e-5)
                    if top_ks[i] > 0:
                        k = min(top_ks[i], lg.numel())
                        thr = lg.topk(k).values[-1]
                        lg = torch.where(
                            lg < thr,
                            torch.full_like(lg, float("-inf")),
                            lg)
                    if top_ps[i] < 1.0:
                        srt, si = lg.sort(descending=True)
                        cum = torch.softmax(srt, -1).cumsum(-1)
                        cut = cum > top_ps[i]
                        cut[0] = False
                        srt[cut] = float("-inf")
                        lg = torch.empty_like(lg).scatter_(
                            0, si, srt)
                    probs = torch.softmax(lg, -1).clamp(min=1e-10)
                    probs = torch.nan_to_num(
                        probs, nan=1.0 / probs.numel())
                    x = int(torch.multinomial(
                        probs, 1, generator=gens[i])[0])
                if cf_topk > 0:
                    # greedy streams never defined lg - record raw
                    # softmax (degenerate pi_s ~ 1; skipped at credit)
                    src = logits if temps[i] == 0 else lg
                    pr = torch.softmax(src.float(), -1)
                    tv, ti = pr.topk(cf_topk)
                    alts_out[i].append(list(zip(
                        ti.tolist(), tv.tolist())))
                outs[i].append(x)
                if x in eos_ids:
                    active[i] = False
                elif stops[i] and tokenizer is not None:
                    tail = tokenizer.decode(
                        outs[i][-24:], skip_special_tokens=False)
                    if any(s in tail for s in stops[i]):
                        active[i] = False
                FluxLM._learn_token(ctx, x)
        return outs, alts_out
