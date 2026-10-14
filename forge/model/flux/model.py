"""FluxLM - sparse online associative-memory LM."""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .helpers import (_HASH_P, _MASK64, _MIX1, _MIX2,
                      _byte_class, _load_teacher_embedding, _mix64,
                      _s64, _scatter_sum)
from .config import FluxConfig
from .journal import FluxJournal
from .tables import _GPUCellTables
from .kernels import HAS_TRITON

if HAS_TRITON:
    from .kernels import (_flux_deep_votes_kernel,
                          _flux_logit_tail_kernel)


from .statemixin import _StateMixin
from .predictmixin import _PredictMixin
from .learnmixin import _LearnMixin
from .revertmixin import _RevertMixin
from .iomixin import _IOMixin


class FluxLM(_StateMixin, _PredictMixin, _LearnMixin, _RevertMixin,
             _IOMixin, nn.Module):
    """Sparse online associative-memory LM.  See module docstring."""

    # Engine capability flags (explicit-compat contract).
    unbounded_context = True     # engine must not truncate the prompt
    supports_kv_cache = False    # state is internal, no past_key_values


    def __init__(self, config: FluxConfig | None = None):
        super().__init__()
        self.config = config or FluxConfig()
        c = self.config
        self.vocab_size = c.vocab_size
        self.eos_token_id = c.eos_token_id

        V = c.vocab_size
        # -- channel weights (Hedge-adapted) ----------------------------
        self.channels = [f"o{k}" for k in c.orders] + ["deep", "topic",
                                                       "sem", "csem",
                                                       "knn",
                                                       "recency",
                                                       "fatigue", "uni",
                                                       "backoff"] \
            + [f"p{j}" for j in range(len(c.sparse_taps))] \
            + [f"g{k}" for k in c.class_orders] \
            + [f"w{j}" for j in range(len(c.word_orders))] \
            + [f"h{j}" for j in range(len(c.hybrid_taps))] \
            + [ch for ch, on in (("gcell", c.gcell), ("gsyn", c.gsyn),
                                 ("gsd", c.gsyn_deep),
                                 ("dnet", c.dnet)) if on]
        self._chix = {ch: i for i, ch in enumerate(self.channels)}
        # fuzzy lift-channel indices: hedged like every channel now
        # (previously fixed-weight bypass - the measured cause of the
        # ablation regression).  Real rewards in the eager and bulk
        # paths; abstain (neutral uni fallback) on the fused device step.
        self._fset = {"gcell", "gsyn", "gsd", "dnet"}
        self._fidx = [self._chix[ch] for ch in
                      ("gcell", "gsyn", "gsd", "dnet")
                      if ch in self._chix]
        # host-table abstain mask: p<j>/g<k>/w<j>/h<j> channels have no
        # device reward on GPU - they take the mixture prob as a
        # neutral reward (weight-preserving) instead of the unigram
        # fallback, which would decay them to hedge_min over training.
        # Fuzzy channels are excluded: their bulk rewards arrive via
        # real_r (overwriting them here would kill that signal).
        self._pmask = np.array([ch.startswith(("p", "g", "w", "h"))
                                and ch not in self._fset
                                for ch in self.channels], dtype=bool)
        self._pidx = np.nonzero(self._pmask)[0].tolist()
        # thin_boost mask: retrieval/semantic channel indices that get
        # the fixed thin-context weight bonus (device tensor built on
        # the GPU init path)
        self._tb_idx = [self._chix[ch] for ch in
                        ("deep", "topic", "sem", "csem", "knn")
                        if ch in self._chix]
        self._w = np.ones(len(self.channels), dtype=np.float64)
        self._w /= self._w.sum()
        self._hedge_mix = 0.0        # AdaHedge cumulative mix gap
        # conditioned hedge: per-regime weight sets + mix-gaps
        # (regime 0 = thin context, 1 = order-evidence warm context);
        # _w stays the "most recently used row" mirror for readers
        # (snapshot/channel_weights show active-regime weights).
        self._wR = np.tile(self._w, (2, 1))
        self._hmxR = np.zeros(2, dtype=np.float64)
        self._regime = 0

        # -- sparse memory cells ----------------------------------------
        self._tables: dict[int, dict[int, dict[int, float]]] = {
            k: {} for k in c.orders}
        # per-cell follower totals (cache - avoids O(followers) sums)
        self._totals: dict[int, dict[int, float]] = {
            k: {} for k in c.orders}
        # sparse-tap tables: same cell semantics as _tables but keyed
        # by the taps tuple (noncontiguous suffix contexts)
        self._ptabs: dict[tuple, dict[int, dict[int, float]]] = {
            taps: {} for taps in c.sparse_taps}
        self._ptots: dict[tuple, dict[int, float]] = {
            taps: {} for taps in c.sparse_taps}
        # character-class tables: contiguous suffix keys over the byte
        # class ring (PAQ word-model analog - coarse keys generalize
        # on novel text); "g<k>" channels, "K" journal entries
        self._gtabs: dict[int, dict[int, dict[int, float]]] = {
            k: {} for k in c.class_orders}
        self._gtots: dict[int, dict[int, float]] = {
            k: {} for k in c.class_orders}
        # byte->class LUT [V]: 0 digit, 1 upper, 2 lower, 3 space/tab,
        # 4 newline, 5 punct, 6 other ascii, 7 high/bpe
        self._cls = np.array([_byte_class(t) for t in range(V)],
                             dtype=np.int32)
        # word-order tables: keys fold the last-w word hashes +
        # current-word length bucket (derived via ring scan - no extra
        # state); "w<j>" channels, "W" journal entries
        self._wtabs: dict[int, dict[int, dict[int, float]]] = {
            j: {} for j in range(len(c.word_orders))}
        self._wtots: dict[int, dict[int, float]] = {
            j: {} for j in range(len(c.word_orders))}
        # hybrid class+raw tap tables: keys fold (offset, is_raw)
        # patterns - "h<j>" channels, "H" journal entries
        self._htabs: dict[int, dict[int, dict[int, float]]] = {
            j: {} for j in range(len(c.hybrid_taps))}
        self._htots: dict[int, dict[int, float]] = {
            j: {} for j in range(len(c.hybrid_taps))}
        self._uni = np.zeros(V, dtype=np.float64)
        self._total = 0
        # bloom doorkeeper (R63): counting-Bloom admission in front of
        # cell creation - a key's first sighting sets bits, the row
        # materializes on its bloom_admit-th sighting.  _bloom_h serves
        # the dict tables (eager + bulk host-table writes); _bloom_g
        # (init below) serves the GPU order-row write gate.  Admission
        # metadata - not journaled; revert restores cells exactly and
        # stale bits only over-promote afterward.
        self._bloom_h = (np.zeros(c.bloom_bits, dtype=np.int16)
                         if c.bloom_admit >= 2 else None)
        self._bloom_g = None

        # device: dense readout tensors live here; sparse memory stays host.
        self._dev = torch.device(
            c.device if (c.device == "cpu" or torch.cuda.is_available())
            else "cpu")
        self._log_uni_num = torch.full(
            (V,), math.log(c.uni_alpha), dtype=torch.float32,
            device=self._dev)

        # -- episodic ring + prefix-hash ring ---------------------------
        self._ring = np.zeros(c.ring_capacity, dtype=np.int32)
        self._H = np.zeros(c.ring_capacity, dtype=np.uint64)
        self._count = 0                     # absolute append position
        # key -> deque of (ctx_end, succ, certs) - succ stored at insert
        # time (deferred by one token) so the entry survives ring wrap;
        # certs = tuple of suffix hashes at config.cert_orders (0 = n/a)
        self._epi: dict[int, deque] = {}
        self._epi_fifo: deque = deque()     # (key, ctx_end) insert order
        self._epi_n = 0                     # live entries (global cap)
        self._epi_hits: dict[int, int] = {}  # key -> cells voted (utility)
        self._epi_dirty: set[int] = set()   # rows needing GPU resync
                                            # (drops batch the probes)
        self._ep_start = 0                  # current episode start (abs pos)
        self._pows = np.empty(
            max(max(c.orders), c.deep_order, c.csem_order,
                *(c.cert_orders or (0,))) + 1,
            dtype=np.uint64)
        self._pows[0] = 1
        for i in range(1, len(self._pows)):
            self._pows[i] = (int(self._pows[i - 1]) * 1099511628211) \
                % (1 << 64)

        # -- topic channel (dense tensors on self._dev) -----------------
        g = torch.Generator().manual_seed(c.seed)
        self._R = ((torch.randint(0, 2, (V, c.topic_dim),
                                  generator=g, dtype=torch.int8) * 2 - 1)
                   .to(self._dev))
        # live features (r_live): additive co-occurrence embedding -
        # _Rl[x] EMAs toward the pre-x context sketch, so tokens seen
        # in similar contexts converge.  KEPT SEPARATE from _R: the
        # context-side sketch/bigram EMAs invert exactly under revert
        # via frozen _R rows; the vocab-facing gen readouts use the
        # blended _Re() = _R + _Rl.  This is the no-SFT substitute for
        # graft_teacher/distill_gen - seeded params + knobs only.
        self._Rl = (torch.zeros(V, c.topic_dim, dtype=torch.float32,
                               device=self._dev) if c.r_live else None)
        # materialized _R + _Rl for vocab-facing reads (gsyn readout,
        # delta-rule targets) - kept current by _rl_write/_rl_bulk and
        # the rL/rB undo paths; None when r_live is off
        self._Reff = (self._R.float()
                      if c.r_live else None)
        self._A = torch.zeros((V, c.topic_dim), dtype=torch.float32,
                              device=self._dev)
        self._c = torch.zeros(c.topic_dim, dtype=torch.float32,
                              device=self._dev)
        # bigram-bound sketch (_c2): EMA over order-sensitive pairs
        # bind(R[prev], R[cur]) = R[prev][bp1] * R[cur][bp2] - gives the
        # distill head / learned-key channels "a follows b" signal the
        # bag-of-features _c cannot express.  Maintained only when
        # c.bigram_sketch; zero vector otherwise.
        self._c2 = torch.zeros(c.topic_dim, dtype=torch.float32,
                               device=self._dev)
        gb = torch.Generator().manual_seed(c.seed ^ 0xB16A)
        self._bp1 = torch.randperm(c.topic_dim, generator=gb) \
            .to(self._dev)
        self._bp2 = torch.randperm(c.topic_dim, generator=gb) \
            .to(self._dev)
        # learned-key cell channel (gcell): SimHash codes of the
        # context fingerprint -> host dict-of-dict cells with delta-
        # rule successor scores.  Host-resident on both backends (the
        # _ptabs/_gtabs precedent - zero VRAM, inspectable, journaled).
        self._gc: dict[int, dict[int, float]] = {}
        self._gct: dict[int, float] = {}
        self._gc_fifo: deque = deque()
        # coarse bucket index for the fuzzy read: code>>shift -> member
        # codes.  Rebuilt from _gc on snapshot load - derived state,
        # never serialized; bucket ops mirror 'G'/'gE' undo inline.
        self._gc_buckets: dict[int, set[int]] = {}
        gk = torch.Generator().manual_seed(c.seed ^ 0x6CE11)
        self._Wk = (torch.randint(0, 2, (self._head_in_dim(),
                                        c.gcell_bits),
                                 generator=gk, dtype=torch.int8)
                    * 2 - 1).to(self._dev)
        # dense delta channel (gsyn): k = normalize(Wq^T feats) keys a
        # delta-rule matrix S mapping context -> expected next-token
        # fingerprint.  Eager host-side update (outside CUDA graphs).
        gq = torch.Generator().manual_seed(c.seed ^ 0x6CE12)
        self._Wq = (torch.randint(0, 2, (self._head_in_dim(),
                                        c.gsyn_dim),
                                 generator=gq, dtype=torch.int8)
                    * 2 - 1).to(self._dev)
        self._S = torch.zeros(c.topic_dim, c.gsyn_dim,
                              dtype=torch.float32, device=self._dev)
        # slow-timescale mirror (gsyn_slow_decay>0): same writes, own
        # retention/write rate - HOPE-style continuum memory pair
        self._Ss = torch.zeros(c.topic_dim, c.gsyn_dim,
                               dtype=torch.float32, device=self._dev)
        # journal factor ring for 'GS' undo:
        # [coeff_fast | coeff_slow | v_err | k] fp16
        self._gsyn_ring = np.zeros(
            (c.vec_delta_cap, 2 + c.topic_dim + c.gsyn_dim),
            dtype=np.float16)
        self._gsyn_cur = 0
        # nonlinear memory channel (gsyn_deep): theta = 2-layer MLP
        # f(normalize(_Wq^T feats)) -> expected _Reff row - the
        # TTT-MLP/Titans-style test-time-trained complement to the
        # linear delta table _S.  Wd2 zero-init: the channel is silent
        # until the first gradient steps land.  Sm* = momentum buffers
        # (heavy-ball online optimizer state, journaled with theta so
        # revert is exact).  All None when disabled.
        if c.gsyn_deep:
            hd = int(c.gsyn_deep_hidden)
            gd = torch.Generator().manual_seed(c.seed ^ 0x6CE13)
            self._Wd1 = (torch.randn(hd, c.gsyn_dim, generator=gd)
                         / math.sqrt(c.gsyn_dim)).float().to(self._dev)
            self._Sb1 = torch.zeros(hd, device=self._dev)
            self._Wd2 = torch.zeros(c.topic_dim, hd,
                                    device=self._dev)
            self._Sb2 = torch.zeros(c.topic_dim, device=self._dev)
            self._Sm1 = torch.zeros(hd, c.gsyn_dim,
                                    device=self._dev)
            self._Smb1 = torch.zeros(hd, device=self._dev)
            self._Sm2 = torch.zeros(c.topic_dim, hd,
                                    device=self._dev)
            self._Smb2 = torch.zeros(c.topic_dim, device=self._dev)
        else:
            (self._Wd1, self._Sb1, self._Wd2, self._Sb2,
             self._Sm1, self._Smb1, self._Sm2, self._Smb2) = (
                None,) * 8
        # iterated memory channel (dnet): a stack of gated delta-rule
        # layers - the linear-attention/delta-net "transformer that
        # learns online".  _DS[l] = recurrent state [topic_dim, dnet_dk]
        # (zero-init, silent at birth); _Dk/_Dv/_Db = seeded bipolar
        # int8 projections (knobs, never trained).  Layer-0 input is
        # the shared gsyn key (gsyn_dim); deeper inputs are relu'd
        # readouts (topic_dim).  No optimizer state, no KV cache.
        if c.dnet:
            nl, dk = int(c.dnet_layers), int(c.dnet_dk)
            self._DS = [torch.zeros(c.topic_dim, dk,
                                    device=self._dev)
                        for _ in range(nl)]
            self._Dk, self._Dv, self._Db = [], [], []
            for l in range(nl):
                din = c.gsyn_dim if l == 0 else c.topic_dim
                gl = torch.Generator().manual_seed(
                    c.seed ^ (0x6CE20 + l))
                self._Dk.append((torch.randint(
                    0, 2, (din, dk), generator=gl,
                    dtype=torch.int8) * 2 - 1).to(self._dev))
                self._Dv.append((torch.randint(
                    0, 2, (din, c.topic_dim), generator=gl,
                    dtype=torch.int8) * 2 - 1).to(self._dev))
                self._Db.append(torch.randn(
                    din, generator=gl).float().to(self._dev))
        else:
            self._DS = self._Dk = self._Dv = self._Db = []
        # open dnet undo session (same convention as _gsd_session)
        self._dnet_session: dict | None = None
        # open deep-memory undo session: first-write snapshot dict
        # {"snap": theta+momentum tuple} flushed as one ("gd", snap)
        # entry at the end of ingest/consolidate - mirrors _rl_session
        self._gsd_session: dict | None = None
        # open live-feature write session (r_live): {tok: pre-span _R
        # row} populated by _rl_write, flushed as one ("rB", rows)
        # journal entry at the end of ingest/teach
        self._rl_session: dict | None = None
        # recall<->generation runtime dial (engine mem_bias): scales
        # gcell/gsyn lift weights; 1.0 = neutral
        self._gen_bias = 1.0
        # block-local journal sink: while _ingest_bulk runs, per-write
        # journal entries collect on the "bc" block ("sub" key) instead
        # of journal.entries - keeps bulk ingests below journal_cap so
        # the "bc" header can't be evicted before it can undo
        self._blk_journal: list | None = None
        # the open "bc" block dict itself - needed so the finally-path
        # in ingest() can seal its payload accounting too
        self._blk_active: dict | None = None
        # expected-next-token fingerprint (semantic-spread channel)
        self._proto = torch.zeros(c.topic_dim, dtype=torch.float32,
                                  device=self._dev)
        # distilled head (distill_head): linear _Wh [V, in] or MLP
        # (_Wh1 [H, in] + _Wh [V, H]) readout from the order-sensitive
        # context fingerprint -> teacher-aligned logits; None until the
        # first distill.  Additive channel, weight = c.head_w, gated
        # to thin contexts by c.head_thin.
        self._Wh: torch.Tensor | None = None
        self._Wh1: torch.Tensor | None = None
        # zero-gradient controller head (KnobHead): when attached, every
        # _predict routes through head.remix(lg, cap) — the head holds
        # counting tables only (patterns/personality layer), all token
        # evidence comes from the weight-derived channel lifts in cap
        self._head = None

        # -- recency channel --------------------------------------------
        self._last_seen: dict[int, int] = {}
        self._recent: deque = deque(        # (abs_pos, tok) for iteration
            maxlen=max(c.recency_span, c.fatigue_span))
        # set by predict-only stream facades (forge/model/flux_stream):
        # gates the shared-memory writes in _learn_token while context
        # state (ring/hash/seen/hedge/uni/proto EMAs) still advances
        self._predict_only = False

        # -- csem: per-context semantic fingerprints --------------------
        # fixed projection A -> csem_dim for cheap per-context cosine
        g2 = torch.Generator().manual_seed(c.seed ^ 0xC5E4)
        self._Pcsem = (torch.randn(c.topic_dim, c.csem_dim,
                                   generator=g2)
                       / math.sqrt(c.topic_dim)).float().to(self._dev)
        self._csem_cpu: dict[int, np.ndarray] = {}   # cpu path rows fp32
        # GPU path: fp16 rows indexed by slot in the shared key matrix
        # (gt.tkeys[gt.row_csem] holds the probe keys)
        self._csem_rows: torch.Tensor | None = None

        # -- knn fingerprint bank (approximate-match episodic memory) --
        # row i = unit-normalized fingerprint of one past context,
        # plus the token that followed it and the context-end position.
        # Rows are unit-normalized at write so a single matvec gives
        # cosines; _knn_pos < 0 marks an empty row.
        P = c.knn_capacity
        # +1 scratch row at index P: the GPU write gate steers
        # suppressed knn writes there - branchless inside the CUDA
        # graph, never scanned (live prefix is [:P]), pos stays -1.
        self._knn_fp = torch.zeros(
            (P + 1, c.csem_dim),
            dtype=torch.float16 if self._dev.type == "cuda"
            else torch.float32, device=self._dev)
        self._knn_succ = torch.full((P + 1,), -1, dtype=torch.int32,
                                    device=self._dev)
        self._knn_pos = torch.full((P + 1,), -1, dtype=torch.int64,
                                   device=self._dev)
        self._knn_cur = 0                   # ring cursor (host mirror)
        # retrieval-utility counter per row (topk membership) - feeds
        # knn_utility_evict; derived stat, not journaled/snapshotted
        self._knn_hits = torch.zeros(P + 1, dtype=torch.int32,
                                     device=self._dev)

        # -- CUDA-primary backend ---------------------------------------
        self._gpu = bool(c.cuda_primary
                         and self._dev.type == "cuda")
        if self._gpu and getattr(c, "compile_dev", False):
            # inductor's generated kernel names run ~200 chars - the
            # cache path overflows Windows MAX_PATH (260) on any base
            # dir longer than ~24 chars (triton adds ~85 chars of hash
            # dirs + tmp + suffix).  Honor an existing env value only
            # if it is short enough to actually work; otherwise pin
            # drive-root dirs.  A long inherited value (e.g. the
            # repo-local .devin/torch_cache) can only serve ~120-char
            # names and silently breaks compile on longer ones.
            _root = os.path.splitdrive(
                tempfile.gettempdir())[0] + os.sep
            for _var, _short in (("TORCHINDUCTOR_CACHE_DIR",
                                  _root + "ti"),
                                 ("TRITON_CACHE_DIR", _root + "tt")):
                _cur = os.environ.get(_var)
                if not _cur or len(_cur) > 24:
                    os.environ[_var] = _short
        # triton fusion gate - eager torch fallbacks cover missing
        # triton / non-CUDA devices (defined before _gpu so helpers can
        # read it unconditionally)
        self._use_triton = bool(
            self._gpu and c.use_triton and HAS_TRITON)
        self._graphs: dict = {}         # (lite, knn_bucket) -> CUDAGraph
        self._next_lite = False         # adaptive-read flag from last step
        self._gt: _GPUCellTables | None = None
        if self._gpu:
            self._gt = _GPUCellTables(c.orders, c.gpu_table_slots,
                                      self._dev, c.gpu_probe_max, V,
                                      M=c.gpu_row_width)
            # device mirrors of host-authoritative structures
            self._Hg = torch.zeros(c.ring_capacity, dtype=torch.int64,
                                   device=self._dev)
            self._ringg = torch.zeros(c.ring_capacity, dtype=torch.int32,
                                      device=self._dev)
            self._unig = torch.zeros(V, dtype=torch.float32,
                                     device=self._dev)
            self._seen_g = torch.full((V,), -10 ** 9,
                                      dtype=torch.int64,
                                      device=self._dev)
            self._pows_g = torch.tensor(
                [int(p) if int(p) < (1 << 63)
                 else int(p) - (1 << 64) for p in self._pows],
                dtype=torch.int64, device=self._dev)
            self._orders_g = torch.tensor(
                list(c.orders), dtype=torch.int64, device=self._dev)
            self._cert_g = torch.tensor(
                list(c.cert_orders) or [0], dtype=torch.int64,
                device=self._dev)
            # device scalars - graph-safe (no host reads inside a step)
            self._pos_g = torch.zeros((), dtype=torch.int64,
                                      device=self._dev)
            self._ep_g = torch.zeros((), dtype=torch.int64,
                                     device=self._dev)
            self._tot_g = torch.zeros((), dtype=torch.float64,
                                      device=self._dev)
            self._bx = torch.zeros((), dtype=torch.int64,
                                   device=self._dev)
            # GPU episodic index: seed key -> ring of context-end
            # positions.  Keys live in gt.tkeys[gt.row_epi] (shared
            # probe matrix); payloads here are indexed by that slot.
            # The host _epi dict stays authoritative for snapshots.
            S, ER = self._gt.S, c.epi_row
            self._epi_pos = torch.full((S, ER), -1,
                                       dtype=torch.int64,
                                       device=self._dev)
            self._epi_cnt = torch.zeros(S, dtype=torch.int64,
                                        device=self._dev)
            # deferred entries carry their successor + certificate
            # suffix hashes, so off-ring positions still vote
            self._epi_succ = torch.full((S, ER), -1,
                                        dtype=torch.int32,
                                        device=self._dev)
            self._epi_cert = torch.zeros((S, ER, len(c.cert_orders)),
                                         dtype=torch.int64,
                                         device=self._dev)
            self._csem_rows = torch.zeros((S, c.csem_dim),
                                          dtype=torch.float16,
                                          device=self._dev)
            if self._bloom_h is not None:
                # device twin of the host filter - gates the add_multi
                # write mask without a D2H round-trip per block
                self._bloom_g = torch.zeros(
                    c.bloom_bits, dtype=torch.int16, device=self._dev)
            # incremental projections/norms of A - updated by the same
            # rank-1 writes that touch A, so predict never rescans it
            self._Ac = torch.zeros((V, c.csem_dim), dtype=torch.float32,
                                   device=self._dev)
            self._Acn = torch.zeros(V, dtype=torch.float32,
                                    device=self._dev)
            self._An = torch.zeros(V, dtype=torch.float32,
                                   device=self._dev)
            # GPU journal rings - pre-update state for exact undo
            # without per-token D2H reads
            RC = c.vec_delta_cap
            # +1 column: AdaHedge mix-gap rides with w for exact undo
            self._w_ring_g = torch.zeros(
                (RC, len(self.channels) + 1), dtype=torch.float64,
                device=self._dev)
            self._avec_g = torch.zeros(
                (RC, c.topic_dim), dtype=torch.float32,
                device=self._dev)
            self._csem_prev = torch.zeros(
                (RC, c.csem_dim), dtype=torch.float16,
                device=self._dev)
            self._jcur = 0
            # device channel weights (authoritative when _gpu)
            self._wg = torch.from_numpy(self._w).to(self._dev)
            # AdaHedge cumulative mix-gap (device scalar, no host sync)
            self._hmg = torch.zeros((), dtype=torch.float64,
                                    device=self._dev)
            # step readback bundle (int64 bits for everything):
            # [O keys, epi key, epi slot, epi bits, csem slot, csem
            #  bits, order bits, gap, prevseen, certs(L), knn slot,
            #  knn prev pos, knn prev succ, w(C), surprise, mix,
            #  deep key, hit flag, next-step lite flag]
            L = len(c.cert_orders)
            self._boff_cert = self._gt.O + 8
            self._boff_knn = self._boff_cert + L
            self._boff_w = self._boff_knn + 3
            # +2 tail slots: predict-time deep key (raw u64, -1 invalid)
            # and the retrieval-hit flag - feeds _epi_hits on the host
            self._boff_hit = self._boff_w + len(self.channels) + 2
            # +1: adaptive-read flag - the step's phase-2 order probe
            # already has the cells, so next-step lite/full graph choice
            # rides this bit (zero extra syncs)
            self._boff_next = self._boff_hit + 2
            # +1: cond_hedge regime used by this step's reward update -
            # host _bookkeep_fused routes the post-update w row back to
            # the right _wR entry
            self._boff_regime = self._boff_next + 1
            # +1: warm/thin flag from the predict-time order probe -
            # host applies sparse-tap lift only on thin contexts
            self._boff_thin = self._boff_regime + 1
            self._bout = torch.zeros(
                (self._boff_thin + 1,),
                dtype=torch.int64, device=self._dev)
            # per-regime device weights + AdaHedge state (cond_hedge)
            self._wgR = torch.from_numpy(self._wR).to(self._dev)
            self._hmxRg = torch.zeros(2, dtype=torch.float64,
                                      device=self._dev)
            self._regime_g = torch.zeros((), dtype=torch.int64,
                                         device=self._dev)
            # regime actually used by this step's reward update -
            # _regime_g is refreshed by _logits_dev for the NEXT
            # context, so bout needs its own slot
            self._reg_used_g = torch.zeros((), dtype=torch.int64,
                                           device=self._dev)
            # knn ring: device cursor + prev-row ring for exact undo
            self._knncur_g = torch.zeros((), dtype=torch.int64,
                                         device=self._dev)
            self._kprev_g = torch.zeros(
                (c.vec_delta_cap, c.csem_dim), dtype=torch.float16,
                device=self._dev)
            self._blogits = torch.zeros(V, dtype=torch.float32,
                                        device=self._dev)
            self._jcur_t = torch.zeros((), dtype=torch.int64,
                                       device=self._dev)
            # precomputed device scalars for fused kernels / bit-pack
            self._pow_deep = self._pows_g[c.deep_order].clone()
            self._bitw = torch.pow(
                torch.full((self._gt.O,), 2, dtype=torch.int64,
                           device=self._dev),
                torch.arange(self._gt.O, dtype=torch.int64,
                             device=self._dev))
            self._warm: dict = {}            # (lite, bucket) -> count
            # clone workers run eager (_step_dev) - concurrent CUDA-graph
            # captures on one device invalidate each other
            self._use_cuda_graph = True
            # graphed standalone predict - the context end rides a
            # device input tensor; separate graph+pool from the step
            # graphs (interleaved replays can't share pool memory)
            self._bend = torch.zeros((), dtype=torch.int64,
                                   device=self._dev)
            self._plogits = torch.zeros(V, dtype=torch.float32,
                                        device=self._dev)
            self._pgraph = None
            self._pgraph_knn_b = -1
            self._pwarm = 0
            # adaptive-read flag: set inside _logits_dev each step -
            # order-cell share computed from the same gather the order
            # channel needs, packed into bout for next-step graph pick
            self._dconf_g = torch.zeros(1, dtype=torch.int64,
                                        device=self._dev)
            # fp64 scalar constants - pre-baked so the graphed step
            # never needs an H2D scalar copy (capture poison)
            self._const_1e3 = torch.full((), 1e-3,
                                         dtype=torch.float64,
                                         device=self._dev)
            self._const_1 = torch.full((), 1.0, dtype=torch.float64,
                                       device=self._dev)
            self._const_lnC = torch.full(
                (), math.log(len(self.channels)),
                dtype=torch.float64, device=self._dev)
            # torch.compile'd _logits_dev/_probes_at (lazy; False =
            # unavailable or failed - falls back to eager permanently)
            self._logits_c = None
            self._probes_c = None
            self._step_c = None
            self._const_eta_max = torch.full(
                (), c.hedge_eta_max, dtype=torch.float64,
                device=self._dev)
            self._const_hedge_lr = torch.full(
                (), c.hedge_lr, dtype=torch.float64,
                device=self._dev)
            self._boff_prevseen = torch.tensor(
                [self._gt.O + 7], dtype=torch.int64,
                device=self._dev)
            # deep-channel retrieval-hit flag: set by _logits_dev each
            # step, packed into bout for host-side utility accounting
            self._dhit_g = torch.zeros(1, dtype=torch.int64,
                                       device=self._dev)
            # predict-context warm flag for host sparse-tap lift (
            # 1 = an order cell had evidence -> sparse channels muted)
            self._dthin_g = torch.zeros(1, dtype=torch.int64,
                                        device=self._dev)
            self._pmask_g = torch.from_numpy(self._pmask).to(self._dev)
            # thin_boost mask over retrieval/semantic channel slots
            self._tb_mask = torch.zeros(
                len(self.channels), dtype=torch.float64,
                device=self._dev)
            if self._tb_idx:
                self._tb_mask[self._tb_idx] = 1.0
            # eviction snapshot slots (one per key row): written
            # in-graph, read by host only when evict bits are set -
            # epi rows carry pos+succ+cert payloads per cell
            EVI = 3 + max(self._gt.M,
                          c.epi_row * (2 + len(c.cert_orders)))
            EVF = 1 + max(self._gt.M, c.csem_dim)
            self._bev_i = torch.zeros(
                (self._gt.R, EVI), dtype=torch.int64,
                device=self._dev)
            self._bev_f = torch.zeros(
                (self._gt.R, EVF), dtype=torch.float32,
                device=self._dev)
            self._brow_epi = torch.tensor(
                [self._gt.row_epi], dtype=torch.int64,
                device=self._dev)
            self._brow_csem = torch.tensor(
                [self._gt.row_csem], dtype=torch.int64,
                device=self._dev)
            # write-gate device constants: discard-slot index (one
            # past the live bank) + a scalar-true for gate-off folds
            self._knn_disc = torch.tensor(
                [c.knn_capacity], dtype=torch.int64, device=self._dev)
            self._const_true = torch.ones(
                (), dtype=torch.bool, device=self._dev)

        # -- provenance -------------------------------------------------
        self.journal = FluxJournal(c.journal_cap, c.vec_delta_cap,
                                   c.topic_dim, len(self.channels),
                                   block_cap=c.journal_bytes)
        self.learning = True
        self.tag = "boot"
        self._graft: dict | None = None   # teacher-embedding graft info
        if c.teacher_embed:
            self.graft_teacher(c.teacher_embed)
        # positions where prediction was thin (for selective distill)
        self._gaps: deque = deque(maxlen=c.gap_cap)

    # -- context sketch --------------------------------------------------
