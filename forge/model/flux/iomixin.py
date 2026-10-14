"""Snapshot / load / device / diagnostics for FluxLM."""
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


class _IOMixin:

    def snapshot(self, path: str | Path,
                 keep_journal: bool = False) -> Path:
        """Full memory checkpoint (tables + rings + channel state).
        Writes the packed v2 layout: dict-of-dict cells become CSR
        arrays so a multi-million-cell model serializes as a few
        hundred MB of columns instead of GBs of pickled objects.
        ``keep_journal=True`` additionally stores the in-memory audit
        journal for post-load reverts; deployment snapshots should
        leave it off (journal blocks dominated v1 file size)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._snapshot_state_v2(keep_journal), path)
        return path

    # -- packed v2 helpers --------------------------------------------
    # v1 snapshots pickled {key -> {tok: cnt}} dicts verbatim; v2 packs
    # each table as CSR columns (keys_u64, rowptr, toks_i32, cnts_f32).

    @staticmethod
    def _pack_celltab(tab: dict) -> tuple:
        n = len(tab)
        keys = np.fromiter(map(int, tab.keys()), np.uint64, n)
        sizes = np.fromiter(map(len, tab.values()), np.int64, n)
        rowptr = np.zeros(n + 1, np.int64)
        np.cumsum(sizes, out=rowptr[1:])
        total = int(rowptr[-1])
        toks = np.empty(total, np.int32)
        cnts = np.empty(total, np.float64)   # f64 - host cells store
        off = 0                              # exact Python floats
        for i, cell in enumerate(tab.values()):
            m = int(sizes[i])
            if m:
                kv = np.fromiter(
                    (e for p in cell.items() for e in p),
                    np.float64, m * 2)
                toks[off:off + m] = kv[0::2].astype(np.int64)
                cnts[off:off + m] = kv[1::2]
                off += m
        return keys, rowptr, toks, cnts

    @staticmethod
    def _unpack_celltab(pack) -> dict:
        keys, rowptr, toks, cnts = pack
        tl, cl, rp = (toks.tolist(), cnts.tolist(),
                      rowptr.tolist())
        return {int(keys[i]): dict(
                    zip(tl[rp[i]:rp[i + 1]],
                        cl[rp[i]:rp[i + 1]]))
                for i in range(len(keys))}

    def _pack_epi(self) -> tuple:
        """_epi {key -> deque[(pos, succ, certs)]} -> columns."""
        epi = self._epi
        lc = len(self.config.cert_orders)
        n = len(epi)
        keys = np.fromiter(map(int, epi.keys()), np.uint64, n)
        sizes = np.fromiter(map(len, epi.values()), np.int64, n)
        rowptr = np.zeros(n + 1, np.int64)
        np.cumsum(sizes, out=rowptr[1:])
        total = int(rowptr[-1])
        pos = np.zeros(total, np.int64)
        succ = np.full(total, -1, np.int32)
        clen = np.zeros(total, np.int8)
        cert = (np.zeros((total, lc), np.uint64)
                if lc else np.zeros((total, 0), np.uint64))
        off = 0
        for dq in epi.values():
            for ent in dq:
                if isinstance(ent, (tuple, list)):
                    p, s_, cv = ent[0], ent[1], (
                        ent[2] if len(ent) > 2 else ())
                else:
                    p, s_, cv = ent, -1, ()
                pos[off] = int(p)
                succ[off] = int(s_)
                if lc and cv:
                    clen[off] = len(cv)
                    cert[off, :len(cv)] = cv
                off += 1
        return keys, rowptr, pos, succ, clen, cert

    @staticmethod
    def _unpack_epi(pack) -> dict:
        keys, rowptr, pos, succ, clen, cert = pack
        rp, cl = rowptr.tolist(), clen.tolist()
        out = {}
        for i, k in enumerate(keys):
            s, e = rp[i], rp[i + 1]
            out[int(k)] = [
                (int(pos[j]), int(succ[j]),
                 tuple(int(v) for v in cert[j, :cl[j]]))
                for j in range(s, e)]
        return out

    @staticmethod
    def _pack_pairs(pairs) -> tuple:
        """deque[(key, pos)] -> two columns."""
        n = len(pairs)
        a = np.empty(n, np.uint64)
        b = np.empty(n, np.int64)
        for i, (k, p) in enumerate(pairs):
            a[i] = int(k)
            b[i] = int(p)
        return a, b

    def _snapshot_state_v2(self, keep_journal: bool) -> dict:
        """Packed counterpart of _snapshot_state(); fmt=2."""
        st = self._snapshot_state()
        st["fmt"] = 2
        st["tables"] = {k: self._pack_celltab(t)
                        for k, t in st["tables"].items()}
        st["ptabs"] = [(list(taps), self._pack_celltab(t))
                       for taps, t in st["ptabs"].items()]
        st["gtabs"] = {k: self._pack_celltab(t)
                       for k, t in st["gtabs"].items()}
        st["wtabs"] = {k: self._pack_celltab(t)
                       for k, t in st["wtabs"].items()}
        st["htabs"] = {k: self._pack_celltab(t)
                       for k, t in st["htabs"].items()}
        st["epi"] = self._pack_epi()
        st["epi_fifo"] = self._pack_pairs(self._epi_fifo)
        if st.get("gc") is not None:
            gc, gct, fifo = st["gc"]
            st["gc"] = (
                self._pack_celltab(gc),
                (np.fromiter(map(int, gct.keys()), np.uint64,
                             len(gct)),
                 np.fromiter(map(float, gct.values()),
                             np.float64, len(gct))),
                np.fromiter(map(int, fifo), np.uint64, len(fifo)))
        st["last_seen"] = (
            np.fromiter(map(int, self._last_seen.keys()),
                        np.int64, len(self._last_seen)),
            np.fromiter(map(int, self._last_seen.values()),
                        np.int64, len(self._last_seen)))
        if not keep_journal:
            st["journal_entries"] = []
        return st

    def _v2_to_v1(self, st: dict) -> dict:
        """Unpack a fmt=2 snapshot dict into the v1 layout so
        ``_restore_state`` handles both formats on one path."""
        s = dict(st)
        s.pop("fmt", None)
        s["tables"] = {k: self._unpack_celltab(p)
                       for k, p in s["tables"].items()}
        s["ptabs"] = {tuple(taps): self._unpack_celltab(p)
                      for taps, p in s["ptabs"]}
        s["gtabs"] = {k: self._unpack_celltab(p)
                      for k, p in s["gtabs"].items()}
        s["wtabs"] = {k: self._unpack_celltab(p)
                      for k, p in s["wtabs"].items()}
        s["htabs"] = {k: self._unpack_celltab(p)
                      for k, p in s["htabs"].items()}
        s["epi"] = self._unpack_epi(s["epi"])
        fk, fp = s["epi_fifo"]
        s["epi_fifo"] = [(int(k), int(p))
                         for k, p in zip(fk.tolist(), fp.tolist())]
        if s.get("gc") is not None:
            gcp, (gk, gv), gf = s["gc"]
            s["gc"] = (self._unpack_celltab(gcp),
                       {int(k): float(v) for k, v in
                        zip(gk.tolist(), gv.tolist())},
                       gf.tolist())
        lt, lp = s["last_seen"]
        s["last_seen"] = {int(t): int(p) for t, p in
                          zip(lt.tolist(), lp.tolist())}
        return s

    def clone(self) -> "FluxLM":
        """In-memory deep copy of the full learned state - tables, rings,
        episodic index, knn bank, journal, hedge weights, graft.  The
        clone shares no state with the source: use it for parallel
        exploration (batch generation workers) while the canonical
        model's stream stays untouched."""
        import io
        buf = io.BytesIO()
        torch.save(self._snapshot_state(), buf)
        buf.seek(0)
        return type(self).load(buf)

    def _snapshot_state(self) -> dict:
        """The serializable state dict behind snapshot() and clone()."""
        state = {
            "config": asdict(self.config),
            "w": self._w,
            "hedge_mix": self._hedge_mix,
            "wR": self._wR, "hmxR": self._hmxR,
            "regime": self._regime,
            "channels": list(self.channels),
            "uni": self._uni,
            "log_uni_num": self._log_uni_num.cpu(),
            "total": self._total,
            "tables": self._tables if not self._gpu else {},
            "totals": self._totals if not self._gpu else {},
            "ptabs": self._ptabs, "ptots": self._ptots,
            "gtabs": self._gtabs, "gtots": self._gtots,
            "wtabs": self._wtabs, "wtots": self._wtots,
            "htabs": self._htabs, "htots": self._htots,
            # GPU backend: portable decoded rows {k: (keys, toks, cnts)}
            "table_triplets": (self._gt.triplets() if self._gpu
                               else None),
            # honest per-row sighting totals - triplets can't rebuild
            # them (truncated followers' mass isn't serialized)
            "table_rowtots": (self._gt.row_totals() if self._gpu
                              else None),
            "csem_cpu": self._csem_cpu if not self._gpu else None,
            "csem_rows_gpu": ((self._gt.tkeys[self._gt.row_csem].cpu(),
                               self._csem_rows.cpu())
                              if self._gpu else None),
            "epi": {k: list(v) for k, v in self._epi.items()},
            "epi_n": self._epi_n,
            "epi_fifo": list(self._epi_fifo),
            # knn associative bank: fp16 fingerprints + successors -
            # the approximate-match channel's entire state; slice to
            # the live prefix so a young model doesn't serialize a
            # mostly-empty capacity allocation
            "knn": (self._knn_fp[:self._knn_live()].cpu(),
                    self._knn_succ[:self._knn_live()].cpu(),
                    self._knn_pos[:self._knn_live()].cpu(),
                    self._knn_cur)
                    if self.config.use_knn else None,
            "ring": self._ring,
            "H": self._H,
            "count": self._count,
            "ep_start": self._ep_start,
            "A": self._A.cpu(),
            "c": self._c.cpu(),
            "c2": self._c2.cpu(),
            "proto": self._proto.cpu(),
            # learned-key cell channel state (gcell): code->succ-score
            # cells + fifo order; the projection is seed-derived
            "gc": (self._gc, self._gct, list(self._gc_fifo))
                  if self.config.gcell else None,
            # dense delta channel state (gsyn); _Wq is seed-derived.
            # tuple (S, Ss) - bare tensor = pre-two-timescale snapshot
            "gs": ((self._S.cpu(), self._Ss.cpu())
                   if self.config.gsyn else None),
            # nonlinear memory channel (gsyn_deep): theta + momentum,
            # 8-tuple in _gsd_params() order; seeded init otherwise
            "gsd": (self._gsd_snap()
                    if self.config.gsyn_deep else None),
            # iterated memory channel (dnet): per-layer recurrent
            # states; projections are seed-derived
            "dn": (tuple(s.cpu() for s in self._DS)
                   if self.config.dnet else None),
            # distilled linear head (distill_head); None until trained
            "Wh": (self._Wh.cpu() if self._Wh is not None else None),
            "Wh1": (self._Wh1.cpu() if self._Wh1 is not None
                    else None),
            # teacher graft: the projected embedding matrix itself -
            # `_R` is otherwise re-seeded from config, which would
            # silently revert the graft on load
            "R_graft": ((self._R.cpu(), dict(self._graft))
                        if self._graft else None),
            # live features (r_live): the drift table is learned
            # state; _Reff re-derives as _R + _Rl on load
            "Rl": (self._Rl.cpu() if self._Rl is not None
                   else None),
            # bloom doorkeeper admission state (counters are metadata,
            # not content - persisted so a resumed model keeps the same
            # admission behavior rather than re-sighting everything)
            "bloom": ((self._bloom_h,
                       self._bloom_g.cpu()
                       if self._bloom_g is not None else None)
                      if self._bloom_h is not None else None),
            "last_seen": self._last_seen,
            "recent": list(self._recent),
            "journal_entries": self.journal.entries,
            "journal_dropped": self.journal.dropped,
            "journal_blocks_dropped": self.journal.dropped_blocks,
            "gaps": list(self._gaps),
            # zero-gradient controller head (KnobHead): counting tables
            # are model state — persisted so a snapshot round-trips the
            # learned routing patterns exactly
            "head": (self._head.state_dict()
                     if self._head is not None else None),
        }
        return state

    def to_device(self, device: str) -> None:
        """Move the dense readout tensors (A/R/c/proto/log_uni_num) to a
        device.  Sparse memory stays host-resident either way."""
        dev = torch.device(
            device if (device == "cpu" or torch.cuda.is_available())
            else "cpu")
        self._dev = dev
        self.config.device = device
        # captured CUDA graphs bake device pointers - drop them so a
        # device move can't replay into freed memory
        self._graphs = {}
        self._pgraph = None
        self._warm = {}
        self._pwarm = 0
        # the compiled _logits_dev/_probes_at are device-specialized -
        # rebuild
        self._logits_c = None
        self._probes_c = None
        self._step_c = None
        for name in ("_R", "_A", "_c", "_c2", "_bp1", "_bp2",
                     "_proto", "_log_uni_num", "_Wk", "_Wq", "_S",
                     "_Ss"):
            setattr(self, name, getattr(self, name).to(dev))
        for name in ("_Rl", "_Reff", "_Wh", "_Wh1",
                     "_Wd1", "_Sb1", "_Wd2", "_Sb2",
                     "_Sm1", "_Smb1", "_Sm2", "_Smb2"):
            t = getattr(self, name)
            if t is not None:
                setattr(self, name, t.to(dev))
        for name in ("_DS", "_Dk", "_Dv", "_Db"):
            lst = getattr(self, name, None)
            if lst:
                setattr(self, name, [t.to(dev) for t in lst])
        if self._bloom_g is not None:
            self._bloom_g = self._bloom_g.to(dev)

    @classmethod
    def load(cls, path: str | Path, device: str | None = None,
             cuda_primary: bool | None = None) -> "FluxLM":
        # TRUST BOUNDARY: .flux snapshots are pickle payloads - load only
        # files you produced or trust.  weights_only=True cannot apply:
        # the format stores dict-of-dict cells, deques, and tuples, none
        # of which survive the safetensors subset.  A migration would
        # need a custom (non-pickle) encoding for the sparse tables.
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = FluxConfig(**state["config"])
        if device is not None:
            cfg.device = device
        if cuda_primary is not None:
            cfg.cuda_primary = cuda_primary
        # the grafted matrix is stored in the snapshot - blank the path
        # so __init__ doesn't re-read the teacher file (or fail if it's
        # gone); R_graft below restores the actual tensor
        cfg.teacher_embed = ""
        model = cls(cfg)
        model._restore_state(state)
        return model

    def _restore_state(self, state: dict) -> None:
        """Restore a snapshot dict into this (freshly constructed)
        model - split out of ``load`` so ``clone``/tests can rehydrate
        a state dict without a filesystem round-trip."""
        if state.get("fmt") == 2:
            state = self._v2_to_v1(state)
        cfg = self.config
        model = self
        rg = state.get("R_graft")
        if rg is not None:
            model._R = rg[0].float().to(model._dev)
            model._graft = dict(rg[1])
        rl = state.get("Rl")
        if rl is not None:
            model._Rl = rl.float().to(model._dev)
        if model._Rl is not None:
            model._Reff = model._R.float() + model._Rl
        # remap channel weights by name - survives channel list changes
        saved_w = np.asarray(state["w"], dtype=np.float64)
        saved_ch = state.get("channels")
        if saved_ch and len(saved_ch) == len(saved_w):
            lut = dict(zip(saved_ch, saved_w))
            model._w = np.array([lut.get(ch, 1.0) for ch in model.channels])
            model._w /= model._w.sum()
        elif len(saved_w) == len(model.channels):
            model._w = saved_w
        model._hedge_mix = float(state.get("hedge_mix", 0.0))
        # per-regime hedge state (cond_hedge) - snapshots older than
        # the flag replicate the global weights into both regimes;
        # snapshots older than the fuzzy-channel hedge rows get the
        # name-remapped global weights broadcast instead
        _wR = np.asarray(
            state.get("wR", np.tile(model._w, (2, 1))),
            dtype=np.float64)
        if _wR.shape[-1] != len(model.channels):
            _wR = np.tile(model._w, (int(_wR.shape[0]) or 2, 1))
        model._wR = _wR
        model._hmxR = np.asarray(
            state.get("hmxR", np.zeros(2)), dtype=np.float64)
        model._regime = int(state.get("regime", 0))
        if model._gpu:
            model._hmg.fill_(model._hedge_mix)
            model._wgR = torch.from_numpy(
                model._wR.astype(np.float64)).to(model._dev)
            model._hmxRg = torch.from_numpy(
                model._hmxR).to(model._dev)
            model._regime_g.fill_(model._regime)
            model._reg_used_g.fill_(model._regime)
        model._uni = state["uni"]
        model._total = state["total"]
        model._log_uni_num = state.get(
            "log_uni_num",
            np.log(state["uni"] + model.config.uni_alpha))
        if not isinstance(model._log_uni_num, torch.Tensor):
            model._log_uni_num = torch.from_numpy(
                np.asarray(model._log_uni_num, dtype=np.float32))
        model._log_uni_num = model._log_uni_num.to(model._dev)
        # GPU snapshots store tables={} (state lives in table_triplets)
        # - keep __init__'s per-order skeleton so the triplet rebuild
        # below has its dicts to write into
        saved_tables = {int(k): v for k, v in state["tables"].items()}
        if saved_tables:
            model._tables = saved_tables
        model._totals = state.get("totals") or {
            k: {key: sum(cell.values()) for key, cell in t.items()}
            for k, t in model._tables.items()}
        # sparse-tap tables (host-side; keys are tuples so restore
        # verbatim - older snapshots simply lack the entries)
        if state.get("ptabs"):
            model._ptabs = {tuple(k): v
                            for k, v in state["ptabs"].items()}
        model._ptots = state.get("ptots") or {
            taps: {key: sum(cell.values()) for key, cell in t.items()}
            for taps, t in model._ptabs.items()}
        # class-order tables (host-side; int keys restore verbatim)
        if state.get("gtabs"):
            model._gtabs = {int(k): v
                            for k, v in state["gtabs"].items()}
        model._gtots = state.get("gtots") or {
            k: {key: sum(cell.values()) for key, cell in t.items()}
            for k, t in model._gtabs.items()}
        # word-order tables (host-side)
        if state.get("wtabs"):
            model._wtabs = {int(k): v
                            for k, v in state["wtabs"].items()}
        model._wtots = state.get("wtots") or {
            j: {key: sum(cell.values()) for key, cell in t.items()}
            for j, t in model._wtabs.items()}
        # hybrid class+raw tables (host-side)
        if state.get("htabs"):
            model._htabs = {int(k): v
                            for k, v in state["htabs"].items()}
        model._htots = state.get("htots") or {
            j: {key: sum(cell.values()) for key, cell in t.items()}
            for j, t in model._htabs.items()}
        # GPU backend: prefer decoded triplets; fall back to converting
        # CPU dicts (recompute pair keys + rebuild tot slots)
        if model._gpu:
            tri = state.get("table_triplets")
            if tri:
                model._gt.load_triplets(tri)
                rtots = state.get("table_rowtots")
                if rtots:
                    model._gt.restore_totals(rtots)
            elif model._tables:
                for k, cells in model._tables.items():
                    kk, tt, cc = [], [], []
                    for key, cell in cells.items():
                        for tok, cnt in cell.items():
                            kk.append(key)
                            tt.append(tok)
                            cc.append(cnt)
                    # no sentinel rows - add() accumulates ttot/tocc
                    # from the follower deltas itself; a tok==V cell
                    # would be read as a real follower and index OOB
                    if kk:
                        model._gt.load_triplets(
                            {k: (np.asarray(kk, dtype=np.uint64),
                                 np.asarray(tt, dtype=np.int64),
                                 np.asarray(cc, dtype=np.float32))})
                model._tables = {k: {} for k in model._tables}
                model._totals = {k: {} for k in model._totals}
            csem_g = state.get("csem_rows_gpu")
            if csem_g is not None:
                keys_g, rows_g = csem_g
                rc = model._gt.row_csem
                n_g = min(keys_g.numel(), model._gt.S)
                model._gt.tkeys[rc, :n_g] = keys_g[:n_g].to(
                    model._dev)
                model._csem_rows[:n_g] = rows_g[:n_g].to(
                    model._dev)
                model._gt.used[rc] = int((keys_g != 0).sum())
        else:
            csem_cpu = state.get("csem_cpu")
            if csem_cpu:
                model._csem_cpu = {int(k): np.asarray(v,
                                                     dtype=np.float32)
                                   for k, v in csem_cpu.items()}
            tri = state.get("table_triplets")
            if tri:
                # rebuild CPU dicts from decoded rows (drop tot rows)
                for k, (kk, tt, cc) in tri.items():
                    k = int(k)
                    for key, tok, cnt in zip(kk.tolist(), tt.tolist(),
                                             cc.tolist()):
                        if tok == model.vocab_size:
                            continue
                        cell = model._tables[k].setdefault(key, {})
                        cell[tok] = cell.get(tok, 0.0) + float(cnt)
                model._totals = {
                    k: {key: sum(cell.values())
                        for key, cell in t.items()}
                    for k, t in model._tables.items()}
        model._epi = {int(k): deque(v, maxlen=cfg.epi_max_per_key)
                      for k, v in state["epi"].items()}
        # FIFO + live-count rebuild: prefer saved values; legacy
        # snapshots fall back to re-deriving from the dict (positions
        # sorted so eviction priority approximates insert order)
        model._epi_n = int(state.get("epi_n") or
                           sum(len(d) for d in model._epi.values()))
        fifo = state.get("epi_fifo")
        if fifo is None:
            fifo = sorted(
                ((k, (e[0] if isinstance(e, (tuple, list)) else e))
                 for k, dq in model._epi.items() for e in dq),
                key=lambda kp: kp[1])
        model._epi_fifo = deque(fifo)
        # knn bank (absent in legacy snapshots -> empty bank)
        knn_st = state.get("knn")
        if knn_st is not None and model.config.use_knn:
            kfp, ksucc, kpos, kcur = knn_st
            P = model.config.knn_capacity
            n_r = min(kfp.shape[0], P)
            model._knn_fp[:n_r] = kfp[:n_r].to(
                model._dev).to(model._knn_fp.dtype)
            model._knn_succ[:n_r] = ksucc[:n_r].to(model._dev)
            model._knn_pos[:n_r] = kpos[:n_r].to(model._dev)
            model._knn_cur = min(int(kcur), 1 << 62)
            if model._gpu:
                model._knncur_g.fill_(model._knn_cur)
        model._ring = state["ring"]
        model._H = state["H"]
        model._count = state["count"]
        model._ep_start = state["ep_start"]
        for name, key in (("_A", "A"), ("_c", "c"), ("_proto", "proto")):
            v = state.get(key)
            if v is None:
                v = torch.zeros((cfg.vocab_size, cfg.topic_dim)
                                if key == "A" else (cfg.topic_dim,))
            if not isinstance(v, torch.Tensor):
                v = torch.from_numpy(np.asarray(v))
            setattr(model, name, v.float().to(model._dev))
        c2 = state.get("c2")
        if c2 is not None:
            if not isinstance(c2, torch.Tensor):
                c2 = torch.from_numpy(np.asarray(c2))
            model._c2 = c2.float().to(model._dev)
        elif cfg.bigram_sketch:
            model._rebuild_c2()        # pre-sketch snapshot
        gc = state.get("gc")
        if gc is not None:
            model._gc = {int(k): dict(v) for k, v in gc[0].items()}
            model._gct = dict(gc[1])
            model._gc_fifo = deque(gc[2])
            # derived fuzzy-read index - rebuilt, never serialized
            sh = cfg.gcell_bucket_shift
            model._gc_buckets = {}
            for code in model._gc:
                model._gc_buckets.setdefault(
                    code >> sh, set()).add(code)
        gs = state.get("gs")
        if gs is not None:
            if isinstance(gs, (tuple, list)):
                gs, gs_s = gs[0], (gs[1] if len(gs) > 1 else None)
            else:
                gs_s = None
            if not isinstance(gs, torch.Tensor):
                gs = torch.from_numpy(np.asarray(gs))
            model._S = gs.float().to(model._dev)
            if gs_s is not None:
                if not isinstance(gs_s, torch.Tensor):
                    gs_s = torch.from_numpy(np.asarray(gs_s))
                model._Ss = gs_s.float().to(model._dev)
            else:
                model._Ss = torch.zeros_like(model._S)
        gsd = state.get("gsd")
        if gsd is not None and model.config.gsyn_deep:
            # theta + momentum tuple -> device tensors; a snapshot
            # without gsyn_deep silently seeds fresh (config-off ablate)
            model._gsd_restore(
                tuple(t.float() for t in gsd))
        dn = state.get("dn")
        if dn is not None and model.config.dnet:
            model._dnet_restore(
                tuple(t.float() for t in dn))
        wh = state.get("Wh")
        if wh is not None:
            if not isinstance(wh, torch.Tensor):
                wh = torch.from_numpy(np.asarray(wh))
            model._Wh = wh.float().to(model._dev)
        wh1 = state.get("Wh1")
        if wh1 is not None:
            model._Wh1 = wh1.float().to(model._dev)
        model._last_seen = state["last_seen"]
        model._recent = deque(state.get("recent", []),
                              maxlen=model._recent.maxlen)
        if model._gpu:
            # rebuild device mirrors from the restored host state
            model._ringg = torch.from_numpy(
                model._ring.astype(np.int32)).to(model._dev)
            model._Hg = torch.from_numpy(
                model._H.view(np.int64)).to(model._dev)
            model._unig = torch.from_numpy(
                model._uni.astype(np.float32)).to(model._dev)
            model._seen_g.fill_(-10 ** 9)
            for t, p in model._last_seen.items():
                model._seen_g[t] = p
            model._pos_g.fill_(model._count)
            model._ep_g.fill_(model._ep_start)
            model._prompt_start = model._count
            model._prompt_g.fill_(model._count)
            model._tot_g.fill_(float(model._total))
            model._wg = torch.from_numpy(
                model._w.astype(np.float64)).to(model._dev)
            # derived A projections/norms - recompute once (the same
            # rank-1 updates that build them live in every write path)
            model._Ac = (model._A @ model._Pcsem).float()
            model._An = model._A.norm(dim=1)
            model._Acn = model._Ac.norm(dim=1)
            model._rebuild_gpu_epi()
        model.journal.entries = list(state.get("journal_entries", []))
        model.journal.dropped = state.get("journal_dropped", 0)
        model.journal.dropped_blocks = state.get(
            "journal_blocks_dropped", 0)
        model.journal.recompute()
        model._gaps = deque(state.get("gaps", []), maxlen=cfg.gap_cap)
        # bloom doorkeeper: restore admission counters (absent on pre-
        # bloom snapshots - filters start empty, which just re-sights
        # existing keys as new; harmless since resident cells skip the
        # gate anyway)
        bl = state.get("bloom")
        if bl is not None and model._bloom_h is not None:
            bh, bg = bl
            n_ = min(bh.shape[0], model._bloom_h.shape[0])
            model._bloom_h[:n_] = bh[:n_]
            if bg is not None and model._bloom_g is not None:
                m_ = min(bg.numel(), model._bloom_g.numel())
                model._bloom_g[:m_] = bg[:m_].to(model._dev)
        # zero-gradient controller head (KnobHead): restores the learned
        # routing tables — inspectable counting state, journaled outside
        # the weight journal by design (knobs are their own layer)
        hd = state.get("head")
        if hd is not None:
            from .knobhead import KnobHead
            model.attach_head(KnobHead.load_state_dict(hd))

    def _rebuild_gpu_epi(self) -> None:
        """Rebuild the GPU episodic index from the host ``_epi`` dict -
        entries are (pos, succ, certs) tuples; legacy bare-pos
        snapshots get succ=-1/empty certs."""
        cfg = self.config
        model = self
        ks_all, ps_all, ss_all, cs_all = [], [], [], []
        LC = len(cfg.cert_orders)
        for key, dq in model._epi.items():
            for ent in dq:
                if isinstance(ent, (tuple, list)):
                    p, s_, cv = ent[0], ent[1], \
                        (ent[2] if len(ent) > 2 else ())
                else:                     # legacy bare position
                    p, s_, cv = ent, -1, ()
                ks_all.append(key)
                ps_all.append(p)
                ss_all.append(s_)
                cs_all.append(
                    [int(v) if int(v) < (1 << 63)
                     else int(v) - (1 << 64)
                     for v in cv] + [0] * (LC - len(cv)))
        if not ks_all:
            return
        kk = torch.tensor(
            [_s64(k) for k in ks_all], dtype=torch.int64,
            device=model._dev)
        pp = torch.tensor(ps_all, dtype=torch.int64,
                          device=model._dev)
        ss = torch.tensor(ss_all, dtype=torch.int32,
                          device=model._dev)
        cc_ = (torch.tensor(cs_all, dtype=torch.int64,
                            device=model._dev)
               if LC else None)
        re_ = model._gt.row_epi
        pk_e = model._epi_hash(kk)
        ekr = model._gt.tkeys[re_]
        err = model._gt.traw[re_]
        se, fe, _ee = model._gt._probe_row_w(re_, pk_e)
        enew = ~fe
        if bool(enew.any()):
            ekr[se[enew]] = pk_e[enew]
            err[se[enew]] = kk[enew]
            model._gt.used[re_] = int(enew.sum())
        se2, _ = model._epi_probe(pk_e)
        ER = model._epi_pos.shape[1]
        # distinct column per position of the same key -
        # occurrence rank within its (duplicate) slot group
        se2_l = se2.tolist()
        rank: dict[int, int] = {}
        cols = []
        for s_ in se2_l:
            r = rank.get(s_, 0)
            cols.append((int(model._epi_cnt[s_]) + r) % ER)
            rank[s_] = r + 1
        col = torch.tensor(cols, dtype=torch.int64,
                           device=model._dev)
        model._epi_pos[se2, col] = pp
        model._epi_succ[se2, col] = ss
        if LC:
            model._epi_cert[se2, col] = cc_
        for s_, r in rank.items():
            model._epi_cnt[s_] += r
        model._gt.tocc[re_][se2] = \
            model._epi_cnt[se2].float()

    # -- diagnostics -----------------------------------------------------

    def memory_report(self) -> dict:
        import sys
        epi_pos = sum(len(d) for d in self._epi.values())
        # shared entries - identical on both backends
        approx = {
            "topic_A_MB": self._A.element_size()
            * self._A.nelement() / 1e6,
            "topic_R_MB": self._R.element_size()
            * self._R.nelement() / 1e6,
            "ring_MB": (self._ring.nbytes + self._H.nbytes) / 1e6,
            "epi_keys": len(self._epi),
            "epi_positions": epi_pos,
            "epi_cap": self.config.epi_total_cap,
            "epi_MB_est": (len(self._epi) * 96 + epi_pos * 48) / 1e6,
            "knn_rows": int((self._knn_pos >= 0).sum()),
            "knn_cap": self.config.knn_capacity,
            "knn_MB": (self._knn_fp.nbytes + self._knn_succ.nbytes
                       + self._knn_pos.nbytes) / 1e6,
            "journal_entries": len(self.journal),
            "journal_blocks_dropped": self.journal.dropped_blocks,
            "journal_MB_est": (len(self.journal.entries) * 160
                               + self.journal.block_bytes
                               + self.journal.vec_ring.nbytes
                               + self.journal.w_ring.nbytes) / 1e6,
            "py_overhead_MB": sys.getsizeof(self._tables) / 1e6,
            "gcell_cells": len(self._gc),
            "gcell_followers": sum(len(d) for d in self._gc.values()),
            "gcell_MB_est": (len(self._gc) * 96
                             + sum(len(d) for d in self._gc.values())
                             * 56) / 1e6,
            "gsyn_MB": (self._S.nbytes + self._Ss.nbytes
                        + self._gsyn_ring.nbytes) / 1e6,
            "gsyn_deep_MB": (sum(t.nbytes for t in self._gsd_params())
                             / 1e6 if self.config.gsyn_deep else 0.0),
            "dnet_MB": (sum(t.nbytes for lst in
                            (self._DS, self._Dk, self._Dv, self._Db)
                            for t in lst) / 1e6
                        if self.config.dnet else 0.0),
            "rlive_MB": ((self._Rl.nbytes + self._Reff.nbytes) / 1e6
                         if self._Rl is not None else 0.0),
        }
        if self._gpu:
            c = self.config
            S, M, O = c.gpu_table_slots, c.gpu_row_width, len(c.orders)
            tab_mb = ((O + 2) * S * (8 + 8) + O * S * 4
                      + O * S * M * 8) / 1e6
            csem_mb = (S * 2 * c.csem_dim) / 1e6
            epi_mb = (S * (8 + 8 * c.epi_row)) / 1e6
            ring_mb = (c.vec_delta_cap * (8 * len(self.channels)
                       + 4 * c.topic_dim + 2 * c.csem_dim)) / 1e6
            approx.update({
                "gpu_tables_MB": tab_mb,
                "gpu_rows_used": sum(self._gt.used),
                "csem_MB": csem_mb,
                "epi_gpu_MB": epi_mb,
                "journal_rings_gpu_MB": ring_mb,
            })
            approx["vram_MB_est"] = (tab_mb + csem_mb + epi_mb
                                     + ring_mb + approx["topic_A_MB"]
                                     + approx["topic_R_MB"]
                                     + approx["knn_MB"] + 20)
        else:
            cells = sum(len(t) for t in self._tables.values())
            followers = sum(len(cell) for t in self._tables.values()
                            for cell in t.values())
            approx.update({
                "tables_cells": cells,
                "tables_followers": followers,
                "tables_MB_est": (cells * 96 + followers * 56) / 1e6,
            })
        approx["total_MB_est"] = sum(
            v for k, v in approx.items()
            if k.endswith("_MB") or k.endswith("_MB_est"))
        return approx

    def channel_weights(self) -> dict:
        return {ch: float(self._w[i])
                for i, ch in enumerate(self.channels)}

    def extra_repr(self) -> str:
        return (f"flux(vocab={self.vocab_size}, cells="
                f"{sum(len(t) for t in self._tables.values())}, "
                f"stream={self._count}, journal={len(self.journal)})")
