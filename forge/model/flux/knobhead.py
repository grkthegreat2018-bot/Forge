"""KnobHead — the zero-gradient controller head for FluxLM.

Design contract (the three-part ontology):
  - weights hold data     -> channel lifts read from memory tables
  - params are pure calc  -> softmax/lift-inversion math, fixed forever
  - knobs are patterns    -> THIS OBJECT: counting tables keyed by
                             (dominant channel, confidence bin) that
                             route the channel mixture.  Learned by
                             counting + EM, never gradient descent.

The head stores NO token knowledge — it stores *which channel is right
in which kind of context*:

    p(tgt) = w_rest[key]·p_flux(tgt)
           + (1-w_rest[key])·[ lam[key]·p_cand(tgt)
                              + (1-lam[key])·p_head(tgt) ]

    p_head(v) = sum_c a_c[key] · softmax(lift_c / w_c)(v)

Open system: every row is inspectable (report), removable
(remove_pattern), checkpointable (checkpoint/restore), and serializable
(state_dict/load_state_dict — persisted inside .flux snapshots).

Attach: flux.attach_head(KnobHead.load_state_dict(...)) — afterwards
every _predict path routes through remix() including engine.generate.
"""
from __future__ import annotations

import math
import numpy as np
import torch

# Channel list is a property of the MODEL (flux.channels / _chix), but
# the canonical experimental order is kept for features collected by
# the .tmp harnesses; fit() records whatever list it was built with.
DEFAULT_CHANNELS = [
    "o3", "o4", "o6", "o8", "o12", "o16", "o24", "o32", "o48", "o64",
    "p0", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "g3", "h0",
    "deep", "knn", "topic", "sem", "csem", "recency", "fatigue",
    "backoff", "uni", "gcell", "gsyn", "gsyn_deep", "dnet"]

# True sparse-detector channels: they produce lifts only when a
# structural pattern matched, so their presence is a context-type
# signal.  Dense fast-weight channels (gsyn/gsyn_deep/dnet) always
# produce a distribution and are excluded from the "which detector
# fired" key component.
SPARSE_NAMES = {"g3", "gcell"}

CONF_EDGES = np.array([1e-5, 1e-3, 0.02, 0.10, 0.25, 0.45, 0.70, 0.90])


def _conf_bin(conf: float) -> int:
    return int(np.searchsorted(CONF_EDGES, conf))


class KnobHead:
    """Pattern-keyed channel-routing tables for FluxLM.

    Tables:
      mixes  (dom, bin) -> np[C]  EM-fitted channel mixture
      cov    (dom, bin) -> [hit, miss]  Beta coverage counts -> w_rest
      lam    (dom, bin) -> float  EM split between candidate lane and
                                  the full-vocabulary head remix
      vpos   (dom, bin) -> np[C]  mean log p_c(target) per channel —
                                  the product-of-experts veto score:
                                  channels that reliably rank truth
                                  up in this context get positive
                                  weight, unreliable ones veto
      gof    (dom, bin) -> float  EM split additive head vs geometric
                                  (PoE) arm — mixture vs veto usage
    Fallback: fine key misses -> (dom=-1, bin) conf-only row.
    """

    def __init__(self, channels=None, min_n: int = 30, k_cand: int = 16):
        self.channels = list(channels or DEFAULT_CHANNELS)
        self.C = len(self.channels)
        self.min_n = min_n
        self.k_cand = k_cand
        self.mixes: dict[tuple, np.ndarray] = {}
        self.cov: dict[tuple, list[float]] = {}
        self.lam: dict[tuple, float] = {}
        self.vpos: dict[tuple, np.ndarray] = {}
        self.gof: dict[tuple, float] = {}
        self.sparse_ids = [i for i, c in enumerate(self.channels)
                           if c.startswith(("p", "w", "h"))
                           or c in SPARSE_NAMES]

    # ---------------- keys ----------------
    def key(self, dom: int, conf: float) -> tuple:
        return (dom, _conf_bin(conf))

    def _fallback(self, key: tuple) -> tuple:
        return (-1, key[1])

    # ---------------- table reads (hierarchical backoff) ----------------
    def channel_mix(self, key: tuple) -> np.ndarray:
        v = self.mixes.get(key)
        if v is None:
            v = self.mixes.get(self._fallback(key))
        if v is None:
            v = np.full(self.C, 1.0 / self.C)
        return v

    def cov_gate(self, key: tuple) -> float:
        """w_rest = P(target outside candidates | context key)."""
        h = self.cov.get(key) or self.cov.get(self._fallback(key))
        if not h:
            return 0.5
        return h[1] / max(h[0] + h[1], 1e-9)

    def lam_of(self, key: tuple) -> float:
        return self.lam.get(key, self.lam.get(self._fallback(key), 0.5))

    def veto_vec(self, key: tuple) -> np.ndarray:
        """Signed PoE exponents b_c[key] in [-1,1]: centered,
        max-normalized mean log p_c(tgt).  Positive = reliable in this
        context, negative = veto power (its top tokens get demoted)."""
        v = self.vpos.get(key)
        if v is None:
            v = self.vpos.get(self._fallback(key))
        if v is None:
            return np.zeros(self.C)
        b = v - v.mean()
        m = np.abs(b).max()
        return b / m if m > 1e-12 else np.zeros(self.C)

    def gof_of(self, key: tuple) -> float:
        return self.gof.get(key, self.gof.get(self._fallback(key), 0.0))

    def thin_cells(self, min_n: int = 100, min_miss: float = 0.5,
                   top: int = 32) -> list[dict]:
        """Context types where the head fails — generation seeds.
        Returns cells sorted by miss mass with their learned mix."""
        out = []
        for k, h in self.cov.items():
            n = h[0] + h[1]
            mr = h[1] / max(n, 1e-9)
            if n >= min_n and mr >= min_miss:
                out.append({"key": k,
                            "dom": (self.channels[k[0]] if k[0] >= 0
                                    else "*"),
                            "conf_bin": k[1], "n": n, "miss": mr,
                            "mass": n * mr,
                            "mix": self.channel_mix(k)})
        out.sort(key=lambda d: -d["mass"])
        return out[:top]

    # ---------------- open-system ops ----------------
    def report(self, key: tuple | None = None, top: int = 8) -> str:
        """Human-readable dump of one cell or the strongest cells."""
        lines = []
        keys = [key] if key is not None else sorted(
            self.mixes, key=lambda k: -self.cov.get(k, [0, 0])[0]
                                            - self.cov.get(k, [0, 0])[1])
        for k in keys[:24]:
            a = self.mixes[k]
            top_c = np.argsort(-a)[:top]
            dist = ", ".join(f"{self.channels[j]}={a[j]:.2f}"
                             for j in top_c if a[j] > 0.01)
            h = self.cov.get(k, [0.0, 0.0])
            n = h[0] + h[1]
            lam = self.lam.get(k)
            b = self.veto_vec(k)
            vb = int(b.argmax()); vl = int(b.argmin())
            lines.append(
                f"key(dom={'*' if k[0] < 0 else self.channels[k[0]]}, "
                f"conf_bin={k[1]}) n={int(n):6d}  "
                f"w_rest={self.cov_gate(k):.2f}"
                + (f" lam={lam:.2f}" if lam is not None else "")
                + (f" gof={self.gof.get(k):.2f}" if k in self.gof
                   else "")
                + (f"  veto +{self.channels[vb]}"
                   f"/-{self.channels[vl]}"
                   if np.abs(b).max() > 0.01 else "")
                + f"  {dist}")
        return "\n".join(lines)

    def remove_pattern(self, key: tuple) -> bool:
        """Forget one pattern cell entirely (post-checkpoints rely on
        the conf-only fallback row)."""
        gone = key in self.mixes
        self.mixes.pop(key, None)
        self.cov.pop(key, None)
        self.lam.pop(key, None)
        self.vpos.pop(key, None)
        self.gof.pop(key, None)
        return gone

    def checkpoint(self) -> dict:
        return {"mixes": {k: v.copy() for k, v in self.mixes.items()},
                "cov": {k: list(v) for k, v in self.cov.items()},
                "lam": dict(self.lam),
                "vpos": {k: v.copy() for k, v in self.vpos.items()},
                "gof": dict(self.gof)}

    def restore(self, cp: dict) -> None:
        self.mixes = {k: v.copy() for k, v in cp["mixes"].items()}
        self.cov = {k: list(v) for k, v in cp["cov"].items()}
        self.lam = dict(cp["lam"])
        self.vpos = {k: v.copy() for k, v in cp.get("vpos", {}).items()}
        self.gof = dict(cp.get("gof", {}))

    # ---------------- serialization ----------------
    def state_dict(self) -> dict:
        return {"channels": self.channels, "min_n": self.min_n,
                "k_cand": self.k_cand,
                "mixes": self.mixes, "cov": self.cov, "lam": self.lam,
                "vpos": self.vpos, "gof": self.gof}

    @classmethod
    def load_state_dict(cls, st: dict) -> "KnobHead":
        h = cls(channels=st["channels"], min_n=st.get("min_n", 30),
                k_cand=st.get("k_cand", 16))
        h.mixes = st["mixes"]
        h.cov = st["cov"]
        h.lam = st["lam"]
        h.vpos = st.get("vpos", {})
        h.gof = st.get("gof", {})
        return h

    # ---------------- live observation (counting, no gradient) --------
    def observe(self, key: tuple, resp: np.ndarray) -> None:
        """EM responsibility accumulation toward a key's channel mix."""
        if key not in self.mixes:
            self.mixes[key] = np.full(self.C, 1.0 / self.C)
        # online soft-EM: shrink toward observed responsibility
        a = self.mixes[key]
        eta = 1.0 / (1.0 + self.cov.get(key, [0, 0])[0]
                     + self.cov.get(key, [0, 0])[1] + 1.0)
        self.mixes[key] = a * (1 - eta) + resp * eta

    def observe_cov(self, key: tuple, hit: bool) -> None:
        h = self.cov.setdefault(key, [0.0, 0.0])
        h[0 if hit else 1] += 1.0

    def observe_vpos(self, key: tuple, ltgt: np.ndarray) -> None:
        """Running mean of log p_c(target) -> the PoE veto stats."""
        v = self.vpos.get(key)
        if v is None:
            self.vpos[key] = ltgt.astype(np.float64).copy()
            return
        n = sum(self.cov.get(key, [0.0, 0.0])) + 1.0
        v += (ltgt - v) / n

    # ---------------- the remix (pure calc over weights) --------------
    def _chan_stats(self, cap: dict) -> np.ndarray:
        """Per-channel (max, log1p count, log1p mass) -> [C*3] stats."""
        st = np.zeros(self.C * 3, dtype=np.float32)
        sp = cap.get("_sparse_stats") or {}
        for j, ch in enumerate(self.channels):
            d = cap.get(ch)
            if torch.is_tensor(d):
                if d.numel() == 0:
                    continue
                st[3 * j] = float(d.float().max())
                st[3 * j + 1] = math.log1p(float((d > 0).sum()))
                st[3 * j + 2] = math.log1p(
                    float(d.float().clamp(max=20).exp().sum()))
            elif ch in sp:
                st[3 * j], st[3 * j + 1], st[3 * j + 2] = sp[ch]
        return st

    def _dom_ch(self, scal: np.ndarray) -> int:
        mx = np.asarray(scal[: self.C * 3:3], dtype=np.float64)
        cnt = np.asarray(scal[1: self.C * 3:3], dtype=np.float64)
        mx = np.nan_to_num(mx, nan=-1e9, posinf=1e9, neginf=-1e9)
        s = np.array([mx[j] if cnt[j] > 0 else -1e9
                      for j in self.sparse_ids])
        if s.size and s.max() > -1e8:
            return int(self.sparse_ids[int(s.argmax())])
        return int(mx.argmax())

    def _chan_probs(self, flux, cap: dict, warm: int,
                    a: np.ndarray, b: np.ndarray | None = None,
                    cand_ids=None):
        """(p_add[V], q_cand, lg_geo[V]) — the channels' own views.
        lg_geo = sum_c b_c·log p_c(v), the product-of-experts arm
        (tokens outside a sparse channel's support get the uniform
        floor, which is a constant under softmax)."""
        c = flux.config
        lw = (flux._wR[warm] if getattr(c, "cond_hedge", 0)
              else flux._w)
        dev = lg_dev = getattr(flux, "_dev", torch.device("cpu"))
        V = flux.vocab_size
        p = torch.zeros(V, device=dev)
        lg_geo = torch.zeros(V, device=dev) if b is not None else None
        per = cap.get("_sparse_raw", {}) or {}
        qc = torch.zeros(len(cand_ids), device=dev) \
            if cand_ids is not None else None
        ids = (torch.as_tensor(cand_ids, device=dev).long()
               if cand_ids is not None else None)
        for j, ch in enumerate(self.channels):
            aj = float(a[j])
            bj = float(b[j]) if b is not None else 0.0
            if aj <= 0 and bj == 0:
                continue
            w = (float(lw[flux._chix[ch]]) if ch in flux._chix
                 else 1.0)
            w = max(abs(w), 1e-6)
            d = cap.get(ch)
            if torch.is_tensor(d) and d.numel() == V:
                lc = torch.log_softmax(d.float() / w, -1)
                if aj > 0:
                    pc = lc.exp()
                    p += aj * pc
                    if qc is not None:
                        qc += aj * pc[ids]
                if bj != 0:
                    lg_geo += bj * lc
            elif isinstance(d, dict) and d:
                # host-path cap: {tok: lift} dicts
                ii = torch.as_tensor(list(d.keys()),
                                     device=dev).long()
                vv = torch.as_tensor(list(d.values()),
                                     device=dev).float() / w
                lpv = vv.log_softmax(-1)
                if aj > 0:
                    p.index_add_(0, ii, aj * lpv.exp())
                    if qc is not None:
                        lut = {int(t): float(x)
                               for t, x in zip(d, lpv.exp().tolist())}
                        qc += aj * torch.as_tensor(
                            [lut.get(int(t), 0.0) for t in cand_ids],
                            device=dev)
                if bj != 0:
                    lg_geo.index_add_(0, ii, bj * lpv)
            elif ch in per:
                v, idx = per[ch]
                vv = torch.as_tensor(np.asarray(v), device=dev).float() / w
                ii = torch.as_tensor(np.asarray(idx), device=dev).long()
                lpv = vv.log_softmax(-1)
                if aj > 0:
                    p.index_add_(0, ii, aj * lpv.exp())
                    if qc is not None:
                        lut = {int(t): float(x)
                               for t, x in zip(ii.tolist(),
                                               lpv.exp().tolist())}
                        qc += aj * torch.as_tensor(
                            [lut.get(int(t), 0.0) for t in cand_ids],
                            device=dev)
                if bj != 0:
                    lg_geo.index_add_(0, ii, bj * lpv)
        return p, qc, lg_geo

    def remix(self, lg: torch.Tensor, cap: dict, flux,
              warm: int | None = None) -> torch.Tensor:
        """Routed next-token logits: w_rest·p_flux + (1-w_rest)·
        [lam·p_cand + (1-lam)·arm], where the head arm blends the
        additive mix with the geometric (PoE) arm by gof[key].
        Returns log-probabilities."""
        dev = lg.device
        lp = torch.log_softmax(lg.float(), -1)
        conf = float(lp.max().exp())
        key = self.key(self._dom_ch(self._chan_stats(cap)), conf)
        a = self.channel_mix(key)
        w_rest = self.cov_gate(key)
        lam = self.lam_of(key)
        gof = self.gof_of(key)
        b = self.veto_vec(key) if gof > 0 else None
        if warm is None:
            warm = int(flux._dthin_g.item()) if flux._gpu else \
                getattr(flux, "_regime", 0)
        cand = torch.topk(lg, min(self.k_cand, lg.numel())).indices
        p_head, qc, lg_geo = self._chan_probs(
            flux, cap, warm, a, b, cand_ids=cand.tolist())
        if lg_geo is not None:
            p_head = (1 - gof) * p_head + gof * torch.softmax(
                lg_geo, -1)
        p = w_rest * lp.exp()
        if qc is not None and float(qc.sum()) > 1e-12:
            qc = qc / qc.sum()
            lane = torch.zeros_like(p_head).index_add_(0, cand, qc)
            p += (1 - w_rest) * (lam * lane + (1 - lam) * p_head)
        else:
            p += (1 - w_rest) * p_head
        return torch.log(p.clamp(min=1e-12))

    # ---------------- fitting (counting + EM, no gradient) ------------
    def fit(self, feats: dict, tgt_ids: np.ndarray,
            rounds: int = 8) -> "KnobHead":
        """Batch-EM fit over collected per-position channel evidence.

        feats: {scal [N,C*3], ltgt [N,C], csup [N,K,C], cid [N,K],
                cpr [N,K]}
        tgt_ids: realized next-token ids [N].
        """
        scal = feats["scal"]
        ltgt = feats["ltgt"].astype(np.float32)
        csup = feats["csup"].astype(np.float32)
        cid = feats["cid"]
        conf = np.exp(np.clip(feats["cpr"][:, 0].astype(np.float64),
                              None, 0))
        N = len(tgt_ids)
        keys = np.empty(N, dtype=object)
        hit = (cid == tgt_ids[:, None]).any(-1)
        for i in range(N):
            k = self.key(self._dom_ch(scal[i]), conf[i])
            keys[i] = k
            self.observe_cov(k, bool(hit[i]))
            self.observe_cov(self._fallback(k), bool(hit[i]))

        p = np.exp(np.clip(ltgt.astype(np.float64), -30, 0))
        resp = {}
        cnt = {}
        for _ in range(rounds):
            resp.clear(); cnt.clear()
            for i in range(N):
                a = self.channel_mix(keys[i])
                mix = max(float((a * p[i]).sum()), 1e-300)
                for k in (keys[i], self._fallback(keys[i])):
                    if k not in resp:
                        resp[k] = np.zeros(self.C); cnt[k] = 0
                    resp[k] += a * p[i] / mix
                    cnt[k] += 1
            self.mixes = {k: resp[k] / max(cnt[k], 1)
                          for k in resp if cnt[k] >= self.min_n}
            for k in list(self.mixes):
                s = self.mixes[k].sum()
                self.mixes[k] = (self.mixes[k] / s if s > 1e-12
                                 else np.full(self.C, 1.0 / self.C))

        # nested lam: per-key EM between candidate lane and head remix
        pc = np.exp(np.clip(csup.astype(np.float64), -30, 0))
        mixes = np.stack([self.channel_mix(k) for k in keys])
        pcand = np.einsum("nc,njc->nj", mixes, pc)
        pcand /= pcand.sum(-1, keepdims=True).clip(min=1e-12)
        p_hit = (pcand * (cid == tgt_ids[:, None])).sum(-1)
        p_hd = (mixes * p).sum(-1)
        self.lam = {}
        for k in set(keys.tolist()):
            m = np.array([kk == k for kk in keys])
            if m.sum() < 50:
                continue
            l = 0.5
            M0, M1 = p_hit[m], p_hd[m]
            for _ in range(60):
                mix = (M0 * l + M1 * (1 - l)).clip(min=1e-300)
                l = float((M0 * l / mix).mean())
            self.lam[k] = l

        # vpos: raw mean log p_c(tgt) per key (veto_vec centers it)
        vsum: dict[tuple, np.ndarray] = {}
        vcnt: dict[tuple, int] = {}
        for i in range(N):
            for k in (keys[i], self._fallback(keys[i])):
                if k not in vsum:
                    vsum[k] = np.zeros(self.C)
                    vcnt[k] = 0
                vsum[k] += ltgt[i]
                vcnt[k] += 1
        self.vpos = {k: vsum[k] / vcnt[k] for k in vsum
                     if vcnt[k] >= self.min_n}

        # gof: EM split additive head vs geometric (PoE) arm.
        # p_geo(tgt) needs a full-vocab normalizer; feats only carry
        # the K candidates, so approximate Z over cand + leftover mass
        # priced at the geo-geo mean — honest on mid/high-conf keys.
        if self.vpos:
            bl = np.stack([self.veto_vec(k) for k in keys])  # [N,C]
            gnum = np.exp(np.clip((bl * ltgt).sum(-1), -60, 0))
            gcand = np.exp(np.clip(
                np.einsum("nc,njc->nj", bl, csup), -60, 0))
            rest = (1 - pcand.sum(-1)).clip(min=0)
            z = gcand.sum(-1) + rest * np.exp(
                np.clip((bl * ltgt).sum(-1) - 4, -60, 0))
            p_geo = gnum / np.clip(z, 1e-300, None)
            self.gof = {}
            for k in set(keys.tolist()):
                if k not in self.vpos:
                    continue
                m = np.array([kk == k for kk in keys])
                if m.sum() < 50:
                    continue
                g = 0.5
                M0, M1 = p_geo[m], p_hd[m]
                for _ in range(60):
                    mix = (M0 * g + M1 * (1 - g)).clip(min=1e-300)
                    g = float((M0 * g / mix).mean())
                self.gof[k] = g
        return self
