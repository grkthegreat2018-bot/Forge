"""Optional Triton fused kernels for FLUX (CUDA-primary).

``HAS_TRITON`` reports availability; every kernel has an eager torch
fallback in the model, so this module is always importable.
"""

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _flux_deep_votes_kernel(
        pos_ptr,            # [ER] int64 - ctx_end per cell (-1 empty)
        succ_ptr,           # [ER] int32 - stored successor (-1 none)
        cert_ptr,           # [ER, LC] int64 - certificate suffix hashes
        hg_ptr,             # [cap] int64 - prefix-hash ring
        ring_ptr,           # [cap] int32 - token ring
        cur_ptr,            # [LC] int64 - current context cert hashes
        certw_ptr,          # [LC] int64 - cert order weights
        mlen_ptr,           # [ER] f32 - OUT: per-cell match mass
        end_ptr, floor_ptr, ep_ptr, dk_ptr, pd_ptr,   # device scalars
        fe_ptr,             # [] bool - probe hit; a miss leaves se
                            # pointing at an unrelated row that must
                            # NOT be voted
        CAP: tl.constexpr, DO: tl.constexpr, MX: tl.constexpr,
        LC: tl.constexpr, ER: tl.constexpr,
    ):
        """One program per episodic cell: seed-hash check + backward
        extension for on-ring entries, certificate-graded weight for
        archived (off-ring) entries.  Emits per-cell match mass; the
        caller scatters it deterministically (atomic_add order is
        timing-dependent -> non-reproducible logits).
        Replaces ~25 tiny torch kernels per predict."""
        j = tl.program_id(0)
        pos = tl.load(pos_ptr + j)
        succ = tl.load(succ_ptr + j).to(tl.int64)
        end = tl.load(end_ptr)
        floor = tl.load(floor_ptr)
        ep = tl.load(ep_ptr)
        dk = tl.load(dk_ptr)
        fnd = tl.load(fe_ptr)
        ok = fnd & (pos >= 0) & (pos < end) & (succ >= 0) \
            & (end - DO + 1 >= ep) & (end >= DO - 1) & (end >= 0)
        mlen = pos * 0                       # int64 scalar zero
        if ok:
            if pos >= floor:
                # on-ring: verify the seed-order suffix hash, then
                # extend backwards token-by-token while bytes match
                hp = tl.load(hg_ptr + pos % CAP)
                hb = tl.load(hg_ptr + tl.maximum(pos - DO, 0) % CAP)
                hv = hp - hb * tl.load(pd_ptr)
                if hv == dk:
                    mlen += DO
                    m = 1
                    while m <= MX:
                        pi = pos - m
                        ei = end - m
                        if (pi < floor) | (pi < 0) | (ei < ep) \
                                | (ei < 0):
                            m = MX + 1
                        else:
                            a = tl.load(ring_ptr + pi % CAP)
                            b = tl.load(ring_ptr + ei % CAP)
                            if a != b:
                                m = MX + 1
                            else:
                                mlen += 1
                                m += 1
            else:
                # archived: seed-order mass + deepest matching cert
                mlen += DO
                for l in tl.static_range(LC):
                    cert = tl.load(cert_ptr + j * LC + l)
                    cur = tl.load(cur_ptr + l)
                    w = tl.load(certw_ptr + l)
                    hit = ((cert != 0) & (cert == cur)
                           & (w > mlen)).to(tl.int64)
                    mlen += (w - mlen) * hit
            tl.store(mlen_ptr + j, mlen.to(tl.float32))

    @triton.jit
    def _flux_logit_tail_kernel(
        logits_ptr,         # [V] f32 in/out - already holds
                            # log_uni + order/backoff/deep/topic terms
        log_uni_ptr,        # [V] f32 - log unigram prior
        uni_pv_ptr,         # [V] f32 - smoothed unigram prob
        sims_ptr,           # [V] f32 - Ac @ proto-fingerprint cosine
        csims_ptr,          # [V] f32 - Ac @ csem-row cosine
        kv_ptr,             # [V] f32 - knn successor vote mass
        seen_ptr,           # [V] int64 - last-seen position
        wg_ptr,             # [C] f64 - hedge weights
        pvn_ptr, rn_ptr, cf_ptr, cok_ptr, kvt_ptr, end_ptr,
        V,
        SEMX: tl.constexpr, CSEMX: tl.constexpr, KNNX: tl.constexpr,
        RECX: tl.constexpr, FATX: tl.constexpr,
        USE_SEM: tl.constexpr, USE_CSEM: tl.constexpr,
        BETA_SEM: tl.constexpr, BETA_CSEM: tl.constexpr,
        ALPHA: tl.constexpr, KAPPA: tl.constexpr,
        LIFT: tl.constexpr,
        SPAN_R: tl.constexpr, LDEC_R: tl.constexpr,
        SPAN_F: tl.constexpr, LDEC_F: tl.constexpr,
        FAT_STR: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Fused post-topic logit tail: sem + csem + knn mixture terms
        and recency/fatigue adjustments in ONE V-map pass.  Channel
        gates arrive as device scalars - multiplicative gating keeps
        the kernel branchless and CUDA-graph safe (no host reads)."""
        pid = tl.program_id(0)
        off = pid * BLOCK + tl.arange(0, BLOCK)
        m = off < V
        lg = tl.load(logits_ptr + off, mask=m)

        pvn = tl.load(pvn_ptr)
        if USE_SEM:
            f = (pvn > 1e-8).to(tl.float32)
            w = tl.load(wg_ptr + SEMX).to(tl.float32)
            lg += f * w * BETA_SEM \
                * tl.load(sims_ptr + off, mask=m, other=0.0)

        if USE_CSEM:
            rn = tl.load(rn_ptr)
            okc = tl.load(cf_ptr) & tl.load(cok_ptr) \
                & (rn > 1e-8) & (rn < float("inf"))
            w = tl.load(wg_ptr + CSEMX).to(tl.float32)
            lg += tl.where(okc, w * BETA_CSEM
                           * tl.load(csims_ptr + off, mask=m,
                                     other=0.0), 0.0)

        kvt = tl.load(kvt_ptr)
        f_k = (kvt > 0).to(tl.float32)
        wk = tl.load(wg_ptr + KNNX).to(tl.float32)
        # evidence lift - massive vote mass is self-evidencing
        if LIFT:
            wk = wk + kvt / (kvt + KAPPA)
        kv = tl.load(kv_ptr + off, mask=m, other=0.0)
        upv = tl.load(uni_pv_ptr + off, mask=m, other=1.0)
        lu = tl.load(log_uni_ptr + off, mask=m, other=0.0)
        pk = (kv + ALPHA * upv) / (kvt + ALPHA)
        lg += f_k * wk * (tl.log(tl.maximum(pk, 1e-12)) - lu)

        end = tl.load(end_ptr)
        seen = tl.load(seen_ptr + off, mask=m, other=-(1 << 40))
        age = end - seen
        is_r = (seen >= 0) & (age >= 0) & (age < SPAN_R)
        lg += tl.load(wg_ptr + RECX).to(tl.float32) * tl.where(
            is_r, tl.exp(age.to(tl.float32) * LDEC_R), 0.0)
        is_f = (seen >= 0) & (age >= 0) & (age < SPAN_F)
        lg -= tl.load(wg_ptr + FATX).to(tl.float32) * FAT_STR \
            * tl.where(is_f, tl.exp(age.to(tl.float32) * LDEC_F), 0.0)

        tl.store(logits_ptr + off, lg, mask=m)

