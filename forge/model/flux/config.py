"""FluxLM configuration - the complete hyperparameter surface."""
from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class FluxConfig:
    """Hyperparameters for FluxLM.  All memory is bounded by construction."""

    name: str = "flux"
    vocab_size: int = 65536
    eos_token_id: int = 2

    # auto episode boundary on EOS: after an eos_token_id token is
    # processed, the next token starts a fresh episode (ep_start
    # advance + sketch/recency reset == soft_reset, journaled for
    # revert).  Covers _push, the fused GPU step/seq paths and
    # _ingest_bulk (segmented at EOS); predict-only stream-pool lanes
    # are NOT split (documented gap - prompt EOS is rare for BPE
    # vocabs).  Measured E4 (.tmp/exp_auto_eos.py): unseen transitions
    # -5% NLL but seen transitions +12x - and train corpora re-ingest
    # in fixed order, so learned transitions are the common case.
    # NOT promoted - opt-in per corpus.  eos_token_id=-1 also disables.
    auto_episode: bool = False

    # n-gram channel orders (suffix-hash table per order).  Sweep
    # (.tmp/flux_head_sweep, 2 corpus families + GPU): orders 1-2 are
    # noise channels (dropping them alone: 0.465->0.272 NLL/tok); long
    # orders 48/64 add real in-domain reach.  This set = -69% NLL and
    # -9% diverse-eval loss vs the old (1..32) default, recall intact.
    orders: tuple = (3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    # episodic channel: seed-match order, then extend backwards in the
    # ring.  order=4 seeds more (shorter) matches -> -7% diverse-eval
    # loss vs 8 across both corpus families (sweep S3/S5).
    deep_order: int = 4
    deep_max_hits: int = 24        # seed positions examined per prediction
    deep_max_extend: int = 96      # extra tokens of match extension allowed
    epi_max_per_key: int = 512     # positions kept per seed key
    epi_evict_scan: int = 64       # oldest fifo candidates scored per evict
    epi_hit_w: float = 4.0         # one retrieval hit ~= hit_w recency units
    # archived entries (positions older than the ring) still vote: each
    # entry stores its successor + suffix-hash "certificates" at these
    # orders; the vote weight is the deepest certificate that matches
    # the current context, else the seed order.  0-length tuple disables.
    cert_orders: tuple = (24, 96)
    epi_total_cap: int = 3_000_000  # global FIFO bound on host entries

    ring_capacity: int = 1 << 20   # episodic ring (tokens), ~4 MB int32

    # topic channel (Hebbian hypervector memory)
    topic_dim: int = 192
    topic_decay: float = 0.995
    hebb_lr: float = 0.05
    topic_beta: float = 4.0        # sharpness of cosine -> pseudo-prob

    # semantic-spread channel (generativity): proto = EMA fingerprint of
    # "what kind of token comes next"; tokens distributionally similar
    # to memorized followers get lift -> recombination, not just recall.
    use_sem: bool = True
    sem_beta: float = 4.0
    proto_decay: float = 0.98

    # recency channel
    recency_span: int = 64
    recency_decay: float = 0.90

    # fatigue channel (anti-repeat): tokens emitted moments ago get
    # penalized - intrinsic degeneration pressure, hedged like the rest
    fatigue_span: int = 16
    fatigue_decay: float = 0.80
    fatigue_strength: float = 2.0

    # smoothing + meta-learning
    alpha: float = 0.4             # count smoothing (backoff to unigram)
    uni_alpha: float = 0.05
    hedge_lr: float = 0.5
    hedge_min: float = 1e-3
    hedge_max:  float = 50.0

    # capacity guards (hard memory bound)
    max_cells_per_order: int = 150_000   # decay-halving sweep beyond this
    journal_cap: int = 200_000           # journaled writes kept for undo
    vec_delta_cap: int = 65_536          # fp16 topic deltas kept for undo

    # speed: during bulk ingest, skip deep-match + topic reads inside the
    # Hedge reward (they fall back to the unigram prior). Channel weights
    # still adapt; full rewards apply in live/generation learning.
    fast_ingest: bool = True

    # adaptive read gating (CPU paths): when some order cell already
    # predicts with >= adaptive_read_conf dominance over >=
    # adaptive_read_min_tot evidence, the deep-scan and knn matvec are
    # skipped - the order channels carry the prediction and the skipped
    # channels fall back to the unigram prior for rewards.  Measured
    # ~31% of tokens skippable at +0.04% NLL (probe: .tmp/flux_rd_probes6).
    adaptive_read: bool = True
    adaptive_read_conf: float = 0.9
    adaptive_read_min_tot: float = 2.0

    # sleeping experts (PAQ/C2-style abstention): a channel with NO
    # evidence this step (order key missed, no deep/knn votes, empty
    # topic/csem/proto sketch, no backoff order) does not get scored on
    # the unigram fallback - it abstains and receives the mixture's own
    # p_mix as a weight-preserving neutral reward.  Mode 1 keeps the
    # legacy p_mix (asleep channels still contribute their uni fallback
    # to the mixture) and only fixes the reward.  Mode 2 additionally
    # renormalizes p_mix over awake channels so surprise reflects the
    # channels that actually responded.  Applies to live steps, bulk
    # ingest (where non-order channels were previously scored as pure
    # unigram - systematically under-weighting retrieval channels), and
    # stream-pool rewards.
    sleeping_experts: int = 0             # 0=off 1=neutral 2=renorm

    # calibrated channel rewards: additive-lift channels (topic, sem,
    # csem, recency, fatigue) express the distribution
    # p_c(v) ∝ p_uni(v)·exp(f_c(v)) in the geometric mixture, so their
    # Hedge reward must be p_uni(x)·exp(f_c(x))/Z_c - NOT the raw
    # lift score.  Without this, fatigue's ~1.0-for-everything score
    # monopolizes the weights (measured collapse: w_fatigue≈0.98) and
    # the genuinely predictive channels are starved.  Orders/backoff/
    # deep/knn/uni are already true probability channels - unchanged.
    calib_rewards: bool = True

    # bulk channel re-grounding: _ingest_bulk used to score every
    # non-order channel's hedge reward at the unigram prior (it never
    # ran the deep/knn/backoff reads), so bulk training floored the
    # retrieval channels regardless of their real quality.  This flag
    # computes real rewards in the bulk pass: knn via one [n,live-bank]
    # sim matvec on the PRE-WRITE bank, backoff via the longest
    # evidenced order's prob, deep via pre-block epi probes (host dict).
    # topic/sem/csem keep the unigram fallback (their per-position Z
    # costs a V-scan per position - deferred).
    bulk_channel_rewards: bool = False

    # conditioned hedge (lpaq-style context-selected mixers): keep one
    # AdaHedge weight set per evidence regime instead of a single
    # global _w.  0 = off; 2 = binary thin/warm split on "any order
    # channel had evidence this context".  Offline oracle probe
    # (.tmp/exp_cond_hedge.py): +0.118 nats/tok headroom, ~85% from the
    # binary split - cold contexts want sem/topic, warm want orders.
    cond_hedge: int = 0

    # sparse/noncontiguous order channels (PAQ8-style gapped contexts):
    # each entry is a tuple of offsets below the context end, e.g.
    # ((0,2),(0,2,4),(1,3),(1,3,5)) - keys skip every-other byte, so a
    # single corrupted/novel token can't break the whole suffix match.
    # Host-resident tables on both backends (GPU bookkeep mirrors the
    # writes + applies the thin-gated lift post-step; sparse arms
    # abstain from device hedge with a neutral pmix reward).
    # Measured: div 5.26->4.10, all 3.67->2.92 (CPU & GPU).
    sparse_taps: tuple = ((0, 2), (0, 2, 4), (0, 2, 4, 6),
                         (1, 3), (1, 3, 5), (1, 3, 5, 7),
                         (0, 4), (1, 5))
    # thin-only gating: sparse votes only fire when no contiguous order
    # cell has evidence (regime-0) - they are a fallback expert family
    # for corrupted/novel contexts, not a competitor to exact keys.
    # Measured: ungated sparse dilutes warm-context predictions
    # (fact/nat regress); gating keeps the div-slice win without it.
    sparse_thin_only: bool = True

    # character-class order channels (PAQ word-model analog): suffix
    # keys over byte CLASSES (digit/upper/lower/space/nl/punct/other/
    # high) instead of raw bytes - coarse contexts that generalize on
    # novel text where verbatim byte keys never repeat.  Host tables,
    # "K" journal entries, g<k> channel names; thin-gated like sparse.
    # Measured: (3,) is the sweet spot - all 2.76 vs 2.92 sparse-only;
    # deeper class orders (4,6) fill 8^k cells but vote coarse-wrong on
    # genuinely novel contexts (nat regressed +0.3).
    class_orders: tuple = (3,)
    class_thin_only: bool = True
    # min cell total before a g<k> channel votes - coarse keys fill
    # fast but low-evidence cells predict worse than the unigram
    # prior on genuinely novel contexts
    class_min_tot: float = 8.0

    # word-order channels (PAQ8 word contexts): key = fold of the last
    # ``w`` completed-word hashes + current-word length bucket.  Pure
    # derived state (backward ring scan - nothing to journal), host
    # tables, "W" entries, w<j> channels, thin-gated like sparse.
    word_orders: tuple = ()
    word_thin_only: bool = True

    # hybrid class+raw taps: each pattern is a tuple of (offset,
    # is_raw) pairs - "the last two byte CLASSES + raw last byte"
    # style keys that stay sharper than pure class keys while still
    # generalizing on novel text.  Host tables, "H" journal entries,
    # h<j> channels, thin-gated like the rest.  Measured: single
    # (c3,c2,c1,raw) pattern is optimal - all 2.676->2.662; adding a
    # second correlated arm dilutes.
    hybrid_taps: tuple = (((3, 0), (2, 0), (1, 0), (0, 1)),)
    hybrid_thin_only: bool = True

    # learned-key cell channel "gcell": cells keyed by a SimHash code
    # of the order-sensitive context fingerprint (_c2 sketch + tail
    # feature rows) instead of a verbatim suffix hash - near-similar
    # contexts share cells, so the channel generalizes where no exact
    # key exists.  Same dict-of-dict substrate as _ptabs/_gtabs (host-
    # resident on both backends, journaled "G"/"gE", snapshot "gc"),
    # successor entries are delta-rule scores s += lr*s*(1-s) (saturate
    # ~1).  Fixed-weight lift (gcell_w), thin-gated by default - a
    # bypass channel, not hedge-registered yet.
    gcell: bool = False
    gcell_bits: int = 24          # SimHash width (<= 62)
    gcell_w: float = 0.5          # fixed channel lift weight
    gcell_thin: bool = True       # vote only in thin contexts
    gcell_lr: float = 1.0         # delta-rule step scale
    gcell_max_cells: int = 1 << 20
    # SDM-style fuzzy read: probe cells whose codes sit within a
    # Hamming radius of the query code (similar contexts pool
    # evidence -> generalization without extra stored state).  A
    # coarse bucket index (code >> bucket_shift) prunes the probe
    # set; neighbor contributions decay by neighbor_decay**dist.
    # 0 = exact-match read (previous behavior).
    gcell_read_radius: int = 1
    gcell_bucket_shift: int = 6
    gcell_neighbor_decay: float = 0.5

    # dense delta channel "gsyn": the smooth-function complement to
    # gcell - a small gated delta-rule matrix S [topic_dim, gsyn_dim]
    # updated per token  S <- gsyn_decay*S + beta*s*(v_x - S@k) x k
    # where k = normalize(_Wq^T @ context_features) and v_x = _R[x]
    # (the token's fingerprint - readout scores candidates by
    # R-similarity to the predicted value).  Same substrate class as
    # _A (surprise-scaled delta rule) but key-addressed by a learned
    # projection, so it generalizes like a tiny linear-attention/
    # delta-net layer.  Host-eager update on both backends (outside
    # the CUDA graph, _ptabs precedent); journaled 'GS' factor rows
    # (v_err, k, coeff) -> fp16-precision inverse like 'a' vec_ring.
    gsyn: bool = False
    gsyn_dim: int = 256           # key projection width
    gsyn_w: float = 0.5           # fixed channel lift weight
    gsyn_thin: bool = True        # vote only in thin contexts
    gsyn_beta: float = 0.5        # delta-rule write rate
    gsyn_decay: float = 0.999     # per-token retention gate
    gsyn_temp: float = 4.0        # read softmax temperature
    # two-timescale state (HOPE-style continuum memory): a second S
    # with slower write rate + slower decay - fast state adapts to the
    # current passage, slow state accumulates long-range regularities.
    # 0 = single-timescale.  Same journaled factors, second coeff.
    gsyn_slow_decay: float = 0.0
    gsyn_slow_beta: float = 0.1

    # "warm" threshold shared by the thin-only gates and cond_hedge
    # regime routing: an order cell needs tot > thin_tot to count as
    # real evidence.  0 = binary; 2.0 measured optimum (weak 1-2-count
    # cells are effectively thin - fallback experts still help there).
    thin_tot: float = 2.0

    # fixed weight bonus for retrieval/semantic channels (deep, knn,
    # topic, sem, csem) in thin contexts - they are the only signal
    # there but hedge converges them to ~floor because their per-token
    # rewards are mediocre in warm contexts (where they still earn
    # weight).  0 = off.
    thin_boost: float = 0.2

    # generation: support-restricted sampling - a sampled token must
    # beat thresh * p_uni (some channel actually voted for it).  Fixes
    # the char-salad failure where temperature sampling falls onto the
    # unigram escape floor in thin contexts.  0 = off; argmax paths are
    # unaffected either way.
    support_thresh: float = 1.5

    # distilled linear readout head (distill_head): additive logit
    # weight for Wh @ chat - a trained parametric channel, the only
    # mechanism that generalizes teacher knowledge to novel contexts.
    # 0 = off until a head exists.
    head_w: float = 0.0
    # head input order sensitivity: concat of the last head_orders raw
    # token feature rows (word order) with the normalized topic sketch.
    # in_dim = (head_orders + 1) * topic_dim; 0 = bag-of-features only.
    head_orders: int = 8
    # thin-gate: the head lift only applies where order cells hold no
    # evidence (same regime-0 gate as sparse/class fallback experts).
    # An always-on head pollutes verbatim recall with its generic
    # votes - the head exists to answer where memory is thin.
    head_thin: bool = True
    # head capacity: 0 = linear readout Wh @ x; >0 = trained hidden
    # layer - Wh @ relu(Wh1 @ x), an MLP channel.  Still a single
    # parametric channel, not a transformer - the nonlinear map is
    # what a pure linear head cannot express (measured: linear tops
    # out at marginal-frequency predictions).
    head_hidden: int = 0

    # order-sensitive context sketch: maintain _c2 - an EMA over
    # bigram-bound feature pairs (R[prev][bp1] * R[cur][bp2], fixed
    # permutations) - alongside the bag-of-features _c.  The sketch is
    # concat'd into the distill head input (in_dim += topic_dim) and
    # later feeds learned-key channels: _c alone cannot express
    # "a follows b".  Off by default; enabling changes _head_in_dim,
    # so heads trained without it stop loading (_head_ready shape
    # guard).
    bigram_sketch: bool = False

    # virtual rollout: on thin contexts (no order evidence - the same
    # gate as cond_hedge regime 0) probe the order tables one hop ahead
    # under the argmax continuation and add the 2-step distribution as a
    # fixed-weight evidence lift.  0 = off; ~1.0 = full channel weight.
    rollout_w: float = 0.0

    # unigram escape (Jelinek-Mercer): the emitted distribution gets a
    # constant ε of unigram mass mixed in arithmetically -
    # p_out = (1-ε)·p_mix + ε·p_uni.  Geometric mixtures let one
    # confident-but-wrong channel veto the true token (measured:
    # calibrated weights push diverse-token NLL BELOW unigram); the
    # escape bounds worst-case surprise at -log(ε·uni_x) at the cost of
    # a tiny confidence dilution.  Applied to output logits only -
    # channel rewards stay geometric (hedge measures components, not
    # the escaped mixture).
    uni_escape: float = 0.12              # 0 = off; 0.12 = optimum on
                                          # the sparse+class mixture
                                          # (swept .02/.05/.10/.12/.15/
                                          # .20 - all asymptotes ~2.69,
                                          # .12 keeps rep/fact sharpest)

    # thin-context write gating: when the order tables already cover a
    # context (probe-order cell mass >= write_gate_thin_tot) AND the
    # token was unsurprising, the episodic + knn-bank writes are
    # redundant - skip them.  Order tables/unigram/topic/csem/recency
    # still write every token; only the expensive episodic structures
    # are gated.  Measured -41% epi / -83% knn writes at zero recall
    # loss (probe: .tmp/flux_rd_probes5).  GPU: folded into dok1 and a
    # branchless discard-slot write - no journal, exact revert.
    write_gate: bool = True
    write_gate_probe_order: int = 8   # cell-mass coverage probe
    write_gate_thin_tot: float = 2.0  # mass below this = thin context
    write_gate_epi_s: float = 0.1     # surprise that forces an epi entry
    write_gate_knn_s: float = 0.3     # surprise needed for a knn row

    # gap tracking for selective distillation: positions where the model
    # was SURPRISED (1 - p_mix > gap_surprise) or the context was unseen
    # are recorded so a teacher can score just those contexts.
    track_gaps: bool = True
    gap_probe_order: int = 8
    gap_min_total: float = 3.0
    gap_surprise: float = 0.5
    gap_cap: int = 500_000

    # surprise-gated writes (Titans/MIRAS-style): every mutation scales
    # with 1 - p_mix(x) - well-predicted tokens barely write, surprising
    # ones commit hard.  Saves bounded-table capacity + collisions.
    surprise_gate: bool = True
    surprise_floor: float = 0.05     # minimum write magnitude

    # delta-rule topic writes (DeltaNet/GDN): the A-row update subtracts
    # the current readout along the context direction - error-corrected
    # associative memory, beats additive Hebbian on in-context recall.
    delta_rule: bool = True

    # backoff channel (infini-gram style): the longest order whose
    # context has >= backoff_min_tot evidence, confidence-discounted -
    # hedged alongside every other channel.
    use_backoff: bool = True
    backoff_min_tot: float = 2.0
    backoff_kappa: float = 8.0      # confidence = tot / (tot + kappa)

    # evidence lift (interpolated-Kneser-Ney semantics): at predict time
    # an order/backoff channel's effective weight is w + tot/(tot+kappa)
    # - hedge learns the reliability PRIOR, but a cell with real mass is
    # self-evidencing.  Without this, hedge collapse (weights -> floor)
    # makes injected/learned facts unreachable no matter their counts.
    evidence_lift: bool = True

    # AdaHedge-style adaptive lr: eta_t = ln(C) / mix_cum where mix_cum
    # accumulates the per-token mixability gap (max reward - p_mix).
    # Constant regret on easy streams, worst-case safe.
    adahedge: bool = True
    hedge_eta_max: float = 4.0

    seed: int = 0xF1A5             # deterministic token->hypervector table
    # "cuda" puts the dense readout on GPU (A/R/c + logits + sem matvec);
    # sparse tables/journal always stay in host RAM - the right split for
    # pointer-chasing hash lookups. Falls back to cpu if CUDA absent.
    device: str = "cpu"

    # -- CUDA-primary mode (device="cuda" + cuda_primary=True) ----------
    # Sparse memory moves into open-addressed pair-key tables ON the GPU:
    # (context_key, follower) -> count.  Prediction = one V-wide probe per
    # order; ingest = vectorized chunks (closed-form EMAs, batch probes).
    cuda_primary: bool = False
    gpu_table_slots: int = 1 << 19   # context rows per order
    gpu_row_width: int = 16          # follower cells per context row
    gpu_probe_max: int = 64          # linear-probe iterations
    gpu_bulk_chunk: int = 8192       # tokens per vectorized ingest block
    epi_slots: int = 1 << 18         # GPU episodic index rows
    # positions per seed key - GPU-table analogue of epi_max_per_key.
    # 16 starved deep coverage; 32 = -19% NLL on GPU bench for +0.23GB
    # (sweep S8; 64 showed no further gain).  CPU keeps 512-key deques.
    epi_row: int = 32                # positions per seed key

    # csem: per-CONTEXT semantic fingerprint (contextual recombination -
    # the per-key version of the global `sem` channel).  Own hash table
    # context_key -> fp16 fingerprint row; EMA'd toward follower A-rows.
    use_csem: bool = True
    csem_order: int = 4              # context key order for csem table
    csem_slots: int = 1 << 19        # gpu: row shares gt.S geometry;
                                     # cpu: cap on _csem_cpu dict rows
                                     # (oldest-insert evicted, journaled)
    csem_dim: int = 64
    csem_beta: float = 2.0
    csem_lr: float = 0.15

    # knn channel: approximate-match episodic memory.  Every learned
    # token writes (normalize(context_EMA @ P_csem), successor, pos)
    # into a fixed-capacity ring bank; prediction retrieves the top-k
    # fingerprint-similar past contexts and votes their successors -
    # "attention" over the entire stream, bounded memory, no ring
    # horizon.  This is the generative counterpart of `deep` (exact
    # match): knn fires on *similar* contexts, not just verbatim ones.
    use_knn: bool = True
    knn_capacity: int = 1 << 19      # fingerprint rows (~64 MB fp16)
    # utility eviction (mirrors epi_hit_w scoring): bank-full writes
    # evict argmin(hits*hit_w + pos/count) instead of the FIFO-oldest
    # row - fingerprints that actually won topk retrievals survive.
    knn_utility_evict: bool = False
    knn_evict_hit_w: float = 4.0
    # retrieved neighbours per predict - k=8 captures all of knn's
    # quality contribution on sweeps (0.1431 vs 0.1455 with knn off);
    # 32 only added topk cost + tie surface.
    knn_k: int = 8
    knn_gamma: float = 2.0           # vote weight = relu(sim)^gamma
    knn_min_sim: float = 0.05        # cosine floor for a vote to count
    knn_scan_min: int = 4096         # live-slice granularity (bucket)

    # triton fused kernels (CUDA-primary only): single-launch deep-vote
    # eval + fused V-tail logit assembly.  Eager torch fallbacks cover
    # missing triton / non-CUDA devices; results are bit-comparable.
    use_triton: bool = True

    # torch.compile the _logits_dev op chain inside CUDA-graph captures
    # (inductor fuses ~200 elementwise kernels into a handful - measured
    # 143us vs 370us graphed-eager at V=512).  Custom triton kernels and
    # topk calls stay dynamo-disabled: inductor's topk picks different
    # tie members than eager, which changed logits.  Falls back to the
    # eager chain permanently on any compile failure.
    compile_dev: bool = True

    # teacher-embedding graft: path to a safetensors checkpoint whose
    # embedding table replaces the random bipolar `_R` feature matrix -
    # one transplant upgrades every semantic channel (topic `_A`, proto,
    # csem rows, knn fingerprints all flow from `_R[tok]`).  Graft
    # BEFORE any learning; snapshots persist the grafted matrix.
    teacher_embed: str = ""
    teacher_embed_method: str = "randproj"   # "randproj" | "pca" -
        # randproj (JL) preserves pairwise cosine structure; PCA's top
        # axes are frequency-dominated and collapse semantics (measured
        # on ForgeLM V2 embed.weight)

    # forward() semantics: True = every observed token is learned
    # (classic online-LM behavior - engine prefills reinforce the
    # prompt).  False = multi-token forwards observe-only (context
    # advances, memory untouched); single-token forwards still learn so
    # decode-time live learning survives.  Explicit training always
    # goes through ingest()/teach() regardless.
    learn_prefill: bool = True

