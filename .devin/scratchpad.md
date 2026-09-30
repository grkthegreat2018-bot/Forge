# R53 research: LLM-like abilities for FLUX without bolting on a transformer

Session notes — literature survey + codebase fit map. Goal: fluent generation,
semantic generalization, ICL/instruction-adjacent behavior — while preserving
FLUX's live learning, unbounded context, ~700 tok/s, journaled writes, <2GB VRAM.

## Framing that makes this NOT "slapping a model on"

MIRAS (Behrouz et al., arXiv:2504.13173) proves the unifying view: every
sequence model = associative memory module + (i) substrate, (ii) internal
objective ("attentional bias"), (iii) retention gate, (iv) update rule.
Transformers, Titans, DeltaNet, linear RNNs are all instances.

FLUX is ALREADY an instance: cells = nonparametric memory w/ count updates;
`_A` = parametric delta-rule matrix (config.py:284 already cites DeltaNet/GDN);
surprise write gates = retention gating. A dense parametric channel is a design
point INSIDE the same family — not a foreign graft. "Attention approximates
SDM" (Bricken & Pehlevan, NeurIPS'21) is the other anchor: softmax attention ≈
continuous read over a sparse distributed memory. FLUX's hash cells ARE an SDM
with hard addressing; the dense channel is its learned continuous counterpart.

## What the literature says works

- LoLCATs (ICLR'25): linearize a trained transformer — attention-transfer MSE
  + LoRA — 0.2% params, 0.04-0.2% of prior tokens, 20+ MMLU pts. Distillation
  into subquadratic is cheap and works at our scale easily.
- Mamba-in-the-Llama (2408.15237): distill transformer -> linear RNN reusing
  attention projection weights; quarter-attention hybrid ~= full teacher on
  chat; perfect needle@20x distillation length. Precedent for "dense channel
  distilled from V2 keeps most ability".
- Titans (NeurIPS'25): deep MLP memory learned at test time via surprise;
  >2M ctx needle wins. Variants: memory-as-layer / gated / persistent.
- HOPE / Nested Learning (NeurIPS'25, 2512.24695): continuum memory system —
  modules at different update frequencies; self-modifying update rules.
  Maps directly onto FLUX's multi-rate structure (per-token cells, EMA _c/_A,
  consolidated ngrams, hedge weights).
- Gated DeltaNet (ICLR'25, NVlabs): gating (fast erase) + delta rule (targeted
  update); beats Mamba2/DeltaNet on LM + in-context retrieval; parallel chunk
  training exists. fla-org/flash-linear-attention has Triton impls (Windows
  Triton status = we already gate on HAS_TRITON; eager fallback fine at T=1).
- kNN-LM (Khandelwal'20; Drozdov'22 analysis): retrieval channel + neural
  channel beats either; gains concentrate where retrieved ctx is SEMANTICALLY
  similar -> learned fingerprints are the lever, not bigger banks.
- Infini-gram (Liu'24): ∞-gram over 5T tokens = ~47% top-1 next-token and
  REDUCES neural PPL when interpolated. FLUX's ngram channels aren't the
  bottleneck — the missing ~half is exactly what a dense channel supplies.
- Memory Layers at Scale (Meta, ICML'25): product-key memory > 2x-compute
  dense on factual tasks. Validates sparse storage — but adds capacity, not
  computation; our tables already do this. Not the missing piece.
- MemoryLLM (ICML'24): fixed-size latent memory pool self-updates ~1M times
  w/o degradation — evidence a small parametric memory can carry live learning.
- Larimar (ICML'24, IBM): Kanerva-machine episodic memory on an LLM — one-shot
  fact write, selective forgetting, 4-10x faster editing. = formal version of
  FLUX's teach()/revert(); supports keeping the journal-based approach.
- EM-LLM (ICLR'25): surprise-segmented episodic events + similarity+contiguity
  retrieval — upgrade path for `deep` channel's fixed-order certificates.
- HippoRAG (NeurIPS'24): KG + Personalized PageRank over memory = multi-hop
  QA +20% vs iterative retrieval — upgrade path for `deep` votes (transitive
  activation through epi graph).

## What FLUX already has (subagent map — verified)

- `distill_head`: `_Wh [V,in]` / MLP `Wh1+Wh` — ONLY autograd path
  (learnmixin.py:2973-3122, AdamW, soft-CE to teacher top-k). Applied as
  fixed-weight additive channel, thin-gated (`predictmixin.py:720-725`).
  Input = concat(normed `_c`, last head_orders=8 `_R` rows) L2-normed;
  in_dim = (1+8)*192 = 1728. Revert via "WH" journal blob.
- `_A [V,192]` delta-rule Hebbian matrix — surprise-gated, per-token,
  `distill_hebbian` already writes teacher probs into it (learnmixin.py:2898).
- `KDALayer` (forge/keys/attention/kda_key.py:98): full gated delta rule
  S_t = Diag(α)S + βk(v−(αS)ᵀk), snapshot_state/reset_state, standalone.
  ~190K params at d=192. Ready-made dense channel.
- `TitanMemory` (forge/keys/architecture/titan_memory_key.py): low-rank
  fast-weight store, surprise write, zero-init gate.
- `track_gaps`: records surprise positions for SELECTIVE teacher scoring —
  selective-distillation infra already wired.
- `graft_teacher`: teacher embed.weight -> `_R` (randproj/PCA). All semantic
  channels flow from `_R`.
- Channel plumbing: `channels` list + `_w` AdaHedge (29 channels), OR the
  `head_w`/`rollout_w` fixed-weight bypass precedent. `_pmask` for
  host-resident channels. Snapshot/revert contracts mapped.
- Quant stack for cheap dense weights: FastINT8Linear (FP8 tensor-core on
  Blackwell), NVFP4Linear, W8A8, Quamba2 (SSM-specific W4A8).
- Mamba3Block standalone (forge/engine/mamba3.py), DSparkHead low-rank
  V×r GRU head pattern, MTPHead tied-embedding pattern.

## Gap analysis (why FLUX isn't "LLM-like" today)

1. No compositional sequence function: head input is a fixed bag (EMA sketch +
   last-k `_R` rows); measured ceiling = "marginal-frequency predictions"
   (config.py:230). No hidden state evolves across the sequence.
2. Fingerprints are order-agnostic + randproj: knn/csem can't encode syntax or
   entity bindings; roadmap already flags "order-sensitive head input".
3. `_A` maps token->topic (192-dim); it doesn't model token->token transition
   structure at sentence scale. DeltaNet state does exactly this.
4. Vocab 65536 fine; not the blocker.
5. Dense-channel generalization requires the gradient path — only
   `distill_head` has one. Extend it, don't add a second framework.

## Recommended design — "dense" channel (R53)

Channel "dnet": small gated-delta-rule stack as a new mixture member.

- Substrate: 1-4 x KDALayer at d=192-384 (or GDN port from fla if Triton
  cooperates). State S per head: KB-scale. Params 0.2-10M -> 0.4-40MB bf16,
  ~10MB int8. FAR inside 12GB budget (tables ~1.7GB already).
- Readout to vocab: `o @ _A.T` (free, reuses 50MB matrix) and/or low-rank
  `W_o = U[V,r] @ W[r,d]` (r=64 -> ~17MB fp32 at V=65536).
- Input: `_R[tok]` rows + topic sketch — same features as head, but now the
  state carries history. Optionally reuse `_head_feats` rows.
- Mixture entry: start as fixed-weight thin-gated bypass (head_w precedent —
  `head_thin` gate already = "vote only where memory is thin"), graduate to
  full Hedge channel (needs `_channel_prob` + `_logits_dev` + reward wiring,
  all mapped in subagent report §2).
- Training: extend `distill_head` -> `distill_dnet`: stream flux-ingested
  corpus; teacher (V2 via ForgeEngine, already loaded by distill_flux.py)
  scores positions — prioritizing `track_gaps` surprise points; soft-CE to
  teacher top-k + optional hidden-state matching for `_R` alignment
  (LoLCATs-style attention-transfer analogue). AdamW offline.
- Live learning preserved: the delta rule IS per-token SGD — channel keeps
  updating at inference exactly like `_A` does; journal S snapshots/deltas
  for revert() ("WH"-style blob; S is tiny so revert is trivial).
- Novel twists (directive C — baseline + novel):
  a. Surprise-gated writes: β_t <- f(1 - p_mix) — Titans does this with its
     own gradient; FLUX has ensemble surprise FREE from the hedge update.
     Wiring mixture surprise into a parametric write gate is a novel cross.
  b. Two-timescale delta states (γ_fast ~0.95 working, γ_slow=1.0 persistent)
     — HOPE continuum memory at ~zero cost. consolidate() then distills
     fast-state -> cells periodically (self-distillation, extends existing
     consolidate() episodic->ngram).
  c. Order-sensitive fingerprint upgrade: replace `_c` EMA for the dense
     channel's key side with a learned contextual projection distilled to
     match teacher hidden states (cosine distill on corpus slice). Upgrades
     knn/csem/sem simultaneously — highest leverage per LoC.

## Supporting upgrades (cheap, additive)

- `_R` semantic distillation: learn P so flux fingerprints ~ teacher hidden
  geometry (replaces frozen randproj; one P upgrades every semantic channel).
- Multi-hop deep votes: PPR/spreading activation over epi graph (HippoRAG).
- Event segmentation: surprise boundaries for epi entries (EM-LLM) — better
  certificates than fixed 24/96-gram at low extra cost.
- Selective distillation at scale: `track_gaps` already targets where teacher
  signal matters — saves 60-90% of teacher forwards vs full-corpus scoring.

## Honest limits

- ~50M-param dense channel + memory tables will NOT reach 7B-LLM open-domain
  reasoning. Reachable: fluent generation, semantic generalization, instant
  factual learning (already exceeds LLMs), strong ICL/associative recall
  (delta nets are SOTA at MQAR-class tasks at small scale), multi-hop QA via
  epi-graph activation. True reasoning stays a teacher-time property — which
  is fine: distill loop already exists.
- KDA._scan is a Python loop — fine for T=1 live steps; bulk training wants
  the chunked parallel form (fla) or accept slower distill throughput.

## Validation plan (directive F — numbers first)

- Baseline: current NLL/tok on held-out diverse eval + teach-recall probes.
- Test 1 (smallest): order-sensitive head input (bigram-bound sketch) —
  roadmap-flagged, ~20 lines, measure NLL delta.
- Test 2: KDA channel off Hedge bypass, distilled on ~10M tokens of teacher
  top-k at gap+uniform positions — measure NLL on THIN contexts specifically
  (that's where it should win; warm contexts should be unchanged by design).
- Test 3: surprise-gated β vs uniform β — A/B on live stream.
- VRAM: torch.cuda.memory_allocated before/after; target < +0.5GB total.
- Speed: keep >=600 tok/s live; dense path adds ~1-3 matvecs/step.

## Rejected / not-now

- Bolting V2 onto inference (user's constraint — correct: doubles mem,
  kills live-learning purity; teacher stays at distill-time only).
- Product-key replacement of hash cells (would void instant writes/journal).
- Bigger n-gram orders alone (infini-gram shows the ~47% ceiling; diminishing).


---

# R53 target spec (user requirements) - two-part gen core

User reqs: (1) keep perfect recall/no-transformer/searchable weights, (2) separate parametric 'gen' weights alongside recall weights, (3) LLM-like next-token gen w/o transformer limits, (4) self-modifying prediction weights like recall weights, (5) perfect recall of active chats, (6) flexible gen params like recall params, (7) readable gen weights - no decompile, (8) recall+gen = agentic LLM.

## Architecture: FLUX = recall core + gen core, Hedge-arbitrated

RECALL CORE (unchanged): orders/backoff/deep/knn/class/sparse/hybrid/word + topic/sem/csem + recency/fatigue. Journal, revert, snapshot as today.

GEN CORE (new, separate weight store, separate journal ops):

### gcell - learned-key cell channel (primary gen mechanism)
- Same substrate as recall cells: _GPUCellTables-style dict {key -> {succ: score}} BUT key is a LEARNED semantic code, not a verbatim suffix hash.
- Key path: _head_input_dev(end) features [1728] -> W_k (learned 1728->192, the only opaque matrix) -> quantize to b-bit code via LSH bands (SimHash: sign(k�P_l)) -> dict key.
- Read: union cells at Hamming ball radius r over band codes -> sparse succ votes -> true prob channel in Hedge (same _channel_prob interface).
- Write: allocate-or-update nearest cell, delta-rule on succ scores (v += beta(y - v)) surprise-gated by existing 1 - p_mix.
- Readable: explain(ctx) -> which cell codes hit, their succ histograms, write provenance via journal. Self-modifying: per-token writes like recall.
- This is literally a differentiable-SDM / learned-LSH channel - MIRAS inside FLUX's native sparse form. Novel vs literature.

### gsyn - dense delta-rule smooth channel (secondary)
- 1-2x KDALayer (kda_key.py:98) or TitanMemory at d=192-384; S updated per token by gated delta rule (parametric self-modify).
- Readout: o @ _A.T (reuses 50MB matrix) + low-rank W_o (V x r, r=64).
- Covers smooth/grammar patterns a sparse store can't. Probe-readable via probe(ctx) (not byte-readable - mitigated by small size + per-channel logit decomposition).

## Req -> mechanism map
- Separate weights: gen core tensors live in own namespace _gen_*, own snapshot keys, own journal ops (g-prefix), revertable independently.
- Self-modifying: gcell per-token delta writes (journaled); gsyn per-token delta rule on S (journaled snapshot deltas); hedge _w already self-modifies.
- Perfect recall preserved: cond_hedge thin/warm gating - gen channels thin-gated (head_thin precedent), recall owns warm; journal keeps recall exact.
- Flexible gen params: extend SamplingParams + expose mem_bias scalar (recall-vs-gen dial), per-channel temp, existing uni_escape/top-p.
- Readable: gcell = dict entries (printable); explain_logits() exact per-channel decomposition (mixture is additive - free); gsyn probe API; journal = audit trail.
- LLM-like: gcell generalizes (similar ctx -> same code); gsyn adds compositionality; distill_head-style training extended -> distill_gen on V2 top-k at track_gaps surprise positions.

## Numbers
- gcell: W_k 1728x192 = 0.33M params; cells O=2, S=2^19, M=16 ~ +160MB GPU (same class as existing tables).
- gsyn: KDA d=192 ~190K params (0.4MB) + S 37KB; or d=384 4-layer ~5-15M ~ 30-60MB bf16 / ~15MB int8 (FastINT8Linear).
- Total delta < +0.5GB. Speed: +2-4 matvecs/step, keep >=600 tok/s.

## Build order
M1 explain_logits() + cell inspector (readability req, small).
M2 order-sensitive head input (roadmap-flagged prereq for W_k).
M3 gcell channel: W_k train via distill (or self-sup: minimize within-cell succ entropy); Hedge-registered; journal ops g*; snapshot keys.
M4 gsyn channel (KDA, thin-gated bypass first, then full channel).
M5 surprise-gated beta + two-timescale states (novel crosses).
M6 mem_bias + gen-param surface in SamplingParams.
M7 distill_gen pipeline (extend distill_head) + eval vs baseline NLL.


## Build progress (session 2)
- M1 DONE: explain()/format_explain()/inspect_cell() - exact additive per-channel decomposition; CPU cap through _predict, device cap through _logits_dev (triton tail skipped when cap set); _sparse_lift_channels_dev refactor; knn neighbor provenance; cells report raw keys+totals+followers.
- M2 DONE: bigram_sketch flag -> _c2 EMA of bind(R[prev],R[cur]) (seeded perms _bp1/_bp2); wired _push/_step_dev/bulk _chunk_c2adv/soft_reset(4-tuple 'b')/revert(t exact-inverse, b, bc)/snapshot 'c2'/_rebuild_c2 fallback/to_device; head input concat [c_hat, c2_hat, rows] (in_dim +topic_dim).
- Tests: +11 in test_flux.py (all pass; 95 total incl train_flux). Engine surface test via ForgeEngine.from_flux.
- M3 DONE (gcell, opt-in `gcell=True`): SimHash code of [_gcell_feats] context fingerprint (x*512 int-quantize x int8 _Wk, fp64 accumulate -> bit-identical CPU/GPU codes; verified) -> host dict-of-dict cells {_gc/_gct/_gc_fifo}; delta-rule succ writes scaled by token surprise s/s_h; fixed gcell_w additive lift, gcell_thin regime-0 gate; 'G'/'gE' journal (pre-write values = bit-exact undo), 'gc' snapshot key, teach() writes via _gcell_code_ctx, inspect_cell('gcell',code) + explain + memory_report. Write sites: _learn_token, _bookkeep_fused(gc_code), _ingest_bulk(_gcell_codes_bulk vectorized, _chunk_cprev/_chunk_c2prev), _fused_seq(_cprev_seq/_c2prev_seq pure closed-form; read=post-token row i+1, write=pre-token row i), _fused_step(gc_pre pre-step). _predict_gpu lift gate += config.gcell. 9 tests, all 93 pass.
- FIXED (pre-existing bug): revert_since(seq_pos=) over-ran one bulk block - 'bc' lands count<=floor but next entries are the PREVIOUS block's non-boundary writes; now breaks after 'bc' undo when count<=seq_pos.
- KNOWN pre-existing divergence (documented in _fused_seq docstring): _next_lite host flag can't update mid-row in _fused_seq -> adaptive_read lite/full graph picks differ vs _fused_step -> s_h (and every surprise-scaled write incl gcell) can differ on toggle-prone streams. adaptive_read=False => bit-equal (verified: gc/gct/logits all equal).
- CPU-learn vs GPU-fused gcell: codes bit-equal; score values differ via s vs s_h (host p_mix vs device p_mix) - same class as all surprise-scaled writes; bulk s_v order-only approx documented at learnmixin:1665. logits max-diff CPU/GPU 0.76 identical with gcell off -> not gcell-caused.
- M4 next: gsyn dense delta channel (KDALayer/TitanMemory d=192-384, readout via _A.T reuse + low-rank W_o, thin-gated bypass then full channel).


## R59 research � counterfactual credit for FLUX (2026-09-24)

Problem: reinforce/revert only scores trajectories actually sampled �
no credit to untried actions. Literature fixes that map onto FLUX's
non-gradient memory substrate:

1. LOGGED-BANDIT / CRM (Swaminathan & Joachims 2015, Wang et al 2017):
   log propensities (top-k sampled probs per generated token) in the
   FluxPacket; at episode end, counterfactual value of untried tokens
   via IPS/DR estimation.  FLUX form: on success, runner-up tokens get
   propensity-scaled positive credit too; on failure, sampled token
   gets negative credit AT the divergent position AND alternatives get
   boosted � "untried action gets credit for what the tried one lost".
   Pure cell writes, journaled.  Needs: packet logs top-k probs.

2. DYNA/branch verification (Sutton 1990; DynaWeb 2601.22149 for LLM
   agents): FLUX is its own world model � at the first divergent
   position, spawn stream-pool continuations from top-2/3 alternative
   tokens, verify each branch (verified tasks only), reinforce the
   winning branch's divergent token via span reinforce (R58 exists).
   True counterfactual by simulation.

3. SELF-VERIFIER cells (V-STaR 2402.06457, CoVerRL 2026): train
   "does this answer verify" as a memory lookup � gcell-style cells
   keyed by (ctx, answer) codes storing verified/unverified stats ->
   generalizes verification to tasks with no programmatic checker.
   Pseudo-rewards confidence-discounted + keep explore eps>0 to avoid
   the consensus-trap/reward-hacking collapse documented in
   "Can Large Reasoning Models Self-Train" (2505.21444).

4. HER relabeling (1707.01495): failed attempt = a valid trajectory
   for the goal it DID achieve � ingest under "neg"/achieved tag;
   negative span credit on the wrong-answer token is now expressible
   via reinforce(span, gain<0).

5. Tabular CFR (NIPS07): regret cells keyed by info-set � gcell codes;
   regret-matching policy.  Needs counterfactual action values �
   supplied by (1) propensities or (2) branch rollouts.

Not fixable: Pearl-style causal intervention reasoning (do-calculus) �
that is architectural, not a training trick.


## ForgeEngine improvement audit (2026-09-25, 3-agent sweep)

See conversation for full detail. Key verified findings:
- ENGINE-KV LAYER DEAD: build_kv_cache strategy (s4r/rotorquant/etc) never append/get'd; decode uses per-request past_kv list + torch.cat (layers.py:267). PreAllocatedKVCache + llm.py preallocated_cache param EXIST but engine never passes it.
- DECODE LAUNCH-BOUND: ~800 kernel launches/token (26 mamba layers x ~30 kernels); mamba3.py:432 per-token python scan; engine never invokes compile_decode_step/cudagraph (llm.py:350,402 exist).
- async_d2h.py (pinned+event readback) EXISTS but only wired to self_play, not engine decode.
- BUGS CONFIRMED: BatchedDecoding eos_set hardcoded {7,151643,151645} -> V2 EOS never fires, batched runs to max_tokens (batched_decoding.py:36-38 vs session_manager.py:722). prefix_cache partial hit corrupts mamba state (_slice_past_kv passes dict states at wrong boundary, prefix_cache.py:348-353). MTP/SelfSpec k,v=layer_kv unpack crashes on dict cache (decoding.py:696,879). stop= truncates output but still generates to max_tokens. _decode_tokens re-decodes full output per token for recovery (O(n^2), crash_recovery only writes %32). xgrammar = char-FSM, ignores schema properties + 65k-iter mask loop/step. RadixPrefixCache sentinel+child-overwrite bugs (prefix_cache.py:668,776). AirLLM blocks called w/o past_key_value -> context-free (airllm_streamer.py:237-250). Left-pad batch prefill pollutes mamba state.
- SECURITY: forge_server no auth + CORS * + 0.0.0.0; sandbox.json inside writable dir; tool/exec/lora/router endpoints exposed; usage fields all zero.
- LIFECYCLE RACES: sleep/wake/lora/merge/_swap_weights not under _gen_lock; boot_config.apply_to resets feature flags to defaults.
- OOM-recovery swaps kv_cache the hot loop ignores -> retries at same pressure.
- DEAD-ALLOC: StandardKVCache.init allocates zeros(B,n_kv,max_seq,hd) per layer at activation even though unused.
- CONSOLIDATION: ~20 kv files dead, ~9 scheduler files dead, 4 sparse-attn wrappers identical boilerplate, quant r44-r50 need common.py extraction, IRIFP4Linear duplicated (novel_quant.py:1075 vs iri_fp4_key.py:197).
- R&D WINS: (1) persistent SSM/conv state pool + single CUDA-graph decode step incl sampling = ~800 launches -> 1 (3-8x est). (2) chunked mamba-state snapshots per 256-tok chunk -> true partial-prefix reuse for hybrid SSM (novel, fixes B1). (3) per-row mamba-state continuous batching (no paged attn needed). (4) unified state pool: KV+SSM+conv slots, per-slot quant policy.




## Engine audit � web research raw notes (2026-09-25)

Two subagent reports condensed into docs/ENGINE_IMPROVEMENT_PLAN.md section 8. Key raw refs:
- vLLM mamba cudagraph: PRs 21401/23035/34571/38270/6214; v0.11.0 default FULL_AND_PIECEWISE; layout = [num_slots,dims] pools + state_indices_tensor + NULL slot.
- Vendor kernels (Apache-2.0, pure Triton, Windows-OK): vllm/model_executor/layers/mamba/ops/{causal_conv1d,mamba_ssm,ssu_dispatch}.py
- triton-windows (pypi triton-windows) >=3.3 for SM120; needs MSVC cl.exe; torch>=2.7 cu>=12.8.
- PDL: tl.extra.cuda.gdc_* + launch_pdl, Triton>=3.3, CC>=9.0; 4-10% additive (llama.cpp 22522).
- SSM prefix caching: vLLM Mamba2 APC align mode (PR 30877, block-boundary snapshots, chunk=256), SGLang MambaRadixCache extra_buffer (PR 11214), Marconi MLSys25 (FLOPs/byte admission, 34x hits).
- Spec: ReplaySSM (vLLM 47576/48018 unmerged; ring-buffer inputs + ckpt; rollback=ptr move; 1.87-1.96x); RecoverSSM closed-form (51855); SuffixDecoding arXiv 2411.04975 model-free ~20us drafts; vLLM bug 39273 = normalize state slot every step.
- Serving: vLLM 23569 fully-overlapped (pinned buf + event + pythonize thread); V1 sampler all-GPU; DecodeStream tokenizers>=0.22 ids= param; stop-str holdback max_len-1; abort flag + aclose() + is_disconnected poll + write-fail detect.
- xgrammar real lib: pypi win wheels, ~36us JSON mask, fill_next_token_bitmask CPU-overlapped, rollback free. llguidance ~25us@65k abi3 wheel. outlines-core regex-only.
- Security baseline: llama.cpp --api-key + auto-restrict CORS agent-mode; vLLM Bearer middleware incomplete (/invocations open); cover ALL paths.
- SKIP: mamba-ssm CUDA ext (no win), FlashInfer (no win wheel), Mirage MPK/Megakernels (no sm120, transformer-only), Medusa-mamba (unpublished), SGLang dep (no win).
- SPOT-CHECK: ForgeLM V2 SSM = Mamba-1 or Mamba-2 math? AI21 Jamba-Reasoning-3B is Mamba-1-flavored on HF -> gates kernel layout choice.




## Perf-tech parity raw notes (2026-09-25, third sweep)

Merged into docs/ENGINE_IMPROVEMENT_PLAN.md Phase 8. Condensed refs:
- Marlin: IST-DASLab, fused W4/W8/FP8/FP4 x FP16 GEMM, act-order+zp+MoE, self-contained .cu, Apache-2.0, RUNS on sm120 (vLLM 5090 fallback). ~3.9x bs<=32.
- GemLite: pure Triton, sm120-focused, W{1,2,4,8}xINT8+MXFP4/NVFP4, MIT -> covers b1.58-as-W2A8+NVFP4 in one dep.
- torch._scaled_mm cuBLASLt FP4 native on sm120 (CUDA 12.8+) = zero-dep NVFP4 floor.
- bitnet.cpp W2A8 GEMV packs 4 ternaries/int8 (official BitNet); BitBLAS linux-only.
- SM120 NVFP4 trap: vLLM#47749 silent Marlin W4A16 fallback -> log kernel selection.
- Split-KV decode: SGLang triton decode_attention_fwd_grouped, max_kv_splits=next_pow2(SM/4); ~48 SMs idle at bs=1 MQA.
- Cascade attention (FlashInfer): LSE-merge shared-prefix+suffix, 26-31x shared-prefix loads; merge op ~50 lines.
- SageAttention 2/3: int8-QK+FP8/FP4-PV, sm120 verified, 2-5x FA2.
- FP8 KV on sm120: Triton dequant-on-load only (FlashInfer FP8 JIT broken <cu12.9).
- Sarathi: decode-first + token budget fill (vLLM V1 default); hybrid chunks only when decodes active (26625).
- Sched: recompute-preempt + LCF victim (vLLM 53723, -20-59% recompute); lpm cache-aware admission (SGLang); stream_interval N (57% e2e, 27869); output coalescing (12298).
- Allocator: expandable_segments Windows port PR189163 (check torch ver); cudaMallocAsync flaky WDDM (probe cudaDevAttrMemoryPoolsSupported); GDS linux-only -> pinned bounce (reuse load_safetensors_pipelined); Tutti lesson: contiguous per-layer pages + N-thread reads.
- WDDM tax: launch 5-20us, sync 4-5us (vs ~1us linux) -> strengthens graph-decode case; ReBAR verify.
- ShadowKV: low-rank K GPU + V CPU-RAM + landmarks + outliers (ICML25, 6x batch/3x tput); Quest: per-page min/max K + top-k (MLSys24, 7x attn); DualDecoder: spec-draft predicts KV -> prefetch layer-ahead (2.62x); ScoutAttention: CPU cold-block attn layer-ahead (2.1x).
- FR-Spec: draft lm_head -> freq top-8k (SGLang speculative-token-map).
- Early-exit: SWIFT/CLaSp/AdaDecode training-free; no engine ships; KV bookkeeping on hybrid unsolved.
- SKIP: BitBLAS(linux), Machete(sm90), FA3(sm90), trtllm-gen/DeepGEMM/cubin(sm90/100), AMX(none on 265F), NIXL/UCX/GDS(linux), MoE kernels(dense), MXFP4-Triton slower than Marlin bs=1.


## R60 lit sweep (2026-09-30) — new pubs since ~Sept 19

### Tier 1 — hits in-flight R53/R49 work
- **Gated DeltaNet-2** (arXiv:2605.22791, NVlabs, code on github NVlabs/GatedDeltaNet-2):
  decouples erase gate b_t (key-side) and write gate w_t (value-side), both
  channel-wise; STRICT generalization — collapses to KDA when gates merge,
  GDN when decay also collapses. Chunkwise WY + gate-aware backward in Triton.
  => gsyn channel (R53-M4) should target GDN2 not KDA; `GDN2Key` port could
  init w_t=b_t=beta -> near-lossless from KDA-key checkpoints (directive A).
  1.3B/100B-token result: best of Mamba2/GDN/KDA/Mamba3, esp. multi-key NIAH.
- **FG2-GDN / FG2-GDN+** (arXiv:2604.19021): channel-wise beta_t write gate,
  separate key/value scaling; <1% FLOPs. Alternative parametrization of same
  idea — pick whichever is cleaner to port.
- **CLVR cross-layer value routing** (arXiv:2607.07953): forward delta-rule
  write value into next layer's aligned hidden stream — modest loss gain,
  cheap twist for multi-layer gsyn. Same paper: KDA+Muon lowest 350M val loss
  (validates kda+muon pairing already in tree).
- **In-Place TTT / TTCD** (ICLR'26; arXiv:2608.01672): MLP last-proj as fast
  weights, NTP-aligned objective (not reconstruction); long-window teacher
  supervises short-window student fast weights = dense self-supervision for
  WHAT to retain. Maps onto distill_head/gsyn training.
- **MoME** (arXiv:2609.15126, Sept 14): mixture-of-M memory slots + learned
  hidden-state gate over context vs deterministic n-gram lookup (DSE). This
  is gcell's learned-code idea validated + a routing baseline. DSE (ACL'26,
  acl-long.226): U-shaped MoE/memory budget law; deterministic-hash memory
  offloads to host RAM w/ negligible tput cost — supports flux host tables.
- **GradMem** (arXiv:2603.13875): gradient-based memory write (few GD steps
  on prefix memory, reconstruction loss, meta-learned init) beats forward-
  only writers at equal size. Candidate upgrade for gcell write rule.
- **RADLADS + KL-guided layer select** (arXiv:2512.20569; fla-org/
  hybrid-distillation): attn-weight transfer + hidden align + KL distill;
  GA-S2 selector ranks which layers keep softmax by marginal utility.
  Production version of our LoLCATs note — use if a V13 goes more-linear.
- **GDN = exact special case of RWKV-7 wkv7** (sirus20x6/rwkv-lab): matched-
  dims remap, cosine 0.999995, +0.013% PPL zero-training on Qwen3.5-9B.
  GDN2 probably folds similarly; RWKV-7 kernel could be universal gsyn base.
- **Kimi-Linear** (moonshotai): KDA kernel open-sourced in FLA — the
  chunked-scan follow-up flagged in R49-2 is available upstream.
- **MiniCPM5-2B** (openbmb, Sept 6-7, Apache-2.0): top open sub-4B
  (AA index 15), ships DSpark draft model + training datasets, vLLM day-0.
  New teacher/draft donor candidate; DSpark = pattern already in tree.
- **Ring-mini-linear-2.0** (inclusionAI): 1/32-active hybrid-linear MoE,
  512k via YaRN — another datapoint for linear+dense hybrids.
- **Naive-N0.5-Flash** (Sept 27): 39 SWA + 9 DSA layers, ZERO full attn,
  1M ctx — frontier-scale existence proof for sparse-only stacks.

### KV cache (V2 engine — strategy zoo additions)
- **BeaconKV** (arXiv:2609.04971): cluster queries -> "beacon" reps predict
  which KV gets revisited by future Thought-Revisiting Tokens; 5.8x mem,
  4.3x tput on reasoning traces. Forward-looking importance vs SnapKV's
  recent-window — complements filler_kv/s4r.
- **AnchorKV** (arXiv:2608.02901): anchor-residual, 20x, NO token dropped;
  token = scaled copy of nearest anchor + refined outliers. Stacks w/ s4r.
- **MetaKV** (arXiv:2609.07966): per-prompt pick among compression configs
  via small predictors under latency/mem budget — meta-scheduler over our
  existing strategies (snapkv/filler/s4r/rotorquant).
- **ReFreeKV** (ACL'26 find.1541): threshold-free budget allocation; fixes
  SnapKV-family input-sensitivity.
- **CompressKV** (2606.24467): Semantic Retrieval Head identification +
  layer-wise budget by offline eviction-error estimate.

### SM120 / quant
- **lna-lab/blackwell-geforce-nvfp4-gemm**: 12 documented SM120 patches for
  vLLM+FlashInfer+CUTLASS NVFP4 (grouped GEMM tile cap, QUTLASS matmul,
  sm120a/f gencode, Marlin W4A8...). Enumerates exactly what breaks on 5070.
- **Rodder5/sm120-quant-bench** + arXiv:2601.09527: consumer-Blackwell
  NVFP4/W4A16/MXFP4 bench; silent Marlin W4A16 fallback trap (vLLM#47749)
  — copy their `kernel_selection` logging into our eval scripts so no
  number ships without kernel-path proof. Per-capability probes > PPL.

### Optimizer (muon_sf_plain upgrade candidates)
- **SAMuon / SAMuon-lite** (arXiv:2608.25990): spectral probing shows Muon
  underuses bulk directions; hold head at Muon scale + amplify bulk w/ static
  prior -> 13-24% fewer tokens to same loss @124M-1B; lite = rank-1 power
  iter, ~zero overhead. Strongest candidate.
- **HTMuon** (ACL'26 find.1819, github TDCSZ327/HTmuon): Schatten-q norm,
  heavier-tailed updates/spectra; plug-in on Muon variants; -0.98 PPL C4.
- **MONA** (arXiv:2605.26842): Nesterov-ish accel via EMA of gradient diffs;
  validated 1B-68B MoE pretrain.
- **AMO** (arXiv:2605.17806): per-operator-type Newton-Schulz budget
  (observe-then-commit); uneven ortho quality under uniform schedule is real.
- **NAMO/NAMO-D** (arXiv:2602.17080): Adam-style noise adapt on orthogonalized
  momentum (scalar or diag right-mult).

### Spec decoding
- **EAL/WTV acceptance-aware draft training** (arXiv:2609.24150): optimize
  expected accepted length directly — EAL for greedy, WTV for sampling;
  +GRPO on simulated acceptance. Retrain MTP/DSpark head on this, not CE.
- **Saguaro / Speculative-Speculative-Decoding** (arXiv:2603.03251): predict
  verification outcome, pre-draft during verify; +30% vs best SD, ~5x vs AR.
  Single-GPU friendly (no extra model).
- **SpecBound** (ACL'26 find.802): self-draft early-exit, layer-wise temp
  annealing + adaptive bound; exact equivalence, training-free, 2.33x.
  Sits on existing DoLa/hidden-state plumbing.

### Hybrid CPU/GPU offload (directive D)
- **ATSInfer** (arXiv:2607.10183): TENSOR-granularity offload (not layer);
  static placement + load-aware dynamic transfer + async coord; 1.94x
  prefill / 3.29x decode vs prior systems. hybrid_offload.py is layer-level —
  tensor-level is the upgrade axis.
- **Pipelined sharding** (MLSys'26 client xLM): sub-layer shards, 3 plans
  (static/dynamic/tiered) picked by benchmark-profile + live conditions.

### Self-play / RL (infinite_loop + V-model)
- **ExGRPO** (ICLR'26): experience replay for RLVR — rollout value ≈
  correctness x entropy; mixed-policy objective; +3.5-7.6 pts, stabilizes
  weak models. Maps onto prompt/trajectory selection in infinite_loop.
- **SC-GRPO** (2606.17): token-level credit = KL(base || self-conditioned-
  on-verified) as multiplicative weight — no external PRM needed;
  complements/alternative to R50-3 PRM head.
- **ToolExpander**: swap never-correct prompts for few-shot demos mid-run +
  self-exemplify reward 0.01; for weak-model GRPO stability (our regime).
- **Dynamic Cheatsheet** (EACL'26): self-curated persistent memory doubles
  AIME-style acc for blackbox LMs — concept support for flux teach()/live.
- **S2L-PO** (2605.30789): small-model explorers -> train big policy.
  Interesting inversion: flux could propose trajectories for V2.

### Deprioritized
- dLLM parallel decoding (Fast-dLLM++/RPS/DiCo/Dynamic-dLLM/Learn2PD): no
  diffusion LM in tree. DiffuSpec (DLM as drafter) is the only crossover.
- SPD pipeline-parallel drafting (2605.30852): needs multi-GPU.
- TALH ablation (2609.29618): small single-seed; MLA+SSM parallel hybrid —
  mostly confirms known tradeoffs.


