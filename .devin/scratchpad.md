# Scratchpad — Hermes-3 8B on ForgeEngine + Local_To_Chub migration

## Objective
Replace LM Studio with ForgeAI (D:\windsurf\ForgeAI) as the Local_To_Chub backend.
Model: mlabonne/Hermes-3-Llama-3.1-8B-lorablated (uncensored ChatML tune).
Already downloaded: research/checkpoints/hermes3-8b-lorablated/ (4x safetensors, ~16GB bf16).

## Hardware budget (RTX 5070, 12GB)
- bf16 weights: 16GB — does NOT fit.
- torchao W4 (TILE_PACKED_TO_4D, g128, tinygemm): attn+ffn = 6.5B params -> ~3.4GB packed.
  embed (0.52B) + head (0.52B) stay bf16 = ~2.1GB. Total ~5.6GB resident.
- KV budget: rotorquant 4bit auto-activation handles; ~4GB headroom at 32k ctx.
- VERIFIED on this box (torch 2.11.0+cu128, torchao 0.17.0):
  quantize_(lin, Int4WeightOnlyConfig(group_size=128,
    int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
    set_inductor_config=False)) works on CUDA. ~9% rel err vs bf16 (typical W4G128),
  forward 0.03ms vs 0.19ms bf16 for 4096x14336 (kernel is fused W4A16, not dequant-cache).
  Weights become Int4TilePackedTo4dTensor subclass on nn.Linear.weight — model.to('cuda') works.

## Tokenizer facts (verified on disk)
- eos_token_id in config.json = 128040 = im_end (ChatML). im_start = 128039.
- bos = 128000 begin_of_text (auto-prepended by tokenizer.json pretokenizer).
- chat_template present (ChatML, 209 chars). Tool markers are tool_call tags —
  same markers forge's qwen_adapter uses for render/parse. Zero template work needed.
- Hermes-3 tokenizer = Llama-3.1 vocab 128256 + ChatML special tokens.

## Integration plan (ForgeAI)
1. config.py: root preset llama31_8b (vocab 128256, d 4096, L32, H32/KV8,
   inter 14336, eps 1e-5, rope_base 5e5, rope_scaling llama3, max_seq 32768).
2. RotaryEmbedding: add _llama3_inv_freq for type/rope_type llama3
   (HF Llama3 scaling: low_wl=orig/low_f, high_wl=orig/high_f, smooth ramp).
   Mirror in loader._reset_non_persistent_buffers.
3. inference_quant.py: quantize_model_w4(model, group_size=128) -> torchao
   quantize_ on nn.Linear targets (q/k/v/out + w_gate/w_up/w_down), skip head.
4. engine_checkpoints.from_checkpoint: new kwarg load_quantize in
   {None,"int4","fp8","w8a8","auto"}. When set: build on CPU (cfg.device=cpu),
   quantize in place, model.to(device). Skips _load_with_fallback entirely.
   Set engine._load_quantize so _auto_activate_optimal does NOT re-quantize.
5. engine_activation._auto_activate_optimal: if self._load_quantize set ->
   overrides["quantize"]=None (+ skip block_fusion/cuda-graph? measure first —
   leave enabled, only guard quantize).
6. engine_common._DEFAULT_EOS_TOKEN_IDS += {128001,128009,128039,128040}
   (llama3 eot/end_of_text + Hermes im_start/im_end).
7. forge_server.py: --load-quantize CLI flag -> register(**engine_kwargs).

## Local_To_Chub plan
- chub_lmstudio_proxy.py -> chub_forge_proxy.py (same logic, renamed terms,
  forge_base_url default http://localhost:8000/v1, auth 'Bearer forge').
- config.yaml: forge_base_url, force_model hermes3-8b-lorablated, context_backend
  comments point at forge /v1/embeddings.
- llm_client.py: default base_url + auth to forge.
- launch.py, start.bat, start_forge_server.bat, README.md, download_model.py.

## Dead ends encountered (recorded per directive C)
- torch._convert_weight_to_int4pack + _weight_int4pack_mm direct calls: operator
  exists but packed layout contract is undocumented; every hand-packed shape
  combo rejected. Solved by torchao Int4TilePackedTo4dTensor which wraps it.
- torchao PLAIN/PLAIN_INT32 packing: PLAIN needs mslk>=1.0 (not installed),
  PLAIN_INT32 is CPU-only. TILE_PACKED_TO_4D is the only CUDA path in 0.17.0.
- ForgeQuant/QuantizedLinear: both cache dequantized bf16 at first forward —
  peak VRAM ~= full model. NOT viable for 8B-on-12GB. int8 no_cache exists but
  is slow per-token. FP8Linear is real but 8B fp8 = ~8.5GB, tight with KV.
- Native int4pack gives ~5.6GB resident -> chosen path.

## Bug session: "VRAM spikes when prompting, response corrupted" (fixed)

Root causes (all confirmed in code; fixes applied + tested):

1. CORRUPTION — SDPA is_causal top-left on continuation forwards.
   layers.py:351 called flash_attention(is_causal=True) whenever T>1, even
   with a non-empty past_kv. On CUDA, PyTorch SDPA anchors the causal mask
   TOP-LEFT for non-square (T < S) shapes -> a T>1 continuation forward
   (prefix-cache suffix, chunked prefill chunk>=2, spec verify) masked out
   the ENTIRE cached prefix. Verified empirically on this box (T=3,S=7).
   Symptom: every 2nd+ chat request sharing the >=32-token prefix key
   (Local_To_Chub sends the same card/system prompt every turn) generated
   corrupted text. Fix: layers.py + fused_qk_norm_rope_cache.py now build
   an explicit bottom-right _causal_mask when total_len > T.

2. VRAM SPIKE — full [1,T,128256] logits on every prefill.
   StandardDecoding (decoding.py:233, the default non-streaming path) and
   the prefix-cache suffix forward (prefix_cache.py:297) ran without
   logits_last_only -> ~512MB bf16 transient at 4k tokens. Fix:
   logits_last_only=True on all prefill-only forwards (decoding.py x7,
   prefix_cache, cacheblend, prefill/__init__).

3. CORRUPTION — _finish_to_stop re-fed the last emitted token.
   past_kv already contains it -> duplicate KV entry -> corrupted tail.
   Fired whenever max_new_tokens hit without EOS (finish_sentence=True
   default). Fix: decode loops now stash _forge_last_logits alongside
   _forge_last_kv; _finish_to_stop uses them (zero extra forwards),
   falls back to _prefill(output_ids) when absent.

4. CORRUPTION (hybrid models) — cache_prompt_prefix stored
   _forge_last_kv = end-of-generation state under a prompt-only key;
   dict recurrent entries can't be sliced -> ghost context on cache hits.
   Fix: _forge_prefill_kv captured at prompt boundary (StandardDecoding,
   _prefill, prefix-cache hit path); cache_prompt_prefix prefers it.

5. STALE captures — strategies that never set _forge_last_kv left the
   previous request's KV/logits visible. _generate_impl now clears
   _forge_last_kv/_forge_last_logits/_forge_prefill_kv per request.

Known remaining issues (confirmed by audit, NOT in user's path — logged):
- spec strategies (NGram/ExternalDraft/MTP/SelfSpecSparse) never roll back
  KV on reject -> corrupt past_kv (decoding.py:338/446, uno.py:101 tuple-only
  unpack also crashes on hybrid dicts). ExternalDraft is dead code (never
  instantiated). UnoDecoding passes past_key_value (singular) -> TypeError.
- Triton conv patch (triton_conv.py:194-218) ignores _conv_state_prefix on
  T>1 -> corrupts continuations on hybrid models when use_triton_conv.
- beacon/anchor/metakv compaction + RoPE offset drift: past_len recomputed
  from compacted cache len -> wrong phases (kv strategy != rotorquant only).
- triton_conv/dspark/eagle/uno etc. dormant on the llama31_8b path.
- per-token torch.cat KV append: O(T) transient each decode step (real KV
  is unbudgeted; rotorquant strategy object is never wired into the model).

Regression tests: tests/unit/test_forge_engine_fixes.py —
  TestContinuationForwardCausalMask (cpu+cuda param),
  TestPrefillLogitsLastOnly, TestFinishToStopNoDoubleFeed,
  TestStaleDecodeCapturesCleared, TestCachePromptPrefixBoundary.
Real-model verify: scripts/test_hermes_w4_generate.py.

## 2026-10-08 follow-up — Chub path VRAM growth + empty output (VERIFIED)

Reproduced end-to-end (server + proxy + in-process probes):

- OUTPUT: proxy routed marker-free Kobold transcripts through
  kobold_to_chat -> single ChatML user turn wrapping a User:/Char:
  transcript. Hermes argmaxes token 128001 (end_of_text, logit 21.0) at
  decode step 0 -> content:null at temp 0, word salad at Chub's temp=2.
  NOT context overflow (failing prompt ~524 tokens; limit 32768).
  LM Studio sent Kobold prompts raw. FIX: handle_kobold always routes
  /v1/completions (raw continuation) + forwards Chub stop_sequence.

- VRAM GROWTH: three pinning sites, all confirmed live
  (7.7->9.2 GB over ~10 reqs before fix; flat +-15MB after):
  1. _slice_past_kv views -> each 32-tok prefix entry pinned the whole
     prompt+gen KV (~1 GB/entry x 256) -> now clones.
  2. semantic_anchors.save_anchor stored captured_kv (3/req x 16,
     write-only) -> now kv_state=None.
  3. _forge_last_kv/_forge_last_logits/_forge_prefill_kv stayed set
     between requests (~1 GB idle pin) -> cleared in _generate_impl after
     _cache_prompt_prefix + generate_stream finally.

Remaining latent: session_cache.update_session_kv stores per-session KV
(64 sessions) — dead path, same pinning class if sessions ever activate.
rotorquant strategy object still bookkeeping-only (never wired into
forward); real presents cache unbudgeted — kv-tokens arg does NOT bound
live KV growth, only the dormant strategy's capacity.

======================================================================
## 2026-10-13 — RESEARCH: selective / non-backprop LLM training systems

Task: survey novel training systems; goal = fast trainer that selectively
updates only relevant weight blocks (cut time+compute). Experiment model:
Qwen2.5-0.5B on RTX 5070 12GB. Three parallel research threads distilled.

### Existing capability (DO NOT duplicate — rule E)
- forge/training/optim/badam.py — block-coordinate Adam, ROUND-ROBIN only
  (desc/asc), BREAD momentum correction, CPU offload of inactive states.
- GaLore, Fira (fira_nlrq), APOLLO, Muon variants, fp8_training,
  hybrid_offload, r20/r21 optimizer R&D — all present.
- NO token-level loss selectivity (sft_train only does -100 masking).
- NO importance-driven block scheduling — THE open niche.

### Landscape verdicts (3 subagent reports, spot-check ids before citing)
- Pure backprop-free does NOT scale: FF/PC/EP/DTP/Hebbian best results are
  CIFAR-scale; iPC reached ~110M masked LM at ≈BP compute; no competitive
  autoregressive LM exists. Don't chase a pure-BP-free LM trainer.
- Proven block-wise precedents at text scale:
  - DiffusionBlocks (ICLR'26): block-local denoising of AR transformer,
    ~85M on LM1B/OpenWebText, 4x memory cut, blocks trained 3-at-a-time.
  - BAdam (NeurIPS'24): 8B full-param FT on one RTX3090.
- Block-selective (direct BAdam successors — round-robin is beatable):
  - MISA (NeurIPS'25, 2511.00056): module-granularity importance sampling,
    ~3% params/step, beats BAdam/LISA/DoRA; provable variance reduction.
  - AdaGradSelect (2512.15764): top-k blocks by cumulative grad-norm +
    Dirichlet + eps-greedy decaying to exploit; near-full-FT @ sub-3B.
  - LCSB (2602.13073): DETACH non-selected layers (residual identity path),
    momentum = implicit updates to skipped blocks; 1.40x real speedup,
    <2% quality loss @3B — only block method cutting backward FLOPs.
  - LISA (2403.17919), HiFT (2401.15207, 89% fewer params/step),
    ChunkFT (2605.21177, sub-tensor blocks, 7B in 13.7GB),
    BlockLLM (2406.17296), BCD+Landscape-Expansion (NeurIPS'25).
- Token-level (second multiplicative axis):
  - RHO-1/SLM (2404.07965): excess-loss vs reference model — 3% of tokens
    matched full-data math training; +6.8% avg on 80B-token CPT.
  - TokenTune (EMNLP'24): backward through token subset only.
  - WARNING (2312.05021): loss/grad-matching subset selection failed to
    beat random baseline — selection must be excess-loss/influence, not
    raw-loss. Use self-reference ckpt (earlier step of same model = free).
  - MATES halves FLOPs to target perf via small influence model.
- Subspace (third axis — composes with existing stack):
  - GUM (2510.17802): unbiased GaLore×Muon via LISA-style layerwise
    sampling for debiasing — literally designed for our composition.
  - Breaking-the-Frozen-Subspace (2502.05790): importance-sample WHICH
    subspace, not always dominant SVD.
  - Q-GaLore (2407.08296): 7B pretrain on 16GB card; LDAdam generalized
    error feedback; GaLore2 (2504.20437) validated Llama-7B x 500B tok.
- ZO/forward-only (utility-probe role, not main trainer):
  - MeZO-BCD (2501.19099): block-coordinate ZO, 2.77x vs MeZO on OPT-13B;
    block selection is literally "choose block, 2 fwd passes, update".
  - Sparse-MeZO (2402.15751): ZO only on small-|w| mask; HiZOO-L
    (2402.15173): diagonal Hessian per block = ready-made relevance score.
  - LOZO (2410.07698) low-rank ZO; SubZero per-layer low-rank perturbation.
  - Only viable for low-dim updates (single block/adapter), not full-param.
- ES (outer-loop candidate, hot 2025-26 area):
  - N=2 ES works at 0.5B-7B without advantage normalization (2607.19408).
  - DART-ES beats GRPO at < GRPO+LoRA memory doing FULL-param (2610.06993).
  - "Functional sparsity" (2608.27351): ES gains concentrated in a sparse
    subset of parameter updates — empirical evidence that selective
    updating isn't a compromise, it's where the signal already lives.
- Nested Learning/HOPE (NeurIPS'25, Behrouz): model = nested optimizers
  at different timescales; formalizes LiveT/TitanMemory. Controller picks
  "slow" blocks for rare expensive updates; fast local writes always-on.
- Gradient Routing (2411.17470): mask machinery for data->subnetwork
  control; the plumbing for data-conditioned block gating.
- Distillation note: GKD/OPCD still need backprop on student; ICLCA
  (2406.02847) converts ICL->bias closed-form, no training.

### Recommended synthesis — "bandit-scheduled selective trainer"
Unclaimed as one system; three multiplicative axes + optional ZO probe:

  1. TOKENS: excess-loss gate (reference = earlier ckpt of SAME model,
     near-free) -> TokenTune-style backward on high-signal tokens only.
     RHO-1 evidence says quality often IMPROVES (denoising effect).
  2. BLOCKS: replace BAdam round-robin with utility-sampled selection —
     score_i = EMA(grad_norm_i) x staleness_i x EMA(dLoss credited to i).
     eps-greedy -> exploit. Detach non-selected block inputs (LCSB trick)
     => frozen blocks cost ZERO backward FLOPs; Adam momentum gives
     implicit updates to skipped blocks. Novelty vs MISA/AdaGradSelect:
     realized-dLoss credit ledger (bandit payoff), not just grad-norm.
  3. SUBSPACE: on active block only, existing GaLore/GUM/Muon projection
     (trivial composition — already in codebase).
  4. OPTIONAL: MeZO-BCD micro-probes (2 fwd passes) to estimate utility of
     ambiguous blocks WITHOUT paying a backward pass — utility estimation
     at inference memory.

  Budget math @0.5B: backward ~2/3 of step FLOPs. Select k-of-24 blocks
  + 30-50% of tokens -> ~3-5x step-time reduction plausible. Full AdamW
  already fits 12GB at 0.5B => the metric is QUALITY-PER-HOUR vs full
  AdamW SFT, NOT memory feasibility. Baselines to beat: full AdamW SFT
  (not LoRA strawman) and BAdam round-robin (rule F: right baseline).

### Unverified citations (subagent-flagged; check arXiv before citing)
2602.13073 LCSB, 2605.21177 ChunkFT, 2607.19408 N=2-ES, 2608.27351
functional-sparsity, 2610.06993 DART-ES, 2512.15764 AdaGradSelect,
GDN-2 2605.22791/2604.21100, Celo2 (report-only), QES (blog-only).

----------------------------------------------------------------------
### EXPERIMENT RESULTS — Qwen2.5-0.5B on RTX 5070 (2026-10-13)
Model: .devin/qwen25-05b (local_dir dl, HF_HOME symlinks broken on this
box — always use local_dir). Scripts: .devin/probe_selective_train.py
(timing/correctness), .devin/selective_trainer.py (arm sweep; arms via
argv). Results: .devin/selective_results.json. Corpus: papers.jsonl
arXiv abstracts (discriminative — test_corpus was saturated @0.013 NLL).
Method: frozen blocks run fwd under no_grad on detached input, residual
re-added outside => h + (f(h.detach())); verified grads land only on
live blocks and identity-path reaches embed_tokens.

**Phase-0 timing (B8xT512, AdamW step included):**
  FULL 281.7ms/9.87GB | BLOCK-12 213/7.54 | BLOCK-6 179/6.37 (1.57x)
  BLOCK-3 162/5.79 (1.74x) | TOK-50% 0.87x LOSES | TOK-25% 0.90x LOSES
  COMBO-6+TOK50 223/6.69 (1.26x) | COMBO-3+TOK25 195/5.17 (1.45x)

**Phase-1 arms (B4xT512, lr 5e-5, 250 steps, held-out eval NLL):**
  FULL      2.6234  in 83s   (quality king)
  FIXED-6   2.6603  in 59s   <- WINNER: top-6 + embed/head always live
  ANCHOR-4+2 2.6772 in 60s   (top-4 anchors + 2 rotating explorers)
  CYCLE-6   2.9299  in 60s   (round-robin: nearly DOESN'T LEARN)
  SAFEUTIL-6 2.9251 in 59s   (grad-norm picks + lr x med_disp/disp_i:
                              stable now, still worse than FIXED-6)
  UTIL-6    4.7648  in 60s   (grad-norm bandit: DIVERGED then partial)
  ACTDELTA-6 5.0293 in 60s   (disp-based picks: DIVERGED)

**Falsified theories (recorded per directive C):**
1. Token-level selection via dedicated scoring pass LOSES wall-clock
   (0.87x) — scoring fwd costs more than 152k-vocab head slicing saves.
   Only viable amortized/proxy-free. Block axis is where FLOPs live.
2. Round-robin block cycling (BAdam-style, hop=6/step) nearly doesn't
   learn — sparse rotating updates can't compose at 5e-5/0.5B. BAdam
   paper used ~100 steps/block dwell; rapid cycling = poison.
3. "Activity" signals select FRAGILE not VALUABLE blocks: both
   ema(grad-norm) and ema(||f(h)-h||/||h||) converge on early blocks
   (disp: layer0~12, layer2~20-71) -> early-layer updates at 5e-5 ->
   distribution cascade -> eval spikes to ~8. Failure = LR x fragility,
   proven by SAFEUTIL (same picks + inverse-disp LR => stable 2.93).
4. Quality-per-hour winner is a STRUCTURAL prior not a learned signal:
   FIXED-6 (top-k + io always) ~ matches FULL at 71% cost. The
   "intelligent" part = knowing WHICH blocks to touch (task-dependent
   prior: readout-adjacent layers for SFT), not online utility.

**Useful artifacts produced:**
- wrap_frozen/unwrap/set_live detach-wrapper = working LCSB machinery
  for HF decoder layers (transformers 4.57, returns-tensor-safe).
- disp instrumentation (per-block ||f(h)-h||/||h|| EMA) for free in fwd.
- VRAM dividend: 6/24 live => 6.4GB => ~2x batch headroom untested.

**Next testable theories:**
- Adaptive-k anchored scheme (grow/shrink k on eval-stall).
- Payoff-credited explorers inside ANCHOR (realized dLoss attribution).
- VRAM->batch dividend: same budget, B=8-12 on selective arms.
- ZO micro-probes (MeZO-BCD) for explorer slots — no backward at all.
- Pretraining regime check: does FIXED-k still win from-scratch?

## 2026-10-13 R68 - FluxLM live generalizer ("gen head") design

User ask: FluxLM recall weights update live; the generalization head should
too.  Ontology: info -> weights -> gen head -> params.  Gen head owns NO
params/weights - a procedure that routes mass between stores.

Existing: consolidate_gen() = BATCH recall->gen (ring replay, order cells'
top-m dist -> gcell/gsyn/gsd/dnet/_Rl).  gen_recall = floor on the realized
token's gen write only (no distribution).  Gap: no LIVE distribution
transfer - self-play ingest leaves gen channels under-trained until a manual
consolidate_gen.

Design (R68, gen_live):
- New knobs: gen_live (mass, 0=off), gen_live_topm (4), gen_live_min_tot
  (-1 -> thin_tot).
- Shared transfer fn _genlive_write(votes, conf, code, k, cvec):
  top-m followers share-weighted; gcell/rl per-follower writes at
  s_i*w_t; dense (gsyn/gsd/dnet) get blended target v=sum(w*Rm[t]) at
  s_i = gen_live*conf.  conf = max_k tot/(tot+backoff_kappa).
- Eager _learn_token: votes from POST-write order cells (all orders with
  tot > min_tot); thin contexts auto-gated (post-write tot ~ s < min_tot).
- _ingest_bulk: votes from PRE-write probes (rt_f/rc_f/tots_f/found_f,
  already gathered for rewards); vectorized top-m via argpartition; dense
  via _gsyn_chunk/_gsd_chunk/_dnet_chunk v= path; rl via _rl_bulk on the
  follower multiset; gcell host loop (bounded by topm).
- _bookkeep_fused (per-token GPU): re-probe keys -> probe_fixed ->
  ttoks/tcnt gather -> host votes -> _genlive_write (post-write cells).
- teach() untouched (already direct gen binds).
- self_play: add cfg flux_gen_consolidate -> consolidate_gen(all_ring)
  after consolidate (batch complement; live path covers ingest).
- Journal: same entries as consolidate_gen (G/GS/gs/rB/rL/gd/dn) ->
  revert_tag exact.
- Verify: NLL on novel-similar contexts, revert parity, thin gating.
  Probe script .tmp/exp_gen_live.py.

## 2026-10-13 R69 � WHY compute params need bits (knowledge/compute split study)

Question: if knowledge lives in external stores, why do high-bit weights
still beat BitNet ternary when params only COMPUTE?  Probes:
`.tmp/exp_why_bits.py` (mechanisms + FluxLM in-situ),
`.tmp/exp_qwen_bits.py` (Qwen2.5-0.5B, .devin/qwen25-05b,
test_corpus_large.txt, 6x1024-token seqs, teacher-forced NLL).

### Measured numbers
Isolated (d=128 learned operator, argmax vs V-row codebook):
- rel-err / top1-flip%: fp16 0.000/0.0 | int8 0.006/1.2 | int4 0.117/24.5 |
  ternary 0.439/74.5 (V=65k; flips grow with V � margins shrink).
- Random +-1 projection preserves pairwise cos as well as fp32 (JL;
  MAE .0716 vs .0695) � untrained projections never needed bits.
- Depth-8 residual chain: fp16/int8 cos~1.0, ternary 0.67 (errors compound).
- STE-QAT on a rigid low-rank task: 100% flip � an ISOLATED operator
  cannot re-design itself ternary-native; paper parity needs whole-net slack.

Qwen2.5-0.5B (baseline NLL 0.218, top1-top2 logit margin 8.18):
- int8 all-linear:  +0.000 (free)        margin 8.23
- int4 all-linear:  +0.225               margin 4.05
- ternary all-lin:  +13.76 (worse than uniform 11.9 � confidently wrong)
                                       margin 0.44
- ternary by role: mlp +10.37 | attn_vo +8.57 | attn_qk +1.95 |
  head/embed +0.38 (tied)              margin tracks NLL
- FluxLM order+gcell counts -> same ternary op: +0.076 trained-NLL
  (~180x less damage than linear weights, per corrupted value).

### Conclusions
1. The premise inverts: knowledge is discrete (counts, ids � values read
   through votes/argmax, top-heavy cells survive quantization); compute
   is continuous (the matrix IS the function � per-entry error applies
   to every input, every forward, systematically).
2. Sensitivity ordering is anti-storage: mlp >> vo > qk >> head.  The
   most storage-like tensor (lm_head codebook, per-token rows read by
   argmax) tolerates ternary best (+0.38).  Operators need bits because
   magnitude ratios within a row ARE the program.
3. Mechanism: operator error aggregates over K-dim sums -> direction
   error; downstream is SELECTION (softmax/argmax) with sub-nat margins
   that shrink with vocab; margin collapse (8.18 -> 0.44) IS the failure.
4. BitNet parity = whole-network QAT re-design + param redundancy
   (~1/sqrt(N) noise averaging) + high-bit activations/row-scales.
   Post-hoc ternary of a converged operator is a different (broken)
   program, not a degraded copy.
5. Design rules for ForgeAI:
   - keep random +-1 projections (_R/_Wq/_Dk/_Dv) at 1 bit � fine already;
   - knowledge stores (order/gcell counts) tolerate extreme value-quant;
   - learned operators: int8 free, int4-per-row is the floor, never
     post-hoc ternary; if ternary wanted, design ternary-native
     end-to-end (slack + QAT), not transcoded;
   - allocate bits by margin-sensitivity (readout/selection side first).

### Caveats / dead ends
- FluxLM hedge mixture throttles weak dense channels � on the easy
  corpus the gen channels contribute ~0.003-0.07 logit lifts vs order
  cells' 2-5, so corrupting _S/_Rl/_DS/_Wd shows +0.000 delta.  In-situ
  compute test needs a corpus where gen channels carry the answer.
- A3 superposition readout: ternary degrades gracefully (SNR 1.14->1.09)
  � pseudoinverse rows are sign-dominated; not all compute is equally
  fragile.

## R69b � iso-byte quality frontier (quality per weight-GB)

Script `.tmp/exp_iso_bytes.py` � bpc = total_nats/(chars*ln2), own
tokenizer per model, 6x1024-token eval, RTX 5070.

qwen-0.5B (25,386 chars):                    hermes-3B (25,522 chars):
  bf16          1.26GB  0.0762                bf16          7.21GB  0.0655
  int8 lin      0.92GB  0.0762                int8 lin      4.50GB  0.0654
  int4+int8mlp  0.89GB  0.0905                int4 lin      3.09GB  0.0797
  int4 lin      0.74GB  0.1549                int4+ternhead 2.40GB  0.1367
  ternary lin   0.63GB  4.8807                int4+int8mlp  4.15GB  0.0662
  int4+ternhead 0.50GB  0.4759                ternary lin   2.25GB  3.3986

Frontier verdicts:
1. int8 on operators = free 2x reduction (0.0762 both). ALWAYS take it.
2. Asymmetric > uniform below int8: int4-lin+int8-mlp (0.89GB/0.0905)
   near-matches bf16; uniform int4 (0.74GB/0.1549) much worse.  Spend
   the saved bytes on the mlp (arithmetic) group.
3. Params-vs-bits at iso-byte: PRECISION wins at this scale �
   qwen-0.5B@int8 (0.92GB) matches hermes-3B@int4 (3.09GB): 0.0762 vs
   0.0797 bpc.  A 6x larger model at 4-bit only ties the small model
   at 8-bit while costing 3.4x memory.
4. 6x params do NOT rescue post-hoc ternary (3B ternary = 3.40 bpc ~
   uniform-random).  Ternary is dead post-hoc at any tested scale;
   it only works QAT-from-scratch.
5. Quant damage compounds multiplicatively: tern-head alone on bf16
   +0.38 NLL, stacked on int4-lin +0.92 marginal.  Never stack two
   "acceptable" losses.
6. Net allocation rule: knowledge -> discrete external stores (ternary
   counts ~free, +0.076 NLL measured); compute -> int8 (free) or
   asymmetric int4/int8-mlp at aggressive budgets; storage-role tensors
   (embed/head codebook) are the most quant-tolerant but stack badly.
