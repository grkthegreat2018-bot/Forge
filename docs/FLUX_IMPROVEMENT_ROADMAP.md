# FLUX Improvement Roadmap — R54+ Planning

Planning doc for FluxLM (sparse online associative-memory LM) and its RSI
loop. Compiled 2026-09-23 from a full scan of `forge/model/flux/`,
`forge/model/flux_stream.py`, the engine flux paths, and
`forge/self_play/infinite_loop.py` (flux mode), plus the documented
literature this architecture descends from (PAQ context-mixing, Titans/
MIRAS surprise memory, DeltaNet delta rule, infini-gram, knn-LM,
prediction-with-expert-advice / sleeping experts / Fixed-Share).

Mental model: **FLUX is a PAQ-style context mixer turned into an LM.**
Many bounded context models (per-order suffix-hash cells, deferred
episodic match index, knn fingerprint bank, cosine feature channels,
recency/fatigue) → geometric (Hedge/AdaHedge) mixture → logits.
"Training" = counts + EMAs + multiplicative weights. There are no
gradient-trained weights; `FluxConfig` = static hyperparams; `_R` = fixed
feature table (random or teacher-grafted). That separation makes the
memory editable, journaled, revertible — the RSI loop's superpower.

---

## 1. Technique audit — documented vs implemented

### Already implemented (verified in code)

| Documented technique | Where |
|---|---|
| Surprise-gated writes (Titans/MIRAS) | `surprise_gate`, `surprise_floor` |
| Delta-rule associative write (DeltaNet/GDN) | `delta_rule` |
| infini-gram style longest-match backoff | `use_backoff` channel |
| AdaHedge adaptive-lr mixture | `adahedge`, `hedge_*` |
| knn-LM retrieval channel | `use_knn` bank + matvec |
| Match model (PAQ "deep") | `deep` episodic channel + cert-graded archived votes |
| Utility-scored eviction | `_epi_hits` (episodic only) |
| Adaptive/conditional compute | `adaptive_read`, `write_gate`, lite graphs |
| GRPO-style grouped attempts | `flux_group_size` G samples/task |
| RLVR (verified reward) | `_flux_check_any` exact match |
| Rejection sampling + teach-on-fail (RAFT/ReST shape) | `apply_packets` + `teach_text` |
| Self-distillation/consolidation | `consolidate()` epi→ngram promotion |
| Eval gate + rollback | promote/demote + `revert_since` |
| Tool-use + RAG + web-goal curriculum | `flux_goals.py`, `FluxTools` |
| Gap-targeted teacher distillation | `distill_from()` — **built but NOT scheduled in the loop** |

### Documented but missing (highest-value gaps)

| Technique | Source | FLUX analogue |
|---|---|---|
| Self-consistency / majority vote over G samples | standard RLVR eval | first-correct wins now; vote = free robustness |
| Difficulty-band task filter (keep tasks in learnable pass-rate band) | SeRL (2505.20347) | uniform-random task mix now |
| Advantage/rarity-scaled update gain | GRPO-style | `flux_gain=4.0` flat now |
| Spaced repetition / retention re-checks | continual-learning lit | `flux_unique` dedups forever — taught tasks never re-probed |
| SSE/APM secondary estimation (recalibrate mixer output by context) | PAQ2+, PAQ8 | no output calibration stage |
| Context-selected mixers (80 nets chosen by context class) | lpaq/PAQ8 | single global hedge vector |
| ~~Sparse/noncontiguous context orders (skip-grams)~~ | PAQ8 | **done (E7)** — `sparse_taps` gapped-order channels, thin-gated |
| Word-level context keys | PAQ word model | token-level only |
| Sleeping experts (abstain vs unigram-fallback) | Blum & Mansour; PAQ mixers | dead channels vote unigram — dilutes mixture |
| Fixed-Share / tracking regret (non-stationary streams) | Herbster & Warmuth | AdaHedge tracks global, not local, optima |
| ICM/state-table compression of cells | PAQ StateMap | full fp32 counts per follower |
| Replay buffer interleaving (anti-forgetting) | continual-learning lit | `replay_buffer.py` exists, not wired to flux |
| Negative-gain unlearning | `reinforce(gain<0)` exists, unused | — |

---

## 2. Q&A verdicts (session findings)

### Q: Image tokens as context? — YES, easier than for transformers
No positional table, no attention — tokens are ints feeding hash
recurrence + `_R` feature rows. Plan: VQ-encode patches into a reserved
vocab band (+row/col separator codes to preserve 2-D); graft encoder
features into `_R[img_tok]` rows (same pattern as `graft_teacher`,
learnmixin.py:1995) so topic/csem/knn channels see image semantics;
mask image ids out of sampling logits; `teach([img]+Q → A)` gives VQA
free. Effort: encoder + engine pre-tokenization hook, not the model.

### Q: Logic games via self-play? — YES, aim at the weak axis
Current builders (`_concise_qa_pairs`, 21 fns) are all one-hop lookups.
Add `logic_goals()`: mini-ARC rule induction (fresh rule per epoch —
unmemorizable, forces in-context binding), bAbI state tracking,
relational chains, parity/state machines, substitution ciphers. All
computed + verifiable + offline. Keep a separate held-out game eval —
do NOT contaminate the `flux_eval_seed` set. Random entity names defeat
hash keys by design: knn/csem must carry generalization — that IS the
measurement. Trajectory tasks map onto the existing `T:`/`O:`/`A:`
two-phase tool machinery.

### Q: Chess? — wrong shape; small repeated-state games are the right shape
Chess positions never repeat → hash/episodic channels can't recall them;
`_R` features don't encode board semantics; full games = ~40 moves for
one reward bit (worst-case credit assignment). Repeated-state games
(Nim, tic-tac-toe, subtraction games, 21) recur constantly → episodic
memory can genuinely learn position→winning-move and `reinforce`/`revert`
gives clean per-game credit. If chess is wanted: mate-in-1 puzzles via
`python-chess` (FEN→move, µs verification, dense single-step reward).

### Q: Massively speed up / improve RSI? — YES, see Phase 1–2
Eval is serial `_fused_step`s WITH writes on the canonical model (also
leaks eval questions into `"live"` memory). Pool eval + predict-only
prompt feed + lazy snapshot + batched teach ≈ 3–8× epoch wall-clock.
Quality side: majority vote, difficulty band, rarity gain, retention.

### Q: Dynamic params (less for easy, more for hard)? — already the design's
center of gravity; extend it. `adaptive_read`/`write_gate`/`surprise_gate`
exist. Add: tiered read depth (conf≥0.95 order-only → +knn → +deep with
scaled `deep_max_extend`/`deep_max_hits`); task-level budget via mean
`-log p_mix` over prompt feed (free signal in `observe_gpu` rewards);
confidence-gated early-exit decode. CUDA-graph caveat: dynamic under
graphs = mask or extra captured variants (lite/full is the template);
pool/eager paths get true savings.

### Q: Weights vs params separate? — YES, cleaner than a transformer
`FluxConfig` = static hyperparams; all learned content lives in memory
stores; `_R` fixed (random/grafted); zero gradient-trained params.
Config values are read per-call → runtime-tunable, EXCEPT values baked
into CUDA-graph shapes (knn bucket at capture; `MX=96` triton constexpr)
→ dynamic = mask or recapture there.

### Q: Better algorithms? — see Phase 4 + Phase 6 (research shelf)

---

## 3. The Plan

### Phase 0 — Baseline (first, one session)
- [ ] `scripts/bench_flux.py` + `model.memory_report()` on current ckpt:
      tok/s live-CUDA, VRAM, `channel_weights()` (which channels hedge
      actually trusts).
- [ ] Verify `flux_teacher_embed="auto"` actually grafts a fresh model
      (`_graft` set post-init). Biggest single quality lever if off.
- [ ] Extract last-N-epoch `acc_pre`/`acc_post`/`reinforced`/`taught`
      from `status.json`/`events.jsonl` — need the curve before claiming
      improvement.

### Phase 1 — RSI loop speed (low risk, pure engineering)
- [ ] Route `_flux_eval_acc` through `FluxStreamPool` (batched,
      predict-only; kills serial `_fused_step`s AND eval leakage into
      `"live"` memory). `engine_generation.py:517` `_generate_batch_flux`
      already exists — eval just isn't using it.
- [x] `prompt_rewards=False` — engine passthrough added
      (`generate_batch(prompt_rewards=)` → `FluxStreamPool.generate`).
      TESTED (E1): 1.00× on 32-stream GPU, ~95-tok prompts; prefill is
      observe/launch-bound not reward-bound; greedy identical 32/32.
      NO WIN — flag kept as diagnostic, not wired into the loop.
- [ ] Lazy snapshot: `snapshot(BytesIO)` → write only on promote
      (`infinite_loop.py:1130` currently writes pre-gate).
- [ ] Batch `teach()`: stack (position × order) into one
      `add_multi`-style call (currently ~100 `gt.add` launches per
      teach on GPU).
- [ ] Port adaptive-read to `_logits_b` (per-stream confidence mask on
      the same gathered cells → skip deep+knn on confident lanes).
- [ ] Two-phase G: G=2 → retry failures at G=6.
- [ ] Delta snapshots: `.flux` full-state pickle every epoch scales with
      memory (epi 3M + knn 512k + journal 200k). Serialize a base
      snapshot once + per-epoch journal-tail deltas; fold on promote.
      (Verified: `distill_from` exists but is NOT wired into the flux
      loop — all `distill` hits in infinite_loop.py are ForgeLM SFT
      flags. FIXED: `engine.generate(learn=False)` is now a predict-only
      gate for memory models; batch eval already routed predict-only
      through `FluxStreamPool`.)

### Phase 2 — RSI loop quality (documented techniques)
- [ ] Majority-vote selection over G samples (vote among DISTINCT
      decoded responses; verifier still gates correctness).
- [ ] Difficulty-band filter: per-builder-kind pass-rate tracking; drop
      100%-pass kinds, prefer the learnable band, upweight weak kinds.
- [ ] Rarity-scaled gain: `gain = flux_gain * (1 - group_pass_frac)`.
- [ ] Spaced retention: per-epoch quota re-asking previously-taught
      tasks; log retention separately in `events.jsonl` (catches silent
      forgetting after demotes).
- [ ] Probe `reinforce(tag, gain<0)` on confidently-wrong packets —
      gated experiment, measure eval-post delta, shelf if it regresses.
- [ ] Schedule `distill_from()` every M epochs on accumulated
      `gap_positions()` — teacher scores thin contexts (code exists,
      just not wired into the loop).
- [ ] Interleaved replay: small `replay` tag re-ingest of older taught
      pairs each epoch (anti-forgetting; `replay_buffer.py` exists).
- [ ] Uncertainty-scaled teach: `teach(weight=25)` is flat — scale by
      attempt wrongness (answer entropy / margin vs correct answer).
- [ ] Eval decode determinism: confirm eval runs temp≈0 argmax; if the
      eval inherits a nonzero temperature, scores jitter epoch to epoch.
- [ ] Eval-gate statistics: `acc_post >= acc_pre` promotes on +0.001
      noise. Add a paired comparison (same eval items, McNemar-style)
      or a small margin — demotes on noise waste an epoch's learning.

### Phase 3 — Curriculum (logic games)
- [ ] `logic_goals()` family beside `_concise_qa_pairs`: mini-ARC rule
      induction, bAbI state tracking, relational chains, parity/state
      machines, substitution ciphers.
- [ ] Separate held-out `flux_eval_games.json` (never trained).
- [ ] Repeated-state game envs (Nim/tic-tac-toe/21): win-rate vs optimal
      is a clean strategy metric; trajectories feed reinforce/revert.
- [ ] Optional: mate-in-1 chess puzzles via `python-chess`.

### Phase 4 — Architecture (each gated by eval delta + probe script)
- [ ] **Virtual rollout (multi-hop)** — biggest structural gap:
      prediction is one-hop. On thin-p_mix tokens push top hypothesis
      into a scratch context (`h*P + tok+1`, one `_c` EMA — register
      math, zero writes), re-probe, mix the 2-step distribution.
      Confidence-gated = the "more compute for hard" mechanism. ~+1
      probe on flagged tokens only.
- [x] **Sleeping experts**: channels with no evidence ABSTAIN instead of
      voting unigram — implemented behind `sleeping_experts` (0=off,
      1=uni-reward, 2=renormalized pmix) across live/bulk/pool reward
      paths. TESTED (E2): ±0.5% = noise on all slices. NO WIN — flag
      kept, default off.
- [x] **Calibrated rewards + unigram escape** — PROMOTED (E3).
      Root-caused the hedge collapse found in E1 (`fatigue≈0.98`):
      additive-lift channels were scored with raw feature values, not
      their implied distribution `p_uni·exp(f)/Z`. `calib_rewards=True`
      (new default) + `uni_escape=0.05` (arithmetic ε-mix on output —
      bounds worst-case NLL; geometric mixtures have no escape mass).
      Measured: GPU fact 0.206→0.111, bound 1.36→0.98, all 4.64→3.68,
      ingest 2265→9461 tok/s; hedge weights healthy (order channels).
- [ ] **APM/SSE output stage**: recalibration table keyed on
      (quantized p_mix, dominant-channel id, evidence regime) →
      corrected prob. PAQ's reliable ~1%/stage; journaled like other
      writes. Easiest conditioned-learned-component to add.
- [x] **Context-selected hedge sets** (lpaq: mixer chosen by context
      class): per-regime `_w` keyed on channel-evidence bits —
      TESTED (E6): binary thin/warm split (`cond_hedge=2`) fully
      implemented (host `_wR`/`_hmxR`, device `_wgR`/`_hmxRg`, bout
      regime slot, `h2` journal, blk snapshots, pool `swR`/`sreg`),
      oracle probe promised +0.118 nats but MEASURED regression on
      every slice (all 3.83 vs 3.70; fact .25 vs .11 GPU). AdaHedge on
      halved per-regime streams can't reach the post-hoc oracle bound.
      Flag kept OFF; shelve unless corpus grows ≫100×.
- [ ] **Fixed-Share tracking**: small weight-mix toward uniform each
      step → tracks locally-best channel under domain drift within a
      stream (wiki→code switches mid-generation).
- [ ] **Recency-decayed counts**: per-row last-write pos;
      effective `cnt·γ^Δpos` at read (+8 B/row). Automatic forgetting;
      replaces decay-halving sweeps.
- [ ] **knn utility eviction**: count topk hits per row; evict cold rows
      instead of FIFO-oldest (mirrors `_epi_hits`). Easiest item here.
- [ ] **Order-sensitive fingerprint**: `_c` is a bag-of-features EMA —
      "dog bites man" = "man bites dog" to knn. Add a second fingerprint
      with position-bound features (HDC bind or small positional
      codebook) for order-sensitive matches.
- [ ] **Byte-level channel**: token-level keys can't see morphology —
      string_op/letter_idx/reverse tasks are hard BECAUSE tokens hide
      characters. Keep a trailing-byte window per stream; hash last-B-
      bytes → small table voting TOKEN successors. Directly targets the
      weakest procedural family.
- [x] **Sparse/noncontiguous orders** (PAQ8): keys over gapped suffixes
      (t-1,t-3 / t-1,t-2,t-5 …) — new channels immune to single-token
      noise that kills contiguous keys. Just different hash taps.
      **PROMOTED (E7)**: `sparse_taps` default = 6-tap stride-2/3 set,
      `sparse_thin_only` gating (fallback experts only — ungated they
      dilute warm contexts). Host-resident `_ptabs`/`_ptots` on both
      backends; div 5.26→4.21, all 3.67→2.99 CPU & GPU. See scratchpad
      E7 for the design notes (device-hedge abstention, bout thin flag,
      post-escape lift asymmetry, pool-lift gap). E8 extended to an
      8-tap set (stride-4 (0,4),(1,5) added): all →2.92.
- [x] **Character-class orders** (PAQ word-model analog): contiguous
      suffix keys over byte CLASSES (`_byte_class` LUT: digit/upper/
      lower/space/nl/punct/other/high) — coarse keys that generalize on
      novel text where verbatim byte keys never repeat. **PROMOTED
      (E8)**: `class_orders=(3,)`, `class_thin_only`, host-resident
      `_gtabs`/`_gtots` + "K" journal entries (same mixed split as
      sparse). g3 wins on every thin-dominated slice: all 2.92→2.76,
      div 4.10→3.88, nat 3.51→3.40. k≥4 regresses (4096-class keys get
      overconfident-wrong on real prose); `class_min_tot` floor had no
      effect — cells are high-evidence, not under-sampled.
- [x] **Hybrid class+raw taps** (PAQ mixed contexts): `(offset,is_raw)`
      patterns — `_hybrid_key` folds class bytes at flagged positions,
      raw bytes elsewhere. **PROMOTED (E9)**: `hybrid_taps=(((3,0),
      (2,0),(1,0),(0,1)),)` — all 2.676→2.662, div→3.69. Second
      correlated arm dilutes. Also promoted: `thin_tot=2.0` (weak
      1-2-count order cells count as thin → fallback experts still
      fire; all −1%) and `uni_escape` retune 0.05→0.12 (all −2%; the
      optimum shifted once host-table arms existed). Word-order
      channels (`word_orders`, w<j> ring-scan word-hash keys) TESTED,
      NOT promoted — (1,)/(2,) both ~2.78 vs 2.76; plumbing kept
      opt-in.
- [x] **Thin-context retrieval boost** (`thin_boost`): deep/topic/sem/
      csem/knn sit at the hedge floor because their rewards are mediocre
      in the warm contexts that dominate training — yet they are the
      only signal when orders are silent. Fixed +weight in regime-0
      contexts only (same gate as the sparse/class fallback experts).
      **PROMOTED (E10)**: `thin_boost=0.2` — swept .1/.2/.3/.5 → all
      2.662→2.647 CPU, 2.663→2.659 GPU (flat optimum .1–.2, overshoots
      at .5); VRAM unchanged. CPU applies it inline in `_predict`; GPU
      passes a boosted weight copy `wrow_b` into `_logit_tail_dev` and
      the eager channel sites. NOT a win at token scale: on the V=65536
      ForgeLM-V2-tokenizer harness it costs ~+0.04 all — cosine votes
      over 65k rows are too noisy to deserve fixed weight.
- [x] **Teacher embedding graft** (`graft_teacher`, R52-6) measured at
      token level: `research/checkpoints/ForgeLM_V2.safetensors`
      embed.weight [65536,2560] → demeaned randproj `_R` (topic_dim=192).
      Graft lands correctly (R changes; teacher cosine geometry healthy:
      pair-cos mean .11/std .06 raw, ~0 demeaned). Result: all
      3.445→3.435, recall 3/8, p0 hedge weight .955 — near-neutral
      because semantic channels hold ~0 weight (see thin_boost), and
      boosting them hurts at this vocab. NOT promoted — flag stays
      opt-in; the bottleneck is channel quality at V=65k, not features.
- [x] **Autoregressive generation**: `FluxLM.generate(prompt_ids,
      max_new, temperature, top_p, learn, tag)` — decode-step learn
      path (fused step on GPU) + support-restricted sampling:
      `support_thresh=1.5` requires a sampled token to beat
      thresh·p_uni, keeping temperature decoding off the unigram
      escape floor. **PROMOTED (E11)** — measured on a shuffled-
      sentence corpus: greedy and temp-0.8-masked both produce fluent
      *novel orderings* (recombination across sentence boundaries);
      unmasked temp-0.8 derails to char-salad. Same mask added to
      `FluxStreamPool`. Also fixed a latent `_ingest_bulk` crash on
      n < recency_span (negative numpy slice). Self-derived PPMI `_R`
      graft measured — real byte-class structure but no NLL gain
      (cosine-readout mechanism is the bottleneck, not features).
- [x] **Parametric distill head** (`distill_head`, E12): linear
      readout `_Wh [V, topic_dim]` trained by AdamW on (normalized
      context fingerprint → teacher top-k + uniform remainder) pairs;
      read as a fixed-`head_w` additive logit lift in `_predict` /
      `_logits_dev` (eager + triton tail). `scope="gaps"` trains at
      thin positions only; `scope="all"`+`ids=` trains at every corpus
      position (ring horizon would otherwise cap it at ~3.3k).
      Journaled as one fp16 `"WH"` blob (now 3-tuple incl. `_Wh1`);
      `revert_tag` restores; snapshot persists; `to_device` moves;
      CUDA graphs invalidated on write/revert (staleness bug
      found+fixed). Measured at V=65536 vs ForgeLM V2 teacher top-8 —
      full tables in scratchpad E12/E13. **E13 upgrade (promoted):
      order-sensitive head input** `concat(ĉ, R[t-1]/√d…R[t-8]/√d)`
      read live from `_ring`/`_ringg` (`head_orders=8`, zero live
      state, graph-safe) + `head_thin=True` gate — all **3.2403 vs
      3.445 baseline, recall 8/8**, head loss 8.79→7.26. MLP variant
      (`head_hidden>0`: `Wh @ relu(Wh1 @ x)`) implemented but measured
      NOT better — capacity is not the bottleneck: head top-6 on any
      prompt is marginals (`<0x0A>`,`.`,`,`). A fixed featurization
      cannot carry transformer-style compositional context.
      **Domain-matched training is mandatory** (off-domain docs head:
      all 4.0+, rep/fact polluted). Cell distillation destroys rep
      (6.59); Hebbian `_A` improves recall not NLL — both shelved.
      **Generation verdict: retrieval is the mechanism** — with real
      content ingested, order/knn channels produce coherent topical
      continuations, and grafted `_R` gives semantic channels genuine
      hits (`capital of France is` → `▁Berlin` rank-2). Ingesting
      60k real prose alone cut `nat` 10.02→9.25. Head recipe stays
      opt-in (`head_w=0` default): graft + `distill_head(scope=all,
      ids=<served corpus>, epochs=4, lr=3e-3)`, `head_w≈2`,
      `head_thin=True`.
- [ ] **Sub-channel splitting**: split `deep` into deep-ring vs
      deep-archived arms, `knn` into sim-band arms — finer-grained hedge
      credit assignment, ~free (more arms in `_w`).
- [ ] **ICM/state-table cell compression** (PAQ StateMap): quantize
      (count,total) per cell → byte state + shared per-order prob table.
      CPU dict cells ~96 B → ~1 B; enables much bigger `max_cells_per_order`.
- [ ] **Per-domain hedge sets**: journal tags already mark domain
      (wiki/dict/arxiv/procedural/live) — keep `_w` per tag-domain,
      blend on retrieval. "Domain expert" memory.
- [ ] **Hedge re-grounding pass** (found this scan): `_ingest_bulk`
      rewards deep/topic/sem/csem/knn/backoff with the UNIGRAM prior
      (order probs are exact; the rest are approximated away) — so bulk
      training systematically teaches hedge "retrieval channels ≈
      unigram", under-weighting the strongest channels vs live
      learning. Fix: periodic live-reward replay — run a held-out
      slice through the full `_learn_token` reward path each epoch
      (~1-2k tokens) to re-ground non-order channel weights.
- [ ] **Retrieval reconsolidation**: `_epi_hits` counts reads but only
      feeds EVICTION. When a deep/knn vote's successor matches the
      observed token (rewards already compute `vx`), bump that entry's
      mass — use-strengthens-memory (documented reconsolidation
      dynamics). Cheap: the reward pass already has the tensors.
- [x] **Auto episode boundaries**: `eos_token_id` was configured but
      `_ep_start` only moved via manual `soft_reset()`. Implemented as
      `auto_episode` (journaled boundary after EOS; covers _push /
      fused step / fused seq / bulk ingest). TESTED (E4): unseen doc
      transitions −5% NLL but seen transitions +12× — fixed-order
      corpora (train_flux re-ingest) make learned transitions the common
      case. NO WIN → default off, opt-in per corpus.

### Phase 5 — Multimodal (after 1–4 land)
- [ ] VQ token band + `_R` row graft + sampling mask + engine
      pre-tokenization hook; `teach()` = VQA. See §2.

### Phase 6 — Research shelf (think/explore, no commitment)
- Meta-confidence channel: predict own correctness from
  (p_mix, dominant channel, evidence mass) history → feeds RSI task
  selection + eval early-exit.
- Symbolic arithmetic channel: detect `a op b` patterns in context →
  computed vote. Neuro-symbolic cheat for the arith family; debatable
  purity but associative-memory-friendly.
- Decode-time rerank: rerank top-k by next-step p_mix estimate (cheap
  single-probe lookahead at sampling level — virtual rollout's cheap
  cousin).
- Hopfield-softmax variant of knn (softmax over bank vs hard top-k) —
  probably marginal over `knn_gamma` shaping; measure only if curious.
- Cross-channel distillation: deep→knn, topic→csem promotions
  (consolidate() generalized).
- LSH/minhash deep seeding (token near-match) — probe corpus first for
  near-miss suffix frequency before building.
- Episode soft-linking: `_ep_start` hard-gates suffix keys per episode;
  relaxed gating for cross-doc patterns — likely deliberate, verify
  before touching.
- Collaborative sampling: streams share a frozen canonical bank; a
  SHARED scratch knn/epi bank would let sample j exploit sample 1's
  mid-batch discoveries (a stream's own writes currently never feed even
  itself). Risk: mode collapse within a batch — diversity vs
  exploitation knob worth a probe.
- Beam-over-pool decode: keep top-B streams by cumulative p_mix —
  beam search without clones; better eval decode than independent
  samples.
- Per-task config profiles: arith wants high-order exact channels,
  semantic tasks want knn — config is runtime-read, so a difficulty/
  domain signal could pick a profile per attempt (pool/eager only;
  graphed path needs variants).
- Teach-order sweep: `teach()` writes every valid order; for short QA
  contexts long orders are dead keys anyway — sweep which orders get
  weight and the base weight=25 itself.
- Learned write policy: surprise/write-gate thresholds are fixed —
  could adapt per-domain from measured retrieval utility.

---

## 4. Standing measurements (per AGENTS.md §F — numbers or it didn't happen)

- Eval delta on BOTH fixed sets after any change: `flux_eval_n`
  procedural + `flux_eval_web_n` frozen web + (Phase 3) games eval.
- `bench_flux.py` tok/s before/after perf items; `memory_report()`
  total_MB_est + VRAM for capacity items (12 GB budget, §D).
- Channel weights trajectory (`channel_weights()` per epoch) — hedge
  collapse (weights → floor) is a known failure mode; `evidence_lift`
  exists because of it.
- Every Phase-4 item gets a minimal probe script in `.tmp/` FIRST
  (sweep 5–10 hyperparam values), then bit-exact/eval verification.
- New behaviors get `tests/unit/test_flux*.py` coverage (CPU where
  possible) — the confirm-then-fix rule applies to features too.
