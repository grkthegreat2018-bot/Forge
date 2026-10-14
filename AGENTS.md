# ForgeAI — Agent Notes

## Agent Operating Directives (READ FIRST)

These directives govern how work is done in ForgeAI. They are non-negotiable
unless the user explicitly overrides them for a specific task.

### A. Model Versioning — Build On The Prior, Never Beside It
- **Every new custom model version MUST be derived from the immediately
  preceding version**, carrying forward all prior keys/architecture as the
  baseline, then adding or replacing only what's new. Current chain:
  `forgelm_v2` (Jamba base) → `forgelm_v12_jamba` (R37 keys) →
  `forgelm_v13_gdn2` (R60: Gated DeltaNet-2 side-paths, KDA-superset).
- **Port-first, train-second**: when introducing a new architecture key or
  attention variant, write the lossless checkpoint-conversion path
  (`XxxKey` class, identity/zero-init warm start, bit-exact load test)
  **before** any training experiment. This eliminates the "train first, port
  later" tax that has bitten past rounds.
- **No silent regressions**: a new preset that drops a prior key must
  document WHY in the preset line and in the R&D round notes. Dropping keys
  silently is a bug.
- **Preset lineage check**: before merging a new preset, run a bit-exact
  forward-pass comparison against the prior preset's checkpoint on the BSP
  base. Max logit diff must be 0.0 (lossless) unless the preset is
  intentionally non-lossless (document the delta).

### B. Confirm-Then-Fix — Never Leave A Known Bug Sitting
- **When you find an issue, confirm it** (reproduce with a minimal script or
  test) **then fix it in the same session**. Do not log it and move on.
- If a fix would be large/risky, scope a minimal failing test first, then
  implement the smallest correct fix. Prefer targeted edits over rewrites.
- **Always add or update a test** for the fix so it cannot silently return.
  Tests live in `tests/unit/` and run on CPU where possible.
- If a fix is genuinely blocked (needs user input, env change, or a
  destructive op), say so explicitly and create a `todo` — do not pretend
  it's done.

### C. R&D Is The Default Mode — Push For Novel Improvements
- **No area is "solved"**. Every existing technique (attention, KV cache,
  quantization, decoding, optimizer, loss, scheduler) is fair game for a
  novel variation. The codebase already has 13 R&D rounds; round 14+ is
  expected, not exceptional.
- **Prefer novel over copy**: when implementing a known technique, always
  ask "what's the novel twist that could beat the paper's number on our
  specific hardware (RTX 5070, 12GB, SM120, Blackwell)?" Implement the
  baseline AND at least one novel variation in the same round.
- **Cross-domain combinations are the highest-value R&D** — see the
  expanded Novel Discovery Protocol in `docs/CHANGELOG.md`.
- **When stuck on a hard optimization, pivot don't quit**: if 2 iterations
  fail to beat the known best, shelf it in `.devin/scratchpad.md` with the
  failed approaches documented, then touch up a *different* area. Fresh
  context often surfaces the missing idea. Return to the hard problem later.
- **Record failures as carefully as successes** — a documented dead end
  saves the next session hours. Use `.devin/scratchpad.md`.

### D. GPU/VRAM — Mixed Approaches Are Mandatory To Consider
- **Never propose a GPU-only or CPU-only solution when a mixed approach is
  viable.** The hardware is RTX 5070 12GB VRAM + 32GB system RAM + pinned
  CPU offload (`hybrid_offload.py`, `cpu_kv_offload.py`). The optimal
  operating point is almost always a split.
- **Always state the VRAM budget** for any new inference or training
  feature. If a feature pushes past 12GB on the 1.2B model, it MUST offer
  a mixed CPU/GPU fallback path (e.g. CPUAdamW for training, CPU KV
  offload for inference, BitNet int8 for weights).
- **Quantization is a first-class citizen**, not a fallback. BitNet b1.58,
  W8A8, NVFP4, OffQ, AAAC, SharQ, MosaicQuant are all production paths on
  this hardware — prefer them over "just use a smaller model".
- **Profile before assuming**: use `torch.cuda.memory_allocated()` /
  `torch.cuda.max_memory_allocated()` in test scripts. Guessing VRAM is
  how we get OOM at 3am.

### E. No Redundant Files — Search Before You Create
- **Before creating ANY new script or module, grep the codebase for an
  existing one that does the same thing.** The codebase has ~600 .py files
  after the aggressive refactor (was 908); the odds are high that a related
  implementation exists.
- **Prefer upgrading an existing file over spawning a new one.** If
  `forge/inference/kv/snapkv.py` exists and you want "smarter SnapKV",
  edit that file — do not create `snapkv_v2.py` or `smart_snapkv.py`.
  Versioned filenames fragment the codebase and hide the canonical path.
- **If two files end up doing the same thing, merge them** and delete the
  inferior one. Document the merge in the "Removed (consolidation)" section.
- **The canonical path wins**: when in doubt, the file already wired into
  `forge_engine.py` / `forge_server.py` / `sft_train.py` is canonical. New
  code hooks into those, not around them.

### F. Math Thinking + Script Testing — Find The True Optimum
- **Every optimization claim must be backed by a number from a script**,
  not a paper citation. Papers report numbers on different hardware/models;
  our numbers come from RTX 5070 + ForgeLM V2 (Jamba-3B).
- **Write the smallest possible test script first** (see Novel Discovery
  Protocol step 1 in `docs/CHANGELOG.md`). A 20-line script that runs in 5 seconds > a 200-line
  design doc.
- **Do the math by hand for small cases** before trusting a benchmark.
  If a KV compression scheme claims "15×", verify: `bytes_full /
  bytes_compressed` with the actual dims (d_model=2048, head_dim=64,
  8 KV heads, 16 layers). Catches off-by-one and per-layer-vs-total
  confusion.
- **Sweep parameters, don't guess them**: when a technique has
  hyperparameters (rank, block size, threshold, alpha), write a sweep
  script that tries 5-10 values and plots/logs the result. The optimum is
  rarely the paper's default.
- **Compare against the RIGHT baseline**: a new KV cache must beat the
  current production cache (`s4r` for V4), not a strawman. A new optimizer
  must beat `muon_sf_plain` (V3) or `cpu_offload` (V4), not raw AdamW.
- **No area is truly solved**: even "obviously optimal" choices (RoPE
  theta, head_dim, SwiGLU intermediate ratio) deserve a periodic
  re-challenge. If the improvement is hard to find, shelf it and touch up
  another area — fresh ideas surface indirectly.

### G. General Best Practices For This Codebase
- **Run the test suite before declaring done**: `pytest tests/ --tb=short -q`
  with the venv python and `PYTHONPATH=D:\windsurf\ForgeAI`.
- **Keep AGENTS.md current**: when you add a new file, feature, or R&D
  round, update the relevant section here in the SAME session. Stale
  AGENTS.md causes the next agent to duplicate your work.
- **Use `.devin/scratchpad.md` for long working notes**, not the main
  conversation. Keeps context clean and persists across sessions.
- **Debate the user's premise when warranted** — if a requested approach
  has a known-better alternative on this hardware, say so and propose the
  alternative. Don't silently implement a worse path.
- **Subagent delegation**: use `subagent_explore` for read-only research
  (file indexing, paper lookups) and `subagent_general` for parallel
  implementation tasks. Spawn 2-3 in parallel for independent work.
- **Skills**: invoke user-level skills (`log-bug`, `sync-memory`,
  `systemspecs`, `glm-supercharge`) — they live in
  `~/.codeium/windsurf/skills/`, not `.devin/skills/` (which does not
  exist). They encode project-specific workflows.
- **IDE crash trigger — batch QA/training-data edits**: The IDE crashes when
  a single edit contains 25+ lines matching Question/Answer or similar
  training-data-like patterns. **Code edits do NOT need batching** — only
  QA pairs, fact lists, training samples, and other data-like content.
  When writing scripts with embedded fact sets or training data, split the
  file write into multiple smaller edits (≤20 QA lines per edit) or write
  the data to a separate `.json`/`.jsonl` file and load it at runtime.

### H. Commit Hygiene — One Logical Change Per Commit
- **One logical change per commit.** If the commit message needs "etc." or a "+"
  to list multiple features, split the commit.
- **Stop amending pushed commits.** Rewriting history breaks bisect and
  collaboration.
- **Commit message format**: imperative mood, focused on WHY not WHAT.
  Example: "Fix int8 dtype mismatch in FastINT8Linear dequantize path"
  NOT: "Add V12 architecture + ForgeQuant + tool-use SFT + GUI fixes"
- **Bisectability**: every commit must build and pass tests. If a commit
  breaks tests, fix it or revert before pushing.

## Current Canonical Layout

Production code: `forge/` (was `research/`, migrated commit f1a7542)
- `forge/config.py` — ModelConfig dataclass + presets
- `forge/model_loader.py` — facade re-exporting `forge/model/` (kv_cache,
  attention_ops, layers, builders, llm/ConfigurableResearchLLM, loader/ModelLoader)
- `forge/checkpoint_io.py` — save/load_checkpoint (safetensors + .pt) and
  `load_safetensors_pipelined()` — the fast CUDA weight loader: N parallel
  file reads into per-thread pinned buffers + per-tensor async H2D on a
  side stream (0.53s vs 3.3s per-tensor safetensors on the 6.4GB V2 ckpt;
  fastsafetensors is broken on this Windows box — missing cudart DLL — so
  pipelined is the de-facto fast path; both have safetensors fallback).
  Wired into `ModelLoader._load_safetensors_mmap/_load_sharded_safetensors`
  and `load_checkpoint`. `from_checkpoint` also overlaps the tokenizer
  load on a daemon thread. Bench: `scripts/bench_weight_io.py` (I/O only),
  `scripts/bench_load_time.py` (full engine load phases — V2: ~2.4s total).
- `forge/engine/forge_engine.py` — ForgeEngine (inference engine; core init +
  8 mixins: engine_checkpoints, engine_activation, engine_generation,
  engine_diagnostics, engine_merging, engine_lora, engine_lifecycle,
  engine_sessions; shared helpers in engine_common.py).
  R60-4: `engine_generation._maybe_compact_past_kv` is the live bridge
  between the registered `kv_cache` strategy (beacon/anchor/metakv) and
  the per-request `past_kv` — it compacts only `(k, v)` tuple entries
  (dict SSM/conv state passes through), throttled to `compact_over`
  watermark crossings + every 32 steps + once forced after prefill.
  IMPORTANT invariant: any T>1 forward mid-sequence (chunked prefill,
  spec verify) must see the committed conv/SSM left context —
  `DoubleGatedConvLayer`/`MambaLayer`/`Mamba3Block` now read live
  state/`past["conv_state"]` as the continuation prefix; do NOT add
  T>1 decode-time paths that bypass this.
- `forge/aux_models.py` — generic non-LLM inference runner
  (`load_aux_model(manifest, ckpt)`): safetensors + manifest-declared
  `mlp`/`module` archs, VRAM-aware CUDA/CPU pick, `layer{i}` key remap.
- `forge/aux_train.py` — train-side sibling: `train_mlp(X, Y, ...)` on
  numpy arrays (AdamW + cosine, input-norm folded into layer0 so exported
  nets take raw features), `export_aux_checkpoint` writes `layer{i}.*`
  safetensors via `save_checkpoint`. Used by Oposition's `neural/train.py`.
- `forge/model/live.py` — LiveT live-training session wrapper
  (`LiveConfig`, `LiveSession`, `livet_config`): journaled live learning
  over ConfigurableResearchLLM. Live surfaces = GDN2 recurrent states
  (per-sequence working memory, auto-update under use_cache) +
  TitanMemory fast weights (persistent cross-sequence store, inner-loop
  surprise-gated updates). Substrate (embed/attn/FFN/head/projections)
  stays frozen — params are never written by ingest; only journaled
  buffers move. `mark()`/`revert()`/`snapshot()`/`load()` give exact
  undo; `eval_nll(use_live_state=)` measures without mutating.
  Design doc: `docs/LIVET_DESIGN.md`; tests `tests/unit/test_livet.py`.
  TitanMemory v2 (titan_memory_key.py): learned k/v/q/o projections +
  buffer fast weights + momentum/forget inner loop; `update()` target
  defaults to successor-token embedding (context→next-token store);
  `LiveConfig.solver` selects gd/cg/solve (ridge-regularized dense
  solve) inner loop. `session.attach_sparse(NGramMemory)` adds an
  exact-suffix sidecar (forge/model/ngram_mem.py): per-order suffix-hash
  successor counts -> gated logit lifts, journaled/reverted with the
  live surface — measured +2 NLL verbatim recall over fuzzy keys alone;
  hybrid (solve+ngram, all blocks) is the winning configuration.
  `forge/model/ngram_mem.py` — NGramMemory exact-suffix sidecar
  (the live-surface contract: update/lift/snapshot/restore/reset/gate);
  R64 fixes: `.to(device)` on lift logits (CUDA inputs), +0.01 uniform
  floor (novel successor in known suffix used to give -inf lift → inf
  surprise → NaN fast weights; real-text-only bug).
- `forge/model/flux/` — FluxLM (R52): sparse online associative-memory
  LM, NOT a transformer.  Package layout: `config.py` (FluxConfig),
  `helpers.py` (`_mix64`/`_s64`/`_scatter_sum`/teacher loader),
  `journal.py` (FluxJournal), `tables.py` (_GPUCellTables),
  `kernels.py` (triton, HAS_TRITON-gated), `model.py` (FluxLM =
  nn.Module + mixins), `statemixin.py` (ring/epi/knn/csem helpers),
  `predictmixin.py` (all predict paths + forward), `learnmixin.py`
  (ingest/teach/fused steps), `revertmixin.py` (undo/audit),
  `iomixin.py` (snapshot/load/report).  `forge/model/flux.py` was
  deleted — `from forge.model.flux import FluxLM` resolves to the
  package facade.  Weights = addressable memory cells (per-order
  n-gram tables keyed by rolling suffix hashes, Hebbian topic matrix,
  episodic index) written live per token; context = order-agnostic
  sketch → unbounded, no positional training.  Prediction = Hedge-weighted
  geometric mixture over channels: `deep` = archived episodic retrieval —
  entries are `(ctx_end, succ, cert_hashes)` tuples deferred-inserted so
  positions past the ring horizon still vote via 24/96-gram suffix-hash
  certificates (R52-2); `knn` = approximate-match bank of normalized
  context fingerprints → successor votes via top-k cosine (R52-1,
  generalizes where `deep` needs verbatim repeats); `sem` = expected-
  next-token fingerprint cosine spread (generativity).  `teach(ctx,
  target)` = instant hot-training — key-addressed multi-order writes +
  csem + knn row, journaled under a tag (R52-3).  Every write is
  journaled (`FluxJournal`) for audit (`audit()`, `writes_since()`) and
  revert (`revert_since`, `revert_tag`, `snapshot`/`load`).  Engine path:
  `ForgeEngine.from_flux(config=FluxConfig(...), checkpoint=..., device=)`
  — bypasses checkpoint detect/quant/KV; `device="cuda"` runs fused
  CUDA-graph steps with GPU-resident tables (`cuda_primary`), knn bank,
  and host-side journal blocks (R52-4 — bulk ingest journal doesn't
  grow VRAM); `unbounded_context=True` skips tokenizer truncation;
  session learning tagged `"live"`.  Train: `scripts/train_flux.py`
  (ingest = training, instant); bench: `scripts/bench_flux.py`.
  R52-5: Triton kernels (deep-votes, logit-tail; `use_triton` flag + HAS_TRITON eager fallback), knn scan sliced to bucketed live prefix (graph recapture on bucket cross), ~692 tok/s live CUDA @V=4096.  R52-6: `graft_teacher` (V2 embed.weight → de-meaned randproj `_R`), `reinforce(tag,gain)` outcome replay, `consolidate()` episodic→ngram self-distillation, utility eviction via `_epi_hits`; distill_flux loads teacher via ForgeEngine.  R52-7: `.flux` snapshots load via ForgeEngine/GUI (models_index + engine_rt → `from_flux`); sleep/wake preserve live memory via temp snapshot; CPU→GPU sentinel-cell OOB fixed.  R52-9: `--training-mode flux` RSI in InfiniteSelfPlayLoop —
  `FluxLM.clone()` (BytesIO round-trip, zero shared state), flux-aware
  `generate_batch` (cloned-worker threads, per-prompt temp/seed, workers
  run `_use_cuda_graph=False` — concurrent captures invalidate);
  cross-epoch prompt dedup + eval-set exclusion; GUI launchable via
  SelfPlay page (mode=flux + .flux checkpoint picker), live events in
  status.json/events.jsonl.  R52-8: `train_flux.py --topic ai,code|science|custom-kw` high-precision
  domain filter (title accept / intro ≥2 distinct hits) for --hf streams,
  jsonl records, and --dict-flatten kaikki entries; corpus-scale flags
  --hf/--wiki-filter/--max-tokens/--ckpt-every; GUI live-learn via
  EngineService.learn + POST /engine/learn (sliced ingest under the
  generation lease).  R52-9: `--training-mode flux` in
  `forge/self_play/infinite_loop.py` — RSI on _concise_qa_pairs:
  tagged attempts, immediate revert on fail, reinforce on success,
  teach_text(Q:/A:) on fail, consolidate, fixed-set eval gate
  (flux_eval_seed), demote = revert_since(epoch_mark).  Tests:
  `tests/unit/test_flux.py` (30), `tests/unit/test_train_flux.py` (12),
  `tests/unit/test_grpo_rsi_mode.py` TestFluxModeBranch.  R52-10:
  monolith split into `forge/model/flux/` package (mixin layout mirrors
  the engine); `forward()` GPU path now batches via `_fused_seq` — T
  device steps + ONE `_bout`/logits D2H readback, deferred host
  journal/mirror replay (epi eviction takes effect at row granularity);
  `learn_prefill` config flag (False = bulk forwards observe-only,
  decode still learns); `_scatter_sum` deterministic vote accumulation
  (knn/deep/order channels + reward path + triton kernel) —
  `index_add_`/triton `atomic_add` float ordering made logits
  timing-dependent; NOTE `_logits_dev` order-channel contrib also
  needed it (tokens repeat across order cells); epi deque maxlen drops
  now journaled ("eE") in the fused path; `_NullJournal.entries`
  instance attr; `load()` → `_restore_state`/`_rebuild_gpu_epi`;
  `memory_report` unified; `.flux` load documents its pickle trust
  boundary (dict-of-dict cells can't go through safetensors).  Engine
  side: `SamplingParams` collapses ~18 duplicated generation args;
  `generate_stream(stop=)` with cross-chunk hold-back (recovery
  checkpointing decodes generated_ids per step — mock decoders in
  tests must disable `eng._recovery`); logprobs captured during decode
  (no re-run — FLUX no longer double-ingests); capability checks
  (`supports_kv_cache`, `unbounded_context`) replace FluxLM
  name-sniffing; `PYTORCH_CUDA_ALLOC_CONF` moved to
  `forge.runtime.configure` and gated off Windows (expandable_segments
  unsupported → per-alloc warnings); CrashRecoveryManager weakref'd
  (atexit/signal roots pinned engine); flux `sleep(1)` uses
  `to_device` (nn.Module.to() was a no-op).  New regression tests:
  `_scatter_sum` order-invariance, `_fused_seq` vs `_fused_step`
  bit-equality, two-model CUDA forward determinism, learn_prefill
  observe-only, _NullJournal isolation, crash-recovery weakref +
  close(), flux sleep to_device, stream stop across chunks, logprobs
  no-double-ingest.  R52-11: finished the atomic-order sweep — every
  remaining float `index_add_` with duplicate-capable keys on the
  write/read paths now routes through `_group_add`/`_group_totals`
  (`helpers.py` — sort keys + cumsum + boundary diff, then ONE
  index_put_ per unique key; O(k log k), no O(k²) matmul, used for
  8k-token bulk chunks).  Converted: bulk `_A`/`_Ac` deltas, `_unig`
  (reuses the host np.unique counts), csem `asum`/`ssum`/`tocc`,
  backoff-channel logits (`_scatter_sum` like the other channels),
  `tables.add`/`add_multi`/`subtract` float writes, and the revert
  mirrors.  int64 counts (`_epi_cnt`, `bincount`) and [O]-unique/
  single-index adds were already order-immune.  NOTE `_group_add` has
  data-dependent shapes — eager paths only, never inside the captured
  `_fused_step` graph.  New test: `test_group_add_deterministic`
  (allclose vs index_add_, bitwise repeatability, exact negation
  round-trip).  `test_cuda_forward_deterministic` was flaky in-suite
  before this (12-cell `_Ac` drift ~3.7e-9); 3 clean runs after.
  R52-12: `write_gate` (thin-context `tot8<2|s>=0.1` epi + `s>=0.3`
  knn write suppression; CPU −80% knn rows, GPU −10%; teach/consolidate
  bypass; GPU gated knn writes steer to a `pos=-1` discard row so the
  captured graph stays branchless; CPU `_csem_cpu` capped at
  `csem_slots` with journaled `s`-op evictions; `adaptive_read`
  two-graph fused steps — `(lite,knn_bucket)` graph keys, the flag
  computed from the phase-2 order gather and riding `_bout`;
  `_finish_to_stop` uses `model._predict()` (past_kv=None re-prefill
  double-ingestion fixed); bounded `_recent` deque + revert rebuild.
  R52-13: `compile_dev` — `torch.compile` of `_logits_dev`/`_probes_at`
  INSIDE CUDA-graph captures (predict 630→150us, fused step
  1038→~800us, bench predict 2.5K→11.4K tok/s; logits 9.5e-07-exact).
  Dynamo-disabled islands keep bit-parity: `_topk_dev`/`_knn_votes_dev`
  (inductor topk picks different tie members → 4.2 logit diff on
  identical fingerprints), `_deep_votes_dev`/`_logit_tail_dev` (custom
  triton).  knn votes moved out of `_logits_dev` into `kv`/`kvt`
  tensor args (python-int args recompile during capture →
  cudaErrorStreamCaptureInvalidated); `kv`/`kvt` always tensors;
  `_warm` is per-(lite,bucket) for the same reason;
  `assert_indirect_indexing=False`; TORCHINDUCTOR_/TRITON_CACHE_DIR
  short paths (Windows MAX_PATH); flux sources ASCII-only (cp1252
  inductor codecache bug); `_scatter_sum` uses scatter_ stores.
  torch.compile(_step_dev) whole crashes triton PassManager — dead
  end; live steps stay ~1ms launch-bound (CPU still faster at V=512;
  GPU wins bulk ingest ~8x + predict ~40x).  Tests:
  `test_cuda_compiled_logits_parity`,
  `test_cuda_to_device_drops_graph_and_compile_state`.
  R52-14: whole-step `torch.compile(_step_dev)` now WORKS — the
  PassManager crash was two separate bugs: (a) the hedge block's fp64
  persistent-reduction kernel (kstar argmax + pmix sum + exp/clamp)
  dies in TritonGPURemoveLayoutConversions on sm_120 → extracted to
  `@torch._dynamo.disable _hedge_dev`; (b) inductor lowers
  `index_copy_`/`index_add_` on big flat views to whole-tensor
  read-modify-write (27.9µs compiled vs ~2µs scatter on a 2M int64
  tensor → ~1.1ms monster kernels on the 16M-cell order tables) → all
  big-tensor writes moved to disabled helpers `_w_order_wipe_dev`/
  `_w_order_cells_dev`/`_w_epi_dev`/`_w_knn_dev`/`_w_topic_dev`/
  `_w_csem_dev` (verbatim blocks, real scatters inside).  Inline knn
  `topk` → `_knn_reward_dev` (same tie hazard).  Fused step
  1044→492µs (~354 kernels vs 625; lite graph 398µs, full 580µs);
  CPU still ~1.8× faster on live steps (~285µs learn+predict) — the
  residual is ~300×~1µs launches; next lever = hand triton write-pack
  kernels or thin-helper producer relocation.  Compiled path is
  deterministic (twins bit-identical) but NOT eager-bitwise: fp16
  knn fingerprints can flip topk ties (≤3.2 logit swings, equal-
  quality alternatives); bout int fields exact vs eager.  Tests:
  `test_cuda_compiled_step_journal_parity`,
  `test_cuda_compiled_step_deterministic`.
  R52-15: head-format tuning shipped new defaults —
  `orders=(3,4,6,8,12,16,24,32,48,64)`, `knn_k=8`, `deep_order=4`,
  `epi_row=32` (sweep `.tmp/flux_head_sweep.py`, 2 corpus families +
  GPU).  Findings: order-1/2 are noise channels (drop alone: NLL
  0.465→0.272, div-eval better too); orders 48/64 add real in-domain
  reach; k=8 captures all of knn's contribution; deep_order=4 seeds
  shorter matches (-7% div); epi_row=32 unstarves GPU deep coverage
  (-19% GPU NLL for +0.23GB); `gpu_row_width` stays 16 (32 = no gain);
  cert_orders flat below ring horizon (untested at scale).  Net vs old
  defaults: -67..-69% CPU NLL, -9% diverse-eval, -30% GPU NLL, -22%
  GPU div, recall intact.  MAX_PATH bug fixed in `FluxLM.__init__`:
  cache-dir env values >24 chars (e.g. `.devin/torch_cache/triton` set
  by `forge/runtime/configure.py`) are overridden to drive-root
  `D:\ti`/`D:\tt` — a ~148-char fused kernel name only fits ≤~125-char
  bases... inverse: base≤24 fits ~170-char names.  Pinned by
  `test_tuned_head_format_defaults`; two tests re-pointed off
  hardcoded order-1/2 semantics.  R54 (measured-experiment round,
  harness `.tmp/flux_harness.py`, log `.devin/scratchpad.md`):
  `calib_rewards=True` — additive-lift channels (recency/fatigue/sem/
  csem) now score hedge rewards as their implied distribution
  `p_uni·exp(f)/Z` instead of raw feature values; fixes the measured
  hedge collapse (`fatigue≈0.98`) — weights spread over order channels.
  `uni_escape=0.05` — arithmetic ε·unigram mix on emitted logits
  bounds worst-case NLL (geometric mixtures let a confident-wrong
  channel veto the true token).  `sleeping_experts` (0/1/2 abstention
  modes) and engine `generate(learn=False)` predict-only gate +
  `generate_batch(prompt_rewards=)` passthrough added but NOT enabled
  (E1/E2 measured no-win; flags are diagnostics).  Numbers: GPU fact
  NLL 0.206→0.111, bound 1.36→0.98, all 4.64→3.68, ingest
  2265→9461 tok/s; CPU fact 0.77→0.25.  Regression tests:
  `test_calib_rewards_no_hedge_collapse`, `test_uni_escape_floor`;
  `test_reinforce_and_revert` teach weight recalibrated 0.4→0.05.
  R54 cont. — `sparse_taps` PROMOTED (E7): PAQ8-style gapped-order
  channels `p<j>` (stride-2/3 6-tap set default) — hash chains over
  (t, t-2, t-4…) survive single-token corruption that breaks every
  contiguous key; `sparse_thin_only` gates them to fallback experts
  (ungated dilutes warm contexts — fact .24→.35; gated keeps the win).
  Host-resident `_ptabs`/`_ptots` on BOTH backends: GPU bookkeep
  mirrors surprise-scaled writes ("C" journal, exact revert), bulk
  does vectorized ring-gather writes, predict thin flag rides bout
  (`_boff_thin`) and the host scatter-adds the lift into `_blogits`/
  `blog_seq`/`_plogits` post-step.  Sparse arms abstain from device
  hedge (neutral pmix via `_pmask` closed form — uni fallback would
  decay them to hedge_min and kill the lift).  Measured: div
  5.26→4.21, all 3.67→2.99 (CPU) / 3.66→2.99 (GPU, ~exact parity,
  zero VRAM).  cond_hedge (E6) and knn_utility_evict (E5) measured,
  NOT promoted — flags stay off (see scratchpad).  R54 cont. — E8:
  `sparse_taps` extended to 8 taps (stride-4 (0,4),(1,5)) → all 2.92,
  and PROMOTED `class_orders=(3,)`: g<k> channels = contiguous suffix
  keys over byte CLASSES (`_byte_class` LUT, PAQ word-model analog) —
  coarse keys generalize on novel text.  Same host-resident split
  (`_gtabs`/`_gtots`, "K" journal entries, thin-gate, abstention via
  `_pmask`/`_pidx`).  g3 alone is optimal (k≥4 keys are overconfident-
  wrong on real prose; class_min_tot floor no-op).  all 2.92→2.76,
  div→3.88, nat→3.40, GPU parity 2.764 @ 89.3MB.  R54 cont. — E9:
  `thin_tot=2.0` (order cells below 2 count as thin → fallback experts
  fire; all −1%), `uni_escape` retuned 0.05→0.12 on the new mixture
  (all −2%), `hybrid_taps=(((3,0),(2,0),(1,0),(0,1)),)` PROMOTED
  (h<j> = (offset,is_raw) class+raw patterns; all 2.676→2.662; extra
  correlated arms dilute).  `word_orders` (w<j> ring-scan word-hash
  keys) measured NOT promoted — plumbing kept opt-in.  Benchmark
  all: 3.67→2.66 cumulative.  R54 cont. — E10: `thin_boost=0.2`
  PROMOTED — fixed +w for deep/topic/sem/csem/knn in regime-0 (thin)
  contexts, bypassing the hedge floor where they are the only signal
  (all 2.662→2.647 CPU / 2.663→2.659 GPU; swept .1–.5).  GPU carries
  it via `wrow_b = wrow + tb*_tb_mask` into `_logit_tail_dev` + eager
  sites.  Teacher graft measured at token level (V=65536 harness,
  ForgeLM_V2 embed→randproj `_R`): graft alone 3.445→3.435, with boost
  3.471 — NOT promoted; semantic votes are too noisy at 65k vocab for
  a fixed boost, and channels hold ~0 hedge weight regardless.  E11:
  `FluxLM.generate()` + `support_thresh=1.5` — support-restricted
  sampling (token must beat thresh·p_uni) keeps temperature decoding
  off the unigram floor; greedy+masked sampling produce fluent NOVEL
  orderings on diverse corpora (the model composes at boundaries -
  earlier "retrieval only" was a fixed-order-corpus artifact).  Also
  fixed `_ingest_bulk` crash on short sequences.  PPMI self-`_R` graft
  measured, no gain — cosine-readout is the bottleneck, not features.
  E12: `distill_head` — parametric teacher channel: linear `_Wh
  [V,topic_dim]` trained by AdamW on (normalized context fingerprint
  -> teacher top-k) pairs, read as fixed-`head_w` additive logit lift
  in `_predict`/`_logits_dev`.  scope='all'+ids= trains at every
  corpus position (ring horizon would cap at ~3.3k).  Journaled 'WH'
  blob (fp16) + snapshot + to_device; CUDA graphs invalidated on
  write/revert (staleness bug found+fixed).  Token-level V=65536 vs
  ForgeLM_V2: graft+head(scope=all,ep4,w2.0) all 3.4046 vs 3.445
  baseline, recall 4/8, rep/fact preserved — the ONLY distill path
  that improves all+recall together (cells: rep 6.59 destroyed;
  hebbian: NLL worse).  Head alone on random _R memorizes (loss 9.84)
  vs grafted _R generalizes (8.79) — they compose.  CEILING: does not
  move greedy generation — topic-EMA fingerprint is bag-of-features,
  no word order; chat/agentic transfer needs an order-sensitive head
  input (bigram-bound sketch beside _c).  Opt-in: head_w=0 default.
  E13: order-sensitive head input landed — `concat(ĉ, R[t-1..t-8]/√d)`
  read live from `_ring`/`_ringg` (`head_orders=8`, graph-safe, zero
  new live state) + `head_thin=True` gate (regime-0 only — ungated
  head polluted rep/fact on off-domain data).  Harness: all
  3.445→3.2403, recall 8/8, head loss 8.79→7.26.  `head_hidden>0` =
  trained MLP channel (`Wh@relu(Wh1@x)`, Wh1 journaled in the 'WH'
  3-tuple + 'Wh1' snapshot key) — measured NOT better (head top-6 is
  marginals on every prompt; capacity is not the bottleneck, the
  fixed featurization is).  Retrieval is the generation mechanism:
  docs-60k ingested → coherent topical continuations; grafted _R
  gives semantic hits via cosine channels ('capital of France is'
  →'▁Berlin'); nat 10.02→9.25 from ingest alone.  Recipe: graft +
  distill_head(scope=all, ids=<served corpus>, ep4, lr3e-3), w≈2,
  thin-gated.  R54 cont. — readability + order-sketch: `explain(top_k)`
  returns per-channel logit attribution per top token + fired-cell
  provenance (raw keys, totals, followers, knn neighbours) — the
  mixture is additive so the decomposition is exact; `inspect_cell(
  'o8'|..'h0', key)` dumps one cell; `format_explain` renders.  CPU
  `_predict(cap=)` and device `_logits_dev(cap=)` record each
  channel's delta (cap forces the eager tail; triton tail skipped);
  host-table lift refactored to `_sparse_lift_channels_dev` (per-
  channel dicts; `_apply_sparse_lift_dev` merges — hot path numerics
  unchanged).  `bigram_sketch=True` maintains `_c2` — EMA over bound
  pairs `R[prev][bp1]*R[cur][bp2]` (seeded perms `_bp1/_bp2`), the
  order-sensitive head input E13 flagged: head in_dim += topic_dim,
  `_push`/`_step_dev`/bulk `_chunk_c2adv`/`_bind_pair`/`_bigram_vector`
  + `_rebuild_c2` (pre-sketch snapshots), 'b' journal entries now
  4-tuples (fp16 sketch), 't'/'bc' undo exact, 'c2' snapshot key.
  R55 — `gcell` channel (opt-in, default OFF): learned-key cell
  family — same dict-of-dict substrate as p/g/w/h host tables but
  the key is a SimHash code of the order-sensitive context
  fingerprint (`_gcell_feats` = [ĉ, ĉ2, R tail rows] like the head
  input; `x*512` int-quantize @ int8 `_Wk` projection, fp64
  accumulate = exact integer math → bit-identical CPU/GPU codes).
  Successor scores by delta-rule `s += lr·s_tok·(1-s)` scaled by the
  token's surprise; fixed-weight `gcell_w` additive lift, thin-gated
  (`gcell_thin`, regime-0 only) so recall owns warm contexts.
  Host-resident both backends (zero VRAM), journaled 'G'
  (pre-write values — bit-exact undo), 'gE' evictions,
  `gcell_max_cells` fifo cap, snapshot key 'gc', `inspect_cell(
  'gcell', code)` + explain()/memory_report wired.  Write sites:
  `_learn_token`, `_bookkeep_fused` (gc_code param — caller scores
  the PRE-x context since device _c is post-step), `_ingest_bulk`
  (`_gcell_codes_bulk` vectorized per-position codes from
  `_chunk_cprev`/`_chunk_c2prev`), `teach()` (`_gcell_code_ctx`,
  weight→repeated delta steps), `_fused_seq` (`_cprev_seq`/
  `_c2prev_seq` pure closed-form context rows — read code for
  emitted logits uses post-token row i+1, write code uses pre-token
  row i).  `_predict_gpu` lift gate now includes `config.gcell`.
  KNOWN (pre-existing, now documented in _fused_seq): `_next_lite`
  is host-side → seq mode can't flip lite/full graphs mid-row;
  surprise-scaled writes can diverge from _fused_step when
  adaptive_read toggles (pass adaptive_read=False for bit-equality).
  ALSO FIXED: `revert_since(seq_pos=)` over-ran one bulk block —
  after a 'bc' restore lands count<=floor the loop kept popping the
  PREVIOUS block's non-boundary writes (epi/knn/gcell); now breaks
  on the 'bc' boundary landing.
  R56 — `gsyn` channel (opt-in, default OFF): dense delta-rule
  generator, the parametric complement to gcell's sparse cells.
  `_S [topic_dim, gsyn_dim]` updated `S ← γS + β·s·(v_err ⊗ k)` per
  learned token where k = `_Wq`-projected `_gcell_feats`, v_err =
  `_R[x] − S@k` (delta rule = per-token SGD — the channel keeps
  self-modifying at inference like `_A`).  Two-timescale mirror
  `_Ss` (config `gsyn_slow_decay`>0, `gsyn_slow_beta`) = HOPE-style
  continuum memory pair: fast adapts to the passage, slow
  accumulates long-range regularities; read sums both.  Read =
  `w·(log p_softmax(R@(S@q)/τ) − log p_uni)` additive lift,
  thin-gated (`gsyn_thin`, regime-0).  Write sites mirror gcell:
  `_learn_token`, `_bookkeep_fused`/`_fused_seq`, `_ingest_bulk`
  (`_gcell_feats_bulk` shares one feature pass with the codes),
  `teach()` (`_gcell_feats_ctx`).  Journal 'GS' rides the fp16
  `_gsyn_ring` [coeff_f|coeff_s|v_err|k] — undo is the exact
  inverse `S_pre=(S−c·v_err⊗k)/γ` (same precision class as 'a').
  Lift is post-graph in `_apply_sparse_lift_dev` (host-computed
  per step like gcell) so CUDA-graph capture/replay is untouched.
  Snapshot 'gs' = (S, Ss) tuple — bare tensor = pre-two-timescale
  load path.  `memory_report()['gsyn_MB']` counts S+Ss+ring.
  `mem_bias` (SamplingParams → `generate(mem_bias=)` →
  `model._gen_bias`) is the recall↔gen runtime dial: scales both
  gcell and gsyn lift weights for the call, restored after; 0 =
  recall-only, >1 = lean generative.
  R60b — `r_live` (default ON, `r_eta` knob): the no-SFT substitute
  for `graft_teacher`/`distill_gen`.  Additive drift table
  `_Rl [V,topic_dim]` — each learned token's row EMAs toward the
  normalized PRE-x context sketch at rate `r_eta*s` (surprise-scaled
  co-occurrence embedding).  `_R` stays the frozen seeded bipolar
  matrix → `_c`/`_c2`/knn context-side EMAs keep bit-exact inverse
  undo; vocab-facing reads use materialized `_Reff = _R + _Rl` (gsyn
  logits + delta-rule targets; `teach` binds target rows too).
  Undo: 'rL'/'rB'/'bc'["rl"] first-touch restores — 'rB'/'rL' are
  revert boundaries (session-scoped, 'bc' class).  Snapshot key
  'Rl'; `_Reff` re-derives on load.  Write sites: `_learn_token`,
  `_bookkeep_fused` (inverts topic EMA for pre-x `_c`),
  `_ingest_bulk` (`_rl_bulk`, deterministic `_group_totals`), teach.
  `_R` dtype stays int8 — `_Rl` is the only live float table.
  R60c — recall→gen (`gen_recall` knob + `consolidate_gen()`): the
  recall side feeds the gen side.  Live: warm contexts (pre-write
  order-cell tot > thin_tot) write gcell/gsyn/_Rl at
  `max(surprise, gen_recall)` — recall-proven successors transfer
  even when unsurprising; wired in `_learn_token` (probe hoisted
  pre-write), `_bookkeep_fused` (`boff_thin` flag), `_ingest_bulk`
  (tots_f/found_f).  Batch: `consolidate_gen(mass, top_m, min_tot,
  tag)` replays on-ring positions (closed-form `_cprev_seq` seeded at
  zero inside the episode), gathers each context's cross-order
  successor distribution, writes top-m share-weighted followers into
  gcell + gsyn (explicit `v=` targets in `_gsyn_chunk`) + `_Rl`.
  One ws/we block; gsyn "gs" snapshot + "rB" session + per-cell "G"
  — revert_tag restores bit-exactly.  `_ringg` host fallback added
  for CPU `_gcell_feats_bulk`.  ~20s to basic topic competence
  measured (21.7K-token corpus, CPU).
  R60d — `gsyn_deep` (default OFF, 9 knobs): the nonlinear memory
  channel — a 2-layer MLP f_theta(k)→expected _Reff row trained
  ONLINE by surprise-scaled gradient steps (TTT-MLP/Titans: the
  memory is a learned function, not a table).  ~115K params +
  momentum buffers, <0.5MB; explicit no-autograd gradients, global
  clip (`gsyn_deep_clip`), heavy-ball (`gsyn_deep_momentum`),
  decay forget-gate (`gsyn_deep_decay`).  SEPARATE from linear
  gsyn (own lift w/thin/temp/journal/explain/report/probe) so
  ablations isolate it; `_gsd_chunk` = LaCT-style batched grad
  steps in bulk/consolidate.  Undo = whole-theta snapshots
  ("gd" entries + blk["gsd"] in "bc" blocks; `_gsd_session` flush
  per ingest span).  NOTE: "gd" had to be added to revert_since's
  boundary set — without it a rewind overshot into the previous
  span's session snapshot (latent bug class for any future
  span-scoped op).  `probe_gsyn_deep()` exposes k/h/o stats +
  param norms + theta drift.  Proof: XOR-corner task — antipodal
  keys → same target — linear gsyn 2/4 (chance, f(-u)=-f(u) is
  structural) vs gsyn_deep 4/4.
  R60e — `dnet` (default OFF, 8 knobs): the iterated-memory channel —
  a stack of gated delta-rule layers, i.e. a linear-attention
  "transformer" that learns online with no training loop, no KV
  cache, no optimizer state (~0.34MB total for 2 layers).  Each
  layer's S_l is the same delta-rule memory as gsyn; the STACK
  supplies what single-step channels structurally lack: iterated
  content routing (layer-l input = relu(o_{l-1}) — one layer's
  readout addresses the next layer's memory; relu breaks the
  odd-symmetry linear ceiling).  Content-gated writes
  `sigmoid(w_b·u)·beta·s` (GDN-style), scalar decay.  Same journal/
  revert/explain/probe/report parity as gsyn_deep ("dn" op +
  blk["dnet"], in the boundary set).  Depth proof: antipodal task
  1-layer=2/4 vs 2-layer=3/4.  Bulk writes are exact-sequential
  (layer l needs l-1's CURRENT state — the WY chunk shortcut
  doesn't lift; optimization path noted).
  R60f — **fuzzy channels are now hedge members + candidate-
  restricted**: `gcell`/`gsyn`/`gsd`/`dnet` sit at the tail of
  `self.channels` (flag-gated).  Real probs in `_channel_prob`
  (eager; `_pfq` memo per position) and `real_r` under
  `bulk_channel_rewards` (bulk, post-block-state approximation);
  abstain-neutral on the fused device step (`_fidx` filled with
  uni_x).  All four lifts multiply by w_rel = C·w (relative to
  uniform share — sinks when the channel loses; replaces the
  fixed-weight bypass that made them unpunishable).  Knobs
  `lift_topk=64`/`lift_topm=8` (0=off): fuzzy lifts vote only on
  top-k of the pre-lift row + own top-m proposals —
  `_sparse_lift_vec_dev` now returns (lift, flift); `_apply_sparse_
  lift_dev` computes the support mask once per predict.  Load:
  `wR` width mismatch replicates the global row.  Measured: grid
  `@2K docs seen_qa` 0.28 → 0.84 (hedge alone carried it;
  restriction = structural guarantee), `backoff_order_sharp=1`
  → 0.98.  Eval harness: `.devin/eval_flux_quality.py` +
  `eval_flux_grid.py`.
  R57 — gcell fuzzy read (SDM semantics, `gcell_read_radius`, default
  1): predict-time Hamming-ball probe over learned codes — similar
  contexts pool evidence, so generalization is a *read* of existing
  cells, not extra stored state.  `_gc_buckets` coarse index
  (`code >> gcell_bucket_shift`, default 6) prunes probes — codes
  within dist r differ in ≤r high-part bits, so enumerating
  high-parts within r covers all candidates; members distance-
  checked exactly (`(cc^q).bit_count()`).  Contributions decay by
  `gcell_neighbor_decay**dist` (default 0.5).  Bucket ops ride the
  existing 'G'/'gE' journal (index mirrors cell lifetime — undo
  maintains it inline); index is derived state, rebuilt on load.
  explain() cells now carry a `dist` field per fired gcell code.
  R58 — fine-grained credit + verifiers + exploration in flux
  self-play: `reinforce(tag, gain, span=(lo,hi))` replays only
  deltas whose token position falls in the span (positions are
  reconstructed from 't'/'bc' markers — 't' CLOSES its write group;
  absolute anchor = _count − journaled tokens, 'bc' blocks carry
  blk["count"]); `gain` may be a callable `gain(pos)` for shaped
  credit.  `flux_verifier` (callable(response, pair) → [0,1])
  replaces the binary grader: `attempt["reward"]` scales
  `reinforce` gain (`flux_gain * reward`); below-threshold-but-
  nonzero attempts (and parsed tool calls under wrong answers)
  apply their packets at `flux_gain * max(reward,
  flux_partial_gain)` before teach-fail.  `flux_explore_eps` /
  `flux_explore_temp` give ε-greedy temperature exploration in
  _temps.
  R59 — counterfactual credit via logged propensities (logged-bandit
  / CRM): `generate_batch(cf_topk=k)` makes FluxStreamPool record
  each generated token's top-k (id, prob) alternatives into
  `FluxPacket.alts` (post-temp/mask sampling distribution — the true
  propensities).  `FluxLM.reinforce_alts(tag, packet, gain, scale,
  block_idx)` walks the applied ws/we block for per-position order-
  cell keys ("c"/"t" groups host-side, "bc" writes_all on GPU;
  block_idx selects among same-tag blocks since apply order =
  journal order) and writes `gain*scale*pi/(1-pi_sampled)` to each
  untried alternative via `_write_counts_keys`, journaled under
  `cf:<tag>` (independently revertible).  Self-play wiring: success
  gives alts a share of the win; partial-credit failures boost alts
  at the same scaled gain; zero-reward failures (flux_cf_fail)
  apply the packet, boost the alternatives, then revert_writes(tag)
  — the untried actions keep credit for what the tried one lost.
  Greedy/degenerate positions (pi_s>=0.999) are skipped.
  R60 — bulk-ingest speedup round (~850-1900 → ~4.5-7.5K tok/s on
  GSM8K text, RTX 5070): (a) `_gsyn_chunk` replaced per-token
  `_gsyn_write` in the bulk path — WY/triangular-solve delta-rule
  blocks (fp64, exact vs sequential, `blk["gsyn"]` S/Ss snapshot for
  undo); block size default 256 with adaptive cap `bs <= 1 +
  -6ln10/ln(gam)` keeping `g^-(bs-1)` fp64-safe. (b) `add_multi`
  collision-repair loop batched: one `_slots2`/probe + grouped undo +
  batched follower-column pick for the whole stuck set; per-element
  sequential writes remain only for ordering, with live re-probe +
  host row-read restricted to elements whose target flat was
  conflicted or written by an earlier repair (`written` set). (c)
  journal flood fix — bulk per-token C/K/W/H/G/GS/eE entries ride a
  per-block sub-journal (`blk["sub"]`, replayed in `_undo_block`), so
  a 40K-token blob emits ONE journal entry instead of ~187K and `bc`
  headers can't be evicted mid-block (previously reverts silently
  no-oped). (d) `_A` stability: grouped topic deltas are damped by
  the geometric-saturation factor `phi` (batch-mode delta rule with
  m duplicate tokens overshoots past m·lr·s>2 → inf→NaN chain);
  proto EMA rewritten as bounded decayed-sum matvec (the cumsum/
  `pd^-j` construction overflowed at chunk ~8K); non-finite surprise
  treated as maximally surprising.  Adaptive knn read-skip already
  shipped (`adaptive_read`).  Deferred: FAISS binary index for
  gcell fuzzy reads (new dep), suffix-array recall core
  (architectural).  Bench: `.devin/bench_corpus.py`,
  `.devin/bench_stuck.py`, parity `.devin/gsyn_parity.py`.
  R61 — packed v2 `.flux` snapshot format: v1 pickled dict-of-dict
  cells verbatim, so a 5.9M-token corpus model serialized ~16GB
  (journal `bc` blocks ~13GB of tensors + 2.9GB `data.pkl` of Python
  objects) and took ~6min/~16GB RAM to load.  `snapshot(path,
  keep_journal=False)` now writes `fmt=2`: every `{key -> {tok: cnt}}`
  table becomes CSR columns (keys_u64/rowptr/toks_i32/cnts_f64 —
  f64 preserves exact Python floats), `_epi` becomes
  keys/rowptr/pos/succ/clen/cert columns, `last_seen`/`_epi_fifo`/
  `_gc_fifo` become column pairs, and journal entries are dropped
  unless `keep_journal=True` (deployment default off; in-session
  journal on the live model is unaffected — post-load reverts only
  reach pre-snapshot writes when the journal is kept).
  `_GPUCellTables.triplets()` vectorized (masked tensor dumps →
  np.repeat/reshape, was a ~100M-step Python loop) and
  `load_triplets` chunks `add` calls (2M/slice) so big restores
  don't spike VRAM.  knn bank slices to the live prefix
  (`_knn_live`) instead of serializing full capacity.  Load path is
  backward compatible: `_restore_state` detects `fmt==2` and
  `_v2_to_v1` unpacks columns back to the v1 structures before the
  shared restore code — v1 files still load.  `clone()` stays on
  the v1 in-memory path (keeps journal).  Migration:
  `.devin/migrate_v2.py`; round-trip test `.devin/v2_roundtrip.py`
  (predict/A/S/epi/gc bit-parity cuda+cpu).
  R68 — live generalizer "gen head" (`gen_live`, default OFF;
  `gen_live_topm`, `gen_live_min_tot`): consolidate_gen made
  incremental.  Every learned position whose order cells hold
  evidence (post-write tot > min_tot) transfers the cells' recalled
  successor DISTRIBUTION — top-M followers, share-weighted — into
  the gen channels at strength `gen_live*conf` with
  `conf = max_k tot/(tot+backoff_kappa)`: gcell per-follower cell
  writes, gsyn/gsd/dnet blended soft targets
  `sum(share*_Reff[t])`, and `_Rl` per-follower row EMAs.  The head
  owns NO weights — it moves mass between existing stores
  (`_genlive_write` in statemixin.py).  Unlike `gen_recall` (floors
  the realized token's own write), it covers followers that were
  not observed at the position, and unlike `gen_recall` it keeps
  writing as cells converge (realized writes stall at s→0, conf
  stays ~1).  All three learn paths read POST-write cells: eager
  `_learn_token` reads `_tables`/`_totals` after the order write;
  `_bookkeep_fused` re-probes post-step (`_fused_seq` batches one
  post-row probe over all positions — consolidate_gen-class
  approximation); `_ingest_bulk` reuses `slot2`/`found2` (the
  reward fix-up's post-add_multi re-probe — pre-write probes made
  first-pass bulk ingests transfer nothing; that was the GPU test
  bug).  `_predict_only` skips it; thin contexts self-gate.  All
  writes ride the existing journal ops — `revert_since` on stream
  ingests restores `_gc`/`_gct`/`_Rl` bit-exact and `_S` to
  fp16-factor precision.  Self-play: `--flux-gen-live` (implies
  `--flux-gen-channels`), `--flux-gen-consolidate` batch complement.
  Probe `.tmp/exp_gen_live.py`: mass +14-38% in gen stores, NLL
  neutral on paraphrase frames (transfers stay distribution-
  consistent — benefit is continuous consolidation, not noisy
  writes).  Tests `test_gen_live_*` (6).
  R69 — extractive knn recall lift for held-out QA (all knobs
  default OFF in `config.py`): `_knn_votes_dev` is now a pipeline —
  broad cosine pool (`knn_pool`) -> high-idf coverage rerank
  (`knn_cover`/`knn_cover_w`) -> votes (successor; `knn_span` span
  voting w/ `knn_dedup`+`knn_span_idf`; `knn_twin` answer-adoption
  from rows inside trained "Q:..A:" lines; `knn_spread*` corpus-wide
  content field, `knn_spread_doc` = flat doc-coverage maxpool x
  own-idf).  Fingerprints: `knn_window` last-W bag + `knn_idf` +
  `knn_residual` (prose-mean removal — fixes the cos~0.99 collapse)
  + `knn_episode` prompt-span queries (`_prompt_start`/`_prompt_g`
  set in `ingest()`).  Merged decode in `generate()`: `lg += knn_lift
  * log p_votes` for `knn_lift_steps` — recall merges into
  generation rather than replacing it, gated by deep-order cell
  existence (`knn_lift_deep`, seen keep verbatim recall),
  `knn_lift_thin`, `knn_lift_conf`, and vote concentration
  (`knn_lift_topconf`).  Answer masks: `attach_vocab(itos)` +
  `knn_atype` (word-initial alpha) + `knn_tokmin`/`knn_tokmax`.
  BUG FIX: the spread field used to include the prompt span in the
  live mask — the question self-matched its own rare tokens; the
  query span is now excluded from field source and read positions.
  Measured (4K trivia+evidence): seen_qa 0.86-0.96; paraphrase
  generalization 54-80%; held_qa ~5% — corpus-knowledge bound
  (answers absent from the docs), not a mechanism bound.
- `forge/model/strata/` — STRATA (R60): FluxLM's successor as a
  three-artifact format — WEIGHTS `.strata` (labeled CellBlocks of
  packed 48B cells: tag32+meta+8×u32 followers+raw64, log-quantized
  counts, 2-choice cuckoo hash, active/sealed/cold lifecycle,
  per-block undo arenas, tag-scoped revert, memmap container IO),
  PARAMS `.iq` (ParamSet — mixer/dense weights; seeded tensors stored
  as {seed,shape} and regenerated), KNOBS `.cond` (KnobSet — scalar
  gates/lifts/sampling with named overlays).  M0 = substrate;
  **M1 `StrataLM` (model.py) DONE**: suffix-hash order channels
  o3..o64 + uni/backoff/recency/fatigue, hedge geometric mixture,
  surprise-gated writes, ingest/teach/observe/learn, forward→
  (logits,None,None), generate+knobs, explain/audit/revert/snapshot —
  CPU-eager reference, ~86 tok/s learn.  Design =
  `docs/STRATA_M1_DESIGN.md`; further milestones (dense gsyn channel
  should reuse FLUX's r_live live-feature pattern — params+knobs, no
  distill).  Bench: `.tmp/bench_strata_step.py` — eager is
  launch-bound dead (~28 tok/s); graph replay 585µs for 10
  probes+dense; `pk_of` per-call salt alloc breaks capture; add() not
  graphable → two-tier thin-write + deferred repair.  Tests:
  `tests/unit/test_strata.py` (22) + `test_strata_m1.py` (25).
- `forge/scrape_feed.py` + `scripts/scrape_feed.py` — R61: continuous
  online scraper -> normalized doc feed.  Sources (`wiki-random` /
  `wiki-all` resumable allpages cursor / `wiki-recent` / `wiki-search` /
  `arxiv` / `wiktionary` / `urls` / `ddg`) -> Processor (persistent
  sha1 dedupe `seen.txt`, min/max chars, injected filter_fn — CLI wires
  `download_wikipedia.should_skip` + train_flux `--topic` scorer; wiki
  keeps `raw` wikitext for the filter since should_skip keys on
  {{disambig}} markup) -> ShardWriter (`feed_NNNNNN.jsonl.wip` ->
  `.ready` rotation; the filesystem queue is the bus) -> sinks:
  `--to shards` (corpus only), `--to api` (serialized POST
  /engine/learn per shard, polls /engine/stats `info.learn` for
  completion — the route is one-at-a-time), `--to model` (in-process
  FluxLM ingest + snapshot, tag "<tag>:<source>" for revert).  Gotcha:
  prop=extracts caps whole-article text at 1 page/request
  (exlimit>1 needs exintro) — live full-text uses prop=revisions
  wikitext (~20 pages/req) + local `wiki_to_text` cleaner.  All fetches
  injectable; tests `tests/unit/test_scrape_feed.py` (16, offline).
- `forge/self_play/flux_goals.py` — R53: source-grounded goal layer for
  flux-mode self-play.  `FluxTools` = budgeted cached executor for the
  model-facing tools (`wiki`/`dict`/`paper`/`web`, `parse_tool_call`
  accepts `wiki X`, `dict: x`, `[paper] x` + aliases); doc fetchers
  return rich payloads for goal building while `call()` returns
  one-line `obs_chars`-clipped observations.  Sources: Wikipedia REST
  summaries (search→summary or random), Wiktionary REST definitions
  (primary; dictionaryapi.dev fallback; local kaikki jsonl via
  `--flux-dict-file`), arXiv Atom (year/first-author/id goals).  Goal
  builders are EXTRACTIVE — every `response` is a verbatim span/field
  of the doc so `_flux_check` stays exact and teach-on-fail writes real
  knowledge.  `FluxKB` = append-only `flux_kb.jsonl` under status_dir
  (deduped (source,key); replayed for new question forms when the fetch
  budget is spent or offline).  `FluxGoalSampler.sample_web` =
  round-robin fresh fetches + KB replay, deduped vs cross-epoch seen
  AND the eval sets; `build_eval` persists a frozen
  `flux_eval_web.json`.  Loop wiring (`infinite_loop.py` R53):
  `flux_web`/`flux_goal_sources`/`flux_web_frac`/`flux_tool_frac`/
  `flux_rag_frac`/`flux_web_fetch_budget`/`flux_eval_web_n`/
  `flux_dict_file`/`flux_obs_chars` config+CLI; web goals get kind=
  "qa"|"rag"|"tool" — tool mode is two-phase (`T:` line → loop executes
  → `O:` injected → `A:` answer), failures teach_text the full
  trajectory so FLUX learns the tool-call binding verbatim; eval =
  procedural + frozen web sets combined.  Offline → `FluxTools.online()`
  probe fails once and everything degrades to the procedural stream.
  GOTCHA: Wikimedia hosts 403 browser-spoofed UAs (robot policy) —
  `web_primitives._ua_for` sends a descriptive BOT_UA to
  *.wikipedia.org/wiktionary.org/etc; also MediaWiki REST paths need
  `_` for spaces (`+` 404s).  Tests: `tests/unit/test_flux_goals.py`.
- `forge/engine/decoding.py` — Decoding strategies (standard, speculative
  family, `dola` self-contrastive R50-2, `uno`, `specbound` adaptive
  early-exit self-speculation R60-5 — greedy bit-identical to standard;
  DRY n-gram penalty R50-1 in the shared `_sample_from_logits` chain +
  all `engine_generation` paths, `dry_multiplier`/`dry_base`/
  `dry_allowed_length`/`dry_penalty_last_n` params, 0=off)
- `forge/engine/gated.py` — ForgeGate three-probe gated generation
  (route/doom/convergence); `ForgeEngine.load_gate_probes()` +
  `generate_gated()`; probe bundle `research/checkpoints/gate_probes.pt`
  (~44KB, version-guarded). VRAM overhead ~0 (logistic heads on hidden
  states the model already computes). Trained on the gate_r 1450-label
  self-verified corpus; retrain pipeline documented in
  `.devin/scratchpad.md` (R&D: ForgeGate probes). Held-out eval on
  ForgeLM V2: +16.4pts acc at −34% tokens vs always-think; numbers are
  corpus-dependent — recalibrate probes on new domains/checkpoints
  before trusting thresholds. GUI wiring:
  `EngineService._load_blocking` auto-loads the bundle when present;
  `chat_loop._route_p_easy` scores the route head once per turn on a
  TOOLS-FREE render of the conversation (the ~17-schema `<tools>` block
  collapses h_mean and drags p_easy ~0.25 below threshold — "Hello":
  0.355 tools-free vs 0.098 with tools; probe was trained tools-free)
  and switches to a closed-think render (`<think></think>` + "Answer:"
  anchor — bare </think> leaves the model musing) when p_easy ≥ 0.2.
  Bare greetings/acks bypass the probe entirely (`_is_trivial_turn`,
  gate_r has no chit-chat class). Emits a `gate` SSE event the Chat page
  shows as a `gate → mode (p=…)` chip. Routing + conv exit run ONLY on
  fresh user turns (last conv msg is `user`) — tool-continuation rounds
  go straight to think+budget: both probes are OOD on synthesis turns
  (observed: conv-exit forced `Answer:` → model restated its plan and
  re-called the same tool 5×). Think turns get TWO exits: the
  conv probe early-exit (`_conv_exit_observer` scores per-step hidden
  states via `generate_stream(hidden_observer=)`, K=2 consecutive
  >0.75 past 32 tokens → inject force-answer; emits a `conv-exit` gate
  event) AND the hard cap backstop (`_think_cap_processor` injects the
  gated.py force-answer suffix after `think_budget` think tokens,
  default 160, per-request via `ChatSendRequest.think_budget` /
  Chat settings "Think budget", 0=no forced exit) — plus a `<think>`
  (id 541) ban in generated text in every mode (a generated <think> can
  only re-open a reasoning pass — observed: model emitted `Answer:` then
  re-opened <think> and re-looped). Direct path passes `budget=None`
  (ban only). Structural bans also apply every mode: `<|im_start|>` 518,
  `<tool_response>`/`</tool_response>` 539/540 (fake tool results), and
  `<tool_call>` 531 is masked while the think block is open — trained
  order is think → `</think>` → text → call; a mid-think call used to
  leave the block unclosed so reasoning leaked into the persisted reply
  (the "thinking escapes the tool call" bug). `tool_calls_allowed=False`
  (tools off, or defs dropped by the repeat guard) bans 531+532 outright.
  Text-level belt: a rolling 24-char tail stops the stream on literal
  `</tool_call>`/`<tool_response>` emitted as ordinary tokens (id-EOS
  can't see them). Post-parse: `_split_reasoning` puts pre-`</think>`
  text in `msg["reasoning_content"]` (re-rendered into the next tool
  round by the template, rendered as ThinkCard in the UI, dropped when
  a new user turn starts) — unclosed output is all-reasoning/empty-body,
  never a visible reply; `_strip_answer_anchor` drops an echoed
  `Answer:`/`Final Answer:` lead. Fed-back tool results are truncated at
  4000 chars. Two output guards: repeat-call detection (`_call_sig` —
  an identical name+args call drops tool defs so the next round must
  synthesize) and `_strip_direct_musing` (direct-path text before a
  stray `</think>` is reasoning voice — dropped; pre-call musing on
  direct tool turns is dropped entirely). Threshold calibrated via
  `scripts/bench_gate_route.py` — rerun it when the probes are
  retrained or the chat template changes. The agent loop
  (`agent_loop.py`) reuses the same processor with a 384-token think
  budget and a bare `\n</think>\n` close (not the "Answer:" suffix) so
  a capped think can still proceed to a tool call.
- `forge/engine/decide.py` — SystemOneEvaluator: TypeSafe AI "System One"-
  compatible typed decisions (noul/choice/score questions → TypeSafe-style
  probability answers) via candidate-continuation scoring — single forward
  passes, no generation. Reasoning-model aware: primes
  `assistant\n<think>\n</think>\nAnswer:` so answer tokens land at the
  read position. `ForgeEngine.decide(state, questions)` wraps it.
  Server exposes `POST /v1/systemone` (typesafe_sdk drop-in: point
  `base_url` at Forge; `/v1/models` sniffs `X-TypeSafe-SDK` header and
  returns the TypeSafe `{"models": [...]}` shape). Without a scorer,
  probabilities are raw LM softmax — consistent within a question but
  NOT calibrated.
- `forge/engine/decision_head.py` — Tier-1 `DecisionScorer`: linear
  verifier head on last-token hidden states, `s(q,c) = w·h(prompt+" "+cand)`
  softmaxed per question; covers noul/choice/score with one head. Also
  `ProcessRewardHead` (R50-3): sigmoid step verifier on step-end hidden
  states — `fit_prm(model, tok, device, dataset)` trains on per-step or
  outcome-broadcast labels (BCE, frozen base), `score_steps` returns
  P(step correct); ~10KB weights, feeds GRPO advantage shaping.
  Train via `fit_decision_scorer(model, tok, device, dataset)` —
  dataset rows `(state, question_spec, correct_candidate_idx)`, group
  softmax CE (proper scoring rule) + LBFGS temperature scaling; ~10 KB
  weights, `scorer.save()`/`DecisionScorer.load()`. Attach with
  `engine.load_decision_scorer(path)` or `decide(..., scorer=...)`.
  Real-model check (340 mixed templated examples): ECE 0.067 vs 0.141
  raw at equal accuracy — calibrated-style, not production-calibrated;
  validate on non-templated data before confidence gating.
- `forge/keys/` — KeyStack architecture keys (25 canonical + `kda` R49-2
  side-path: `use_kda` config flag, gate=0 bit-exact, KDAKey BI port;
  + `gdn2` R60-1: `use_gdn2` flag, Gated DeltaNet-2 erase/write gates,
  collapses to KDA at warm start — mutually exclusive with `use_kda`).
  `forge/keys/safety.py` is the production `safe_apply`/rollback harness
  (used by pit/lerope/attn_residual/mhc keys `safe=True` paths).
- `forge/quant/` — Quantization implementations. `inference_quant.py`
  `quantize_model_w4()` packs nn.Linear weights into torchao
  `Int4TilePackedTo4dTensor` (W4A16, group=128, tinygemm kernel —
  `aten::_convert_weight_to_int4pack` is CUDA-only, so each linear is
  moved to GPU, quantized from bf16, then released; peak VRAM = packed
  total + one layer). Wired into `from_checkpoint` via
  `load_quantize="int4"` (engine_checkpoints `_load_quantized`): loads
  bf16→meta→assign on CPU, per-module CUDA quantize, skips the
  post-load auto-quant pass AND compile/graph features
  (`_auto_activate_optimal` hard guards — torchao tensor subclasses
  hang inductor, capture is fragile; guards also override caller flags).
  Compatible features auto-enabled for packed pure-attention models:
  learned prefix cache, adaptive+suffix spec decode, FASER, lookahead
  gate, POD attention, hybrid prefill, hot prefix. Excluded: lossy
  KV/attention pruners, RoPE replacers, P-EAGLE (needs trained draft),
  arch-specific (quamba2/replay_ssm/MoE/triton_conv/fa4), extra weight
  quant. `from_checkpoint` splits ActivationConfig-named kwargs into
  `_user_activation_overrides` — forge_server flags:
  `--load-quantize int4`, `--kv-tokens N` (caps KV prealloc).
  Validated on Hermes-3-8B: ~11.5s load, ~7.6GB VRAM incl. KV,
  ~34 tok/s, 28k-token KV budget. Tests: `tests/unit/test_llama31_w4.py`.
- `forge/decoding/` — Decoding implementations
- `forge/training/` — Training runners + optimizers; `training/data/`
  holds the dataset pipeline modules (`efficient_pipeline`,
  `parquet_dataset`, `curriculum_augment`, data-prep one-shots) — it is
  tracked source, exempted from the `data/` gitignore rule.
  `runners/sft_train.py` renders all examples via the CANONICAL
  `chat_template.apply_chat_template` (`<|im_start|>`/`<|im_end|>`,
  `<tool_call>`/`<tool_response>`, thinking=False direct answers). The
  legacy `<|startofsegment|>`/`<|tool_call_start|>` markers are gone —
  they don't exist in the V2 tokenizer, so SFT was training a format
  inference never sees (fixed 2026-09-21; see BUG_LOG).
- `forge/evolution/` — Evolutionary optimizer (ForgeEvolve); CLI: `python -m forge.evolution --domain <name> --steps <N>` (use `--list-domains` to see all domains). Domain specs + run/focus profiles live in `forge/evolution/configs/` (canonical; the legacy `tests/evolution/configs/` copy was removed — the tests/evolution harness scripts now point at the canonical dir). `--fixate` pins a run to one domain: no cross-domain transfer, no refinement spawning, plateaus re-diversify in place (`_fixate_rediversify`); `run_evolve.py --fixate` additionally kills auto-discover/revisit/parallel waves. `forge/evolution/domains/livet_domains.py` — `MicroLiveLM` (R64): the micro live-format search domain; REAL evaluator (builds ConfigurableResearchLLM+LiveSession on CUDA, measures fact-recall live_gain, bs=1 decode tps, ingest tps, params, revert parity — not synthetic). Structural grid via `discrete_choices` (240 cells) + 9 continuous dims. Focus topic: `--focus micro_format`. Results: conv+titan+ngram substrate dominates the micro frontier (d64-class ~0.2-0.5M params, gain ~5, ~250tps); see scratchpad R64. `forge/evolution/domains/flux_domains.py` — `FluxTune` (R65): FluxLM self-tuning domain; builds real FluxLM (cuda_primary, graphs+compile_dev off — concurrent captures/poisoned RNG state crash threaded evals) and measures held-out NLL, 1-hop + 2-hop chain recall, pred/ingest/gen tps, memory_report+VRAM, argmax diversity; score = weighted log2 ratios vs production defaults (baseline anchors at 25). `forge/evolution/domains/meta_domains.py` — `EvolverTune` (R65): meta-evolution — candidates are ForgeEvolveConfig knobs scored by real inner ForgeEvolve runs on SyntheticDomain (progress vs random-floor + eval efficiency); fixated+warm_start=False inner runs isolate proposal quality. Both set `eval_thread_safe=False` (engine opt-out of `_threaded_eval` side-stream pools — added in engine.py `_gpu_batch_eval`). Categories/focus topics: `fluxlm`, `evolver`. DBs: `forge_evolve_flux.db`, `forge_evolve_meta.db`, `forge_evolve_meta_inner.db` (scratch). R66 proposal frontier (engine.py): proposals are submitted to a pending frontier every gen and popped for eval later — submission is decoupled from eval throughput (`_submitted` counter in logs; measured ~2.1K unique proposals/s on synthetic). `ForgeEvolveConfig.propose_mult` (proposals/gen = n_generators×propose_mult, multi-sample grouped-einsum forward in `BatchedGenerator`), `frontier_cap` (evict lowest surrogate-scored pending), `quick_ratio` (multi-fidelity: pop quick_ratio×, domain `quick_eval` ranks, top fraction promotes to full eval). Domains may set `cpu_eval_ok=False` to opt out of the `parallel_eval` CPU pool (FluxTune/EvolverTune do — GPU-bound/nested-CUDA). Dedup is one D2H transfer + vectorized round for all candidates (was per-candidate `.cpu()`, ~100x). FluxTune quick_eval: ~1/3 corpus, 200 held positions, no decode probe, quality+mem terms only (tps is wall-clock noise at that fidelity), ~3x cheaper, deterministic prod=25.0 anchor. FluxTune hop2 probe is real chain recall (`#V>` -> V_next, the hop-1 value becomes the hop-2 key — measures association chaining, baseline ~2.35 nats). R66b literature fixes (AlphaEvolve/FunSearch/pyribs/MEMES parity): dedup keys on the DECODED config (discrete-bin domains were re-evaluating identical configs under different param vectors); per-generator archive-conditioned contexts (60% UCB elite / 15% random cell / 15% two-elite blend / 10% best — was one shared elite for all generators); rank-shaped scores for trainer+evolve (OpenAI-ES centered-rank fitness shaping — -1e9 failure spikes no longer poison fitness_ema); explore emitters (first 20% of generators credited on rank-shaped novelty, MEMES-style); crossover emitter (`crossover_frac`, blends two elite encodings + jitter as direct proposals, generator_idx=-2); islands (`n_islands` deme archives — generator g feeds island g%K, weakest half import global elites every `island_migrate_gens`).
- `forge/evolution/domains/mech_domains.py` — `MechSearch` (R67): mechanism discovery for the FluxLM/micro-format dev plan. 22-dim candidate space spanning the *mechanism surface*: attn_type (gqa/gta/gla), delta path (none/kda/gdn2), titan fast-weight rank compression (0/d2/d4), solver, ngram sidecar+orders, live_projections (bounded substrate drift), write_threshold (surprise admission gating), forget/surprise_cap/chunk — real eval builds ConfigurableResearchLLM+LiveSession on CUDA (live_gain, decode/ingest tps, params, revert parity; same scoreboard as MicroLiveLM). `quick_eval` tier + `gpu_pool_ok` (spawn workers; small eager models). Smoke: gla+live_proj+gate 49.36 ≥ R64 winner 49.33. Category `mech`.
- `forge/evolution/engine.py` GPU eval pool (R66c): `gpu_eval_workers>1` spawns N `spawn`-context worker processes (`_gpu_worker_init`/`_gpu_eval`), each owning its own CUDA context + domain instance — evals share the device but never a context/graph/RNG. Opt-in ONLY for domains declaring `gpu_pool_ok=True` + `gpu_worker_kwargs()` (currently just `FluxTune`; quick tier routes through `quick_eval` inside workers). Worker exceptions → -1e9 fail-score; pool failure → serial `_gpu_batch_eval` fallback. VRAM rule of thumb on the 12GB 5070: workers x worst-case candidate VRAM (prod mem_preset ≈ 4.2GB) must fit — `gpu_eval_workers=2` measured ~1.8x eval throughput; 3 is risky. Cleaned up in `run()` finally alongside `cpu_pool`.
- `forge/evolution/program_vm.py` + `domains/math_prog_domain.py` — `math_prog` / `MathProgDomain`: batched program-synthesis domain. Candidates ARE algorithms — fixed-length register-machine programs over a closed opcode table (15 total ops, protected div/mod/pow, no eval/IO/loops → nothing to scan; ISA is the sandbox). Decoded from generator `[0,1]^45` vectors (12 steps × op/a-addr/b-addr + 8 consts ±64 + out-sel). `run_programs` executes [B,T,R] register buffers fully vectorized on CUDA (~2700 programs/s; bench: `scripts/bench_math_prog.py`). Tasks randomized per eval: FUNC (match hidden f, held-out graded), GOAL ("A must equal T" — compose bounded consts; pinned via `goal_target`), GOAL_IO (constraint pairs incl. pinned `goal_inputs`+`goal_target`, oracle shifts `h(u)+(T-h(u0))` so held-out still forces real math), FIT (kind 3: approximate a pinned `target_fn` — e.g. tanh/sigmoid/log2/sin/rsqrt — graded by `fit_score`=-log10(max_err) + continuous proximity kernel; op whitelist via `ProgramSpec.op_ids`/`--ops cheap|cheapdiv` restricts the ISA to deployable ops). Anti-gaming: `heldout_exact`, `uses_inputs` (dependency-cone only — dead-end reads don't count), `const_cheat`, `degenerate` metrics → RewardGuard in `configs/domains/math_prog.json`. Batched path: `MathProgDomain.batch_evaluate` is auto-delegated by `BatchedEvaluator` (any domain defining `batch_evaluate` wins). Driver: `scripts/evolve_math_prog.py --target 794 [--inputs 12,7] | --fit tanh --ops cheapdiv`. Tests: `tests/unit/test_math_prog.py` (26).
- `forge/self_play/` — Self-play + discovery (`infinite_loop.py` is the RSI
  loop; `live_status.py` writes live telemetry — status.json, heartbeat.json
  with progress-coupled stall detection, events.jsonl — to
  `research/checkpoints/self_play/` for the GUI Self-Play page + CLI polling;
  `--status-dir`/`--no-live-status` flags on the loop). Live spectation:
  `Services._selfplay_watch` (deps.py) tails `EventsReader` every ~400ms and
  republishes onto the hub — `selfplay_event` / `selfplay_status` ws events,
  while `/api/selfplay/status` stays for hydration; `EventsReader.poll`
  survives shrink/ctime-rotation/torn O_TRUNC rewrites (rewinds mid-line).
  `task_done` carries bounded `preview` (model output ≤200 chars) +
  `expected` (≤80); flux mode pushes `flux_cells/journal/knn/epi/mb` +
  `flux_channels` hedge weights into status.json for the FLUX memory card. RSI data path
  (anti-overthinking): `_assemble_solution` AST-cleans every reply
  (comments + docstrings stripped — model musing can't leak into SFT/GRPO
  data); export picks the SHORTEST passing attempt and mixes
  `concise_qa_per_epoch` verified short-answer pairs (default 12, computed/
  curated answers — never model-generated) so each epoch trains direct
  minimal answers; GRPO groups get `grpo_concise_bonus` (default 0.1)
  length-efficiency shaping — correctness dominates, and all-pass groups
  regain nonzero advantage. CLI: `--concise-qa-per-epoch`,
  `--grpo-concise-bonus`. Debug: `--debug-round` runs ONE self-play round
  (propose→solve) with NO training — a `GenTracer` on the curriculum
  records every generation (rendered prompt, raw pre-strip output,
  per-token ids), prints it live, and dumps
  `status_dir/debug_round_ep*.txt|.jsonl`. Use it to check model behavior
  before a real RSI run.
- `forge_gui/` — Qt-free domain layer (`api/` only; PySide6 shell removed).
  Web tools live in `forge_gui/api/web_tools.py` (defs + dispatch) backed by
  stdlib-only `forge/web_primitives.py`: `web_search` (multi-engine chain:
  DDG HTML → DDG Lite → Bing → Google News RSS — first parseable result
  wins), `news_search` (Google News RSS — use for "the news"/current
  events; empty query → top headlines), `web_fetch` (ANY http(s) URL or
  bare domain — auto-https, follows redirects, reports `final_url`; HTML
  → markdown-ish text with inline `[label](url)` links + a deduped
  `links` list of every anchor so the model can follow link names;
  `offset`/`next_offset` page long docs; non-HTML bodies pass through raw
  with `content_type`), `wikipedia_search`, `arxiv_search`. All keyless
  GETs; never gated by safety checks, agent approval modes, or read-only
  chat mode. `parse_ddg_html` drops DDG ad/tracker links (y.js
  redirectors carry u3= not uddg= and can't be unwrapped). `web_fetch`
  clamp is 200–16000 chars via `max_chars`. The
  agent-loop fallback harness also wires `WebTools` so web tools exist even
  if the deps harness factory fails.
- `forge_gui_server/` — FastAPI GUI backend (REST `/api` + WebSocket `/ws`)
  serving the React UI; ports `forge_gui/api/` to plain async services.
  Launch: `python -m forge_gui_server` (desktop window via pywebview),
  `--browser`, `--no-window`, or `--dev` (API only, Vite dev on :5173).
  EngineService preloads the resident model in the background at startup
  (`FORGE_GUI_NO_PRELOAD=1` to disable, `FORGE_GUI_PRELOAD` /
  `FORGE_GUI_PRELOAD_CONFIG` to pick a different checkpoint/config);
  a different-model request mid-load is queued (not dropped), a bare
  re-request for the resident model is a no-op, and unload() uses
  sleep(level=2) — no wasted GPU→CPU copy on discard.
- `forge_ui/` — React 19 + TS + Vite + Tailwind v4 frontend
  (`npm run dev` / `npm run build`; design tokens in `src/index.css`).
  Shared chat/agent primitives live in `src/components/chat/` (Markdown,
  blocks=ThinkCard/ToolActivityCard/SegmentedBody/UserBubble/CopyBtn,
  Composer, ScrollFeed) — reuse them instead of duplicating message cards.
  Backend endpoints the pages rely on: `POST /api/chats/{id}/truncate`
  (regenerate/edit-and-resend), `POST /api/chats/import` (paste a
  transcript → new conversation; `parse_transcript` in
  `forge_gui/api/chat_store.py` handles JSON/ChatML/role-marked/plain
  alternating-paragraph formats), `DELETE /api/agent/runs/{id}`,
  `GET /api/agent/tools` (real harness tool defs for the Agent picker —
  do not hardcode tool lists in the UI).
- `tests/unit/` — Unit tests (CPU-runnable where possible)
- `tests/integration/` — Integration tests (GPU required)
- `docs/` — Documentation + R&D round notes; `docs/FLUX_IMPROVEMENT_ROADMAP.md`
  = FluxLM phased improvement plan + technique audit (R54+ planning);
  `docs/ENGINE_IMPROVEMENT_PLAN.md` = ForgeEngine phased plan (correctness
  bugs → wiring → SSM snapshots → graph decode → continuous batching; the
  "dead-wired" R&D-zoo diagnosis + external precedents)
- `scripts/` — Standalone scripts
- `research/` — Legacy paths (tokenizer cache, checkpoints only)

## Current Architecture: ForgeLM V2 Jamba-3B (SOLE BASE)

**Base model**: ForgeLM V2 — lossless port of AI21 Jamba-Reasoning-3B.
- 28-layer hybrid: 26 Mamba-2 SSM + 2 GQA attention layers
- d_model=2560, 20 heads, 1 KV head (MQA), vocab=65536, max_seq_len=262144
- No RoPE (Mamba handles position), untied embeddings, ~3.2B params
- bf16 checkpoint ~6.1 GB; server VRAM budget 8.0 GB

**Base checkpoint**: `research/checkpoints/ForgeLM_V2.safetensors`
**Tokenizer**: `research/checkpoints/forgelm_v2_tokenizer/`
**Default config**: `forgelm_v2` (load_default_model() defaults to this)
**Path constant**: `research.paths.V2_CHECKPOINT`, `research.paths.FORGE_TOKENIZER_DIR`
**Server model ID**: `forgelm-v2-jamba` (DEFAULT_MODELS in forge_server.py)

All prior checkpoints (Jamba_Reasoning_3B source port, ForgeLM_V2_Light*,
evolution R30/R31 artifacts, LFM2.5 GGUFs) were deleted 2026-09 — ForgeLM V2
Jamba is the sole base model.

## Config Presets

Config presets in `forge/config.py`:
- `forgelm_v2` — **ForgeLM V2 Jamba-3B: THE DEFAULT AND SOLE BASE.** 28 layers
  (26 Mamba + 2 GQA), d_model=2560, ~3.2B params.
- `forgelm_v12_jamba` — R37 research preset derived from `forgelm_v2`
  (parent declared; Mamba-3, Kronecker embed, PIT, OutRo, ForgeHybrid — all
  zero/identity-init for lossless warm start).
- `forgelm_v13_gdn2` — R60 preset derived from `forgelm_v12_jamba`;
  adds `use_gdn2` (Gated DeltaNet-2 side-paths alongside each block,
  gate=0 → bit-exact warm start; supersedes KDA — mutually exclusive).
- `forgelm_tiny` — 4-layer tiny conv+attention model for fast tests
  (no checkpoint, config-only).
- `gen_model_tiny` — tiny generator config for evolution experiments.
- `qwen25_05b` — HF-compat preset for Qwen2.5-0.5B checkpoint detection.
- `qwen3_4b` — HF-compat preset for the Qwen3-4B-Instruct-2507 family
  (e.g. `0xA50C1A1/Qwen3-4B-Nymphaea-RP`). First preset to use the
  explicit `ModelConfig.head_dim` field (128; `d_model/n_heads`=80 would
  be wrong). GQA 32Q/8KV, QK-norm, no QKV bias, RoPE 5M, 256K ctx, tied
  embed. Auto-detected by `_detect_config_from_header` on
  (vocab=151936, d_model=2560, n_layers=36).
- `llama31_8b` — HF-compat preset for the Llama-3.1-8B family (e.g.
  Hermes-3-Llama-3.1-8B lorablated/abliterated tunes; ChatML format).
  GQA 32Q/8KV, no attn/MLP bias, SwiGLU 14336, RMSNorm eps 1e-5, RoPE
  base 500K + llama3 frequency scaling (factor 8 over orig 8K ctx,
  implemented in `forge/model/layers.py` + mirrored in
  `loader._reset_non_persistent_buffers`), untied embed/head, vocab
  128256, max_seq_len capped at 32768 for the 12GB card (native 131072).
  EOS set includes 128000/128001/128040 (im_end). Auto-detected by
  `_detect_config_from_header` (embed 128256x4096, q 4096x4096, kv
  1024x4096). The server renders plain ChatML for `llama*` configs via
  `render_messages_for_config` (no Jamba thinking prefix/`<think>`
  block — see `qwen_adapter.py`).
- `llama32_3b` — HF-compat preset for the Llama-3.2-3B family (e.g.
  NousResearch/Hermes-3-Llama-3.2-3B; same ChatML + tool-call format as
  the 8B). GQA 24Q/8KV head_dim=128, SwiGLU 8192, RoPE 500K + llama3
  scaling factor 32 (3.2 scales 32x over the 8K base, vs 3.1's 8x),
  TIED embed/lm_head (unlike 8B/70B), vocab 128256, ctx capped 32768.
  ~3x faster decode than the 8B; first consumer is ForgeChat's
  hermes3_3b catalog entry as the fast card-conversion/chat model.

All legacy presets (`forgelm_v2_light`, `forgelm_v2_pro`, `forgelm_v12`,
`forgelm_v10_1.2b`, `forgelm_v11_3b_vl`, `lfm25_tiny`, `lfm25_1.2b`, V3–V9)
have been DELETED.

### Evolution-Discovered Promotions (2026-08-24, from forge_evolve.db)
Promoted after validation against evolution data (39,631 discoveries scanned):
- **MTP**: n_heads=4, loss_weight=0.495 (was 2/0.3, score 27.69) ✓ validated
- **SpecDecode (PEAGLE)**: n_draft=7 (was 4, score 57.05) ✓ validated
- **BatchQueue**: batch_window=52ms, max_batch=15 (was 50ms/8, score 44.25) ✓ validated
- **SFT training**: grad_accum=5, grad_compression=int4 (score 30.00/11.10) ✓ validated
- **AirMoEKey**: cache_strategy=lfu, disk_cache_size=4096 (score 8.83, 0% miss rate) ✓ validated
- **ForgeEngine**: CREATIVE_SAMPLING preset (temp=1.98, top_p=0.989, top_k=69, score 10.45) ✓ validated
- Already applied prior: W8A8 fp8+alpha=0.999, PagedEvictKV page=64/LRU, ModConfig aux_loss=1e-8, CheckpointRecompute selective/block=512, FocalLoss gamma=4.93

### Reverted promotions (scoring artifacts — synthetic metrics didn't match real behavior)
- **MoE routing** (4/3/switch/6e-5): REVERTED to 8/2/aux_free/0.01. Evolution found top_k=1 scored higher (24.95 vs 24.87) but top_k=1 is a trivial solution (no ensemble). Scoring fixed: diversity penalty for top_k=1.
- **RoPE theta=10M**: REVERTED to 1M. Synthetic metric rewarded angle diversity, not attention quality. Scoring fixed: checkpoint compat penalty + frozen-dimension detection at long range.
- **Scheduler warmup=0**: REVERTED to warmup=500/20. Synthetic AUC rewarded no warmup, ignored training stability. Scoring fixed: stability penalty for zero warmup.
- **Label smoothing 0.29**: REVERTED to 0.1. Synthetic metric rewarded high smoothing for grad magnitude. Scoring fixed: smoothing penalty above 0.2 + focus ratio metric.

> Detailed evolution domain scoring fixes, DB rescores, focus profiles, and R&D round
> documentation have been moved to `docs/CHANGELOG.md`.

## Build & Test Commands

```powershell
# Run tests
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -m pytest tests/ --tb=short -q

# Verify model loads
$env:PYTHONPATH="D:\windsurf\ForgeAI"; D:\windsurf\ForgeAI\venv\Scripts\python.exe -c "from forge.model_loader import ConfigurableResearchLLM; print('OK')"

# Benchmark INT4 quantization
D:\windsurf\ForgeAI\venv\Scripts\python.exe D:\windsurf\ForgeAI\.devin\benchmark_int4.py
```

## Environment

- OS: Windows 11, GPU: RTX 5070 12GB
- Python venv: `D:\windsurf\ForgeAI\venv\`
- Key packages: torch, transformers, safetensors, bitsandbytes, pytest

## Historical Notes
R&D round documentation, refactor retrospectives, and historical notes
have been moved to `docs/CHANGELOG.md` (extracted 2026-09-09, critique F1).
