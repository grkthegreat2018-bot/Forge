# Bug Log

## 2026-08-11 â€” Self-play instability: model corruption, crashes, stalls

### Symptom
Self-play training was unstable: crashed on bad samples/epochs, stalled with
no detection, and silently corrupted the model after the first fine-tune epoch
(cascading garbage outputs on all subsequent epochs/validation).

### Root Causes

1. **LoRA merge corruption (CRITICAL)** â€” `_finetune_expert` merge loop
   assigned `merged` only inside the `if hasattr(layer, 'merge_and_unload')`
   branch but called `setattr(target, attr_name, merged)` unconditionally.
   In dense_bypass MoE mode, `experts[0]` was never LoRA-wrapped (only
   `shared` was), so `merged` retained `shared`'s last merged tensor and
   overwrote `experts[0]` weights with stale data.
   File: `research/training/self_play_expert_training.py` (~L1134)

2. **No per-topic/per-epoch exception isolation** â€” `main()` topic loop and
   `train_topic()` fine-tune call had no try/except. One bad sample/epoch
   killed the entire run, skipping all remaining topics.
   File: `research/training/self_play_expert_training.py` (L1745, L582)

3. **Cooperative stop sentinel never checked** â€” GUI wrote `STOP_REQUESTED`
   sentinel but training loop never polled it. Only abort path was
   SIGINT/SIGTERM handler, unreliable on Windows (taskkill doesn't deliver
   catchable SIGTERM â†’ no emergency checkpoint, process hangs).
   File: `research/training/self_play_expert_training.py`, `research/self_play/live_status.py`

4. **Heartbeat decoupled from training progress** â€” `LiveStatusWriter` daemon
   thread wrote `heartbeat.json` every interval regardless of training-loop
   progress. A hung loop still looked "alive" to the GUI's staleness check.
   File: `research/self_play/live_status.py`

### Resolution

1. **LoRA merge**: Guarded `setattr` â€” only call it when `merge_and_unload`
   was actually invoked. Unwrapped layers are left intact.

2. **Exception isolation**: Wrapped each topic in `main()` with try/except
   (log, cleanup VRAM, continue to next topic). Wrapped fine-tune call in
   `train_topic()` with try/except (log, `empty_cache`, `model.eval()`,
   skip to next epoch). Followed PyTorch forum guidance: clear graph
   references + `empty_cache` after exceptions to prevent OOM cascades.

3. **Cooperative stop**: Added `stop_requested()` method to
   `LiveStatusWriter` (checks `STOP_REQUESTED` sentinel next to
   `status.json`). Training loop polls at epoch boundaries, before each
   task batch, and at topic boundaries. TF Coordinator `should_stop()`-
   style pattern â€” reliable on Windows where signals aren't.

4. **Progress-coupled heartbeat**: Added `_last_progress_ts` to
   `LiveStatusWriter`, updated on every public API call
   (`task_started`, `round_done`, `task_done`, `set_phase`, `update`).
   Heartbeat thread checks stall age; if >120s since last progress,
   writes `{"ts": ..., "stalled": true, "stall_age_s": ...}`. GUI
   `EventsReader.heartbeat_stalled()` reads this flag; `SelfPlayPage`
   shows "STALLED" phase tag and warns the user the run is hung.

### Files Modified
- `research/training/self_play_expert_training.py` â€” Fixes 1, 2, 3
- `research/self_play/live_status.py` â€” Fixes 3, 4
- `forge_gui/api/events_reader.py` â€” Fix 4 (GUI-side reader)
- `forge_gui/pages/selfplay.py` â€” Fix 4 (GUI-side display)

### Research References
- PEFT `merge_and_unload` is not in-place; must assign result
  (huggingface/peft PR #2871, Issue #2032)
- PyTorch exception isolation: exceptions keep differentiable output alive
  in frame â†’ must `del` refs + `empty_cache` (discuss.pytorch.org #108619)
- TF Coordinator `request_stop()`/`should_stop()` pattern for cooperative
  shutdown (tensorflow.org API docs)
- PyTorch NCCL watchdog heartbeat pattern: monitor thread detects stale
  progress, escalates (pytorch/pytorch PR #112518)

## 2026-08-24 â€” OptimizerConfig metadata JSON serialization failure

### Symptom
ForgeEvolve `boot` run reported `OptimizerConfig: Object of type Tensor is not JSON serializable` and skipped the domain (56/57 valid). The error occurred during ForgeEvolve runs when discoveries were written to the SQLite DB, which serializes metadata via `json.dumps`.

### Root Cause
In `research/evolution/domains/training_domains.py`, `OptimizerConfig.evaluate()` line 69 stored `losses[-1]` (a CUDA tensor) directly into the `metadata` dict instead of a Python float. All other domains in the file (SchedulerConfig, LossConfig, MuonConfig) correctly wrap tensor values with `float(...)` or `.item()`. The discoveries DB path serializes metadata via `json.dumps`, which raises `TypeError` on torch.Tensor.

### Resolution
Changed line 69 from `"final_loss": losses[-1]` to `"final_loss": float(losses[-1].item())` in `research/evolution/domains/training_domains.py`.

### Files Modified
- `research/evolution/domains/training_domains.py` â€” Fix (line 69)
- `tests/unit/test_evolution_domains.py` â€” Regression tests (new file)

### Regression Tests Added
- `test_optimizer_config_metadata_final_loss_is_float` â€” Direct regression check that final_loss is a float and metadata round-trips through json.dumps
- `test_metadata_is_json_serializable[name]` â€” Parametrized over all 57 registered domains, verifying each domain's evaluate() returns JSON-serializable metadata, score, and behavioral values

### Verification
- All 58 new tests pass
- Full unit suite (636 tests) passes
- Re-ran `run_evolve.py --profile boot --domains OptimizerConfig` â†’ best=15.99, 29 discoveries, no error

## 2026-08-24 â€” CrossLayerKV negative recon_err + engine metadata mismatch

### Symptom
ForgeEvolve `boot` run showed two suspicious results:
1. `CrossLayerKV` scored 218,453 â€” wildly higher than any other domain.
   `recon_err` in metadata was **-233.1** (negative reconstruction error).
2. `XQuantKV` best config said `recomputation_ratio=1.0, quant_bits=4` but
   metadata said `recomputation_ratio=0.0, quant_bits=8` â€” config and
   metadata disagreed.

### Root Causes

1. **CrossLayerKV recon_err (kv_domains.py L485)**: The reconstruction
   error formula was:
   ``python
   (group[-n:] - shared.expand(...).norm()).mean()
   ``
   `.norm()` was applied to `shared.expand(...)` (a scalar), not to the
   difference `(group - shared)`. This subtracted a large scalar from each
   group element, producing large negative values. The score formula
   `param_reduction * 100 - recon_err * 500` then turned negative recon_err
   into a massive positive bonus (+116k).

2. **Engine metadata mismatch (engine.py L547)**: When saving discoveries
   to the DB, the engine used `self.all_results[-1].get("metadata", {})` â€”
   always the **last** result's metadata â€” for every discovery in the
   generation. If configs A and B both entered the archive, both got B's
   metadata.

### Resolution

1. **CrossLayerKV**: Replaced the broken formula with proper relative L2
   reconstruction error:
   ``python
   diff = target - recon
   recon_err += float(diff.norm().item() / (target.norm().item() + 1e-8))
   ``
   This is always non-negative and in [0, ~2] range.

2. **Engine**: Added `metadata_list` alongside the existing `scores` and
   `behavioral_list`, populated in the same `zip(configs, results)` loop.
   The discovery loop now zips `metadata_list` too, ensuring each
   discovery gets its own metadata.

### Files Modified
- `research/evolution/domains/kv_domains.py` â€” CrossLayerKV fix (L483-489)
- `research/evolution/engine.py` â€” metadata_list fix (L449, L508-518, L534-548)
- `tests/unit/test_evolution_domains.py` â€” regression tests added

### Regression Tests Added
- `test_cross_layer_kv_recon_err_non_negative` â€” parametrized over all
  mode/ratio/n_groups combinations, asserts recon_err >= 0 and score < 1000
- `test_xquant_kv_metadata_matches_config` â€” parametrized over all
  ratio/bits/interval combinations, asserts metadata matches config
- `test_engine_discovery_metadata_matches_config` â€” runs a full
  ForgeEvolve engine on SyntheticDomain, verifies all_results have
  consistent metadata
- `test_engine_metadata_list_aligned_with_configs` â€” verifies
  all_results ordering and metadata presence

### Verification
- All 160 domain tests pass
- Full unit suite (738 tests) passes
- CrossLayerKV: all 160 mode/ratio/groups combos produce recon_err >= 0
- XQuantKV: all 40 ratio/bits/interval combos produce matching metadata

## 2026-08-24 â€” JSON round-trip type coercion (4 domains)

### Symptom
After enabling canonical warm-start (loading past configs from the DB),
4 domains crashed with type errors:
- `GroupQuant`: `view(): argument 'size' failed to unpack`
- `KvZipKV`: `'<' not supported between instances of 'int' and 'str'`
- `RotorQuantKV`: `'str' object cannot be interpreted as an integer`
- `SyntheticDomain`: `ufunc 'subtract' did not contain a loop with signature matching types (dtype('<U...'))`

### Root Cause
Configs saved to the DB via `json.dumps()` and loaded back via
`json.loads()` can lose type information â€” integer values become strings.
The domain `evaluate()` methods assumed correct types without coercion.
This only manifested when warm-started generators produced configs that
were saved to the DB and re-evaluated on the next run.

### Resolution
Added `int()` / `float()` / `np.array(..., dtype=np.float64)` coercion at
the point of use in each affected domain:
- `quant_domains.py` GroupQuant: `int(config["group_size"])`, `int(config["n_bits"])`
- `kv_domains.py` RotorQuantKV: `int(c["n_rotations"])`, `int(c["quant_bits"])`
- `kv_domains.py` KvZipKV: `int(c["codebook_size"])`, `int(c["n_iter"])`
- `synthetic.py` SyntheticDomain: `np.array(config["x"], dtype=np.float64)`

### Files Modified
- `research/evolution/domains/quant_domains.py` â€” GroupQuant fix
- `research/evolution/domains/kv_domains.py` â€” RotorQuantKV + KvZipKV fixes
- `research/evolution/domains/synthetic.py` â€” SyntheticDomain fix

### Verification
- All 4 domains now handle string-typed configs correctly
- Full unit suite (747 tests) passes

## 2026-09-02 â€” Agent/Chat tool calls rendered as plain text (never executed)

### Symptom
In the Forge GUI Agent page and Chat Studio agent loop, the model's tool
calls appeared as plain text in the output and were never executed. The
agent loop terminated after one round with no tools invoked.

### Root Cause
`ForgeEngine.generate()` defaults to `skip_special_tokens=True`, which
strips the `<|tool_call_start|>` (id 10) and `<|tool_call_end|>` (id 11)
special tokens from the decoded output. The agent runner and chat agent
loop both called `engine.generate(...)` without overriding this default,
so `qwen_parse_tool_calls` could not find the marker-wrapped tool calls
(its primary parse path). It fell back to bare-JSON detection, which
fails on a model trained to emit the marker-wrapped format, so the raw
text â€” including the JSON tool-call body â€” was returned as plain "musing"
text and no tools were executed.

The engine docstring documents the fix: *"Set to False for tool-call
parsing (preserves <|tool_call_start|>/<|tool_call_end|> markers)."* â€”
but the GUI call sites did not pass it.

### Resolution
- `forge_gui/api/agent_runner.py` (line ~150): added
  `skip_special_tokens=False` to the `engine.generate(...)` call.
- `forge_gui/pages/chat.py` (line ~150, agent-loop path): added
  `skip_special_tokens=False` to the `engine.generate(...)` call.
- `forge_gui/api/sub_agent.py` was checked â€” it does not parse tool
  calls (returns raw text), so no change needed.

### Verification
- `tests/unit/test_gui_agent_tools.py` (14) + `test_gui_chat_store.py`
  (8) pass: 22 passed, 0 failed.


## 2026-09-13 - ForgeLM V2 Jamba: decorrelated outputs (use_rope), bf16 SSM scan drift, EOS blind spot, Quamba2+LoRA dtype crashes

### Symptom
Booting ForgeLM V2 (Jamba-Reasoning-3B port) produced fluent but semantically
incoherent text: think-traces hallucinated alternate user prompts, never
converged to answers, and ran past <|im_end|> to the token cap. Full-model
parity vs HF JambaForCausalLM showed logit cosine ~0.0 (per-layer probe:
divergence jumped at attention blocks 7/21). Separately, Quamba2+LoRA crashed
in add_lora_adapters (no .weight on Quamba2Linear) and warmup hit
fp32-vs-bf16 matmul errors.

### Root Cause
- forgelm_v2 preset set use_rope=True, but Jamba attention layers carry NO
  positional encoding (HF JambaAttention has no rotary call - Mamba supplies
  position). RoPE-scrambled Q/K decorrelated the only 2 attention layers.
- MambaLayer._selective_scan_ref ran the SSM recurrence in bf16; HF keeps the
  state and discretization in fp32 (~8%/element/layer drift, compounded over
  26 layers). Same issue in Quamba2Block._selective_scan_ref and _rmsnorm.
- StandardDecoding hardcoded eos_set={7,151643,151645} (LFM/Qwen only),
  ignoring engine _DEFAULT_EOS_TOKEN_IDS - Jamba 2/519 never stopped decode.
  eos_set.add(list) would also crash on HF list-valued eos_token_id.
- add_lora_adapters only probed weight/weight_packed/dense_packed; Quamba2
  uses weight_int4_packed. Quamba2Linear.forward never applied lora_adapter.
- Quamba2Block cast the SSM core to fp16, promoting activations to fp32 and
  breaking downstream bf16 matmuls (out_proj/LoRA).

### Resolution
- forge/config.py: forgelm_v2 use_rope=False (Jamba attention is
  position-agnostic; rope_base kept for API shape only).
- mamba_probe.py + quamba2.py: fp32 SSM recurrence (h/A_bar/B_bar/x in fp32,
  per-step output cast to activation dtype like HF), fp32 RMSNorm.
- quamba2.py: Quamba2Linear.forward applies lora_adapter with dtype
  alignment; block output cast back to model dtype.
- bitnet_lora.py: device detection falls back to weight_int4_packed.
- decoding.py: eos_set now includes Jamba {2,519} at both StandardDecoding
  and speculative fallback sites; list-valued eos handled.
- forge_engine.py: AVMP/VirtualTensorPool log lines divided bytes by 1024^2
  labeled GB -> now GiB (printed 12226GB for a 12GB GPU).

### Files Modified
- forge/config.py, forge/keys/architecture/mamba_probe.py,
  forge/quant/quamba2.py, forge/training/bitnet_lora.py,
  forge/engine/decoding.py, forge/engine/forge_engine.py,
  tests/unit/test_forgelm_v2_jamba.py, tests/unit/test_quamba2.py

### Verification
- Layer parity vs HF Jamba (bf16, GPU): cos >= 0.9993 every block, logit
  max diff 0.44, identical argmax (was cos ~0.0).
- 61 unit tests pass (test_forgelm_v2_jamba incl. 3 new EOS/RoPE
  regressions, test_quamba2 incl. LoRA coverage, test_r39_sparse_grammar).
- End-to-end boot: correct rate arithmetic, correct Python-semantics answer
  ([36,81]), well-formed <tool_call> JSON, clean <|im_end|> stop.

## 2026-09-13 â€” GUI boot blocked: VRAM pre-flight demanded ~16GB for ForgeLM V2

### Symptom
Booting ForgeLM V2 from the GUI always failed with "not enough free VRAM:
~11.5 GB free, loading this checkpoint needs ~16.0 GB" on the 12GB RTX 5070.

### Root Cause
`_LoadWorker` in forge_gui/api/engine_runtime.py estimated the load
requirement as `ckpt_size * 2.5` (~16GB for the 6.39GB checkpoint) while the
engine's own fast-path gate in ForgeEngine.from_checkpoint uses
`ckpt_size * 1.3` (~8.3GB). The GUI gate could never pass on 12GB even though
the validated mamba_hybrid profile peaks at ~7.5GB.

### Resolution
- engine_common.py: new `_FAST_LOAD_VRAM_HEADROOM = 1.3` +
  `_fast_load_vram_required(ckpt_size)` shared by both checks so the GUI and
  engine thresholds can never drift apart again.
- engine_checkpoints.py + engine_runtime.py: both call the shared helper.

### Files Modified
- forge/engine/engine_common.py, forge/engine/engine_checkpoints.py,
  forge_gui/api/engine_runtime.py, tests/unit/test_engine_checkpoint_guard.py

### Verification
- 12/12 tests pass in test_engine_checkpoint_guard.py (3 new: headroom value,
  preflight blocks below gate, preflight passes and reaches from_checkpoint).
- End-to-end load via from_checkpoint: 80s, peak 7.37GB allocated, 4.0GB
  free; auto-activated rotorquant 4-bit KV, QuaRot-KV, Quamba2 W4A8 (26 SSM
  blocks), AVMP, VirtualTensorPool, prefix cache, CacheBlend, chunked
  prefill, adaptive+suffix spec decode, FASER, SeqSplit, ReplaySSM.

### Follow-up (same boot session): chat crash on `thinking` kwarg
- **Symptom:** every Chat Studio send failed with
  `TypeError: render_messages_for_config() got an unexpected keyword
  argument 'thinking'`.
- **Root cause:** the GUI wrapper `forge_gui/api/chat_render.py` never
  forwarded `thinking` to the engine renderer
  (`forge/self_play/discovery/qwen_adapter.py`, which accepts it).
- **Resolution:** wrapper now takes `thinking: bool = True` and forwards
  it. Regression test in tests/unit/test_gui_chat_stream.py
  (TestChatRenderWrapper).

## 2026-09-16 â€” Mixin split dropped heavy imports: NameError at class-def / call time

### Symptom
`ForgeEngine` was unimportable in the working tree: `NameError: name
'CacheBlend' is not defined` at class-definition time in
engine_generation.py. After that was repaired, runtime probing plus the
evolutionary-merge suite surfaced further `NameError`s at call time:
`build_kv_cache`, `build_decoding` (engine_activation.py),
`build_health_report` (engine_diagnostics.py), and
`unpack_output_with_kv` (engine_sessions.py).

### Root Cause
The uncommitted monolith-to-mixin split of forge_engine.py moved methods
into engine_*.py files but dropped the heavy imports the monolith relied
on (engine_common.py intentionally stays import-light). Annotations and
call sites referenced symbols that were never imported into the new
modules.

### Resolution
Restored per-mixin imports for every symbol each file actually uses:
- engine_generation.py: CacheBlend, StandardDecoding, and other
  generation-path deps.
- forge_engine.py: StandardDecoding and core-path deps.
- engine_activation.py: build_kv_cache (kv_backend), build_decoding
  (decoding), plus activation-path deps.
- engine_checkpoints.py / engine_diagnostics.py / engine_lifecycle.py /
  engine_merging.py: their respective call-site deps (incl.
  build_health_report in engine_diagnostics).
- engine_sessions.py: unpack_output_with_kv from forge.model_loader.
- SemanticKVAnchors imported from prefix_cache; GGUFInfo from
  forge_loader; CREATIVE_SAMPLING resolved as class attribute.

### Files Modified
- forge/engine/forge_engine.py, engine_activation.py,
  engine_checkpoints.py, engine_diagnostics.py, engine_generation.py,
  engine_lifecycle.py, engine_merging.py, engine_sessions.py

### Verification
- `import forge.engine.forge_engine` succeeds; end-to-end `decide()`
  smoke test runs on CPU.
- tests/unit/test_evolutionary_merge.py: 8 errors -> all pass.
- 157 passed across test_evolutionary_merge, test_decide,
  test_forge_engine_fixes, test_engine_concurrency_fixes,
  test_chat_features.

### Follow-up (same session): Windows MAX_PATH triton cache crash
- **Symptom:** `test_model_loader.py::test_compile_for_inference_gpu`
  failed with `InductorError: FileNotFoundError` writing
  `triton_poi_fused_..._10.source` into the torchinductor cache.
- **Root cause:** the triton cache `put()` path is 268 chars â€”
  `%TEMP%/torchinductor_tmk68/triton/0/<48-char hash>/tmp.<id>/<138-char
  kernel name>.source` exceeds the 260-char Windows MAX_PATH, so
  `open(temp_path)` fails even though `os.makedirs` succeeded.
- **Resolution:** `tests/conftest.py` sets
  `TORCHINDUCTOR_CACHE_DIR=%TEMP%/ti` (short base) on Windows before
  torch import. Test passes; no repo code change needed.

### Follow-up (System One work): silent random-weight load + reasoning-template prompt

- **Symptom:** `load_default_model("forgelm_v2")` produced a model that
  generated pure gibberish on every prompt (flat ~2.0 logits, nonsense
  tokens). `decide()` on it returned near-uniform probabilities.
- **Root cause (two bugs):**
  1. `load_default_model` documented "checkpoint_path defaults to config
     default" but passed `None` through â€” `build_model_fast` then skipped
     the weight-load branch and printed only a generic `Total:` line.
     **Silent random weights.** (The pending-bug list had flagged this.)
  2. `decide.py` built a plain ChatML prompt ending at `assistant\n` â€”
     ForgeLM V2 is a reasoning model: the canonical template (see
     `qwen_adapter.render_messages_for_config`) primes
     `assistant\n<think>\n`, so the first generated token is reasoning
     text and answer tokens sat ~15 logits below the top.
- **Resolution:**
  - `model_loader.py`: `checkpoint_path=None` now resolves to
    `V2_CHECKPOINT` when it exists; `build_model_fast` prints a loud
    "no checkpoint_path â€” RANDOM weights" warning when it doesn't.
  - `decide.py`: prompt now uses the canonical think-hint injection and
    ends `assistant\n<think>\n</think>\nAnswer:` (empty think block +
    explicit answer field); candidates are space-prefixed
    (" yes"/" no", " finance", " 3") matching what the model emits
    after `Answer:`.
- **Verification:** real ForgeLM V2 via `ForgeEngine.from_checkpoint`
  + `activate_optimal` (GUI path): sector choices 0.99/0.95 correct,
  sentiment/hype scores land on the right level, noul directions all
  correct (0.65-0.79 true / 0.13-0.15 false â€” direction right,
  calibration soft as expected pre-Tier-1). 31 decide tests pass.

## 2026-09-19 ï¿½ KV-cache eviction mask overflow + min-k dead sensitivity

### Symptom
1. `SnapKVCache._evict` (and the new `FillerKVCache._evict` modeled on it)
   crashed with `IndexError: The shape of the mask [total] does not match
   the indexed tensor` whenever the K/V buffer had grown beyond the live
   sequence length ï¿½ i.e. any single append with `T > 1` crossing the
   capacity boundary, or after a buffer-doubling growth.
2. `_min_k_filter` (`engine_common.py`) and `_min_k_filter_logits`
   (`decoding.py`) computed `weighted_diffs > sensitivity * max_decay` and
   discarded the result; the `sensitivity` argument had no effect ï¿½ every
   call truncated at the global argmax regardless of the configured value.

### Root Cause
1. The eviction `keep` mask is sized `total` (= live seq_len) but was
   applied to the full buffer axis (capacity > total after growth slack).
   Trigger: `cache.append(k, v)` with T=20 on a 12-capacity cache
   allocated 32 slots; `k_cache[:, :, keep]` then fails to broadcast.
2. Dead expression ï¿½ the boolean cliff-mask was computed and thrown away;
   `cliff_pos = weighted_diffs.argmax(...)` ignored the threshold.

### Resolution
1. Mask only the first `total` buffer slots:
   `k_cache[:, :, :total][:, :, keep]` (same for v/scores/filler flags) in
   both `snapkv.py::_evict` and `filler_kv.py::_evict`.
2. Truncate at the *rightmost* position whose weighted decay exceeds
   `sensitivity * max_decay` (higher sensitivity ? higher threshold ?
   earlier rightmost cliff ? more aggressive truncation, matching the
   docstring); fall back to the argmax when no position exceeds
   (sensitivity=1 edge). Applied identically in both copies.

### Verification
`test_snapkv_multi_token_overflow_regression` (T=20 single-shot append ?
seq_len==12, no crash) and `test_min_k_sensitivity_controls_truncation`
(sens 0.1 keeps 5 tokens vs sens 0.9 keeps 1 on a two-cliff distribution).
Pre-existing `test_min_k_*` cases still pass (single-cliff inputs give
identical results). Full unit suite: 2607 passed.

### Follow-up (R50 feature smoke): DoLa inference-tensor crash

- **Symptom:** `DoLaDecoding.generate` crashed on the first sampled step
  with `RuntimeError: Inference tensors cannot be saved for backward`
  inside `ln_f`/`head` ï¿½ only on real models, not in unit tests (stubs
  had requires_grad=False params).
- **Root cause:** `_contrast_logits`/`_early_logits` ran outside any
  no-grad scope while their `hidden_list` inputs were inference tensors
  produced by the cached prefill/decode forwards. Model params
  (`requires_grad=True`) tried to save them for backward.
- **Resolution:** `@torch.no_grad()` on `_contrast_logits` ï¿½ `no_grad`
  (not `inference_mode`) so the returned logits stay normal tensors the
  sampling chain can mutate in place.
- **Verification:** `test_dola_contrast_accepts_inference_tensors`
  (inference-mode inputs + requires_grad params + in-place mutation on
  the output); GPU smoke `scripts/smoke_r50_gpu.py` ï¿½ DoLa generates on
  real ForgeLM V2 ("Paris and the currency the euro" vs greedy
  "Paris. The capital of Germany is Berlin").


## 2026-09-20 - Cleanup sweep: three broken import paths found + fixed

### Symptom
1. `engine_activation.block_reconstruct` checkpoint-reload branch was
   unreachable-by-accident: read `self._checkpoint_path` (never set -
   the real attr is `self.checkpoint_path`) AND imported
   `forge.engine.model_loader` (module does not exist) AND called
   `load_default_model` as a `ModelLoader` method (it is a module
   function in the `forge.model_loader` facade).
2. `build_decoding('speculative')` mapped to `SpeculativeDecoding`,
   whose `generate()` imported `research.speculative_decode` - deleted
   in the research->forge migration, so the strategy crashed on first call.
3. Four production keys (`pit_key`, `lerope_key`, `attn_residual_key`,
   `mhc_key`) import `forge.keys.safety.safe_apply` for their
   `safe=True` path, but `safety.py` had been moved to
   `tests/fixtures/keys/` - the safe path raised ImportError.

### Root Cause
Refactor leftovers: file moves (safety.py -> fixtures, sandbox ->
training/runners, research/ -> forge/) updated the movers but not all
import sites; dormant paths had no test coverage so the breakage stayed
latent.

### Resolution
1. `block_reconstruct` now uses `self.checkpoint_path` +
   `self.config` and calls the canonical
   `ModelLoader.build_model_fast(cfg, checkpoint_path=...)`.
2. Removed `SpeculativeDecoding`; `build_decoding('speculative')`
   now maps to `ExternalDraftSpeculativeDecoding` (same draft_model
   interface) - the strategy name works again instead of crashing.
3. `safety.py` moved back to `forge/keys/safety.py` (production code
   imported by production keys); the 4 `_load_module` test paths in
   `test_key_transforms.py` updated.
4. Deleted unwired `airllm_streamer.py` (its one dependency,
   `forge.keys.moe.airllm_key.AirLLMKey`, no longer exists; the real
   streaming fallback is `engine_checkpoints._load_streaming`) and
   pruned 19 stale entries from `vast_connector.CRITICAL_SOURCE_FILES`
   / `CRITICAL_INIT_FILES`.

### Verification
`git grep` confirms zero references to removed modules;
`vast_connector` manifest now resolves 73/73 paths; unit suite run in
the same cleanup commit (see git log).

## 2026-09-20 - Chat gating/tool loop: probes OOD on tools block + continuations, reasoning leaked into replies, junk news results

### Symptom
1. Bare greetings (`Hello`) burned ~150 think tokens parroting the master
   prompt, then emitted a bland `Answer:` — route probe scored p_easy
   ~0.10 on prompts that should route direct.
2. Tool turns looped: conv-exit fired on the tool-continuation round, the
   forced `Answer:` made the model restate its plan and re-emit the SAME
   web_search 5x — never synthesized.
3. Direct-routed tool turns persisted think-voice musing as the visible
   reply; a mid-think `<tool_call>` left the think block unclosed so
   reasoning leaked into the persisted reply (`thinking escapes the tool
   call`); after the cap fired the model re-opened `<think>` and
   re-looped; literal `</tool_call>`/`<tool_response>` emitted as
   ordinary tokens bypassed the id-based EOS and the model hallucinated
   its own tool results.
4. `check the news` returned portal homepages (cnn.com/) + y.js ad
   redirectors — zero real headlines; `web_fetch` stripped all markup so
   link names/urls were destroyed and long pages had no paging.

### Root Cause
1. `_route_p_easy` scored the production render including the ~17-schema
   `<tools>` block; gate_r probes were trained tools-free — the schema
   block collapses h_mean, dragging p_easy ~0.25 under the 0.2 threshold
   (`Hello`: 0.355 tools-free vs 0.098 with tools). gate_r also has no
   chit-chat class.
2. Route probe + conv observer ran on EVERY round — both are OOD on
   tool-continuation (synthesis) turns.
3. No repeat-call detection; direct-path pre-call musing persisted
   verbatim; `<tool_call>` (531) unmasked while think open (trained
   order: think ? </think> ? text ? call); generated `<think>` (541)
   unbanned; no `reasoning_content` split — the template supports it
   but the loop never populated it.
4. DDG HTML returns portal homepages + ad redirectors for news queries —
   a data problem, not a gating/restriction problem; single-engine search
   had no fallback when an endpoint bot-walls; `html_to_text` flattened
   pages to one line with zero link/structure info.

### Resolution
- `chat_loop.py`: tools-free probe render (generation keeps tools
  render); `_is_trivial_turn` regex bypass; `fresh_turn` gate (probes
  only when conv[-1] is user); `_call_sig` repeat guard drops defs;
  `_think_cap_processor` now masks 531 while the think block is open,
  bans 541/518/539/540 in generated text always, and bans 531+532 when
  `tool_calls_allowed=False`; rolling 24-char literal-marker tail stop;
  `_split_reasoning` ? `msg[reasoning_content]` (re-rendered into the
  next tool round by the template, shown as ThinkCard, dropped on the
  next user turn); `_strip_answer_anchor`; tool results truncated at
  4000 chars before re-render.
- `agent_loop.py`: reuses `_think_cap_processor` with a 384-token
  think budget + bare `\n</think>\n` close (a capped think can still
  proceed to a call); same reasoning split; empty-body fallback so an
  early EOS inside think still yields the run's final text.
- `routes/chat.py`: persists `reasoning_content`; master prompt now
  gets real `tools_enabled`/`thinking` flags.
- `master_prompt.py`: enforceable blunt/brief contract (lead with the
  answer, bullets over paragraphs, match length to question) + actionable
  tool-use policy instead of soft guidelines.
- `web_primitives.py`: `fetch_url` reads any http(s) URL — bare
  domains auto-https, redirects followed (`final_url`), HTML ?
  markdown-ish text via a streaming `HTMLParser` (headings/lists/links
  kept; `[label](url)` inline + deduped `links` list; script/style/
  nav chrome dropped), non-HTML bodies pass through raw, `offset`/
  `next_offset` paging; search cascade DDG HTML ? DDG Lite ? Bing ?
  Google News RSS.
- `web_tools.py`: `web_fetch` clamp raised to 16000 + `offset` arg;
  descriptions rewritten so the model knows it can open ANY url and
  follow links.
- `streamParse.ts`/`Chat.tsx`: `implicitThink` marks the lead
  segment as thinking while the prompt-opened block is unclosed
  (suppressed when gate routed direct); persisted `reasoning_content`
  renders as a ThinkCard; `toolResultSummary` gives JSON-aware one-line
  tool previews.

### Files Modified
- `forge_gui_server/services/chat_loop.py`,
  `forge_gui_server/services/agent_loop.py`,
  `forge_gui_server/routes/chat.py`, `forge_gui/api/master_prompt.py`,
  `forge_gui/api/web_tools.py`, `forge/web_primitives.py`,
  `forge_ui/src/lib/streamParse.ts`, `forge_ui/src/pages/Chat.tsx`,
  `forge_ui/src/components/chat/blocks.tsx`,
  `tests/unit/test_gated.py`, `tests/unit/test_gui_web_tools.py`,
  `tests/unit/test_chat_features.py`

### Verification
- `pytest tests/unit/test_gui_web_tools.py test_gated.py
  test_gui_server.py test_gui_chat_store.py test_gui_tool_harness.py
  test_jamba_tool_format.py test_chat_features.py test_thinking_pipeline.py
  test_gui_agent_tools.py test_gui_mcp_client.py test_agent_safety_backup.py`
  — 281 passed.
- `npm run build` (tsc + vite) clean.

## 2026-09-21 — RSI loop: SFT trained a fake chat format + reasoning leaked into exported code

### Symptom
RSI (self-play) epochs produced little/no improvement on the eval gate:
the model stayed prone to overthinking and weak at short, simple answers
even though those are exactly what the loop should be sharpening.

### Root Cause
1. **SFT rendered a non-existent chat format (CRITICAL)** — `sft_train.py`
   emitted `<|startofsegment|>`/`<|endofsegment|>` + `<|tool_call_start|>`/
   `<|tool_call_end|>` markers. None exist in the ForgeLM V2 (Jamba
   Reasoning 3B) tokenizer — they tokenize as literal text fragments.
   Every SFT epoch trained completions conditioned on a format the model
   never sees at inference (`<|im_start|>` ChatML + `<tool_call>` ids
   531/532). The module docstring already claimed im_start markers —
   the code was left stale through the Jamba migration.
2. **Reasoning leaked into training data** — the solver's post-fence
   musing was captured as `#` comments/docstrings inside exported
   solutions (observed in `azr_epoch1.jsonl`: 40+ comment lines of
   second-guessing before the actual function). SFT literally taught
   "write a wall of reasoning comments before the answer".
3. **No direct-answer training signal** — epoch data was 100% code
   tasks; nothing taught "easy prompt ? short direct answer", so the
   `concise` eval category could never move.
4. **First-pass selection** exported the first passing attempt, not the
   best one; GRPO groups carried raw binary rewards, so all-pass groups
   had zero advantage (loss 0).

### Resolution
1. `sft_train.py` renders via the canonical
   `chat_template.apply_chat_template` (`<|im_start|>`/`<|im_end|>`,
   `<tool_call>`/`<tool_response>`, thinking=False direct answers) —
   verified real special ids 518/519/531/532/539 in encoded output.
2. `infinite_curriculum._assemble_solution` AST-cleans every reply
   (`_clean_code` + `_DocstringStripper`): comments + docstrings dropped,
   code canonicalized; unparseable bodies fall back to raw text so
   verification still fails them naturally. Applies to SFT export AND
   GRPO group completions.
3. `infinite_loop._concise_qa_pairs` mixes `concise_qa_per_epoch`
   (default 12) verified short-answer pairs per epoch — computed answers
   (math/string/list/logic/conversions) + curated facts, ~half with
   explicit brevity hints, half bare so brevity becomes the default.
4. Export picks the SHORTEST passing attempt; `_concise_adjusted_rewards`
   adds `grpo_concise_bonus` (default 0.1) length-efficiency shaping —
   correctness still dominates; all-pass groups regain nonzero advantage.

### Files Modified
- `forge/training/runners/sft_train.py`,
  `forge/self_play/infinite_curriculum.py`,
  `forge/self_play/infinite_loop.py`,
  `tests/unit/test_grpo_rsi_mode.py`, `AGENTS.md`

### Verification
- `pytest tests/unit/test_grpo_rsi_mode.py test_curriculum_engine_gen.py
  test_sft_data_loader.py test_thinking_pipeline.py
  test_jamba_tool_format.py test_gui_chat_store.py test_gated.py`
  ? 163 passed (38 new: canonical render, code cleaning, QA stream,
  concise rewards).
- Tokenizer check: rendered examples now encode ids 518/519/531/532/539
  (previously literal text).

## 2026-10-08 — Continuation-forward causal mask + prefill-logits VRAM spike + decode-capture corruption (Hermes-3/llama31_8b)

### Symptom
"VRAM spikes when prompting, response corrupted" during the Hermes-3 8B
(`llama31_8b`) Local_To_Chub bring-up: every 2nd+ chat request sharing the
>=32-token prefix key (same card/system prompt each turn) generated corrupted
text; ~512MB bf16 transient VRAM spike on every prefill; corrupted
continuations after max_new_tokens cutoffs (finish_sentence=True default);
ghost context on prefix-cache hits for hybrid models; cross-request state
leakage.

### Root Causes
1. **SDPA is_causal top-left on continuation forwards (CRITICAL)** —
   `GroupedQueryAttention` called `flash_attention(is_causal=True)` whenever
   T>1, even with a non-empty past_kv. PyTorch SDPA anchors the causal mask
   TOP-LEFT on non-square (T<S) shapes, so a T>1 continuation forward
   (prefix-cache suffix, chunked prefill chunk>=2, spec verify) masked out
   the ENTIRE cached prefix (verified empirically T=3,S=7). Every chat
   message after the first hit the >=32-token prefix key and generated
   corrupted text. Same defect in `fused_qk_norm_rope_cache.py`.
   File: `forge/model/layers.py` (~L348), `forge/engine/attention/fused_qk_norm_rope_cache.py` (~L295)

2. **Full [1,T,V] prefill logits** — every DecodingStrategy prefilled via
   `model(ids, use_cache=True)` without `logits_last_only` (only the last
   position is ever read): ~512MB bf16 transient at 4k tokens / V=128256.
   File: `forge/engine/decoding.py` (x7 sites), `forge/engine/prefix_cache.py` (suffix forward)

3. **_finish_to_stop re-fed the last emitted token** — the decode loop
   already consumed it into past_kv; re-feeding appended a duplicate KV
   position and corrupted the <=32-token continuation tail. Fired whenever
   output hit max_new_tokens without EOS (finish_sentence=True default).
   File: `forge/engine/engine_generation.py` (~L1953)

4. **Prefix cache stored end-of-generation KV under a prompt-only key** —
   `cache_prompt_prefix` sliced `model._forge_last_kv` (prompt+generated).
   Tuple entries slice fine, but dict recurrent (mamba/SSM) entries can't be
   sliced, so generated tokens leaked into a prompt-keyed entry -> ghost
   context on hybrid-model hits.
   File: `forge/engine/prefix_cache.py` (~L263)

5. **Stale decode captures across requests** — `_forge_last_kv` was never
   cleared; strategies that don't set it (NGram/ExternalDraft/Medusa/DSpark/
   Eagle3/MTPSelfSpec/SelfSpecSparse) left the previous request's KV visible
   to `_cache_prompt_prefix` and `_finish_to_stop` (poisoned entries /
   continued a different sequence).
   File: `forge/engine/engine_generation.py` (~L301)

### Resolution
1. Explicit bottom-right `_causal_mask(T, total_len, total_len-T)` whenever
   total_len > T; `is_causal=True` only for the square (total_len == T)
   case. Applied in layers.py and fused_qk_norm_rope_cache.py.
2. `logits_last_only=True` on every prefill-only / last-position-only
   forward: decoding.py x7, prefix_cache.py suffix, kv/cacheblend.py x2,
   prefill/__init__.py x3, engine_generation.py.
3. Decode loops stash `_forge_last_logits` alongside `_forge_last_kv`
   (decoding.py x3, engine_generation._decode_tokens); `_finish_to_stop`
   uses them (zero extra forwards), falls back to `_prefill(output_ids)`
   when absent — never re-feeds the last token.
4. `_forge_prefill_kv` captured at the prompt boundary (StandardDecoding,
   engine `_prefill`, prefix-cache hit path); `cache_prompt_prefix` prefers
   it, so stored dict state is at the true boundary.
5. `_generate_impl` clears `_forge_last_kv`/`_forge_last_logits`/
   `_forge_prefill_kv` at request start.

### Known remaining issues (confirmed by audit, logged — outside this path)
- Spec strategies (NGram/ExternalDraft/MTP/SelfSpecSparse) never roll back
  KV on reject -> corrupt past_kv (decoding.py:338/446); `uno.py`/`dspark.py`
  dormant/broken paths (singular kwarg TypeError, no snapshot/restore).
- Triton conv patch (triton_conv.py:194-218) ignores `_conv_state_prefix`
  on T>1 continuations (hybrid models only).
- beacon/anchor/metakv compaction + RoPE offset drift (past_len recomputed
  from compacted cache len; only when a non-rotorquant strategy selected).
- Per-token torch.cat KV append: O(T) transient per decode step; the real
  per-layer `presents` cache is unbudgeted (rotorquant strategy object is
  instantiated but never wired into the model's forward).

### Files Modified
- `forge/model/layers.py`,
  `forge/engine/attention/fused_qk_norm_rope_cache.py`,
  `forge/engine/decoding.py`, `forge/engine/prefix_cache.py`,
  `forge/engine/kv/cacheblend.py`, `forge/engine/prefill/__init__.py`,
  `forge/engine/engine_generation.py`
- `tests/unit/test_forge_engine_fixes.py` — regression tests
- `scripts/test_hermes_w4_generate.py` — real-model verify harness

### Verification
- `pytest tests/unit/test_forge_engine_fixes.py tests/unit/test_llama31_w4.py
  tests/unit/test_rd14_speedup.py tests/unit/test_engine_concurrency_fixes.py
  tests/unit/test_replay_ssm.py tests/unit/test_r60_kv.py` — all pass
  (145, incl. 9 new regression tests).
- TestContinuationForwardCausalMask (cpu+cuda parametrized): continuation
  logits match single-pass prefill <1e-4 on both devices.
- Real-model: `scripts/test_hermes_w4_generate.py` on RTX 5070 12GB —
  Hermes-3 8B loads W4 (~5.6 GiB resident); prefix-cache HIT output equals
  no-cache reference; peak transient ~0.5-0.8 GiB (was GBs at long prompts).

---

## 2026-10-08 - VRAM growth to limit + empty/garbled output on Chub path

### Symptom
"context still grows mem to limit. model still horrid in output,
nonsensical" - through Local_To_Chub (Kobold mode): VRAM climbed ~1.5 GB
across ~10 requests until the server OOM'd and went unreachable; responses
returned content:null or pseudo-German word salad with mojibake.

### Root causes
1. Kobold routing wrapped transcripts in a ChatML user turn.
   chub_forge_proxy.kobold_to_chat sent Chub's pre-formatted transcript
   (card + User:/Char: history, no native markers) as one user message to
   /v1/chat/completions. Verified in-process: on that wrapped prompt
   Hermes-3 argmaxes token 128001 end_of_text (logit 21.0) at step 0 -
   the model sees a completed document, not a conversation, and ends the
   turn. Every marker-free Kobold prompt took this path -> empty output
   (temp 0) or word salad (temp 2, what Chub's preset sent). LM Studio's
   Kobold mode sent the prompt raw; wrapping was a regression.
   File: Local_To_Chub/chub_forge_proxy.py (handle_kobold)

2. Prefix-cache slices pinned whole KV buffers.
   _slice_past_kv returned seq-axis views; a 32-token LearnedPrefixCache
   entry shared storage with the full prompt+generation KV (~1 GB at 8k
   ctx). Up to 256 entries -> VRAM grew to limit -> server crash
   ("Cannot reach ForgeAI" in proxy.log).
   File: forge/engine/prefix_cache.py (_slice_past_kv)

3. SemanticKVAnchors stored raw captured_kv per anchor.
   Up to 3 anchors/save x 16 retained x full request KV, write-only in
   production (nothing reads kv_state back).
   File: forge/engine/engine_generation.py (~L354)

4. Decode captures retained request KV between requests.
   _forge_last_kv/_forge_last_logits/_forge_prefill_kv stayed set until
   the NEXT request cleared them - one full prompt+gen KV (~1 GB) pinned
   while idle.
   File: forge/engine/engine_generation.py (_generate_impl, generate_stream)

### Resolution
1. Local_To_Chub Kobold endpoint now always routes to /v1/completions
   (raw continuation - matches LM Studio behavior) and forwards Chub's
   stop_sequence merged into payload stop.
2. _slice_past_kv clones tuple slices and dict tensor values - entries
   own ~4 MB instead of pinning GBs.
3. save_anchor called with kv_state=None (anchor positions kept for
   future reuse without pinning KV).
4. _generate_impl clears the three capture attrs after
   _cache_prompt_prefix; generate_stream clears them in its finally.

### Files modified
- forge/engine/prefix_cache.py, forge/engine/engine_generation.py
- Local_To_Chub/chub_forge_proxy.py
- tests/unit/test_forge_engine_fixes.py (+3 regression tests)

### Verification
- Server /v1/completions replay of the Chub transcript prompt:
  VRAM flat across 7 requests (7.9 GB +- 15 MB; was +1.5 GB/10 req).
- Proxy /api/v1/generate end-to-end: coherent in-character continuation.
- 40/40 tests in test_forge_engine_fixes.py pass.
- NOT a context-overflow issue: failing prompt was ~524 tokens; the model
  genuinely emits EOS on user-turn-wrapped transcripts.
- Note: temp=2.0 (Chub preset value observed in proxy.log) still produces
  degraded sampling by design - user-side sampler choice, not a bug.
