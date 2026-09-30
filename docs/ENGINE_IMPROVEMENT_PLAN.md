# ForgeEngine Improvement Plan

**Date:** 2026-09-25 · **Status:** research-complete, awaiting execution approval
**Sources:** 3-agent static sweep + `docs/CODEBASE_CRITIQUE_2026-09-16.md` (P0–P18, still-open items verified) + live web research (section 8)

**Target model:** ForgeLM V2 (Jamba-hybrid — 26 Mamba-2 SSM + 2 GQA/MQA attention,
d_model=2560, head_dim=128, vocab=65536, no RoPE, bf16 ~6.1 GB)
**Hardware:** RTX 5070 12 GB / 32 GB RAM / Windows 11 / SM120 Blackwell

## 0. Core Diagnosis

The engine's decode path is **launch-bound and mis-wired**:

- ~800 kernel launches/token (26 mamba layers × ~30 kernels each), `.item()`
  GPU→CPU sync per token, O(T²) `torch.cat` KV growth on the 2 attention layers.
- Most registered features (KV strategies, schedulers, sparse-attention
  wrappers, spec-decode, CUDA graphs, async D2H) are **constructed at
  activation but never called by `generate()`** — logs report capabilities
  that are not on the hot path.
- The good substrate already exists: `PreAllocatedKVCache`,
  `preallocated_cache=` forward param, `compile_decode_step`,
  `cudagraph_mark_step_begin`, `async_d2h.py`, `unified_radix.py`,
  `forge/quant/paged_kv.py`. The work is mostly **wiring, not invention**.

Rule-B applies to every bug below: minimal failing repro first, fix, add a
regression test in `tests/unit/` (CPU-runnable where possible).

---

## Phase 0 — Correctness Bugs (silent wrong output; do first)

| # | Bug | Location | Fix | Effort |
|---|-----|----------|-----|--------|
| 0.1 | **Prefix-cache partial hits corrupt Mamba state.** `_slice_past_kv` slices only `(k,v)` tuples; dict states pass through at end-of-entry, not hit boundary | `prefix_cache.py:348-353` | Reject partial hits for stateful layers until Phase 3 lands chunked SSM snapshots; store per-chunk `(ssm, conv)` snapshots | M |
| 0.2 | **Batched EOS never fires on V2** — `eos_set` hardcoded `{7,151643,151645}` | `batched_decoding.py:36-38` vs `session_manager.py:722` | ~~Build `eos_set` from request `eos_token_ids`~~ **DONE** — call site passes `engine._eos_token_ids()` (merges defaults `{2,7,519,...}` + tokenizer/model/config EOS) | S |
| 0.3 | **Spec-decode crashes on V2** — `k,v = layer_kv` unpacks dict keys; EAGLE3 no KV rollback + off-by-one accept | `decoding.py:696,879,1118-1200` | Guard dict layers (skip or state-aware rollback); fix acceptance index; rejection sampling for T>0 | M |
| 0.4 | **`stop=` doesn't stop** — truncates returned text, decodes to EOS anyway | `engine_generation.py:285-292` | Incremental detokenize + ring-buffer suffix match inside decode loop | M |
| 0.5 | **Left-pad batch prefill pollutes Mamba state** — `attention_mask` guards attention only, pads integrated into SSM/conv | `batched_decoding.py:97-121` | Per-row sequential prefill through state slots, or right-pad + row-length bookkeeping | M |
| 0.6 | **RoPE-only modules crash V2** (`self.rope=None`): `atflash`, `triroute`, `mosa`, `fused_qk_norm_rope_cache`; mosa/triroute also drop `past_key_value` on sparse path | `atflash.py:210`, `triroute.py:282-290`, `mosa.py:201-209` | ~~`if self.rope is not None` guards~~ **DONE** (also `seq_aware_split`, `compact_attention`, `cosa` — same pattern). Sparse-path `past_key_value` drop still open | S |
| 0.7 | **`RadixPrefixCache`** sentinel `object()` returned as cached KV; `insert` overwrites child edges | `prefix_cache.py:668,776` | ~~Fix or delete~~ **DONE** — `kv_state=None`, real mid-edge splits via `_split_edge` + parent tracking, `ref_count=0` on insert so `_evict_lru` can fire. Tests added | S |
| 0.8 | **AirLLMStreamer context-free decode** — `blocks[li](x)` without `past_key_value` | `airllm_streamer.py:237-250` | Thread past_kv through block calls; prefetch shard i+1 on side stream | M |
| 0.9 | **Lifecycle races** — `sleep/wake/load_lora/_swap_weights/boot_config.apply_to` mutate model outside `_gen_lock`; `boot_config` resets feature flags to defaults | `engine_lifecycle.py:28-157`, `engine_lora.py:23-126`, `engine_merging.py:31-49`, `session_manager.py:575` | ~~Funnel through `_gen_lock`~~ **DONE** (locked `*_impl` delegates; RLock nests). Boot-flags sparse-apply still open | M |
| 0.10 | **OOM recovery swaps a dead cache** — retries at identical pressure; successful recovery leaves degraded strategy permanently | `engine_generation.py:1548-1595` | Make recovery operate on the real (prealloc) cache after Phase 1; restore strategy after success | S |

**Verification:** repro scripts per bug (e.g. prefix-hit logits vs fresh
prefill; batched EOS at V2 ids; spec-decode on dict cache), then
`pytest tests/unit --tb=short -q`.

## Phase 1 — Wiring Wins (substrate exists; honesty + free speed)

| # | Change | Location | Payoff | Effort |
|---|--------|----------|--------|--------|
| 1.1 | Pass `PreAllocatedKVCache` through `_decode_tokens`/`StandardDecoding`; V2 needs only 2 attention-layer slots | `engine_generation.py`, `model/kv_cache.py` | O(1) append; kills O(T²) cat; activates existing int8-KV path; makes kv_cache strategies real or deletable | M |
| 1.2 | Wire `async_d2h.py` (pinned+event readback) into decode loop — exists, only used by self_play | `scheduler/async_d2h.py` → `engine_generation.py` | Removes per-token pipeline drain | S |
| 1.3 | `head(hidden[:, -1:])` when `use_cache and targets is None` | `model/llm.py:662` | **DONE** — opt-in `logits_last_only` kwarg (auto-slice would break `loader.py` bit-exact test + prompt_logprobs); wired into `_prefill`/`_prefill_chunked`/CacheBlend/logprobs-rerun/`BatchedDecoding` prefill | S |
| 1.4 | Recovery decode inside `%32` gate (currently re-decodes whole output per token — O(n²)) | `engine_generation.py:1074-1081` | **DONE** — `_safe_decode_ids` now runs only on `_CHECKPOINT_INTERVAL` boundary; KV snapshot still self-gates at %256 | S |
| 1.5 | Vectorize rep-penalty → single `index_fill_` before temperature; same fix in `batched_decoding.py:158-163` | `decoding.py:121-123` | **DONE** — `index_copy_` on clone, applied BEFORE temperature + to greedy (HF ordering); same in `batched_decoding` + `_sample_next_token` | S |
| 1.6 | Skip/zero-alloc dead `StandardKVCache.init` (allocates `zeros(B,n_kv,262144,hd)` at activation) | `kv_backend.py:72-75` | **DONE (adjusted)** — `StandardKVCache` was already lazy; real eager alloc was `KVzipCacheStrategy.init` (`zeros(n_kv,262144,hd)` ×2 ≈128MB on V2) — now lazy `_ensure_allocated` | S |
| 1.7 | Server pass-through: `top_k/seed/repetition_penalty/logprobs` dropped in streaming routes; populate `usage` (all zeros today); `top_k=0` validation consistency | `forge_server.py:1844-1855`, `:885-1017` | **DONE** — both SSE routes forward `top_k/rep_penalty/seed` (seed added to `registry.generate_stream`); `usage` populated via tokenizer counts on chat/completions/embeddings/infill/messages. `top_k=0` validation still open | S |
| 1.8 | `compile_decode_step` + cudagraph for decode — exists, never invoked (superseded by Phase 4 if it lands; still a cheap intermediate) | `llm.py:350,402` | Removes per-op launch cost where graphable | S-M |
| 1.9 | **Vendor vLLM's Triton SSM kernels** — `causal_conv1d_update` + `selective_state_update` (`vllm/model_executor/layers/mamba/ops/`, Apache-2.0, pure Triton → runs on Windows via `triton-windows` ≥3.3). Replaces the `mamba3.py:432` Python scan (P1 — THE decode bottleneck) and is graph-capture-ready out of the box (`state_batch_indices` indirection). Requires `triton-windows` + MSVC `cl.exe` + torch ≥2.7/cu≥12.8 for SM120 | new file under `forge/model/` or `forge/engine/` | Kills P1; prerequisite for Phase 4 | M |
| 1.10 | **Spot-check: is the SSM actually Mamba-2 (SSD) or Mamba-1 math?** | `forge/engine/mamba3.py` vs checkpoint config | **DONE → Mamba-1.** V2 uses `MambaLayer` (`keys/architecture/mamba_probe.py`, `ssm_type="mamba2"` is a mislabel): `A_log (d_inner,d_state)` per-channel, `x_proj→dt_rank+2·d_state`, shared `(B,d_state,L)` B/C, elementwise `exp(dt·A)` scan — no heads/ngroups/segsum. → vendor `selective_state_update` + `causal_conv1d_update` (Mamba-1 semantics). The Python scan is at `mamba_probe.py:164-189`, not `mamba3.py` (that file is `use_mamba3` research-only) | S |

## Phase 2 — Decode Core Convergence

Today there are **two divergent decode loops**: `StandardDecoding.generate`
(default path — has degeneration guards, lacks recovery/observers/stop-early)
and `_decode_tokens` (logprobs/schema/stream paths — has observers, lacks
guards). Bugs fixed in one don't reach the other.

- **2.1** Converge on `_decode_tokens` as the single core with per-step hook
  lists (`on_logits`, `on_sampled`, `on_step`) — recovery, logprobs,
  probes, stop-checks become registered optional callbacks. **Effort: M**
- **2.2** Device-resident sampling state: `generated_ids` in preallocated GPU
  int32 buffer; rep-penalty/DRY as scatter ops on last-64 window; EOS via
  `torch.isin`; sync only when a stop fires. Enables zero-sync greedy
  decode inside Phase 4's graph. **Effort: M**
- **2.3** xgrammar → **real xgrammar library** (mlc-ai `xgrammar`, Windows
  wheels on PyPI, ~36 µs/token): `TokenizerInfo(vocab, vocab_size=65536)` →
  `GrammarCompiler.compile_json_schema` → `GrammarMatcher` with
  `fill_next_token_bitmask` on CPU **overlapped with the GPU forward** +
  `apply_token_bitmask_inplace`. `need_apply=False` skips unconstrained
  steps; `rollback(n)` gives spec-decode retraction free. Replaces both the
  65k-iter Python mask loop AND the schema-blind FSM. Wire `response_format`
  into server request. Fallback: `llguidance` (~25 µs @65k vocab, abi3 win
  wheel) or `outlines-core` (regex-level only). **Effort: S-M**
- **2.4** Spec-decode sanity: device-resident accept counts; dict-cache
  rollback via state-pool slots (cheap, fixed-size — or closed-form
  reconstruction, see RecoverSSM in §8). Draft method: **SuffixDecoding**
  (arXiv 2411.04975, NeurIPS 2025 Spotlight — suffix trees over
  prompt+generated, ~20 µs drafting, up to 5.3× on agentic loops; zero
  training) before any draft-model spec. **Design lesson from vLLM #39273:**
  normalize state position after every step — zero-draft steps must not
  read stale state. **Effort: M**
- **2.4b** Watch: **ReplaySSM** (Dao AI Lab × NVIDIA, vLLM PRs #47576/#48018,
  unmerged but public) — ring buffer of SSM inputs + periodic checkpoint;
  rollback = pointer move, parallel verify in one GEMM, 1.87–1.96× on spec
  decode. Vendor once stable; the pattern works at bs=1 too.
- **2.5** Early-exit hygiene: cancellation token checked in decode loop;
  `generate_stream` releases `_gen_lock` on generator close/disconnect;
  SSE disconnect checks. **Effort: M**
- **2.6** Incremental detokenization: one HF `DecodeStream` per request
  (`tokenizers` ≥0.22, `ids=` prefill param) — handles partial UTF-8
  (returns `None` until boundary) and strip-decoder correctness. Stop
  *ids* checked on host per step (cheap); stop *strings* matched on
  incremental text with a hold-back of `max(len(s))-1` chars so partial
  stop strings never leak (vLLM `FastIncrementalDetokenizer` pattern).
  Replaces per-token `tokenizer.decode` + full-buffer `find` scans.
  **Effort: M**

## Phase 3 — Chunked Mamba-State Prefix Snapshots (NOVEL)

**The highest value/effort item in this plan.** Fixes 0.1 properly.
Precedent confirms the approach: vLLM's **Mamba2 APC** (PR #30877 "align"
mode — SSM state cached only at block boundaries, `mamba_block_size`
multiple of chunk size), SGLang **MambaRadixCache** (`extra_buffer` mode —
2 ping-pong slots/request snapshotting at radix branch points, PR #11214),
and the **Marconi** paper (MLSys 2025 — FLOPs-saved-per-byte admission
policy, 34× hit rate) all validate state-snapshot prefix caching. Our
bs≤8 scale allows a simpler version: `{prefix_hash → (state_slot, kv_len)}`
dict + clone-on-boundary — no block-table machinery needed.

- Store `(ssm_real, ssm_imag, conv)` per **256-token** chunk alongside KV in
  `ChunkedPrefixCache` (alignment must match the SSM kernel's chunk size —
  vLLM uses 256; intermediate states only exist at chunk boundaries).
  Mamba state is **fixed-size** (~26 × (nheads·d_head·d_state) ≈ 35–70 MB
  fp32 + ~1 MB conv) — cheap to snapshot at chunk boundaries.
- Enables **true partial-prefix reuse on a hybrid SSM** — something
  vLLM-class engines cannot do (they disable prefix caching on mamba layers).
- Tiering: GPU-hot chunks / pinned-RAM warm / NVMe cold — prefix state is
  read-once-on-hit, offload nearly free. 32 GB RAM + fast NVMe is a real tier.
- Hybrid with `unified_radix.py` (real radix tree, currently dead) for
  cross-session sharing.
- **Effort: M** · Test: partial-hit logits must equal fresh-prefill logits
  bit-for-bit on V2.

## Phase 4 — Unified State Pool + Single-Graph Decode Step

The throughput multiplier. ~800 launches → 1 graph replay (est. 3–8× decode
at bs=1 on launch-bound hardware; must be measured, not assumed). This is
**proven design, not speculation**: vLLM made full CUDA graphs the default
for mamba models in v0.11.0 (PRs #21401, #23035; reviewers: "makes such a
difference in perf"). Copy their layout exactly (see §8).

- **4.1** Persistent state pools owned by engine — copy vLLM's
  `BaseMambaAttentionMetadataBuilder` layout: `[num_slots, state_dims]`
  conv+SSM pools per layer + `state_indices_tensor` indirection (int32 block
  table written per step into a **static tensor** so captured kernels re-read
  updated pointers) + NULL/PAD slot for padding rows. Per-slot quant policy
  (int8 K/V; bf16 SSM with fp32 in-kernel accumulate). ~35–70 MB state per
  sequence → graph-capture-friendly. **GOTCHA (vLLM #34571/#38270):**
  #slots ≥ max captured batch size — each decode sequence needs its own
  state slot. **Effort: L**
- **4.2** Rewrite mamba `T==1` step as in-place update (no `torch.cat`, no
  fresh dicts) via vendored `causal_conv1d_update` + `selective_state_update`
  (Phase 1.9). Cache `A_neg` (recomputed per call today, `mamba3.py`).
  **Effort: M**
- **4.3** Capture whole step — 28 blocks + `ln_f` + `head` + sampling — as one
  CUDA graph. Small graph family keyed by sampling mode (greedy/top-k/top-p);
  feed RNG via pre-drawn uniform tensor per step. **Effort: L**
- **4.4** Fixed-T "prefill chunk" graph (T=512) interleaved between decode
  replays — hides TTFT behind decode throughput, zero scheduler complexity.
  **Effort: M**
- **4.5** **PDL (programmatic dependent launch)** — SM120 is CC 12.0 ≥ 9.0,
  Triton ≥3.3 exposes `tl.extra.cuda.gdc_wait/gdc_launch_dependents` +
  `launch_pdl=True`. Chains the 28-layer sequence to hide residual launch
  latency inside/between graphs — llama.cpp measured 4–10% additive on
  Blackwell. Spot-check `gdc_*` presence in triton-windows first. **Effort: S**
- **4.6** Async readback (vLLM PR #23569 pattern): sampled token → pinned CPU
  buffer via `non_blocking` copy on side stream + `cuda.Event`; pythonize/
  detok thread blocks on the event. Next step's input_ids copied on-GPU —
  CPU never needs the token to keep running. **Effort: S**
- Depends on Phase 2.2 (device-resident sampling) for clean capture.

## Phase 5 — Continuous Batching for the Hybrid

- Per-row mamba-state slots (fixed-size, no masks) + row scheduler evicting
  finished rows / admitting queued ones between graph replays — this is
  exactly vLLM's `MambaManager`/`SingleTypeKVCacheManager` pattern: one
  state block per request, `state_indices_tensor` maps batch row → slot;
  slot alloc/free on admit/finish. Far simpler than paged attention.
- Share one prefill's prefix chunks across batch rows via Phase-3 snapshots.
- Cooperative cancellation: per-request `cancelled` flag checked at step
  assembly (bounded latency = one step); SSE write-failure +
  `request.is_disconnected()` polling sets it; abort frees slot + KV.
  (llama.cpp lesson: poll liveness inside the batch loop, not only on emit.)
- Replaces today's static left-padded `BatchedDecoding` (EOS bug 0.2,
  polluted-state bug 0.5, no row eviction).
- **Effort: L** — the serving-tier differentiator.

## Phase 6 — Server & Security Hardening

| # | Item | Effort |
|---|------|--------|
| 6.1 | API-key dependency + CORS lockdown; `/v1/tools/*`, `/v1/lora/*`, `/v1/router/*`, `/v1/security/*` behind auth (server binds 0.0.0.0 today, all exposed) | M |
| 6.2 | `sandbox.json` out of writable paths (`research/data`); deny-list it in `PROTECTED_ENGINE_FILES`; blacklist secrets for **reads** not just writes; auth-gate `approve_request` | S |
| 6.3 | Cooperative cancellation + SSE disconnect close; fix batched-stream single-chunk emit (`session_manager.py:751-759`) | M |
| 6.4 | OpenAI-compat: `usage` real counts, error envelope `{error:{type,message}}`, 422 mapping, `presence/frequency_penalty`, `logit_bias`, `response_format`, `stream_options` | S-M |
| 6.5 | Registry VRAM: measure `torch.cuda.mem_get_info` delta at load instead of `1.5×ckpt` declared budget; fix `total_tokens` cumulative over-count; `is_awake` reflect external sleeps | M |
| 6.6 | Crash recovery: atomic writes (temp+rename+checksum), boot-time opt-in resume or remove the `kv_snapshot` D2H stall (it also can't fire — dead strategy object) | M |
| 6.7 | Hotswap dead fields (`max_batch_size`, `batch_timeout_ms`, `auto_offload`, `vram_safety_margin_gb`, `default_*`, `warmup`): wire or strip; sync `_current` with real config; real rollback on reactivation failure | M |

## Phase 6b — Hosting Parity (competitor audit, 2026-09-25)

Three research passes across vLLM/SGLang (engine-grade), llama.cpp/Ollama/
LM Studio/LocalAI/KoboldCpp/TGW/Jan/GPT4All (desktop-grade), TGI/TRT-LLM/
Triton/NIM/LMDeploy/TEI/Replicate (production-grade). Fleet-only items
(TP/PP/DP, PD-disaggregation, KV-aware multi-worker routing, autoscalers,
K8s CRDs) deliberately excluded — single GPU, single box.

**Differentiators ForgeEngine already has that competitors lack:** native
Windows (vLLM/SGLang are Linux-only), `/v1/systemone` (TypeSafe decisions),
`/v1/engine/*` live hotswap, `/v1/library` memory, `/v1/tasks` agent harness,
sandboxed `/v1/tools/*` + approval flow, FluxLM live-learning, GUI, crash
recovery, session cache, multi-format quant auto-select.

### Gaps — high value, small effort

| Item | Who has it | Status here |
|---|---|---|
| `/tokenize` + `/detokenize` endpoints | vLLM, SGLang, TGI, TEI, NIM, llama.cpp, LMDeploy | **Missing** — S, pure tokenizer calls |
| `/metrics` Prometheus endpoint | all of them | **Missing** — M (registry + counters exist inside) |
| `/version`, `/info` (capabilities) | NIM, TGI, TEI, vLLM, LCPP `/props` | **Missing** — S |
| Health live-vs-ready split (503 while loading) | Triton, NIM, KServe, LCPP | Partial `/health` — S |
| Latency-breakdown response headers (`x-queue-time`, `x-inference-time`, `x-time-per-token`) + timings in response body | TGI, Ollama, LMS, KCPP | **Missing** — S |
| `stream_options.include_usage` + usage counts | universal | **Missing** (all zeros today — Phase 6.4 covers) |
| HF-hub auto-download + `model`-field JIT load + `keep_alive`/TTL unload | vLLM, Ollama, LMS, llama.cpp router | **Missing** — M; registry+router exist, add pull/load API |
| Per-request `priority`/`timeout`, `max_concurrent_requests` backpressure + queue timeout | Triton, TGI, vLLM, SGLang | **Missing** — S-M |
| `--api-key` Bearer + localhost default bind + restricted CORS | llama.cpp, TGI, vLLM, NIM | **Missing** — in Phase 6.1 |
| Graceful SIGTERM drain (503-on-health, finish in-flight) | vLLM, SGLang | **Missing** — S |
| Abort endpoint (`/abort_request` rid/all) | SGLang, vLLM(dev), LMDeploy, llama.cpp | **Missing** — needs Phase 2.5 cancel flag first |
| `n>1` parallel samples, `min_tokens`, `stop_token_ids` (in req), `presence/frequency_penalty`, `logit_bias`, `seed` fwd everywhere | universal | Partial/missing fields — S |
| `prompt_token_ids`/`input_ids` request input | vLLM, SGLang, llama.cpp `/completion` | **Missing** — S |
| `response_format`/json_schema in request | universal | Engine has (fake) FSM, not exposed — Phase 2.3 |
| `chat_template_kwargs` + `/apply-template` dry-run | vLLM, SGLang, LCPP | **Missing** — S |
| Model aliases (`--served-model-name`) | vLLM, SGLang | Partial (model_id) — S |
| LoRA per-request `adapter_id`/`model` field + runtime load/unload | vLLM, SGLang, TGI, NIM, LCPP `/lora-adapters` | Load/unload exists; per-request selection unclear — verify |
| GGUF pull path | vLLM, SGLang, Ollama, LCPP | GGUF loader exists (`forge_loader`); no hub fetch — M |
| Revision/source pinning (`hf://id@rev`, `HF_HUB_OFFLINE`) | TGI, vLLM, Ray | **Missing** — S |
| Response cache (exact-match request→response, Triton) | Triton | **Missing** — S (trivial dedup on single GPU) |
| Batch JSONL runner (`vllm run-batch` shape) | vLLM | `/v1/batch/completions` exists; no file/offline runner — S |
| Server bench script (`bench_serving` TTFT/TPOT) | vLLM, SGLang, TGI | **Missing** — S |
| Request logging (`--log-requests`, JSON logs, X-Request-Id) | vLLM, SGLang, TGI, JAN | **Missing** — S |
| OTel tracing endpoint | TGI, vLLM, SGLang | **Missing** — M (nice-to-have) |
| Watermark flag | TGI | **Missing** — niche |
| Context-shift/overflow policy (`contextOverflowPolicy`, `--context-shift`, `--cache-reuse`) | LCPP, LMS, KCPP | `/v1/engine/infinite-context` exists — verify coverage |
| Sleep level-1/2 + `is_sleeping` | vLLM, LMDeploy, SGLang | Have sleep/wake; verify `is_sleeping` state endpoint — S |
| Weight-update endpoints (RL refit) | SGLang rich, vLLM `/collective_rpc` | **Missing** — niche for self-play; maybe |
| `sessions` endpoint (server-side pinned session) | SGLang | Have session_cache — different model, verify fit |
| Env-var config mirroring (`LLAMA_ARG_*`) | llama.cpp, LAI | **Missing** — S |
| Single-binary/embedded-WebUI distribution | LCPP, KCPP, Ollama | Have GUI; no single-binary — M |

### Gaps — deliberate skips (fleet-only or out-of-scope)
TP/PP/DP, PD-disaggregation + NIXL/Mooncake KV transfer, multi-worker
routers (vllm-router/sgl-model-gateway), autoscalers, K8s/Helm/KServe,
Triton model-repository protocol, ensembles, instance-group GPU pinning,
shared-memory/CUDA-IPC infer extensions, gRPC surface, audio/vision/
image-gen endpoints (text-only model), Modal-style container scaling.

## Phase 7 — Consolidation (rule E)

| Action | Files | Note |
|--------|-------|------|
| Delete/quarantine dead KV strategy layer | `kv_backend.py` + ~20 `engine/kv/*.py` (keep `cacheblend`, `learned_prefix_cache`) | Nothing calls append/get; `StandardKVCache` still allocates VRAM |
| Merge 4 sparse-attn wrappers | `cosa`, `compact_attention`, `seq_aware_split`, `flex_decoding` | Identical ~60-line boilerplate, all gated on `preallocated_cache` |
| Delete RoPE-dependent / Windows-dead modules | `fa4_attention`, `atflash`, `lampe`, `jet_long` | No-RoPE model / no Windows install path |
| Archive dead schedulers | `fastserve/libra/kairos/corun/hotprefix/moe_optim/feather/triroute` | R&D residue → docs archive (keep `unified_radix`, `async_d2h` — wire them) |
| Quant consolidation | extract `quant/common.py` (hadamard/hessian/ternary/FP4 helpers from `novel_quant`/`r44`/`r46`); dedupe `IRIFP4Linear` (`novel_quant.py:1075` vs `iri_fp4_key.py:197`); archive `r45`/`r50` | Keep r46/r48 dispatch modes |
| Dead flags | `use_avmp`, `use_virtual_tensor`, `use_progressive_load`, `use_adamx/sharq/mosaic_quant` | Wire or strip — they misreport capability today |
| Capability contract | registry declares `touches: [prefill/decode/sampling/kv]`; `activate` warns when touch-points have no consumer | Prevents this class of dead wiring permanently |

## Phase 8 — Performance-Tech Parity (competitor kernel/scheduler audit)

Third research pass — quant/GEMM kernels, fusion/compile/allocator,
hybrid-offload, scheduler tech. Already-covered items (graphs, PDL, SSM
kernels, async D2H, spec variants, xgrammar, detok, batching) deduped.
Existing-but-dark engine assets noted per item.

### 8A. Fused dequant-GEMM kernels — the biggest open gap

Our quant paths (forge_quant, quamba2 W4A8, w8a8, bitnet b1.58, nvfp4)
dequantize-then-matmul (NC6, confirmed open since Sept-9 critique).
Every competitor runs fused mixed-precision kernels:

| Tech | What | For us | Effort |
|---|---|---|---|
| **Marlin** | Fused W{4,8,FP8,FP4}×FP16 GEMM w/ in-kernel dequant + act-order + zp + MoE; self-contained `.cu`, Apache-2.0. ~3.9× vs FP16 at bs≤32. **Already runs on SM120** (it IS vLLM's 5090 fallback). Buildable via torch cpp_extension | Vendor for W4A16/W8A16/W4A8-INT/NVFP4-weight-only paths | M-L |
| **GemLite** | Pure-Triton fused quant GEMM, **explicit sm_120 focus**, W{1,2,4,8}×INT8 + MXFP4/NVFP4 dtypes, torch.compile-friendly, MIT | **Covers b1.58 as W2A8 + NVFP4 in one dep** — runs native on Windows via triton-windows | M |
| `torch._scaled_mm` | cuBLASLt — native FP4 (float4_e2m1) GEMM on sm120, zero new deps | Cheap floor for real NVFP4 tensor cores | S |
| **ExLlamaV3** | EXL3 codebook quants, official win_amd64 wheels | Optional format support | S |
| **bitnet.cpp W2A8 GEMV** | packs 4 ternaries/int8, official BitNet decode kernel, MIT | Port for b1.58 decode (BitBLAS is Linux-locked — can't use it) | M |
| llama.cpp MMQ pattern | convert all quants → q8_1 tiles in-kernel, int8-mma | Reference design for any future format | — |
| ⚠️ **sm120 NVFP4 trap** | vLLM #47749: "FP4" on 5090 silently resolves to Marlin W4A16 (FP4 hardware idle) | **Must log kernel selection** in our nvfp4 path — verify we're not dequant-only | S |

Skip list: BitBLAS (Linux-assert), Machete (sm90 only), DeepGEMM/trtllm-gen
cubins (sm90/100), torchao int4 tinygemm (Linux-first), AMX (Core Ultra
265F has none).

### 8B. Attention + prefill kernels

| Tech | What | For us | Effort |
|---|---|---|---|
| **Split-KV / FlashDecoding** | decode attention parallelizes over KV length; at bs≤8 + MQA (1 KV head) most of the 5070's ~48 SMs idle. SGLang heuristic: `max_kv_splits = next_pow2(SM_count/4)` | Triton split-KV kernel for our 2 attention layers — biggest long-ctx decode win after graphs | M |
| **Cascade attention** (FlashInfer) | LSE-merge operator ⊕: shared-prefix KV in SMEM + suffix decode, 26-31× on shared-prefix batches | Our self-play/agentic loads have heavy shared prefixes (system prompt). Merge op ~50 lines if FlashInfer wheel unavailable | M |
| **SageAttention2/3** | int8-QK^T + FP8/FP4-PV attention, sm120 verified, 2-5× FA2 | Optional backend for the 2 attn layers at long ctx | S-M |
| **Sarathi mixing** | decode-first per step, fill token budget with prefill chunks (vLLM V1 default); hybrid variant chunks only when decodes active (PR #26625) | Scheduler policy — pairs with Phase 4.4 prefill graph | S-M |
| **varlen packing reset** | FA varlen/cu_seqlens exists (`attention_ops.py:50`) but conv/SSM state never resets at doc boundaries → cross-doc leak (P16) | Correctness, not just perf | M |
| **fp8_e4m3 KV + Triton dequant-on-load** | only FP8-KV path proven on sm120 (FlashInfer FP8 JIT broken <CUDA12.9) | Slot into xquant/hqe paths | S-M |

We already have: fused RMSNorm Triton, fused SwiGLU, fused QK-norm+RoPE
(env opt-in), fused_qk_norm_rope_cache — these are NOT gaps, just opt-in.

### 8C. Scheduler + memory policy (cheap wins)

| Tech | Source | Effort |
|---|---|---|
| Recompute-preemption + LCF victim (least-computed-first, −20-59% wasted recompute) | vLLM #53723 | S |
| `max_num_batched_tokens` ~1-2k on 12GB (small→ITL, large→TTFT) | vLLM/TGI tuning docs | S |
| Cache-aware admission (sort queue by prefix-hit length, `lpm` policy) | SGLang schedule_policy | S |
| `stream_interval N` + output coalescing (57% e2e at N=10 in I/O-bound test) | vLLM #27869/#12298 | S |
| KV-slot LCP reuse + `--cache-reuse` + RAM prompt-cache slots | llama.cpp (-sps) | S-M |
| llama.cpp `-ub` microbatch sweep = our prefill-chunk granularity | bench flag | S |

### 8D. Memory allocator / platform (Windows-specific)

- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — was Linux-only;
  PR #189163 ports to Windows single-process; check our torch version.
  Kills fragmentation between weights + KV + graph pools.
- **WDDM launch tax**: kernel launch 5-20µs + sync ~4-5µs vs ~1µs Linux —
  makes our launch-bound decode *worse* on Windows → strengthens Phase 4.
  Also: prefer events over syncs; never `CUDA_LAUNCH_BLOCKING` patterns.
- **ReBAR**: verify enabled — CPU→GPU streamed weights/KV benefit.
- `cudaMallocAsync` backend: flaky/absent on WDDM — probe
  `cudaDevAttrMemoryPoolsSupported` first; don't assume.
- GDS: Linux-only confirmed — disk-KV must use pinned-CPU bounce (reuse
  `load_safetensors_pipelined` pattern for `DiskKVCache`; Tutti lesson:
  few large contiguous reads >> many small).
- Model-load: our 12.3GB/s pipelined loader ≈ NVMe ceiling; only mmap
  warm-cache (llama.cpp trick) beats it for repeat loads — minor.

### 8E. R&D-grade novel items (cross-domain, our hardware)

1. **ShadowKV-style KV tiering** — low-rank pre-RoPE K on GPU + V in CPU
   RAM + Quest page-metadata (min/max per page) + top-k gather kernel.
   Without top-k selection, full-KV gather from RAM caps ~6 tok/s @128K.
   Fuse `matryoshka_kv` (low-rank K) + `cpu_kv_offload` (V→RAM) + Quest
   metadata — **no engine does this on an SSM-hybrid**. M-L. ★ top pick
2. **Draft-guided KV prefetch (DualDecoder)** — spec drafts predict
   next-step tokens → prefetch cold KV one layer ahead on a side stream.
   Composes our existing suffix/spec with the KV tier. M.
3. **Tree-attention verify kernel** — required to harvest multi-branch
   drafts (Medusa/EAGLE/suffix produce trees); check which attention
   backend accepts arbitrary masks first. M.
4. **FR-Spec** — draft lm_head restricted to freq-ranked top-8k vocab —
   cheap win at 65k vocab. S.
5. **LayerSkip/SWIFT/CLaSp early-exit** — training-free self-spec; no
   engine ships it (differentiator) BUT KV bookkeeping for skipped
   layers on a hybrid is unsolved — flag risk. M-L, R&D round.
6. **Free-threaded Python 3.13t** — parallel CPU drafters/prefetchers
   in-process; PyO3 ecosystem fragile — medium-risk. L.

### 8F. Skip list (verified dead-ends for this stack)

BitBLAS (Linux-only assert), Machete (sm90), FA3 (sm90 wgmma),
trtllm-gen cubins + flashinfer-cubin closed wheels + DeepGEMM
(sm90/100), KTransformers AMX (Xeon-only, no AMX on 265F), GDS/NIXL/UCX
(Linux/POSIX), AllReduce fusions (1 GPU), MoE fused kernels (dense
model — revisit if MoE lands), vLLM MXFP4-Triton on sm120 (slower than
Marlin bs=1).

## 8. External Research Notes

Two web-research passes (2026-09-25). Full detail in `.devin/scratchpad.md`.

### 8.1 SSM/Mamba decode + CUDA graphs (precedent = adopt, don't invent)

- **vLLM full-CUDA-graph for mamba is default since v0.11.0** (Oct 2025;
  `FULL_AND_PIECEWISE`). Layout to copy: per-layer `[num_slots, dims]`
  conv+SSM pools, `state_indices_tensor` indirection, NULL slot for padding,
  conv update fully in-graph via Triton `causal_conv1d_update`.
  Gotcha: `#state_slots ≥ max_captured_batch_size` (vLLM #34571/#38270).
  SGLang equivalent: `HybridReqToTokenPool`/`MambaPool` + per-capture-size
  static index tensors (`hybrid_linear_attn_backend.py`); WAR-event-after-
  replay gotcha (PR #31072).
- **Vendor these Triton kernels** (Apache-2.0, self-contained, Windows-OK
  via `triton-windows`): `vllm/model_executor/layers/mamba/ops/
  causal_conv1d.py` (PR #18218), `mamba_ssm.py` (`selective_state_update`,
  supports `state_batch_indices`/`dst_state_batch_indices`/multi-token),
  `ssu_dispatch.py`. Baseline `selective_state_update` only hits 13–22%
  BW at bs=1 — **launch overhead dominates, not bandwidth** → graphs are
  the right first move.
- **Env requirement**: `triton-windows` ≥3.3 (PyPI `triton-windows`, needs
  MSVC `cl.exe`), PyTorch ≥2.7 + cu≥12.8 for SM120 (torch 2.9/2.10 stable
  wheels exist). FLA's delta-rule ops also run under triton-windows but
  target DeltaNet/GDN, not vanilla Mamba-2.
- **PDL**: CC ≥9.0 (SM120 ✓); `tl.extra.cuda.gdc_*` + `launch_pdl` in
  Triton ≥3.3; 4–10% additive with graphs on Blackwell (llama.cpp #22522).
- **Megakernel**: Hazy "no bubbles" (78% BW bs=1) is H100/B200-only,
  transformer-only, CUDA-C++ — **skip**; Mirage MPK same (no sm_120, SSM
  is open issue #324). Graphs+PDL capture most of the win at 1% effort.
- **⚠️ Verify**: whether ForgeLM V2 SSM is Mamba-1 or Mamba-2 math —
  AI21 Jamba-Reasoning-3B is Mamba-1-flavored on HF; kernel choice +
  state layout depend on it (spot-check 1.10).

### 8.2 SSM prefix caching / spec decode (precedent + what to build)

- **vLLM Mamba2 APC**: PR #30877 "align" mode — SSM state cached only at
  block boundaries (multiple of `mamba_chunk_size`=256); `mamba_cache_mode`
  none/all/align; `--prefix-match-unit` for finer hits; hybrid-miss vetoes
  attention hits (issue #45238 — design our own per-layer hit logic).
- **SGLang MambaRadixCache**: `extra_buffer` = 2 ping-pong slots/request
  snapshotting at radix branch points (PR #11214); graph-decline-to-replay
  rule when tracking disabled (#32606 — apply same rule to our snapshots).
- **Marconi** (MLSys 2025): exact-match-only SSM entries; admit/evict by
  FLOPs-saved-per-byte + reuse forecast → 34× hit rate vs LRU. Later
  refinement for our eviction policy.
- **Spec rollback options** (ranked for us): (a) clone state pre-verify
  (~35–70 MB, <0.3 ms — simplest); (b) ReplaySSM ring-buffer inputs +
  checkpoint (vLLM #47576/#48018, unmerged, 1.87–1.96× spec — watch);
  (c) closed-form RecoverSSM reconstruction (vLLM #51855, delta-rule math).
  **Bug lesson**: vLLM #39273 — zero-draft steps must normalize state to
  a canonical slot or they read stale state.
- **Drafting**: SuffixDecoding (arXiv 2411.04975) — model-free suffix-tree
  drafts, ~20 µs, up to 5.3× on agentic loops; best fit. MTP/EAGLE heads
  need training (round-2+). Medusa-on-Mamba unpublished.

### 8.3 Serving loop (async sampling, detok, cancel, grammar)

- **Async sampling (vLLM V1)**: `--async-scheduling` one-step-ahead (#19970,
  NanoFlow-style); fully-overlapped output (#23569): `sampled_token_ids`
  CUDA tensor → `non_blocking` copy to pinned buffer on side stream +
  `cuda.Event` + pythonize thread; next step's `input_ids` fed on-GPU —
  CPU never needs the value. TRT-LLM overlap scheduler = same pattern
  (stop checks 1 step delayed — standard price).
- **Sampling on GPU**: vLLM V1 sampler = greedy→argmax else temp→min_p→
  top_k/top_p→Gumbel sampling, all device-resident, no host sync.
  FlashInfer fused sampling kernels exist but no official Windows wheel —
  `torch.compile` top-k/top-p + multinomial is sufficient for us.
- **Stop checks**: vLLM checks token-id stops every step on CPU (cheap);
  stop *strings* in detokenizer on incremental text; `--stream-interval N`
  batches emission only (57% e2e gain at interval 10 under load — optional
  for us).
- **Incremental detok**: HF `DecodeStream` (tokenizers ≥0.22 `ids=` param)
  handles partial UTF-8 + strip decoders; hold back `max_stop_len-1` chars;
  skip detok of final token when finish was token-based.
- **Cancellation**: vLLM `abort(request_id)` frees blocks + pushes ABORT;
  async-gen `aclose()` propagates disconnects (don't `asyncio.shield`);
  `request.is_disconnected()` is a poll — pair with write-failure
  detection; llama.cpp lesson — poll liveness inside batch loop.
- **xgrammar real lib**: `xgrammar` PyPI win wheels (~36 µs/token JSON
  mask, CPU-side `fill_next_token_bitmask` overlapped with GPU forward,
  `need_apply=False` skip, `rollback(n)` free); `llguidance` abi3 wheel
  (~25 µs @65k vocab); `outlines-core` win wheel (regex-level only).
  **Drop-in replaces our char-FSM.**
- **Security baseline**: llama.cpp `--api-key` + auto-restricted CORS when
  agent mode on; vLLM Bearer middleware (but `/invocations` stays open —
  cover ALL paths); LMDeploy abort endpoint flags. Baseline for us:
  localhost default bind, Bearer key on all paths incl. tool/lora/security
  endpoints, CORS=localhost when mutating endpoints exist.

### 8.4 Skip list (verified dead-ends for this stack)

mamba-ssm CUDA ext / causal-conv1d CUDA pkg (no official Win builds — use
Triton), FlashInfer (no Win wheel; torch-native suffices), Mirage MPK /
Hazy Megakernels (no sm_120, transformer-only), Medusa-on-Mamba
(unpublished), SGLang as dependency (no Windows support — reference only),
FLA CUDA conv (Triton backend covers it).

## 9. Execution Order (recommended)

1. **Phase 0** — all correctness bugs (each: repro → fix → test)
2. **Phase 1** — wiring wins incl. 1.9 Triton kernel vendoring + 1.10
   Mamba-1-vs-2 spot-check (spot-check gates kernel layout)
3. **Phase 3** — mamba-state snapshots (small, novel, unlocks prefix reuse)
4. **Phase 2** — decode core convergence incl. real xgrammar + DecodeStream
   (prereq for 4/5)
5. **Phase 4** — unified state pool + graph decode + PDL + async readback
   (the multiplier — copy vLLM's proven layout)
6. **Phase 5/6/7** — interleave per appetite

Bench gates: `scripts/bench_load_time.py` (load); a new decode tok/s
micro-bench before/after Phases 1/4; prefix-hit parity test for Phase 3;
graph-capture smoke test asserting `#slots ≥ capture_bs`.
