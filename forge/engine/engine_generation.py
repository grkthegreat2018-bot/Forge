"""Generation, batching, tool-calling, streaming, and logprobs mixin for ForgeEngine."""
import dataclasses
import json
import time
from collections.abc import Iterator

import torch.nn.functional as F

from .engine_common import *  # noqa: F403
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _apply_dry,
    _dry_penalties,
    _is_memory_model,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    logger,
    SamplingParams,
)
from .airllm_streamer import AirLLMStreamer  # noqa: F401
from .cascade import ModelCascade  # noqa: F401
from .crash_recovery import _CHECKPOINT_INTERVAL as _RECOVERY_CKPT_EVERY  # noqa: F401
from .errors import ConfigurationError, GenerationOOMError  # noqa: F401
from .kv.cacheblend import CacheBlend  # noqa: F401
from .test_time_scaling import (  # noqa: F401
    BeamSearch,
    FirstFinishSearch,
    MCTSDecoder,
)
from .prefix_cache import (  # noqa: F401
    cache_prompt_prefix as _cache_prompt_prefix,
    generate_from_prefix_cache as _generate_from_prefix_cache,
)
from forge.model_loader import unpack_output_with_kv  # noqa: F401


class _GenerationMixin:
    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.0, top_p: float = 1.0,
                 top_k: int = 80, repetition_penalty: float = 1.05,
                 finish_sentence: bool = True,
                 context_limit: int | None = None,
                 skip_special_tokens: bool = True,
                 min_p: float = 0.0, min_k: float = 0.0,
                 dry_multiplier: float = 0.0, dry_base: float = 1.75,
                 dry_allowed_length: int = 2, dry_penalty_last_n: int = 512,
                 json_schema: dict | None = None,
                 stop: list[str] | None = None,
                 logprobs: int | None = None,
                 prompt_logprobs: int | None = None,
                 return_logprobs: bool = False,
                 learn: bool | None = None,
                 mem_bias: float | None = None) -> str | dict:
        """Generate text from a prompt using active strategies.

        Args:
            top_k: LFM2.5-recommended top-k sampling (only applied when
                temperature > 0; ignored for greedy decoding).
            repetition_penalty: LFM2.5-recommended repetition penalty (only
                applied when temperature > 0; ignored for greedy decoding).
            finish_sentence: If True, when max_new_tokens is hit mid-sentence,
                continue generating up to 32 extra tokens to reach a natural
                stopping point (period, newline, code block close, EOS).
            context_limit: Override max context tokens for this request only.
                If None, uses the hotswap current setting. Set to a large
                number (e.g. 1_000_000) for infinite context with eviction.
            skip_special_tokens: If True (default), strips special tokens from
                output. Set to False for tool-call parsing (preserves
                <|tool_call_start|>/<|tool_call_end|> markers).
            min_p: Min-p sampling threshold (0 = disabled). Filters tokens
                below min_p * max_prob. Temperature-invariant.
            min_k: Min-k semantic-cliff sampling sensitivity (0 = disabled).
                Detects sharp logit transitions for dynamic truncation.
                Temperature-invariant (ACL 2026).
            dry_multiplier: DRY n-gram repetition penalty (0 = disabled,
                llama.cpp ``dry_multiplier``). Penalizes tokens that would
                continue a repeated suffix longer than
                ``dry_allowed_length``; penalty grows as
                ``dry_base ** excess``. Scans prompt + generated context
                over the trailing ``dry_penalty_last_n`` tokens.
            json_schema: Optional JSON schema dict for constrained decoding.
                When provided, an XGrammarConstrainer is built and used as a
                logits processor to mask tokens that would produce
                schema-invalid JSON. Bypasses prefix cache / CacheBlend /
                AirLLM paths (constrained decoding needs per-step masking).
            learn: memory-model gate (FluxLM only). ``False`` runs the whole
                call observe-only — no prompt ingestion, no decode-time
                writes, no journal entries — so held-out evaluation cannot
                contaminate live memory.  ``None``/``True`` keeps the
                default live-learning behavior.  Ignored by KV-cache models.
            mem_bias: recall<->generation dial (FluxLM only). Scales the
                gcell/gsyn lift weights for this call — <1 leans on
                verbatim recall, >1 leans on generalized generation.
                ``None`` keeps the model default. Ignored by KV-cache models.
        """
        params = SamplingParams(
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, min_p=min_p,
            min_k=min_k, dry_multiplier=dry_multiplier, dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n, stop=stop,
            skip_special_tokens=skip_special_tokens, learn=learn,
            mem_bias=mem_bias)
        with self._gen_lock:
            if params.learn is False and _is_memory_model(self.model):
                prev = self.model.learning
                self.model.learning = False
                try:
                    return self._generate_with_oom_recovery(
                        self._generate_impl, prompt, params,
                        finish_sentence, context_limit,
                        json_schema=json_schema,
                        logprobs=logprobs,
                        prompt_logprobs=prompt_logprobs,
                        return_logprobs=return_logprobs)
                finally:
                    self.model.learning = prev
            if params.mem_bias is not None \
                    and _is_memory_model(self.model):
                prev_b = self.model._gen_bias
                self.model._gen_bias = float(params.mem_bias)
                try:
                    return self._generate_with_oom_recovery(
                        self._generate_impl, prompt, params,
                        finish_sentence, context_limit,
                        json_schema=json_schema,
                        logprobs=logprobs,
                        prompt_logprobs=prompt_logprobs,
                        return_logprobs=return_logprobs)
                finally:
                    self.model._gen_bias = prev_b
            return self._generate_with_oom_recovery(
                self._generate_impl, prompt, params,
                finish_sentence, context_limit,
                json_schema=json_schema,
                logprobs=logprobs, prompt_logprobs=prompt_logprobs,
                return_logprobs=return_logprobs)

    def _generate_impl(self, prompt, params: SamplingParams,
                       finish_sentence, context_limit,
                       json_schema: dict | None = None,
                       logprobs: int | None = None,
                       prompt_logprobs: int | None = None,
                       return_logprobs: bool = False) -> str | dict:
        """Internal generate implementation (no OOM wrapper)."""
        self._require_awake()
        self._validate_generation_params(prompt, params)
        self._check_vram_and_offload_if_needed()

        # Apply pending hot-swap changes before generation
        self.hotswap.apply_pending()

        # Library lorebook injection: augment prompt with relevant entries
        if self._library_enabled and self.library is not None:
            prompt = self.library.inject(
                prompt, max_tokens=self._library_injection_budget)

        # Per-request context limit override
        ctx_limit = context_limit or self.hotswap.current.max_context_tokens

        _t0 = time.perf_counter()
        # Models with unbounded_context (e.g. FluxLM) never truncate —
        # the context sketch has no maximum length by construction.
        unbounded = getattr(self.model, "unbounded_context", False)
        ids = self.tokenizer(
            prompt, return_tensors="pt",
            truncation=not unbounded,
            max_length=None if unbounded else ctx_limit,
            add_special_tokens=True,  # BOS <|startoftext|> — required for sane raw prompts
        ).input_ids.to(self.device)

        # Logprobs capture: record raw (pre-sampling) logits per decode
        # step via step_observer — avoids the old second full generation
        # pass, and on memory models (FluxLM) avoids re-ingesting the
        # sequence into live memory (which silently double-counted every
        # token and appended duplicate journal entries).
        lp_steps: list[dict] = []
        lp_observer = None
        if logprobs is not None and logprobs > 0:
            def lp_observer(lg, tid):
                lp = torch.log_softmax(lg[0].float(), dim=-1)
                n = min(logprobs, lp.shape[-1])
                tv, ti = lp.topk(n)
                lp_steps.append({
                    "token_id": tid, "token_lp": float(lp[tid]),
                    "top_ids": ti.tolist(), "top_vals": tv.tolist()})

        # Per-step decode paths: constrained decoding (json_schema) and
        # logprobs capture share the _decode_tokens loop — the schema
        # masks per step, the observer records raw logits per step.  Both
        # bypass prefix cache / CacheBlend / AirLLM (those fast paths
        # can't run per-step processors or observers).
        if json_schema is not None or lp_observer is not None:
            processor = (self._build_xgrammar_processor(json_schema)
                         if json_schema is not None else None)
            eos_set = self._eos_token_ids()
            generated_ids: list[int] = []
            logits, past_kv = self._prefill(ids)
            gen_tokens = []
            for next_token, _ in self._decode_tokens(
                logits, past_kv, params, eos_set, generated_ids, processor,
                context_ids=ids[0].tolist(), step_observer=lp_observer,
            ):
                gen_tokens.append(next_token)
            if gen_tokens:
                output_ids = torch.cat([ids] + [
                    t.unsqueeze(0) if t.dim() == 1 else t for t in gen_tokens
                ], dim=1)
            else:
                output_ids = ids
        else:
            # CacheBlend (R&D14): non-prefix KV reuse for RAG / tool-use.
            # Attempted before prefix caching; on a productive blend it
            # assembles a KV buffer from pre-computed chunks and decodes the
            # suffix, skipping most of the prefill.  Falls through to the
            # prefix-cache / standard path on a miss (zero overhead).
            output_ids = None
            if self._cache_blend is not None:
                blend_result = self._cache_blend.blend_prefill(self, ids)
                if blend_result is not None:
                    blend_kv, covered_len = blend_result
                    suffix = ids[:, covered_len:]
                    if suffix.shape[1] > 0 and blend_kv is not None:
                        with torch.inference_mode():
                            out = self.model(
                                suffix, past_key_values=blend_kv,
                                use_cache=True, logits_last_only=True)
                            logits, past_kv = unpack_output_with_kv(out)
                        output_ids = self._decode_with_kv(
                            ids, logits, past_kv, params)
                        self._log(f"CacheBlend HIT (reused {covered_len} "
                                  f"tokens, suffix {suffix.shape[1]} to "
                                  f"prefill)")

            # Prefix caching: check if we've seen this prompt prefix before
            if output_ids is None:
                output_ids = _generate_from_prefix_cache(
                    self, ids, params)
            if output_ids is None and self.acceleration == "airllm_streaming":
                output_ids = AirLLMStreamer.generate(
                    self, ids, params.max_new_tokens, params.temperature)
            elif output_ids is None:
                output_ids = self.decoding.generate(
                    self.model, ids, params.max_new_tokens,
                    params.temperature, params.top_p,
                    top_k=params.top_k,
                    repetition_penalty=params.repetition_penalty,
                    min_p=params.min_p, min_k=params.min_k,
                    dry_multiplier=params.dry_multiplier,
                    dry_base=params.dry_base,
                    dry_allowed_length=params.dry_allowed_length,
                    dry_penalty_last_n=params.dry_penalty_last_n)

        # Capture KV cache from decoding step for fast finish-to-stop path
        captured_kv = getattr(self.model, '_forge_last_kv', None)

        # Smart cutoff: if we hit max_new_tokens without EOS, extend to next
        # natural stopping point (up to 32 extra tokens).
        if finish_sentence and output_ids.shape[1] - ids.shape[1] >= params.max_new_tokens:
            output_ids = self._finish_to_stop(
                output_ids, ids.shape[1], params,
                extra_budget=32, past_kv=captured_kv)

        # Store prefix KV cache for future reuse — reuse the KV captured
        # during decoding instead of paying a second full prefill.
        _cache_prompt_prefix(self, ids, captured_kv)

        n_gen = output_ids.shape[1] - ids.shape[1]
        self._record_generation(n_gen)
        prompt_len = ids.shape[1]
        generated_ids = output_ids[0, prompt_len:]
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=params.skip_special_tokens)

        # Stop-string truncation: cut at the EARLIEST occurrence across all
        # stop strings (OpenAI API spec — the first stop hit ends output).
        if params.stop:
            cut = -1
            for s in params.stop:
                idx = result.find(s)
                if idx != -1 and (cut == -1 or idx < cut):
                    cut = idx
            if cut != -1:
                result = result[:cut]

        _gen_ms = (time.perf_counter() - _t0) * 1000
        self._record_output(prompt, result, n_gen, _gen_ms,
                            params.temperature)

        # Logprobs extraction (OpenAI API compat). Any logprobs request
        # returns the dict form — str cannot carry an attribute, so the
        # previous _forge_logprobs attach crashed on this line.
        if return_logprobs or logprobs is not None or prompt_logprobs is not None:
            lp_data = self._extract_logprobs(
                ids, output_ids, logprobs, prompt_logprobs,
                captured_steps=lp_steps)
            return {"text": result, "logprobs": lp_data}

        # Save semantic KV anchors for future reuse (FreeToken-style)
        try:
            anchors = self.semantic_anchors.detect_anchors(prompt + result)
            if anchors and captured_kv is not None:
                for a in anchors[:3]:  # Save up to 3 anchors per generation
                    self.semantic_anchors.save_anchor(
                        token_pos=ids.shape[1],  # approximate
                        text_pos=a["text_pos"],
                        anchor_type=a["type"],
                        kv_state=captured_kv,
                        source_text=prompt,
                    )
        except Exception:
            logger.debug("Failed to save semantic KV anchors", exc_info=True)
        return result

    # ── Test-time scaling & cascade routing (R39-6 / R39-8) ──────────────

    def generate_with_scaling(self, prompt: str, strategy: str = "ffs",
                              n_samples: int = 8, beam_width: int = 4,
                              n_iterations: int = 32, max_tokens: int = 512,
                              **kwargs) -> str:
        """Generate with test-time scaling.

        Trades extra inference FLOPs for higher answer quality by running
        a search strategy (FFS / beam search / MCTS) over the engine.

        Args:
            strategy: ``"ffs"`` (first-finish search), ``"beam"`` (beam
                search), or ``"mcts"`` (Monte-Carlo tree search).
            n_samples: Number of parallel samples for FFS.
            beam_width: Beam width for beam search.
            n_iterations: MCTS iterations.
            max_tokens: Maximum tokens per sample / beam / rollout.
            **kwargs: Forwarded to the underlying ``generate()`` calls
                (e.g. ``temperature``, ``top_p``).

        Returns:
            The best-scoring generation as a string.
        """
        adapter = _ScalingModelAdapter(self, **kwargs)
        if strategy == "ffs":
            scaler = FirstFinishSearch(
                n_samples=n_samples, max_tokens=max_tokens,
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.95))
        elif strategy == "beam":
            scaler = BeamSearch(
                beam_width=beam_width, max_tokens=max_tokens)
        elif strategy == "mcts":
            scaler = MCTSDecoder(
                n_iterations=n_iterations, max_tokens=max_tokens)
        else:
            raise ValueError(
                f"Unknown strategy {strategy!r}; expected "
                "'ffs', 'beam', or 'mcts'")
        return scaler.generate(adapter, prompt)

    def generate_cascade(self, prompt: str, small_engine: "ForgeEngine",
                         difficulty_threshold: float = 0.5,
                         **kwargs) -> str:
        """Generate using model cascade — routes easy queries to small_engine.

        Wraps ``self`` (the large engine) and ``small_engine`` in the
        string-based ``generate(prompt, **kwargs) -> str`` interface that
        :class:`ModelCascade` expects, then delegates routing to it.

        Args:
            prompt: The input prompt.
            small_engine: A smaller/cheaper ``ForgeEngine`` for easy queries.
            difficulty_threshold: Queries with estimated difficulty >=
                threshold go to ``self`` (large); the rest to ``small_engine``.
            **kwargs: Forwarded to the underlying ``generate()`` call.

        Returns:
            The generated string from the routed model.
        """
        small_adapter = _ScalingModelAdapter(small_engine, **kwargs)
        large_adapter = _ScalingModelAdapter(self, **kwargs)
        cascade = ModelCascade(
            small_model=small_adapter,
            large_model=large_adapter,
            difficulty_threshold=difficulty_threshold)
        return cascade.generate(prompt)

    # ── CacheBlend public API (R&D14) ────────────────────────────────

    def enable_cache_blend(self, chunk_size: int = 256,
                           max_chunks: int = 512) -> CacheBlend:
        """Enable CacheBlend non-prefix KV reuse at runtime.

        Returns the ``CacheBlend`` instance so the caller can register
        chunks via ``register_chunk`` / ``register_text``.
        """
        if not isinstance(self._cache_blend, CacheBlend):
            self._cache_blend = CacheBlend(
                chunk_size=chunk_size, max_chunks=max_chunks)
        self._log("CacheBlend: enabled (non-prefix KV reuse)")
        return self._cache_blend

    def register_blend_chunk(self, text: str) -> int:
        """Pre-compute and store a chunk's KV for CacheBlend reuse."""
        if self._cache_blend is None:
            self.enable_cache_blend()
        return self._cache_blend.register_text(self, text)

    @torch.no_grad()
    def generate_adaptive(
        self,
        prompt: str,
        think_max_tokens: int = 512,
        no_think_max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        think_prefix: str = "Let me think about this step by step.\n",
    ) -> tuple[str, bool]:
        """Adaptive thinking generation (RPO-trained models).

        Uses the model's root token to decide whether to think or not:
        1. Forward pass on prompt to get root token logits
        2. If root token indicates thinking → generate with think_prefix
           and higher token budget
        3. If root token indicates direct answer → generate without
           think_prefix and lower token budget

        This gives ~50% token reduction on easy problems while maintaining
        accuracy on hard ones (ACL 2026, Kim et al.).

        Args:
            think_max_tokens: token budget when thinking is triggered
            no_think_max_tokens: token budget for direct answers
            think_prefix: text prepended when thinking mode is selected

        Returns:
            (generated_text, did_think) tuple
        """
        self._require_awake()
        self._check_vram_and_offload_if_needed()

        # Step 1: Forward pass on prompt to get root token logits
        ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        with torch.autocast(
            device_type="cuda", dtype=torch.bfloat16,
            enabled=("cuda" in str(self.device)),
        ):
            logits, _ = self.model(ids)

        root_logits = logits[0, -1, :].float()
        root_token = root_logits.argmax().item()

        # Step 2: Decide think vs no-think based on root token
        # After RPO training, the root token encodes this decision.
        # Heuristic: if the root token matches common thinking markers,
        # use thinking mode; otherwise direct answer.
        think_markers = set()
        for marker in ["Let", "Let me", "First", "To solve", "I need"]:
            t_ids = self.tokenizer(marker, add_special_tokens=False).input_ids
            if t_ids:
                think_markers.add(t_ids[0])

        did_think = root_token in think_markers

        # Step 3: Generate with appropriate budget
        if did_think:
            full_prompt = prompt + think_prefix
            result = self.generate(
                full_prompt, max_new_tokens=think_max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty)
            return result, True
        else:
            result = self.generate(
                prompt, max_new_tokens=no_think_max_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty)
            return result, False

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        temperatures: list[float] | None = None,
        top_ps: list[float] | None = None,
        top_ks: list[int] | None = None,
        repetition_penalties: list[float] | None = None,
        seeds: list[int | None] | None = None,
        stops: list[list[str] | None] | None = None,
        logits_processors: list | None = None,
        skip_special_tokens: bool = True,
        prompt_rewards: bool = True,
        cf_topk: int = 0,
    ) -> list[str]:
        """Generate text for multiple prompts in a single batched forward pass.

        Uses BatchedDecoding for 3-5x throughput vs serial generation.
        All prompts are processed simultaneously — the model's KV cache
        handles all sequences in parallel.

        Args:
            prompts: list of prompt strings (1-8 recommended for 12GB VRAM)
            max_new_tokens: max tokens to generate per prompt
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling threshold
            top_k: top-k sampling limit
            repetition_penalty: repetition penalty
            temperatures / top_ps / top_ks / repetition_penalties / seeds /
                stops: optional per-sequence overrides (same length as
                prompts). Each sequence in the batch gets its own
                sampling parameters, RNG seed, and stop strings.
            logits_processors: optional per-sequence callables
                ``(logits, generated_ids) -> logits`` — same contract as
                generate_raw's logits_processor (e.g. think-cap /
                structural-token bans). Applied before sampling.
            skip_special_tokens: decode flag. Pass False when callers need
                structural markers preserved (e.g. ``</think>`` for
                reasoning-split post-processing).
            prompt_rewards: FluxLM only — ``False`` skips the per-prompt-token
                reward gather during pool prefill (faster; stream-internal
                hedge drift only — canonical memory is never touched and
                packets are re-ingested on apply anyway). Ignored by
                KV-cache models.
            cf_topk: FluxLM only — >0 logs each generated token's top-k
                alternative propensities into ``FluxPacket.alts`` for
                counterfactual credit (``reinforce_alts``).

        Returns:
            list of generated text strings (same order as prompts)
        """
        self._require_awake()
        self.hotswap.apply_pending()
        self._check_vram_and_offload_if_needed()

        with self._gen_lock:
            if _is_memory_model(self.model):
                return self._generate_batch_flux(
                    prompts, max_new_tokens, temperature, top_p, top_k,
                    repetition_penalty,
                    temperatures=temperatures, top_ps=top_ps,
                    top_ks=top_ks,
                    repetition_penalties=repetition_penalties,
                    seeds=seeds, stops=stops,
                    logits_processors=logits_processors,
                    skip_special_tokens=skip_special_tokens,
                    prompt_rewards=prompt_rewards,
                    cf_topk=cf_topk)
            return self._generate_batch_impl(
                prompts, max_new_tokens, temperature, top_p, top_k,
                repetition_penalty,
                temperatures=temperatures, top_ps=top_ps, top_ks=top_ks,
                repetition_penalties=repetition_penalties,
                seeds=seeds, stops=stops,
                logits_processors=logits_processors,
                skip_special_tokens=skip_special_tokens)

    def _generate_batch_flux(
        self, prompts, max_new_tokens, temperature, top_p, top_k,
        repetition_penalty, temperatures=None, top_ps=None, top_ks=None,
        repetition_penalties=None, seeds=None, stops=None,
        logits_processors=None,
        skip_special_tokens=True,
        prompt_rewards: bool = True,
        cf_topk: int = 0,
    ) -> list[str]:
        """FluxLM batched generation via shared-memory stream pool.

        Standard LLM batching for a memory model: all learned state is
        read-only at predict time, so N prompts run as N independent
        streams over the SAME tables — one batched [N,V] logits step per
        round on the model's device (no clones, no serialization, no
        VRAM multiplication; ~1MB of context state per stream).

        Streams are predict-only: the canonical model is never mutated,
        no journal writes, no revert pass.  Each stream instead emits a
        ``FluxPacket`` (prompt+gen ids under a tag); the single learning
        process applies selected packets via ``model.apply_packets`` —
        available on ``self.last_flux_packets`` after the call, applied
        with ``self.apply_flux_packets(accept)``.

        Per-prompt temperature/top_p/top_k/repetition_penalty/seed/
        stops/logits_processors are honoured.  ``flux_batch_workers``
        caps pool width (0/None = all prompts in one batch);
        ``flux_workers_device`` is retained for compatibility but the
        pool always runs on the model's own device.
        """
        from forge.model.flux_stream import FluxStreamPool

        n = len(prompts)
        if n == 0:
            return []
        pool = getattr(self, "_flux_pool", None)
        if pool is None or pool.model is not self.model:
            pool = FluxStreamPool(self.model)
            self._flux_pool = pool

        max_par = getattr(self, "flux_batch_workers", 0) or 0
        if max_par <= 0:
            max_par = n

        norm = []
        for p in prompts:
            ids = self.tokenizer(
                p, return_tensors="pt", add_special_tokens=True,
            ).input_ids[0].tolist()
            norm.append([int(t) for t in ids])

        eos_ids = self._eos_token_ids()
        outs: list[list[int]] = [[] for _ in range(n)]
        packets = []
        for off in range(0, n, max_par):
            chunk = norm[off:off + max_par]
            idx = list(range(off, off + len(chunk)))
            sub = lambda name_list: ([name_list[i] for i in idx]
                                     if name_list else None)
            o, pk = pool.generate(
                chunk,
                max_new_tokens=max_new_tokens,
                temperatures=sub(temperatures)
                or ([temperature] * len(chunk)
                    if temperature is not None else None),
                top_ps=sub(top_ps)
                or ([top_p] * len(chunk) if top_p is not None else None),
                top_ks=sub(top_ks)
                or ([top_k] * len(chunk) if top_k is not None else None),
                repetition_penalties=sub(repetition_penalties)
                or ([repetition_penalty] * len(chunk)
                    if repetition_penalty is not None else None),
                seeds=sub(seeds),
                stops=sub(stops),
                logits_processors=sub(logits_processors),
                tokenizer=self.tokenizer,
                eos_ids=eos_ids,
                tag_fn=lambda j: f"batch:{off + j}",
                prompt_rewards=prompt_rewards,
                cf_topk=cf_topk,
            )
            for j, gi in enumerate(o):
                outs[off + j] = gi
            packets.extend(pk)
        self.last_flux_packets = packets
        return [self.tokenizer.decode(g,
                                      skip_special_tokens=skip_special_tokens)
                for g in outs]

    def apply_flux_packets(self, accept=None) -> dict:
        """Apply selected packets from the last flux batch through the
        model's single learning process (``FluxLM.apply_packets``).

        ``accept``: None = all, bool list aligned with the last batch's
        prompt order, or predicate ``accept(packet) -> bool``.
        """
        packets = getattr(self, "last_flux_packets", None) or []
        return self.model.apply_packets(packets, accept=accept)

    def _generate_batch_impl(
        self,
        prompts: list[str],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        temperatures: list[float] | None = None,
        top_ps: list[float] | None = None,
        top_ks: list[int] | None = None,
        repetition_penalties: list[float] | None = None,
        seeds: list[int | None] | None = None,
        stops: list[list[str] | None] | None = None,
        logits_processors: list | None = None,
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if not prompts:
            return []

        # For single prompt, fall back to regular generate — generate_raw
        # when a logits processor is supplied (generate() has no such arg).
        if len(prompts) == 1:
            if logits_processors and logits_processors[0] is not None:
                return [self.generate_raw(
                    prompts[0], max_new_tokens=max_new_tokens,
                    temperature=temperature, top_p=top_p, top_k=top_k,
                    repetition_penalty=repetition_penalty,
                    logits_processor=logits_processors[0],
                    skip_special_tokens=skip_special_tokens)]
            return [self.generate(
                prompts[0], max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty)]

        # Tokenize all prompts
        all_ids = []
        for p in prompts:
            ids = self.tokenizer(
                p, return_tensors="pt", truncation=True,
                max_length=self.hotswap.current.max_context_tokens
            ).input_ids.to(self.device)
            all_ids.append(ids)

        # Use BatchedDecoding — pass the engine's resolved EOS set so
        # per-sequence termination honors this model's real stop tokens
        # (Jamba V2: <|endoftext|>=2, <|im_end|>=519 — the legacy default
        # {7, 151643, 151645} never fires).
        from forge.engine.batched_decoding import BatchedDecoding
        batched = BatchedDecoding(eos_token_ids=self._eos_token_ids())

        n = len(prompts)
        _t0 = time.perf_counter()
        try:
            output_ids = batched.generate_batch(
                self.model, all_ids,
                max_tokens_list=[max_new_tokens] * n,
                temperatures=temperatures or [temperature] * n,
                top_ps=top_ps or [top_p] * n,
                top_k_list=top_ks or [top_k] * n,
                repetition_penalty_list=(
                    repetition_penalties or [repetition_penalty] * n),
                seed_list=seeds,
                stop_list=stops,
                tokenizer=self.tokenizer,
                processor_list=logits_processors,
            )
        except torch.cuda.OutOfMemoryError:
            # Fallback: serial generation
            self._log(
                f"Batched OOM ({len(prompts)} prompts), falling back to serial",
                level="warn")
            self._clear_cuda_cache()
            return [self.generate(
                p, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty) for p in prompts]

        # Decode each output
        results = []
        for i, ids in enumerate(output_ids):
            prompt_len = all_ids[i].shape[1]
            generated = ids[0, prompt_len:]
            text = self.tokenizer.decode(
                generated, skip_special_tokens=skip_special_tokens)
            results.append(text)

        _gen_ms = (time.perf_counter() - _t0) * 1000
        total_tokens = sum(len(r) for r in results)
        self._record_generation(total_tokens)
        self._log(
            f"Batch generate: {len(prompts)} prompts, "
            f"{total_tokens} tokens, {_gen_ms:.0f}ms",
            source="batch")

        return results

    @torch.no_grad()
    def generate_with_tools(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        max_tool_rounds: int = 5,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        extra_tools: list[dict] | None = None,
    ) -> dict:
        """Agentic generation loop with built-in tool execution.

        The model generates a response. If it makes tool calls, they are
        executed server-side and the results are fed back. This continues
        for up to `max_tool_rounds` rounds or until the model stops
        calling tools.

        Built-in tools available to the model:
          - library_save/search/lookup/get/delete/stats/optimize/set_config
          - engine_set_kv_cache/decoding/context_limit/infinite_context/
            generation_params/feature/apply_changes
          - engine_get_settings/stats/pending
          - engine_batch_generate
          - engine_generate_adaptive

        Args:
            prompt: the user's prompt
            max_new_tokens: max tokens per generation round
            max_tool_rounds: max number of tool execution rounds
            temperature: sampling temperature
            extra_tools: additional tool definitions from the caller

        Returns:
            dict with:
              - "content": final text response
              - "tool_calls": list of all tool calls made
              - "tool_results": list of all tool results
              - "rounds": number of rounds executed
        """
        from forge.self_play.discovery.qwen_adapter import (
            qwen_parse_tool_calls,
            qwen_render_messages,
        )

        # Build tool definitions: built-in + extra
        tool_defs = self.tools.get_tool_defs()
        if extra_tools:
            tool_defs.extend(extra_tools)

        all_tool_calls: list[dict] = []
        all_tool_results: list[dict] = []
        messages = [
            {"role": "user", "content": prompt}
        ]

        for round_idx in range(max_tool_rounds):
            # Render conversation with tools
            rendered = qwen_render_messages(
                messages, tools=tool_defs, add_generation_prompt=True)

            # Library injection on the full conversation
            if self._library_enabled and self.library is not None:
                rendered = self.library.inject(
                    rendered, max_tokens=self._library_injection_budget)

            # Generate
            raw = self.generate(
                rendered, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty)

            # Parse tool calls
            tool_calls, content = qwen_parse_tool_calls(raw)
            # Convert to OpenAI format if needed
            if tool_calls:
                parsed_calls = []
                for tc in tool_calls:
                    if isinstance(tc, dict) and "name" in tc:
                        parsed_calls.append(tc)
                    elif isinstance(tc, dict) and "function" in tc:
                        fn = tc["function"]
                        args = fn.get("arguments", {})
                        if not isinstance(args, dict):
                            try:
                                args = json.loads(args or "{}")
                            except (json.JSONDecodeError, TypeError):
                                self._log(
                                    f"Malformed tool arguments for "
                                    f"{fn.get('name', '?')}: {args!r:.120}",
                                    level="warn")
                                args = {}
                        if not isinstance(args, dict):
                            args = {}
                        parsed_calls.append({
                            "name": fn.get("name", ""),
                            "arguments": args,
                        })
                tool_calls = parsed_calls

            # Add assistant message to conversation
            messages.append({
                "role": "assistant",
                "content": content or "",
                "tool_calls": tool_calls or None,
            })

            if not tool_calls:
                # No more tool calls — we're done
                break

            # Execute tool calls
            results = self.tools.execute_calls(tool_calls)
            all_tool_calls.extend(tool_calls)
            all_tool_results.extend(results)

            # Feed results back to the model
            for call, result in zip(tool_calls, results):
                tool_name = call.get("name", "tool")
                messages.append({
                    "role": "tool",
                    "name": tool_name,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return {
            "content": content or "",
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "rounds": round_idx + 1,
        }

    def _tokenize(self, prompt: str,
                  add_special_tokens: bool = True) -> torch.Tensor:
        """Tokenize a prompt and move token IDs to the engine's device.

        Shared by ``generate_raw`` and ``generate_stream`` so they use
        identical tokenization semantics. Special tokens (BOS
        ``<|startoftext|>``) are added by default — the model was trained
        with a BOS-prefixed sequence and raw prompts tokenize to garbage
        without it.
        """
        return self.tokenizer(
            prompt, return_tensors="pt",
            add_special_tokens=add_special_tokens).input_ids.to(self.device)

    def _prefill(self, ids: torch.Tensor):
        """Run prefill: process ``ids`` through the model with KV cache.

        Returns ``(logits, past_kv)`` — the full-sequence logits and the KV
        cache state ready for autoregressive decoding.

        Shared by ``generate_raw``, ``generate_stream``, and
        ``_finish_to_stop`` (slow path) to avoid duplicating the
        ``model(ids, use_cache=True) + unpack`` pattern.
        """
        with torch.inference_mode():
            out = self.model(ids, use_cache=True, logits_last_only=True)
            return unpack_output_with_kv(out)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float,
                           top_k: int, top_p: float,
                           repetition_penalty: float,
                           generated_ids: list[int],
                           min_p: float = 0.0,
                           min_k: float = 0.0,
                           dry_penalties: dict[int, float] | None = None
                           ) -> torch.Tensor:
        """Sample the next token from logits with top-k / top-p / rep-penalty.

        Centralised sampling used by generate_raw, generate_stream,
        _decode_with_kv and _finish_to_stop so they all share identical
        filtering semantics.

        Args:
            logits: (batch, vocab) already-sliced logits for the next
                position (i.e. ``logits[:, -1, :]`` from the model output).
            temperature: 0 = greedy argmax, >0 = probabilistic sampling.
            top_k: top-k filter (0 = disabled).
            top_p: nucleus filter (1.0 = disabled).
            repetition_penalty: divisor applied to last 64 generated tokens.
            generated_ids: token IDs generated so far (for rep penalty).
            min_p: minimum-probability sampling (0 = disabled). Tokens with
                probability < min_p * max_prob are filtered. Temperature-
                invariant dynamic truncation (Min-p sampling paper, 2024).
            min_k: Min-k semantic-cliff sampling (0 = disabled). Analyzes
                local shape of sorted logit distribution to find "semantic
                cliffs" — sharp transitions from high-confidence to long-tail.
                Temperature-invariant (ACL 2026, Ding et al.). Range [0, 1]:
                0 = disabled, higher = more aggressive cliff detection.

        Returns:
            next_token tensor of shape (batch, 1).
        """
        if repetition_penalty <= 0:
            raise ConfigurationError("repetition_penalty must be positive")

        # Repetition penalty BEFORE temperature (HF ordering; also applies
        # to greedy argmax). One index_copy instead of a kernel per unique
        # token; clone first since `logits` may be an inference tensor.
        next_logits = logits
        if repetition_penalty != 1.0 and generated_ids:
            next_logits = logits.clone()
            repeated = torch.tensor(
                tuple(set(generated_ids[-64:])), device=logits.device)
            next_logits.index_copy_(
                -1, repeated,
                next_logits.index_select(-1, repeated) / repetition_penalty)
        if temperature <= 0:
            return next_logits.argmax(-1, keepdim=True)
        next_logits = next_logits / temperature

        # DRY penalty: precomputed {token_id: penalty} logit subtraction
        # for tokens that would continue a repeated n-gram suffix.
        if dry_penalties:
            next_logits = _apply_dry(next_logits, dry_penalties)

        # Min-p sampling: filter tokens below min_p * max_prob.
        # Temperature-invariant because it's applied in probability space
        # relative to the top token. Applied early as a coarse filter.
        if min_p > 0.0:
            probs = torch.softmax(next_logits, dim=-1)
            max_prob = probs.max(dim=-1, keepdim=True).values
            threshold = min_p * max_prob
            next_logits = torch.where(
                probs < threshold,
                torch.full_like(next_logits, float("-inf")),
                next_logits,
            )

        # Min-k sampling: detect semantic cliffs in sorted logit distribution.
        # Temperature-invariant — operates on relative logit dynamics, not
        # absolute probabilities. Finds the sharpest transition from
        # high-confidence core to uncertain long-tail and truncates there.
        if min_k > 0.0:
            next_logits = _min_k_filter(next_logits, min_k)

        # Top-k filtering
        if top_k > 0:
            candidate_logits, candidate_ids = torch.topk(
                next_logits, min(top_k, next_logits.shape[-1]), dim=-1)
        else:
            candidate_logits = next_logits
            candidate_ids = torch.arange(
                next_logits.shape[-1], device=logits.device).expand_as(next_logits)
        # Top-p filtering
        if top_p < 1.0:
            candidate_logits, order = torch.sort(
                candidate_logits, descending=True, dim=-1)
            candidate_ids = candidate_ids.gather(-1, order)
            cumulative_probs = torch.cumsum(
                torch.softmax(candidate_logits, dim=-1), dim=-1)
            remove = cumulative_probs > top_p
            remove[..., 1:] = remove[..., :-1].clone()
            remove[..., 0] = False
            candidate_logits = candidate_logits.masked_fill(
                remove, float("-inf"))
        sampled = torch.multinomial(
            torch.softmax(candidate_logits, dim=-1), num_samples=1)
        return candidate_ids.gather(-1, sampled)

    def _eos_token_ids(self, custom_ids=None) -> set[int]:
        token_ids = set(custom_ids) if custom_ids else set(_DEFAULT_EOS_TOKEN_IDS)
        sources = (
            self.tokenizer,
            self.model,
            getattr(self.model, "config", None),
        )
        for source in sources:
            token_id = getattr(source, "eos_token_id", None)
            if isinstance(token_id, (list, tuple, set, frozenset)):
                token_ids.update(token_id)
            elif token_id is not None:
                token_ids.add(token_id)
        return token_ids

    def _decode_tokens(
        self, logits, past_kv, params: SamplingParams,
        stop_token_ids, generated_ids=None,
        logits_processor=None, context_ids: list[int] | None = None,
        hidden_observer=None, step_observer=None,
    ):
        generated_ids = generated_ids if generated_ids is not None else []
        dry_ctx = context_ids if context_ids is not None else []
        try:
            for _ in range(params.max_new_tokens):
                raw_logits = logits[:, -1, :]
                next_logits = raw_logits

                # Constrained decoding: apply logits processor BEFORE sampling
                if logits_processor is not None:
                    next_logits = logits_processor(next_logits, generated_ids)

                dry_pen = (
                    _dry_penalties(
                        dry_ctx + generated_ids, params.dry_penalty_last_n,
                        params.dry_allowed_length, params.dry_multiplier,
                        params.dry_base)
                    if params.dry_multiplier > 0.0 else None
                )
                next_token = self._sample_next_token(
                    next_logits, params.temperature, params.top_k,
                    params.top_p, params.repetition_penalty, generated_ids,
                    params.min_p, params.min_k, dry_penalties=dry_pen)
                token_id = next_token.item()
                generated_ids.append(token_id)
                if step_observer is not None:
                    # Raw (pre-processor, pre-temperature) logits + the
                    # actually-sampled id — logprobs capture needs both.
                    step_observer(raw_logits, token_id)
                should_stop = token_id in stop_token_ids
                yield next_token, should_stop
                if should_stop:
                    break

                # Crash recovery: checkpoint generation + KV snapshot.
                # _safe_decode_ids is O(n) — only run it on the same
                # %32 boundary checkpoint_generation writes at, otherwise
                # detokenization alone is O(n^2) across the generation.
                n_gen = len(generated_ids)
                if hasattr(self, '_recovery') and self._recovery.enabled:
                    if n_gen % _RECOVERY_CKPT_EVERY == 0:
                        partial = self._safe_decode_ids(
                            generated_ids, skip_special_tokens=False)
                        self._recovery.checkpoint_generation(
                            "", partial, n_gen)  # prompt filled by caller
                    self._recovery.snapshot_kv_cache(n_gen)

                with torch.inference_mode():
                    kw = {"past_key_values": past_kv, "use_cache": True}
                    if hidden_observer is not None:
                        kw["return_hidden"] = True
                    out = self.model(next_token, **kw)
                    logits, past_kv = unpack_output_with_kv(out)
                    if hidden_observer is not None:
                        # hidden of the just-consumed token (the state that
                        # will produce the next logits); generated_ids
                        # already includes it — ForgeGate conv/doom probes
                        # score exactly this per-step hidden.
                        hidden_observer(out[-1][0, -1], generated_ids)
        finally:
            if self.model is not None:
                self.model._forge_last_kv = past_kv

    def _build_xgrammar_processor(self, schema: dict):
        """Build an XGrammar logits processor from a JSON schema.

        Creates an :class:`XGrammarConstrainer` compiled with the given
        schema and returns a closure compatible with the
        ``logits_processor`` parameter of ``_decode_tokens``.

        The closure signature is ``(logits, generated_ids) -> logits``,
        matching the existing logits-processor pattern.  At each call:

        1. If ``generated_ids`` is non-empty, the constrainer's FSM is
           advanced with the last generated token (reflecting the token
           selected in the *previous* step).  On the first call the FSM
           is at its initial state (set by ``compile_json`` → ``reset``).
        2. A boolean mask ``(vocab_size,)`` is obtained from
           ``constrainer.get_mask``.
        3. Disallowed tokens are masked to ``-inf``.
        4. The modified logits are returned for sampling.

        Args:
            schema: JSON schema dict (e.g. ``{"type": "object", ...}``).

        Returns:
            A callable ``logits_processor(logits, generated_ids) -> logits``.
        """
        from forge.engine.structured import XGrammarConstrainer

        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None:
            # Fallback: derive from the model's embedding/output layer.
            vocab_size = getattr(self.model, "vocab_size", 65536)
        constrainer = XGrammarConstrainer(vocab_size, self.tokenizer)
        constrainer.compile_json(schema)

        def logits_processor(logits, generated_ids):
            # Advance the FSM with the last generated token (if any).
            # On the first call generated_ids is empty → FSM stays at START.
            if generated_ids:
                constrainer.advance(generated_ids[-1])
            # Get the allowed-token mask for the current FSM state.
            mask = constrainer.get_mask(
                generated_ids[-1] if generated_ids else 0)
            mask = mask.to(logits.device)
            # Mask out disallowed tokens (set to -inf).
            if logits.dim() > 1:
                # (batch, vocab) — broadcast mask across batch dim.
                logits = logits.masked_fill(
                    ~mask.unsqueeze(0), float("-inf"))
            else:
                logits = logits.masked_fill(~mask, float("-inf"))
            return logits

        return logits_processor

    @torch.no_grad()
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for the input text.

        Uses the model's last hidden state (mean-pooled) as the embedding.
        Falls back to a hash-based pseudo-embedding if the model doesn't
        expose hidden states.
        """
        ids = self.tokenizer(text, return_tensors="pt",
                             truncation=True, max_length=512).input_ids.to(self.device)
        # A no-cache forward still resets live conv/recurrent buffers —
        # serialize against in-flight generation.
        with self._gen_lock, torch.inference_mode():
            out = self.model(ids, use_cache=False, return_hidden=True)
            # Model returns (logits, loss, presents, hidden) with return_hidden
            if isinstance(out, tuple) and len(out) >= 4:
                hidden = out[3]
            elif hasattr(out, 'last_hidden_state'):
                hidden = out.last_hidden_state
            else:
                # Fallback: use logits as pseudo-embedding
                hidden = out[0] if isinstance(out, tuple) else out.logits
            # Mean-pool over sequence dimension
            emb = hidden[0].float().mean(dim=0)
            # L2 normalize
            emb = emb / emb.norm().clamp(min=1e-8)
        return emb.cpu().tolist()

    @torch.no_grad()
    def rerank(self, query: str, document: str) -> float:
        """Score the relevance of a document to a query.

        Uses cross-attention scoring: concatenates query + document,
        runs a forward pass, and uses the final hidden state's cosine
        similarity between query and document segments as the relevance score.
        """
        # Simple approach: embed both and compute cosine similarity
        q_emb = torch.tensor(self.embed(query), device=self.device)
        d_emb = torch.tensor(self.embed(document), device=self.device)
        score = torch.cosine_similarity(q_emb.unsqueeze(0), d_emb.unsqueeze(0)).item()
        return score

    def decide(self, state, questions: dict, context_limit: int | None = None,
               max_batch: int = 32, scorer=None) -> dict:
        """System One-style typed decisions (TypeSafe API-compatible).

        Evaluates typed questions (noul / choice / score) against ``state``
        via candidate-continuation scoring — single forward passes, no
        generation, no parsing.

        When ``scorer`` (a DecisionScorer, or one loaded via
        ``load_decision_scorer``) is present, candidate probabilities come
        from the trained verifier head — calibrated Tier-1.  Otherwise
        probabilities are raw LM softmax values: consistent within a
        question but NOT calibrated.

        Args:
            state: text or JSON object the questions are about.
            questions: {key: {"type": "noul"|"choice"|"score",
                              "instructions"?, "criteria"?}} — see
                forge/engine/decide.py for the wire contract.
            context_limit: max prompt tokens per question (state is
                truncated to fit). Default 8192.
            max_batch: candidate rows per forward pass.
            scorer: DecisionScorer override (defaults to the engine's
                loaded scorer, if any).

        Returns:
            {"model": "", "answers": {...},
             "usage": {"billing_units": n, "input_tokens": n,
                       "output_tokens": 0}}
        """
        # _gen_lock held for the whole batch: the model keeps mutable
        # per-layer state that concurrent forwards would corrupt.
        scorer = scorer if scorer is not None else getattr(
            self, "_decision_scorer", None)
        with self._gen_lock:
            self._require_awake()
            evaluator = getattr(self, "_system_one", None)
            if evaluator is None or evaluator.max_batch != max_batch:
                from .decide import SystemOneEvaluator
                evaluator = SystemOneEvaluator(
                    self.model, self.tokenizer, self.device,
                    max_batch=max_batch)
                self._system_one = evaluator
            evaluator.scorer = scorer
            return evaluator.evaluate(
                state, questions, context_limit=context_limit)

    def load_decision_scorer(self, path: str) -> None:
        """Load a trained DecisionScorer for calibrated decide() calls."""
        from .decision_head import DecisionScorer
        self._decision_scorer = DecisionScorer.load(path, self.device)

    def load_gate_probes(self, path: str) -> None:
        """Load the ForgeGate probe bundle (route/doom/convergence)."""
        from .gated import GateProbes
        self._gate_probes = GateProbes.load(path, self.device)

    def generate_gated(self, question: str, cfg=None,
                       return_result: bool = False):
        """ForgeGate cascade: route -> monitored direct -> escalate ->
        convergence-exit think. Requires load_gate_probes() first.

        Chat-mode fast path: wraps `question` in the chat template the
        probes were trained on. Returns GateResult (or its text)."""
        if getattr(self, '_gate_probes', None) is None:
            raise ConfigurationError(
                "generate_gated requires load_gate_probes(path) first")
        from .gated import GatedDecoder
        dec = GatedDecoder(self.model, self.tokenizer, self.device,
                           self._gate_probes, cfg)
        res = dec.generate(question)
        return res if return_result else res.text

    @torch.no_grad()
    def generate_raw(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.2,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        logits_processor=None,
        eos_token_ids: list[int] | None = None,
        skip_special_tokens: bool = False,
        min_p: float = 0.0,
        min_k: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n: int = 512,
        json_schema: dict | None = None,
    ) -> str:
        """Generate text with raw control — for self-play / agentic loops.

        Unlike ``generate()``, this method:
          - Supports a ``logits_processor`` callback for constrained decoding
            (e.g. xgrammar bitmask for tool-call JSON).
          - Returns the decoded string with configurable ``skip_special_tokens``
            (self-play needs special tokens preserved for tool-call parsing).
          - Does NOT do prefix caching or finish_sentence extension (the
            agentic loop manages its own stopping logic).
          - Uses the active KV cache strategy + Triton conv + torch.compile
            if activated on the engine.

        Args:
            prompt: input prompt string
            max_new_tokens: max tokens to generate
            temperature: sampling temperature (0 = greedy)
            top_p: nucleus sampling threshold
            top_k: top-k sampling
            repetition_penalty: repetition penalty
            logits_processor: optional callback ``(logits, token_ids) -> logits``
                called BEFORE top-k/temperature. Use for grammar constraints.
            eos_token_ids: custom EOS token IDs to stop on. If None, uses
                {7, 151643, 151645} (LFM2.5 + Qwen2.5 defaults).
            skip_special_tokens: if True, strips special tokens from output.
                Self-play needs False to preserve tool-call markers.
            min_p: Min-p sampling threshold (0 = disabled).
            min_k: Min-k semantic-cliff sampling (0 = disabled).
            json_schema: Optional JSON schema dict for constrained decoding.
                When provided (and logits_processor is None), an
                XGrammarConstrainer is built and used as the logits
                processor to mask tokens that would produce schema-invalid
                JSON. If both json_schema and logits_processor are given,
                json_schema takes precedence.

        Returns:
            Decoded string of generated tokens (not including prompt).
        """
        self._require_awake()
        params = SamplingParams(
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, min_p=min_p,
            min_k=min_k, dry_multiplier=dry_multiplier, dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            skip_special_tokens=skip_special_tokens)
        self._validate_generation_params(prompt, params)
        self._check_vram_and_offload_if_needed()
        self.hotswap.apply_pending()

        # Build XGrammar logits processor from JSON schema if provided.
        # Takes precedence over a caller-supplied logits_processor.
        if json_schema is not None:
            logits_processor = self._build_xgrammar_processor(json_schema)

        def _run():
            ids = self._tokenize(prompt)
            eos_set = self._eos_token_ids(eos_token_ids)
            generated_ids: list[int] = []

            logits, past_kv = self._prefill(ids)
            for _ in self._decode_tokens(
                logits, past_kv, params, eos_set, generated_ids,
                logits_processor,
                context_ids=ids[0].tolist(),
            ):
                pass

            self._record_generation(len(generated_ids))
            return self._safe_decode_ids(generated_ids, skip_special_tokens)

        with self._gen_lock:
            return self._generate_with_oom_recovery(_run)

    @torch.no_grad()
    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 80,
        repetition_penalty: float = 1.05,
        skip_special_tokens: bool = True,
        logits_processor=None,
        eos_token_ids: list[int] | None = None,
        min_p: float = 0.0,
        min_k: float = 0.0,
        dry_multiplier: float = 0.0,
        dry_base: float = 1.75,
        dry_allowed_length: int = 2,
        dry_penalty_last_n: int = 512,
        stop: list[str] | None = None,
        hidden_observer=None,
    ) -> Iterator[str]:
        """Token-by-token streaming generator.

        Yields decoded text chunks (one per generated token) as they are
        produced, enabling true token-level SSE streaming with low
        time-to-first-token.

        Mirrors :meth:`generate_raw` decoding logic but yields each token's
        decoded text incrementally instead of collecting all tokens first.

        Args:
            logits_processor: optional callback ``(logits, token_ids) -> logits``
                called BEFORE top-k/temperature. Use for grammar constraints.
            eos_token_ids: custom EOS token IDs to stop on. If None, uses
                {7, 151643, 151645} (LFM2.5 + Qwen2.5 defaults).
            min_p: Min-p sampling threshold (0 = disabled).
            min_k: Min-k semantic-cliff sampling (0 = disabled).
            dry_multiplier: DRY n-gram repetition penalty (0 = disabled).
            stop: optional stop strings — output ends at the earliest
                occurrence (OpenAI semantics, same as ``generate``).
                Matching holds back up to ``max(len(s))`` decoded chars so
                a stop string spanning chunks is still caught.
        """
        self._require_awake()
        params = SamplingParams(
            max_new_tokens=max_new_tokens, temperature=temperature,
            top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, min_p=min_p,
            min_k=min_k, dry_multiplier=dry_multiplier, dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n, stop=stop,
            skip_special_tokens=skip_special_tokens)
        self._validate_generation_params(prompt, params)
        self._check_vram_and_offload_if_needed()
        self.hotswap.apply_pending()

        # Note: OOM recovery wraps the entire generator, but since generators
        # can't be retried mid-yield, we wrap the setup + first yield.
        # If OOM occurs, the generator raises GenerationOOMError.
        #
        # _gen_lock is held for the WHOLE iteration (acquired on first
        # next(), released on close/exhaustion) — the model's mutable
        # conv/recurrent state must not interleave with another request.
        # Callers that stop early should close() the generator.
        self._gen_lock.acquire()
        try:
            ids = self._tokenize(prompt)
            eos_set = self._eos_token_ids(eos_token_ids)
            generated_ids: list[int] = []

            # Stop-string holdback: never emit the last ``holdback``
            # decoded chars until a stop check passes — a match may start
            # inside that tail (stop strings can span token boundaries).
            holdback = (max((len(s) for s in stop), default=0)
                        if stop else 0)
            buf = ""

            logits, past_kv = self._prefill(ids)
            for next_token, _ in self._decode_tokens(
                logits, past_kv, params, eos_set, generated_ids,
                logits_processor,
                context_ids=ids[0].tolist(),
                hidden_observer=hidden_observer,
            ):
                chunk = self._safe_decode_ids([next_token.item()],
                                              skip_special_tokens)
                if not chunk:
                    continue
                buf += chunk
                if stop:
                    cut = -1
                    for s in stop:
                        idx = buf.find(s)
                        if idx != -1 and (cut == -1 or idx < cut):
                            cut = idx
                    if cut != -1:
                        if cut > 0:
                            yield buf[:cut]
                        return
                if len(buf) > holdback:
                    emit = len(buf) - holdback
                    yield buf[:emit]
                    buf = buf[emit:]
            if buf:
                yield buf

            self._record_generation(len(generated_ids))
        except torch.cuda.OutOfMemoryError as e:
            self._clear_cuda_cache()
            vram = self.vram_usage() if self.device.type == "cuda" else {}
            raise GenerationOOMError(
                f"OOM during streaming generation: {e}",
                context={"vram": vram},
                suggestion=("Use generate_raw() instead of generate_stream() "
                            "for OOM recovery, or reduce max_new_tokens."),
            ) from e
        finally:
            self._gen_lock.release()

    def _record_generation(self, n_gen: int):
        """Update generation counters (shared by all generate methods)."""
        self.generation_count += 1
        self.total_tokens_generated += n_gen

    def _validate_generation_params(self, prompt: str,
                                    params: SamplingParams):
        """Validate generation parameters before starting.

        Raises ``ConfigurationError`` or ``GenerationError`` on invalid input.
        """
        if not isinstance(prompt, str) or not prompt:
            raise ConfigurationError("prompt must be a non-empty string")
        if params.max_new_tokens <= 0:
            raise ConfigurationError(
                f"max_new_tokens must be positive, got {params.max_new_tokens}")
        if params.max_new_tokens > 100_000:
            raise ConfigurationError(
                f"max_new_tokens={params.max_new_tokens} is unreasonably large "
                f"(max 100000)", suggestion="Use a smaller value or add a timeout.")
        if not (0 <= params.temperature <= 2.0):
            raise ConfigurationError(
                f"temperature must be in [0, 2.0], got {params.temperature}")
        if not (0 < params.top_p <= 1.0):
            raise ConfigurationError(
                f"top_p must be in (0, 1.0], got {params.top_p}")
        if params.top_k <= 0:
            raise ConfigurationError(
                f"top_k must be positive, got {params.top_k}")
        if params.repetition_penalty <= 0:
            raise ConfigurationError(
                f"repetition_penalty must be positive, got {params.repetition_penalty}")
        if not (0 <= params.min_p <= 1.0):
            raise ConfigurationError(
                f"min_p must be in [0, 1.0], got {params.min_p}")
        if not (0 <= params.min_k <= 1.0):
            raise ConfigurationError(
                f"min_k must be in [0, 1.0], got {params.min_k}")
        if params.dry_multiplier < 0:
            raise ConfigurationError(
                f"dry_multiplier must be >= 0, got {params.dry_multiplier}")

        # Check prompt length vs model max_seq_len (warn, don't block —
        # tokenizer may handle truncation, and estimate is rough)
        # With infinite context mode, the limit is the KV cache budget
        # (adjustable via hotswap.set_context_limit()).
        max_seq = self.hotswap.current.max_context_tokens
        est_prompt_tokens = len(prompt) // 4
        if est_prompt_tokens > max_seq and not self.hotswap.current.infinite_context:
            self._log(
                f"Prompt ~{est_prompt_tokens} tokens may exceed max_seq_len "
                f"({max_seq}). Generation may truncate.",
                level="warn")

    def _generate_with_oom_recovery(self, generate_fn, *args, **kwargs):
        """Wrap a generation function with OOM detection and auto-degradation.

        If ``torch.cuda.OutOfMemoryError`` is raised during generation, this
        attempts recovery by:
          1. Clearing CUDA cache and retrying
          2. Reducing KV bits to 4 and switching to S4R cache
          3. Switching to CPU offload KV cache
          4. Giving up with a helpful error message
        """
        if self.device.type != "cuda":
            return generate_fn(*args, **kwargs)

        try:
            return generate_fn(*args, **kwargs)
        except torch.cuda.OutOfMemoryError as e:
            self._log(f"OOM during generation: {e}", level="error")
            self._clear_cuda_cache()

            # Attempt 1: retry after cache clear
            try:
                self._log("Retrying after CUDA cache clear...")
                return generate_fn(*args, **kwargs)
            except torch.cuda.OutOfMemoryError:
                pass

            # Attempt 2: reduce KV bits + switch to S4R
            old_kv = self.kv_cache
            getattr(self, '_active_kv_bits', 8)
            try:
                self._log("Retrying with S4R 4-bit KV cache...")
                self._activate_kv_cache("s4r", None)
                result = generate_fn(*args, **kwargs)
                self._log("OOM recovery successful (S4R 4-bit)")
                return result
            except torch.cuda.OutOfMemoryError:
                self.kv_cache = old_kv  # restore
                pass

            # Attempt 3: CPU offload KV cache
            try:
                self._log("Retrying with CPU offload KV cache...")
                self._activate_kv_cache("cpu_offload", None)
                result = generate_fn(*args, **kwargs)
                self._log("OOM recovery successful (CPU offload KV)")
                return result
            except torch.cuda.OutOfMemoryError:
                self.kv_cache = old_kv
                pass

            # All recovery attempts failed
            vram = self.vram_usage()
            raise GenerationOOMError(
                f"Out of memory after all recovery attempts. "
                f"VRAM: {vram['used_gb']:.1f}/{vram['total_gb']:.1f} GB used.",
                context={"vram_used_gb": vram["used_gb"],
                         "vram_total_gb": vram["total_gb"],
                         "model": getattr(self.config, "name", "unknown")},
                suggestion=("Try engine.sleep(1) to offload weights to CPU, "
                            "or use quantize='int4' for 4x weight compression.")
            ) from e

    def _safe_decode_ids(self, token_ids: list[int],
                         skip_special_tokens: bool = True) -> str:
        """Decode token IDs, clamping to tokenizer vocab range.

        The model vocab may be larger than the tokenizer vocab (e.g. padding
        for tensor-parallel alignment). This clamps out-of-range IDs to the
        last valid token before decoding, preventing IndexError.

        Shared by ``generate_raw`` and ``generate_stream``.
        """
        tok_vocab = len(self.tokenizer)
        safe_ids = [t if t < tok_vocab else tok_vocab - 1 for t in token_ids]
        return self.tokenizer.decode(
            safe_ids, skip_special_tokens=skip_special_tokens)

    def _record_output(self, prompt, result, n_gen, gen_ms, temperature):
        """Record output + timing event for diagnostics."""
        kv_cache_info = self.kv_cache.info() if self.kv_cache else {}
        self.outputs.record(
            prompt, result, n_gen, gen_ms, temperature=temperature,
            kv_cache=kv_cache_info.get("name", kv_cache_info.get("type", "none")),
            decoding=self.decoding.name,
        )
        self.events.log(f"generate: {n_gen} tokens in {gen_ms:.0f}ms",
                        source="engine", level="profile",
                        tokens=n_gen, time_ms=round(gen_ms, 1),
                        tok_s=round(n_gen / (gen_ms / 1000), 1) if gen_ms > 0 else 0)

    @torch.no_grad()
    def _chunked_prefill(self, ids: torch.Tensor, chunk_size: int = 512,
                         past_kv=None) -> tuple:
        """Chunked prefill: process prompt in chunks, mixing with decode.

        vLLM V1 / SGLang-style chunked prefill splits long prompts into
        chunks and processes them incrementally, interleaving with decode
        steps from other requests. This prevents a single long prompt from
        blocking the batch.

        Args:
            ids: full prompt token ids [1, seq_len]
            chunk_size: tokens per prefill chunk (default 512)
            past_kv: optional existing KV cache to extend

        Returns:
            (logits, past_kv) after processing all chunks
        """
        seq_len = ids.shape[1]
        if seq_len <= chunk_size:
            # Short prompt — single pass
            with torch.inference_mode():
                out = self.model(ids, past_key_values=past_kv,
                                 use_cache=True, logits_last_only=True)
                logits, past_kv = unpack_output_with_kv(out)
            return logits, past_kv

        # Process in chunks
        offset = 0
        kv = past_kv
        logits = None
        with torch.inference_mode():
            while offset < seq_len:
                end = min(offset + chunk_size, seq_len)
                chunk = ids[:, offset:end]
                out = self.model(chunk, past_key_values=kv,
                                 use_cache=True, logits_last_only=True)
                logits, kv = unpack_output_with_kv(out)
                offset = end
        # Return logits from the last chunk (last position)
        return logits, kv

    def _decode_with_kv(self, ids, logits, past_kv,
                        params: SamplingParams):
        """Standard autoregressive decode from existing KV cache state.

        Used by prefix cache fast path: prefill already done, just decode.
        """
        eos_set = self._eos_token_ids()
        generated_ids: list[int] = []
        generated_tokens = []

        for next_token, is_eos in self._decode_tokens(
            logits, past_kv, params, eos_set, generated_ids,
            context_ids=ids[0].tolist(),
        ):
            if not is_eos:
                generated_tokens.append(next_token)

        if not generated_tokens:
            return ids
        return torch.cat([ids, *generated_tokens], dim=-1)

    # Token IDs for natural stopping points (Qwen2.5)
    def _get_stop_tokens(self) -> set[int]:
        """Get token IDs that indicate natural sentence/code boundaries."""
        if self._stop_tokens is not None:
            return self._stop_tokens
        tok = self.tokenizer
        stops = self._eos_token_ids()
        for text in [".", "!", "?", ".\n", "!\n", "?\n", ".\"", "!", "?",
                     "```\n", "```\n\n", ")\n", ")\n\n", "}\n", "}\n\n"]:
            ids = tok.encode(text, add_special_tokens=False)
            if ids:
                stops.add(ids[-1])
        self._stop_tokens = stops
        return stops

    @torch.no_grad()
    def _extract_logprobs(self, prompt_ids: torch.Tensor,
                          output_ids: torch.Tensor,
                          logprobs: int | None,
                          prompt_logprobs: int | None,
                          captured_steps: list[dict] | None = None) -> dict:
        """Extract token logprobs for OpenAI API compatibility.

        ``captured_steps`` carries per-step records made during decode by
        ``step_observer`` (raw logits + sampled id) — the normal path, so
        extraction costs zero extra forwards.  Legacy callers passing
        ``None`` fall back to re-running generation; on memory models the
        re-run is wrapped in a journal mark + ``revert_since`` so the
        model's live memory is restored afterwards (forward() learns).

        Returns dict with:
          - content: list of {token, logprob, bytes, top_logprobs} per
            generated token (up to ``logprobs`` top alternatives each)
          - prompt: list of {token, logprob, bytes, top_logprobs} per
            prompt token (up to ``prompt_logprobs`` alternatives each)
        """
        prompt_len = prompt_ids.shape[1]
        gen_ids = output_ids[0, prompt_len:]

        content_logprobs = []
        prompt_lp = []

        # Memory models learn on every forward — journaled writes are
        # undone via revert_since; predict-only pushes leave no journal
        # entries, so the stream scalars are snapshotted directly too
        # (ring/H bytes are position-keyed and unreachable once _count
        # is restored — documented harmless in _undo_entry).
        mem = _is_memory_model(self.model)
        saved = None
        if mem and hasattr(self.model, "journal"):
            saved = {"mark": len(self.model.journal.entries),
                     "count": self.model._count,
                     "c": self.model._c.clone(),
                     "proto": self.model._proto.clone()}
            if self.model._gpu:
                saved["pos_g"] = self.model._pos_g.clone()
        try:
            with torch.inference_mode():
                # Prompt logprobs: one forward pass over the prompt
                if prompt_logprobs is not None and prompt_logprobs > 0:
                    out = self.model(prompt_ids, use_cache=False)
                    logits = out[0] if isinstance(out, tuple) else out.logits
                    log_probs = F.log_softmax(logits[0].float(), dim=-1)
                    for i in range(min(prompt_len, len(log_probs))):
                        token_id = prompt_ids[0, i].item()
                        lp = log_probs[i, token_id].item()
                        top_n = min(prompt_logprobs, log_probs.shape[-1])
                        top_vals, top_ids = log_probs[i].topk(top_n)
                        top_lp = [
                            {"token": self.tokenizer.decode([tid.item()]),
                             "logprob": lp_val.item(),
                             "bytes": tid.item()}
                            for tid, lp_val in zip(top_ids, top_vals)
                        ]
                        prompt_lp.append({
                            "token": self.tokenizer.decode([token_id]),
                            "logprob": lp,
                            "bytes": token_id,
                            "top_logprobs": top_lp,
                        })

                if logprobs is not None and logprobs > 0 and len(gen_ids) > 0:
                    if captured_steps is not None:
                        # Zero-cost path: logits were recorded at decode.
                        for st in captured_steps[:len(gen_ids)]:
                            top_lp = [
                                {"token": self.tokenizer.decode([tid]),
                                 "logprob": lv,
                                 "bytes": tid}
                                for tid, lv in zip(st["top_ids"],
                                                   st["top_vals"])
                            ]
                            content_logprobs.append({
                                "token": self.tokenizer.decode(
                                    [st["token_id"]]),
                                "logprob": st["token_lp"],
                                "bytes": st["token_id"],
                                "top_logprobs": top_lp,
                            })
                    else:
                        # Legacy: re-run generation for per-step logits.
                        ids = prompt_ids.clone()
                        past_kv = None
                        for step in range(min(len(gen_ids), output_ids.shape[1] - prompt_len)):
                            with torch.inference_mode():
                                if past_kv is not None:
                                    out = self.model(ids[:, -1:], past_key_values=past_kv, use_cache=True, logits_last_only=True)
                                else:
                                    out = self.model(ids, use_cache=True, logits_last_only=True)
                                logits = out[0] if isinstance(out, tuple) else out.logits
                                past_kv = out[2] if isinstance(out, tuple) and len(out) > 2 else out.past_key_values
                            log_probs = F.log_softmax(logits[0, -1].float(), dim=-1)
                            token_id = gen_ids[step].item()
                            lp = log_probs[token_id].item()
                            top_n = min(logprobs, log_probs.shape[-1])
                            top_vals, top_ids = log_probs.topk(top_n)
                            top_lp = [
                                {"token": self.tokenizer.decode([tid.item()]),
                                 "logprob": lp_val.item(),
                                 "bytes": tid.item()}
                                for tid, lp_val in zip(top_ids, top_vals)
                            ]
                            content_logprobs.append({
                                "token": self.tokenizer.decode([token_id]),
                                "logprob": lp,
                                "bytes": token_id,
                                "top_logprobs": top_lp,
                            })
                            ids = torch.cat([ids, gen_ids[step:step+1].unsqueeze(0)], dim=1)
        finally:
            if saved is not None:
                # Undo the scoring-pass writes — restore live memory to
                # exactly its pre-extraction state.
                n_new = len(self.model.journal.entries) - saved["mark"]
                if n_new > 0:
                    self.model.revert_since(n_new)
                self.model._count = saved["count"]
                self.model._c.copy_(saved["c"])
                self.model._proto.copy_(saved["proto"])
                if "pos_g" in saved:
                    self.model._pos_g.copy_(saved["pos_g"])

        return {"content": content_logprobs, "prompt": prompt_lp}

    def _finish_to_stop(self, output_ids, prompt_len,
                        params: SamplingParams, extra_budget=32,
                        past_kv=None):
        """Continue generation until a natural stopping point or extra_budget.

        If past_kv is provided (captured from the decoding step), skips the
        expensive full-sequence re-run and continues directly from the last state.
        Otherwise falls back to a full prefill to recover KV cache state.
        """
        stop_tokens = self._get_stop_tokens()
        generated_ids = output_ids[0, prompt_len:].tolist()

        if past_kv is not None:
            last_token = output_ids[:, -1:]
            with torch.inference_mode():
                out = self.model(last_token, past_key_values=past_kv, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)
        elif _is_memory_model(self.model) and \
                hasattr(self.model, "_predict"):
            # Memory models already learned every generated token during
            # decode — a re-prefill would double-ingest the whole sequence
            # into live memory.  The next-token distribution comes
            # straight from the current sketch at zero cost.
            with torch.inference_mode():
                logits = self.model._predict()[None, None, :]
        else:
            logits, past_kv = self._prefill(output_ids)

        budget_params = dataclasses.replace(
            params, max_new_tokens=extra_budget)
        generated_tokens = [
            next_token
            for next_token, _ in self._decode_tokens(
                logits, past_kv, budget_params, stop_tokens, generated_ids,
                context_ids=output_ids[0, :prompt_len].tolist(),
            )
        ]
        if not generated_tokens:
            return output_ids
        return torch.cat([output_ids, *generated_tokens], dim=-1)

