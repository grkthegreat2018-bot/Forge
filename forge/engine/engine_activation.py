"""Feature activation, quantization, and warmup mixin for ForgeEngine."""
import os
from pathlib import Path

from .engine_common import *  # noqa: F403
import torch.nn as nn  # noqa: F401
from .activation import ActivationConfig  # noqa: F401
from .airllm_streamer import AirLLMStreamer  # noqa: F401
from .errors import ActivationError, ConfigurationError  # noqa: F401
from .feature_registry import _FEATURE_REGISTRY  # noqa: F401
from .innovations import (  # noqa: F401
    MRLAdaptiveContext,
    ProgressiveKV,
    QuaRotKV,
    V0WarmStart,
)
from .kv.cacheblend import CacheBlend  # noqa: F401
from .kv_backend import build_kv_cache  # noqa: F401
from .decoding import build_decoding  # noqa: F401
from .prefix_cache import ChunkedPrefixCache, LRUPrefixCache  # noqa: F401
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _is_memory_model,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    logger,
)


class _ActivationMixin:
    def _detect_keystack_features(self):
        """Detect which KeyStack transforms are in the checkpoint.

        Degrades gracefully on I/O errors (missing/corrupt checkpoint):
        returns empty features + metadata instead of crashing.
        """
        features = []
        ckpt = Path(self.checkpoint_path) if self.checkpoint_path else None
        metadata = {}
        keys = set()
        if ckpt is None or not ckpt.exists():
            self._log("KeyStack feature detection skipped (no checkpoint)",
                      level="warn")
            self.keystack_features = features
            self._checkpoint_metadata = metadata
            return
        try:
            from safetensors import safe_open
            if ckpt.is_dir():
                shards = sorted(ckpt.glob("model-*.safetensors"))
                if shards:
                    with safe_open(str(shards[0]), framework="pt") as f:
                        keys = set(f.keys())
                        metadata = f.metadata() or {}
            else:
                with safe_open(str(ckpt), framework="pt") as f:
                    keys = set(f.keys())
                    metadata = f.metadata() or {}
        except (OSError, RuntimeError, ValueError, ImportError) as e:
            self._log(
                f"KeyStack feature detection failed: {e} — "
                f"continuing with empty features", level="warn")
            self.keystack_features = features
            self._checkpoint_metadata = metadata
            return

        if "value_residual_v0" in keys:
            features.append("value_residual")
        if "rotorquant_rotations" in keys:
            features.append("rotorquant")
        if "mtp_head.heads.0.weight" in keys:
            features.append("mtp")
        if "_airllm_streamable" in keys:
            features.append("airllm")
        if metadata.get("_bitnet_prequant") == "1":
            features.append("bitnet_prequant")
            self._log(f"Pre-quantized BitNet checkpoint detected "
                      f"(mode={metadata.get('_prequant_mode', 'int8')})")
        # QuaRot detection: check if V/O weights are Hadamard-rotated
        # (heuristic: compare against original if available)
        features.append("quarot")  # Assume applied by pipeline
        features.append("mrl")     # Assume applied by pipeline

        self.keystack_features = features
        self._checkpoint_metadata = metadata
        self._log(f"KeyStack features detected: {features}")

    # ── Activation ────────────────────────────────────────────────────────

    def activate_optimal(self, **overrides) -> None:
        """Activate with optimal settings for max VRAM efficiency + speed.

        Picks the best combination of strategies for the current hardware:
          - RotorQuant KV cache (Givens rotation + Lloyd-Max, ~8x compression)
          - torch.compile (1.3-2x decode speedup)
          - Fused QK-Norm+RoPE+Cache-Write (5-10% decode speedup)
          - Triton conv kernel (89% conv bottleneck cut)
          - Prefix cache (avoids re-computing repeated prefixes)
          - Chunked prefill (interleaves with decode for long prompts)

        For pre-quantized BitNet checkpoints, weights are already int8 —
        no quantize= needed (the model uses ternary GEMM natively).

        Mamba/SSM hybrids (e.g. ForgeLM V2 Jamba) are routed to a
        validated mamba-safe profile — no torch.compile, no KV-slicing
        caches that assume (k, v) tuple state.

        Args:
            **overrides: Override any activate() parameter.
        """
        config = ActivationConfig.optimal_for(self.config, **overrides)
        return self.activate_config(config)

    def activate(self, **kwargs) -> None:
        """Activate runtime strategies.

        All keyword arguments map directly to ``ActivationConfig`` fields.
        See ``ActivationConfig`` for the full parameter list and defaults.

        Common args:
            kv_cache: "standard", "paged", "rotorquant", "hadamard_int4", "compressed",
                      "streaming", "snapkv", "filler", "snapkv_4bit", "paged_eviction", "xquant",
                      "cpu_offload", "s4r", "hqe_kv", "hyquant",
                      "evo_sparse", "vegas", "hisparse", "capture",
                      "vtoken", "auto_context"
            decoding: "standard", "speculative", "ngram_speculative",
                      "external_draft_speculative", "medusa", "dspark", "eagle3",
                      "mtp_selfspec", "self_speculative_sparse"
            quantize: None, "int8", "int4", "fp8", "w8a8", "nvfp4",
                      "forge_quant", "grinqh", "mixllm", "acbq", "quamba2",
                      "awq_fp4", "nanoquant", "btc", "ternary_ptq"
            acceleration: None, "cuda_graph", "airllm_streaming", "megakernel", "flex_decoding"
            mrl_keep_ratio: if set (e.g. 0.75), truncate to that fraction of dims
            kv_bits: 4 or 8, for KV cache quantization
            use_compile: torch.compile the model for 1.3-2x decode speedup
            use_prefix_cache: cache KV for repeated prompt prefixes
            warmup: pre-run a dummy token to initialize CUDA kernels (avoids
                    first-generation slowdown). Like llama.cpp's graph reservation.

        For the full set of 50+ feature flags, see ``ActivationConfig``.
        """
        self._require_awake()
        config = ActivationConfig.from_kwargs(**kwargs)
        self.activate_config(config)

    def activate_config(self, config: ActivationConfig) -> None:
        """Activate runtime strategies from an ``ActivationConfig``.

        This is the primary activation path — ``activate()`` and
        ``activate_optimal()`` both delegate here.
        """
        self._require_awake()
        feature_flags = config.feature_flags

        self._activate_core_innovations(
            config.quantize, config.mrl_keep_ratio, config.kv_cache,
            config.kv_bits, feature_flags)
        self._activate_kv_cache(config.kv_cache, config.kv_cache_tokens)
        self._activate_decoding(config.decoding)
        self._activate_acceleration(config.acceleration)
        self._activate_compile_runtime(feature_flags)
        self._apply_feature_registry(feature_flags)
        self._finalize_activation(feature_flags)

        # Store activation parameters for level-2 wake re-activation.
        self._last_activation_params = config.to_dict()

    def _activate_core_innovations(self, quantize, mrl_keep_ratio, kv_cache,
                                   kv_bits, feature_flags):
        # 1. Quantization
        if quantize:
            # Save original weights before quantization modifies them in-place.
            # This allows restore when activate(quantize=None) is called later.
            if not hasattr(self, '_original_weights') or self._original_weights is None:
                self._save_original_weights()
            self._apply_quantization(quantize)
            self.quantize = quantize
        elif self.quantize is not None:
            # quantize=None but model was previously quantized → restore originals
            self._log(f"Restoring unquantized weights (was: {self.quantize})")
            self._restore_original_weights()
            self.quantize = None

        # 2. MRL adaptive context
        if mrl_keep_ratio and mrl_keep_ratio < 1.0:
            cfg = getattr(self.model, "config", None)
            d_model = getattr(cfg, "d_model", 1536)
            self.mrl_adapter = MRLAdaptiveContext(d_model, mrl_keep_ratio)
            self.mrl_adapter.apply_to_model(self.model)

        # 3. QuaRot-KV
        if "quarot" in self.keystack_features and kv_cache in ("hadamard_int4", "rotorquant"):
            self.quarot_kv = QuaRotKV(bits=kv_bits, has_quarot=True)
            self._log(f"QuaRot-KV active: V pre-rotated, "
                      f"K runtime-Hadamard, {kv_bits}-bit")

        # 4. V0 warm start
        if feature_flags["use_v0_warm"] and "value_residual" in self.keystack_features:
            self.v0_warm = V0WarmStart.from_checkpoint(self.checkpoint_path)
            if self.v0_warm:
                self._log(f"V0-WarmStart active: {self.v0_warm.info()}")
            else:
                self._log("V0-WarmStart: no V_0 found in checkpoint", level="warn")

        # 5. Progressive KV
        if feature_flags["use_progressive_kv"]:
            self.progressive_kv = ProgressiveKV(anchor_bits=8, residual_bits=8)
            self._log(f"ProgressiveKV active: {self.progressive_kv.info()}")

        # 6. R35: AVMP — Asymmetric virtual memory paging for hybrid models
        if feature_flags.get("use_avmp"):
            try:
                from forge.engine.memory.avmp import AVMPManager
                # Budget against FREE VRAM — weights are already resident, so
                # total_memory would oversubscribe the card (0 free → KV
                # forced to CPU offload).
                gpu_budget = 12 * 1024**3  # 12GB default
                if self.device.type == "cuda":
                    free_b, _ = torch.cuda.mem_get_info(0)
                    gpu_budget = int(free_b * 0.85)
                self.avmp = AVMPManager(gpu_budget_bytes=gpu_budget, kv_ratio=0.6)
                self._log(f"AVMP active: KV/SSM pools, "
                          f"{gpu_budget / (1024**3):.1f}GB GPU budget")
            except Exception as e:
                self._log(f"AVMP init failed: {e}", level="warn")

        # 7. R35: Virtual tensor pool for elastic GPU/CPU memory
        if feature_flags.get("use_virtual_tensor"):
            try:
                from forge.engine.memory.virtual_tensor import VirtualTensorPool
                budget = 10 * 1024**3  # 10GB default, leave room for weights
                if self.device.type == "cuda":
                    free_b, _ = torch.cuda.mem_get_info(0)
                    budget = int(free_b * 0.7)
                self.vtensor_pool = VirtualTensorPool(gpu_budget_bytes=budget)
                self._log(f"VirtualTensorPool active: {budget / (1024**3):.1f}GB budget")
            except Exception as e:
                self._log(f"VirtualTensorPool init failed: {e}", level="warn")

    # Fallback order for KV cache init failures (OOM, unsupported backend, etc.)
    _KV_FALLBACK_CHAIN = {
        "rotorquant": ["s4r", "hadamard_int4", "standard", "cpu_offload"],
        "hadamard_int4": ["s4r", "standard", "cpu_offload"],
        "paged": ["s4r", "standard", "cpu_offload"],
        "compressed": ["s4r", "standard", "cpu_offload"],
        "snapkv": ["s4r", "standard", "cpu_offload"],
        "filler": ["snapkv", "s4r", "standard", "cpu_offload"],
        "snapkv_4bit": ["s4r", "standard", "cpu_offload"],
        "paged_eviction": ["s4r", "standard", "cpu_offload"],
        "xquant": ["s4r", "standard", "cpu_offload"],
        "hqe_kv": ["s4r", "standard", "cpu_offload"],
        "spectral": ["s4r", "standard", "cpu_offload"],
        "residual_stream": ["s4r", "standard", "cpu_offload"],
        "hyquant": ["s4r", "hadamard_int4", "standard", "cpu_offload"],
        "evo_sparse": ["snapkv", "s4r", "standard", "cpu_offload"],
        "vegas": ["snapkv", "s4r", "standard", "cpu_offload"],
        "hisparse": ["cpu_offload", "s4r", "standard"],
        "capture": ["cpu_offload", "s4r", "standard"],
        "vtoken": ["paged_eviction", "snapkv", "standard", "cpu_offload"],
        "auto_context": ["s4r", "standard", "cpu_offload"],
        "s4r": ["standard", "cpu_offload"],
        "standard": ["cpu_offload"],
        "cpu_offload": [],
        "streaming": [],
    }

    def _activate_kv_cache(self, kv_cache, kv_cache_tokens):
        cfg = getattr(self.model, "config", None)
        n_heads = getattr(cfg, "n_heads", 12)
        n_kv = getattr(cfg, "n_kv_heads", 2) or n_heads
        head_dim = getattr(cfg, "head_dim", None) or (
            getattr(cfg, "d_model", 1536) // n_heads)
        max_seq = getattr(cfg, "max_seq_len", 4096)
        if kv_cache_tokens is not None and kv_cache_tokens < max_seq:
            self._log(f"KV cache limited to {kv_cache_tokens} tokens "
                      f"(was {max_seq})")
            max_seq = kv_cache_tokens

        # VRAM-aware cap: don't allocate KV cache larger than free VRAM allows
        if self.device.type == "cuda":
            try:
                from forge.runtime.vram_manager import VRAMManager
                n_layers = getattr(cfg, "n_layers", 16)
                dtype_bytes = self.dtype.itemsize if hasattr(self.dtype, 'itemsize') else 2
                vram_mgr = VRAMManager(safety_margin_gb=0.5)
                vram_max = vram_mgr.max_gen_tokens(
                    n_layers=n_layers, n_heads=n_kv,
                    head_dim=head_dim, dtype_bytes=dtype_bytes,
                    overhead_mb=256,
                )
                if vram_max < max_seq:
                    self._log(f"VRAM-aware KV cap: {vram_max} tokens "
                              f"(was {max_seq}, free VRAM limited)")
                    max_seq = vram_max
            except Exception as e:
                self._log(f"VRAM-aware KV sizing skipped: {e}", level="warn")

        # Try the requested KV cache, then fall back through progressively
        # simpler caches on OOM or unsupported-backend errors.
        chain = [kv_cache] + self._KV_FALLBACK_CHAIN.get(kv_cache, ["standard"])
        last_error = None
        for try_cache in chain:
            try:
                # Build + init into a local first: self.kv_cache must never
                # reference a half-initialized cache (a concurrent stats()
                # read crashed on missing attributes during the swap).
                cache = build_kv_cache(try_cache)
                cache.init(n_heads, head_dim, n_kv, max_seq,
                           str(self.device), self.dtype)
                self.kv_cache = cache
                # R50-4: filler eviction needs the tokenizer's filler id set.
                if try_cache == "filler" and hasattr(cache, "set_filler_ids"):
                    try:
                        from forge.engine.kv.filler_kv import filler_token_ids
                        cache.set_filler_ids(filler_token_ids(self.tokenizer))
                    except Exception as e:
                        self._log(f"Filler id set skipped: {e}", level="warn")
                # Track active KV bits for OOM-recovery fallback (s4r 4-bit, etc.)
                self._active_kv_bits = getattr(cache, "bits", 8)
                self._active_kv_cache_name = try_cache
                if try_cache != kv_cache:
                    self._log(
                        f"KV cache fallback: '{kv_cache}' failed, "
                        f"using '{try_cache}' instead", level="warn")
                self._log(f"KV cache: {cache.info()}")
                return
            except (torch.cuda.OutOfMemoryError, RuntimeError, ImportError,
                    ValueError) as e:
                last_error = e
                self._log(
                    f"KV cache '{try_cache}' init failed: {e}",
                    level="warn")
                self._clear_cuda_cache()
        # All fallbacks failed — this should be extremely rare
        raise ActivationError(
            f"All KV cache strategies failed (last: {last_error})",
            context={"requested": kv_cache, "tried": chain},
            suggestion="Try kv_cache='cpu_offload' or reduce max_seq_len.")

    def _activate_decoding(self, decoding):
        decode_kwargs = {}
        if decoding == "mtp_selfspec":
            decode_kwargs["k"] = 4
            if hasattr(self.model, "mtp_head"):
                decode_kwargs["mtp_module"] = self.model.mtp_head
        elif decoding == "eagle3":
            if hasattr(self.model, "eagle_head"):
                decode_kwargs["eagle_head"] = self.model.eagle_head
            elif self.checkpoint_path:
                eagle_path = self.checkpoint_path.replace(
                    ".safetensors", ".eagle3.safetensors")
                if os.path.exists(eagle_path):
                    from forge.decoding.eagle import add_eagle3_to_model
                    head = add_eagle3_to_model(self.model)
                    from safetensors.torch import load_file
                    head.load_state_dict(load_file(eagle_path))
                    head = head.to(self.device)
                    decode_kwargs["eagle_head"] = head
                    self._log(f"EAGLE-3 head loaded from {eagle_path}")
            decode_kwargs.setdefault("draft_length", 4)
        elif decoding == "self_speculative_sparse":
            # R39-4: same model as draft+target with sparse attention.
            # Defaults: draft_len=4, sparse_k=64. Can be overridden via
            # engine.activate(decoding="self_speculative_sparse",
            #                  draft_len=8, sparse_k=128).
            decode_kwargs.setdefault("draft_len", 4)
            decode_kwargs.setdefault("sparse_k", 64)
        elif decoding == "uno":
            # R49-1: Uno diffusion-augmented block decoding (arXiv:2609.04010).
            # Lossless Psi-Spec verification; n-gram proposer by default.
            # Overrides: engine.activate(decoding="uno", block_size=8,
            #                            entropy_stop=0.1).
            decode_kwargs.setdefault("block_size", 4)
            decode_kwargs.setdefault("entropy_stop", None)
        elif decoding == "dola":
            # R50-2: DoLa self-contrastive decoding (ICLR 2024). Per-step
            # auto layer selection by default; pin with
            # engine.activate(decoding="dola", early_layer=7) or a custom
            # candidate list via early_candidates=[...].
            decode_kwargs.setdefault("early_layer", None)
            decode_kwargs.setdefault("early_candidates", None)
            decode_kwargs.setdefault("candidate_top_k", 64)
        self.decoding = build_decoding(decoding, **decode_kwargs)
        self._log(f"Decoding: {self.decoding.name}")

    def _activate_acceleration(self, acceleration):
        if acceleration == "cuda_graph" and self.device.type == "cuda":
            from forge.runtime.cuda_graph import CudaGraphRunner
            self._graph_runner = CudaGraphRunner(
                self.model, batch_size=1, seq_len=1,
                device=str(self.device), use_cache=True)
            self._graph_runner.capture()
            self.acceleration = "cuda_graph"
            self._log("CUDA graphs: active")
        elif acceleration == "megakernel" and self.device.type == "cuda":
            from forge.decoding.megakernel import CompiledMegakernelDecode
            self._megakernel = CompiledMegakernelDecode(
                self.model, device=str(self.device))
            try:
                self._megakernel.capture()
                self.acceleration = "megakernel"
                self._log("Megakernel decode: active (compiled + graph)")
            except Exception as e:
                self._log(f"Megakernel decode: failed ({e}), falling back",
                          level="warn")
                self._megakernel = None
                self.acceleration = None
        elif acceleration == "flex_decoding" and self.device.type == "cuda":
            from forge.engine.attention.flex_decoding import FlexDecodingWrapper
            self._flex_decoding = FlexDecodingWrapper()
            if self._flex_decoding.apply(self.model):
                self.acceleration = "flex_decoding"
            else:
                self._flex_decoding = None
                self.acceleration = None
        elif acceleration == "airllm_streaming":
            AirLLMStreamer.setup(self)
        else:
            self._graph_runner = None
            self.acceleration = None

    def _activate_compile_runtime(self, feature_flags):
        # torch.compile
        if feature_flags["use_compile"] and self.device.type == "cuda":
            try:
                self.model = torch.compile(
                    self.model, mode="reduce-overhead", dynamic=True)
                self._log("torch.compile: active (reduce-overhead)")
            except Exception as e:
                self._log(f"torch.compile: failed ({e})", level="warn")

        # Triton fused conv kernel
        if feature_flags["use_triton_conv"] and self.device.type == "cuda":
            try:
                from forge.decoding.triton_conv import patch_conv_layers
                patch_conv_layers(self.model)
            except Exception as e:
                self._log(f"Triton conv: failed ({e})", level="warn")

        # Prefix caching — bounded LRU by default; ChunkedPrefixCache
        # (LMCache-style rolling-hash, R&D14) when use_chunked_prefix_cache.
        if feature_flags["use_prefix_cache"]:
            use_chunked = feature_flags.get("use_chunked_prefix_cache", False)
            if use_chunked:
                if not isinstance(self._prefix_cache, ChunkedPrefixCache):
                    self._prefix_cache = ChunkedPrefixCache(max_entries=64)
                self._log("Prefix caching: active (ChunkedPrefixCache, "
                          "256-token rolling hash)")
            else:
                if not isinstance(self._prefix_cache, LRUPrefixCache):
                    self._prefix_cache = LRUPrefixCache(max_entries=64)
                self._log("Prefix caching: active (LRU, max 64 entries)")
        else:
            self._prefix_cache = None

        # CacheBlend (R&D14): non-prefix KV reuse for RAG / tool-use.
        if feature_flags.get("use_cache_blend", False):
            if not isinstance(self._cache_blend, CacheBlend):
                self._cache_blend = CacheBlend()
            self._log("CacheBlend: active (non-prefix KV reuse)")
        else:
            self._cache_blend = None

        # Chunked prefill
        self._chunked_prefill = None
        if (feature_flags["use_chunked_prefill"]
                and not feature_flags["use_hybrid_prefill"]):
            from forge.engine.prefill import ChunkedPrefiller
            self._chunked_prefill = ChunkedPrefiller(
                self.model, chunk_size=512, device=str(self.device))
            self._log("Chunked prefill: active (chunk_size=512)")

    def _apply_feature_registry(self, feature_flags):
        """Activate all registered inference features in a single pass.

        Iterates ``_FEATURE_REGISTRY`` (see ``feature_registry.py``) and
        invokes each enabled feature's handler.  Replaces the 11 per-category
        ``_activate_*`` methods (~500 lines) with one declarative loop.

        Each handler performs its own lazy import, instantiation, and
        ``.apply(model)`` call, then returns a status message (or ``None``).
        Exceptions are caught per-feature so one failure doesn't block the rest.
        """
        for spec in _FEATURE_REGISTRY:
            if not feature_flags.get(spec.flag, False):
                continue
            if spec.cuda_only and self.device.type != "cuda":
                continue
            try:
                message = spec.handler(self, feature_flags)
                if message:
                    self._log(message)
            except Exception as e:
                self._log(f"{spec.flag}: failed ({e})", level="warn")

    def _finalize_activation(self, feature_flags):
        # Warmup — pre-run a dummy token to initialize CUDA kernels
        if feature_flags["warmup"] and self.device.type == "cuda" and not self._needs_streaming:
            self._warmup()

        if self.device.type == "cuda":
            vram_free, vram_total = torch.cuda.mem_get_info(self.device)
            used_gb = (vram_total - vram_free) / 1e9
            free_gb = vram_free / 1e9
            self._log(f"VRAM: {used_gb:.2f} GB used, {free_gb:.2f} GB free",
                      level="profile")

    @torch.no_grad()
    def _warmup(self):
        """Pre-compile all CUDA kernels with dummy forward passes.

        The first real generation triggers JIT compilation of CUDA kernels,
        cuDNN algorithm selection, and memory pool initialization. This warmup
        runs a multi-token dummy pass through every layer type (conv + attention)
        with KV cache enabled, so all kernel variants are compiled upfront.

        This reduces the Layer 0 cold start from ~300ms (JIT compile) to ~1ms.
        """
        try:
            vocab_size = getattr(self.model, 'config', None)
            vocab_size = getattr(vocab_size, 'vocab_size', 65536) if vocab_size else 65536
            dummy = torch.randint(0, min(vocab_size, 32767), (1, 4),
                                  device=self.device, dtype=torch.long)
            with torch.inference_mode():
                self.model(dummy, use_cache=True)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self._log("Warmup: all CUDA kernels pre-compiled "
                      "(conv + attn + KV cache)")
        except Exception as e:
            self._log(f"Warmup: skipped ({e})", level="warn")

    # Fallback order: if a quantization mode fails, try the next lower one.
    _QUANT_FALLBACK_CHAIN = {
        "nvfp4": ["w8a8", "fp8", "int8", "int4", None],
        "awq_fp4": ["nvfp4", "w8a8", "int8", "int4", None],
        "nanoquant": ["awq_fp4", "nvfp4", "int4", None],
        "btc": ["awq_fp4", "nvfp4", "int4", None],
        "ternary_ptq": ["awq_fp4", "int4", None],
        "forge_quant": ["nvfp4", "w8a8", "int8", "int4", None],
        "grinqh": ["forge_quant", "int4", None],
        "mixllm": ["forge_quant", "int8", "int4", None],
        "acbq": ["int4", None],
        "w8a8": ["fp8", "int8", "int4", None],
        "fp8": ["int8", "int4", None],
        "int8": ["int4", None],
        "int4": [None],
        # Quamba2: SSM-specific W4A8. Falls back to w8a8 (same bit-budget for
        # non-SSM layers) then the standard chain. If no SSM blocks are found,
        # quantize_model_quamba2 is a no-op (returns 0) and we fall through.
        "quamba2": ["w8a8", "fp8", "int8", "int4", None],
        None: [],
    }

    def _save_original_weights(self):
        """Snapshot model weights to CPU RAM before in-place quantization.

        Stores a CPU-side copy of every parameter so ``activate(quantize=None)``
        can restore the original bf16 weights after a prior quantization pass.
        Without this, quantization is "sticky" — re-activating without
        ``quantize=`` leaves the quantized (slower) weights in place.
        """
        self._original_weights = {}
        for name, param in self.model.named_parameters():
            self._original_weights[name] = param.data.cpu().clone()
        self._log(f"Saved {len(self._original_weights)} weight tensors "
                  f"for quantization restore")

    def _restore_original_weights(self):
        """Restore model weights from the CPU snapshot saved by _save_original_weights."""
        if not hasattr(self, '_original_weights') or self._original_weights is None:
            self._log("No original weights snapshot to restore", level="warn")
            return
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._original_weights:
                    saved = self._original_weights[name]
                    param.data = saved.to(param.device, dtype=param.dtype)
        self._clear_cuda_cache()
        self._log(f"Restored {len(self._original_weights)} weight tensors "
                  f"from snapshot")

    def _apply_quantization(self, mode: str):
        """Apply weight-only quantization with automatic fallback.

        If the requested mode fails (unsupported hardware, OOM, ImportError),
        falls back through progressively lower-bit modes, finally to
        unquantized bf16. Never leaves the model in a broken state.
        """
        chain = [mode] + self._QUANT_FALLBACK_CHAIN.get(mode, [None])
        last_error = None
        for try_mode in chain:
            if try_mode is None:
                self._log(
                    f"Quantization '{mode}' failed; falling back to "
                    f"unquantized bf16 (no compression)",
                    level="warn")
                if last_error:
                    self._log(f"  Last error: {last_error}", level="warn")
                return  # leave model unquantized
            try:
                self._apply_quantization_single(try_mode)
                if try_mode != mode:
                    self._log(
                        f"Quantization fallback: '{mode}' failed, "
                        f"using '{try_mode}' instead", level="warn")
                return
            except (ImportError, RuntimeError, ValueError,
                    torch.cuda.OutOfMemoryError) as e:
                last_error = e
                self._log(
                    f"Quantization '{try_mode}' failed: {e}",
                    level="warn")
                self._clear_cuda_cache()
        # Should not reach here (chain always ends with None), but just in case:
        raise ConfigurationError(
            f"All quantization modes failed for '{mode}'",
            context={"mode": mode, "last_error": str(last_error)},
            suggestion="Try quantize=None to run unquantized.")

    def block_reconstruct(
        self,
        model_orig: nn.Module | None = None,
        calibration_data: torch.Tensor | None = None,
        n_iters: int = 50,
        lr: float = 0.05,
        mode: str = "progressive",
        kl_iters: int = 20,
        verbose: bool = True,
    ) -> dict[int, float]:
        """Run block-level reconstruction on the quantized model.

        Optimizes quantized layer parameters (scales + latent binary matrices
        via STE) to minimize block-level output error against the original
        model. This is the key technique from NanoQuant (ICML 2026) for
        improving extreme low-bit PTQ quality.

        Modes:
            - "progressive": sequential block reconstruction (default)
            - "standard": independent block reconstruction
            - "error_mitigation": sequential with error propagation mitigation
              (adjusts targets to account for accumulated quantization errors)
            - "kl_calib": progressive block recon + model-level KL calibration

        Args:
            model_orig: original unquantized model (for targets). If None,
                loads from the same checkpoint.
            calibration_data: (batch, seq_len) input_ids for calibration.
                If None, generates random tokens from the model's vocab.
            n_iters: optimization iterations per block
            lr: learning rate for scale optimization
            mode: reconstruction mode (see above)
            kl_iters: KL calibration iterations (only for "kl_calib" mode)
            verbose: print progress

        Returns:
            dict mapping block_idx → final loss

        Example:
            >>> engine = ForgeEngine.from_checkpoint("model.pt", quantize="nanoquant")
            >>> engine.block_reconstruct(n_iters=50, lr=0.05, mode="kl_calib")
        """
        self._require_awake()
        from forge.engine.quant.block_recon import BlockReconstructor

        # Get or load original model
        if model_orig is None:
            # Reload from checkpoint if available
            ckpt_path = getattr(self, 'checkpoint_path', None)
            cfg = getattr(self, 'config', None)
            if ckpt_path is not None and cfg is not None:
                from forge.model_loader import ModelLoader
                model_orig = ModelLoader.build_model_fast(
                    cfg, checkpoint_path=str(ckpt_path))
            else:
                self._log("block_reconstruct: no original model available "
                          "(need model_orig or checkpoint_path)", level="warn")
                return {}
        model_orig = model_orig.to(self.device).eval()

        # Generate calibration data if not provided
        if calibration_data is None:
            cfg = getattr(self.model, 'config', None)
            vocab_size = getattr(cfg, 'vocab_size', 32000) if cfg else 32000
            calibration_data = torch.randint(0, vocab_size, (4, 128),
                                             device=self.device)
        calibration_data = calibration_data.to(self.device)

        recon = BlockReconstructor(
            model_orig=model_orig,
            model_quant=self.model,
            calibration_data=calibration_data,
            device=str(self.device),
        )

        if mode == "progressive":
            results = recon.reconstruct_progressive(
                n_iters=n_iters, lr=lr, verbose=verbose)
        elif mode == "error_mitigation":
            results = recon.reconstruct_with_error_mitigation(
                n_iters=n_iters, lr=lr, verbose=verbose)
        elif mode == "kl_calib":
            results = recon.reconstruct_progressive(
                n_iters=n_iters, lr=lr, verbose=verbose)
            if kl_iters > 0:
                recon.calibrate_kl(n_iters=kl_iters, lr=lr * 0.1,
                                   verbose=verbose)
        else:
            results = recon.reconstruct(
                n_iters=n_iters, lr=lr, verbose=verbose)

        self._log(f"Block reconstruction complete: {len(results)} blocks, "
                  f"avg loss={sum(results.values())/max(len(results),1):.4f}")
        return results

    def _apply_quantization_single(self, mode: str):
        """Apply a single quantization mode (no fallback). Raises on failure."""
        if mode == "int8":
            from forge.quant.inference_quant import quantize_model_int8
            fast = torch.cuda.is_available()
            quantize_model_int8(self.model, fast=fast)
        elif mode == "int4":
            from forge.quant.inference_quant import quantize_model_int4
            quantize_model_int4(self.model, group_size=128)
        elif mode == "fp8":
            from forge.quant.fp8_infer import quantize_model_fp8
            quantize_model_fp8(self.model)
        elif mode == "w8a8":
            from forge.engine.quant.w8a8_quant import quantize_model_w8a8
            w8a8_mode = getattr(self.model, 'config', None)
            w8a8_mode = getattr(w8a8_mode, 'w8a8_mode', 'int8') if w8a8_mode else 'int8'
            quantize_model_w8a8(self.model, mode=w8a8_mode)
            self._log(f"W8A8 quantization: {w8a8_mode}")
        elif mode == "nvfp4":
            from forge.engine.quant.nvfp4_quant import quantize_model_nvfp4
            cfg = getattr(self.model, 'config', None)
            block_size = getattr(cfg, 'nvfp4_block_size', 32) if cfg else 32
            w4a8 = getattr(cfg, 'nvfp4_w4a8', False) if cfg else False
            quantize_model_nvfp4(self.model, block_size=block_size, w4a8=w4a8)
            self._log(f"NVFP4 quantization: active (Blackwell native FP4, "
                      f"block={block_size}, w4a8={w4a8})")
        elif mode == "forge_quant":
            from forge.engine.quant.forge_quant import quantize_model_forge_quant
            cfg = getattr(self.model, 'config', None)
            gs = getattr(cfg, 'forge_quant_group_size', 128) if cfg else 128
            sr = getattr(cfg, 'forge_quant_sparse_ratio', 0.10) if cfg else 0.10
            n_q = quantize_model_forge_quant(self.model, group_size=gs, sparse_ratio=sr)
            self._log(f"ForgeQuant: {n_q} layers quantized (INT4 dense + 2-bit sparse, "
                      f"group={gs}, sparse_ratio={sr}). SM120-tuned.")
        elif mode == "grinqh":
            from forge.engine.quant.grinqh import quantize_model_grinqh
            cfg = getattr(self.model, 'config', None)
            gs = getattr(cfg, 'grinqh_group_size', 128) if cfg else 128
            target_bits = getattr(cfg, 'grinqh_target_bits', 2.5) if cfg else 2.5
            n_q = quantize_model_grinqh(model=self.model, group_size=gs,
                                        target_effective_bits=target_bits)
            self._log(f"GRINQH: {n_q} layers quantized (effective {target_bits}-bit, "
                      f"group={gs})")
        elif mode == "mixllm":
            from forge.engine.quant.mixllm import quantize_model_mixllm
            cfg = getattr(self.model, 'config', None)
            gs = getattr(cfg, 'mixllm_group_size', 128) if cfg else 128
            hf = getattr(cfg, 'mixllm_high_fraction', 0.10) if cfg else 0.10
            n_q = quantize_model_mixllm(self.model, group_size=gs, high_fraction=hf)
            self._log(f"MixLLM: {n_q} layers quantized (global mixed-precision, "
                      f"high_fraction={hf})")
        elif mode == "acbq":
            from forge.engine.quant.acbq import quantize_model_acbq
            cfg = getattr(self.model, 'config', None)
            gs = getattr(cfg, 'acbq_group_size', 128) if cfg else 128
            attn_bits = getattr(cfg, 'acbq_attn_bits', 4) if cfg else 4
            ffn_bits = getattr(cfg, 'acbq_ffn_bits', 4) if cfg else 4
            n_q = quantize_model_acbq(self.model, group_size=gs,
                                      attn_bits=attn_bits, ffn_bits=ffn_bits)
            self._log(f"ACBQ: {n_q} layers quantized (adaptive cross-block, "
                      f"attn={attn_bits}bit, ffn={ffn_bits}bit)")
        elif mode == "quamba2":
            from forge.quant.quamba2 import quantize_model_quamba2
            cfg = getattr(self.model, "config", None)
            gs = getattr(cfg, "quamba2_group_size", 128) if cfg else 128
            alpha = getattr(cfg, "quamba2_smoothquant_alpha", 0.5) if cfg else 0.5
            n_q = quantize_model_quamba2(self.model, group_size=gs,
                                         smoothquant_alpha=alpha)
            self._log(f"Quamba2: {n_q} SSM blocks quantized (W4A8, "
                      f"group={gs}, smoothquant_alpha={alpha}). "
                      f"SSM core (A_log, dt, scan) kept in FP16.")
        elif mode == "awq_fp4":
            from forge.engine.quant.novel_quant_r46 import (
                collect_activations as _collect_acts_r46,
            )
            from forge.engine.quant.novel_quant_r46 import (
                quantize_model_awq_fp4,
            )
            # AWQ-FP4 needs calibration data — use a short forward pass
            cfg = getattr(self.model, "config", None)
            block_size = getattr(cfg, "nvfp4_block_size", 32) if cfg else 32
            # Generate calibration input from model's vocab
            vocab_size = getattr(cfg, "vocab_size", 32000) if cfg else 32000
            calib_ids = torch.randint(0, vocab_size, (4, 128))
            self._log("AWQ-FP4: collecting calibration activations...")
            acts = _collect_acts_r46(self.model, calib_ids.to(self.device),
                                     n_samples=128, device=str(self.device))
            n_q = quantize_model_awq_fp4(self.model, acts, block_size=block_size,
                                         verbose=True)
            self._log(f"AWQ-FP4: {n_q} layers quantized (activation-aware FP4, "
                      f"block={block_size}). Best quality FP4 in R&D rounds.")
        elif mode == "nanoquant":
            from forge.engine.quant.novel_quant_r48 import quantize_model_nanoquant
            cfg = getattr(self.model, "config", None)
            rank = getattr(cfg, "nanoquant_rank", 128) if cfg else 128
            n_q = quantize_model_nanoquant(self.model, rank=rank,
                                           admm_iters=50, verbose=True)
            self._log(f"NanoQuant: {n_q} layers quantized (low-rank binary, "
                      f"rank={rank}). Sub-1-bit extreme compression.")
        elif mode == "btc":
            from forge.engine.quant.novel_quant_r48 import quantize_model_btc
            cfg = getattr(self.model, "config", None)
            K = getattr(cfg, "btc_codebook_size", 256) if cfg else 256
            n_q = quantize_model_btc(self.model, codebook_size=K,
                                     use_rotation=True, verbose=True)
            self._log(f"BTC-LLM: {n_q} layers quantized (binary codebook, "
                      f"K={K}). Sub-1-bit via pattern clustering.")
        elif mode == "ternary_ptq":
            from forge.engine.quant.novel_quant_r48 import (
                collect_activations as _collect_acts_r48,
            )
            from forge.engine.quant.novel_quant_r48 import (
                quantize_model_ternary_ptq,
            )
            cfg = getattr(self.model, "config", None)
            vocab_size = getattr(cfg, "vocab_size", 32000) if cfg else 32000
            calib_ids = torch.randint(0, vocab_size, (4, 128))
            acts = _collect_acts_r48(self.model, calib_ids.to(self.device),
                                     n_samples=128, device=str(self.device)) \
                if hasattr(_collect_acts_r48, '__call__') else None
            # Use R46's collect_activations (same signature)
            from forge.engine.quant.novel_quant_r46 import collect_activations as _ca46
            acts = _ca46(self.model, calib_ids.to(self.device),
                         n_samples=128, device=str(self.device))
            n_q = quantize_model_ternary_ptq(self.model, activations=acts,
                                             refine_iters=20, verbose=True)
            self._log(f"TernaryPTQ: {n_q} layers quantized (1.58-bit ternary, "
                      f"Hessian-refined scales). BitNet-style PTQ.")
        else:
            raise ConfigurationError(
                f"Unknown quantization mode: {mode}",
                context={"mode": mode},
                suggestion="Use one of: int8, int4, fp8, w8a8, nvfp4, "
                           "forge_quant, grinqh, mixllm, acbq, quamba2, "
                           "awq_fp4, nanoquant, btc, ternary_ptq")

    # ── Generation ────────────────────────────────────────────────────────

    _LOW_VRAM_THRESHOLD_BYTES = 500 * 1024 * 1024  # 500 MB

    def _check_vram_and_offload_if_needed(self):
        """Proactively check free VRAM and switch to CPU offload KV if low.

        If free VRAM drops below 500 MB, switches the KV cache strategy to
        ``cpu_offload`` before an OOM can occur. This avoids the expensive
        OOM recovery path (which retries the entire generation).
        """
        if self.device.type != "cuda":
            return
        # Memory models (FluxLM) have no KV cache — nothing to offload
        if _is_memory_model(self.model):
            return
        try:
            free, _ = torch.cuda.mem_get_info(self.device)
        except Exception:
            return
        if free < self._LOW_VRAM_THRESHOLD_BYTES:
            kv_info = self.kv_cache.info() if self.kv_cache else {}
            kv_type = kv_info.get("type", kv_info.get("name", "none"))
            if kv_type != "cpu_offload":
                self._log(
                    f"Low VRAM ({free / 1e9:.2f} GB free < "
                    f"{self._LOW_VRAM_THRESHOLD_BYTES / 1e9:.2f} GB) — "
                    f"proactively switching KV cache to cpu_offload",
                    level="warn")
                self._clear_cuda_cache()
                try:
                    self._activate_kv_cache("cpu_offload", None)
                except Exception as e:
                    self._log(
                        f"Proactive CPU offload failed: {e}", level="warn")

    # Evolution-discovered creative sampling preset (score 10.45):
    # High temperature + high top_p + moderate top_k + low penalties.
    # Use for creative/diverse generation tasks.
    CREATIVE_SAMPLING = {
        "temperature": 1.98, "top_p": 0.989, "top_k": 69,
        "repetition_penalty": 1.011, "frequency_penalty": 0.014,
    }

