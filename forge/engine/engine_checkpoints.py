"""Checkpoint loading and auto-detection mixin for ForgeEngine."""
import json
import os
from pathlib import Path

import torch.nn.functional as F

from .engine_common import *  # noqa: F403
from .errors import CheckpointError  # noqa: F401
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _fast_load_vram_required,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    logger,
)


class _CheckpointLoadingMixin:
    @classmethod
    def from_checkpoint(cls, checkpoint: str, config_name: str = "forgelm_v2",
                        tokenizer_path: str | None = None,
                        device: str = "cuda",
                        auto_activate: bool = True,
                        config_overrides: dict | None = None,
                        **kwargs) -> "ForgeEngine":
        """Build engine from a KeyStack checkpoint.

        Auto-checks VRAM capacity and picks the best loading strategy:
          1. Pre-quantized BitNet → int8 direct load
          2. Fits in VRAM → fast meta-init load
          3. Fits with hybrid offload → conv on CPU, attention on GPU
          4. Too large → AirLLM layer-streaming (meta device + shard loading)

        Args:
            auto_activate: If True (default), automatically calls
                ``activate_optimal()`` with keystack-aware overrides after
                loading. Detected features (MTP, value_residual, etc.) are
                auto-enabled. Set to False for manual activation.
            config_overrides: Optional dict of config field overrides applied
                to the base config before model construction (e.g.
                ``{"use_mamba3": True}`` to enable Mamba-3 warm start).
        """
        from forge.config import get_config
        from research.tokenizer_cache import get_tokenizer
        from forge.runtime.configure import configure

        # Env-side runtime config (CUDA alloc conf etc.) — must run before
        # the first CUDA allocation below, else expandable_segments is a
        # no-op (previously set in __init__, which runs after weight load).
        configure()

        cfg = get_config(config_name, device=device,
                         **(config_overrides or {}))

        # Tokenizer auto-dispatch: the canonical LFM tokenizer (vocab 65536)
        # cannot tokenize for Qwen-family checkpoints (vocab 151936). Dispatch
        # on the config's vocab so HF-style checkpoints get a matching
        # tokenizer (explicit tokenizer_path always wins).
        tok_path = tokenizer_path or _tokenizer_for_vocab(cfg.vocab_size)
        # Resolve the tokenizer lazily on a daemon thread — the ~0.4s fast-path
        # load overlaps with weight I/O instead of serializing before it.
        _tok_box: dict = {}

        def _load_tok():
            try:
                _tok_box["t"] = get_tokenizer(tok_path)
            except Exception as e:
                _tok_box["e"] = e

        _tok_thread = threading.Thread(target=_load_tok, daemon=True)
        _tok_thread.start()

        def _tokenizer():
            _tok_thread.join()
            if "e" in _tok_box:
                raise _tok_box["e"]
            return _tok_box["t"]

        # GGUF checkpoint detection — route to ForgeLoader for dequant + load
        if str(checkpoint).lower().endswith(".gguf"):
            return cls._load_gguf_checkpoint(
                checkpoint, _tokenizer(), device, auto_activate, **kwargs)

        ckpt_size = _checkpoint_size_cache.get(checkpoint)
        if ckpt_size is None:
            try:
                ckpt_size = Path(checkpoint).stat().st_size
            except OSError as e:
                raise CheckpointError(
                    f"Cannot access checkpoint '{checkpoint}': {e}",
                    context={"checkpoint": checkpoint},
                    suggestion="Verify the path exists and is a valid "
                               "safetensors file or directory of shards.",
                ) from e
            with _ckpt_cache_lock:
                _checkpoint_size_cache[checkpoint] = ckpt_size
                while len(_checkpoint_size_cache) > _CKPT_CACHE_MAX:
                    _checkpoint_size_cache.popitem(last=False)
        dev = torch.device(device)
        vram_free, _ = cls._memory_info(dev)
        needed = _fast_load_vram_required(ckpt_size)
        fits = vram_free > needed

        metadata = cls._read_checkpoint_metadata(checkpoint)
        is_prequant = metadata.get("_bitnet_prequant") == "1"

        # Architecture guard: a checkpoint whose tensor shapes don't match
        # the selected config can NEVER load — the old behavior (fall through
        # to AirLLM streaming with the same wrong config) silently produced
        # a random-weight model that "loaded" but generated garbage.
        try:
            cls._validate_checkpoint_config(checkpoint, cfg, config_name)
        except CheckpointError:
            detected = cls._detect_config_from_header(checkpoint)
            if detected and detected != config_name:
                logger.info(
                    "Checkpoint does not match config '%s' — auto-detected "
                    "matching preset '%s' from checkpoint shapes",
                    config_name, detected)
                config_name = detected
                cfg = get_config(detected, device=device)
                cls._validate_checkpoint_config(checkpoint, cfg, config_name)
            else:
                raise

        engine = None
        try:
            if is_prequant:
                engine = cls._load_prequant(
                    cfg, checkpoint, _tokenizer(), device, metadata, **kwargs)
            elif fits:
                engine = cls._load_standard(
                    cfg, checkpoint, _tokenizer(), device, **kwargs)
            else:
                # Check if hybrid offload can bridge the gap
                engine = cls._load_with_fallback(
                    cfg, checkpoint, _tokenizer(), device,
                    ckpt_size, vram_free, **kwargs)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "size mismatch" in str(e).lower():
                # Weights don't fit the built model — streaming the same
                # shapes would silently produce a random-weight model.
                raise CheckpointError(
                    f"checkpoint tensors do not match config "
                    f"'{config_name}': {e}",
                    context={"checkpoint": checkpoint,
                             "config": config_name},
                    suggestion="Select the config matching this checkpoint."
                ) from e
            # If the primary load path fails (OOM, corrupt weights, etc.),
            # fall through to the streaming path as a last resort.
            logger.warning(
                "Load path failed (%s), falling back to AirLLM streaming...", e)
            cls._clear_cuda_cache_static(dev)
            engine = cls._load_streaming(
                cfg, checkpoint, _tokenizer(), device,
                ckpt_size, vram_free, **kwargs)

        if auto_activate and engine is not None:
            engine._auto_activate_optimal()
        return engine

    @classmethod
    def from_flux(cls, config=None, checkpoint: str | None = None,
                  tokenizer_path: str | None = None, device: str = "cpu",
                  cuda_primary: bool = False,
                  **_kwargs) -> "ForgeEngine":
        """Load a FluxLM sparse associative-memory model.

        Explicit compatibility path — skips checkpoint-format detection,
        quantization, KV cache and strategy activation entirely (FluxLM
        has no KV cache and no dense weights to quantize; its state is
        the memory itself).

        Args:
            config: FluxConfig instance (default: FluxConfig()).
            checkpoint: optional path to a ``model.snapshot()`` file to
                restore learned memory from.
            tokenizer_path: tokenizer dir (default: auto by vocab_size).
            device: engine device.  "cuda" puts FluxLM's dense readout
                (A/R/c/proto + logits + sem matvec) on GPU; the sparse
                memory tables stay host-resident either way.
            cuda_primary: with device="cuda", move the sparse memory
                itself into GPU open-addressed pair-key tables and use
                the vectorized bulk-ingest path (much faster training +
                V-wide probe prediction).
        """
        from forge.model.flux import FluxConfig, FluxLM
        from research.tokenizer_cache import get_tokenizer

        cfg = config or FluxConfig()
        cfg.device = device
        cfg.cuda_primary = cuda_primary or cfg.cuda_primary
        if checkpoint:
            model = FluxLM.load(checkpoint, device=device,
                                cuda_primary=cuda_primary or None)
            cfg = model.config
        else:
            model = FluxLM(cfg)
        tok_path = tokenizer_path or _tokenizer_for_vocab(cfg.vocab_size)
        tokenizer = get_tokenizer(tok_path)
        # session learning is tagged "live" — revert_tag("live") forgets
        # everything the model picked up during engine use; "live"/"gen"
        # tags also get full-fidelity Hedge rewards (see fast_ingest).
        model.tag = "live"
        engine = cls(model, tokenizer, device=device)
        # set post-init: passing it into __init__ would trigger safetensors
        # KeyStack detection on a pickle; the path is needed for sleep(2)/wake
        engine.checkpoint_path = checkpoint
        engine._log(
            f"FLUX loaded — sparse associative memory "
            f"({sum(len(t) for t in model._tables.values())} cells, "
            f"stream={model._count} tok)")
        return engine

    def _auto_activate_optimal(self):
        """Auto-activate optimal strategies based on detected KeyStack features.

        Inspects ``self.keystack_features`` and enables feature-appropriate
        strategies that are safe and beneficial:
          - MTP detected → mtp_selfspec decoding (2-4x speedup)
          - value_residual detected → use_v0_warm (lossless quality boost)
          - bitnet_prequant → skip quantize (already ternary)
          - CUDA device → enable block fusion + breakable CUDA graphs
          - VRAM-aware: auto-select highest quantization that fits
        """
        overrides = {}

        # torch.compile is broken on some triton/GPU stacks (inductor
        # mis-types scalar args of @triton.jit kernels → InductorError on
        # every forward). FORGE_NO_COMPILE=1 opts out at load time — the
        # GUI fast-load path sets this before importing the engine.
        if os.environ.get("FORGE_NO_COMPILE", "").strip().lower() in (
                "1", "true", "yes"):
            overrides["use_compile"] = False

        # Keystack-aware overrides
        if "mtp" in self.keystack_features:
            overrides["decoding"] = "mtp_selfspec"
        if "value_residual" in self.keystack_features:
            overrides["use_v0_warm"] = True
        # Mamba/SSM hybrids (e.g. ForgeLM V2 Jamba) keep the validated
        # mamba_hybrid profile: no extra weight quant on top of Quamba2,
        # and no attention-only features that assume (k, v) tuple state.
        _lt = getattr(self.config, "layer_types", None) or []
        is_mamba_hybrid = any(lt in ("mamba", "mamba3") for lt in _lt)

        if "bitnet_prequant" in self.keystack_features:
            # Weights are already ternary int8 — no quantize needed
            overrides["quantize"] = None
        elif is_mamba_hybrid:
            # Quamba2 already quantizes SSM blocks in the mamba profile;
            # an extra whole-model quant pass is unvalidated there.
            overrides.setdefault("quantize", None)
        else:
            # Auto-select highest quantization that fits VRAM
            quant = self._auto_select_quantization()
            if quant:
                overrides["quantize"] = quant

        # VRAM-aware feature selection
        if self.device.type == "cuda":
            vram_free, vram_total = self._memory_info(self.device)
            vram_ratio = vram_free / vram_total if vram_total > 0 else 0

            # RotorQuant KV cache: Givens rotation + Lloyd-Max quantization
            # More refined than TurboQuant (block-diagonal vs dense rotation,
            # deferred quantization for zero error compounding during prefill).
            # 3-4 bit KV with 0.94% error, ~8x compression.
            overrides.setdefault("kv_cache", "rotorquant")
            overrides.setdefault("kv_bits", 4)

            if vram_ratio > 0.5 and not is_mamba_hybrid:
                # Ample VRAM — enable aggressive speed features
                overrides["use_block_fusion"] = True
                overrides["use_breakable_cuda_graph"] = True
                overrides["use_learned_prefix_cache"] = True
            elif vram_ratio < 0.25:
                # Tight VRAM — skip graph features, keep RotorQuant + 4-bit
                overrides["use_block_fusion"] = False
                overrides["use_breakable_cuda_graph"] = False

        # Don't activate if model is on meta (streaming mode)
        if self._needs_streaming:
            self._log("Auto-activate skipped (streaming mode)", level="info")
            return

        self._log(f"Auto-activating optimal strategies "
                  f"(overrides: {list(overrides.keys()) or 'none'})")
        self.activate_optimal(**overrides)

    def _auto_select_quantization(self) -> str | None:
        """Pick the highest quantization level that fits available VRAM.

        Priority (highest quality first):
          1. nvfp4 — if Blackwell GPU (native FP4, ~99% quality, 3.8x compression)
             NOW THE DEFAULT on Blackwell — quality is near-lossless and it
             frees VRAM for larger KV cache / longer context.
          2. None (bf16) — only if nvfp4 unavailable and VRAM is ample
          3. fp8 — if Hopper+ (hardware-native FP8)
          4. w8a8 — 2-3x speedup, minimal quality loss
          5. int4 — 4x compression, last resort

        Returns None if no quantization is needed (ample VRAM + no Blackwell).
        """
        if self.device.type != "cuda":
            return None

        # Check GPU capability for hardware-native quantization
        cap = torch.cuda.get_device_capability(self.device)
        # Blackwell = SM 100+ (RTX 5070 = SM 120)
        is_blackwell = cap[0] >= 10
        # Hopper = SM 90 (H100)
        is_hopper = cap[0] >= 9

        # NVFP4 is now the default on Blackwell — near-lossless quality,
        # 3.8x compression, frees VRAM for KV cache / longer context.
        # Only skip if the model is already BitNet (ternary) or explicitly
        # configured to use a different quant.
        if is_blackwell:
            # ForgeQuant (R32-6) is the best quantization for SM120 (RTX 5070):
            # INT4 dense + INT8 sparse outliers, ~3.6 effective bits, better
            # quality than NVFP4 due to outlier preservation, and lower
            # conversion memory overhead (processes layers one at a time).
            # NVFP4 requires holding original + quantized weights simultaneously,
            # which OOMs on 12GB for models >2B params.
            try:
                from forge.engine.quant.forge_quant import quantize_model_forge_quant  # noqa
                # ForgeQuant replaces nn.Linear with ForgeQuantLinear per-layer.
                # Each replacement frees the original bf16 weight and allocates
                # a smaller packed INT4 weight. Peak overhead per layer is just
                # the ForgeQuantLinear storage (~50% of one layer's bf16 weight),
                # NOT the full model. The old 1.3x model-size check was wrong
                # and prevented ForgeQuant on 12GB GPUs with 3B+ models.
                # We need only enough free VRAM for the largest single layer's
                # quantization temp (~256MB for a 2560×10240 in_proj).
                vram_free, _ = self._memory_info(self.device)
                _per_layer_overhead = 256 * 1024 * 1024  # 256MB conservative
                if vram_free > _per_layer_overhead:
                    return "forge_quant"
                # Not enough VRAM even for per-layer conversion
                self._log(
                    f"ForgeQuant skipped: only {vram_free/1e9:.1f}GB free VRAM "
                    f"(need {_per_layer_overhead/1e9:.1f}GB per-layer overhead)",
                    level="warn")
                return "int4"
            except ImportError:
                pass

            # Fallback to NVFP4 if ForgeQuant unavailable (ample VRAM only)
            n_params = sum(p.numel() for p in self.model.parameters())
            model_bytes_bf16 = n_params * 2
            vram_free, _ = self._memory_info(self.device)
            if vram_free > model_bytes_bf16 * 2.5:
                try:
                    from forge.engine.quant.nvfp4_quant import quantize_model_nvfp4  # noqa
                    return "nvfp4"
                except ImportError:
                    pass

            try:
                from forge.quant.fp8_infer import quantize_model_fp8
                return "fp8"
            except ImportError:
                pass

        # Estimate model size in VRAM for non-Blackwell path
        n_params = sum(p.numel() for p in self.model.parameters())
        model_bytes_bf16 = n_params * 2  # bf16 = 2 bytes/param
        vram_free, _ = self._memory_info(self.device)

        # If we have > 2x model size free, no quantization needed
        if vram_free > model_bytes_bf16 * 2:
            return None

        if is_hopper:
            try:
                from forge.quant.fp8_infer import quantize_model_fp8  # noqa
                return "fp8"
            except ImportError:
                pass

        # W8A8: 2-3x speedup, works on all CUDA GPUs with torch._int_mm
        if vram_free > model_bytes_bf16 * 0.6:
            return "w8a8"

        # INT4: 4x compression, last resort for very tight VRAM
        return "int4"

    @staticmethod
    def _read_checkpoint_metadata(checkpoint: str) -> dict:
        """Read safetensors metadata from a checkpoint (single or sharded).

        Uses a module-level cache keyed by (path, mtime) to avoid re-reading
        metadata from the same file on every call.
        """
        try:
            mtime = Path(checkpoint).stat().st_mtime
        except OSError:
            mtime = 0.0
        cache_key = (checkpoint, mtime)
        with _ckpt_cache_lock:
            cached = _checkpoint_metadata_cache.get(cache_key)
        if cached is not None:
            return cached

        from safetensors import safe_open
        metadata = {}
        try:
            with safe_open(checkpoint, framework="pt") as checkpoint_file:
                metadata = checkpoint_file.metadata() or {}
        except (OSError, RuntimeError, ValueError):
            # File missing, corrupt, or not a valid safetensors archive —
            # return empty metadata rather than crashing the engine.
            metadata = {}
        with _ckpt_cache_lock:
            _checkpoint_metadata_cache[cache_key] = metadata
            while len(_checkpoint_metadata_cache) > _CKPT_CACHE_MAX:
                _checkpoint_metadata_cache.popitem(last=False)
        return metadata

    @staticmethod
    def _read_safetensors_header(checkpoint) -> dict | None:
        """Raw safetensors header (tensor name → {shape, dtype, ...}).

        Reads only the JSON header (8-byte length prefix + payload) — no
        torch, no tensor loading. Returns None for non-safetensors files.
        """
        import struct
        try:
            with open(checkpoint, "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                return json.loads(f.read(n))
        except (OSError, ValueError, struct.error):
            return None

    @classmethod
    def _validate_checkpoint_config(cls, checkpoint, cfg, config_name) -> None:
        """Fail fast when a checkpoint cannot possibly match ``cfg``.

        Compares shapes read from the safetensors header (cheap, no tensor
        loading) against the config. Skips silently for non-safetensors or
        sharded checkpoints where per-file shapes are not authoritative.
        """
        path = Path(checkpoint)
        if path.suffix.lower() != ".safetensors" or path.is_dir():
            return
        hdr = cls._read_safetensors_header(checkpoint)
        if not hdr:
            return
        shapes = {k: v["shape"] for k, v in hdr.items()
                  if isinstance(v, dict) and v.get("shape")}
        if not shapes:
            return

        problems: list[str] = []
        # Embedding: ForgeLM "embed.weight" or HF "model.embed_tokens.weight"
        embed = shapes.get("embed.weight") or shapes.get("model.embed_tokens.weight")
        if embed and len(embed) == 2:
            if cfg.vocab_size and embed[0] != cfg.vocab_size:
                problems.append(
                    f"vocab mismatch: checkpoint embed {embed[0]} vs config "
                    f"{cfg.vocab_size}")
            if cfg.d_model and embed[1] != cfg.d_model:
                problems.append(
                    f"d_model mismatch: checkpoint embed is [{embed[0]}, "
                    f"{embed[1]}] vs config d_model={cfg.d_model}")
        # Depth: count distinct block indices (blocks.N / model.layers.N)
        import re
        idx = {int(m.group(1)) for k in shapes
               for m in [re.match(r"(?:blocks|model\.layers)\.(\d+)\.", k)] if m}
        if idx and cfg.n_layers and len(idx) != cfg.n_layers:
            problems.append(
                f"depth mismatch: checkpoint has {len(idx)} transformer "
                f"blocks vs config n_layers={cfg.n_layers}")
        if not problems:
            return

        list(shapes)
        hints = []
        if any(k.startswith("model.") for k in shapes):
            hints.append("transformers/HF-style key names")
        if any("U_latent" in k or "V_latent" in k for k in shapes):
            hints.append("low-rank (SVD/ASVD) factorized weights")
        hint = f" (detected: {', '.join(hints)})" if hints else ""
        raise CheckpointError(
            f"Checkpoint '{Path(checkpoint).name}' does not match config "
            f"'{config_name}': {'; '.join(problems)}.{hint}",
            context={"checkpoint": str(checkpoint), "config": config_name,
                     "problems": problems},
            suggestion="Select the config matching this checkpoint, or "
                       "re-export the checkpoint in ForgeLM KeyStack format.")

    @classmethod
    def _detect_config_from_header(cls, checkpoint) -> str | None:
        """Infer a matching MODEL_CONFIGS preset from checkpoint tensor shapes.

        Reads the safetensors header (no tensor loading), extracts
        (vocab, d_model, n_layers) from the embedding + block indices, and
        returns the first preset whose shapes match exactly. None when no
        preset matches (or the file has no readable header).
        """
        import re

        from forge.config import MODEL_CONFIGS
        path = Path(checkpoint)
        if path.suffix.lower() != ".safetensors" or path.is_dir():
            return None
        hdr = cls._read_safetensors_header(checkpoint)
        if not hdr:
            return None
        shapes = {k: v["shape"] for k, v in hdr.items()
                  if isinstance(v, dict) and v.get("shape")}
        embed = shapes.get("embed.weight") or shapes.get("model.embed_tokens.weight")
        if not embed or len(embed) != 2:
            return None
        idx = {int(m.group(1)) for k in shapes
               for m in [re.match(r"(?:blocks|model\.layers)\.(\d+)\.", k)] if m}
        if not idx:
            return None
        vocab, d_model, n_layers = embed[0], embed[1], len(idx)
        for name, preset in MODEL_CONFIGS.items():
            if (getattr(cfg := preset, "vocab_size", None) == vocab
                    and cfg.d_model == d_model and cfg.n_layers == n_layers):
                return name
        return None

    @classmethod
    def _load_prequant(cls, cfg, checkpoint, tokenizer, device,
                       metadata, **kwargs):
        """Load a pre-quantized BitNet checkpoint directly into int8 storage.

        Avoids the wasteful int8→bf16→int8 round-trip of the old path.
        Instead, loads int8 tensors from the safetensors file and calls
        BitNetLinear.load_prequantized() to store them directly as int8
        buffers. Non-BitNet tensors (embeddings, norms, etc.) are loaded
        normally as bf16. Peak CPU RAM is ~6 GB (int8 state dict) instead
        of ~17 GB (int8 + bf16 intermediate).
        """
        import gc
        import time

        from safetensors import safe_open

        from forge.config import ModelConfig
        from forge.keys.quantization.bitnet_b158_key import (
            BitNetConv1d,
            BitNetEmbedding,
            BitNetLinear,
        )
        from forge.model_loader import ModelLoader

        t0 = time.time()
        logger.info(
            "Pre-quantized BitNet checkpoint: direct int8 loading "
            "(no bf16 intermediate)")

        # 1. Build model on meta device (fast, no real tensors)
        cfg_meta = ModelConfig(**{**cfg.__dict__, "device": "meta"})
        with torch.device("meta"):
            from forge.model_loader import ConfigurableResearchLLM
            model = ConfigurableResearchLLM(cfg_meta)
        logger.info("Meta-init architecture in %.1fs", time.time() - t0)

        # 2. Load safetensors to CPU (int8 tensors stay int8 — no cast!)
        t_weights = time.time()
        state = {}
        with safe_open(checkpoint, framework="pt", device="cpu") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
        n_int8 = sum(1 for t in state.values() if t.dtype == torch.int8)
        n_bf16 = sum(1 for t in state.values() if t.dtype != torch.int8)
        int8_params = sum(t.numel() for t in state.values() if t.dtype == torch.int8)
        other_params = sum(t.numel() for t in state.values() if t.dtype != torch.int8)
        logger.info(
            "Loaded %d tensors (%d int8=%.2fB, %d other=%.2fB) in %.1fs",
            len(state), n_int8, int8_params / 1e9,
            n_bf16, other_params / 1e9, time.time() - t_weights)

        # 3. Build a map from parameter name → module for all BitNet types
        #    (BitNetLinear, BitNetConv1d, BitNetEmbedding) so we can call
        #    load_prequantized for int8 weights.
        bitnet_modules = {}
        for name, module in model.named_modules():
            if isinstance(module, (BitNetLinear, BitNetConv1d, BitNetEmbedding)):
                bitnet_modules[name + ".weight"] = module
        # Also handle head (nn.Linear) — store int8 as a buffer manually
        head_module = getattr(model, 'head', None)

        # 4. Assign tensors: int8 → load_prequantized, others → assign
        t_gpu = time.time()
        int8_loaded = 0
        other_keys = {}
        # Collect qscale tensors to pair with their int8 weights
        qscale_map = {}
        for key, tensor in state.items():
            if key.endswith(".qscale") and tensor.dtype != torch.int8:
                qscale_map[key[:-len(".qscale")] + ".weight"] = tensor
        for key, tensor in state.items():
            if key in bitnet_modules and tensor.dtype == torch.int8:
                # Direct int8 loading — no bf16 intermediate!
                module = bitnet_modules[key]
                # Use checkpoint's qscale if available, else compute from absmean
                # (use bf16 not fp32 to minimize memory for the fallback)
                if key in qscale_map:
                    qscale = qscale_map[key]
                else:
                    absmean = tensor.to(torch.bfloat16).abs().float().mean().clamp(min=1e-8)
                    qscale = absmean / 0.7
                # Move to target device as int8
                module.load_prequantized(
                    tensor.to(device).to(torch.int8), qscale.to(device))
                int8_loaded += 1
            elif key == "head.weight" and tensor.dtype == torch.int8 and head_module is not None:
                # Head is nn.Linear — convert to int8 buffer storage manually
                dev = torch.device(device)
                w_int8 = tensor.to(dev).to(torch.int8)
                qscale = qscale_map.get(key)
                if qscale is None:
                    absmean = tensor.to(torch.bfloat16).abs().float().mean().clamp(min=1e-8)
                    qscale = absmean / 0.7
                qscale = qscale.to(dev)
                del head_module.weight
                head_module.register_buffer("weight_int8", w_int8)
                head_module.register_buffer("qscale_buf", qscale)
                head_module._prequantized = True
                # Monkey-patch forward to use int8 buffer
                _orig_forward = head_module.forward
                def _int8_forward(x, _w=w_int8, _s=qscale, _b=head_module.bias):
                    return F.linear(x, _w.to(x.dtype), _b) * _s.to(x.dtype)
                head_module.forward = _int8_forward
                int8_loaded += 1
            elif key.endswith(".qscale") and (
                key[:-len(".qscale")] + ".weight" in bitnet_modules
                or key[:-len(".qscale")] + ".weight" == "head.weight"):
                # qscale for a BitNet/head layer already handled — skip
                continue
            else:
                other_keys[key] = tensor

        # 5. Load remaining (non-int8) tensors via assign=True
        if other_keys:
            missing, unexpected = model.load_state_dict(
                other_keys, strict=False, assign=True)
            if missing:
                real_missing = [k for k in missing if k != "head.weight"]
                if real_missing:
                    logger.warning(
                        "Missing keys: %s%s", real_missing[:5],
                        "..." if len(real_missing) > 5 else "")
            if unexpected:
                logger.warning(
                    "Unexpected keys: %s%s", unexpected[:5],
                    "..." if len(unexpected) > 5 else "")

        # 6. Re-tie weights if needed (assign breaks sharing)
        if getattr(cfg, 'tie_word_embeddings', True) \
                and not getattr(cfg, 'use_pit', False):
            model.head.weight = model.embed.weight

        # 7. Move non-int8 params/buffers to target device
        dev = torch.device(device)
        for module in model.modules():
            for pname, param in list(module._parameters.items()):
                if param is None:
                    continue
                if param.is_meta:
                    module._parameters[pname] = torch.nn.Parameter(
                        torch.zeros(param.shape, dtype=param.dtype, device=dev),
                        requires_grad=param.requires_grad)
                elif param.device != dev:
                    module._parameters[pname] = torch.nn.Parameter(
                        param.data.to(dev),
                        requires_grad=param.requires_grad)
            for bname, buf in list(module._buffers.items()):
                if buf is not None:
                    if buf.is_meta:
                        module._buffers[bname] = torch.zeros(
                            buf.shape, dtype=buf.dtype, device=dev)
                    elif buf.device != dev:
                        module._buffers[bname] = buf.to(dev)

        # 8. Reset non-persistent buffers (RoPE cos/sin)
        ModelLoader._reset_non_persistent_buffers(model, dev)

        if dev.type == "cuda":
            torch.cuda.synchronize()

        # 9. Free the state dict immediately
        del state, other_keys, bitnet_modules
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 10. Post-load QK-norm identity scan
        for block in model.blocks:
            attn = block.attn
            if hasattr(attn, 'q_norm') and hasattr(attn, '_qk_norm_identity'):
                q_id = (attn.q_norm.weight == 1.0).all()
                k_id = (attn.k_norm.weight == 1.0).all()
                attn._qk_norm_identity = bool(q_id and k_id)

        t_total = time.time() - t0
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(
            "Direct int8 load: %d BitNet layers | assign: %.1fs | "
            "Total: %.1fs (%.1fM params)",
            int8_loaded, time.time() - t_gpu, t_total, param_count)
        model.eval()

        engine = cls(model, tokenizer, device=device,
                     checkpoint_path=checkpoint, **kwargs)
        engine._checkpoint_metadata = metadata
        engine.keystack_features = ["bitnet_prequant", "quarot", "mrl"]
        engine._log(f"KeyStack features: {engine.keystack_features}")
        return engine

    @classmethod
    def _load_gguf_checkpoint(cls, checkpoint, tokenizer, device,
                              auto_activate=True, **kwargs) -> "ForgeEngine":
        """Load a GGUF checkpoint with dequantization.

        Uses ForgeLoader to parse the GGUF file, dequantize quantized tensors
        (Q4_0, Q8_0, Q4_K, Q6_K, etc.) to float16, and build a model from
        the extracted weights. The GGUF metadata is used to auto-detect the
        architecture and build an appropriate ModelConfig.
        """
        from forge.config import ModelConfig
        from forge.engine.forge_loader import GGUFInfo
        from forge.model_loader import ModelLoader

        logger.info("Loading GGUF: %s", checkpoint)
        info = GGUFInfo(checkpoint)
        arch = info.get_architecture()
        logger.info("GGUF architecture: %s, tensors: %d", arch, len(info.tensors))

        # Build config from GGUF metadata
        d_model = info.metadata.get(f"{arch}.embedding_length", 2048)
        n_layers = info.metadata.get(f"{arch}.block_count", 16)
        n_heads = info.metadata.get(f"{arch}.attention.head_count", 16)
        n_kv_heads = info.metadata.get(
            f"{arch}.attention.head_count_kv", n_heads)
        intermediate = info.metadata.get(
            f"{arch}.feed_forward_length", int(4 * d_model))
        vocab_size = info.metadata.get(f"{arch}.vocab_size", 32000)
        max_seq = info.metadata.get(
            f"{arch}.context_length", info.get_context_length())

        cfg = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            intermediate_size=intermediate,
            attn_type="gqa" if n_kv_heads < n_heads else "mha",
            attn_bias=False,
            ffn_type="swiglu",
            norm_type="rmsnorm",
            norm_eps=1e-5,
            use_embed_norm=False,
            use_final_norm=True,
            rope_base=10000.0,
            max_seq_len=max_seq,
            layer_types=["attention"] * n_layers,
            use_qk_norm=False,
            use_bitnet=False,
            use_bitnet_residual=False,
            ffn_compression="none",
            nlrq_rank=0,
            use_factorized_embeddings=False,
            embed_factorized_rank=0,
            use_pit=False,
            use_iri_fp4=False,
            use_spectral_kv=False,
            zero_init_residual=True,
            batch_size=1,
            seq_len=2048,
            max_steps=50000,
            warmup_steps=2000,
            max_lr=3e-4,
            min_lr=3e-5,
        )

        # Build model on meta device, then load dequantized weights
        model = ModelLoader.build_model(cfg, checkpoint_path=None)
        model = model.to(device).to(torch.float16)

        # Load and dequantize weights from GGUF
        state_dict = {}
        for t_info in info.tensors:
            name = t_info["name"]
            try:
                tensor = info.get_tensor(name, dequantize=True)
                state_dict[name] = tensor.to(device).to(torch.float16)
            except Exception as e:
                logger.warning("GGUF: skipping tensor %s: %s", name, e)

        # Map GGUF tensor names to Forge model names
        # GGUF uses: blk.N.attn_q.weight, blk.N.ffn_gate.weight, etc.
        # Forge uses: blocks.N.attn.q_proj.weight, blocks.N.ffn.w_gate.weight, etc.
        mapped = {}
        for name, tensor in state_dict.items():
            forge_name = _map_gguf_to_forge(name)
            if forge_name:
                mapped[forge_name] = tensor

        # Load mapped weights into model (strict=False for missing/extra keys)
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        if missing:
            logger.warning("GGUF missing keys: %d (first 5: %s)",
                           len(missing), missing[:5])
        if unexpected:
            logger.warning("GGUF unexpected keys: %d", len(unexpected))

        info.close()
        engine = cls(model, tokenizer, device=device,
                     checkpoint_path=checkpoint, **kwargs)
        engine._log(f"GGUF loaded: {arch}, {len(state_dict)} tensors, "
                    f"{len(mapped)} mapped")
        if auto_activate:
            engine._auto_activate_optimal()
        return engine

    @classmethod
    def _load_standard(cls, cfg, checkpoint, tokenizer, device, **kwargs):
        """Fast path: model fits in VRAM, load normally."""
        from forge.model_loader import ModelLoader

        # Pass dtype=bfloat16 to prevent fp32 upcasting (saves 2x VRAM)
        model = ModelLoader.build_model_fast(
            cfg, checkpoint_path=checkpoint, dtype=torch.bfloat16)
        return cls(model, tokenizer, device=device,
                   checkpoint_path=checkpoint, **kwargs)

    @classmethod
    def _load_with_fallback(cls, cfg, checkpoint, tokenizer, device,
                            ckpt_size, vram_free, **kwargs):
        """Model doesn't fit entirely — try hybrid offload before streaming.

        Decision tree:
          1. If model has hybrid layer types (conv + attention), try
             hybrid_offload (conv on CPU, attention on GPU). This is much
             faster than full streaming since only conv weights go to CPU.
          2. If hybrid offload still doesn't fit, fall back to AirLLM
             layer-streaming (meta device + per-forward shard loading).
        """
        from forge.model_loader import ModelLoader

        layer_types = getattr(cfg, "layer_types", None)
        has_hybrid = layer_types and any(
            lt != "attention" for lt in layer_types)

        if has_hybrid:
            # Estimate: attention layers on GPU, conv on CPU
            n_attn = sum(1 for lt in layer_types if lt == "attention")
            n_conv = len(layer_types) - n_attn
            # Rough estimate: attention layers are ~70% of model params
            attn_size = int(ckpt_size * 0.7)
            if vram_free > int(attn_size * 1.3):
                logger.info(
                    "HybridOffload: checkpoint %.2f GB > VRAM free %.2f GB, "
                    "but model has %d attn + %d conv layers",
                    ckpt_size / 1e9, vram_free / 1e9, n_attn, n_conv)
                logger.info(
                    "HybridOffload: trying attention on GPU, conv on CPU...")
                try:
                    model = ModelLoader.build_model_fast(
                        cfg, checkpoint_path=checkpoint, dtype=torch.bfloat16)
                    model = ModelLoader.hybrid_offload(
                        model, gpu_layers=-1, device=device)
                    engine = cls(model, tokenizer, device=device,
                                 checkpoint_path=checkpoint, **kwargs)
                    engine._log("Hybrid offload active: conv layers on CPU, "
                                "attention on GPU")
                    return engine
                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    if "size mismatch" in str(e).lower():
                        raise
                    logger.warning(
                        "HybridOffload failed (%s), falling back to AirLLM "
                        "streaming...", e)
                    cls._clear_cuda_cache_static(torch.device(device))

        # Fall back to full streaming
        return cls._load_streaming(
            cfg, checkpoint, tokenizer, device,
            ckpt_size, vram_free, **kwargs)

    @classmethod
    def _load_streaming(cls, cfg, checkpoint, tokenizer, device,
                        ckpt_size, vram_free, **kwargs):
        """Slow path: model too large for VRAM, build on meta for streaming."""
        from forge.model_loader import ModelLoader

        logger.info(
            "AirLLM-Smart: checkpoint %.2f GB > VRAM free %.2f GB",
            ckpt_size / 1e9, vram_free / 1e9)
        logger.info("AirLLM-Smart: building model on meta device (zero VRAM)...")
        model = ModelLoader.build_model(cfg, checkpoint_path=None)
        model.eval()
        engine = cls(model, tokenizer, device=device,
                     checkpoint_path=checkpoint, **kwargs)
        engine._needs_streaming = True
        return engine

