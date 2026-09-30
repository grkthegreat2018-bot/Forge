"""Forge Inference Engine — unified runtime backend for ForgeAI models.

Pluggable strategy architecture with auto-detection and auto-activation:
  - KV cache: standard, paged, rotorquant (4-bit default), hadamard_int4,
    compressed, streaming, snapkv, snapkv_4bit, paged_eviction, xquant,
    cpu_offload, s4r (15x compression), hqe_kv, 2bit
  - Decoding: standard, speculative, medusa, dspark, eagle3, mtp_selfspec
  - Quantization: none, int8, int4, fp8, w8a8, nvfp4, BitNet ternary
    (auto-selected based on VRAM + GPU capability)
  - Acceleration: none, cuda_graph, airllm_streaming, megakernel, flex_decoding
  - Innovations: MRL-AdaptiveContext, QuaRot-KV, V0-WarmStart, ProgressiveKV
  - 42 feature-registry flags for attention, prefill, KV, scheduling, MoE, etc.

Auto-detects KeyStack features from checkpoint metadata and auto-activates
optimal strategies (auto_activate=True by default). Picks the highest
quantization that fits VRAM (nvfp4 on Blackwell, w8a8 default, int4 when
tight). Uses RotorQuant 4-bit KV cache by default (Givens rotation +
Lloyd-Max quantization, ~8x compression, 0.94% error, deferred quantization
for zero error compounding during prefill).

Loading strategies (auto-selected by VRAM capacity):
  1. Pre-quantized BitNet → int8 direct load (4x VRAM cut)
  2. Fits in VRAM → fast meta-init + background thread weight load
  3. Hybrid offload → conv on CPU, attention on GPU (LFM2.5 hybrid arch)
  4. Too large → AirLLM layer-streaming with CPU RAM shard caching

Usage:
    from forge.engine.forge_engine import ForgeEngine

    # Auto-activates optimal strategies (S4R KV, torch.compile, prefix cache, etc.)
    engine = ForgeEngine.from_checkpoint(
        checkpoint="research/checkpoints/ForgeLM_V2.safetensors",
        config_name="forgelm_v2",
        tokenizer_path="research/checkpoints/forgelm_v2_tokenizer",
    )
    # Strategies auto-activated — just generate:
    output = engine.generate("def fibonacci(n):", max_new_tokens=50)

    # Or manually control activation:
    engine = ForgeEngine.from_checkpoint(..., auto_activate=False)
    engine.activate_optimal(kv_cache="hadamard_int4", decoding="mtp_selfspec")

    # Streaming, raw control, benchmarking, diagnostics:
    for chunk in engine.generate_stream("Hello", max_new_tokens=100):
        print(chunk, end="", flush=True)
    engine.benchmark("test prompt", max_new_tokens=50)
    engine.bottleneck()  # per-layer timing
    engine.diagnose()    # full health report

    # Sleep/wake for VRAM management:
    engine.sleep(level=1)  # offload to CPU
    engine.wake()          # restore to GPU

    # Context manager (auto-sleeps on exit):
    with ForgeEngine.from_checkpoint(...) as engine:
        engine.generate("...")
"""
import logging
import threading

import torch

logger = logging.getLogger(__name__)

# Helpers moved to engine_common.py (re-exported for backward compatibility).
from .engine_activation import _ActivationMixin
from .engine_checkpoints import _CheckpointLoadingMixin
from .activation import ActivationConfig  # noqa: F401
from .feature_registry import _FEATURE_REGISTRY  # noqa: F401
from .engine_common import (  # noqa: F401
    _CKPT_CACHE_MAX,
    _DEFAULT_CPU_MEMORY_BYTES,
    _DEFAULT_EOS_TOKEN_IDS,
    _QWEN_TOKENIZER_PATH,
    _QWEN_VOCAB,
    _checkpoint_metadata_cache,
    _checkpoint_size_cache,
    _ckpt_cache_lock,
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    SamplingParams,
)
from .engine_diagnostics import _DiagnosticsMixin
from .engine_generation import _GenerationMixin
from .engine_lifecycle import _LifecycleMixin
from .engine_lora import _LoRAMixin
from .engine_merging import _MergingMixin
from .engine_sessions import _SessionMixin

# Heavy runtime dependencies used by ForgeEngine.__init__ and class methods.
from .crash_recovery import CrashRecoveryManager  # noqa: F401
from .decoding import DecodingStrategy, StandardDecoding  # noqa: F401
from .diagnostics import (  # noqa: F401
    BandwidthProfiler,
    EngineProfiler,
    EventLog,
    OutputHistory,
)
from .engine_tools import EngineToolRegistry  # noqa: F401
from .errors import ForgeEngineError  # noqa: F401
from .hotswap import HotSwapManager  # noqa: F401
from .innovations import (  # noqa: F401
    MRLAdaptiveContext,
    ProgressiveKV,
    QuaRotKV,
    V0WarmStart,
)
from .kv.cacheblend import CacheBlend  # noqa: F401
from .kv_backend import KVCacheStrategy  # noqa: F401
from .library import Library  # noqa: F401
from .prefix_cache import SemanticKVAnchors  # noqa: F401
from .session_cache import SessionCacheManager  # noqa: F401


class ForgeEngine(_CheckpointLoadingMixin, _ActivationMixin,
                  _GenerationMixin, _DiagnosticsMixin, _MergingMixin,
                  _LoRAMixin, _LifecycleMixin, _SessionMixin):
    """Unified inference engine for ForgeAI XP models.

    Orchestrates all runtime strategies and innovations. Auto-detects
    KeyStack features from checkpoint and activates matching strategies.
    """

    def __init__(self, model, tokenizer, device="cuda",
                 checkpoint_path: str | None = None):
        # Runtime side effects (CUDA alloc conf, cache dirs, TF32) live in
        # forge.runtime.configure — idempotent, effective only if it runs
        # before CUDA init (entrypoints call it at startup; direct engine
        # construction still gets the non-CUDA settings).
        from forge.runtime.configure import configure
        configure()
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path

        # LoRA hot-loading state
        self._lora_config: dict | None = None

        # Serialization of ALL generation paths.  The model keeps mutable
        # per-layer state that past_kv does not cover (conv `_conv_state`
        # buffers, `_forge_last_kv`, hotswap flags) — two concurrent
        # generations interleave forward passes and corrupt each other.
        # RLock because generate_with_tools() -> generate() nests.
        self._gen_lock = threading.RLock()

        # Strategy slots
        self.kv_cache: KVCacheStrategy | None = None
        self.decoding: DecodingStrategy = StandardDecoding()
        self.quantize: str | None = None
        self.acceleration: str | None = None
        self._active_kv_bits = 8  # default; updated by _activate_kv_cache
        self._active_kv_cache_name: str | None = None

        # Innovation slots
        self.mrl_adapter: MRLAdaptiveContext | None = None
        self.quarot_kv: QuaRotKV | None = None
        self.v0_warm: V0WarmStart | None = None
        self.progressive_kv: ProgressiveKV | None = None

        # Detected KeyStack features
        self.keystack_features: list[str] = []

        # Stats
        self.generation_count = 0
        self.total_tokens_generated = 0
        self._prefix_cache = None
        self._cache_blend: CacheBlend | None = None  # R&D14: CacheBlend
        self.semantic_anchors = SemanticKVAnchors()  # R&D15: FreeToken-style
        self._graph_runner = None
        self._stop_tokens = None
        self._system_one = None        # cached SystemOneEvaluator (decide())
        self._decision_scorer = None   # trained Tier-1 DecisionScorer
        self._gate_probes = None       # ForgeGate probe bundle
        self._awake = True
        self._sleep_level = 0
        self._last_activation_params: dict | None = None

        # Built-in diagnostics (replaces need for one-off scripts)
        self.events = EventLog(capacity=500)
        self.outputs = OutputHistory(capacity=100)
        self._profiler = EngineProfiler(self.model, self.device)
        self.bandwidth_profiler = BandwidthProfiler(self.device)
        self._log("ForgeEngine initialized",
                  device=str(self.device),
                  checkpoint=checkpoint_path or "none")

        # Crash recovery: signal handlers + atexit + disk checkpointing
        self._recovery = CrashRecoveryManager(self, enabled=True)

        # Session-aware KV cache: multi-turn optimization with radix prefix matching + TTL
        self._session_cache = SessionCacheManager(self)

        # Hot-swap manager: runtime config changes without reload
        self.hotswap = HotSwapManager(self)

        # Library: persistent knowledge base with lorebook-style injection
        from research.paths import LIBRARY_DIR
        self.library = Library(
            tokenizer=tokenizer, path=str(LIBRARY_DIR),
        )
        self._library_enabled = True
        self._library_injection_budget = 2048

        # Built-in tool registry: gives the LLM tools to use engine features
        self.tools = EngineToolRegistry(self)

        # Move model to device (unless it's on meta — streaming mode)
        self._needs_streaming = False
        first_param = next(self.model.parameters(), None)
        if first_param is not None and first_param.device.type != "meta":
            self.model.to(self.device)
        self.model.eval()

        # Auto-detect KeyStack features
        self._checkpoint_metadata = {}
        if checkpoint_path:
            self._detect_keystack_features()
            # Note: int8 conversion is handled in from_checkpoint() for pre-quant
            # checkpoints, since it requires casting int8->fp32 before load_state_dict

    # ── Logging ──────────────────────────────────────────────────────────

    def _log(self, message: str, level: str = "info",
             source: str = "engine", **data):
        """Print a status message and record it in the event log.

        Replaces scattered ``print()`` calls with a single chanel that
        both shows the message to the user and stores it for
        ``read_log()`` / ``diagnose()`` diagnostics.

        Emits through the ``logging`` module when handlers are configured
        (server/GUI contexts capture it properly); falls back to stdout
        for interactive/script use so the status line still shows.
        """
        line = (f"  [{source.title()}] {message}" if source != "engine"
                else f"  [ForgeEngine] {message}")
        lvl = {"debug": logging.DEBUG, "info": logging.INFO,
               "warn": logging.WARNING, "warning": logging.WARNING,
               "error": logging.ERROR}.get(level, logging.INFO)
        if logging.getLogger().handlers:
            logger.log(lvl, "%s%s", line,
                       f" {data}" if data else "")
        else:
            print(line)
        self.events.log(message, level=level, source=source, **data)

    def _clear_cuda_cache(self):
        """Synchronize + empty CUDA cache. Deduplicates the 6x repeated
        ``synchronize / empty_cache / synchronize`` pattern across generate,
        sleep, and OOM-recovery paths."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(self.device)

    def __del__(self):
        """Release GPU tensors and CPU weight snapshots to prevent VRAM leaks
        between engine loads. Safe to call multiple times (gc also calls this).
        """
        try:
            # Free CPU weight snapshot from _save_original_weights
            if hasattr(self, '_original_weights'):
                self._original_weights = None
            # Free cached quantized weights
            if hasattr(self, 'model') and self.model is not None:
                for m in self.model.modules():
                    if hasattr(m, '_cached_weight'):
                        m._cached_weight = None
            # Move model to CPU to release GPU tensors
            if hasattr(self, 'model') and self.model is not None:
                try:
                    self.model.cpu()
                except Exception:
                    logger.debug("Failed to move model to CPU during cleanup", exc_info=True)
            if hasattr(self, 'device') and self.device.type == "cuda":
                torch.cuda.empty_cache()
        except Exception:
            logger.debug("Error during engine cleanup", exc_info=True)

    @staticmethod
    def _clear_cuda_cache_static(device):
        """Static version for use in classmethods (before an instance exists)."""
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)

    def _release_acceleration_resources(self):
        """Release CUDA graph / megakernel / flex-decoding / chunked-prefill
        resources that hold GPU memory independent of ``self.model``.

        Called from ``sleep(level=2)`` before discarding the model to prevent
        leaks of CUDA graph pools and megakernel capture buffers.
        """
        for attr in ("_graph_runner", "_megakernel",
                     "_flex_decoding", "_chunked_prefill"):
            obj = getattr(self, attr, None)
            if obj is None:
                continue
            # If the object exposes a cleanup/release method, call it.
            for method_name in ("release", "cleanup", "destroy", "close"):
                method = getattr(obj, method_name, None)
                if callable(method):
                    try:
                        method()
                    except Exception:
                        logger.debug("Error releasing %s via %s", attr, method_name, exc_info=True)
                    break
            setattr(self, attr, None)
        self.acceleration = None

    # ── Convenience properties ──────────────────────────────────────────
    # These derive from the loaded model so callers don't need to reach
    # into self.model.config / self.model.parameters() manually.

    @property
    def config(self):
        """ModelConfig attached to the loaded model (or None)."""
        return getattr(self.model, "config", None)

    # ── Library control ──────────────────────────────────────────────────

    def library_save(self, content: str, category: str = "custom",
                     tags: list[str] | None = None,
                     description: str = "",
                     triggers: list[str] | None = None,
                     priority: int = 0) -> str:
        """Save an entry to the library (model self-write).

        Categories: "failure", "win", "research", "common_data", "custom".
        Returns entry_id.
        """
        return self.library.save(
            content=content, category=category, tags=tags,
            description=description, triggers=triggers,
            priority=priority, source="model")

    def library_set_enabled(self, enabled: bool) -> None:
        """Enable/disable library lorebook injection globally."""
        self._library_enabled = enabled

    def library_set_budget(self, tokens: int) -> None:
        """Set the injection token budget (max tokens injected per request)."""
        self._library_injection_budget = tokens

    def library_lookup(self, **kwargs) -> list:
        """Lookup library entries by tags/category."""
        return self.library.lookup(**kwargs)

    def library_search(self, query: str, limit: int = 20) -> list:
        """Full-text search the library."""
        return self.library.search(query, limit=limit)

    def library_optimize(self) -> dict:
        """Run library optimization (merge similar, trim, re-index)."""
        return self.library.optimize()

    def library_stats(self) -> dict:
        """Get library statistics."""
        return self.library.stats()

    @property
    def dtype(self) -> torch.dtype:
        """Dtype of the model's first parameter (bf16 by default)."""
        if self.model is None:
            return getattr(self, "_stored_dtype", torch.bfloat16)
        parameter = next(self.model.parameters(), None)
        return parameter.dtype if parameter is not None else torch.bfloat16

    @staticmethod
    def _memory_info(device: torch.device) -> tuple[int, int]:
        if device.type == "cuda":
            return torch.cuda.mem_get_info(device)
        return _DEFAULT_CPU_MEMORY_BYTES, _DEFAULT_CPU_MEMORY_BYTES

    @property
    def _kv_dimensions(self) -> tuple[int, int]:
        return (
            getattr(self.config, "n_kv_heads", 8),
            getattr(self.config, "head_dim", 64),
        )

    def _require_awake(self):
        if not self._awake:
            raise ForgeEngineError(
                "ForgeEngine is asleep; call wake() before inference",
                context={"sleep_level": self._sleep_level},
                suggestion="Call engine.wake() to restore the model to GPU.")

    @property
    def active_config(self) -> ActivationConfig | None:
        """The ``ActivationConfig`` from the last ``activate()`` call, or None."""
        params = getattr(self, "_last_activation_params", None)
        if params is None:
            return None
        return ActivationConfig.from_kwargs(**params)

    def reset_stats(self) -> None:
        """Reset generation counters and diagnostics history.

        Useful for benchmarking: call before a measurement run to get
        clean stats without interference from prior generations.
        """
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.events.clear()
        self.outputs = OutputHistory(capacity=100)

    def __repr__(self) -> str:
        status = "awake" if self._awake else f"asleep(L{self._sleep_level})"
        n_params = 0
        if self.model is not None:
            n_params = sum(p.numel() for p in self.model.parameters())
        cfg_name = getattr(self.config, "name", "unknown")
        return (f"ForgeEngine(model={cfg_name}, "
                f"params={n_params/1e6:.0f}M, "
                f"device={self.device}, "
                f"kv={self.kv_cache.info().get('type', 'none') if self.kv_cache else 'none'}, "
                f"decoding={self.decoding.name}, "
                f"{status})")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Auto-sleep level 1 on context exit to release VRAM."""
        self.close()
        return False

    def close(self):
        """Explicit teardown: disable crash recovery + release VRAM.

        Preferred over relying on ``__del__`` (which still runs as a
        backstop) — deterministic and safe to call more than once.
        """
        rec = getattr(self, "_recovery", None)
        if rec is not None:
            rec.close()
        if self._awake:
            self.sleep(level=1)

    # ── Checkpoint loading ────────────────────────────────────────────────

