"""Model Registry — multi-engine VRAM-budgeted manager.

Manages multiple ForgeEngine instances on a single GPU with explicit
byte-level VRAM budgeting. Supports sleep/wake hot-swap, concurrent
multi-model loading, and automatic VRAM defragmentation.

Key differentiator vs vLLM/LM Studio:
  - Byte budgets, not utilization fractions (avoids vLLM V1 bug)
  - Concurrent multi-model when VRAM allows (LM Studio does one at a time)
  - Sleep/wake with sub-3s round-trip target
  - Automatic eviction of idle models under VRAM pressure

Usage:
    from forge.engine.model_registry import ModelRegistry

    registry = ModelRegistry()
    registry.register("forgelm-v2-jamba", checkpoint="...", config="forgelm_v2", vram_budget_gb=8.0)
    registry.register("qwen2.5", checkpoint="...", config="qwen25_05b", vram_budget_gb=3.5)

    # Generate with either model — registry handles wake/sleep automatically
    out = registry.generate("forgelm-v2-jamba", "def fibonacci(n):")
    out = registry.generate("qwen2.5", "Explain quantum computing")
"""
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from forge.engine.forge_engine import ForgeEngine, _checkpoint_size_cache

logger = logging.getLogger(__name__)


@dataclass
class EngineEntry:
    """Metadata for a registered engine."""
    model_id: str
    engine: ForgeEngine
    vram_budget_bytes: int
    checkpoint_path: str
    config_name: str
    tokenizer_path: str
    is_awake: bool = True
    last_used: float = field(default_factory=time.time)
    generation_count: int = 0
    total_tokens: int = 0


class VRAMBudgetExceeded(Exception):
    """Not enough VRAM to load the model even after evicting idle engines."""
    pass


class ModelRegistry:
    """Multi-engine manager with explicit VRAM budgeting.

    Key properties:
      - Byte-level budgets: each engine reserves an absolute byte budget.
        This avoids the vLLM V1 bug where `gpu_memory_utilization` fractions
        don't account for already-loaded models.
      - Sleep/wake: idle engines are put to sleep (Level 1: CPU offload)
        when VRAM is needed for a new model.
      - Concurrent loading: if VRAM budget allows, multiple models stay
        awake simultaneously.
      - Thread-safe: all public methods use a reentrant lock.
    """

    def __init__(self, device: str = "cuda", safety_margin: float = 0.10):
        self.device = torch.device(device)
        self.safety_margin = safety_margin  # 10% VRAM headroom
        self._entries: dict[str, EngineEntry] = {}
        self._lock = threading.RLock()

        # VRAM capacity
        if self.device.type == "cuda":
            self._total_vram = torch.cuda.get_device_properties(self.device).total_memory
        else:
            self._total_vram = 32 * 1024**3  # Assume 32GB system RAM

    # ── Public API ────────────────────────────────────────────────────────

    def register(self, model_id: str, checkpoint: str, config_name: str,
                 tokenizer_path: str | None = None,
                 vram_budget_gb: float = 0,
                 **engine_kwargs) -> ForgeEngine:
        """Register and load a model, managing VRAM automatically.

        If VRAM is insufficient, idle engines are put to sleep to free space.
        If vram_budget_gb is 0, auto-calculates from checkpoint file size.

        Returns the loaded ForgeEngine (or existing if already registered).
        """
        with self._lock:
            # Return existing if already loaded
            if model_id in self._entries:
                entry = self._entries[model_id]
                self._ensure_awake(model_id)
                entry.last_used = time.time()
                return entry.engine

            # Calculate VRAM budget
            if vram_budget_gb <= 0:
                ckpt_size = _checkpoint_size_cache.get(checkpoint)
                if ckpt_size is None:
                    ckpt_size = Path(checkpoint).stat().st_size
                    _checkpoint_size_cache[checkpoint] = ckpt_size
                vram_budget_gb = (ckpt_size * 1.5) / 1e9  # 50% overhead for KV + activations
                logger.info("Auto budget for %s: %.2f GB", model_id, vram_budget_gb)
            budget_bytes = int(vram_budget_gb * 1e9)

            # Ensure VRAM is available
            self._make_room(budget_bytes, exclude=model_id)

            # Load the model
            tok_path = tokenizer_path or "research/checkpoints/forgelm_v2_tokenizer"
            engine = ForgeEngine.from_checkpoint(
                checkpoint=checkpoint,
                config_name=config_name,
                tokenizer_path=tok_path,
                device=str(self.device),
                **engine_kwargs,
            )

            entry = EngineEntry(
                model_id=model_id,
                engine=engine,
                vram_budget_bytes=budget_bytes,
                checkpoint_path=checkpoint,
                config_name=config_name,
                tokenizer_path=tok_path,
            )
            self._entries[model_id] = entry

            logger.info("Registered: %s (budget=%.2fGB, free=%.2fGB)",
                        model_id, vram_budget_gb, self._free_vram() / 1e9)
            return engine

    def generate(self, model_id: str, prompt: str, max_new_tokens: int = 100,
                 temperature: float = 0.0, top_p: float = 1.0,
                 finish_sentence: bool = True,
                 top_k: int = 80, repetition_penalty: float = 1.05,
                 stop: list[str] | None = None, seed: int | None = None,
                 **kwargs) -> str:
        """Generate text from a registered model. Auto-wakes if asleep."""
        with self._lock:
            if model_id not in self._entries:
                raise KeyError(f"Model '{model_id}' not registered. "
                               f"Available: {list(self._entries.keys())}")

            self._ensure_awake(model_id)
            entry = self._entries[model_id]
            entry.last_used = time.time()

        # Seed handling: set on the engine for reproducible sampling
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        output = entry.engine.generate(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p,
            finish_sentence=finish_sentence,
            top_k=top_k, repetition_penalty=repetition_penalty,
            stop=stop,
            min_p=kwargs.get('min_p', 0.0),
            min_k=kwargs.get('min_k', 0.0),
            dry_multiplier=kwargs.get('dry_multiplier', 0.0),
            dry_base=kwargs.get('dry_base', 1.75),
            dry_allowed_length=kwargs.get('dry_allowed_length', 2),
            dry_penalty_last_n=kwargs.get('dry_penalty_last_n', 512),
            logprobs=kwargs.get('logprobs'),
            prompt_logprobs=kwargs.get('prompt_logprobs'),
            return_logprobs=kwargs.get('return_logprobs', False),
        )
        entry.generation_count += 1
        entry.total_tokens += entry.engine.total_tokens_generated
        return output

    def generate_stream(self, model_id: str, prompt: str, max_new_tokens: int = 256,
                        temperature: float = 0.0, top_p: float = 1.0,
                        top_k: int = 80, repetition_penalty: float = 1.05,
                        min_p: float = 0.0, min_k: float = 0.0,
                        dry_multiplier: float = 0.0, dry_base: float = 1.75,
                        dry_allowed_length: int = 2,
                        dry_penalty_last_n: int = 512,
                        eos_token_ids: list[int] | None = None,
                        skip_special_tokens: bool = True,
                        logits_processor=None,
                        seed: int | None = None,
                        stop: list[str] | None = None):
        """Token-by-token streaming generator. Auto-wakes if asleep.

        Yields decoded text chunks (one per token) as they are generated.
        Sampling args mirror ForgeEngine.generate_stream.
        """
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        with self._lock:
            if model_id not in self._entries:
                raise KeyError(f"Model '{model_id}' not registered. "
                               f"Available: {list(self._entries.keys())}")
            self._ensure_awake(model_id)
            entry = self._entries[model_id]
            entry.last_used = time.time()
            engine = entry.engine

        for chunk in engine.generate_stream(
            prompt, max_new_tokens=max_new_tokens,
            temperature=temperature, top_p=top_p, top_k=top_k,
            repetition_penalty=repetition_penalty, min_p=min_p,
            min_k=min_k, eos_token_ids=eos_token_ids,
            dry_multiplier=dry_multiplier, dry_base=dry_base,
            dry_allowed_length=dry_allowed_length,
            dry_penalty_last_n=dry_penalty_last_n,
            skip_special_tokens=skip_special_tokens,
            logits_processor=logits_processor,
            stop=stop,
        ):
            yield chunk

        with self._lock:
            entry.generation_count += 1
            entry.total_tokens += engine.total_tokens_generated

    def sleep(self, model_id: str, level: int = 1):
        """Put a model to sleep to free VRAM."""
        with self._lock:
            if model_id not in self._entries:
                return
            entry = self._entries[model_id]
            if entry.is_awake:
                entry.engine.sleep(level=level)
                entry.is_awake = False
                logger.info("%s → sleep level %s (free=%.2fGB)",
                            model_id, level, self._free_vram() / 1e9)

    def wake(self, model_id: str):
        """Wake a sleeping model."""
        with self._lock:
            self._ensure_awake(model_id)

    def switch(self, from_model: str, to_model: str):
        """Sleep one model, wake another. Target: <3s round-trip."""
        with self._lock:
            self.sleep(from_model, level=1)
            self._ensure_awake(to_model)

    def unregister(self, model_id: str):
        """Remove a model from the registry and free all its resources."""
        with self._lock:
            if model_id not in self._entries:
                return
            entry = self._entries[model_id]
            if entry.is_awake:
                entry.engine.sleep(level=2)  # Discard weights
            del entry.engine
            del self._entries[model_id]
            torch.cuda.empty_cache()
            logger.info("Unregistered: %s", model_id)

    def list_models(self) -> list[dict]:
        """List all registered models with status."""
        with self._lock:
            return [
                {
                    "id": e.model_id,
                    "awake": e.is_awake,
                    "vram_budget_gb": e.vram_budget_bytes / 1e9,
                    "config": e.config_name,
                    "last_used_ago_s": time.time() - e.last_used,
                    "generations": e.generation_count,
                }
                for e in self._entries.values()
            ]

    def get_engine(self, model_id: str):
        """Get the ForgeEngine for a registered model (auto-wakes if asleep).

        Returns None if model_id is not registered.
        """
        with self._lock:
            if model_id not in self._entries:
                return None
            self._ensure_awake(model_id)
            entry = self._entries[model_id]
            entry.last_used = time.time()
            return entry.engine

    def stats(self) -> dict:
        """Aggregate registry statistics."""
        with self._lock:
            awake_count = sum(1 for e in self._entries.values() if e.is_awake)
            asleep_count = len(self._entries) - awake_count
            awake_vram = sum(
                e.vram_budget_bytes for e in self._entries.values() if e.is_awake)
            return {
                "total_models": len(self._entries),
                "awake": awake_count,
                "asleep": asleep_count,
                "awake_vram_gb": awake_vram / 1e9,
                "total_vram_gb": self._total_vram / 1e9,
                "free_vram_gb": self._free_vram() / 1e9,
                "models": self.list_models(),
            }

    # ── Internal ──────────────────────────────────────────────────────────

    def _free_vram(self) -> int:
        """Get currently free VRAM in bytes."""
        if self.device.type == "cuda":
            return torch.cuda.mem_get_info(self.device)[0]
        return self._total_vram  # CPU: assume all free

    def _usable_vram(self) -> int:
        """VRAM available for new models (after safety margin)."""
        return int(self._total_vram * (1.0 - self.safety_margin))

    def _reserved_vram(self) -> int:
        """Total VRAM reserved by awake engines."""
        return sum(e.vram_budget_bytes for e in self._entries.values() if e.is_awake)

    def _ensure_awake(self, model_id: str):
        """Wake a model, sleeping others if VRAM is tight."""
        entry = self._entries[model_id]
        if entry.is_awake:
            return

        # Check if we need to free VRAM
        needed = entry.vram_budget_bytes
        free = self._free_vram()
        reserved = self._reserved_vram()

        if free + reserved < needed:
            # Not enough total VRAM even if we sleep everything
            raise VRAMBudgetExceeded(
                f"Model '{model_id}' needs {needed/1e9:.2f}GB but only "
                f"{(free + reserved)/1e9:.2f}GB is available on the device"
            )

        if free < needed:
            # Sleep idle models (LRU order) until we have enough
            self._evict_lru(needed - free, exclude=model_id)

        entry.engine.wake()
        entry.is_awake = True

    def _make_room(self, needed_bytes: int, exclude: str | None = None):
        """Ensure `needed_bytes` of free VRAM by sleeping idle engines."""
        free = self._free_vram()
        reserved = self._reserved_vram()

        if free + reserved < needed_bytes:
            raise VRAMBudgetExceeded(
                f"Need {needed_bytes/1e9:.2f}GB but device only has "
                f"{(free + reserved)/1e9:.2f}GB total (free={free/1e9:.2f}GB)"
            )

        if free < needed_bytes:
            self._evict_lru(needed_bytes - free, exclude=exclude)

    def _evict_lru(self, needed_bytes: int, exclude: str | None = None):
        """Sleep engines in LRU order until `needed_bytes` freed."""
        # Sort awake engines by last_used (oldest first), exclude the target
        candidates = sorted(
            (e for e in self._entries.values()
             if e.is_awake and e.model_id != exclude),
            key=lambda e: e.last_used,
        )

        freed = 0
        for entry in candidates:
            entry.engine.sleep(level=1)
            entry.is_awake = False
            freed += entry.vram_budget_bytes
            logger.info("Evicted %s (idle %.0fs, freed %.2fGB)",
                        entry.model_id, time.time() - entry.last_used,
                        entry.vram_budget_bytes / 1e9)
            if freed >= needed_bytes:
                break
        else:
            raise VRAMBudgetExceeded(
                f"Could not free {needed_bytes/1e9:.2f}GB after evicting "
                f"{len(candidates)} idle engines"
            )


# ── Quick test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    registry = ModelRegistry()
    print(f"Device: {registry.device}")
    print(f"Total VRAM: {registry._total_vram / 1e9:.2f} GB")
    print(f"Free VRAM: {registry._free_vram() / 1e9:.2f} GB")
    print(f"Usable VRAM: {registry._usable_vram() / 1e9:.2f} GB")
