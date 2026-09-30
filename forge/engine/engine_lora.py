"""LoRA adapter management mixin for ForgeEngine."""
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
    _map_gguf_to_forge,
    _min_k_filter,
    _ScalingModelAdapter,
    _tokenizer_for_vocab,
    logger,
)


class _LoRAMixin:
    # ── LoRA hot-loading ───────────────────────────────────────────────────

    def load_lora(self, lora_path: str, rank: int = 32, alpha: int | None = None,
                  target_modules: list[str] | None = None) -> int:
        """Hot-load a LoRA adapter onto the model.

        Attaches LoRA adapters to the specified modules and loads weights from
        a safetensors checkpoint. The base model weights are frozen — only the
        LoRA adapters are trainable. This enables dynamic skill injection at
        runtime (e.g. loading a tool-calling LoRA for self-play, then swapping
        it for a code-generation LoRA).

        Args:
            lora_path: Path to LoRA adapter safetensors file.
            rank: LoRA rank (must match the checkpoint).
            alpha: LoRA alpha (scale = alpha / rank). If None, uses rank * 2.
            target_modules: Module name substrings to attach LoRA to.
                If None, defaults to FFN+attention modules:
                ["w_gate", "w_up", "w_down", "q_proj", "v_proj",
                 "out_proj", "in_proj"]
                Pass ["w_gate", "w_up", "w_down"] for FFN-only.

        Returns:
            Number of LoRA parameters loaded.

        Raises:
            FileNotFoundError: If lora_path doesn't exist.
            ValueError: If LoRA checkpoint doesn't match model structure.
        """
        # Serialise against in-flight generation: adapter attachment
        # monkey-patches module forwards — unsafe mid-forward.
        with self._gen_lock:
            return self._load_lora_impl(
                lora_path, rank, alpha, target_modules)

    def _load_lora_impl(self, lora_path: str, rank: int,
                        alpha: int | None,
                        target_modules: list[str] | None) -> int:
        from pathlib import Path

        from safetensors.torch import load_file as _load

        from forge.training.bitnet_lora import add_lora_adapters

        if alpha is None:
            alpha = rank * 2
        if target_modules is None:
            target_modules = ["w_gate", "w_up", "w_down", "q_proj", "v_proj",
                              "out_proj", "in_proj"]

        lora_file = Path(lora_path)
        if not lora_file.exists():
            raise FileNotFoundError(f"LoRA checkpoint not found: {lora_path}")

        # Remove any existing LoRA adapters first
        self.unload_lora()

        self._log(f"Loading LoRA adapter: {lora_file.name} (rank={rank})")
        n_adapters, _ = add_lora_adapters(
            self.model, rank=rank, alpha=alpha,
            target_modules=target_modules)

        state = _load(str(lora_file))
        loaded = 0
        for name, param in self.model.named_parameters():
            if "lora_" in name and name in state:
                param.data.copy_(state[name])
                loaded += 1

        if loaded < len(state):
            missing = set(state.keys()) - {
                n for n, _ in self.model.named_parameters() if "lora_" in n}
            self._log(f"WARNING: {len(state) - loaded} LoRA tensors not found "
                      f"in model (checkpoint has {len(state)}, loaded {loaded}). "
                      f"Missing: {list(missing)[:5]}...",
                      level="warning")

        n_params = sum(v.numel() for v in state.values())
        self._lora_config = {
            "path": str(lora_file), "rank": rank, "alpha": alpha,
            "target_modules": target_modules, "n_adapters": n_adapters,
            "n_params": n_params,
        }
        self._log(f"LoRA loaded: {loaded}/{len(state)} tensors, "
                  f"{n_adapters} adapters, {n_params / 1e6:.1f}M params")
        return loaded

    def unload_lora(self) -> bool:
        """Remove LoRA adapters from the model.

        Detaches all LoRA modules and restores the original forward functions.
        The base model weights are unaffected.

        Returns:
            True if any LoRA adapters were removed, False if none were attached.
        """
        with self._gen_lock:
            return self._unload_lora_impl()

    def _unload_lora_impl(self) -> bool:
        removed = 0
        for name, module in self.model.named_modules():
            # IRIFP4Linear: lora_adapter is a submodule attribute
            if hasattr(module, 'lora_adapter') and module.lora_adapter is not None:
                del module.lora_adapter
                removed += 1
            # nn.Linear with monkey-patched forward: restore original
            if hasattr(module, '_lora_orig_forward'):
                module.forward = module._lora_orig_forward
                del module._lora_orig_forward
                # Also remove the LoRA adapter module if present
                if hasattr(module, 'lora_adapter'):
                    del module.lora_adapter
                removed += 1

        if removed > 0:
            self._log(f"LoRA unloaded: {removed} adapters removed")
            self._lora_config = None
        return removed > 0

    def has_lora(self) -> bool:
        """Check if LoRA adapters are currently loaded."""
        return getattr(self, '_lora_config', None) is not None

    def lora_info(self) -> dict | None:
        """Return info about the currently loaded LoRA, or None."""
        return getattr(self, '_lora_config', None)

