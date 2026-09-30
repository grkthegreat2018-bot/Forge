"""ConfigurableResearchLLM — the modular ForgeAI language model."""
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from forge.config import ModelConfig

logger = logging.getLogger(__name__)

from .attention_ops import _causal_mask
from .kv_cache import KVCache, PreAllocatedKVCache
from .layers import ModularBlock, RMSNorm


class ConfigurableResearchLLM(nn.Module):
    """Full config-driven research language model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        # V5: Factorized embedding (ALBERT pattern, 7.8x param reduction).
        # Init via SVD of original embedding when loading from checkpoint.
        # Combined with BitNet embedding for ~60x total reduction.
        if getattr(config, 'use_factorized_embeddings', False):
            from forge.keys.architecture.factorized_embed_key import FactorizedEmbedding, FactorizedLMHead
            rank = getattr(config, 'embed_factorized_rank', 256)
            self.embed = FactorizedEmbedding(config.vocab_size, config.d_model, rank=rank)
            # Apply BitNet to factorized embedding when both are enabled
            if getattr(config, 'use_bitnet', False):
                from forge.keys.quantization.bitnet_b158_key import build_bitnet_embedding, build_bitnet_linear
                self.embed.embed = build_bitnet_embedding(
                    config, config.vocab_size, rank)
                self.embed.project = build_bitnet_linear(
                    config, rank, config.d_model, bias=False)
            self.head = FactorizedLMHead(self.embed)
        # V12: Kronecker-factored byte-level embedding (drop-in nn.Embedding
        # replacement). Lossless warm-start via from_embedding() SVD when
        # loading from a full-vocab checkpoint. ~8x param reduction.
        elif getattr(config, 'use_kronecker_embed', False):
            from forge.keys.architecture.kronecker_embed_key import KroneckerEmbedding
            d_char = getattr(config, 'kronecker_d_char', 64)
            max_char_len = getattr(config, 'kronecker_max_char_len', 8)
            self.embed = KroneckerEmbedding(
                config.vocab_size, config.d_model,
                d_char=d_char, max_char_len=max_char_len)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # PIT: Pseudo-Inverse Tying (L=I → standard weight tying, lossless).
        elif getattr(config, 'use_pit', False):
            from forge.keys.misc.pit_key import PITEmbedding, PITLMHead
            self.embed = PITEmbedding(config.vocab_size, config.d_model)
            self.head = PITLMHead.from_embedding(self.embed)
        # V5: BitNet embedding (ternary QAT on embedding weight).
        elif getattr(config, 'use_bitnet_embedding', False):
            from forge.keys.quantization.bitnet_b158_key import build_bitnet_embedding
            self.embed = build_bitnet_embedding(config, config.vocab_size, config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.embed = nn.Embedding(config.vocab_size, config.d_model)
            self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.blocks = nn.ModuleList([ModularBlock(config, layer_idx=i) for i in range(config.n_layers)])
        norm = RMSNorm if getattr(config, 'norm_type', 'layernorm') == 'rmsnorm' else nn.LayerNorm
        _use_triton_f = getattr(config, 'use_triton_kernels', False)
        _norm_eps = getattr(config, 'norm_eps', 1e-6)
        # Embedding norm: applied after embedding, before layers (LFM2.5 has this)
        if getattr(config, 'use_embed_norm', False):
            if norm is RMSNorm:
                self.embed_norm = RMSNorm(config.d_model, eps=_norm_eps, use_triton=_use_triton_f)
            else:
                self.embed_norm = norm(config.d_model)
        else:
            self.embed_norm = None
        # Final norm: applied after layers, before head (standard, LFM2.5 doesn't have this)
        if getattr(config, 'use_final_norm', True):
            if norm is RMSNorm:
                self.ln_f = RMSNorm(config.d_model, eps=_norm_eps, use_triton=_use_triton_f)
            else:
                self.ln_f = norm(config.d_model)
        else:
            self.ln_f = None
        # Weight tying: skip if PIT, factorized, or Kronecker is enabled
        # (they handle tying), or if config explicitly disables it (e.g., Qwen2.5).
        if (not getattr(config, 'use_pit', False)
                and not getattr(config, 'use_factorized_embeddings', False)
                and not getattr(config, 'use_kronecker_embed', False)
                and getattr(config, 'tie_word_embeddings', True)):
            self.embed.weight = self.head.weight  # Weight tying

        # V5: Expert tying — share expert weights across consecutive layer groups.
        # Applied after blocks are built but before weight loading.
        # Pointer aliasing: odd layers in each group point to even layers.
        if getattr(config, 'use_moe', False) and getattr(config, 'moe_expert_tying', False):
            from forge.keys.moe.expert_tying_key import ExpertTyingKey
            tie_key = ExpertTyingKey()
            tie_key.apply(self)
            self._expert_tying_applied = True

        # AttnRes: cross-layer retrieval (shared module, gates=0 → lossless).
        # Applied after each block; maintains a buffer of past layer outputs.
        self._attn_res: nn.Module | None = None
        if getattr(config, 'use_attn_residual', False):
            from forge.keys.architecture.attn_residual_key import AttnResModule
            self._attn_res = AttnResModule(
                config.d_model, config.n_layers,
                k=getattr(config, 'attn_res_k', 4),
                n_heads=min(4, config.n_heads))
            self._attn_res_gate_zero: bool | None = None

        # ValueResidual (ResFormer): add V_0 residual to all layers' V.
        # gate=0 at init → lossless. Training opens the gate.
        # V_0 is captured from layer 0's first attention forward and stored
        # as a buffer (detached, no grad) for use by subsequent layers.
        self._use_value_residual = getattr(config, 'use_value_residual', False)
        self._v0_mode = getattr(config, 'value_residual_mode', 'resformer')
        self._v0: torch.Tensor | None = None  # captured during first forward
        self._v0_gates = None
        if self._use_value_residual:
            # Per-layer gate (scalar), init=0 → lossless at start.
            self._v0_gates = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(config.n_layers)
            ])

        # Zero-init residual: output projections start at zero so the residual
        # stream is unchanged at init — cleaner gradient flow early in training
        # (NanoGPT speedrun technique by @Grad62304977).
        if getattr(config, 'zero_init_residual', False):
            for block in self.blocks:
                if hasattr(block.attn, 'out_proj') and hasattr(block.attn.out_proj, 'weight'):
                    block.attn.out_proj.weight.detach().zero_()
                if hasattr(block.ffn, 'w_down') and hasattr(block.ffn.w_down, 'weight'):
                    block.ffn.w_down.weight.detach().zero_()

        self.draft_head: nn.Module | None = None
        if config.use_gradient_checkpointing:
            self.enable_gradient_checkpointing(
                strategy=getattr(config, 'selective_gradient_checkpointing', 'all'))

        # MTP heads (Nemotron Lightning): shared-weight multi-token prediction
        self.mtp_module: nn.Module | None = None
        if getattr(config, 'use_mtp', False):
            from forge.decoding.mtp import MTPModule
            self.mtp_module = MTPModule(
                d_model=config.d_model,
                vocab_size=config.vocab_size,
                n_heads=getattr(config, 'mtp_n_heads', 2),
                loss_weight=(getattr(config, 'mtp_weight', None)
                             if getattr(config, 'mtp_weight', None) is not None
                             else getattr(config, 'mtp_loss_weight', 0.3)),
                identity_init=getattr(config, 'zero_init_residual', True),
            )
            # Tie MTP head to model head (shared weight design)
            # Use tie_head_from_module so the tie survives model.to(device)
            if hasattr(self.mtp_module, 'tie_head_from_module'):
                self.mtp_module.tie_head_from_module(self, "head")
            else:
                self.mtp_module.tie_head_to_model(self.head.weight)

        # V5.2: LiSA — cross-layer Q/K sharing with alignment FFN.
        # Lossless at init: shared Q/K + alignment are zero-init (gate=0).
        # Per-layer Q/K loaded from checkpoint unchanged. Training opens gate.
        self.lisa: nn.Module | None = None
        if getattr(config, 'use_lisa', False):
            from forge.keys.attention.lisa_key import LisaKey
            LisaKey.apply(self, config)

        # V5.2: Hyperloop — looped middle blocks for layer param reduction.
        # Lossless at init: all layers unique, loop gate=0. Training opens gate.
        # We only register the loop block + gates as submodules (not the wrapper,
        # which would create a circular ref: model → wrapper → model).
        if getattr(config, 'use_hyperloop', False):
            from forge.keys.architecture.hyperloop_key import HyperloopKey
            _hl = HyperloopKey.apply(
                self,
                begin=getattr(config, 'hyperloop_begin', 2),
                end=getattr(config, 'hyperloop_end', 2),
                loop_iters=getattr(config, 'hyperloop_loop_iters', 3))
            # Register loop block and gates as direct submodules (no wrapper)
            self.loop_block = _hl.loop_block
            self.loop_gates = _hl.loop_gates
            self.middle_gates = _hl.middle_gates
            # Store config for forward-time hyperloop logic
            self._hyperloop_begin = _hl.begin
            self._hyperloop_end = _hl.end
            self._hyperloop_loop_iters = _hl.loop_iters
            self._hyperloop_n_middle = _hl.n_middle

        # ── R19 keys wiring ────────────────────────────────────────────────
        # These modules are created AFTER all other model modules so their
        # random init does not perturb the main model's weight init sequence
        # (critical for the lossless-at-init test: same seed → same main
        # weights whether keys are ON or OFF).
        #
        # QSA (Qwen Sparse Attention): QSALayer modules are registered but
        # not used in the forward path at init (budget=all blocks = full
        # attention, identity warm start).  Training would gradually swap
        # them in as the indexer learns to skip blocks.
        self.qsa_layers: nn.ModuleList | None = None
        if getattr(config, 'use_qsa', False):
            from forge.keys.attention.qsa_key import QSALayer
            _head_dim = getattr(config, 'head_dim', None) or (
                config.d_model // config.n_heads)
            _budget = getattr(config, 'qsa_top_k', 512)
            _n_kv = getattr(config, 'n_kv_heads', None) or config.n_heads
            self.qsa_layers = nn.ModuleList([
                QSALayer(
                    d_model=config.d_model, n_heads=config.n_heads,
                    n_kv_heads=_n_kv, head_dim=_head_dim,
                    budget_blocks=_budget)
                for _ in range(config.n_layers)
            ])

        # GatedResidual: GatedResidualLayer modules registered but not used
        # in the forward path at init (gate=1.0 = identity).  Training opens
        # the multi-branch residual.
        self.gated_residuals: nn.ModuleList | None = None
        if getattr(config, 'use_gated_residual', False):
            from forge.keys.architecture.gated_residual_key import GatedResidualLayer
            _gr_rank = min(256, config.d_model)
            self.gated_residuals = nn.ModuleList([
                GatedResidualLayer(
                    d_model=config.d_model, n_branches=4,
                    bottleneck_rank=_gr_rank)
                for _ in range(config.n_layers)
            ])

        # NgramEmbedding: adds n-gram lookup embeddings to token embeddings.
        # Lossless at init: table = all zeros → additive zero.  This one IS
        # used in the forward path (the zero table makes it a no-op at init,
        # and training fills in useful n-gram embeddings).
        self.ngram_embed: nn.Module | None = None
        if getattr(config, 'use_ngram_embedding', False):
            from forge.keys.knowledge.ngram_embedding_key import NGramEmbeddingLayer
            self.ngram_embed = NGramEmbeddingLayer(
                vocab_size=config.vocab_size, d_model=config.d_model,
                n_gram=getattr(config, 'ngram_n', 3),
                table_size=getattr(config, 'ngram_dim', 256),
                device=getattr(config, 'device', 'cpu'),
                host_table=getattr(config, 'ngram_host', True))

        # Cache config flags to avoid getattr/hasattr per forward.
        self._use_liger_ce = getattr(config, 'use_liger_ce', False)
        self._liger_fce = None  # lazy-init on first use

        # Device placement cache — scanning next(param).device per layer per
        # forward (17+ Python calls) is pure CPU overhead. Placement is fixed
        # after load/hybrid-offload, so cache it once on first forward.
        self._embed_device: torch.device | None = None
        self._ln_f_device: torch.device | None = None
        self._block_devices: list[torch.device] | None = None

    def cache_devices(self):
        """Scan and cache module device placement (lazy, on first forward)."""
        # Use buffers if no parameters exist (pre-quantized int8 storage)
        embed_params = list(self.embed.parameters())
        if embed_params:
            self._embed_device = embed_params[0].device
        else:
            embed_bufs = list(self.embed.buffers())
            self._embed_device = embed_bufs[0].device if embed_bufs else torch.device("cpu")
        self._ln_f_device = (next(self.ln_f.parameters()).device
                             if self.ln_f is not None else self._embed_device)
        block_devs = []
        for b in self.blocks:
            params = list(b.parameters())
            if params:
                block_devs.append(params[0].device)
            else:
                bufs = list(b.buffers())
                block_devs.append(bufs[0].device if bufs else torch.device("cpu"))
        self._block_devices = block_devs

    def invalidate_device_cache(self):
        """Clear cached device placement (call after moving modules)."""
        self._embed_device = None
        self._ln_f_device = None
        self._block_devices = None

    @torch.no_grad()
    def apply_iri_fp4_quantization(self) -> int:
        """Quantize all 2D nn.Linear weights to IRI-FP4 in-place.

        Replaces every nn.Linear (with 2D weight) in the model with an
        IRIFP4Linear that stores the IRI-FP4 packed weights. This is the
        build-time quantization path for ForgeLM V2 — call after training
        (or after loading a non-quantized checkpoint) to get the 3.5x
        weight compression V2 is designed for.

        Uses ``iri_fp4_rounds`` and ``iri_fp4_block_size`` from the config.

        Returns:
            Number of nn.Linear layers replaced with IRIFP4Linear.
        """
        from forge.keys.quantization.iri_fp4_key import convert_model_to_iri_fp4
        _block = int(getattr(self.config, 'iri_fp4_block_size', 32))
        _rounds = int(getattr(self.config, 'iri_fp4_rounds', 2))
        n_before = sum(1 for m in self.modules()
                       if isinstance(m, nn.Linear))
        convert_model_to_iri_fp4(self, block_size=_block, n_rounds=_rounds)
        self.invalidate_device_cache()
        return n_before

    def enable_gradient_checkpointing(self, strategy: str = "all"):
        """Enable activation checkpointing on all transformer blocks.

        Args:
            strategy: "all" (full block recompute), "ffn" (recompute only the
                FFN — largest activation consumer), "attn" (recompute only
                attention), "none" (no recomputation).
        """
        valid = {"all", "ffn", "attn", "none"}
        if strategy not in valid:
            import warnings
            warnings.warn(
                f"Unknown gradient checkpointing strategy '{strategy}' — "
                f"falling back to 'none' (no checkpointing). "
                f"Valid strategies: {sorted(valid)}",
                stacklevel=2,
            )
        for block in self.blocks:
            block._gradient_checkpointing = True
            block._gradient_checkpointing_strategy = strategy

    def disable_gradient_checkpointing(self):
        """Disable activation checkpointing (e.g., for inference)."""
        for block in self.blocks:
            block._gradient_checkpointing = False

    def compile_for_inference(self, mode: str = "default"):
        """torch.compile the model for inference speedup.

        Uses mode="default" (kernel fusion, no CUDA graphs) by default because
        the pre-allocated KV cache has dynamic fill lengths that are incompatible
        with CUDA graph capture (which requires static memory addresses).

        For training or non-cached inference, use mode="reduce-overhead" for
        CUDA graph acceleration (1.3-2x additional speedup).

        Args:
            mode: "default" (kernel fusion only), "reduce-overhead" (+CUDA graphs),
                  "max-autotune" (kernel autotuning, slower compile).

        Returns:
            The compiled model (replaces self in-place via torch.compile wrapper).
        """
        self.eval()
        return torch.compile(self, mode=mode, dynamic=True)

    def compile_decode_step(self, batch_size: int = 1):
        """Compile a dedicated decode-step forward for CUDA graph acceleration.

        The decode step (single token, B×1) has fixed shapes, enabling
        mode="reduce-overhead" with dynamic=False for proper CUDA graph capture.
        This eliminates per-kernel CPU launch overhead — the primary source of
        CPU spikes during autoregressive generation.

        Must be called once per unique batch_size. The returned callable is
        a compiled forward that accepts (idx, past_key_values, use_cache, attention_mask).

        Args:
            batch_size: fixed batch size for this compiled decode step.

        Returns:
            Compiled forward callable for decode steps of shape (batch_size, 1).
        """
        self.eval()
        # Wrap just the forward with reduce-overhead + static shapes for CUDA graphs.
        return torch.compile(
            self.forward,
            mode="reduce-overhead",
            dynamic=False,
            fullgraph=False,  # allow graph breaks for attention_bias conditional
        )

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[KVCache | None] | None = None,
        use_cache: bool = False,
        return_hidden: bool = False,
        preallocated_cache: Optional["PreAllocatedKVCache"] = None,
        attention_mask: torch.Tensor | None = None,
        # DiffusionBlocks support: run only specific layers, with noise conditioning
        layer_indices: list[int] | None = None,
        noisy_embeds: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
        # Varlen attention (R&D round 14): cu_seqlens for packed sequences.
        cu_seqlens: torch.Tensor | None = None,
        # AdaLN-zero conditioning (DiT): cond embedding for adaptive layer norm.
        cond: torch.Tensor | None = None,
        # SIGReg: return per-layer hidden states for spectral regularization.
        return_hidden_states: bool = False,
        # Generation fast path: compute lm_head only on the last position.
        # Prefill otherwise materializes [B, T, vocab] logits (~500MB /
        # ~1.3 TFLOP at 4k tokens on V2) when callers only read [:, -1].
        logits_last_only: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | tuple[torch.Tensor, torch.Tensor | None, list[KVCache | None]]:
        # Mark a new step for CUDAGraph tree manager. Conv layers clone state
        # buffers in forward (e.g. _conv_state = ...clone()), which triggers
        # "tensor output overwritten by subsequent run" errors under
        # torch.compile + CUDAGraphs. Marking the step boundary tells the
        # manager to start a fresh capture window, preventing the overwrite
        # detection from firing on legitimate state updates.
        torch.compiler.cudagraph_mark_step_begin()
        # FP8 training autocast: wrap forward in FP8 for 2x throughput on
        # Hopper/Blackwell. Falls back to BF16 on older GPUs.
        if getattr(self.config, 'use_fp8_training', False):
            from forge.training.optim.fp8_training import enable_fp8_training
            with enable_fp8_training():
                return self._forward_impl(
                    idx, targets, past_key_values, use_cache, return_hidden,
                    preallocated_cache, attention_mask, layer_indices,
                    noisy_embeds, modulation, cu_seqlens, cond,
                    return_hidden_states, logits_last_only)
        return self._forward_impl(
            idx, targets, past_key_values, use_cache, return_hidden,
            preallocated_cache, attention_mask, layer_indices,
            noisy_embeds, modulation, cu_seqlens, cond,
            return_hidden_states, logits_last_only)

    def _forward_impl(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        past_key_values: list[KVCache | None] | None = None,
        use_cache: bool = False,
        return_hidden: bool = False,
        preallocated_cache: Optional["PreAllocatedKVCache"] = None,
        attention_mask: torch.Tensor | None = None,
        layer_indices: list[int] | None = None,
        noisy_embeds: torch.Tensor | None = None,
        modulation: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        cond: torch.Tensor | None = None,
        return_hidden_states: bool = False,
        logits_last_only: bool = False,
    ):
        position_ids = None
        attention_bias = None  # additive mask for SDPA: (B, 1, T, total_len)
        if attention_mask is not None:
            B, T = idx.shape[:2]
            total_len = attention_mask.shape[1]  # cached + new tokens
            # position_ids: cumsum of mask - 1, clamped to 0 for pad positions.
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids = position_ids.clamp(min=0)
            position_ids = position_ids[:, -T:]  # only for tokens being processed

            # Build additive attention bias ONCE: (B, 1, T, total_len)
            # 0 for real tokens, -inf for pad. Combined with causal for prefill.
            dtype = next(self.parameters()).dtype
            pad_mask = (attention_mask == 0)  # (B, total_len)
            if T == 1 and total_len > 1:
                # Decode: only padding mask, no causal needed.
                # Shape: (B, 1, 1, total_len)
                attention_bias = torch.zeros(B, 1, 1, total_len, device=idx.device, dtype=dtype)
                attention_bias = attention_bias.masked_fill(
                    pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
            elif total_len == T:
                # Prefill: combine causal + padding.
                causal = _causal_mask(T, total_len, 0, idx.device, dtype)
                pad_add = torch.zeros(B, 1, T, total_len, device=idx.device, dtype=dtype)
                pad_add = pad_add.masked_fill(
                    pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
                attention_bias = causal + pad_add
            else:
                # Chunked prefill with cache.
                past_len = total_len - T
                causal = _causal_mask(T, total_len, past_len, idx.device, dtype)
                pad_add = torch.zeros(B, 1, T, total_len, device=idx.device, dtype=dtype)
                pad_add = pad_add.masked_fill(
                    pad_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
                attention_bias = causal + pad_add

        # Move input to embed's device (handles hybrid offload).
        # Device placement is cached after the first forward (it only changes
        # via explicit .to() / hybrid_offload, which call invalidate_device_cache).
        if self._embed_device is None:
            self.cache_devices()
        embed_device = self._embed_device
        if idx.device != embed_device:
            idx = idx.to(embed_device)
        x = self.embed(idx)
        # LFM2.5: apply embedding norm after embedding, before layers
        if self.embed_norm is not None:
            x = self.embed_norm(x)
        # N-gram embedding — add n-gram lookup embeddings to token
        # embeddings.  Lossless at init (table = all zeros → additive zero).
        if self.ngram_embed is not None:
            x = self.ngram_embed(idx, x)
        # DiffusionBlocks: add noisy target embeddings to the input
        if noisy_embeds is not None:
            # noisy_embeds: (B, T, d_model) — added to input embeddings
            if noisy_embeds.shape[:2] == x.shape[:2]:
                x = x + noisy_embeds.to(x.dtype).to(x.device)
            else:
                # Broadcast (B, d_model) → (B, T, d_model)
                x = x + noisy_embeds.unsqueeze(1).expand(-1, x.shape[1], -1).to(x.dtype).to(x.device)
        # DiffusionBlocks: convert layer_indices to a set for O(1) lookup
        active_layers = set(layer_indices) if layer_indices is not None else None
        presents: list[KVCache | None] = []
        # Track device for hybrid offload: move x to each block's device
        cur_device = x.device
        block_devices = self._block_devices
        # AttnRes: only build past_outputs buffer when gates are non-zero.
        # At init (gates=0), skip entirely — zero overhead, bit-exact.
        use_attn_res = self._attn_res is not None
        if use_attn_res and self._attn_res_gate_zero is None:
            self._attn_res_gate_zero = (
                self._attn_res.gates.abs().max().item() == 0.0)
        attn_res_active = use_attn_res and not self._attn_res_gate_zero
        past_outputs: list[torch.Tensor] = [] if attn_res_active else None
        # New sequence: when past_key_values is None, reset conv layer states
        # so stale state from a previous sequence is not reused. This is the
        # correct place to reset — conv layers always get past_key_value=None
        # (they don't have KV cache entries), so they can't detect new vs.
        # continuation at the per-layer level.
        if past_key_values is None:
            for block in self.blocks:
                if hasattr(block, 'attn') and hasattr(block.attn, '_conv_state'):
                    block.attn._conv_state_reset = True
                if hasattr(block, 'attn') and hasattr(block.attn, '_ssm_state'):
                    block.attn._ssm_state = None  # reset Mamba recurrent state
                # Mamba-3: reset complex state (real + imag parts)
                if hasattr(block, 'attn') and hasattr(block.attn, '_ssm_state_real'):
                    block.attn._ssm_state_real = None
                    block.attn._ssm_state_imag = None
                # R49-2 KDA + ForgeHybrid side-paths: same new-sequence reset
                if getattr(block, '_kda', None) is not None:
                    block._kda._state_reset = True
                if getattr(block, '_forge_hybrid_ssm', None) is not None:
                    if hasattr(block._forge_hybrid_ssm, '_ssm_state'):
                        block._forge_hybrid_ssm._ssm_state = None
                    if hasattr(block._forge_hybrid_ssm, '_conv_state'):
                        block._forge_hybrid_ssm._conv_state_reset = True
        # SIGReg: collect per-layer hidden states for spectral regularization.
        hidden_states_list: list[torch.Tensor] = [] if return_hidden_states else None
        for i, block in enumerate(self.blocks):
            # DiffusionBlocks: skip layers not in the active set
            if active_layers is not None and i not in active_layers:
                if use_cache:
                    presents.append(past_key_values[i] if past_key_values is not None else None)
                continue
            block_device = block_devices[i]
            if block_device != cur_device:
                x = x.to(block_device)
                cur_device = block_device
            # ValueResidual: inject V_0 into layers 1+ (gate=0 → lossless).
            # V_0 is captured from layer 0's V projection on the first forward.
            if self._use_value_residual and i > 0 and self._v0 is not None:
                attn = block.attn
                if hasattr(attn, '_v0_residual'):
                    attn._v0_residual = self._v0
                    attn._v0_gate = self._v0_gates[i] if self._v0_gates is not None else None
            # Enable V_0 capture on layer 0 (first forward only).
            if self._use_value_residual and i == 0 and self._v0 is None and not use_cache:
                attn = block.attn
                if hasattr(attn, '_v0_capture'):
                    attn._v0_capture = True  # signal to capture
            past = past_key_values[i] if past_key_values is not None else None
            x, present = block(x, past_key_value=past, use_cache=use_cache,
                               preallocated_cache=preallocated_cache, layer_idx=i,
                               attention_bias=attention_bias, position_ids=position_ids,
                               modulation=modulation, cu_seqlens=cu_seqlens, cond=cond)
            # SIGReg: collect hidden state after this block.
            if hidden_states_list is not None:
                hidden_states_list.append(x)
            # Capture V_0 from layer 0 after its first forward (for training).
            if self._use_value_residual and i == 0 and self._v0 is None and not use_cache:
                attn = block.attn
                if hasattr(attn, '_v0_capture') and attn._v0_capture is not None:
                    self._v0 = attn._v0_capture.detach()
            # AttnRes: cross-layer retrieval (gates=0 → lossless at start).
            if attn_res_active:
                x = x + self._attn_res(x, i, past_outputs)
                past_outputs.append(x)
            if use_cache:
                presents.append(present)
        # Snapshot per-layer recurrent state at the end of a fresh prefill
        # (past_key_values=None).  Decode steps mutate the live _conv_state
        # buffers, so this frozen copy is the only way prefix/session caches
        # can reconstruct the conv context at the prompt boundary later.
        if use_cache and past_key_values is None and idx.shape[1] > 1:
            snap = {}
            for i, block in enumerate(self.blocks):
                attn = getattr(block, 'attn', None)
                if attn is None:
                    continue
                st = {}
                if getattr(attn, '_conv_state', None) is not None:
                    st['conv'] = attn._conv_state.clone()
                if getattr(attn, '_ssm_state', None) is not None:
                    st['ssm'] = attn._ssm_state.clone()
                kda = getattr(block, '_kda', None)
                if kda is not None:
                    kda_snap = kda.snapshot_state()
                    if any(v is not None for v in kda_snap.values()):
                        st['kda'] = kda_snap
                if st:
                    snap[i] = st
            self._last_prefill_recurrent = snap
        # Advance the pre-allocated cache position after all layers processed.
        if preallocated_cache is not None:
            preallocated_cache.advance(idx.shape[1])
        # Final norm (if model has one — LFM2.5 doesn't)
        if self.ln_f is not None:
            ln_f_device = self._ln_f_device
            if x.device != ln_f_device:
                x = x.to(ln_f_device)
            hidden = self.ln_f(x)
        else:
            hidden = x

        # Collect MoE aux_loss from all blocks that have it.
        # Stored as self._last_moe_aux_loss so callers that don't pass targets
        # (e.g. GRPO/RLVR policy-gradient forward) can still add it to their
        # loss — otherwise the MoE router gets no load-balancing signal during
        # RL and can collapse to a single expert.
        moe_aux_loss = torch.tensor(0.0, device=idx.device, dtype=hidden.dtype)
        for block in self.blocks:
            if hasattr(block, '_last_aux_loss') and block._last_aux_loss is not None:
                moe_aux_loss = moe_aux_loss + block._last_aux_loss
        self._last_moe_aux_loss = moe_aux_loss

        # Chunked CE path: skip materializing full [B*T, V] logits for loss.
        # The head Linear + CE are fused into chunked passes over the token dim,
        # saving ~2.8 GB at batch 2 / seq 1024 / vocab 151665.
        if self.config.use_chunked_ce and targets is not None and not use_cache:
            ent_alpha = getattr(self.config, 'entropy_alpha', 0.0)
            if ent_alpha > 0.0:
                from forge.training.losses.chunked_ce import chunked_entropy_weighted_ce
                loss = chunked_entropy_weighted_ce(
                    hidden.view(-1, hidden.size(-1)),
                    self.head.weight,
                    targets.view(-1),
                    chunk_size=self.config.ce_chunk_size,
                    entropy_alpha=ent_alpha,
                )
            else:
                from forge.training.losses.chunked_ce import chunked_linear_cross_entropy
                loss = chunked_linear_cross_entropy(
                    hidden.view(-1, hidden.size(-1)),
                    self.head.weight,
                    targets.view(-1),
                    chunk_size=self.config.ce_chunk_size,
                )
            # Logits are not computed in this path; return None for the logits
            # slot since the training loop only uses the loss.
            logits = None
        elif self._use_liger_ce and targets is not None and not use_cache:
            # Liger-Kernel fused linear cross-entropy: one Triton kernel for
            # head matmul + CE, avoids [B*T, V] logits entirely. Saves ~620MB
            # VRAM (bf16 logits at batch 2 / seq 1024 / vocab 151665).
            # NOT compatible with --compile (graph break kills backward compilation,
            # 5x slower). Use with --no-compile for memory-constrained scenarios.
            from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
            if self._liger_fce is None:
                self._liger_fce = LigerFusedLinearCrossEntropyLoss()
            loss = self._liger_fce(
                self.head.weight,
                hidden.view(-1, hidden.size(-1)),
                targets.view(-1),
            )
            logits = None
        else:
            logits = self.head(
                hidden[:, -1:] if logits_last_only and targets is None
                else hidden)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        if self.draft_head is not None and targets is not None and targets.size(1) > 2:
            draft_logits, draft_loss = self.draft_head(hidden[:, :-2], targets[:, 2:])
            if loss is not None and draft_loss is not None:
                loss = loss + 0.1 * draft_loss

        # Add MoE aux_loss (load balancing) to total loss
        if loss is not None and moe_aux_loss.requires_grad:
            loss = loss + moe_aux_loss

        # MTP loss (Nemotron Lightning): multi-token prediction with shared weights
        if self.mtp_module is not None and targets is not None and not use_cache and targets.size(1) > 3:
            token_embeds = self.embed(idx)  # ground truth token embeddings
            mtp_loss, _ = self.mtp_module(hidden, token_embeds, targets)
            if mtp_loss is not None and loss is not None:
                loss = loss + mtp_loss

        if return_hidden_states:
            # SIGReg: return per-layer hidden states alongside standard outputs.
            if return_hidden:
                if use_cache:
                    return logits, loss, presents, hidden, hidden_states_list
                return logits, loss, hidden, hidden_states_list
            if use_cache:
                return logits, loss, presents, hidden_states_list
            return logits, loss, hidden_states_list
        if return_hidden:
            if use_cache:
                return logits, loss, presents, hidden
            return logits, loss, hidden
        if use_cache:
            return logits, loss, presents
        return logits, loss


