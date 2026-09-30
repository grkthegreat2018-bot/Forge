"""MoSA: Mixture of Sparse Attention via Expert-Choice Routing.

Based on "Mixture of Sparse Attention: Content-Based Learnable Sparse
Attention via Expert-Choice Routing" (arXiv 2505.00315).

Key insight: instead of dense attention where every head attends to every
token, MoSA treats each attention head as an "expert" and uses expert-choice
routing to let each head SELECT which tokens it attends to.

This creates arbitrary sparse attention patterns that are:
  - Content-based (learned, not fixed like local/windowed)
  - Head-specific (different heads attend to different token subsets)
  - Perfectly balanced (expert-choice routing ensures equal load)

Results:
  - Up to 27% better perplexity than dense attention at same compute budget
  - Faster wall-clock time despite no custom kernel
  - Less memory for training
  - Drastically smaller KV cache

For our model (6 attention layers, 32 heads):
  - Dense: each head attends to all T tokens → O(T²) per head
  - MoSA with k=T/2: each head selects T/2 tokens → O(T²/4) per head
  - With 2× heads (same compute): O(T²/4) × 2 = O(T²/2) → 2× speedup
  - Or: same heads, k=T/4 → 4× attention speedup

This implementation provides:
  1. MoSAAttention: expert-choice routed sparse attention
  2. Integration with existing attention layers
  3. Load balancing via expert-choice (no auxiliary loss needed)
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MoSAAttention(nn.Module):
    """Mixture of Sparse Attention with expert-choice routing.

    Each attention head is an "expert" that selects k tokens from the
    sequence to attend to. Selection is based on a router score.

    Args:
        n_heads: number of attention heads (experts)
        head_dim: dimension per head
        n_kv_heads: number of KV heads (for GQA)
        k_ratio: fraction of tokens each head selects (0.5 = half)
        router_dim: router hidden dimension
    """

    def __init__(self, n_heads: int, head_dim: int, n_kv_heads: int,
                 k_ratio: float = 0.5, router_dim: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.k_ratio = k_ratio
        self.scale = 1.0 / math.sqrt(head_dim)

        # Router: scores each token for each head
        self.router = nn.Linear(head_dim, n_heads, bias=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                attention_bias: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            q: (B, n_heads, T, head_dim)
            k: (B, n_kv, T, head_dim) — KV cache keys
            v: (B, n_kv, T, head_dim) — KV cache values

        Returns:
            out: (B, n_heads, T, head_dim)
        """
        B, n_h, T, hd = q.shape
        k_tokens = max(1, int(T * self.k_ratio))

        # Router scores: (B, T, n_heads)
        # Use mean-pooled Q as router input
        router_input = q.mean(dim=1)  # (B, T, hd)
        router_scores = self.router(router_input)  # (B, T, n_heads)

        # Expert-choice routing: each head selects top-k tokens
        # Transpose: (B, n_heads, T)
        router_scores_t = router_scores.transpose(1, 2)

        # Top-k tokens per head
        topk_scores, topk_indices = router_scores_t.topk(k_tokens, dim=-1)

        # For each head, gather selected K/V and compute attention
        # q stays the same (all T queries), but K/V are selected per head
        outputs = []
        for h in range(n_h):
            # Selected token indices for this head: (B, k_tokens)
            indices_h = topk_indices[:, h]  # (B, k_tokens)

            # Gather K/V for selected tokens
            # k: (B, n_kv, T, hd) → need to gather along T dimension
            # For GQA: map head h to kv head h * n_kv // n_heads
            kv_head = h * self.n_kv_heads // n_h
            k_h = k[:, kv_head]  # (B, T, hd)
            v_h = v[:, kv_head]  # (B, T, hd)

            # Gather: (B, k_tokens, hd)
            k_selected = torch.gather(
                k_h, 1,
                indices_h.unsqueeze(-1).expand(-1, -1, hd))
            v_selected = torch.gather(
                v_h, 1,
                indices_h.unsqueeze(-1).expand(-1, -1, hd))

            # Q for this head: (B, T, hd)
            q_h = q[:, h]  # (B, T, hd)

            # Attention: (B, T, k_tokens)
            attn = torch.matmul(q_h, k_selected.transpose(-1, -2)) * self.scale

            # Apply router scores as attention bias (encourage attending to selected)
            # topk_scores: (B, k_tokens) → expand to (B, T, k_tokens)
            attn = attn + topk_scores[:, h].unsqueeze(1).log() * 0.1

            attn = F.softmax(attn, dim=-1)
            out_h = torch.matmul(attn, v_selected)  # (B, T, hd)
            outputs.append(out_h)

        # Stack: (B, n_heads, T, hd)
        out = torch.stack(outputs, dim=1)
        return out


class MoSAWrapper:
    """Wraps model attention to use MoSA for long sequences.

    Patches attention forward to use MoSA when sequence length exceeds
    a threshold. Short sequences use standard dense attention.
    """

    def __init__(self, k_ratio: float = 0.5, min_seq_len: int = 512):
        self.k_ratio = k_ratio
        self.min_seq_len = min_seq_len
        self._active = False
        self._original_forwards = {}
        self._mosa_layers = {}

    def apply(self, model: nn.Module):
        from forge.model_loader import GroupedQueryAttention
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, (GroupedQueryAttention,)) or \
               type(module).__name__ in ("GroupedTiedAttention", "GroupedLatentAttention"):
                # Create MoSA attention for this layer
                mosa = MoSAAttention(
                    n_heads=module.n_heads,
                    head_dim=module.head_dim,
                    n_kv_heads=module.n_kv_heads,
                    k_ratio=self.k_ratio)
                mosa = mosa.to(next(module.parameters()).device)
                self._mosa_layers[name] = mosa
                self._patch(module, name)
                count += 1
        self._active = True
        logger.info(f"  [MoSA] Patched {count} attention layers "
              f"(k_ratio={self.k_ratio}, min_seq={self.min_seq_len})")

    def _patch(self, attn_module, name: str):
        original_forward = attn_module.forward
        self._original_forwards[name] = original_forward
        mosa = self._mosa_layers[name]

        def mosa_forward(self, x, past_key_value=None, use_cache=False,
                         preallocated_cache=None, layer_idx=0,
                         attention_bias=None, position_ids=None):
            B, T, C = x.shape
            hd = self.head_dim

            # Use MoSA only for long sequences
            if self._mosa_min_seq > T or attention_bias is not None:
                return original_forward(
                    x, past_key_value=past_key_value, use_cache=use_cache,
                    preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                    attention_bias=attention_bias, position_ids=position_ids)

            # Standard Q/K/V projection
            q = self.q_proj(x).view(B, T, self.n_heads, hd).transpose(1, 2)
            k = self.k_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)
            if hasattr(self, '_identity') and self._identity:
                v = k
            else:
                v = self.v_proj(x).view(B, T, self.n_kv_heads, hd).transpose(1, 2)

            if self.use_qk_norm and not getattr(self, '_qk_norm_identity', True):
                q = self.q_norm(q)
                k = self.k_norm(k)

            if self.rope is not None:  # use_rope=False (Jamba V2): skip
                q = self.rope(q, offset=0, position_ids=position_ids)
                k = self.rope(k, offset=0, position_ids=position_ids)

            if preallocated_cache is not None:
                preallocated_cache.append(layer_idx, k, v)
                k_cache = preallocated_cache.k_caches[layer_idx][:, :, :T]
                v_cache = preallocated_cache.v_caches[layer_idx][:, :, :T]
            else:
                k_cache, v_cache = k, v

            new_kv = (k_cache, v_cache) if use_cache else None

            # MoSA sparse attention
            out = self._mosa_module(q, k_cache, v_cache)
            out = out.transpose(1, 2).reshape(B, T, -1)
            return self.out_proj(out), new_kv

        attn_module._mosa_min_seq = self.min_seq_len
        attn_module._mosa_module = mosa
        attn_module.forward = mosa_forward.__get__(attn_module, type(attn_module))

    def revert(self, model: nn.Module):
        for name, module in model.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        self._original_forwards.clear()
        self._mosa_layers.clear()
        self._active = False
