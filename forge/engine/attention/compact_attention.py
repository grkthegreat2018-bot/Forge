"""CompactAttention: block-union KV selection for chunked prefill.

Based on "CompactAttention: Accelerating Chunked Prefill with Block-Union
KV Selection" (arXiv 2605.16839).

Problem: chunked prefill (which we already built) splits long prompts into
chunks. But existing sparse attention methods don't work well with chunked
prefill: block-sparse kernels lose efficiency when Q is short (chunk size),
and fine-grained pattern search is costly when repeated per chunk.

CompactAttention treats 2D block-sparse masks as KV-selection signals (not
execution plans). It converts masks into GQA-aware per-group KV block tables
through:
  1. Q-block union: merge selected KV blocks across query blocks
  2. Intra-group union: merge selections across GQA groups

This produces minimal block tables that preserve all selected KV blocks under
paged execution, enabling selected blocks to be accessed in-place without
explicit KV compaction.

Results: 2.72× attention speedup at 128K context under chunked prefill,
accuracy close to dense attention on RULER benchmark.

For our setup:
  - We already have chunked prefill (ChunkedPrefiller)
  - CompactAttention makes the per-chunk attention efficient
  - Especially valuable for long prompts (8K+ tokens)
  - Composes with our paged KV cache

This implementation provides:
  1. Block-union KV selection from per-chunk attention masks
  2. GQA-aware block table construction
  3. In-place sparse attention on selected blocks
"""
from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def block_union_selection(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_size: int = 16,
    budget_ratio: float = 0.5,
    n_kv_groups: int = 8,
) -> torch.Tensor:
    """Select KV blocks using block-union across Q blocks and GQA groups.

    Instead of per-query-block sparse attention (which is inefficient for
    short Q chunks), this:
      1. Scores all KV blocks against the chunk's queries (lightweight proxy)
      2. Unions the selected blocks across all Q positions in the chunk
      3. Unions across GQA groups (so all heads in a group use the same blocks)
      4. Returns a compact block table for in-place sparse attention

    Args:
        q: (B, n_heads, T_chunk, head_dim) — current chunk queries
        k_cache: (B, n_kv, S, head_dim) — full KV cache
        block_size: KV block size
        budget_ratio: fraction of blocks to select
        n_kv_groups: number of GQA groups (n_kv_heads)

    Returns:
        block_table: (n_selected,) — indices of selected KV blocks
    """
    B, n_heads, T, hd = q.shape
    S = k_cache.shape[2]
    n_blocks = (S + block_size - 1) // block_size

    # Score each KV block against all Q positions in the chunk
    # Use key-query dot product as the proxy (lightweight)
    # Average over Q positions and heads to get per-block score

    # Pad k_cache to block boundary
    pad = block_size - (S % block_size) if S % block_size != 0 else 0
    if pad > 0:
        k_padded = F.pad(k_cache, (0, 0, 0, pad))
    else:
        k_padded = k_cache

    # Reshape to blocks: (B, n_kv, n_blocks, block_size, hd)
    k_blocks = k_padded.view(B, k_cache.shape[1], n_blocks, block_size, hd)

    # Score: max dot product between any Q and any K in the block
    # q: (B, n_heads, T, hd), k_blocks: (B, n_kv, n_blocks, bs, hd)
    # For GQA: average q across heads in each group
    n_rep = n_heads // n_kv_groups
    if n_rep > 1:
        q_grouped = q.view(B, n_kv_groups, n_rep, T, hd).mean(dim=2)  # (B, n_kv, T, hd)
    else:
        q_grouped = q

    # Dot product: q @ k_block_mean^T
    # q: (B, n_kv, T, hd), k_block_mean: (B, n_kv, n_blocks, hd)
    k_block_mean = k_blocks.mean(dim=3)  # (B, n_kv, n_blocks, hd)
    scores = torch.matmul(q_grouped, k_block_mean.transpose(-1, -2))  # (B, n_kv, T, n_blocks)

    # Max over T (any query in the chunk needs this block)
    scores = scores.max(dim=2).values  # (B, n_kv, n_blocks)

    # Union across GQA groups: max over n_kv
    scores = scores.max(dim=1).values  # (B, n_blocks)

    # Select top-k blocks
    n_select = max(1, int(n_blocks * budget_ratio))
    _, block_indices = scores.topk(n_select)
    block_indices = block_indices.sort().values  # keep temporal order

    return block_indices


def compact_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_size: int = 16,
    budget_ratio: float = 0.5,
) -> torch.Tensor:
    """CompactAttention: block-union KV selection + in-place sparse attention.

    Args:
        q: (B, n_heads, T_chunk, head_dim) — chunk queries
        k_cache: (B, n_kv, S, head_dim) — full KV cache
        v_cache: (B, n_kv, S, head_dim) — full KV cache
        block_size: KV block size
        budget_ratio: fraction of blocks to compute

    Returns:
        out: (B, n_heads, T_chunk, head_dim)
    """
    B, n_heads, T, hd = q.shape
    S = k_cache.shape[2]
    n_blocks = (S + block_size - 1) // block_size

    # Short sequence or full budget: use dense attention
    if budget_ratio >= 1.0 or n_blocks <= 4:
        n_kv = k_cache.shape[1]
        n_rep = n_heads // n_kv
        if n_rep > 1:
            k = k_cache[:, :, None, :, :].expand(B, n_kv, n_rep, S, hd).reshape(B, n_heads, S, hd)
            v = v_cache[:, :, None, :, :].expand(B, n_kv, n_rep, S, hd).reshape(B, n_heads, S, hd)
        else:
            k, v = k_cache, v_cache
        return F.scaled_dot_product_attention(q, k, v, is_causal=T > 1)

    # Block-union selection
    block_table = block_union_selection(
        q, k_cache, block_size=block_size,
        budget_ratio=budget_ratio,
        n_kv_groups=k_cache.shape[1],
    )
    # block_table: (B, n_select) — squeeze batch (B=1 for decode/prefill)
    block_table = block_table.squeeze(0) if block_table.dim() > 1 else block_table

    # Gather selected blocks
    k_parts = []
    v_parts = []
    for blk_idx in block_table:
        s = blk_idx.item() * block_size
        e = min(s + block_size, S)
        k_parts.append(k_cache[:, :, s:e])
        v_parts.append(v_cache[:, :, s:e])

    k_selected = torch.cat(k_parts, dim=2)
    v_selected = torch.cat(v_parts, dim=2)

    # GQA repeat
    n_kv = k_selected.shape[1]
    n_rep = n_heads // n_kv
    if n_rep > 1:
        S_sel = k_selected.shape[2]
        k_selected = k_selected[:, :, None, :, :].expand(B, n_kv, n_rep, S_sel, hd).reshape(B, n_heads, S_sel, hd)
        v_selected = v_selected[:, :, None, :, :].expand(B, n_kv, n_rep, S_sel, hd).reshape(B, n_heads, S_sel, hd)

    # Dense attention on selected blocks
    return F.scaled_dot_product_attention(q, k_selected, v_selected, is_causal=T > 1)


class CompactAttentionWrapper:
    """Wraps model attention to use CompactAttention for chunked prefill.

    Patches attention to use compact_attention during prefill (T > 1) when
    the KV cache is long. Decode (T=1) uses standard attention.
    """

    def __init__(self, block_size: int = 16, budget_ratio: float = 0.5,
                 min_kv_len: int = 2048):
        self.block_size = block_size
        self.budget_ratio = budget_ratio
        self.min_kv_len = min_kv_len
        self._active = False
        self._original_forwards = {}

    def apply(self, model: torch.nn.Module):
        from forge.model_loader import GroupedQueryAttention
        count = 0
        for name, module in model.named_modules():
            if isinstance(module, (GroupedQueryAttention,)) or \
               type(module).__name__ in ("GroupedTiedAttention", "GroupedLatentAttention"):
                self._patch(module, name)
                count += 1
        self._active = True
        logger.info(f"  [CompactAttn] Patched {count} attention layers "
              f"(budget={self.budget_ratio}, min_kv={self.min_kv_len})")

    def _patch(self, attn_module, name: str):
        original_forward = attn_module.forward
        self._original_forwards[name] = original_forward

        def compact_forward(self, x, past_key_value=None, use_cache=False,
                            preallocated_cache=None, layer_idx=0,
                            attention_bias=None, position_ids=None):
            B, T, C = x.shape
            hd = self.head_dim

            # Only use CompactAttention for prefill (T > 1) with long KV
            if T <= 1 or preallocated_cache is None or attention_bias is not None:
                return original_forward(
                    x, past_key_value=past_key_value, use_cache=use_cache,
                    preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                    attention_bias=attention_bias, position_ids=position_ids)

            past_len = preallocated_cache.position
            if past_len < self._compact_min_kv:
                return original_forward(
                    x, past_key_value=past_key_value, use_cache=use_cache,
                    preallocated_cache=preallocated_cache, layer_idx=layer_idx,
                    attention_bias=attention_bias, position_ids=position_ids)

            # Standard Q/K/V projection + RoPE + cache
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
                q = self.rope(q, offset=past_len, position_ids=position_ids)
                k = self.rope(k, offset=past_len, position_ids=position_ids)

            preallocated_cache.append(layer_idx, k, v)
            k_cache = preallocated_cache.k_caches[layer_idx][:, :, :past_len + T]
            v_cache = preallocated_cache.v_caches[layer_idx][:, :, :past_len + T]

            new_kv = (k_cache, v_cache) if use_cache else None

            # CompactAttention: block-union KV selection + sparse attention
            out = compact_attention(
                q, k_cache, v_cache,
                block_size=16,
                budget_ratio=self._compact_budget,
            )
            out = out.transpose(1, 2).reshape(B, T, -1)
            return self.out_proj(out), new_kv

        attn_module._compact_min_kv = self.min_kv_len
        attn_module._compact_budget = self.budget_ratio
        attn_module.forward = compact_forward.__get__(attn_module, type(attn_module))

    def revert(self, model: torch.nn.Module):
        for name, module in model.named_modules():
            if name in self._original_forwards:
                module.forward = self._original_forwards[name]
        self._original_forwards.clear()
        self._active = False
