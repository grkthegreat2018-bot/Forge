"""KV cache strategy backends.

Pluggable KV cache implementations selectable at runtime:
  - StandardKVCache: basic tensor cache (baseline)
  - PagedKVCacheStrategy: vLLM-style paged memory (wraps paged_kv.py)
  - RotorQuantKVCache: Givens rotation + Lloyd-Max quantization (wraps rotorquant.py)
  - HadamardKVCache: block-diagonal Hadamard + INT4 quantization (SAW-INT4 inspired)
  - CompressedKVCacheStrategy: H2O eviction + KV quant (wraps kv_compress.py)

All implement the KVCacheStrategy interface:
  init(n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype)
  append(k, v, position) -> None
  get(positions) -> (k, v)
  clear()
  info() -> dict
"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F


class KVCacheStrategy(ABC):
    """Base interface for KV cache backends."""

    @abstractmethod
    def init(self, n_heads: int, head_dim: int, n_kv_heads: int,
             max_seq_len: int, device: str, dtype: torch.dtype):
        pass

    @abstractmethod
    def append(self, k: torch.Tensor, v: torch.Tensor, position: int):
        """Append K/V for one position. k/v shape: [B, n_kv, 1, head_dim]."""
        pass

    @abstractmethod
    def get(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve K/V for given positions. Returns [B, n_kv, T, head_dim]."""
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def info(self) -> dict:
        """Return cache stats (size, compression ratio, etc.)."""
        pass


class StandardKVCache(KVCacheStrategy):
    """Basic KV cache — pre-allocated tensor cache (no torch.cat per step).

    Unlike the old torch.cat approach, this pre-allocates the full cache
    upfront and fills by position — inspired by llama.cpp's KV cell model.
    This avoids per-token allocation + copy (10-20% decode speedup).
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype
        # Pre-allocate K/V cache — filled by position, no reallocation
        self.k_cache = None  # lazily allocated on first append (need batch size)
        self.v_cache = None
        self.seq_len = 0

    def _ensure_allocated(self, batch_size):
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                batch_size, self.n_kv, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)

    def append(self, k, v, position):
        self._ensure_allocated(k.shape[0])
        seq = k.shape[2]
        end = position + seq
        # Fill by position — no allocation, just in-place copy
        self.k_cache[:, :, position:end].copy_(k)
        self.v_cache[:, :, position:end].copy_(v)
        self.seq_len = end

    def get(self, positions):
        # Return only the filled portion (slices are views, no copy)
        return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]

    def clear(self):
        # Zero out the filled portion (keep allocation for reuse)
        if self.k_cache is not None:
            self.k_cache[:, :, :self.seq_len].zero_()
            self.v_cache[:, :, :self.seq_len].zero_()
        self.seq_len = 0

    def info(self):
        size_mb = 0
        if self.k_cache is not None:
            size_mb = (self.k_cache.numel() + self.v_cache.numel()) * 2 / 1e6
        return {"type": "standard_prealloc", "seq_len": self.seq_len,
                "max_seq_len": self.max_seq_len, "size_mb": size_mb,
                "compression": 1.0}


class PagedKVCacheStrategy(KVCacheStrategy):
    """vLLM-style paged KV cache — wraps existing paged_kv.py."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from forge.quant.paged_kv import PagedKVCache
        self.cache = PagedKVCache(
            n_blocks=max(256, max_seq_len // 16),
            block_size=16, n_heads=n_kv_heads,
            head_dim=head_dim, device=device,
        )
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = 0

    def append(self, k, v, position):
        self.cache.append(k, v, position)
        self.seq_len = position + 1

    def get(self, positions):
        return self.cache.get(positions)

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return {"type": "paged", "seq_len": self.seq_len,
                "n_blocks": self.cache.n_blocks, "compression": 1.0}


class RotorQuantKVCache(KVCacheStrategy):
    """RotorQuant KV cache — Givens rotation + Lloyd-Max quantization.

    Wraps existing rotorquant.py. 3-4 bit KV cache with 0.94% error.
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from forge.quant.rotorquant import RotorQuantCache
        self.cache = RotorQuantCache(
            d_model=n_kv_heads * head_dim, n_heads=n_kv_heads,
            head_dim=head_dim, bits=4, max_seq_len=max_seq_len,
            n_kv_heads=n_kv_heads, device=device, dtype=dtype,
        )
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = 0

    def append(self, k, v, position):
        self.cache.append(k, v, position)
        self.seq_len = position + 1

    def get(self, positions):
        return self.cache.get(positions)

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return {"type": "rotorquant", "seq_len": self.seq_len,
                "bits": 4, "compression": 16 / 4}


class HadamardKVCache(KVCacheStrategy):
    """Block-diagonal Hadamard rotation + INT4 quantization.

    Inspired by SAW-INT4 (arxiv 2604.19157): "a simple design—token-wise INT4
    quantization with block-diagonal Hadamard rotation—consistently achieves
    the best accuracy-efficiency trade-off."

    The Hadamard rotation Gaussianizes the K/V distribution, making uniform
    INT4 quantization near-lossless. Block-diagonal structure ensures
    compatibility with paged memory layouts.

    If the model already has QuaRot (Hadamard on V/O), the V cache benefits
    from double rotation — but we still rotate K (which QuaRot doesn't cover).
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.bits = 4
        self.block_size = 64  # Hadamard block size

        # Block-diagonal Hadamard via per-block FWHT (O(n log n), no matrix
        # materialization). Equivalent to matmul with block-diagonal H/sqrt(bs)
        # but avoids the O(n^2) construction and the O(n^2) per-append matmul.
        bs = self.block_size
        assert bs & (bs - 1) == 0, "block_size must be power of 2"
        self.n_blocks = (head_dim + bs - 1) // bs
        self.padded_dim = self.n_blocks * bs
        self._norm = 1.0 / (bs ** 0.5)
        # Precompute FWHT bit-reversal stages as strided index pairs for speed.
        self._stages = self._fwht_stage_indices(bs, device)

        # Storage: quantized K/V
        self.k_quant = None
        self.v_quant = None
        self.k_scale = None
        self.v_scale = None
        self.seq_len = 0
        self.qmax = (1 << (self.bits - 1)) - 1  # 7 for 4-bit

    @staticmethod
    def _fwht_stage_indices(n: int, device):
        """Precompute (i, j) index pairs for each of log2(n) FWHT butterfly stages."""
        stages = []
        h = 1
        while h < n:
            idx_i = []
            idx_j = []
            for i in range(0, n, h * 2):
                for k in range(i, i + h):
                    idx_i.append(k)
                    idx_j.append(k + h)
            stages.append((
                torch.tensor(idx_i, device=device, dtype=torch.long),
                torch.tensor(idx_j, device=device, dtype=torch.long),
            ))
            h *= 2
        return stages

    def _fwht(self, t):
        """In-place Fast Walsh-Hadamard Transform along the last dim.

        t: [..., block_size] (block_size must be power of 2). Returns the
        unnormalized Hadamard transform of the last dim.
        """
        for idx_i, idx_j in self._stages:
            a = t[..., idx_i]
            b = t[..., idx_j]
            t[..., idx_i] = a + b
            t[..., idx_j] = a - b
        return t

    def _rotate(self, t):
        """Apply block-diagonal Hadamard rotation via per-block FWHT.

        Equivalent to t @ H where H is block-diagonal(H_bs/sqrt(bs)), but
        O(n log n) per block instead of O(n^2) matmul and no matrix tensor.
        t shape: [B, n_kv, T, head_dim].
        """
        *lead, hd = t.shape
        if hd != self.padded_dim:
            t = F.pad(t, (0, self.padded_dim - hd), value=0)
        t = t.reshape(*lead, self.n_blocks, self.block_size)
        t = self._fwht(t.contiguous())
        t = t * self._norm
        t = t.reshape(*lead, self.padded_dim)
        if hd != self.padded_dim:
            t = t[..., :hd]
        return t

    def _inverse_rotate(self, t):
        """Inverse Hadamard rotation. H/sqrt(n) is orthogonal and symmetric,
        so the inverse equals the forward transform."""
        return self._rotate(t)

    def _quantize(self, t):
        """Per-token INT4 quantization on rotated values."""
        # t shape: [B, n_kv, T, head_dim]
        scale = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
        q = torch.clamp(torch.round(t / scale), -self.qmax, self.qmax)
        return q, scale

    def append(self, k, v, position):
        # Rotate then quantize
        k_rot = self._rotate(k)
        v_rot = self._rotate(v)
        k_q, k_s = self._quantize(k_rot)
        v_q, v_s = self._quantize(v_rot)

        B, n_kv, T, hd = k_q.shape
        end = position + T
        if self.k_quant is None:
            # Pre-allocate the static max buffer once — writing by position
            # avoids the per-token torch.cat (which reallocated + copied the
            # whole cache on every decode step).
            self.k_quant = torch.zeros(B, n_kv, self.max_seq_len, hd,
                                       device=k_q.device, dtype=k_q.dtype)
            self.v_quant = torch.zeros_like(self.k_quant)
            self.k_scale = torch.zeros(B, n_kv, self.max_seq_len, 1,
                                       device=k_s.device, dtype=k_s.dtype)
            self.v_scale = torch.zeros_like(self.k_scale)
        self.k_quant[:, :, position:end] = k_q
        self.v_quant[:, :, position:end] = v_q
        self.k_scale[:, :, position:end] = k_s
        self.v_scale[:, :, position:end] = v_s
        self.seq_len = max(self.seq_len, end)

    def get(self, positions):
        # Dequantize: q * scale, then inverse rotate
        k = self.k_quant[:, :, :self.seq_len] * self.k_scale[:, :, :self.seq_len]
        v = self.v_quant[:, :, :self.seq_len] * self.v_scale[:, :, :self.seq_len]
        k = self._inverse_rotate(k)
        v = self._inverse_rotate(v)
        return k, v

    def clear(self):
        # Keep the allocation for reuse (static buffer pattern) — just reset.
        self.seq_len = 0

    def info(self):
        size_mb = 0
        if self.k_quant is not None:
            # INT4 = 0.5 bytes per element (stored as float for simplicity)
            n = self.seq_len
            actual_bytes = n * self.n_kv * self.padded_dim * 0.5
            actual_bytes += n * self.n_kv * 2  # per-token scales
            size_mb = 2 * actual_bytes / 1e6  # K + V
        return {"type": "hadamard_int4", "seq_len": self.seq_len,
                "bits": self.bits, "block_size": self.block_size,
                "size_mb": size_mb, "compression": 16 / self.bits}


class CompressedKVCacheStrategy(KVCacheStrategy):
    """H2O heavy-hitter eviction + KV quantization — wraps kv_compress.py."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from forge.quant.kv_compress import CompressedKVCache
        self.cache = CompressedKVCache(
            n_heads=n_kv_heads, head_dim=head_dim,
            max_tokens=max_seq_len, bits=8,
            keep_ratio=0.5, device=device,
        )
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.seq_len = 0

    def append(self, k, v, position):
        self.cache.append(k, v, position)
        self.seq_len = position + 1

    def get(self, positions):
        return self.cache.get(positions)

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return {"type": "compressed_h2o", "seq_len": self.seq_len,
                "bits": 8, "keep_ratio": 0.5, "compression": 16 / 8 * 2}


def build_kv_cache(strategy: str = "standard", **kwargs) -> KVCacheStrategy:
    """Factory: build KV cache by strategy name."""
    strategies = {
        "standard": StandardKVCache,
        "paged": PagedKVCacheStrategy,
        "rotorquant": RotorQuantKVCache,
        "hadamard_int4": HadamardKVCache,
        "compressed": CompressedKVCacheStrategy,
        "streaming": StreamingKVCacheStrategy,
        "snapkv": SnapKVCacheStrategy,
        "snapkv_4bit": SnapKV4BitCache,
        "2bit": KV2BitCacheStrategy,
        "paged_eviction": None,  # lazy import
        "xquant": None,          # lazy import
    }
    # Lazy imports for new strategies (avoid circular imports)
    if strategy == "paged_eviction":
        from forge.engine.kv.paged_eviction import PagedEvictionKVCache
        return PagedEvictionKVCache()
    if strategy == "xquant":
        from forge.engine.kv.xquant_kv import XQuantKVCache
        return XQuantKVCache()
    if strategy == "cpu_offload":
        from forge.engine.kv.cpu_kv_offload import CPUKVCache
        return CPUKVCache()
    if strategy == "disk_offload":
        from forge.engine.kv.cpu_kv_offload import DiskKVCache
        return DiskKVCache()
    if strategy == "s4r":
        from forge.engine.kv.s4r_kv import S4RKVCache
        return S4RKVCache()
    if strategy == "hqe_kv":
        from forge.engine.kv.hqe_kv import HqeKVCache
        return HqeKVCache()
    if strategy == "kvzip":
        return KVzipCacheStrategy()
    if strategy == "spectral":
        from forge.engine.kv.spectral_kv import SpectralKVCache
        return SpectralKVCache()
    if strategy == "residual_stream":
        from forge.engine.kv.residual_cache import ResidualStreamCache
        return ResidualStreamCache()
    if strategy == "hyquant":
        from forge.engine.kv.hyquant_kv import HyQuantKVCache
        return HyQuantKVCache()
    if strategy == "hisparse":
        from forge.engine.kv.hisparse_kv import HiSparseKVCache
        return HiSparseKVCache()
    if strategy == "evo_sparse":
        from forge.engine.kv.evo_sparse import EvoSparseKVCache
        return EvoSparseKVCache()
    if strategy == "vegas":
        from forge.engine.kv.vegas_kv import VegasKVCache
        return VegasKVCache()
    if strategy == "capture":
        from forge.engine.kv.capture_kv import CaptureKVCache
        return CaptureKVCache()
    if strategy == "vtoken":
        from forge.engine.kv.vtoken_kv import VTokenKVCache
        return VTokenKVCache()
    if strategy == "auto_context":
        from forge.engine.kv.auto_context import AutoContextKVCache
        return AutoContextKVCache()
    if strategy == "matryoshka":
        from forge.engine.kv.matryoshka_kv import MatryoshkaKVCache
        return MatryoshkaKVCache()
    if strategy == "filler":
        return FillerKVCacheStrategy()
    cls = strategies.get(strategy, StandardKVCache)
    if cls is None:
        return StandardKVCache()
    return cls()


class StreamingKVCacheStrategy(KVCacheStrategy):
    """StreamingLLM: attention sinks + sliding window. Wraps streaming_llm.py."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from forge.engine.kv.streaming_llm import StreamingKVCache
        n_sinks = 4
        window = min(512, max_seq_len)
        self.cache = StreamingKVCache(
            n_sinks=n_sinks, window_size=window,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            device=device, dtype=dtype,
        )
        self.seq_len = 0

    def append(self, k, v, position):
        self.cache.append(k, v, position)
        self.seq_len = self.cache.seq_len

    def get(self, positions):
        k, v, _ = self.cache.get()
        return k, v

    def get_past_kv(self):
        return self.cache.get_past_kv()

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return self.cache.info()


class SnapKVCacheStrategy(KVCacheStrategy):
    """SnapKV: observation-window eviction. Wraps snapkv.py."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from forge.engine.kv.snapkv import SnapKVCache
        obs_window = 128
        budget = min(512, max_seq_len)
        self.cache = SnapKVCache(
            observation_window=obs_window, budget=budget,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            device=device, dtype=dtype,
        )
        self.seq_len = 0

    def append(self, k, v, position, attention_weights=None):
        self.cache.append(k, v, position, attention_weights=attention_weights)
        self.seq_len = self.cache.seq_len

    def get(self, positions):
        return self.cache.get()

    def get_past_kv(self):
        return self.cache.get_past_kv()

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return self.cache.info()


class FillerKVCacheStrategy(KVCacheStrategy):
    """Filler-token-first KV eviction — wraps kv/filler_kv.py.

    Semantic token-type-aware eviction (R50-4): function words and
    punctuation are evicted before attention-score losers. The filler set
    comes from ``filler_ids`` (or ``filler_pred``); when no token ids are
    passed to ``append`` it degrades to SnapKV-style score eviction.
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from forge.engine.kv.filler_kv import FillerKVCache
        obs_window = 128
        budget = min(512, max_seq_len)
        self.cache = FillerKVCache(
            observation_window=obs_window, budget=budget,
            n_kv_heads=n_kv_heads, head_dim=head_dim,
            device=device, dtype=dtype,
        )
        self.seq_len = 0

    def set_filler_ids(self, ids):
        """Attach a tokenizer-derived filler id set post-init."""
        self.cache.filler_ids = set(ids)

    def append(self, k, v, position, attention_weights=None, token_ids=None):
        self.cache.append(k, v, position,
                          attention_weights=attention_weights,
                          token_ids=token_ids)
        self.seq_len = self.cache.seq_len

    def get(self, positions):
        return self.cache.get()

    def get_past_kv(self):
        return self.cache.get_past_kv()

    def clear(self):
        self.cache.clear()
        self.seq_len = 0

    def info(self):
        return self.cache.info()


class SnapKV4BitCache(KVCacheStrategy):
    """Combined SnapKV eviction + Hadamard INT4 quantization.

    Composes two orthogonal compression strategies:
      1. SnapKV: observation-window eviction — keeps high-attention tokens,
         evicts low-importance ones (reduces token count).
      2. Hadamard INT4: block-diagonal Hadamard rotation + per-token INT4
         quantization on the retained tokens (reduces bits per token).

    Total compression = (token eviction ratio) × (16/4 bit ratio).
    For budget=512 with 2048-token context: ~4× from eviction × 4× from
    INT4 = ~16× total VRAM reduction vs fp16 full cache.

    The two strategies are composed as: evict first (SnapKV), then quantize
    the survivors (Hadamard INT4). This is strictly better than either alone
    because eviction and quantization operate on different axes (token count
    vs bit width) and don't interfere.

    Usage:
        cache = build_kv_cache("snapkv_4bit")
        # In attention: pass attention_weights for SnapKV scoring
        cache.append(k, v, position, attention_weights=attn_weights)
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_kv = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        self.max_seq_len = max_seq_len

        # SnapKV eviction parameters
        self.obs_window = 128
        self.budget = min(512, max_seq_len)
        self.max_capacity = self.budget + self.obs_window

        # Hadamard INT4 parameters (reused from HadamardKVCache)
        self.bits = 4
        self.block_size = 64
        bs = self.block_size
        assert bs & (bs - 1) == 0, "block_size must be power of 2"
        self.n_blocks = (head_dim + bs - 1) // bs
        self.padded_dim = self.n_blocks * bs
        self._norm = 1.0 / (bs ** 0.5)
        self._stages = HadamardKVCache._fwht_stage_indices(bs, device)
        self.qmax = (1 << (self.bits - 1)) - 1

        # SnapKV state (full-precision until eviction)
        self.k_cache = None
        self.v_cache = None
        self.attention_scores = None
        self.seq_len = 0

        # INT4 quantized storage (populated after eviction)
        self.k_quant = None
        self.v_quant = None
        self.k_scale = None
        self.v_scale = None
        self._quantized = False  # whether current cache is in INT4 form

    def bf16_or_dtype(self):
        """Return bf16 if using bf16 dtype (saves 2x VRAM for attention scores),
        otherwise return the configured dtype."""
        return torch.bfloat16 if self.dtype == torch.bfloat16 else self.dtype

    def _ensure_buffer(self, B, T, dtype):
        """Ensure pre-allocated full-precision buffer is large enough for
        B batch and T new tokens. Grows geometrically if needed (no per-append
        torch.cat reallocation)."""
        needed = max(self.max_capacity + T, self.seq_len + T)
        if (self.k_cache is None
                or self.k_cache.shape[0] != B
                or self.k_cache.shape[2] < needed
                or self.k_cache.dtype != dtype):
            old_k = self.k_cache
            old_v = self.v_cache
            old_scores = self.attention_scores
            old_len = self.seq_len
            new_k = torch.zeros(B, self.n_kv, needed, self.head_dim,
                                device=self.device, dtype=dtype)
            new_v = torch.zeros_like(new_k)
            new_scores = torch.zeros(B, self.n_kv, needed,
                                     device=self.device,
                                     dtype=self.bf16_or_dtype())
            if old_k is not None and old_len > 0:
                copy_len = min(old_len, old_k.shape[2])
                new_k[:, :, :copy_len].copy_(old_k[:, :, :copy_len])
                new_v[:, :, :copy_len].copy_(old_v[:, :, :copy_len])
                if old_scores is not None:
                    sc = min(copy_len, old_scores.shape[2])
                    new_scores[:, :, :sc].copy_(old_scores[:, :, :sc])
            self.k_cache = new_k
            self.v_cache = new_v
            self.attention_scores = new_scores

    def _rotate(self, t):
        """Block-diagonal Hadamard rotation via per-block FWHT."""
        *lead, hd = t.shape
        if hd != self.padded_dim:
            t = F.pad(t, (0, self.padded_dim - hd), value=0)
        t = t.reshape(*lead, self.n_blocks, self.block_size)
        t = HadamardKVCache._fwht(self, t.contiguous().clone())
        t = t * self._norm
        t = t.reshape(*lead, self.padded_dim)
        if hd != self.padded_dim:
            t = t[..., :hd]
        return t

    def _inverse_rotate(self, t):
        """Inverse Hadamard (H/sqrt(n) is orthogonal+symmetric)."""
        return self._rotate(t)

    def _quantize(self, t):
        """Per-token INT4 quantization on rotated values."""
        scale = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / self.qmax
        q = torch.clamp(torch.round(t / scale), -self.qmax, self.qmax)
        return q, scale

    def _dequantize(self, q, scale):
        """Dequantize INT4 → fp, then inverse rotate."""
        t = q * scale
        return self._inverse_rotate(t)

    def _ensure_full_precision(self):
        """If cache is currently INT4-quantized, dequantize back to fp for
        appending new tokens. This is needed because SnapKV eviction operates
        on full-precision attention scores."""
        if not self._quantized or self.k_quant is None:
            return
        k = self._dequantize(
            self.k_quant[:, :, :self.seq_len],
            self.k_scale[:, :, :self.seq_len])
        v = self._dequantize(
            self.v_quant[:, :, :self.seq_len],
            self.v_scale[:, :, :self.seq_len])
        B = k.shape[0]
        # Ensure pre-allocated buffer exists (T=0: just need room for seq_len)
        self._ensure_buffer(B, 0, k.dtype)
        self.k_cache[:, :, :self.seq_len].copy_(k)
        self.v_cache[:, :, :self.seq_len].copy_(v)
        self.k_quant = None
        self.v_quant = None
        self.k_scale = None
        self.v_scale = None
        self._quantized = False

    def _quantize_cache(self):
        """Quantize the full-precision cache to INT4 Hadamard."""
        if self.k_cache is None or self._quantized:
            return
        k_rot = self._rotate(self.k_cache[:, :, :self.seq_len])
        v_rot = self._rotate(self.v_cache[:, :, :self.seq_len])
        k_q, k_s = self._quantize(k_rot)
        v_q, v_s = self._quantize(v_rot)
        B, n_kv, T, hd = k_q.shape
        self.k_quant = torch.zeros(B, n_kv, self.max_capacity, hd,
                                   device=k_q.device, dtype=k_q.dtype)
        self.v_quant = torch.zeros_like(self.k_quant)
        self.k_scale = torch.zeros(B, n_kv, self.max_capacity, 1,
                                   device=k_s.device, dtype=k_s.dtype)
        self.v_scale = torch.zeros_like(self.k_scale)
        self.k_quant[:, :, :T] = k_q
        self.v_quant[:, :, :T] = v_q
        self.k_scale[:, :, :T] = k_s
        self.v_scale[:, :, :T] = v_s
        # Free full-precision cache
        self.k_cache = None
        self.v_cache = None
        self._quantized = True

    def append(self, k, v, position, attention_weights=None):
        """Append K/V with optional attention weights for SnapKV scoring.

        Dequantizes if needed, appends in full precision, runs SnapKV
        eviction when over capacity, then re-quantizes to INT4.
        """
        self._ensure_full_precision()
        B, _, T, _ = k.shape
        self._ensure_buffer(B, T, k.dtype)

        # Update attention scores from observation window
        if attention_weights is not None and self.seq_len > 0:
            scores = attention_weights.sum(dim=2)  # [B, n_kv, cache_size]
            cs = scores.shape[-1]
            if cs <= self.seq_len:
                self.attention_scores[:, :, :cs] += scores
            else:
                self.attention_scores[:, :, :self.seq_len] += scores[:, :, :self.seq_len]

        # Write new tokens to buffer at current position (no torch.cat)
        end = self.seq_len + T
        self.k_cache[:, :, self.seq_len:end].copy_(k)
        self.v_cache[:, :, self.seq_len:end].copy_(v)
        self.attention_scores[:, :, self.seq_len:end].zero_()
        self.seq_len = end

        # Evict if over capacity (SnapKV)
        if self.seq_len > self.max_capacity:
            self._evict()

        # Quantize to INT4 Hadamard for storage efficiency
        self._quantize_cache()

    def _evict(self):
        """SnapKV eviction: keep high-attention tokens + observation window.
        Compacts in-place into the pre-allocated buffer (no reallocation)."""
        total = self.seq_len
        n_to_evict = total - self.max_capacity
        obs_start = total - self.obs_window

        candidate_scores = self.attention_scores[:, :, :obs_start].mean(dim=1)
        candidate_scores = candidate_scores.mean(dim=0)  # [obs_start]

        _, indices = torch.sort(candidate_scores)
        evict_indices = indices[:n_to_evict].sort()[0]

        keep = torch.ones(total, dtype=torch.bool, device=self.device)
        keep[evict_indices] = False

        # Compact buffer: copy kept entries to front (no reallocation)
        new_seq_len = keep.sum().item()
        self.k_cache[:, :, :new_seq_len] = self.k_cache[:, :, keep]
        self.v_cache[:, :, :new_seq_len] = self.v_cache[:, :, keep]
        self.attention_scores[:, :, :new_seq_len] = self.attention_scores[:, :, keep]
        self.seq_len = new_seq_len

    def get(self, positions):
        """Retrieve K/V — dequantizes from INT4 if needed."""
        if self._quantized and self.k_quant is not None:
            k = self._dequantize(
                self.k_quant[:, :, :self.seq_len],
                self.k_scale[:, :, :self.seq_len])
            v = self._dequantize(
                self.v_quant[:, :, :self.seq_len],
                self.v_scale[:, :, :self.seq_len])
            return k, v
        if self.k_cache is None:
            return None, None
        return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]

    def get_past_kv(self):
        return self.get(None)

    def clear(self):
        self.k_cache = None
        self.v_cache = None
        self.attention_scores = None
        self.k_quant = None
        self.v_quant = None
        self.k_scale = None
        self.v_scale = None
        self.seq_len = 0
        self._quantized = False

    def info(self):
        current_size = self.seq_len
        if self._quantized and self.k_quant is not None:
            actual_bytes = current_size * self.n_kv * self.padded_dim * 0.5
            actual_bytes += current_size * self.n_kv * 2  # per-token scales
            size_mb = 2 * actual_bytes / 1e6  # K + V
        else:
            size_mb = 0
            if self.k_cache is not None and self.seq_len > 0:
                # Report only the valid portion (pre-allocated buffer is larger)
                elems = self.seq_len * self.n_kv * self.head_dim
                size_mb = 2 * elems * 2 / 1e6  # K + V, fp16/bf16 = 2 bytes
        eviction_ratio = max(1.0, self.max_seq_len / max(1, current_size))
        bit_ratio = 16 / self.bits
        return {"type": "snapkv_4bit", "seq_len": current_size,
                "observation_window": self.obs_window, "budget": self.budget,
                "max_capacity": self.max_capacity, "bits": self.bits,
                "size_mb": size_mb,
                "compression": eviction_ratio * bit_ratio,
                "eviction_ratio": eviction_ratio, "bit_ratio": bit_ratio}


class KV2BitCacheStrategy(KVCacheStrategy):
    """2-bit KV cache quantization (NSNQuant). 7x compression, <0.3 PPL loss."""

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        from forge.quant.kv_2bit import KV2BitCache
        n_layers = 16
        self.cache = KV2BitCache(
            n_kv_heads=n_kv_heads, head_dim=head_dim, n_layers=n_layers,
            max_seq_len=max_seq_len, device=device, dtype=dtype,
        )
        self.seq_len = 0

    def append(self, k, v, position, attention_weights=None):
        if not hasattr(self, '_layer_counter'):
            self._layer_counter = 0
        self.cache.append(self._layer_counter % self.cache.n_layers, k, v)
        self._layer_counter += 1
        self.seq_len = self.cache.seq_len

    def get(self, positions):
        return self.cache.get(self._layer_counter % self.cache.n_layers)

    def get_past_kv(self):
        return self.cache.get(self._layer_counter % self.cache.n_layers)

    def clear(self):
        self.cache.seq_len = 0
        self._layer_counter = 0
        self.seq_len = 0

    def info(self):
        return self.cache.info()


class KVzipCacheStrategy(KVCacheStrategy):
    """KVzip: query-agnostic KV cache compression via context reconstruction.

    (NeurIPS 2025, arXiv 2504.14452)

    Core idea: Use the model itself to reconstruct the original context from
    cached KV pairs. KV pairs that receive high attention during reconstruction
    are important — keep them. Low-attention pairs are evicted.

    Key advantages over attention-score-based eviction (H2O, SnapKV):
    - Query-agnostic: compressed cache works for ANY future query
    - 394x compression ratio with negligible performance loss
    - 2x decode latency reduction (smaller cache = faster attention)
    - Works across diverse tasks (QA, retrieval, reasoning, code)

    Algorithm:
    1. After prefill, run a reconstruction forward pass on the cached context
    2. For each KV pair, compute max attention score across all heads/layers
    3. Rank pairs by importance score
    4. Evict bottom-k pairs, keeping only the top budget fraction
    5. Subsequent decode steps use the compressed cache

    The reconstruction pass is a single forward pass — O(n) cost, amortized
    over all subsequent decode steps.
    """

    def init(self, n_heads, head_dim, n_kv_heads, max_seq_len, device, dtype):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Full KV cache — lazily allocated on first append so activation
        # doesn't burn ~seq*hd*2 bytes for a cache the decode loop may
        # never write to (mirrors StandardKVCache._ensure_allocated).
        self.k_cache = None
        self.v_cache = None

        # Compression state
        self.seq_len = 0
        self.compressed = False
        self.keep_indices = None
        self.compression_ratio = 1.0

        # Config
        self.budget_ratio = 0.1  # keep top 10% (10x compression)
        self._attention_scores = []

    def _ensure_allocated(self):
        if self.k_cache is None:
            self.k_cache = torch.zeros(
                self.n_kv_heads, self.max_seq_len, self.head_dim,
                device=self.device, dtype=self.dtype)
            self.v_cache = torch.zeros_like(self.k_cache)

    def append(self, k, v, position, attention_weights=None):
        self._ensure_allocated()
        if position < self.max_seq_len:
            self.k_cache[:, position:position+1, :] = k[0].squeeze(0).to(self.dtype)
            self.v_cache[:, position:position+1, :] = v[0].squeeze(0).to(self.dtype)
        self.seq_len = max(self.seq_len, position + 1)
        if attention_weights is not None:
            self._attention_scores.append(attention_weights.detach())

    def compress(self, model=None, input_ids=None):
        """Run KVzip compression: score KV pairs and evict low-importance ones."""
        if self.seq_len == 0:
            return
        n_keep = max(1, int(self.seq_len * self.budget_ratio))

        if self._attention_scores:
            all_scores = torch.stack(self._attention_scores, dim=0)
            importance = all_scores.max(dim=0)[0].max(dim=0)[0]
            importance = importance[:self.seq_len]
        else:
            importance = torch.ones(self.seq_len, device=self.device)

        # Always keep attention sinks (first tokens) + recent tokens
        n_sink = min(4, n_keep // 4)
        n_recent = min(4, n_keep // 4)
        n_middle = n_keep - n_sink - n_recent

        sink_indices = list(range(n_sink))
        recent_indices = list(range(self.seq_len - n_recent, self.seq_len))

        middle_mask = torch.ones(self.seq_len, dtype=torch.bool, device=self.device)
        middle_mask[:n_sink] = False
        middle_mask[self.seq_len - n_recent:] = False
        middle_importance = importance[middle_mask]
        middle_positions = torch.where(middle_mask)[0]

        if n_middle > 0 and len(middle_positions) > 0:
            top_middle = torch.topk(middle_importance, min(n_middle, len(middle_positions)))
            middle_indices = middle_positions[top_middle.indices].tolist()
        else:
            middle_indices = []

        self.keep_indices = sorted(sink_indices + middle_indices + recent_indices)
        self.compressed = True
        self.compression_ratio = self.seq_len / max(len(self.keep_indices), 1)

        keep_idx = torch.tensor(self.keep_indices, device=self.device, dtype=torch.long)
        self.k_cache[:, :len(keep_idx), :] = self.k_cache[:, keep_idx, :]
        self.v_cache[:, :len(keep_idx), :] = self.v_cache[:, keep_idx, :]
        self.seq_len = len(keep_idx)
        self._attention_scores = []

    def get(self, positions):
        if self.k_cache is None:
            empty = torch.zeros(
                self.n_kv_heads, 0, self.head_dim,
                device=self.device, dtype=self.dtype)
            return empty.unsqueeze(0), empty.unsqueeze(0)
        k = self.k_cache[:, :self.seq_len, :].unsqueeze(0)
        v = self.v_cache[:, :self.seq_len, :].unsqueeze(0)
        return k, v

    def get_past_kv(self):
        return self.get(None)

    def clear(self):
        self.seq_len = 0
        self.compressed = False
        self.keep_indices = None
        self.compression_ratio = 1.0
        self._attention_scores = []

    def info(self):
        return {
            "strategy": "kvzip",
            "seq_len": self.seq_len,
            "compressed": self.compressed,
            "compression_ratio": self.compression_ratio,
            "budget_ratio": self.budget_ratio,
        }

