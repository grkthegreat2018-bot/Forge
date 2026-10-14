"""Shared helpers for the FLUX sparse-memory LM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

_HASH_P = np.uint64(1099511628211)
_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _byte_class(t: int) -> int:
    """Coarse byte class for the g<k> channels (PAQ word-model analog):
    0 digit, 1 upper, 2 lower, 3 space/tab, 4 newline, 5 punctuation,
    6 other ASCII/control, 7 high byte or BPE id (>=128)."""
    if t >= 256 or t < 0:
        return 7
    if 48 <= t <= 57:
        return 0
    if 65 <= t <= 90:
        return 1
    if 97 <= t <= 122:
        return 2
    if t in (9, 32):
        return 3
    if t in (10, 13):
        return 4
    if t in b".,;:!?'\"-()[]{}<>/\\|@#$%^&*+=~`":
        return 5
    return 6 if t < 128 else 7




def _load_teacher_embedding(path) -> torch.Tensor:
    """Load a teacher's embedding table from a checkpoint file -
    safetensors (slice-read, no full load) or torch .pt.  The target is
    auto-detected: prefer a key containing 'embed'/'wte', else the
    largest first-dim 2-D tensor (vocab-sized rows)."""
    path = Path(path)
    if path.is_dir():
        cands = sorted(path.glob("*.safetensors"))
        if not cands:
            raise FileNotFoundError(f"no .safetensors in {path}")
        path = cands[0]
    if path.suffix == ".safetensors":
        from safetensors import safe_open
        with safe_open(str(path), framework="pt") as f:
            keys = [k for k in f.keys()
                    if "embed" in k.lower() or "wte" in k.lower()]
            if not keys:
                keys = [k for k in f.keys()
                        if len(f.get_slice(k).get_shape()) == 2]
            best = max(keys, key=lambda k: f.get_slice(k)
                       .get_shape()[0])
            return f.get_tensor(best).float()
    sd = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(sd, dict):
        raise ValueError(f"{path} is not a state dict")
    keys = [k for k in sd
            if isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2
            and ("embed" in k.lower() or "wte" in k.lower())]
    if not keys:
        keys = [k for k in sd
                if isinstance(sd[k], torch.Tensor) and sd[k].dim() == 2]
    return sd[max(keys, key=lambda k: sd[k].shape[0])].float()



def _s64(v: int) -> int:
    """int64 bit pattern (signed) for a uint64 constant."""
    return v - (1 << 64) if v >= (1 << 63) else v


_MIX1 = _s64(0x9E3779B97F4A7C15)
_MIX2 = _s64(0xBF58476D1CE4E5B9)
_MIX1U = np.uint64(0x9E3779B97F4A7C15)
_MIX2U = np.uint64(0xBF58476D1CE4E5B9)


def _mix64(x: torch.Tensor) -> torch.Tensor:
    """SplitMix64 finalizer on int64 (wraps mod 2^64 - fine, it's a
    hash).  Returns nonzero values."""
    x = (x + _MIX1) * _MIX2
    x = (x ^ (x >> 29)) * _MIX2
    x = x ^ (x >> 32)
    return torch.where(x == 0, torch.ones_like(x), x)


def _mix64_np(x: np.ndarray) -> np.ndarray:
    """Vectorized SplitMix64 on a uint64 array (bloom index hashing -
    the device path wraps int64, this one stays unsigned)."""
    x = (x + _MIX1U) * _MIX2U
    x = (x ^ (x >> np.uint64(29))) * _MIX2U
    return x ^ (x >> np.uint64(32))


def _mix64_py(x: int) -> int:
    """Scalar SplitMix64 on a python int (eager-path bloom hashing)."""
    x = ((x + 0x9E3779B97F4A7C15) * 0xBF58476D1CE4E5B9) \
        & 0xFFFFFFFFFFFFFFFF
    x = ((x ^ (x >> 29)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    return x ^ (x >> 32)


def _scatter_sum(keys: torch.Tensor, vals: torch.Tensor,
                 n: int) -> torch.Tensor:
    """Deterministic ``zeros(n).index_add_(0, keys, vals)``.

    ``index_add_`` on CUDA accumulates via float atomics - arrival
    order (and therefore the rounded result) varies with kernel
    timing, which made logits non-reproducible when the caller's
    enqueue pattern changed (batched ``_fused_seq`` vs synced
    ``_fused_step``).  A sorted O(k^2) equality matmul gives each
    element its group total; duplicate keys then scatter identical
    values, so store order cannot matter.  All shapes are static -
    CUDA-graph capture safe (``unique_consecutive`` would force a
    data-dependent D2H sync)."""
    out = torch.zeros(n, dtype=vals.dtype, device=vals.device)
    k = keys.numel()
    if k == 0:
        return out
    srt, order = torch.sort(keys)
    v = vals[order]
    same = (srt[:, None] == srt[None, :]).to(v.dtype)
    gsum = same @ v                          # per-element group total
    # scatter_ (store) not index_put_: duplicate keys write identical
    # values so the store race is benign, and inductor does not turn
    # it into an accumulate (index_put_ miscompiled under compile).
    out.scatter_(0, srt, gsum)
    return out


def _group_totals(keys: torch.Tensor,
                  vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(sorted_unique_keys, per_key_sums)`` - deterministic grouped sum.

    Companion to ``_scatter_sum`` for the k>>1 case where the O(k^2)
    equality matmul is too big (bulk-ingest chunks are 8k tokens):
    sort keys, cumsum the sorted vals, and difference at group
    boundaries.  Sort and cumsum have fixed reduction order, so equal
    (keys, vals) inputs always produce identical totals - float atomic
    arrival order never enters.
    """
    srt, order = _group_sort(keys)
    return _group_apply(srt, order, keys, vals)


def _group_sort(keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The sort half of _group_totals - hoistable when several value
    tensors share the same keys (the EMA loop groups 4 tensors over
    the same token ids)."""
    return torch.sort(keys)


def _group_apply(srt: torch.Tensor, order: torch.Tensor,
                 keys: torch.Tensor,
                 vals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-sum ``vals`` under a presorted ``keys`` ordering."""
    if keys.numel() == 0:
        z = torch.zeros((0,) + vals.shape[1:], dtype=vals.dtype,
                        device=vals.device)
        return keys.clone(), z
    v = vals[order]
    k = keys.numel()
    head = torch.ones(k, dtype=torch.bool, device=keys.device)
    head[1:] = srt[1:] != srt[:-1]
    starts = head.nonzero().squeeze(1)
    ends = torch.cat([starts[1:], starts.new_full((1,), k)])
    cum0 = torch.cat([torch.zeros_like(v[:1]), torch.cumsum(v, dim=0)])
    return srt[head], cum0[ends] - cum0[starts]


def _group_prefix(keys: torch.Tensor, vals: torch.Tensor,
                  keys2: torch.Tensor | None = None
                  ) -> torch.Tensor:
    """Exclusive in-group prefix sum of ``vals``, returned in ORIGINAL
    position order.  Groups are equal ``keys`` (or ``(keys, keys2)``
    pairs); within a group, cumsum runs in ascending position order.
    Deterministic - stable sort + cumsum have fixed reduction order.

    Bulk-ingest use: position i's prefix value = the mass earlier
    positions in the same block already wrote under the same key -
    the in-block evidence the eager path sees live.
    """
    n = keys.numel()
    if n == 0:
        return torch.zeros_like(vals)
    if keys2 is None:
        order = torch.argsort(keys, stable=True)
        k1s = keys[order]
        same = k1s[1:] == k1s[:-1]
    else:
        o1 = torch.argsort(keys2, stable=True)
        order = o1[torch.argsort(keys[o1], stable=True)]
        k1s = keys[order]
        k2s = keys2[order]
        same = (k1s[1:] == k1s[:-1]) & (k2s[1:] == k2s[:-1])
    head = torch.ones(n, dtype=torch.bool, device=keys.device)
    head[1:] = ~same
    # exclusive in-group prefix = global exclusive prefix minus the
    # prefix value at the group's first element
    v = vals[order]
    cum = torch.cumsum(v, dim=0)
    exg = cum - v
    gid = torch.cumsum(head.long(), dim=0) - 1
    starts = head.nonzero().squeeze(1)
    excl = exg - exg.index_select(0, starts.index_select(0, gid))
    out = torch.empty_like(excl)
    out[order] = excl
    return out


def _group_add(dst: torch.Tensor, keys: torch.Tensor,
               vals: torch.Tensor) -> None:
    """Deterministic ``dst.index_add_(0, keys, vals)`` (see _scatter_sum).

    Groups duplicate keys via _group_totals, then writes each unique
    key's row exactly once with index_put_ - no float atomics, so the
    result cannot depend on kernel scheduling."""
    if keys.numel() == 0:
        return
    u, tot = _group_totals(keys, vals)
    dst.index_put_((u,), dst.index_select(0, u) + tot)

