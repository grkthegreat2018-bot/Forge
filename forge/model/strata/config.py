"""STRATA configuration - block-store hyperparameters (M0 substrate)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StrataConfig:
    """Memory layout is bounded by construction: every kind gets a fixed
    slab of blocks, each block a fixed slab of cells."""

    name: str = "strata"
    vocab_size: int = 65536
    eos_token_id: int = 2

    # cell geometry (see cells.py for the bit layout)
    followers: int = 8            # M - top-M followers per context key
    slots_per_block: int = 1 << 18   # 262144 cells -> 10 MB/block @ M=8
    store_raw: bool = True        # keep raw ctx key per cell (+8B) for
                                  # labeled export/debug; off = lean mode

    # context channels -> block kinds (order-k suffix-hash tables;
    # FLUX-swept set, sparse/class/epi kinds ride the same substrate)
    orders: tuple = (3, 4, 6, 8, 12, 16, 24, 32, 48, 64)

    # capacity guards
    fill_cap: float = 0.85        # seal+roll the active block past this load
    max_blocks_per_kind: int = 8  # hot+sealed blocks kept per kind (host+dev)
    cold_blocks_per_kind: int = 16  # demoted-to-host blocks before drop

    # write policy
    surprise_gate: bool = True    # scale deltas by 1 - p_mix upstream
    surprise_floor: float = 0.05

    # undo arenas (per active block)
    undo_cap: int = 200_000       # batched undo entries kept per block

    device: str = "cpu"           # "cuda" puts hot blocks on GPU
    seed: int = 0x57A7A
