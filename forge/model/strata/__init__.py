"""STRATA - cortex-over-database LM substrate (R60).

Three separated systems:
  WEIGHTS (.strata)  - labeled CellBlocks of packed info cells.
                       Info's only job: hold facts.  Sealed/active
                       lifecycle + per-block undo arenas + tag-scoped
                       revert + bounded hot/cold budget.
  PARAMS  (.iq)      - the tiny compute machinery (IQ).  Mixer
                       weights, fingerprint table, optional dense
                       channel.  Decides HOW info is used.
  KNOBS   (.cond)    - conditioning scalars (gates, lifts, sampling).
                       Ingest never touches them, so weights can be
                       pumped with info without changing behavior.

M0 (this package): the storage substrate + artifact IO.  The model
dynamics (channels, context sketch, fused decode) build on top.
"""
from .config import StrataConfig
from .cells import (
    count_lut, quantize_counts, decode_counts, pack_follower,
    foll_tok, foll_cntq, foll_aux, foll_valid, pack_meta, meta_nfoll,
    meta_wtag, meta_flags, cell_bytes, EMPTY_FOLL,
)
from .blocks import CellBlock, BlockStore, BlockMeta
from .io import (save_container, load_container, save_store,
                 load_store, MAGIC)
from .knobs import KnobSet, DEFAULT_KNOBS
from .params import ParamSet

__all__ = [
    "StrataConfig", "CellBlock", "BlockStore", "BlockMeta",
    "KnobSet", "DEFAULT_KNOBS", "ParamSet",
    "save_container", "load_container", "save_store", "load_store",
    "count_lut", "quantize_counts", "decode_counts", "pack_follower",
    "foll_tok", "foll_cntq", "foll_aux", "foll_valid", "pack_meta",
    "meta_nfoll", "meta_wtag", "meta_flags", "cell_bytes",
    "EMPTY_FOLL", "MAGIC",
]
