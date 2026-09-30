"""FLUX - sparse online associative-memory language model.

Not a transformer.  The "weights" are addressable memory cells that are
written locally and live: every observed token updates a handful of
cells (n-gram tables, a Hebbian topic row, episodic index entries, a
kNN context-fingerprint row) and nothing else - one change cannot
perturb unrelated regions.

Context format = a *sketch*, not a window:
  - rolling polynomial suffix-hashes at multiple orders (features over
    the whole history, order-agnostic),
  - an episodic ring buffer enabling longest-suffix-match retrieval of
    effectively unbounded order, PLUS archived entries that survive the
    ring horizon: each index entry carries its successor token and
    suffix-hash "certificates" at longer orders, so a position that has
    scrolled out of the ring still votes (graded confidence = deepest
    matching certificate).  Deep memory is therefore truly unbounded,
    not capped at ring_capacity,
  - an EMA topic hypervector,
  - a recency table.
No positional encoding, no maximum length, no training needed to handle
longer contexts - longer history just means more memory cells.

Prediction = geometric mixture over channels:
    logit(t) = log_uni(t) + sum_c  w_c * (log p_c(t) - log uni(t))
where each channel p_c is the empirical distribution of one feature
family and w_c adapts online by Hedge (multiplicative weights).

The "knn" channel is the approximate-match counterpart of the exact
episodic channel: every learned token writes one row
(normalize(context_EMA @ P_csem), successor) into a fixed-capacity
fingerprint bank; prediction is a top-k cosine retrieval whose
successors vote.  It generalizes deep recall to *similar* contexts -
the memorizing-transformer pattern, online and journaled.

``teach()`` is instant hot-training: a (context -> target) association
is written directly into the order tables (all orders), the csem
fingerprint of the context, and the knn bank - no gradients, no ring
requirement, revertible via ``revert_tag``.

Live learning is instant and journaled: every mutation is appended to
``FluxJournal`` with its source tag, so "what is stored in which
weights" is auditable and revertible (``revert_since``, ``revert_tag``,
``snapshot``/``load``).

Engine contract (forge/engine/decoding.py StandardDecoding):
    model(ids, past_key_values=None, use_cache=True) -> (logits, loss, None)
    logits: float32 [B, T, vocab]; past is always None (state is internal).

Package facade - ``from forge.model.flux import FluxLM`` keeps working;
the implementation is split across focused modules (config, helpers,
journal, tables, kernels, state/predict/learn/revert/io mixins, model).
"""

from .helpers import (_HASH_P, _MASK64, _MIX1, _MIX2,
                      _group_add, _group_totals,
                      _load_teacher_embedding, _mix64, _s64,
                      _scatter_sum)
from .config import FluxConfig
from .journal import FluxJournal
from .tables import _GPUCellTables
from .kernels import HAS_TRITON
from .model import FluxLM

__all__ = [
    "FluxLM", "FluxConfig", "FluxJournal", "HAS_TRITON",
    "_GPUCellTables", "_HASH_P", "_MASK64", "_MIX1", "_MIX2",
    "_load_teacher_embedding", "_mix64", "_s64", "_scatter_sum",
    "_group_add", "_group_totals",
]
