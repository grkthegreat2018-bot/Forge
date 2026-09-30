"""Explicit, centralized runtime configuration for ForgeAI.

Historically, several import-time global side effects (CUDA env vars, cache
dir overrides, TF32 toggles, orphaned-tmp cleanup) were scattered across
``forge/model_loader.py`` and ``forge/__init__.py``.  Critique finding F17
asked to make these side effects **explicit and centralized** so that any
entrypoint (GUI, server, tests, CLI) can opt into the same environment with a
single, discoverable call instead of relying on an import-ordering accident.

This module exposes :func:`configure`, which is **idempotent** — safe to call
multiple times.  Entrypoints (``forge_gui_server``, ``forge_server``,
``sft_train``, ``infinite_loop``) call it explicitly at startup; importing
``forge`` or ``forge.model_loader`` no longer triggers it (critique NC7).
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_CONFIGURED = False


def configure() -> None:
    """Apply ForgeAI's global runtime settings (idempotent).

    Performs the side effects that used to live at import time in
    ``forge/model_loader.py`` and ``forge/__init__.py``:

    * ``SAFETENSORS_FAST_CUDA=1`` — pinned-memory + async DMA weight loading.
    * Short Triton/Inductor cache dirs (avoids Windows MAX_PATH overflow).
    * ``torch.set_float32_matmul_precision("high")`` — TF32 matmuls.
    * ``cudnn.allow_tf32`` / ``cuda.matmul.allow_tf32`` — TF32 convs.
    * ``cleanup_orphaned_tmp()`` — remove crashed-write leftovers.

    Calling this more than once is a no-op after the first successful run.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    # --- CUDA allocator ------------------------------------------------------
    # expandable_segments cuts VRAM fragmentation — critical on the 12GB
    # card where long-running engines fragment the heap across many
    # generate/sleep cycles.  Must be set before CUDA context init, which is
    # why it lives here (entrypoint startup) rather than in ForgeEngine
    # (constructed after weights are already on GPU — too late to matter).
    # cuMemMap-backed expandable segments are unsupported on Windows
    # (torch warns "not supported on this platform" on every alloc) — the
    # native Windows allocator already handles fragmentation adequately.
    import sys
    if sys.platform != "win32":
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # --- safetensors fast CUDA loading -------------------------------------
    # Pinned host memory + async CPU->GPU copies instead of synchronous
    # per-tensor copies.  Dramatically speeds up weight loading on CUDA.
    os.environ.setdefault("SAFETENSORS_FAST_CUDA", "1")

    # --- Triton / Inductor cache dirs --------------------------------------
    # The default (%TEMP%\torchinductor_<user> with a "triton\<device>\<key>"
    # suffix) plus Triton's ~130-char fused kernel names exceeds Windows
    # MAX_PATH (260), so open() fails during torch.compile.  Project-local
    # short dirs keep paths well under the limit and persist kernels.
    try:
        from research.paths import TORCH_CACHE_DIR
        os.environ.setdefault("TRITON_CACHE_DIR", str(TORCH_CACHE_DIR / "triton"))
        os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(TORCH_CACHE_DIR))
        os.environ.setdefault("TORCHINDUCTOR_PERSISTENT_AUTOTUNE_DIR", str(TORCH_CACHE_DIR))
    except ImportError:
        # research.paths may be unavailable in minimal test environments;
        # the torch settings below are still applied.
        logger.debug("research.paths unavailable, skipping cache dir setup", exc_info=True)

    # --- torch precision / TF32 --------------------------------------------
    try:
        import torch
        # Enable TensorFloat32 tensor cores for float32 matmuls (RTX 5070
        # supports this): ~8x speedup on fp32 matmuls, ~1e-5 precision loss.
        torch.set_float32_matmul_precision("high")
        # TF32 for cuDNN convolutions (conv layers, attention padding ops).
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except ImportError:
        # torch may not be installed in some minimal environments.
        logger.debug("torch not available, skipping TF32 configuration", exc_info=True)

    # --- orphaned checkpoint cleanup ---------------------------------------
    try:
        from forge.checkpoint_io import cleanup_orphaned_tmp
        cleanup_orphaned_tmp()
    except Exception:
        logger.debug("Orphaned checkpoint cleanup failed", exc_info=True)

    _CONFIGURED = True
