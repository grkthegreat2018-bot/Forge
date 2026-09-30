"""Shared pytest fixtures for ForgeAI test suite.

All tests use GPU (CUDA) with the full ForgeEngine pipeline.
CPU fallback only when CUDA is critically unavailable.
"""
import os
import tempfile

# expandable_segments is unsupported on Windows (torch warns on every
# CUDA alloc) — set it only where the cuMemMap path exists.
if os.name != "nt":
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Triton kernel filenames can exceed Windows MAX_PATH under the default
# %TEMP%/torchinductor_<user> cache root — use a shorter cache dir.
if os.name == "nt":
    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", os.path.join(tempfile.gettempdir(), "ti"))

import pytest
import torch

from forge.config import ModelConfig, get_config

CUDA_AVAILABLE = torch.cuda.is_available()

# ── GPU-first config fixtures ──────────────────────────────────────────────

def _gpu_config(preset="forgelm_tiny", vocab=256, dtype="bfloat16"):
    """Build config on GPU with bf16. Falls back to CPU only if no CUDA."""
    cfg = get_config(preset)
    cfg.vocab_size = vocab
    cfg.dtype = dtype
    cfg.device = "cuda" if CUDA_AVAILABLE else "cpu"
    return cfg


@pytest.fixture
def tiny_config():
    """Tiny config on GPU (bf16). CPU fallback only if no CUDA."""
    return _gpu_config("forgelm_tiny")


@pytest.fixture
def tiny_config_cpu(tiny_config):
    """Alias — same as tiny_config (GPU-first now)."""
    return tiny_config


@pytest.fixture
def tiny_config_gpu():
    """GPU config (explicit). Skips if no CUDA."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    return _gpu_config("forgelm_tiny")


@pytest.fixture
def v10_config():
    """Full V10 config on GPU (bf16). For integration tests."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available — V10 requires GPU")
    return _gpu_config("forgelm_v2", vocab=65536)


@pytest.fixture
def gpu_available():
    """Skip test if CUDA is not available."""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
    return True


@pytest.fixture
def forge_engine(tiny_config_gpu):
    """Build a ForgeEngine with full feature activation on GPU.

    Uses the tiny config for fast tests. activate_optimal() enables:
      - RotorQuant KV cache, torch.compile, Triton conv, prefix cache,
        fused QK-Norm+RoPE+Cache-Write, chunked prefill, seq split, warmup.
    """
    from forge.model_loader import ConfigurableResearchLLM
    from forge.engine.forge_engine import ForgeEngine
    from research.tokenizer_cache import get_tokenizer

    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            model = ConfigurableResearchLLM(tiny_config_gpu)
    finally:
        torch.set_default_dtype(old_dtype)
    model.eval()

    tok = get_tokenizer("research/checkpoints/forgelm_v2_tokenizer")
    engine = ForgeEngine(model, tok, device="cuda")
    engine.activate_optimal()
    yield engine
    # Cleanup
    del engine
    del model
    torch.cuda.empty_cache()


@pytest.fixture
def tmp_checkpoint_dir(tmp_path):
    """Clean temporary directory for checkpoint I/O tests."""
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    return ckpt_dir


@pytest.fixture
def small_state_dict():
    """Small state dict with mixed tensors and metadata for checkpoint tests."""
    return {
        "weight_a": torch.randn(4, 8, dtype=torch.float32),
        "weight_b": torch.randn(16, dtype=torch.float32),
        "step": 100,
        "config": {"lr": 1e-4, "epochs": 10},
    }


@pytest.fixture
def bf16_state_dict():
    """State dict with bf16 tensors (common in ForgeAI)."""
    return {
        "weight_a": torch.randn(4, 8, dtype=torch.bfloat16),
        "weight_b": torch.randn(16, dtype=torch.bfloat16),
        "step": 50,
    }
