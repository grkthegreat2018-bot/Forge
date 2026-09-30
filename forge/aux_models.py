"""Auxiliary (non-LLM) model runner — small task nets loaded from
checkpoints for external consumers (e.g. render-side neural passes).

ForgeEngine serves language models end-to-end; this module is the
sibling entry point for *auxiliary* models: checkpoints that carry a
plain state dict plus a manifest describing a small torch module, run
through the same conventions ForgeAI uses for LLM weights:

* safetensors-first loading via ``forge.checkpoint_io.load_checkpoint``
  (pipelined pinned→async-H2D on CUDA, mmap on CPU).
* VRAM-aware device pick: ``torch.cuda.mem_get_info`` against the
  caller's budget — CPU fallback is first-class, never an error.
* Manifest-declared arch: ``{"type": "mlp", "layers": [...]}`` builds a
  ``nn.Sequential``; ``{"type": "module", "target": "pkg.mod:cls",
  "kwargs": {...}}`` imports any user module and feeds it the raw state
  dict (the escape hatch for conv/recurrent nets).

Usage::

    from forge.aux_models import load_aux_model
    runner = load_aux_model(manifest, "models/sun.safetensors",
                            vram_budget=1 << 30)
    y = runner.infer(x)          # numpy in -> numpy out
"""
from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn

_ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
         "gelu": nn.GELU, "silu": nn.SiLU}


def _act(name):
    if name in (None, "none"):
        return None
    cls = _ACTS.get(name)
    if cls is None:
        raise ValueError(f"unknown aux activation {name!r}")
    return cls()


def _build_mlp(arch):
    layers = [int(v) for v in arch["layers"]]
    hid, out = arch.get("activation", "relu"), arch.get("out_activation")
    seq = []
    linear_at = []      # module index of each Linear inside Sequential
    for li, (din, dout) in enumerate(zip(layers[:-1], layers[1:])):
        linear_at.append(len(seq))
        seq.append(nn.Linear(din, dout))
        a = _act(out if li == len(layers) - 2 else hid)
        if a is not None:
            seq.append(a)
    net = nn.Sequential(*seq)
    return net, layers[0], linear_at


def _import_target(dotted):
    mod_name, _, attr = dotted.partition(":")
    if not attr:
        mod_name, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(mod_name), attr)


class AuxModelRunner:
    """Loaded aux model + inference call. ``infer`` takes and returns
    numpy arrays on whatever device the model landed on."""

    def __init__(self, net, device, meta):
        self.net = net.eval()
        self.device = device
        self.meta = meta
        for p in self.net.parameters():
            p.requires_grad_(False)

    def infer(self, x):
        x = np.asarray(x, dtype=np.float32)
        with torch.no_grad():
            t = torch.from_numpy(x).to(self.device)
            if t.ndim == 1:
                t = t.unsqueeze(0)
                return self.net(t)[0].cpu().numpy()
            return self.net(t).cpu().numpy()

    __call__ = infer


def load_aux_model(manifest: dict, checkpoint_path: str,
                   device: str = "auto",
                   vram_budget: int | None = None) -> AuxModelRunner:
    """Build and load an auxiliary model from manifest + checkpoint.

    ``manifest`` carries ``arch`` (see module docstring) and optional
    ``vram_bytes`` / ``key_prefix`` (stripped from checkpoint tensor
    names). ``device="auto"`` picks CUDA when the manifest's weight
    estimate fits both ``vram_budget`` and *currently free* VRAM, else
    CPU — callers can force either side explicitly.
    """
    arch = manifest.get("arch", {})
    kind = arch.get("type", "mlp")

    if kind == "mlp":
        net, _, linear_at = _build_mlp(arch)
    elif kind == "module":
        cls = _import_target(arch["target"])
        net = cls(**arch.get("kwargs", {}))
        linear_at = None
    else:
        raise ValueError(f"unsupported aux arch {kind!r}")

    # Device pick: CUDA iff it is present, allowed, and the model fits.
    need = int(manifest.get("vram_bytes", 0))
    use_cuda = device == "cuda" or (
        device == "auto" and torch.cuda.is_available()
        and need <= (vram_budget if vram_budget is not None else need))
    if use_cuda and torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        use_cuda = need <= free
    dev = torch.device("cuda" if use_cuda else "cpu")

    state = _load_state(checkpoint_path, dev)
    prefix = manifest.get("key_prefix")
    if prefix:
        state = {k[len(prefix):] if k.startswith(prefix) else k: v
                 for k, v in state.items()}
    if kind == "mlp":
        # Checkpoints name layers "layer{i}"; Sequential numbers them.
        remapped = {}
        for k, v in state.items():
            nk = k
            if k.startswith("layer"):
                tag, _, rest = k.partition(".")
                li = int(tag[5:])
                nk = f"{linear_at[li]}.{rest}"
            remapped[nk] = v
        state = remapped
    missing, unexpected = net.load_state_dict(state, strict=False)
    # Shape mismatches surface here — loud, not silent (ForgeAI
    # convention: no silent regressions).
    bad = [k for k in list(missing) + list(unexpected)
           if not k.endswith("num_batches_tracked")]
    if bad:
        raise ValueError(f"checkpoint does not satisfy arch {kind!r}: "
                         f"missing/unexpected keys {bad[:4]}")
    return AuxModelRunner(net.to(dev), dev, manifest)


def _load_state(checkpoint_path, device):
    """forge.checkpoint_io.load_checkpoint on the chosen device; on CUDA
    this uses the pipelined pinned-staging loader automatically."""
    from .checkpoint_io import load_checkpoint
    state = load_checkpoint(checkpoint_path, map_location=device)
    # load_checkpoint merges meta sidecar keys into the result — drop
    # non-tensor entries before load_state_dict.
    return {k: v for k, v in state.items()
            if torch.is_tensor(v)}
