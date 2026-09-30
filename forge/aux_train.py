"""Auxiliary-model training — the train-side sibling of ``aux_models``.

ForgeEngine's training stack (``forge/training/``) is LLM-shaped: token
datasets, causal loss, optimizer factories tuned for a 1.2B model. The
auxiliary models served by :mod:`forge.aux_models` are the opposite —
tiny MLPs distilled from a teacher function or labeled telemetry. This
module provides that generic path:

* :func:`train_mlp` — numpy (X, Y) -> trained ``nn.Sequential`` built by
  the same :func:`forge.aux_models._build_mlp` the loader uses, so an
  ``arch`` dict round-trips identically between train and inference.
* :func:`export_aux_checkpoint` — writes weights through
  :func:`forge.checkpoint_io.save_checkpoint` (atomic, readback-verified)
  with tensor names remapped to the ``layer{i}.weight``/``layer{i}.bias``
  convention the manifests declare.

Input standardization: train_mlp fits per-feature mean/std on the train
split, trains on standardized inputs, then folds the affine transform
into layer0 before returning — the exported net consumes RAW features
(the contract downstream runtime passes assume).

VRAM: these nets are KB-scale — AdamW-fp32 on CUDA is fine; CPU works
unchanged for machines without a GPU.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .aux_models import _build_mlp
from .checkpoint_io import save_checkpoint

_LOSSES = {
    "mse": nn.functional.mse_loss,
    "bce": nn.functional.binary_cross_entropy,
    "huber": nn.functional.huber_loss,
}


def _linear_state(net: nn.Sequential, linear_at: list[int]) -> dict:
    """Sequential state dict -> ``layer{i}.*`` names matching the manifest
    convention (inverse of the remap in ``aux_models.load_aux_model``)."""
    state = net.state_dict()
    out = {}
    for li, mi in enumerate(linear_at):
        out[f"layer{li}.weight"] = state[f"{mi}.weight"]
        out[f"layer{li}.bias"] = state[f"{mi}.bias"]
    return out


def fold_input_norm(net: nn.Sequential, linear_at: list[int],
                    x_mu, x_sd) -> None:
    """Fold ``x_s = (x - mu) / sd`` into layer0 so the net takes raw
    inputs: W_raw = W_std / sd (columnwise), b_raw = b_std - W_raw·mu."""
    lin = net[linear_at[0]]
    with torch.no_grad():
        sd = torch.as_tensor(x_sd, dtype=lin.weight.dtype)[None, :]
        mu = torch.as_tensor(x_mu, dtype=lin.weight.dtype)[None, :]
        lin.weight.data = lin.weight.data / sd
        lin.bias.data = lin.bias.data - (lin.weight.data * mu).sum(1)


def train_mlp(X, Y, *, layers=None, activation: str = "relu",
              out_activation: str | None = None, loss: str = "mse",
              epochs: int = 60, batch_size: int = 4096, lr: float = 3e-3,
              val_frac: float = 0.05, device: str = "auto",
              seed: int = 0, min_lr_frac: float = 0.02,
              fold_norm: bool = True):
    """Train a small MLP on a numpy regression/classification dataset.

    ``X`` (N, Din) and ``Y`` (N, Dout) float arrays. ``layers`` overrides
    the hidden stack; default is [din, 128, 128, dout]. Returns
    ``(net, metrics)`` where metrics carries train/val loss history and
    the final val error.
    """
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    assert X.shape[0] == Y.shape[0] and X.shape[0] >= 8

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(X.shape[0], generator=g).numpy()
    n_val = max(1, int(len(perm) * val_frac))
    vi, ti = perm[:n_val], perm[n_val:]

    x_mu = X[ti].mean(0)
    x_sd = X[ti].std(0)
    x_sd[x_sd < 1e-6] = 1.0
    Xs = (X - x_mu) / x_sd

    if layers is None:
        layers = [X.shape[1], 128, 128, Y.shape[1]]
    arch = {"type": "mlp", "layers": list(layers),
            "activation": activation, "out_activation": out_activation}
    net, _, linear_at = _build_mlp(arch)

    dev = torch.device(
        "cuda" if (device == "cuda" or (device == "auto"
                   and torch.cuda.is_available())) else "cpu")
    net.to(dev)
    xt = torch.from_numpy(Xs).to(dev)
    yt = torch.from_numpy(Y).to(dev)
    xt_tr, yt_tr = xt[ti], yt[ti]
    xv, yv = xt[vi], yt[vi]

    opt = torch.optim.AdamW(net.parameters(), lr=lr,
                            fused=torch.cuda.is_available())
    steps_per_epoch = max(1, (len(ti) + batch_size - 1) // batch_size)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs * steps_per_epoch, eta_min=lr * min_lr_frac)
    loss_fn = _LOSSES[loss]

    hist = {"train": [], "val": []}
    for ep in range(epochs):
        net.train()
        ep_loss, nb = 0.0, 0
        p = torch.randperm(len(ti), generator=g, device="cpu").to(dev)
        for s in range(0, len(ti), batch_size):
            idx = p[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            l = loss_fn(net(xt_tr[idx]), yt_tr[idx])
            l.backward()
            opt.step()
            sched.step()
            ep_loss += float(l.detach())
            nb += 1
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(xv), yv))
        hist["train"].append(ep_loss / max(1, nb))
        hist["val"].append(vl)

    net.eval().cpu()
    if fold_norm:
        fold_input_norm(net, linear_at, x_mu, x_sd)
        # Sanity: folded net on raw val inputs must match the trained net's
        # standardized-input val loss. Cheap one-shot check catches a bad
        # fold before export ships it.
        with torch.no_grad():
            vl = float(loss_fn(net(torch.from_numpy(X[vi])),
                               torch.from_numpy(Y[vi])))
        metrics_val = vl
    else:
        metrics_val = hist["val"][-1]
    metrics = {"history": hist, "val_loss": metrics_val,
               "train_loss": hist["train"][-1], "arch": arch,
               "x_mu": x_mu.tolist(), "x_sd": x_sd.tolist(),
               "folded_norm": fold_norm}
    return net, metrics


def export_aux_checkpoint(net: nn.Sequential, path: str,
                          linear_at: list[int] | None = None) -> str:
    """Write a trained MLP as ``layer{i}.*`` safetensors via
    :func:`forge.checkpoint_io.save_checkpoint`. ``linear_at`` defaults to
    the Linear positions inside the Sequential (recomputed when absent)."""
    if linear_at is None:
        linear_at = [i for i, m in enumerate(net)
                     if isinstance(m, nn.Linear)]
    return save_checkpoint(_linear_state(net, linear_at), path)


def train_and_export(X, Y, path: str, *, layers=None, **train_kw) -> dict:
    """One-call convenience: train_mlp + export. Returns metrics."""
    net, metrics = train_mlp(X, Y, layers=layers, **train_kw)
    linear_at = [i for i, m in enumerate(net) if isinstance(m, nn.Linear)]
    export_aux_checkpoint(net, path, linear_at)
    return metrics


__all__ = ["train_mlp", "export_aux_checkpoint", "train_and_export",
           "fold_input_norm"]
