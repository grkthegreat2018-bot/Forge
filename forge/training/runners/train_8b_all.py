"""Full training run on ALL processed datasets with V10 model.

Datasets:
  1. research/data/v7_train/  — 16.5M tokens (SFT + pretrain mix, 7,660 seqs)
  2. research/data/           — 100M tokens (original LFM pretrain, ~48,800 seqs)

Model:     forgelm_v2 (d=4096, L=32, NLRQ rank=1024, ~2.8B params, 8 GB)
Optimizer: BAdam (block-wise, 1 layer at a time, fp32 states, fits 12 GB)

Features:
  - NLRQ factor training (STE masters, on by default) — trains U/V factors,
    not just the singular values S. Checkpoints stay pure-INT8 format.
  - Chunked fused linear-CE (no [B*T, V] logits materialization) — saves
    ~0.8 GB VRAM at seq 2048 / vocab 65536.
  - Resume: --resume CKPT continues step count, LR schedule, BAdam schedule,
    RNG state, and active-block optimizer states.
  - Stratified epoch sampling (--sampling stratified --v7-weight W) or plain
    shuffled epochs (default). No replacement within an epoch.
  - Best-checkpoint tracking by val loss + retention (--keep-checkpoints N).
  - VRAM preflight gate: refuses to start a run that would spill into Windows
    shared GPU memory (18x slowdown) unless --allow-spill.
  - Serial background checkpoint writer (one 8 GB file at a time).

Usage:
    python -m research.training.runners.train_8b_all --steps 500 --seq-len 2048
    python -m research.training.runners.train_8b_all --resume research/checkpoints/ForgeLM_V10_step300.safetensors
    python -m research.training.runners.train_8b_all --eval-only --checkpoint research/checkpoints/ForgeLM_V10_final.safetensors
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import queue
import random
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

if os.name != "nt":  # expandable_segments unsupported on Windows
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
V7_DATA_DIR = ROOT / "research" / "data" / "v7_train"
LFM_DATA_DIR = ROOT / "research" / "data"
FINEWEB_EDU_DIR = ROOT / "research" / "data" / "fineweb_edu"
CKPT_DIR = ROOT / "research" / "checkpoints"
LOG_DIR = ROOT / "research" / "results"

DATASET_DIRS: dict[str, Path] = {
    "v7": V7_DATA_DIR,
    "lfm": LFM_DATA_DIR,
    "fineweb_edu": FINEWEB_EDU_DIR,
}

VRAM_LOG_INTERVAL = 50  # steps between VRAM snapshots
FLCE_CHUNK = 256        # tokens per chunked-CE matmul (67 MB fp32 at vocab 65536)
EMA_BETA = 0.92         # per-step loss EMA for display


# ────────────────────────── logging / reporting ──────────────────────────

def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def log(msg: str) -> None:
    print(f"  {msg}", flush=True)


def vram_snapshot(label: str = "") -> dict:
    if not torch.cuda.is_available():
        return {}
    alloc = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    free, _total = torch.cuda.mem_get_info()
    log(f"VRAM[{label}]: {alloc:.2f} GB alloc, {reserved:.2f} GB reserved, "
        f"{peak:.2f} GB peak, {free / 1e9:.2f} GB free")
    return {"alloc_gb": alloc, "reserved_gb": reserved, "peak_gb": peak, "free_gb": free / 1e9}


# ─────────────────────────────── data ────────────────────────────────────

@dataclass
class PackedDataset:
    name: str
    train: torch.Tensor
    val: torch.Tensor

    @property
    def n_tokens(self) -> int:
        return self.train.numel() + self.val.numel()


def load_packed_data(path: Path, seq_len: int, name: str,
                     vocab_size: int | None = None) -> torch.Tensor:
    """Load a packed int32 token stream as an (N, seq_len) CPU tensor.

    vocab_size: if given, validate that every token id fits the model vocab.
    Mismatched packs (e.g. a 151k-vocab LFM corpus against the 65536 lfm25
    vocab) cause out-of-bounds embedding gathers, which surface as confusing
    downstream CUDA errors (gather assert → poisoned context → CUBLAS
    failures) — fail fast with the real reason instead.
    """
    if not path.exists():
        raise FileNotFoundError(f"[{name}] missing packed data file: {path}")
    arr = np.fromfile(str(path), dtype=np.int32)
    if vocab_size is not None:
        max_id = int(arr.max())
        if max_id >= vocab_size:
            raise ValueError(
                f"[{name}] {path.name}: token id {max_id} >= vocab_size "
                f"{vocab_size} — this pack was built for a different tokenizer. "
                f"Re-pack it with the model's tokenizer or use "
                f"--skip-incompatible / a different --datasets selection.")
    n_seqs = arr.shape[0] // seq_len
    if n_seqs == 0:
        raise ValueError(f"[{name}] {path.name} too small for seq_len={seq_len}")
    tensor = torch.from_numpy(arr[: n_seqs * seq_len].reshape(n_seqs, seq_len)).long()
    log(f"[{name}] {path.name}: {tuple(tensor.shape)} "
        f"({n_seqs} seqs, {n_seqs * seq_len / 1e6:.1f}M tokens)")
    return tensor


def load_datasets(selection: str, seq_len: int, vocab_size: int | None = None,
                  skip_incompatible: bool = False) -> list[PackedDataset]:
    names = list(DATASET_DIRS) if selection == "all" else [selection]
    loaded = []
    for name in names:
        try:
            loaded.append(PackedDataset(
                name=name,
                train=load_packed_data(DATASET_DIRS[name] / "train.bin", seq_len,
                                       f"{name}_train", vocab_size),
                val=load_packed_data(DATASET_DIRS[name] / "val.bin", seq_len,
                                     f"{name}_val", vocab_size),
            ))
        except ValueError as exc:
            if not skip_incompatible:
                raise
            log(f"SKIP incompatible dataset: {exc}")
    if not loaded:
        raise ValueError("no usable datasets remain after compatibility check")
    return loaded


class EpochBatchSampler:
    """Shuffled, without-replacement batch stream over the concatenated train
    sets. Rebuilds (reshuffles) at each epoch boundary.

    weights=None → plain epoch (each seq seen exactly once per epoch).
    weights={"v7": 3.0} → stratified epochs: each dataset's token share per
    epoch is proportional to its weight (v7 seqs repeat up to 3x more often
    than uniform), still without replacement inside an epoch.
    """

    def __init__(self, datasets: list[PackedDataset], batch_size: int,
                 weights: dict[str, float] | None, seed: int):
        self.sizes = [d.train.shape[0] for d in datasets]
        self.names = [d.name for d in datasets]
        self.total = sum(self.sizes)
        self.offsets = [0]
        for s in self.sizes:
            self.offsets.append(self.offsets[-1] + s)
        self.batch_size = batch_size
        self.weights = weights
        self.g = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self._pool = torch.empty(0, dtype=torch.long)
        self._pos = 0

    def _rebuild(self) -> None:
        if not self.weights:
            self._pool = torch.randperm(self.total, generator=self.g)
        else:
            w = np.array([self.weights.get(n, 1.0) for n in self.names], dtype=np.float64)
            tokens = np.array(self.sizes, dtype=np.float64)
            share = w * tokens / (w * tokens).sum()          # epoch token share
            target = np.round(share * self.total).astype(int)  # seqs per epoch
            chunks = []
            for i, (n, t) in enumerate(zip(self.sizes, target)):
                reps = max(1, math.ceil(t / max(n, 1)))
                idx = torch.randperm(n, generator=self.g).repeat(reps)[:t]
                chunks.append(idx + self.offsets[i])
            self._pool = torch.cat(chunks)[torch.randperm(sum(c.numel() for c in chunks),
                                                          generator=self.g)]
        self._pos = 0
        self.epoch += 1

    def next(self) -> torch.Tensor:
        if self._pos + self.batch_size > self._pool.numel():
            self._rebuild()
        idx = self._pool[self._pos:self._pos + self.batch_size]
        self._pos += self.batch_size
        return idx


# ─────────────────────────────── model ───────────────────────────────────

def materialize_meta_(model: torch.nn.Module, device: torch.device,
                      dtype: torch.dtype) -> None:
    """Replace meta-device params/buffers with real tensors on `device` (in place)."""
    for module in model.modules():  # includes the root module — one pass suffices
        for name, param in list(module._parameters.items()):
            if param is not None and param.is_meta:
                module._parameters[name] = torch.nn.Parameter(
                    torch.empty(param.shape, dtype=dtype, device=device),
                    requires_grad=param.requires_grad,
                )
        for name, buf in list(module._buffers.items()):
            if buf is not None and buf.is_meta:
                module._buffers[name] = torch.empty(
                    buf.shape, dtype=buf.dtype, device=device)


def reset_nlrq_layers_(model: torch.nn.Module) -> None:
    """Meta-device construction skips reset_parameters(); redo it on real tensors."""
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    for module in model.modules():
        if isinstance(module, NLRQLinear):
            module.reset_parameters()


def enable_factor_training_all_(model: torch.nn.Module) -> int:
    """Enable STE factor training on every NLRQ layer. Returns layer count."""
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    n = 0
    for module in model.modules():
        if isinstance(module, NLRQLinear):
            module.enable_factor_training_()
            n += 1
    return n


def export_nlrq_(model: torch.nn.Module) -> None:
    """Refresh INT8 buffers from STE masters (call before snapshotting state)."""
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    for module in model.modules():
        if isinstance(module, NLRQLinear):
            module.export_quantized_()


def disable_bitnet_qat_(model: torch.nn.Module) -> None:
    """Ternary QAT on random weights wastes 2+ GB of compute for no benefit.

    Train in full bf16; QAT can be re-enabled later for inference.
    """
    from forge.keys.quantization.bitnet_b158_key import BitNetLinear
    for module in model.modules():
        if isinstance(module, BitNetLinear):
            module.quantize = False
            module.force_quant = False
            if module.qscale is not None:
                with torch.no_grad():
                    module.qscale.data = module.weight.abs().mean().clamp(min=1e-8) / 0.7


def initialize_weights_(model: torch.nn.Module, n_layers: int) -> None:
    """Conservative init for stable training from scratch.

    S / gates / sinks are deliberately untouched: they are either already
    initialized by their module's reset_parameters, or should keep the module
    default (gates=0, sinks=0).
    """
    depth_scale = 0.5 ** (1.0 / max(n_layers, 1))  # scale down for deep stacks
    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.ndim >= 2 and "weight" in name:
                torch.nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
                param.mul_(depth_scale)
            elif "bias" in name:
                torch.nn.init.zeros_(param)
            elif param.ndim == 1 and ("norm" in name or "ln" in name):
                torch.nn.init.ones_(param)


def normalize_logit_scale_(model, device: torch.device, cfg) -> float:
    """Rescale the LM head path so from-scratch logits start at std ~1.

    Kaiming init on the two-stage factorized head compounds: project (std
    sqrt(2/rank)) × embed (std sqrt(2/rank)) gives logit std ≈ 5.4, so
    CE at init ≈ ln(V) + σ²/2 ≈ 24 — the model starts CONFIDENTLY WRONG
    (worse than uniform's 11.09) and burns its whole budget unlearning the
    scale. One probe forward, then scale the head path by 1/σ. Scaling
    `project.weight` also shrinks the input embedding path, but the first
    RMSNorm absorbs that — training dynamics are unaffected.

    Returns the applied scale factor.
    """
    model.eval()
    vocab = getattr(cfg, "vocab_size", 100)
    ids = torch.randint(0, vocab, (1, 16), device=device)
    with torch.no_grad(), autocast_ctx(device):
        logits = forward_model(model, ids)
    std = float(logits.float().std())
    if std < 1e-3 or not math.isfinite(std):
        return 1.0
    scale = 1.0 / std

    embed = getattr(model, "embed", None)
    target = None
    if embed is not None and hasattr(embed, "project"):
        target = embed.project.weight  # FactorizedEmbedding (head is tied)
    elif getattr(model, "head", None) is not None and hasattr(model.head, "weight") \
            and not hasattr(model.head, "embed_ref"):
        target = model.head.weight
    if target is not None:
        with torch.no_grad():
            target.mul_(scale)
    model.train()
    return scale


def enable_checkpointing_(model: torch.nn.Module) -> None:
    if hasattr(model, "use_grad_checkpoint"):
        model.use_grad_checkpoint = True
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()


def build_model(config_name: str, device: torch.device, dtype: torch.dtype,
                use_checkpointing: bool, grad_clip: float):
    """Build on the meta device, materialize to GPU, then initialize weights."""
    from forge.config import get_config
    from forge.model_loader import ConfigurableResearchLLM

    cfg = get_config(config_name)
    cfg.device = "meta"
    cfg.dtype = "bfloat16"
    cfg.use_gradient_checkpointing = use_checkpointing
    # NOTE: the model only implements "all" | "ffn" | "attn" | "none".
    # "optimal" silently maps to "none" (full activation materialization —
    # seq 2048 on 8B-B OOMs). The sft_train optimal-checkpoint PLANNER is a
    # separate mechanism; here we want plain full-block checkpointing.
    cfg.selective_gradient_checkpointing = "all"
    cfg.grad_clip = grad_clip

    with torch.device("meta"):
        model = ConfigurableResearchLLM(cfg)
    materialize_meta_(model, device, dtype)
    cfg.device = str(device)  # reflect the real device post-materialization

    reset_nlrq_layers_(model)
    disable_bitnet_qat_(model)
    initialize_weights_(model, cfg.n_layers)
    scale = normalize_logit_scale_(model, device, cfg)
    if scale != 1.0:
        log(f"Init: logit scale normalized by {scale:.3f} "
            f"(init CE ≈ ln(vocab) = {math.log(cfg.vocab_size):.2f})")
    if use_checkpointing:
        enable_checkpointing_(model)

    model.train()
    return model, cfg


def print_model_stats(model: torch.nn.Module, cfg) -> None:
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    storage = sum(t.numel() * t.element_size()
                  for t in list(model.buffers()) + list(model.parameters()))
    log(f"Params:     {n_params / 1e6:.1f}M")
    log(f"Trainable:  {n_trainable / 1e6:.1f}M")
    log(f"Storage:    {storage / 1e9:.2f} GB")
    log(f"NLRQ rank:  {cfg.nlrq_rank}")


# ────────────────────────── training primitives ──────────────────────────

def autocast_ctx(device: torch.device):
    dev_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    if dev_type != "cuda" or torch.cuda.device_count() == 0:
        # CPU/meta/no-GPU: no autocast needed (fp32 training). Use nullcontext
        # to avoid a torch bug where autocast(cpu) still probes CUDA on some builds.
        from contextlib import nullcontext
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True)


def forward_model(model, input_ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids)
    return out[0] if isinstance(out, tuple) else out


def unwrap_compiled(model):
    """Get the underlying module from torch.compile's OptimizedModule."""
    return getattr(model, "_orig_mod", model)


def next_token_cross_entropy(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """Standard next-token CE over shifted logits/labels."""
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
    )


def chunked_next_token_ce(model, input_ids: torch.Tensor,
                          chunk: int = FLCE_CHUNK) -> torch.Tensor:
    """Next-token CE without materializing full [B*T, V] logits.

    Uses the model's pre-head hidden states (return_hidden=True) and computes
    logits per token chunk in fp32. Numerically identical to
    next_token_cross_entropy (sum/n reduction), saves ~0.8 GB at seq 2048.
    Handles both plain Linear heads and FactorizedLMHead (two-stage matmul).
    """
    _, _, hidden = model(input_ids, return_hidden=True)
    head = unwrap_compiled(model).head
    x = hidden[:, :-1, :].reshape(-1, hidden.size(-1))
    target = input_ids[:, 1:].reshape(-1)
    n = target.numel()

    if hasattr(head, "embed_ref"):  # FactorizedLMHead: logits = (x@P)@E.T
        hidden_in = x @ head.embed_ref.project.weight   # (n, rank)
        proj = head.embed_ref.embed.weight              # (vocab, rank)
    else:                            # nn.Linear head: logits = x@W.T
        hidden_in = x
        proj = head.weight

    total = torch.zeros((), device=x.device, dtype=torch.float32)
    bias = getattr(head, "bias", None)
    for start in range(0, n, chunk):
        logits = (hidden_in[start:start + chunk] @ proj.t()).float()
        if bias is not None:
            logits = logits + bias.float()
        total = total + F.cross_entropy(logits, target[start:start + chunk],
                                        reduction="sum")
    return total / n


def compute_loss(model, input_ids: torch.Tensor, use_flce: bool) -> torch.Tensor:
    if use_flce:
        return chunked_next_token_ce(model, input_ids)
    return next_token_cross_entropy(forward_model(model, input_ids), input_ids)


# ────────────────────────── LR schedules ─────────────────────────────────

def lr_at(step: int, base_lr: float, warmup: int, total_steps: int,
          schedule: str = "linear", decay_frac: float = 0.3) -> float:
    """LR for 1-indexed `step`.

    linear:  warmup → linear decay-to-zero (D2Z, arXiv:2502.15938 — saves
             ~60% compute vs cosine-to-10% at matched loss).
    wsd:     warmup → stable → linear decay over the final `decay_frac`.
    cosine:  warmup → cosine decay-to-zero.
    """
    if step <= warmup:
        return base_lr * step / max(warmup, 1)
    prog = (step - warmup) / max(total_steps - warmup, 1)
    if schedule == "wsd":
        decay_start = 1.0 - decay_frac
        if prog < decay_start:
            return base_lr
        return max(0.0, base_lr * (1.0 - (prog - decay_start) / max(decay_frac, 1e-6)))
    if schedule == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * prog))
    return base_lr * max(0.0, 1.0 - prog)  # linear D2Z


# ─────────────────────────── evaluation ──────────────────────────────────

@torch.inference_mode()
def evaluate(model, val_data: torch.Tensor, device: torch.device,
             batch_size: int, max_batches: int, use_flce: bool = True) -> dict:
    """Mean val loss over the leading val batches (deterministic)."""
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(val_data), batch_size):
        if count >= max_batches:
            break
        input_ids = val_data[start:start + batch_size].to(device)
        with autocast_ctx(device):
            loss = compute_loss(model, input_ids, use_flce)
        total += loss.item()
        count += 1
    model.train()
    val_loss = total / max(count, 1)
    return {"val_loss": val_loss, "ppl": math.exp(min(val_loss, 20.0))}


# ─────────────────────────── checkpointing ───────────────────────────────

def snapshot_state(model, step: int, extra: dict | None = None) -> dict:
    """CPU copy of the state dict — safe to hand to a background thread.

    NLRQ masters are exported to their INT8 buffers first and stripped, so
    saved checkpoints stay in the pure-INT8 inference format (loadable by
    ForgeEngine without any training-mode flags).
    """
    m = unwrap_compiled(model)
    export_nlrq_(m)
    state = {k: v.detach().cpu().contiguous()
             for k, v in m.state_dict().items()
             if not (k.endswith(".U_m") or k.endswith(".V_m"))}
    state["step"] = step
    if extra:
        state.update(extra)
    return state


def state_size_gb(state: dict) -> float:
    return sum(v.numel() * v.element_size() for v in state.values()
               if hasattr(v, "numel")) / 1e9


def save_state(state: dict, path: Path) -> None:
    from forge.checkpoint_io import save_checkpoint
    save_checkpoint(state, str(path))


class CheckpointWriter(threading.Thread):
    """Serial background writer — one 8 GB file at a time, GPU keeps training.

    An idle GPU downclocks, and the first batches after a blocking save run at
    lower clocks → a persistent ~4x slowdown until clocks ramp back up.
    Serializing writes also avoids two 8 GB files competing for NVMe.
    Handles retention: keeps only the newest `keep` step checkpoints.
    """

    def __init__(self, keep: int, ckpt_dir: Path, prefix: str):
        super().__init__(daemon=True)
        self.keep = keep
        self.ckpt_dir = ckpt_dir
        self.prefix = prefix
        self.q: queue.Queue = queue.Queue()

    def submit(self, state: dict, path: Path, is_step_ckpt: bool = False) -> None:
        self.q.put((state, path, is_step_ckpt))

    def close(self) -> None:
        self.q.put(None)
        self.join()

    def run(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                break
            state, path, is_step_ckpt = item
            try:
                save_state(state, path)
                log(f"Saved {path.name} ({state_size_gb(state):.1f} GB)")
                if is_step_ckpt:
                    self._rotate()
            except Exception as exc:
                log(f"Save FAILED ({path.name}): {exc}")

    def _rotate(self) -> None:
        if self.keep <= 0:
            return
        pattern = str(self.ckpt_dir / f"{self.prefix}_step*.safetensors")
        def step_of(p: str) -> int:
            m = re.search(r"_step(\d+)\.safetensors$", p)
            return int(m.group(1)) if m else -1
        ckpts = sorted(glob.glob(pattern), key=step_of)
        for old in ckpts[: max(len(ckpts) - self.keep, 0)]:
            try:
                os.remove(old)
                log(f"Rotated out {Path(old).name}")
            except OSError:
                pass


# ─────────────────────── resume bundle ───────────────────────────────────

def slim_badam_state(optimizer) -> dict:
    """BAdam state dict with only the ACTIVE block's optimizer states.

    Full state for every visited block would be ~2x model size (fp32 m+v for
    all params) — unacceptable as a resume file. Moment estimates for
    previously-visited blocks are rebuilt within ~100 steps of resume; block
    scheduling position is preserved exactly.

    The param_groups index list is kept FULL so torch's load_state_dict
    group-size validation passes; step() creates missing states on demand.
    """
    raw = optimizer.state_dict()
    inner = raw["optimizer_state"]
    # Next block (will be stepped on resume) + the block that was just
    # updated (its states are parked on CPU and otherwise unrecoverable).
    keep_blocks = {optimizer._block_idx,
                  (optimizer._block_idx - 1) % optimizer._n_blocks}
    keep_ids = {id(p) for i in keep_blocks
                for p in optimizer._blocks[i]["params"]}
    keep_positions = {i for i, p in enumerate(optimizer.param_groups[0]["params"])
                      if id(p) in keep_ids}
    slim_state = {}
    for key, st in inner["state"].items():
        if int(key) in keep_positions:
            slim_state[str(key)] = {k: (v.cpu() if torch.is_tensor(v) else v)
                                    for k, v in st.items()}
    return {
        "optimizer_state": {"state": slim_state,
                            "param_groups": [dict(inner["param_groups"][0])]},
        "step_count": raw["step_count"],
        "block_idx": raw["block_idx"],
        "steps_in_block": raw["steps_in_block"],
    }


def capture_rng() -> dict:
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(rng: dict) -> None:
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"].cpu())
    if rng.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
        except RuntimeError:
            pass
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("python") is not None:
        random.setstate(rng["python"])


# ─────────────────────── VRAM preflight ──────────────────────────────────

def freeze_dead_params_(model, device: torch.device, use_flce: bool) -> int:
    """Freeze params that never receive gradients.

    Models carry architecture-conditional modules (MTP draft head, loop_block,
    gated-off AttnRes/MHC paths, unused heads). They can never train, but
    BAdam still partitions them — when a block contains ONLY dead params,
    every other block is frozen, the loss has no grad_fn at all, and
    backward() crashes with 'element 0 of tensors does not require grad'.

    Detection: one tiny forward+backward with all params trainable and
    `p.grad is None` afterwards = autograd's own reachability truth. (A
    forward grad_fn walk is NOT reliable here: non-reentrant activation
    checkpointing hides interior param leaves from the forward graph.)

    Returns the frozen count (includes anything frozen before, e.g. MTP).
    """
    saved_flags = [(p, p.requires_grad) for p in model.parameters()]
    for p in model.parameters():
        p.requires_grad_(True)

    was_training = model.training
    model.train()
    # token id 0 is valid for any embedding — only reachability matters
    ids = torch.zeros((1, 8), dtype=torch.long, device=device)
    with autocast_ctx(device):
        loss = compute_loss(model, ids, use_flce)
    loss.backward()
    had_grad = {id(p): p.grad is not None for p in model.parameters()}
    model.zero_grad(set_to_none=True)  # don't leak probe grads into step 1
    model.train(was_training)

    n_dead = 0
    for p, was in saved_flags:
        if not had_grad[id(p)]:
            p.requires_grad_(False)
            p._forge_frozen = True  # BAdam excludes explicitly frozen params
            n_dead += 1
    return n_dead


def preflight_vram_check(model, optimizer, args, device, use_flce: bool) -> dict:
    """Estimate the incremental training footprint vs available VRAM.

    Oversubscribing VRAM spills into Windows shared GPU memory (system RAM
    over PCIe) — measured ~18x slowdown (see .devin/scratchpad.md). Better
    to fail fast with actionable guidance than to start a doomed run.

    Returns {"ok": bool, "est_inc_gb", "available_gb"} so the caller can
    auto-fall-back (e.g. drop factor training) before aborting.
    """
    result = {"ok": True, "est_inc_gb": 0.0, "available_gb": 0.0}
    if device.type != "cuda":
        return result
    weights_b = sum(t.numel() * t.element_size()
                    for t in list(model.parameters()) + list(model.buffers())
                    if t.device.type == "cuda")
    cfg = unwrap_compiled(model).config
    block_sizes = sorted(sum(p.numel() for p in b["params"])
                         for b in optimizer._blocks)
    # Median block, not max: the max block may contain params that never
    # receive grads (frozen MTP head etc.). A hard runtime check after step 1
    # catches under-estimates with only seconds wasted.
    med_block = block_sizes[len(block_sizes) // 2]
    state_bytes = 4.5 if getattr(optimizer, "bf16_large_states", False) else 8.0
    optimizer_b = med_block * state_bytes * 1.5   # m+v, switch-overlap safety
    grads_b = med_block * 2
    tokens = args.batch_size * args.seq_len
    act_b = (tokens * cfg.d_model * cfg.n_layers * 2 * 1.5          # ckpt boundaries
             + tokens * cfg.vocab_size * (2.5 if not use_flce else 1.0))
    result["est_inc_gb"] = (optimizer_b + grads_b + act_b) / 1e9

    free_b, total_b = torch.cuda.mem_get_info(device)
    alloc_b = torch.cuda.memory_allocated(device)
    reserved_b = torch.cuda.memory_reserved(device)
    # Conservative bound: reserved-but-unallocated cache is reusable by torch,
    # but WDDM soft-spill (silent PCIe paging, ~5-18x slowdown) starts BEFORE
    # torch's allocator reports exhaustion. Never plan for more than
    # total - weights - a 0.8 GB OS/driver reserve.
    os_reserve_b = 0.8e9
    reclaim_b = min(free_b + (reserved_b - alloc_b),
                    total_b - weights_b - os_reserve_b)
    result["available_gb"] = max(reclaim_b, 0.0) / 1e9
    total_gb = total_b / 1e9

    log(f"Preflight VRAM: weights {weights_b/1e9:.2f} (already allocated) + "
        f"optimizer ~{optimizer_b/1e9:.2f} + grads ~{grads_b/1e9:.2f} + "
        f"activations ~{act_b/1e9:.2f} = incremental {result['est_inc_gb']:.2f} GB "
        f"vs {result['available_gb']:.2f} GB available ({total_gb:.2f} GB total)")
    if weights_b / 1e9 > total_gb * 0.9:
        log("Weights alone exceed 90% of VRAM — this config cannot train here")
    elif result["est_inc_gb"] <= result["available_gb"] * 0.95:
        return result
    result["ok"] = False
    return result


def abort_over_budget(args) -> None:
    log("PROJECTED OVER BUDGET — this run would spill into Windows shared GPU")
    log("memory (PCIe paging, ~18x slower). Options:")
    log("  - --badam-blocks-per-layer 2 (halves the optimizer spike)")
    log("  - reduce --seq-len or --batch-size")
    log("  - a smaller --config (factor training fits forgelm_v2, not 8B-B)")
    if not args.allow_spill:
        sys.exit("Aborting (use --allow-spill to override)")


# ─────────────────────────────── main ────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V10 full training run")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--lr-schedule", choices=("linear", "wsd", "cosine"),
                        default="linear",
                        help="linear=D2Z (default), wsd=warmup-stable-decay, cosine")
    parser.add_argument("--decay-frac", type=float, default=0.3,
                        help="wsd: fraction of steps in final decay")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=0,
                        help="validate every N steps (0 = final validation only)")
    parser.add_argument("--val-batches", type=int, default=25,
                        help="number of val batches per evaluation")
    parser.add_argument("--config", type=str, default="forgelm_v2")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume", type=str, default="",
                        help="checkpoint to resume from (restores step/LR/BAdam/RNG)")
    parser.add_argument("--datasets", type=str, default="all",
                        choices=list(DATASET_DIRS) + ["all"],
                        help="which datasets to use")
    parser.add_argument("--skip-incompatible", action="store_true",
                        help="skip datasets whose token ids exceed the model "
                             "vocab (e.g. the 151k-vocab lfm pack vs 65536)")
    parser.add_argument("--sampling", choices=("uniform", "stratified"),
                        default="uniform")
    parser.add_argument("--v7-weight", type=float, default=1.0,
                        help="stratified: v7 token-frequency multiplier vs lfm")
    parser.add_argument("--badam-switch-every", type=int, default=10,
                        help="BAdam steps per block (paper uses 100)")
    parser.add_argument("--badam-blocks-per-layer", type=int, default=1,
                        help="2 = split each layer (attn|FFN) — halves the "
                             "optimizer VRAM spike (use with factor training)")
    parser.add_argument("--badam-switch-mode", choices=("descending", "ascending"),
                        default="descending")
    parser.add_argument("--fp32-optimizer-states", action="store_true",
                        help="force fp32 optimizer states (default: bf16 states "
                             "for >1M-element params when factor training is on)")
    parser.add_argument("--compile", action="store_true",
                        help="enable torch.compile (4x on SM120)")
    parser.add_argument("--no-checkpointing", action="store_true",
                        help="disable gradient checkpointing (saves recompute)")
    parser.add_argument("--factor-training", choices=("auto", "on", "off"),
                        default="auto",
                        help="NLRQ STE factor training: auto = use if VRAM "
                             "fits (falls back to S-only with a warning), "
                             "on = require, off = S-only training")
    parser.add_argument("--no-flce", action="store_true",
                        help="disable chunked fused linear-CE (use full logits)")
    parser.add_argument("--allow-spill", action="store_true",
                        help="skip the VRAM preflight abort (shared-memory spill = ~18x slower)")
    parser.add_argument("--keep-checkpoints", type=int, default=3,
                        help="retained {prefix}_step*.safetensors (0 = keep all)")
    parser.add_argument("--ckpt-prefix", type=str, default="ForgeLM_V10",
                        help="checkpoint filename prefix")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="load --checkpoint, evaluate, exit (no training)")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    seed = args.seed if args.seed is not None else int(time.time()) % (2 ** 31)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    # FLCE for anything beyond small smoke sizes (saves ~0.8 GB at seq 2048)
    use_flce = (not args.no_flce) and args.batch_size * args.seq_len >= 1024

    banner("V10 TRAINING")
    log(f"Config: {args.config} | datasets: {args.datasets} | seed: {seed}")
    log(f"Steps: {args.steps} | seq_len={args.seq_len} | lr={args.lr} "
        f"({args.lr_schedule}) | clip={args.grad_clip} | wd={args.weight_decay}")
    log(f"Batch: {args.batch_size} | grad_accum={args.grad_accum} | save every: {args.save_every}")
    log(f"Sampling: {args.sampling}" +
        (f" (v7 weight {args.v7_weight})" if args.sampling == "stratified" else ""))
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info(device)
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {free / 1e9:.2f} GB free / {total / 1e9:.2f} GB total")

    # ── Data ──
    banner("LOADING DATASETS")
    from forge.config import get_config as _get_cfg
    vocab_size = _get_cfg(args.config).vocab_size
    datasets = load_datasets(args.datasets, args.seq_len, vocab_size=vocab_size,
                             skip_incompatible=args.skip_incompatible)
    train_data = torch.cat([d.train for d in datasets]).pin_memory()
    val_data = torch.cat([d.val for d in datasets])
    total_tokens = train_data.numel() + val_data.numel()
    log(f"Datasets: {[d.name for d in datasets]}")
    log(f"Combined train: {tuple(train_data.shape)} | val: {tuple(val_data.shape)}")
    log(f"Total tokens: {total_tokens / 1e6:.1f}M")

    sampler = EpochBatchSampler(
        datasets, args.batch_size,
        weights={"v7": args.v7_weight} if args.sampling == "stratified" else None,
        seed=seed)

    # ── Model ──
    banner(f"BUILDING MODEL: {args.config}")
    t_build = time.perf_counter()
    model, cfg = build_model(
        args.config, device, dtype,
        use_checkpointing=not args.no_checkpointing,
        grad_clip=args.grad_clip,
    )
    load_path = args.resume or args.checkpoint
    if load_path:
        from forge.checkpoint_io import load_checkpoint as _load_ckpt
        log(f"Loading checkpoint: {load_path}")
        sd = _load_ckpt(load_path)
        # drop snapshot metadata (step, val_loss) — not module params
        sd = {k: v for k, v in sd.items() if "." in k}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            raise RuntimeError(f"checkpoint has unexpected keys: {unexpected[:5]}")
        if missing:
            log(f"  note: {len(missing)} missing keys kept at init "
                f"(e.g. {missing[:2]})")
        del sd
    if args.factor_training != "off":
        n = enable_factor_training_all_(model)
        log(f"NLRQ factor training: {n} layers (STE masters, INT8 buffers parked on CPU)")
    # The MTP draft head (268M params) gets no gradients here — the trainer
    # computes CE from logits/hidden states and never passes `targets`, so the
    # model's built-in MTP loss never fires. Freeze it explicitly: keeps the
    # BAdam block honest and the preflight estimate tight. Train MTP heads
    # separately with forge/decoding/mtp.py.
    mtp = getattr(model, "mtp_module", None)
    if mtp is not None:
        for p in mtp.parameters():
            p.requires_grad_(False)
            p._forge_frozen = True
        log("MTP draft head frozen (no CE pathway; train via decoding/mtp.py)")
    if device.type == "cuda":
        torch.cuda.empty_cache()  # probe transiently materializes all grads
    n_dead = freeze_dead_params_(model, device, use_flce)
    log(f"Froze {n_dead} dead param tensors (no grad path — MTP/loop_block/"
        f"gated modules); BAdam blocks now contain only live params")
    log(f"Build time: {time.perf_counter() - t_build:.1f}s")
    print_model_stats(model, cfg)
    vram_snapshot("post-build")

    if args.eval_only:
        banner("EVAL-ONLY")
        result = evaluate(model, val_data, device, args.batch_size,
                          args.val_batches, use_flce=use_flce)
        log(f"Val: loss={result['val_loss']:.4f} (PPL={result['ppl']:.1f})")
        return

    # ── Optimizer ──
    banner("OPTIMIZER: BAdam")
    from forge.keys.compression.nlrq_ffn_key import NLRQLinear
    from forge.training.optim.badam import configure_badam

    factor_on = any(m.factor_training_enabled() for m in model.modules()
                    if isinstance(m, NLRQLinear))

    def make_optimizer():
        return configure_badam(
            model, lr=args.lr, weight_decay=args.weight_decay,
            blocks_per_layer=args.badam_blocks_per_layer,
            switch_every=args.badam_switch_every,
            switch_mode=args.badam_switch_mode,
            bf16_large_states=(factor_on and not args.fp32_optimizer_states),
        )

    optimizer = make_optimizer()

    # ── Preflight + auto-fallback (BEFORE resume: the optimizer mode must be
    # final before loading its saved state) ──
    check = preflight_vram_check(model, optimizer, args, device, use_flce)
    if not check["ok"] and factor_on and args.factor_training == "auto":
        log("AUTO FALLBACK: NLRQ factor masters do not fit VRAM at this "
            "config/seq-len — dropping to S-only training. Use a smaller "
            "--config or --seq-len to keep factor training.")
        for m in model.modules():
            if isinstance(m, NLRQLinear):
                m.disable_factor_training_(export=False)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        factor_on = False
        optimizer = make_optimizer()
        check = preflight_vram_check(model, optimizer, args, device, use_flce)
    if not check["ok"]:
        abort_over_budget(args)

    # ── Resume ──
    start_step = 1
    best_val = float("inf")
    resume_pt = CKPT_DIR / f"{args.ckpt_prefix}_resume.pt"
    if args.resume:
        if resume_pt.exists():
            bundle = torch.load(resume_pt, map_location="cpu", weights_only=False)
            try:
                optimizer.load_state_dict(bundle["badam"])
            except (ValueError, RuntimeError) as exc:
                log(f"Optimizer state not reusable ({str(exc)[:60]}...) — "
                    f"starting fresh optimizer (moments rebuild in ~100 steps)")
            restore_rng(bundle["rng"])
            best_val = bundle.get("best_val", float("inf"))
            start_step = bundle["step"] + 1
            log(f"Resumed at step {start_step} (block {optimizer.active_block_idx}, "
                f"best_val={best_val:.4f})")
        else:
            m = re.search(r"_step(\d+)", Path(args.resume).stem)
            start_step = int(m.group(1)) + 1 if m else 1
            log(f"No resume bundle found — weights-only resume at step {start_step}")

    if args.compile:
        log("Compiling model with torch.compile (mode='reduce-overhead')...")
        t_compile = time.perf_counter()
        model = torch.compile(model, mode="reduce-overhead")
        log(f"Compile setup: {time.perf_counter() - t_compile:.1f}s "
            f"(first step will compile)")

    if device.type == "cuda":
        torch.cuda.empty_cache()  # return build-time scratch to the OS pool
        torch.cuda.reset_peak_memory_stats()  # forget fallback/build transients

    # ── Training loop ──
    banner("TRAINING LOOP")
    log(f"Steps {start_step}..{args.steps} | batch={args.batch_size} "
        f"| grad_accum={args.grad_accum} | seq_len={args.seq_len}")
    log(f"lr={args.lr} ({args.lr_schedule}) | warmup={args.warmup} | FLCE={'on' if use_flce else 'off'}")
    log(f"Save every: {args.save_every} | keep: {args.keep_checkpoints or 'all'} "
        f"| val every: {args.val_every or 'end only'}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    writer = CheckpointWriter(args.keep_checkpoints, CKPT_DIR, args.ckpt_prefix)
    writer.start()

    log_path = LOG_DIR / f"train_8b_{args.ckpt_prefix}_{int(time.time())}.jsonl"
    log_f = open(log_path, "a", encoding="utf-8")

    log_entries: list[dict] = []
    val_history: list[dict] = []
    all_losses: list[float] = []
    ema_loss = None
    skipped_steps = 0
    tokens_seen = 0
    t0 = time.perf_counter()
    step_times: list[float] = []
    best_toks = [0.0]        # tracked across steps for soft-spill detection
    spill_warned = [False]

    def validate(step: int) -> dict:
        result = evaluate(model, val_data, device, args.batch_size,
                          args.val_batches, use_flce=use_flce)
        val_history.append({"step": step, **result})
        log(f"Val @ {step}: loss={result['val_loss']:.4f} (PPL={result['ppl']:.1f})")
        return result

    for step in range(start_step, args.steps + 1):
        t_step = time.perf_counter()
        lr = lr_at(step, args.lr, args.warmup, args.steps,
                   args.lr_schedule, args.decay_frac)

        # ── Micro-batch accumulation ──
        micro_losses = []
        skip_step = False
        for _ in range(args.grad_accum):
            idx = sampler.next()
            input_ids = train_data[idx].to(device, non_blocking=True)

            with autocast_ctx(device):
                loss = compute_loss(model, input_ids, use_flce) / args.grad_accum
            loss_value = loss.item() * args.grad_accum  # un-normalized
            micro_losses.append(loss_value)
            all_losses.append(loss_value)

            if math.isfinite(loss_value):
                if loss.requires_grad:
                    loss.backward()
                else:
                    # Active BAdam block contains only graph-dead params —
                    # nothing to update this step; skip cleanly.
                    log(f"Step {step}: loss has no grad path (dead block), skipping")
                    skip_step = True
            else:
                log(f"[NaN] loss={loss_value:.4f} at step {step}")
                skip_step = True

            # Break autograd references to intermediates. Without this, PyTorch
            # retains forward/backward buffers and VRAM creeps up monotonically.
            del loss, input_ids

        # ── Optimizer step ──
        grad_norm = None
        if skip_step:
            log(f"Step {step}: non-finite loss, skipping optimizer step")
            optimizer.zero_grad()
            skipped_steps += 1
        else:
            for group in optimizer.param_groups:
                group["lr"] = lr
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip))
            if not math.isfinite(grad_norm):
                log(f"Step {step}: non-finite grad norm ({grad_norm}), "
                    f"skipping optimizer step")
                optimizer.zero_grad()
                skipped_steps += 1
            else:
                optimizer.step()
                optimizer.zero_grad()

        # ── Logging ──
        step_t = time.perf_counter() - t_step
        step_times.append(step_t)

        # Hard spill check after the first full step: real peak beats any
        # estimate. torch only sees its own allocations, so also compare
        # against total VRAM — reserved ≈ total means we are spilling.
        if step == start_step and device.type == "cuda" and not args.allow_spill:
            total_b = torch.cuda.get_device_properties(device).total_memory
            peak_b = torch.cuda.max_memory_allocated()
            if peak_b > total_b * 0.92:
                vram_snapshot("spill check")
                sys.exit(f"Aborting: peak {peak_b/1e9:.2f} GB ≈ total VRAM — "
                         f"this run would spill (use --allow-spill to override, "
                         f"or reduce --seq-len / --batch-size)")

        tokens_seen += args.batch_size * args.seq_len * args.grad_accum
        avg_loss = sum(micro_losses) / len(micro_losses)
        ema_loss = avg_loss if ema_loss is None else \
            EMA_BETA * ema_loss + (1 - EMA_BETA) * avg_loss
        window = step_times[-20:]
        tok_s = (args.batch_size * args.seq_len * args.grad_accum) / max(sum(window) / len(window), 1e-6)
        if step > 10:
            best_toks[0] = max(best_toks[0], tok_s)
            if tok_s < 0.4 * best_toks[0] and not spill_warned[0]:
                spill_warned[0] = True
                log(f"WARNING: throughput collapsed ({tok_s:.0f} vs peak "
                    f"{best_toks[0]:.0f} tok/s) — likely WDDM shared-memory "
                    f"paging. Reduce --seq-len/--batch-size or drop factor "
                    f"training.")
        eta_s = (args.steps - step) * (sum(step_times) / len(step_times))
        eta = f"{int(eta_s // 60)}m{eta_s % 60:04.1f}s" if eta_s > 90 else f"{eta_s:.1f}s"

        log(f"Step {step:4d}/{args.steps} | loss={avg_loss:.4f} (ema {ema_loss:.3f}) "
            f"| lr={lr:.2e} | gn={'-' if grad_norm is None else f'{grad_norm:.2f}'} "
            f"| {tok_s:.0f} tok/s | eta {eta}"
            f"{' [NaN]' if skip_step else ''}")

        entry = {
            "step": step, "loss": avg_loss, "ema": ema_loss, "lr": lr,
            "tok_s": tok_s, "step_s": step_t, "grad_norm": grad_norm,
            "block": getattr(optimizer, "active_block_name", "?"),
        }
        log_entries.append(entry)
        log_f.write(json.dumps(entry) + "\n")
        if step % 10 == 0:
            log_f.flush()

        # ── Periodic checkpoint (async, GPU keeps running) ──
        if step % args.save_every == 0 or step == args.steps:
            path = CKPT_DIR / f"{args.ckpt_prefix}_step{step}.safetensors"
            writer.submit(snapshot_state(model, step), path, is_step_ckpt=True)
            try:
                torch.save({
                    "step": step,
                    "badam": slim_badam_state(optimizer),
                    "rng": capture_rng(),
                    "best_val": best_val,
                    "args": {k: v for k, v in vars(args).items()},
                }, resume_pt)
            except Exception as exc:
                log(f"Resume bundle save FAILED: {exc}")

        # ── Periodic validation + best checkpoint ──
        if args.val_every and step % args.val_every == 0 and step != args.steps:
            vl = validate(step)["val_loss"]
            if vl < best_val:
                best_val = vl
                writer.submit(snapshot_state(model, step, {"val_loss": vl}),
                              CKPT_DIR / f"{args.ckpt_prefix}_best.safetensors")

        if step % VRAM_LOG_INTERVAL == 0:
            vram_snapshot(f"step {step}")

    writer.close()
    log_f.close()

    # ── Final validation & save ──
    final_val = validate(args.steps)["val_loss"]
    if final_val < best_val:
        best_val = final_val
        try:
            state = snapshot_state(model, args.steps, {"val_loss": final_val})
            save_state(state, CKPT_DIR / f"{args.ckpt_prefix}_best.safetensors")
            log(f"New best: {final_val:.4f}")
        except Exception as exc:
            log(f"Best save FAILED: {exc}")

    elapsed = time.perf_counter() - t0
    window = [l for l in all_losses[-args.grad_accum:] if math.isfinite(l)]
    final_loss = sum(window) / len(window) if window else float("nan")

    banner("TRAINING COMPLETE")
    log(f"Steps {start_step}..{args.steps} | Time: {elapsed:.1f}s "
        f"({elapsed / max(args.steps - start_step + 1, 1):.1f}s/step)")
    log(f"Final train loss: {final_loss:.4f} (ema {ema_loss:.4f})")
    log(f"Final val: {final_val:.4f} | best: {best_val:.4f} "
        f"| skipped: {skipped_steps}")
    log(f"Tokens seen: {tokens_seen / 1e6:.1f}M")

    final_path = CKPT_DIR / f"{args.ckpt_prefix}_final.safetensors"
    log(f"Saving final checkpoint: {final_path.name}...")
    try:
        state = snapshot_state(model, args.steps, {"val_loss": final_val})
        save_state(state, final_path)
        log(f"Saved ({state_size_gb(state):.1f} GB)")
    except Exception as exc:
        log(f"Save FAILED: {exc}")

    summary_path = LOG_DIR / f"train_8b_{args.ckpt_prefix}_{int(time.time())}.json"
    summary_path.write_text(json.dumps({
        "config": args.config,
        "steps": args.steps,
        "start_step": start_step,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "lr_schedule": args.lr_schedule,
        "grad_clip": args.grad_clip,
        "weight_decay": args.weight_decay,
        "factor_training": factor_on,
        "flce": use_flce,
        "seed": seed,
        "time_s": elapsed,
        "final_loss": final_loss,
        "final_val": final_val,
        "best_val": best_val,
        "skipped_steps": skipped_steps,
        "tokens_seen": tokens_seen,
        "val_history": val_history,
        "losses": [l for l in all_losses if math.isfinite(l)],
        "log_entries": log_entries,
        "datasets": [d.name for d in datasets],
        "total_tokens": total_tokens,
    }, indent=2))
    log(f"Step log: {log_path}")
    log(f"Summary:  {summary_path}")
    log(f"Checkpoints: {CKPT_DIR}/{args.ckpt_prefix}_{{step*,best,final}}.safetensors")


if __name__ == "__main__":
    run(parse_args())
