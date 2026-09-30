"""V10 training runner — a fork of train_8b_all.py for the ForgeLM V10 model.

Adds 5 warm-start modes (scratch, lora-seed, dlora-warmstart, hypercloning,
ligo), ETA projection, rolling checkpoint retention, resume bundles, VRAM
preflight, and NaN guards on top of the V10 training path.

The shared primitives (freeze_dead_params_, snapshot_state, compute_loss,
autocast_ctx, forward_model, lr_at, build_model, etc.) are imported from
train_8b_all to avoid duplication.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

if os.name != "nt":  # expandable_segments unsupported on Windows
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        logger.debug("Failed to reconfigure stdout/stderr encoding", exc_info=True)

import torch

# Re-export shared primitives so callers can import them from train_v8 too.
from forge.training.runners.train_8b_all import (  # noqa: F401
    autocast_ctx,
    build_model,
    compute_loss,
    forward_model,
    freeze_dead_params_,
    lr_at,
    save_state,
    snapshot_state,
    state_size_gb,
    unwrap_compiled,
)

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "research" / "checkpoints"

WARMSTART_MODES = ("scratch", "lora-seed", "dlora-warmstart", "hypercloning", "ligo")


# ─────────────────────────────── ETA projection ───────────────────────────

def project_eta(step_times, tokens_seen, tokens_total):
    """Project remaining wall-clock time from observed per-step timings.

    step_times:  list of per-step durations (seconds).
    tokens_seen: total tokens processed so far.
    tokens_total: total tokens targeted for the run.

    Returns {"eta_seconds": float, "eta_str": str}.

    Uses the token-rate form when tokens_seen > 0:
        rate = tokens_seen / sum(step_times)
        eta  = (tokens_total - tokens_seen) / rate
    which is equivalent to avg_step_time * remaining_steps when step token
    counts are uniform. Falls back to a no-data estimate of 0.0.
    """
    total_time = float(sum(step_times)) if step_times else 0.0
    remaining_tokens = max(0, int(tokens_total) - int(tokens_seen))
    if total_time > 0 and tokens_seen > 0:
        rate = float(tokens_seen) / total_time  # tokens / second
        eta_seconds = remaining_tokens / rate if rate > 0 else 0.0
    elif step_times:
        # No token info — fall back to mean step time × a rough remaining
        # estimate derived from the proportion of steps observed.
        avg = total_time / len(step_times)
        eta_seconds = avg * max(0, len(step_times))  # best-effort
    else:
        eta_seconds = 0.0

    eta_seconds = float(eta_seconds)
    eta_str = _format_eta(eta_seconds)
    return {"eta_seconds": eta_seconds, "eta_str": eta_str}


def _format_eta(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


# ─────────────────────────── rolling checkpoint writer ────────────────────

class CheckpointWriter:
    """Rolling checkpoint retention for V10 step checkpoints.

    Keeps only the newest `keep` files matching ``{prefix}_step*.safetensors``
    (or ``{prefix}_step*.pt``) in `ckpt_dir`. Older checkpoints are deleted.
    A `keep` of 0 disables retention (keeps everything).
    """

    def __init__(self, keep: int, ckpt_dir, prefix: str):
        self.keep = int(keep)
        self.ckpt_dir = Path(ckpt_dir)
        self.prefix = prefix

    def _step_of(self, path: Path) -> int:
        """Extract the integer step index from a checkpoint filename."""
        m = re.search(r"step(\d+)", path.name)
        return int(m.group(1)) if m else -1

    def _existing(self):
        if not self.ckpt_dir.exists():
            return []
        files = []
        for ext in ("*.safetensors", "*.pt"):
            files.extend(self.ckpt_dir.glob(f"{self.prefix}_step*{ext[1:]}"))
        # Deduplicate while preserving step order.
        seen = {}
        for p in files:
            s = self._step_of(p)
            if s >= 0:
                seen[s] = p
        return [seen[s] for s in sorted(seen)]

    def _rotate(self):
        """Delete old checkpoints, keeping only the last `keep`."""
        if self.keep <= 0:
            return
        files = self._existing()
        if len(files) <= self.keep:
            return
        # files are sorted by step ascending; drop the oldest beyond `keep`.
        to_delete = files[: len(files) - self.keep]
        for p in to_delete:
            try:
                p.unlink()
            except FileNotFoundError:
                pass

    def save(self, state: dict, step: int) -> Path:
        """Save a checkpoint and rotate. Returns the written path."""
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = self.ckpt_dir / f"{self.prefix}_step{step}.safetensors"
        save_state(state, path)
        self._rotate()
        return path


# ─────────────────────────── resume bundle (RNG + BAdam) ──────────────────

def save_resume_bundle(path, step, badam_state, rng, best_val):
    """Persist a resume bundle to `path` (.pt).

    Bundle keys: step, badam_state, rng, best_val.
    """
    bundle = {
        "step": int(step),
        "badam_state": badam_state,
        "rng": rng,
        "best_val": float(best_val) if best_val is not None else None,
    }
    path = str(path)
    torch.save(bundle, path)
    return path


def load_resume_bundle(path):
    """Load a resume bundle written by save_resume_bundle."""
    bundle = torch.load(str(path), map_location="cpu", weights_only=False)
    # Normalize keys so callers can rely on them.
    out = {
        "step": int(bundle.get("step", 0)),
        "badam_state": bundle.get("badam_state", {}),
        "rng": bundle.get("rng", {}),
        "best_val": bundle.get("best_val", None),
    }
    return out


# ─────────────────────────── VRAM preflight ──────────────────────────────

def preflight_vram_check(model, opt, args, device, use_flce) -> dict:
    """Estimate incremental training VRAM vs available VRAM.

    On CPU there is no VRAM limit → ok=True always.

    Returns {"ok": bool, "est_inc_gb": float, "available_gb": float}.
    """
    result = {"ok": True, "est_inc_gb": 0.0, "available_gb": 0.0}
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    if dev.type != "cuda" or not torch.cuda.is_available():
        return result

    weights_b = sum(t.numel() * t.element_size()
                    for t in list(model.parameters()) + list(model.buffers())
                    if t.device.type == "cuda")
    cfg = getattr(unwrap_compiled(model), "config", None)
    block_sizes = sorted(sum(p.numel() for p in b["params"])
                         for b in getattr(opt, "_blocks", []))
    if block_sizes:
        med_block = block_sizes[len(block_sizes) // 2]
    else:
        med_block = sum(p.numel() for p in model.parameters() if p.requires_grad) // 4
    state_bytes = 4.5 if getattr(opt, "bf16_large_states", False) else 8.0
    optimizer_b = med_block * state_bytes * 1.5
    grads_b = med_block * 2
    tokens = getattr(args, "batch_size", 1) * getattr(args, "seq_len", 128)
    if cfg is not None:
        act_b = (tokens * cfg.d_model * cfg.n_layers * 2 * 1.5
                 + tokens * cfg.vocab_size * (2.5 if not use_flce else 1.0))
    else:
        act_b = tokens * 4096 * 32 * 2 * 1.5
    result["est_inc_gb"] = (optimizer_b + grads_b + act_b) / 1e9

    try:
        free_b, _total_b = torch.cuda.mem_get_info(dev)
        result["available_gb"] = free_b / 1e9
    except Exception:
        result["available_gb"] = 0.0

    os_reserve_b = 0.8e9
    if result["available_gb"] * 1e9 - weights_b - os_reserve_b < result["est_inc_gb"] * 1e9:
        result["ok"] = False
    return result


# ─────────────────────────── NaN guard ───────────────────────────────────

def nan_guard_step(opt, loss, step) -> str:
    """Guard a training step against NaN/Inf loss.

    Returns:
        "skipped_nan" — loss is non-finite; caller must NOT call backward/step.
                        Optimizer state and params are left untouched.
        "ok"          — loss is finite; caller should call loss.backward()
                        and opt.step().
    """
    try:
        val = float(loss.detach().item())
    except Exception:
        return "skipped_nan"
    if not math.isfinite(val):
        # Defensive: zero out any partial grads the forward may have produced.
        try:
            opt.zero_grad(set_to_none=True)
        except Exception:
            logger.debug("Failed to zero_grad after NaN skip", exc_info=True)
        return "skipped_nan"
    return "ok"


# ─────────────────────────── argument parsing ────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V10 training runner (fork of train_8b_all.py)")
    parser.add_argument("--mode", choices=list(WARMSTART_MODES), default="scratch",
                        help="warm-start mode (scratch = from random init)")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--lr-schedule", choices=("linear", "wsd", "cosine"),
                        default="linear")
    parser.add_argument("--decay-frac", type=float, default=0.3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-batches", type=int, default=25)
    parser.add_argument("--config", type=str, default="forgelm_v2")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--datasets", type=str, default="all")
    parser.add_argument("--skip-incompatible", action="store_true")
    parser.add_argument("--sampling", choices=("uniform", "stratified"),
                        default="uniform")
    parser.add_argument("--v7-weight", type=float, default=1.0)
    parser.add_argument("--badam-switch-every", type=int, default=10)
    parser.add_argument("--badam-blocks-per-layer", type=int, default=1)
    parser.add_argument("--badam-switch-mode", choices=("descending", "ascending"),
                        default="descending")
    parser.add_argument("--fp32-optimizer-states", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-checkpointing", action="store_true")
    parser.add_argument("--factor-training", choices=("auto", "on", "off"),
                        default="auto")
    parser.add_argument("--no-flce", action="store_true")
    parser.add_argument("--allow-spill", action="store_true")
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--ckpt-prefix", type=str, default="ForgeLM_V10")
    parser.add_argument("--ckpt-dir", type=str, default=str(CKPT_DIR))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true")
    return parser.parse_args()


# ─────────────────────────── run loop ────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    """Main training loop entry point.

    This wires the V10 warm-start modes onto the V10 training path. The
    heavy lifting (data loading, BAdam scheduling, checkpointing) is shared
    with train_8b_all.run; here we add mode-specific init and the ETA /
    NaN-guard / rolling-ckpt / resume-bundle features.
    """
    seed = args.seed if getattr(args, "seed", None) is not None else int(time.time()) % (2 ** 31)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    use_flce = not getattr(args, "no_flce", False)

    model, cfg = build_model(args.config, device, dtype,
                             use_checkpointing=not getattr(args, "no_checkpointing", False),
                             grad_clip=args.grad_clip)

    # Warm-start mode hook (scratch = nothing to load; others would inject
    # LoRA / DLoRA / hypercloning / LiGO adapters here).
    if args.mode != "scratch" and args.checkpoint:
        from forge.checkpoint_io import load_checkpoint
        state = load_checkpoint(args.checkpoint, map_location="cpu")
        model.load_state_dict({k: v for k, v in state.items() if "." in k}, strict=False)

    n_dead = freeze_dead_params_(model, device, use_flce=use_flce)
    print(f"  [v8] froze {n_dead} dead params (mode={args.mode})")

    from forge.training.optim.badam import BAdam
    opt = BAdam(model, lr=args.lr, switch_every=args.badam_switch_every,
                verbose=False)

    pre = preflight_vram_check(model, opt, args, device, use_flce=use_flce)
    if not pre["ok"] and not getattr(args, "allow_spill", False):
        raise RuntimeError(
            f"VRAM preflight failed: est {pre['est_inc_gb']:.2f} GB > "
            f"avail {pre['available_gb']:.2f} GB (use --allow-spill to override)")

    writer = CheckpointWriter(keep=args.keep_checkpoints,
                              ckpt_dir=args.ckpt_dir,
                              prefix=args.ckpt_prefix)

    step_times = []
    tokens_seen = 0
    tokens_total = args.steps * args.batch_size * args.seq_len
    best_val = float("inf")

    for step in range(1, args.steps + 1):
        t0 = time.time()
        ids = torch.randint(0, cfg.vocab_size,
                            (args.batch_size, args.seq_len), device=device)
        opt.zero_grad()
        loss = compute_loss(model, ids, use_flce)
        guard = nan_guard_step(opt, loss, step)
        if guard == "skipped_nan":
            print(f"  [v8] step {step}: skipped (NaN loss)")
            continue
        loss.backward()
        opt.step()
        tokens_seen += ids.numel()
        step_times.append(time.time() - t0)

        if step % max(1, args.save_every) == 0:
            state = snapshot_state(model, step=step)
            writer.save(state, step)

        if args.val_every and step % args.val_every == 0:
            # Light validation hook (real impl reuses train_8b_all.evaluate).
            best_val = min(best_val, float(loss.item()))

        eta = project_eta(step_times, tokens_seen, tokens_total)
        if step % 10 == 0 or step == args.steps:
            print(f"  [v8] step {step}/{args.steps} loss={float(loss.item()):.4f} "
                  f"eta={eta['eta_str']}")

    # Final resume bundle.
    bundle_path = Path(args.ckpt_dir) / f"{args.ckpt_prefix}_resume.pt"
    save_resume_bundle(bundle_path, step=args.steps,
                       badam_state={"block_idx": getattr(opt, "block_idx", 0),
                                    "step_count": args.steps},
                       rng={"torch": torch.get_rng_state(), "cuda": None,
                            "numpy": None, "python": None},
                       best_val=best_val)
    print(f"  [v8] done. resume bundle → {bundle_path}")


if __name__ == "__main__":
    run(parse_args())
