"""Selective teacher distillation into a FluxLM snapshot.

Instead of scoring the whole corpus, FLUX only asks the teacher about
contexts where its memory was thin (gap positions collected during
ingest).  Teacher top-k predictions are written as fractional counts
into the order tables, journaled under tag="distill" — fully revertible
via ``model.revert_tag("distill")`` and diffable via ``writes_since``.

Measured on this box: ForgeLM V2 prefill ~350 tok/s continuous, but gap
contexts are only 32 tokens and batched — effective throughput is far
higher per gap.  A same-vocab small teacher (LFM2 family shares the
65536 tokenizer) covers more gaps per minute; pass --teacher hf:<id>.

Usage:
    python scripts/distill_flux.py --model ckpt.flux --max-gaps 20000
    python scripts/distill_flux.py --model ckpt.flux \
        --teacher hf:LiquidAI/LFM2-350M --batch 128
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from forge.model.flux import FluxLM  # noqa: E402


def _v2_score_fn(model, device):
    """ForgeLM V2 / ConfigurableResearchLLM scorer."""
    def score(ctxs):
        x = torch.tensor(ctxs, device=device)
        with torch.inference_mode():
            out = model(x, use_cache=False)
        logits = out[0] if isinstance(out, tuple) else out.logits
        probs = torch.softmax(logits[:, -1].float(), dim=-1)
        return torch.topk(probs, k=8, dim=-1)[::-1]  # (values, indices)
    return score


def _hf_score_fn(hf_id, device, vocab_size):
    """HF CausalLM scorer — requires matching vocab size."""
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16).to(device).eval()
    n_emb = model.get_input_embeddings().num_embeddings
    if n_emb != vocab_size:
        raise SystemExit(
            f"teacher vocab {n_emb} != flux vocab {vocab_size} — "
            f"only same-vocab teachers work (LFM2 family matches 65536)")

    def score(ctxs):
        x = torch.tensor(ctxs, device=device)
        with torch.inference_mode():
            out = model(x)
        logits = out.logits if hasattr(out, "logits") else out[0]
        probs = torch.softmax(logits[:, -1].float(), dim=-1)
        return torch.topk(probs, k=8, dim=-1)[::-1]
    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="flux snapshot path")
    ap.add_argument("--teacher", default="forgelm_v2",
                    help="'forgelm_v2' or 'hf:<model-id>'")
    ap.add_argument("--teacher-ckpt",
                    default="research/checkpoints/ForgeLM_V2.safetensors",
                    help="teacher checkpoint path (forgelm_v2 only)")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--scale", type=float, default=2.0,
                    help="teacher prob -> count multiplier")
    ap.add_argument("--mode", default="tables",
                    choices=["tables", "hebbian", "head"],
                    help="distillation target: order/key tables (verbatim + "
                         "chosen families), _A Hebbian delta writes "
                         "(generalizes via fingerprint cosine), or a "
                         "trained linear readout head (function "
                         "approximation - strongest transfer, reads in "
                         "_predict at weight config.head_w)")
    ap.add_argument("--families", default="orders",
                    help="tables mode only: comma list of key families - "
                         "orders,class,sparse,hybrid,word. Non-order "
                         "families recur on NOVEL text so teacher mass "
                         "written there transfers off-verbatim")
    ap.add_argument("--eta", type=float, default=1.0,
                    help="hebbian mode: write strength multiplier")
    ap.add_argument("--epochs", type=int, default=2,
                    help="head mode: training epochs over collected pairs")
    ap.add_argument("--lr", type=float, default=3e-3,
                    help="head mode: readout-head learning rate")
    ap.add_argument("--scope", default="gaps",
                    choices=["gaps", "all"],
                    help="head mode: train at thin-gap positions only, or "
                         "every ring-retained position")
    ap.add_argument("--head-hidden", type=int, default=0,
                    help="head mode: hidden width >0 trains the MLP "
                         "channel (Wh @ relu(Wh1 @ x)) - measured better "
                         "than linear, which tops out at marginal-"
                         "frequency predictions. Sets config.head_hidden "
                         "on the loaded model before training")
    ap.add_argument("--max-gaps", type=int, default=50_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ctx-len", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", help="resave snapshot here (default: in-place)")
    args = ap.parse_args()

    m = FluxLM.load(args.model, device="cpu")  # memory stays host-side
    gaps = m.gap_positions()
    if not gaps:
        n = m.rescan_gaps()
        gaps = m.gap_positions()
        print(f"gap rescan: {n:,} thin contexts")
    print(f"flux: {m._count:,} stream tokens, {len(gaps):,} gap positions")
    if not gaps:
        print("no gaps — memory already covers every context")
        return 0

    t0 = time.perf_counter()
    if args.teacher == "forgelm_v2":
        # ForgeEngine path: pipelined safetensors loader + auto strategy
        # activation — ~2.4s load vs ~58s raw ModelLoader on this box
        from forge.engine.forge_engine import ForgeEngine
        engine = ForgeEngine.from_checkpoint(
            args.teacher_ckpt, config_name="forgelm_v2",
            device=args.device, auto_activate=False)
        score_fn = _v2_score_fn(engine.model, args.device)
    elif args.teacher.startswith("hf:"):
        score_fn = _hf_score_fn(args.teacher[3:], args.device,
                                m.vocab_size)
    else:
        raise SystemExit(f"unknown teacher {args.teacher!r}")
    print(f"teacher loaded in {time.perf_counter() - t0:.1f}s")

    def progress(done, total):
        el = time.perf_counter() - t0
        print(f"  {done:,}/{total:,} gaps  ({done / max(el, 1e-9):,.0f}/s)",
                flush=True)

    if args.mode == "tables":
        rep = m.distill_from(
            score_fn, top_k=args.top_k, scale=args.scale,
            max_gaps=args.max_gaps, batch_size=args.batch,
            ctx_len=args.ctx_len, tag="distill",
            families=tuple(f.strip() for f in args.families.split(",")
                           if f.strip()),
            progress=progress)
    elif args.mode == "hebbian":
        rep = m.distill_hebbian(
            score_fn, top_k=args.top_k, eta=args.eta,
            max_gaps=args.max_gaps, batch_size=args.batch,
            ctx_len=args.ctx_len, tag="distill", progress=progress)
    else:
        if args.head_hidden:
            m.config.head_hidden = args.head_hidden
        rep = m.distill_head(
            score_fn, top_k=args.top_k, ctx_len=args.ctx_len,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch,
            max_gaps=args.max_gaps, scope=args.scope,
            tag="distill", progress=progress)
    print(rep)
    out = args.out or args.model
    m.snapshot(out)
    print(f"saved {out} — revert with model.revert_tag('distill')")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
