"""Bench FluxLM: ingest tok/s, generation tok/s, RSS memory, recall probe.

    python scripts/bench_flux.py [--vocab 65536] [--tokens 200000]
"""
import argparse
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from forge.model.flux import FluxConfig, FluxLM  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=1024)
    ap.add_argument("--tokens", type=int, default=200_000)
    ap.add_argument("--gen", type=int, default=200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cuda-primary", action="store_true")
    ap.add_argument("--channels", default="",
                    help="comma list: gcell,gsyn,gsyn_deep,dnet,r_live")
    ap.add_argument("--tracemalloc", action="store_true",
                    help="enable tracemalloc (adds ~3x overhead)")
    args = ap.parse_args()

    if args.tracemalloc:
        tracemalloc.start()
    cfg = FluxConfig(vocab_size=args.vocab, device=args.device,
                     cuda_primary=args.cuda_primary)
    for ch in [s.strip() for s in args.channels.split(",") if s.strip()]:
        setattr(cfg, ch, True)
    m = FluxLM(cfg)

    # synthetic corpus: repeated sentence patterns + noise
    import random
    rng = random.Random(0)
    words = [rng.randrange(args.vocab) for _ in range(256)]
    corpus = []
    for i in range(args.tokens):
        if i % 97 < 40:
            corpus.append(words[i % 256])
        else:
            corpus.append(rng.randrange(args.vocab))

    t0 = time.perf_counter()
    m.ingest(corpus, tag="bench")
    dt = time.perf_counter() - t0
    print(f"ingest: {len(corpus) / dt:,.0f} tok/s "
          f"({dt:.2f}s for {len(corpus):,})")

    # generation (predict-only loop, argmax)
    m.soft_reset()
    t0 = time.perf_counter()
    import torch
    for _ in range(args.gen):
        logits = m._predict()
        nxt = int(torch.argmax(logits).item())
        m._learn_token(nxt)
    dt = time.perf_counter() - t0
    print(f"generate: {args.gen / dt:,.0f} tok/s ({dt:.2f}s for {args.gen})")

    rep = m.memory_report()
    line = f"memory_est: {rep['total_MB_est']:.0f} MB"
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
        line += f" | tracemalloc peak: {peak / 1e6:.0f} MB"
    if torch.cuda.is_available() and args.device == "cuda":
        line += (f" | vram: {torch.cuda.memory_allocated() / 1e6:.0f} MB "
                 f"alloc / {torch.cuda.memory_reserved() / 1e6:.0f} MB "
                 f"reserved / {torch.cuda.max_memory_allocated() / 1e6:.0f} MB peak")
    print(line)
    print(f"channels: {m.channel_weights()}")
    print(f"journal: {len(m.journal):,} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
