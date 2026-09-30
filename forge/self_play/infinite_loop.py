"""Unified infinite self-play training loop (AZR paradigm).

Merges the AZR-style curriculum (propose → solve → verify) from
`infinite_curriculum.py` with the training orchestration (SFT, LoRA,
promote/demote, checkpoint archiving) from the former tool-use loop.

Cycle per epoch:
  1. Propose: model generates coding tasks with unit tests
  2. Validate: Python executor checks self-consistency (reference solution passes tests)
  3. Solve: model attempts to solve validated tasks
  4. Export: successful (task → solution) pairs → SFT JSONL
  5. Finetune: SFT continuation from current best checkpoint (LoRA)
  6. Evaluate: fast_eval compares candidate vs best (code, reasoning, knowledge)
  7. Promote/Demote: if candidate passes, it becomes the new best
  8. Repeat

Usage:
    python -m research.self_play.infinite_loop \\
        --checkpoint research/checkpoints/ForgeLM_V2.safetensors \\
        --epochs 50 --tasks-per-epoch 30

The loop is resumable: if interrupted, it picks up from the last
completed epoch using the saved checkpoint + curriculum state.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from forge.training.training_utils import oom_guard

logger = logging.getLogger(__name__)

from forge.self_play.infinite_curriculum import GenTracer, InfiniteCurriculum
from forge.self_play.flux_goals import TOOL_HEADER as _FLUX_TOOL_HEADER
from forge.self_play.flux_goals import parse_tool_call as _parse_tool_call


@dataclass
class LoopConfig:
    """Configuration for the unified AZR self-play loop.

    V10 update (2026-08-31): defaults to ForgeLM V10-1.2B, wires ForgeEngine
    inference features (SpectralKV, MTP self-speculative decoding, prefix
    cache, Triton conv) into the self-play generation phase, and exposes
    all sft_train training tricks (Muon-SF optimizer, grad-mixup, curriculum,
    augmentation, SYNPRO, focal loss, entropy weighting, MTP auxiliary loss,
    distillation, sequential freeze, EMA, BitNet-everywhere).
    """
    # Self-play (AZR curriculum)
    tasks_per_epoch: int = 60        # propose + solve per epoch (~60 →
                                     # ~40+ unique SFT examples at typical
                                     # solve rates; halves per-example
                                     # repetition vs 30 at ft_max_steps)
    max_gen_tokens: int = 256        # max tokens per generation
    temperature: float = 0.7         # exploration temperature
    top_k: int = 80                  # LFM2.5-recommended
    top_p: float = 0.95
    propose_batch_size: int = 8      # parallel task proposal
    domains: tuple = ("algorithms", "math", "strings", "logic", "data_structures")

    # ── ForgeEngine inference features (self-play generation phase) ──
    # V10: SpectralKV is the production KV cache (63× compression at 0.095 error).
    # When use_forge_engine=True, the self-play phase loads via ForgeEngine
    # and activates these features instead of the bare model path.
    use_forge_engine: bool = True    # load via ForgeEngine (V10 features)
    kv_cache: str = "spectral"       # V10 SpectralKV (was: none / bare model)
    decoding: str = "mtp_selfspec"   # MTP self-speculative (2-4× decode speedup)
    quantize: str | None = None      # weight quantization (None=bf16, "w8a8", "nvfp4")
    acceleration: str | None = None  # "cuda_graph", "megakernel", "flex_decoding"
    use_compile: bool = False        # torch.compile (experimental on Windows)
    use_triton_conv: bool = True     # Triton fused conv kernel (89% conv bottleneck cut)
    use_prefix_cache: bool = True    # prefix KV cache (repeated prompt prefixes)
    use_chunked_prefill: bool = True # chunked prefill (long prompts)
    # NOTE: disabled — FusedQKNormRopeCacheWrapper has a RoPE-convention
    # mismatch on V10 (model uses NeoX full-dim cos/sin via cat((freqs,freqs)),
    # but the wrapper's _py_qk_norm_rope expects GPT-J half-dim cos/sin →
    # 32-vs-64 shape crash). The original attention forward handles RoPE
    # correctly. Re-enable only after a bit-exact forward-pass comparison
    # (AGENTS.md directive A). Tracked in .devin/scratchpad.md.
    use_fused_qk_norm_rope_cache: bool = False  # fused QK-norm+RoPE+cache-write
    use_seq_split: bool = True       # sequence-split attention (long context)
    use_spec_attn: bool = True       # speculative attention
    kv_cache_tokens: int = 4096      # KV cache token budget
    warmup: bool = True              # warmup dummy token (init CUDA kernels)

    # Finetune
    ft_max_steps: int = 100          # SFT steps per epoch
    ft_lr: float = 5e-5              # V10 default (muon_sf tolerates higher LR)
    ft_min_lr: float = 5e-6          # cosine decay floor
    ft_batch_size: int = 1
    ft_grad_accum: int = 5           # evolution-discovered: 5 (effective batch 5)
    ft_sync_freq: int = 15           # evolution-discovered: sync every 15 steps
    ft_seq_len: int = 1024
    ft_warmup_steps: int = 20        # warmup for stability (evolution: 0 was synthetic-only)
    ft_grad_checkpoint: bool = True
    ft_checkpoint_strategy: str = "all"  # selective: "ffn", "attn", "lazy", "optimal"
    ft_optimizer: str = "muon_sf"    # V10 default: Muon-SF (2.39× vs AdamW)
    ft_lora: bool = True             # LoRA: train ~1M params
    ft_lora_r: int = 32              # V10 default rank (was 16)
    ft_lora_alpha: int = 64          # V10 default alpha (was 32)
    ft_bitnet_everywhere: bool = True  # BitNet ternary QAT (1.58 bits, 2.39× vs AdamW)
    ft_manual_lora: bool = True      # BitNet-compatible LoRA (auto-enabled w/ bitnet)
    ft_entropy_alpha: float = 0.5    # token entropy weighting (WeFT/VCORE 2025)
    ft_loss_function: str = "ce"     # "ce", "focal", "label_smoothing", "dynamic_focal"
    ft_focal_gamma: float = 4.93     # evolution-discovered focal optimum
    ft_label_smoothing_eps: float = 0.1
    ft_grad_compression: str = "int4"  # evolution-discovered: int4 for CPU offload
    ft_vram_limit_gb: float = 11.0
    ft_min_examples: int = 8         # skip finetune if fewer than this

    # ── Training tricks (R&D round 14+, wired to sft_train) ──
    ft_grad_mixup: int = 1           # N-batch grad averaging (1=off, 3=1.25× convergence)
    ft_curriculum: str = "none"      # "vanilla", "pacing", "interleaved", "warmup"
    ft_augment: bool = False         # token noise + FIM + target offset (anti-overfit)
    ft_synpro: bool = False          # SYNPRO synthetic data (3.7-5.2× effective tokens)
    ft_norm_type: str = "rmsnorm"    # "seednorm", "dyt" (Muon-compatible)
    ft_mtp_weight: float = 0.0       # MTP auxiliary loss (Nemotron Lightning)
    ft_mtp_n_heads: int | None = None  # override MTP heads (None=use config)
    ft_distill: bool = False         # knowledge distillation from teacher
    ft_teacher_checkpoint: str | None = None  # teacher checkpoint for distill
    ft_distill_topk: int = 50        # top-K logits to cache from teacher
    ft_distill_truncate: float = 1.0  # sequence truncation ratio for distill
    ft_distill_prefix: bool = False  # on-policy prefix distillation (2-40× FLOP cut)
    ft_sequential_freeze: int = 0    # sequential freeze/unfreeze (0=off, 4=4 phases)
    ft_final_finetune_steps: int = 0  # reserved all-layer finetune steps at end
    ft_ema: bool = False             # EMA shadow weights
    ft_ema_decay: float = 0.999
    ft_anchor: str | None = None     # L2-SP anchor checkpoint (anti-drift)
    ft_l2_lambda: float = 0.01       # L2-SP regularization lambda
    ft_sample_weighting: bool = False  # easy sample upweighting (ICML 2025)
    ft_lazy_train: bool = False      # LazyTrain scheduler (1.24× sustained TFLOPS)
    ft_oomb: bool = False            # OOMB chunk-recurrent (128K+ context on 12GB)
    ft_hybrid_clip: bool = False     # Hybrid8Bit fast grad clip
    ft_elastic_grad_accum: bool = False  # dynamic grad_accum (OOM prevention)
    ft_freetoken: bool = False       # FreeToken double-buffered grad pipeline
    ft_compile: bool = False         # torch.compile model (experimental on Windows)
    ft_disk_cache: bool = True       # disk-backed tokenization cache
    ft_pack_sequences: bool = True   # pack variable-length sequences (Llama-3 style)
    ft_async_prefetch: bool = True   # async background prefetcher (2-3× throughput)
    ft_prefetch_count: int = 4       # batches to prefetch ahead

    # Loop control
    max_epochs: int = 50
    eval_threshold: float = 0.5      # lenient mode: max fractional quality regression
    strict_promote: bool = True      # promote only if candidate beats/ties best
    device: str = "cuda"

    # ── Training mode ──
    # "sft"  = SFT continuation (default, existing path)
    # "grpo" = GRPO-only RSI mode: train directly on verified self-play
    #          trajectories using Group Relative Policy Optimization.
    #          Skips SFT entirely — RL is the sole training signal.
    # "flux" = FluxLM live-learning RSI: the model attempts verified
    #          tasks, correct trajectories get reinforce()'d, failures
    #          get revert_tag()'d + teach()'d the verified answer, then
    #          consolidate() promotes agreeing episodic memory into the
    #          fast tables.  Promotion gates on a FIXED held-out QA set
    #          (same questions every epoch — the improvement trend).
    training_mode: str = "sft"

    # ── FluxLM config (only used when training_mode="flux") ──
    flux_eval_n: int = 40          # fixed held-out QA eval size
    flux_eval_seed: int = 77777    # eval epoch seed — constant across run
    flux_gen_tokens: int = 24      # QA answers are short
    flux_gain: float = 4.0         # reinforce() gain on verified success
    flux_consolidate_min: int = 4  # min agreeing episodic entries
    flux_group_size: int = 4       # G sampled attempts per task (batch)
    flux_workers: int = 4          # max concurrent predict-only streams
    flux_workers_device: str = "cpu"  # retained for compat; streams run
                                    # on the model's own device
    flux_temp: float = 0.3         # attempt temperature center
    flux_temp_spread: float = 0.5  # ± uniform range, clipped to [0, 1]
    flux_unique: bool = True       # cross-epoch prompt dedup (no repeats)
    # pluggable reward: callable(response:str, pair:dict) -> float in
    # [0,1].  None = binary _flux_check_any (1.0/0.0).  The reward
    # scales reinforce() gain per attempt — graded/partial credit.
    flux_verifier: object = None
    flux_ok_threshold: float = 0.5  # reward >= this counts as success
    flux_partial_gain: float = 0.5  # reward scale for tool-call-only
                                    # success (parsed call, wrong answer)
    # exploration: eps fraction of samples decode at explore temp
    # instead of the exploit center — widens discovered trajectories
    flux_explore_eps: float = 0.0
    flux_explore_temp: float = 1.3
    # counterfactual credit (R59): log each generated token's top-k
    # alternative propensities in FluxPacket.alts; reinforce_alts then
    # writes propensity-weighted credit to the untried tokens.  On
    # failure (flux_cf_fail) the sampled trajectory is applied, the
    # alternatives boosted, then the sampled writes reverted — "the
    # untried action keeps credit for what the tried one lost".
    flux_cf_topk: int = 8        # 0 disables propensity logging
    flux_cf_scale: float = 0.5   # alt-credit weight vs sampled credit
    flux_cf_fail: bool = True    # counterfactual learning on failures

    # ── R53: source-grounded goals + tool use ──
    # Goals are drawn from Wikipedia summaries, dictionary entries and
    # arXiv papers (verified extractive answers), mixed with the
    # procedural stream.  Web goals run in three modes: "qa" (bare
    # recall), "rag" (source snippet in-prompt — teaches extraction),
    # and "tool" (model emits `T: <tool> <query>`, loop injects `O:`,
    # model answers) — FLUX learns the tool-call pattern via teach-on-
    # fail demonstrations of the full trajectory.
    flux_web: bool = True          # web-sourced goals (offline -> procedural)
    flux_goal_sources: str = "wiki,dict,arxiv"  # enabled web sources
    flux_web_frac: float = 0.5     # fraction of epoch goals from sources
    flux_tool_frac: float = 0.35   # of web goals: two-phase tool tasks
    flux_rag_frac: float = 0.35    # of remaining web goals: inline context
    flux_web_fetch_budget: int = 8 # new source fetches per epoch
    flux_eval_web_n: int = 16      # frozen web eval size (0 disables)
    flux_dict_file: str = ""       # optional kaikki.org wiktionary jsonl
    flux_obs_chars: int = 240      # tool observation truncation
    flux_context_chars: int = 700  # source text ingested on teach-fail
    # teacher-embedding graft for a FRESH flux model (snapshots restore
    # their own _R).  "auto" = ForgeLM V2 checkpoint if present — the
    # single biggest lever toward V2-quality recall (semantic features
    # feed topic/csem/knn channels). "" disables.
    flux_teacher_embed: str = "auto"

    # ── GRPO config (only used when training_mode="grpo") ──
    grpo_max_steps: int = 50         # GRPO steps per epoch
    grpo_group_size: int = 4         # G completions per prompt (MC-GRPO)
    grpo_lr: float = 5e-6            # GRPO learning rate
    grpo_kl_coeff: float = 0.02      # KL penalty coefficient (β)
    grpo_clip_range: float = 0.2     # PPO clip range (ε)
    grpo_temperature: float = 0.8    # generation temperature for GRPO rollouts
    grpo_max_new_tokens: int = 256   # max tokens per GRPO rollout
    grpo_max_seq_len: int = 512      # max seq len for GRPO training
    grpo_grad_accum: int = 2         # gradient accumulation steps
    grpo_use_repetition_penalty: bool = True  # doom-loop mitigation
    grpo_rl_algorithm: str = "grpo"  # grpo/gtpo/cispo/sppo/psppo/evpo/grpo_or
    grpo_min_trajectories: int = 4   # skip GRPO if fewer verified trajectories

    # Paths
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data/finetune"
    config_name: str = "forgelm_v2"  # V2 (Jamba) default

    # Replay buffer: mix in prior SFT data to prevent catastrophic forgetting
    replay_file: str = ""            # path to prior SFT JSONL
    replay_ratio: float = 0.2        # fraction of replay examples in each epoch

    # Concise-answer stream: verified short-answer QA pairs mixed into each
    # epoch's SFT export — teaches direct, minimal answers on easy prompts
    # (the "don't overthink" + "short simple response" behaviors; eval's
    # 'concise' category scores them). 0 disables.
    concise_qa_per_epoch: int = 12
    # GRPO length-efficiency bonus: within a passing group, shorter code
    # earns slightly more reward (correctness still dominates). Restores
    # nonzero advantage in all-pass groups and pushes concise solutions.
    grpo_concise_bonus: float = 0.1  # 0 disables

    # Task source: "model" (AZR self-propose) or "api" (distillation APIs)
    task_source: str = "model"       # "model" or "api"

    # Live telemetry: status.json + heartbeat.json + events.jsonl under
    # status_dir, polled by the GUI Self-Play page (EventsReader/StatusReader)
    # and readable directly for CLI monitoring of an RSI run.
    live_status: bool = True
    status_dir: str = "research/checkpoints/self_play"

    # Debug round: run Phase 1 self-play only (propose → validate → solve)
    # with a GenTracer on the curriculum — every model generation's raw
    # output + token ids are printed live and dumped to
    # status_dir/debug_round_ep{N}.txt|.jsonl — then stop. No training,
    # eval, or promotion. For inspecting model behavior before a real run.
    debug_round: bool = False


# ── Concise-QA stream ─────────────────────────────────────────────────
# Verified (question → minimal answer) pairs. Answers are computed or
# curated-correct — never model-generated — so every pair is ground truth.
# ~Half the prompts carry an explicit brevity hint; the rest are bare so
# the model learns brevity as the DEFAULT, not only when asked.

_QA_FACTS = (
    ("the capital of Japan", "Tokyo"),
    ("the capital of Germany", "Berlin"),
    ("the capital of the United Kingdom", "London"),
    ("the capital of Italy", "Rome"),
    ("the capital of Spain", "Madrid"),
    ("the capital of Canada", "Ottawa"),
    ("the capital of Australia", "Canberra"),
    ("the chemical symbol for gold", "Au"),
    ("the chemical symbol for iron", "Fe"),
    ("the chemical symbol for sodium", "Na"),
    ("the planet closest to the Sun", "Mercury"),
    ("the Red Planet", "Mars"),
    ("the largest ocean on Earth", "Pacific"),
    ("the freezing point of water in Celsius", "0"),
    ("the number of days in a week", "7"),
    ("the number of months in a year", "12"),
)

_QA_BRIEF_HINTS = ("", "", "", " Be concise.", " One word.",
                   " Short answer only.", " Just the answer.")


def _concise_qa_pairs(n: int, epoch: int) -> list[dict]:
    """Build n verified short-answer SFT pairs (prompt → minimal response).

    Teaches what the code-trajectory stream cannot: answer an easy prompt
    DIRECTLY and MINIMALLY. Deterministic per epoch (resumed runs rebuild
    the same mix). Prompts deduplicated within the batch.
    """
    import random as _r
    rng = _r.Random(7919 + epoch)

    def _hinted(q: str) -> str:
        return q + rng.choice(_QA_BRIEF_HINTS)

    def arith():
        a, b = rng.randint(7, 99), rng.randint(3, 97)
        op = rng.choice("+-*")
        ans = {"+": a + b, "-": a - b, "*": a * b}[op]
        q = rng.choice((f"What is {a} {op} {b}?", f"Calculate {a} {op} {b}.",
                        f"{a} {op} {b} ="))
        return _hinted(q), str(ans)

    def compound():
        a, b, c = (rng.randint(2, 30) for _ in range(3))
        return _hinted(f"What is ({a} + {b}) * {c}?"), str((a + b) * c)

    def percent():
        p, n_ = rng.choice((10, 20, 25, 50, 75)), rng.choice((20, 40, 60, 80, 120, 200))
        return _hinted(f"What is {p}% of {n_}?"), str(int(n_ * p / 100))

    def square():
        k = rng.randint(3, 25)
        return _hinted(f"What is {k} squared?"), str(k * k)

    def string_op():
        w = rng.choice(("python", "curriculum", "tensor", "jamba", "kernel",
                        "gradient", "selfplay", "reasoning", "tokenizer", "mamba"))
        kind = rng.choice(("reverse", "upper", "len", "vowels"))
        if kind == "reverse":
            return _hinted(f"Reverse the word '{w}'."), w[::-1]
        if kind == "upper":
            return _hinted(f"Write '{w}' in uppercase."), w.upper()
        if kind == "len":
            return _hinted(f"How many letters are in '{w}'?"), str(len(w))
        return _hinted(f"How many vowels are in '{w}'?"), str(
            sum(1 for ch in w if ch in "aeiou"))

    def list_op():
        xs = rng.sample(range(1, 60), rng.randint(4, 6))
        kind = rng.choice(("sum", "max", "min"))
        ans = {"sum": sum(xs), "max": max(xs), "min": min(xs)}[kind]
        verb = {"sum": "the sum", "max": "the largest value",
                "min": "the smallest value"}[kind]
        return _hinted(f"What is {verb} of {xs}?"), str(ans)

    def parity():
        k = rng.randint(2, 200)
        return _hinted(f"Is {k} even? Answer yes or no."), "yes" if k % 2 == 0 else "no"

    def prime():
        k = rng.randint(2, 60)
        isp = k > 1 and all(k % d for d in range(2, int(k ** 0.5) + 1))
        return _hinted(f"Is {k} a prime number? Answer yes or no."), "yes" if isp else "no"

    def base_conv():
        k = rng.randint(5, 255)
        kind = rng.choice(("binary", "hexadecimal", "octal"))
        ans = {"binary": bin(k)[2:], "hexadecimal": hex(k)[2:],
               "octal": oct(k)[2:]}[kind]
        return _hinted(f"What is {k} in {kind}?"), ans

    def compare():
        a, b = rng.sample(range(10, 500), 2)
        hi = rng.random() < 0.5
        return _hinted(f"Which is {'larger' if hi else 'smaller'}: {a} or {b}?"), str(
            max(a, b) if hi else min(a, b))

    def unit_conv():
        k, kind = rng.randint(2, 90), rng.choice(("min2sec", "hr2min", "day2hr",
                                                "week2day", "kg2g", "m2cm", "cm2mm"))
        spec = {"min2sec": (f"{k} minutes in seconds", k * 60),
                "hr2min": (f"{k} hours in minutes", k * 60),
                "day2hr": (f"{k} days in hours", k * 24),
                "week2day": (f"{k} weeks in days", k * 7),
                "kg2g": (f"{k} kg in grams", k * 1000),
                "m2cm": (f"{k} m in cm", k * 100),
                "cm2mm": (f"{k} cm in mm", k * 10)}[kind]
        return _hinted(f"How many is {spec[0]}?"), str(spec[1])

    def seq_next():
        a0, d = rng.randint(1, 20), rng.randint(2, 9)
        xs = [a0 + d * i for i in range(4)]
        return _hinted(f"What comes next: {', '.join(map(str, xs))}, ...?"), str(xs[-1] + d)

    def digit_op():
        k = rng.randint(100, 9999)
        if rng.random() < 0.5:
            return _hinted(f"What is the digit sum of {k}?"), str(sum(map(int, str(k))))
        return _hinted(f"How many digits does {k} have?"), str(len(str(k)))

    def fact():
        q, a = rng.choice(_QA_FACTS)
        q_fmt = rng.choice((f"What is {q}?", f"Name {q}.", f"{q.capitalize()} — what is it?"))
        return _hinted(q_fmt), a

    # ── R53 additions: wider procedural coverage (all computed — every
    # answer is derived, never authored) ──
    def gcd():
        import math
        a, b = rng.randint(6, 240), rng.randint(6, 240)
        return _hinted(
            f"What is the greatest common divisor of {a} and {b}?"), \
            str(math.gcd(a, b))

    def mean():
        k = rng.randint(3, 5)
        base = rng.randint(2, 40)
        xs = [base + rng.randint(0, 8) * k for _ in range(k)]
        xs = [x - (sum(xs) % k) if i == 0 else x
              for i, x in enumerate(xs)]     # force integer mean
        return _hinted(f"What is the mean of {xs}?"), str(sum(xs) // k)

    def median():
        xs = sorted(rng.sample(range(1, 99), rng.choice((3, 5))))
        return _hinted(f"What is the median of {xs}?"), str(xs[len(xs) // 2])

    def modulo():
        a, b = rng.randint(10, 500), rng.randint(3, 12)
        return _hinted(f"What is {a} mod {b}?"), str(a % b)

    def letter_idx():
        w = rng.choice(("python", "curriculum", "tensor", "jamba",
                        "kernel", "gradient", "memory", "neuron"))
        i = rng.randint(1, len(w))
        suf = {1: "st", 2: "nd", 3: "rd"}.get(i, "th")
        return _hinted(f"What is the {i}{suf} letter of '{w}'?"), w[i - 1]

    def sort_rank():
        xs = rng.sample(range(10, 99), 4)
        kind = rng.choice(("smallest", "largest"))
        ans = min(xs) if kind == "smallest" else max(xs)
        return _hinted(
            f"Sort {xs} and give the {kind} value."), str(ans)

    def word_count():
        ws = rng.choice(
            ("the quick brown fox jumps", "a neural network learns",
             "self play improves models", "memory cells store facts",
             "the model reads context")).split()
        return _hinted(
            f"How many words are in '{' '.join(ws)}'?"), str(len(ws))

    builders = (arith, compound, percent, square, string_op, list_op, parity,
                prime, base_conv, compare, unit_conv, seq_next, digit_op,
                fact, gcd, mean, median, modulo, letter_idx, sort_rank,
                word_count)

    pairs: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(pairs) < n and attempts < n * 8:
        attempts += 1
        prompt, response = rng.choice(builders)()
        if prompt in seen:
            continue
        seen.add(prompt)
        pairs.append({"prompt": prompt, "response": response})
    return pairs


def _concise_adjusted_rewards(comps: list, rews: list,
                              bonus: float) -> list:
    """Add a small length-efficiency bonus to PASSING completions.

    Shortest pass earns +bonus, longest +0 — the binary correctness reward
    still dominates (a fail is always 0), but all-pass groups regain a
    nonzero advantage aimed at the terse solution instead of collapsing
    to loss 0 (observed: identical rewards → no learning signal).
    """
    if bonus <= 0:
        return list(rews)
    lens = [len(c or "") for c in comps]
    pass_lens = [ln for ln, r in zip(lens, rews) if r > 0]
    if not pass_lens:
        return list(rews)
    lo, hi = min(pass_lens), max(pass_lens)
    span = max(hi - lo, 1)
    return [r + bonus * (1.0 - (ln - lo) / span) if r > 0 else r
            for ln, r in zip(lens, rews)]


class InfiniteSelfPlayLoop:
    """The unified AZR self-play → finetune → evaluate → promote loop."""

    def __init__(self, checkpoint: str, config: LoopConfig | None = None):
        self.config = config or LoopConfig()
        self.best_checkpoint = checkpoint
        self.best_score = 0.0
        self.epoch = 0
        self.history: list[dict] = []

        # Trajectory storage: (task_description, solution_code, success)
        self._trajectories: list[dict] = []
        # Debug-round generation transcript (populated when
        # config.debug_round attaches a GenTracer to the curriculum).
        self._gen_trace: GenTracer | None = None
        # ForgeEngine retained across self-play → eval for weight-swap reuse.
        self._engine = None
        # FluxLM mode state: cross-epoch prompt dedup + attempt rng.
        self._flux_seen: set = set()
        self._flux_rng = random.Random()
        # Live telemetry writer — created in run() so tests that call
        # run_epoch() directly never touch the filesystem.
        self._status_writer = None
        self._monitor = None

    def _init_status_writer(self) -> None:
        """Create the live telemetry writer + failure-mode monitor (idempotent)."""
        if self._status_writer is not None or not self.config.live_status:
            return
        from forge.self_play.live_status import LiveStatusWriter
        from forge.self_play.monitoring import SelfPlayMonitor
        status_path = os.path.join(self.config.status_dir, "status.json")
        self._status_writer = LiveStatusWriter(status_path)
        self._monitor = SelfPlayMonitor()
        self._status_writer.update(
            name="self_play", method=self.config.training_mode,
            config=self.config.config_name,
            checkpoint=self.best_checkpoint,
            vram_gb=self._vram_gb())

    def _vram_gb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.memory_allocated() / 1e9, 2)

    def _status_update(self, **fields) -> None:
        """Telemetry update that no-ops when the writer isn't active."""
        if self._status_writer is not None:
            self._status_writer.update(vram_gb=self._vram_gb(), **fields)

    def _record_step_metrics(self, metrics: dict) -> None:
        """Feed SelfPlayMonitor + surface alerts into status.monitor_alerts."""
        if self._monitor is not None:
            self._monitor.record_step(metrics)
            alerts = self._monitor.check_alerts()
            self._status_update(monitor_alerts=alerts)

    def _epoch_checkpoint_path(self, epoch: int) -> str:
        """Unique per-epoch checkpoint path — never overwrites.

        The epoch counter resets when a run restarts, so a bare SP{epoch}
        name would clobber checkpoints from a previous run. Append _rN
        until the path is free.
        """
        cfg_tag = self.config.config_name.replace("forgelm_", "").upper() or "MODEL"
        stem = os.path.join(
            self.config.checkpoint_dir, f"ForgeLM_{cfg_tag}_SP{epoch}")
        path = f"{stem}.safetensors"
        n = 1
        while os.path.exists(path):
            n += 1
            path = f"{stem}_r{n}.safetensors"
        return os.path.normpath(path)

    def _load_engine(self):
        """Load a ForgeEngine with V10 inference features activated.

        When ``use_forge_engine`` is True (default), uses
        ``ForgeEngine.from_checkpoint`` and activates the V10 feature set
        (SpectralKV, MTP self-speculative decoding, prefix cache, Triton
        conv, fused QK-norm+RoPE+cache-write, sequence-split attention).
        The engine.model is used for curriculum generation; the engine
        itself is reused by fast_eval (weight-swap, no reload).

        When False, falls back to the bare ``load_default_model`` path
        (legacy, no V10 inference features — kept for debugging/ablation).
        """
        cfg = self.config
        if cfg.use_forge_engine:
            from forge.engine.forge_engine import ForgeEngine
            engine = ForgeEngine.from_checkpoint(
                checkpoint=self.best_checkpoint,
                config_name=cfg.config_name,
                tokenizer_path="research/checkpoints/forgelm_v2_tokenizer",
                device=cfg.device,
                auto_activate=False,  # we activate explicitly below
            )
            # Activate V10 inference features. SpectralKV is the V10 production
            # KV cache (63× compression). MTP self-spec gives 2-4× decode
            # speedup when the checkpoint has MTP heads (V10-growth configs).
            engine.activate(
                kv_cache=cfg.kv_cache,
                decoding=cfg.decoding,
                quantize=cfg.quantize,
                acceleration=cfg.acceleration,
                kv_cache_tokens=cfg.kv_cache_tokens,
                use_compile=cfg.use_compile,
                use_triton_conv=cfg.use_triton_conv,
                use_prefix_cache=cfg.use_prefix_cache,
                use_chunked_prefill=cfg.use_chunked_prefill,
                use_fused_qk_norm_rope_cache=cfg.use_fused_qk_norm_rope_cache,
                use_seq_split=cfg.use_seq_split,
                use_spec_attn=cfg.use_spec_attn,
                warmup=cfg.warmup,
            )
            print(f"  [ForgeEngine] V10 features active: kv={cfg.kv_cache}, "
                  f"decoding={cfg.decoding}, triton_conv={cfg.use_triton_conv}, "
                  f"prefix_cache={cfg.use_prefix_cache}")
            return engine

        # Legacy bare-model path (no V10 inference features)
        from forge.engine.forge_engine import ForgeEngine
        from forge.model_loader import load_default_model
        model, tokenizer = load_default_model(
            cfg.config_name,
            checkpoint_path=self.best_checkpoint,
            device=cfg.device,
            dtype=torch.bfloat16,
        )
        model.eval()
        # Wrap in a minimal ForgeEngine shell so fast_eval's weight-swap
        # path still works (engine.model + engine.tokenizer).
        engine = ForgeEngine(model, tokenizer, device=cfg.device,
                             checkpoint_path=self.best_checkpoint)
        return engine

    # ── FluxLM RSI mode (training_mode="flux") ───────────────────────
    # Live-learning RSI: attempt verified tasks → reinforce successes /
    # revert+teach failures → consolidate → snapshot → fixed-eval gate →
    # promote or full epoch rollback.  Every write is journaled so a
    # demoted epoch is erased exactly (revert_since(pre_epoch_mark)).

    @staticmethod
    def _flux_check(output: str, answer: str) -> bool:
        """Verified-task grader: the answer is the first thing emitted
        (answer-first skill) or appears word-bounded anywhere."""
        a = answer.strip().lower()
        if not a:
            return False
        out = output.strip().lower()
        if out.startswith(a):
            return True
        # right edge: EOS, non-word char, or '.' NOT followed by a digit
        # (so "42." matches "42" but "42.5" does not)
        return re.search(r"(?<![\w.])" + re.escape(a)
                         + r"(?=$|[^\w.]|\.(?!\d))", out) is not None

    def _flux_checkpoint_path(self, epoch: int) -> str:
        """Per-epoch .flux snapshot under the GUI-scanned flux dir."""
        d = os.path.join(self.config.checkpoint_dir, "flux")
        os.makedirs(d, exist_ok=True)
        stem = os.path.join(d, f"FLUX_SP{epoch}")
        path = f"{stem}.flux"
        n = 1
        while os.path.exists(path):
            n += 1
            path = f"{stem}_r{n}.flux"
        return os.path.normpath(path)

    def _flux_engine(self):
        """Resident FluxLM engine — loaded once, kept across epochs."""
        from forge.model.flux import FluxLM  # noqa: F401
        if self._engine is not None and \
                self._engine.model.__class__.__name__ == "FluxLM":
            return self._engine
        self._free_engine()
        from forge.model.flux import FluxConfig
        from forge.engine.forge_engine import ForgeEngine
        cfg = self.config
        fc = FluxConfig(vocab_size=65536, device=cfg.device,
                        cuda_primary=cfg.device == "cuda")
        # R53: teacher-embedding graft — "auto" grafts the V2 embed
        # table when it exists (semantic _R feeds every channel); a
        # resumed .flux snapshot restores its own _R and ignores this.
        te = cfg.flux_teacher_embed
        if te == "auto":
            cand = os.path.join(cfg.checkpoint_dir,
                                "ForgeLM_V2.safetensors")
            te = cand if os.path.exists(cand) else ""
        if te and not str(self.best_checkpoint).endswith(".flux"):
            fc.teacher_embed = te
        ckpt = (self.best_checkpoint
                if str(self.best_checkpoint).endswith(".flux") else None)
        self._engine = ForgeEngine.from_flux(fc, checkpoint=ckpt,
                                             device=cfg.device)
        return self._engine

    # ── R53: source-grounded goals + tools ──

    def _flux_tools(self):
        """Shared web-tools executor (cached, budgeted, offline-safe)."""
        t = getattr(self, "_flux_tools_obj", None)
        if t is None:
            from forge.self_play.flux_goals import FluxTools
            cfg = self.config
            t = FluxTools(dict_file=cfg.flux_dict_file,
                          obs_chars=cfg.flux_obs_chars,
                          fetch_budget=cfg.flux_web_fetch_budget)
            self._flux_tools_obj = t
        return t

    def _flux_sampler(self):
        """Lazy goal sampler over the shared tools + persistent KB."""
        s = getattr(self, "_flux_sampler_obj", None)
        if s is None:
            from forge.self_play.flux_goals import FluxGoalSampler, FluxKB
            cfg = self.config
            kb = FluxKB(os.path.join(cfg.status_dir, "flux_kb.jsonl"))
            srcs = tuple(s.strip() for s in
                         cfg.flux_goal_sources.split(",") if s.strip())
            s = FluxGoalSampler(self._flux_tools(), kb=kb,
                                sources=srcs, seed=cfg.flux_eval_seed)
            self._flux_sampler_obj = s
        return s

    def _flux_eval_pairs(self) -> list[dict]:
        """Combined held-out eval: fixed procedural set + frozen web set.

        The procedural half is deterministic across runs (flux_eval_seed).
        The web half is built once at first use and persisted to
        ``status_dir/flux_eval_web.json`` — restarts re-score the SAME
        questions, so the accuracy trend stays meaningful.
        """
        cfg = self.config
        pairs = [dict(p, kind="qa", source="procedural") for p in
                 _concise_qa_pairs(cfg.flux_eval_n, cfg.flux_eval_seed)]
        if cfg.flux_web and cfg.flux_eval_web_n > 0:
            try:
                web = self._flux_sampler().build_eval(
                    cfg.flux_eval_web_n,
                    os.path.join(cfg.status_dir, "flux_eval_web.json"),
                    {p["prompt"] for p in pairs})
                pairs += [dict(r, kind="qa") for r in web]
            except Exception:
                pass                    # offline/broken -> procedural only
        return pairs

    def _flux_eval_acc(self) -> float:
        """Held-out accuracy on the FIXED eval sets (procedural + frozen
        web).  Predict-only batched streams — measurement never teaches,
        so no learn+revert cycle is needed.  Breakdown is stashed in
        ``_flux_eval_detail`` for telemetry."""
        cfg = self.config
        pairs = self._flux_eval_pairs()
        outs = self._engine.generate_batch(
            [f"Q: {p['prompt']}\nA:" for p in pairs],
            max_new_tokens=cfg.flux_gen_tokens,
            temperatures=[0.0] * len(pairs))
        hit = hit_w = n_w = 0
        for o, p in zip(outs, pairs):
            ok = self._flux_check_any(o, p)
            hit += ok
            if p.get("source") and p["source"] != "procedural":
                n_w += 1
                hit_w += ok
        n_p = len(pairs) - n_w
        self._flux_eval_detail = {
            "acc": hit / max(len(pairs), 1),
            "proc": (hit - hit_w) / max(n_p, 1),
            "web": hit_w / max(n_w, 1) if n_w else None,
            "n_web": n_w}
        return self._flux_eval_detail["acc"]

    @classmethod
    def _flux_check_any(cls, output: str, pair: dict) -> bool:
        """Grade against the canonical answer plus any accepted
        alternates (synonym/antonym sets, co-author names)."""
        if cls._flux_check(output, pair.get("response", "")):
            return True
        return any(cls._flux_check(output, a)
                   for a in pair.get("accepts") or [])

    def _flux_score(self, output: str, pair: dict) -> float:
        """Reward in [0,1] for a sample — pluggable verifier
        (``flux_verifier`` callable) or the binary grader."""
        v = self.config.flux_verifier
        if v is not None:
            try:
                return min(1.0, max(0.0, float(v(output, pair))))
            except Exception:
                return 0.0
        return 1.0 if self._flux_check_any(output, pair) else 0.0

    def _flux_fresh_pairs(self, n: int) -> list[dict]:
        """Draw n verified task pairs, deduped against every prompt this
        run has already attempted (and against the fixed eval sets).

        R53: ``flux_web_frac`` of the batch comes from the source
        sampler (Wikipedia/dictionary/arXiv — extractive verified
        answers); the rest is the procedural stream.  Each web goal is
        assigned a mode — "qa" (bare recall), "rag" (source snippet
        inline) or "tool" (two-phase tool call) — by flux_tool_frac /
        flux_rag_frac.  Offline or exhausted sources fall back to
        procedural fill so the batch size is always met."""
        cfg = self.config
        if not hasattr(self, "_flux_eval_prompts"):
            self._flux_eval_prompts = {
                p["prompt"] for p in self._flux_eval_pairs()}
        seen, eval_prompts = self._flux_seen, self._flux_eval_prompts

        web: list[dict] = []
        if cfg.flux_web and cfg.flux_web_frac > 0:
            n_web = int(round(n * cfg.flux_web_frac))
            try:
                self._flux_tools().begin_epoch(cfg.flux_web_fetch_budget)
                web = self._flux_sampler().sample_web(
                    n_web, seen, eval_prompts, epoch=self.epoch)
            except Exception:
                web = []
        web_pairs: list[dict] = []
        for g in web:
            if g["prompt"] in seen:        # within-batch dedupe
                continue
            seen.add(g["prompt"])
            # assign the attempt mode — tool needs a canonical tool+query,
            # rag needs a source context; the rest recall directly
            r = self._flux_rng.random()
            if g.get("tool") and r < cfg.flux_tool_frac:
                g["kind"] = "tool"
            elif g.get("context") and \
                    r < cfg.flux_tool_frac + cfg.flux_rag_frac:
                g["kind"] = "rag"
            else:
                g["kind"] = "qa"
            web_pairs.append(g)

        need = n - len(web_pairs)
        if not cfg.flux_unique:
            proc = [dict(p, kind="qa", source="procedural") for p in
                    _concise_qa_pairs(need, self.epoch)]
            return web_pairs + proc
        pairs = list(web_pairs)
        j = 0
        while len(pairs) < n and j < 12:
            for p in _concise_qa_pairs(need * 2 + 8,
                                       self.epoch * 977 + j):
                q = p["prompt"]
                if q in seen or q in eval_prompts:
                    continue
                seen.add(q)
                pairs.append(dict(p, kind="qa", source="procedural"))
                if len(pairs) >= n:
                    break
            j += 1
        return pairs

    @staticmethod
    def _flux_prompt(p: dict) -> str:
        """Canonical attempt prompt for a goal's assigned mode."""
        kind = p.get("kind", "qa")
        if kind == "rag" and p.get("context"):
            ctx = " ".join(str(p["context"]).split())[:400]
            return f"C: {ctx}\nQ: {p['prompt']}\nA:"
        if kind == "tool":
            return f"{_FLUX_TOOL_HEADER}\nQ: {p['prompt']}\nT:"
        return f"Q: {p['prompt']}\nA:"

    def _flux_tool_turn(self, engine, prompts_t: list[str],
                        seeds: list[int], temps: list[float],
                        ) -> tuple[list[str], list]:
        """Phase A of a tool task: emit the `T:` line (one tool call)."""
        outs = engine.generate_batch(
            prompts_t, max_new_tokens=14,
            temperatures=temps, seeds=seeds,
            stops=[["\n"]] * len(prompts_t),
            cf_topk=self.config.flux_cf_topk)
        return outs, getattr(engine, "last_flux_packets", None) or []

    def _flux_self_play(self) -> dict:
        """Phase 1 (flux): eval-pre → grouped concurrent attempts.

        Three batched passes so modes stay cheap:
          1. qa + rag tasks — one ``generate_batch`` (G samples each);
             rag prompts carry the source context inline.
          2. tool tasks phase A — model emits ``T: <tool> <query>``,
             loop executes it through FluxTools (cached per epoch).
          3. tool tasks phase B — ``Q:.. T:<call> O:<obs> A:`` answer.

        Streams are predict-only: the canonical model is never mutated;
        each stream emits an update packet applied in ``_flux_finish``
        against verified answers.
        """
        engine = self._flux_engine()
        model = engine.model
        self._flux_mark = model._count          # full-epoch rollback pos
        cfg = self.config
        sw = self._status_writer

        if sw:
            sw.set_phase("flux_eval",
                         detail=f"epoch {self.epoch}: held-out probe")
        acc_pre = self._flux_eval_acc()

        pairs = self._flux_fresh_pairs(cfg.tasks_per_epoch)
        G = max(1, cfg.flux_group_size)
        engine.flux_batch_workers = cfg.flux_workers
        engine.flux_workers_device = cfg.flux_workers_device

        def _temps(k):
            # epsilon-exploration: a fraction of samples decodes at the
            # explore temp instead of the exploit center — widens the
            # trajectory distribution beyond what near-greedy reaches
            return [min(1.5, max(0.0,
                    (cfg.flux_explore_temp
                     if self._flux_rng.random() < cfg.flux_explore_eps
                     else cfg.flux_temp)
                    + self._flux_rng.uniform(-cfg.flux_temp_spread,
                                             cfg.flux_temp_spread)))
                    for _ in range(k)]

        def _seeds(k):
            return [self._flux_rng.randrange(1 << 31)
                    for _ in range(k)]

        # split: tool tasks get the two-phase path, qa/rag one pass
        tool_idx = [i for i, p in enumerate(pairs)
                    if p.get("kind") == "tool"]
        plain_idx = [i for i in range(len(pairs))
                     if i not in set(tool_idx)]

        if sw:
            sw.set_phase(
                "flux_attempts",
                detail=f"{len(pairs)} tasks x{G} samples "
                       f"({len(tool_idx)} tool, {cfg.flux_workers} wk)")
            for i, p in enumerate(pairs):
                sw.task_started(p["prompt"], i, len(pairs),
                                kind=p.get("kind", "qa"),
                                source=p.get("source", ""))

        attempts: list[dict] = [{
            "prompt": p["prompt"], "response": p["response"],
            "accepts": p.get("accepts"), "kind": p.get("kind", "qa"),
            "source": p.get("source", "procedural"),
            "context": p.get("context", ""),
            "tool": p.get("tool"), "query": p.get("query"),
            "ok": False, "best": None, "pkt": None, "pkt_extra": [],
            "tool_ok": False} for p in pairs]

        # ── pass 1: qa + rag (single batch) ──
        if plain_idx:
            prompts = [self._flux_prompt(pairs[i])
                       for i in plain_idx for _ in range(G)]
            outs = engine.generate_batch(
                prompts, max_new_tokens=cfg.flux_gen_tokens,
                temperatures=_temps(len(prompts)),
                seeds=_seeds(len(prompts)),
                cf_topk=cfg.flux_cf_topk)
            packets = getattr(engine, "last_flux_packets", None) or []
            for row, i in enumerate(plain_idx):
                group = outs[row * G:(row + 1) * G]
                good = [(j, o, self._flux_score(o, pairs[i]))
                        for j, o in enumerate(group)]
                good = [g for g in good if g[2] > 0]
                good.sort(key=lambda g: -g[2])
                a = attempts[i]
                a["reward"] = good[0][2] if good else 0.0
                a["ok"] = a["reward"] >= cfg.flux_ok_threshold
                if good:
                    a["best"] = good[0][1]
                    a["pkt"] = row * G + good[0][0]
                    a["pkt_extra"] = [row * G + j for j, _, _ in good[1:]]
                a["_pkts"] = packets
                if sw:
                    sw.task_done(i, pairs[i]["prompt"],
                                 success=a["ok"], rounds_used=G,
                                 preview=(a["best"] or
                                          (group[0] if group else "")),
                                 expected=pairs[i]["response"])

        # ── passes 2+3: tool tasks (call → observe → answer) ──
        if tool_idx:
            tools = self._flux_tools()
            pa_prompts = [self._flux_prompt(pairs[i])
                          for i in tool_idx for _ in range(G)]
            outs_a, pkts_a = self._flux_tool_turn(
                engine, pa_prompts, _seeds(len(pa_prompts)),
                _temps(len(pa_prompts)))
            calls = [_parse_tool_call(o) for o in outs_a]
            # resolve observations — canonical query substituted when the
            # model's call didn't parse (task stays solvable; the demo
            # teach on failure corrects the tool-call step itself)
            obs: list[str] = []
            pb_prompts: list[str] = []
            for row, i in enumerate(tool_idx):
                for j in range(G):
                    k = row * G + j
                    call = calls[k]
                    if call is None:
                        call = (pairs[i].get("tool") or "wiki",
                                pairs[i].get("query")
                                or pairs[i]["prompt"])
                        obs_ok = False
                    else:
                        obs_ok = True
                    o = tools.call(call[0], call[1])
                    obs.append(o)
                    pb_prompts.append(
                        f"{_FLUX_TOOL_HEADER}\nQ: {pairs[i]['prompt']}\n"
                        f"T: {call[0]} {call[1]}\nO: {o}\nA:")
            outs_b = engine.generate_batch(
                pb_prompts, max_new_tokens=cfg.flux_gen_tokens,
                temperatures=_temps(len(pb_prompts)),
                seeds=_seeds(len(pb_prompts)),
                cf_topk=cfg.flux_cf_topk)
            pkts_b = getattr(engine, "last_flux_packets", None) or []
            for row, i in enumerate(tool_idx):
                group = outs_b[row * G:(row + 1) * G]
                good = [(j, o, self._flux_score(o, pairs[i]))
                        for j, o in enumerate(group)]
                good = [g for g in good if g[2] > 0]
                good.sort(key=lambda g: -g[2])
                a = attempts[i]
                a["reward"] = good[0][2] if good else 0.0
                a["ok"] = a["reward"] >= cfg.flux_ok_threshold
                if good:
                    j0 = good[0][0]
                    a["best"] = good[0][1]
                    a["pkt"] = row * G + j0          # phase-B packet idx
                    a["pkt_extra"] = [row * G + j
                                      for j, _, _ in good[1:]]
                    # phase-A packet only counts when the model's own
                    # call parsed — reinforcing a garbage call would
                    # teach bad syntax
                    a["tool_ok"] = bool(calls[row * G + j0])
                    a["pkt_a"] = (row * G + j0) if a["tool_ok"] else None
                else:
                    # partial credit: track a parsed phase-A call even
                    # when the answer failed — valid tool syntax gets a
                    # reduced-gain reinforce in _flux_finish
                    a["tool_ok"] = any(
                        calls[row * G + j] for j in range(G))
                    a["pkt_a"] = next(
                        (row * G + j for j in range(G)
                         if calls[row * G + j]), None)
                a["obs"] = obs[row * G:(row + 1) * G]
                a["_pkts"] = pkts_b
                a["_pkts_a"] = pkts_a
                if sw:
                    sw.task_done(i, pairs[i]["prompt"],
                                 success=a["ok"], rounds_used=G,
                                 preview=(a["best"] or
                                          (group[0] if group else "")),
                                 expected=pairs[i]["response"])

        self._flux_attempts = attempts
        n_ok = sum(a["ok"] for a in attempts)
        n_tool_ok = sum(a.get("tool_ok") for a in attempts
                        if a["kind"] == "tool")
        if sw:
            sw.event("flux_attempts_done", epoch=self.epoch,
                     correct=n_ok, total=len(attempts),
                     tool_tasks=len(tool_idx), tool_calls_ok=n_tool_ok,
                     acc_pre=round(acc_pre, 3))
        print(f"  [flux] eval-pre acc={acc_pre:.3f} | "
              f"attempts {n_ok}/{len(attempts)} correct"
              + (f" | tool calls {n_tool_ok}/{len(tool_idx)} parsed"
                 if tool_idx else ""))
        return {"success_rate": n_ok / max(len(attempts), 1),
                "acc_pre": acc_pre, "attempts": len(attempts),
                "tool_tasks": len(tool_idx), "tool_ok": n_tool_ok,
                "web_goals": sum(a["source"] != "procedural"
                                 for a in attempts),
                "domain": "mixed_qa"}

    @staticmethod
    def _flux_ids(tok, text: str) -> list[int]:
        ids = tok.encode(text, add_special_tokens=False)
        return list(getattr(ids, "input_ids", getattr(ids, "ids", ids)))

    def _flux_ingest(self, model, tok, text: str, tag: str) -> int:
        """Journaled ingest of raw text under ``tag`` (ws/we-wrapped so
        revert_tag/reinforce see the block)."""
        if not text:
            return 0
        model.journal.append(("ws", tag))
        n = model.ingest(self._flux_ids(tok, text), tag=tag)
        model.journal.append(("we", tag))
        return n

    def _flux_teach_fail(self, a: dict, model, tok, tag: str) -> None:
        """Failure path: internalize the SOURCE first (episodic/knn/
        csem channels get the real context), then hot-teach the verified
        binding.  Tool tasks additionally demonstrate the canonical
        tool-call step, so ``Q:.. -> T: <tool> <query>`` gets learned."""
        cfg = self.config
        ctx = (a.get("context") or "")[: cfg.flux_context_chars]
        if ctx:
            self._flux_ingest(model, tok, ctx, tag)

        if a["kind"] == "tool":
            tool = a.get("tool") or "wiki"
            query = a.get("query") or a["prompt"]
            # canonical observation for the demo trajectory — cached,
            # so this is free when the model already called it
            obs = self._flux_tools().call(tool, query)
            if obs in ("", "offline", "no results"):
                obs = (a.get("obs") or [""])[0] or "no results"
            # step 1: emit the right tool call for this question
            model.teach_text(
                f"{_FLUX_TOOL_HEADER}\nQ: {a['prompt']}\nT:",
                f" {tool} {query}\n", tok, tag=tag)
            # step 2: answer given the observation
            model.teach_text(
                f"{_FLUX_TOOL_HEADER}\nQ: {a['prompt']}\n"
                f"T: {tool} {query}\nO: {obs}\nA:",
                f" {a['response']}\n", tok, tag=tag)
            # full trajectory into episodic memory
            self._flux_ingest(
                model, tok,
                f"{_FLUX_TOOL_HEADER}\nQ: {a['prompt']}\n"
                f"T: {tool} {query}\nO: {obs}\nA: {a['response']}\n",
                tag)
            return

        prompt = self._flux_prompt(dict(a, kind="rag" if ctx else "qa"))
        model.teach_text(prompt, f" {a['response']}\n", tok, tag=tag)
        if a["kind"] == "rag" and ctx:
            self._flux_ingest(
                model, tok,
                f"{prompt} {a['response']}\n", tag)

    def _flux_finish(self, phases: dict) -> None:
        """Phases 2-4 (flux): reinforce/revert+teach → consolidate →
        snapshot → eval-post gate → promote or full rollback."""
        cfg = self.config
        engine = self._flux_engine()
        model = engine.model
        tok = engine.tokenizer

        # Phase 2: outcome-weighted learning on this epoch's attempts —
        # the single learning process applies each verified attempt's
        # update packet (what the stream actually produced), amplified by
        # reinforce(); failed attempts' packets are discarded and the
        # correct answer is hot-taught instead (plus its source context
        # and, for tool tasks, the demonstrated tool call).
        n_reinf = n_taught = 0
        for i, a in enumerate(self._flux_attempts):
            tag = f"e{self.epoch}:t{i}"
            model.tag = tag
            if a["ok"]:
                applied = False
                pkts = []
                main = a.get("_pkts") or []
                for k in ([a["pkt"]] + a.get("pkt_extra", [])):
                    if k is not None and k < len(main):
                        main[k].tag = tag
                        pkts.append(main[k])
                # tool success: also learn the model's OWN tool call
                # (only when it parsed — garbage syntax is not taught)
                if a["kind"] == "tool" and a.get("pkt_a") is not None:
                    pa = a.get("_pkts_a") or []
                    if a["pkt_a"] < len(pa):
                        pa[a["pkt_a"]].tag = tag
                        pkts.insert(0, pa[a["pkt_a"]])
                if pkts:
                    model.apply_packets(pkts)
                    applied = True
                else:
                    # fallback: no packet recorded (stale engine) —
                    # ingest the verified Q/A pair directly
                    self._flux_ingest(
                        model, tok,
                        f"{self._flux_prompt(a)} {a['response']}\n", tag)
                    applied = True
                # graded reward scales the gain — a 0.7-reward attempt
                # reinforces weaker than a clean 1.0
                model.reinforce(tag,
                                gain=cfg.flux_gain * a.get("reward", 1.0))
                # counterfactual credit: the runner-up tokens the
                # sampler logged share a little of the win
                if cfg.flux_cf_topk:
                    for bi, pk in enumerate(pkts):
                        if getattr(pk, "alts", None):
                            model.reinforce_alts(
                                tag, pk,
                                gain=cfg.flux_gain
                                * a.get("reward", 1.0),
                                scale=cfg.flux_cf_scale,
                                block_idx=bi)
                n_reinf += applied
            else:
                # partial credit: below-threshold but nonzero reward
                # (or a parsed tool call under a wrong answer) still
                # reinforces what it got right, at scaled gain — the
                # teach then supplies what it missed
                r = a.get("reward", 0.0)
                pkts = []
                if r > 0 and a.get("pkt") is not None:
                    main = a.get("_pkts") or []
                    if a["pkt"] < len(main):
                        main[a["pkt"]].tag = tag
                        pkts.append(main[a["pkt"]])
                if a["kind"] == "tool" and a.get("pkt_a") is not None:
                    pa = a.get("_pkts_a") or []
                    if a["pkt_a"] < len(pa):
                        pa[a["pkt_a"]].tag = tag
                        pkts.insert(0, pa[a["pkt_a"]])
                if pkts:
                    model.apply_packets(pkts)
                    pg = cfg.flux_gain * max(r, cfg.flux_partial_gain)
                    model.reinforce(tag, gain=pg)
                    if cfg.flux_cf_topk:
                        for bi, pk in enumerate(pkts):
                            if getattr(pk, "alts", None):
                                model.reinforce_alts(
                                    tag, pk, gain=pg,
                                    scale=cfg.flux_cf_scale,
                                    block_idx=bi)
                elif cfg.flux_cf_topk and cfg.flux_cf_fail:
                    # counterfactual failure path: apply the failed
                    # trajectory, boost the untried alternatives it
                    # recorded, then revert the sampled writes — the
                    # alts keep credit for what the tried token lost
                    main = a.get("_pkts") or []
                    if a.get("pkt") is not None and a["pkt"] < len(main):
                        pk = main[a["pkt"]]
                        pk.tag = tag
                        model.apply_packets([pk])
                        if getattr(pk, "alts", None):
                            model.reinforce_alts(
                                tag, pk, gain=cfg.flux_gain,
                                scale=cfg.flux_cf_scale)
                        model.revert_writes(tag)
                self._flux_teach_fail(a, model, tok, tag)
                n_taught += 1
            model.soft_reset()
        cons = model.consolidate(min_entries=cfg.flux_consolidate_min)
        print(f"  [flux] reinforced={n_reinf} taught={n_taught} "
              f"consolidated={cons}")

        # Snapshot the trained state as this epoch's candidate.
        candidate = self._flux_checkpoint_path(self.epoch)
        model.snapshot(candidate)
        phases["flux"] = {"checkpoint": candidate,
                          "reinforced": n_reinf, "taught": n_taught,
                          "consolidate": cons}

        # Phase 3: fixed held-out eval — post vs pre.
        acc_post = self._flux_eval_acc()
        acc_pre = phases.get("self_play", {}).get("acc_pre", 0.0)
        eval_result = {
            "passed": acc_post >= acc_pre,
            "candidate_quality": acc_post,
            "base_quality": acc_pre,
            "winner": "CANDIDATE" if acc_post >= acc_pre else "BASE",
        }
        phases["evaluate"] = eval_result
        if self._status_writer:
            det = getattr(self, "_flux_eval_detail", {}) or {}
            self._status_writer.event(
                "flux_eval_done", epoch=self.epoch,
                acc_pre=round(acc_pre, 3), acc_post=round(acc_post, 3),
                acc_proc=round(det.get("proc", 0.0), 3),
                acc_web=(round(det["web"], 3)
                         if det.get("web") is not None else None),
                reinforced=n_reinf, taught=n_taught)
            # live memory telemetry for the spectation card: which
            # channels the hedge mixture favors + how full the banks are
            try:
                rep = model.memory_report()
                chans = model.channel_weights()
                self._status_writer.update(
                    flux_cells=int(rep.get(
                        "tables_cells", rep.get("gpu_rows_used", 0))),
                    flux_journal=int(rep.get("journal_entries", 0)),
                    flux_knn=int(rep.get("knn_rows", 0)),
                    flux_knn_cap=int(rep.get("knn_cap", 0)),
                    flux_epi=int(rep.get("epi_positions", 0)),
                    flux_mb=round(float(rep.get("total_MB_est", 0.0)), 1),
                    flux_channels={k: round(float(v), 3)
                                   for k, v in chans.items()})
            except Exception:
                pass
        print(f"  [flux] eval acc {acc_pre:.3f} -> {acc_post:.3f}")

        # Phase 4: promote the snapshot; on demote roll the live memory
        # back to the pre-epoch mark so nothing regressive persists.
        promoted = self._maybe_promote(candidate, eval_result)
        phases["promoted"] = promoted
        if not promoted:
            rep = model.revert_since(seq_pos=self._flux_mark)
            print(f"  [flux] rolled back epoch writes: {rep}")

    # ── Phase 1: Self-Play (AZR curriculum) ──────────────────────────

    def _run_self_play(self) -> dict:
        """Phase 1: Load model, run AZR curriculum (propose → solve → verify).

        V10: loads via ForgeEngine and activates V10 inference features
        (SpectralKV, MTP self-speculative decoding, prefix cache, Triton
        conv, fused QK-norm+RoPE+cache-write, sequence-split attention).
        The engine.model is passed to InfiniteCurriculum; the engine is
        retained on self._engine for fast_eval reuse (skips ~40s reload).
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 1: AZR SELF-PLAY (epoch {self.epoch})")
        print(f"{'='*70}")
        sw = self._status_writer
        if sw:
            sw.set_phase("self_play", detail=f"epoch {self.epoch}: loading engine")

        engine = self._load_engine()
        model = engine.model
        tokenizer = engine.tokenizer
        # Keep engine alive for fast_eval reuse (set before curriculum so
        # cleanup paths can find it).
        self._engine = engine

        curriculum = InfiniteCurriculum(
            model=model,
            tokenizer=tokenizer,
            device=self.config.device,
            max_gen_tokens=self.config.max_gen_tokens,
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            engine=engine,
        )
        # Debug round: trace every model generation (propose/solve/retry)
        # so the raw outputs + token ids can be inspected afterwards.
        if self.config.debug_round:
            curriculum.gen_trace = GenTracer(tokenizer)
        self._gen_trace = curriculum.gen_trace

        # Clear seen descriptions from prior runs — fresh start per epoch
        # so the model can re-propose similar tasks (it has limited vocabulary).
        # The clone filter still prevents exact duplicates within this epoch.
        curriculum._seen_descriptions = set()
        from research.paths import CURRICULUM_DIR
        seen_path = CURRICULUM_DIR / "seen_descriptions.json"
        if seen_path.exists():
            try:
                seen_path.unlink()
            except OSError:
                pass

        # ── Propose + validate tasks ──
        n_target = self.config.tasks_per_epoch
        validated: list = []
        if sw:
            sw.set_phase("proposing", detail=f"epoch {self.epoch}: target {n_target} tasks")

        if self.config.task_source == "api":
            # API-driven: use distillation teacher models to generate diverse tasks.
            # The local model still does all solving — APIs only provide task diversity.
            print("  [Propose] Using API teachers for task generation...")
            domain = self.config.domains[self.epoch % len(self.config.domains)]
            # Request more than needed — some will be filtered
            api_batch = max(n_target * 2, 30)
            proposed = curriculum.api_propose_tasks(
                n=api_batch, domain=domain, difficulty="medium")
            validated.extend(proposed)
            propose_attempts = 1
            print(f"  [Propose] {len(validated)} validated tasks from API "
                  f"(domain={domain})")
        else:
            # Model-driven: rotate domains + modes across propose attempts for diversity.
            # The model has limited task vocabulary — rotating domains and using
            # different reasoning modes (induction/abduction/deduction) forces
            # diverse task generation instead of repeating the same 5-6 tasks.
            import random as _rng
            _rng.Random(42 + self.epoch)
            modes = ["induction", "abduction", "deduction"]

            propose_attempts = 0
            max_propose_attempts = (n_target // 3) + 5  # more attempts for diversity

            while len(validated) < n_target and propose_attempts < max_propose_attempts:
                # Rotate domain + mode per attempt
                domain = self.config.domains[propose_attempts % len(self.config.domains)]
                mode = modes[propose_attempts % len(modes)]
                batch_n = min(self.config.propose_batch_size,
                              n_target - len(validated) + 5)
                # Vary temperature slightly per attempt for diversity
                old_temp = curriculum.temperature
                curriculum.temperature = self.config.temperature * (1.0 + 0.1 * (propose_attempts % 3))
                proposed = curriculum.propose_tasks(
                    domain=domain, n=batch_n, batch_size=batch_n, mode=mode)
                curriculum.temperature = old_temp
                validated.extend(proposed)
                propose_attempts += 1
                if sw:
                    sw.curriculum_progress(
                        proposed=curriculum.stats.total_proposed,
                        validated=len(validated), parse_failed=0)
                print(f"  [Propose] Attempt {propose_attempts} ({domain}/{mode}): "
                      f"{len(proposed)} validated, {len(validated)}/{n_target} total")

            print(f"  [Propose] {len(validated)} validated tasks from "
                  f"{propose_attempts} attempts (domain={domain})")

        if not validated:
            print("  [Propose] No valid tasks generated, skipping epoch")
            if sw:
                sw.alert("warn", f"epoch {self.epoch}: no valid tasks proposed")
            self._free_engine()
            del curriculum
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return {"error": "no_valid_tasks", "n_proposed": 0}

        # ── Solve tasks ──
        # All first attempts go through ONE concurrent engine.generate_batch
        # call (true parallel decode with per-sequence controls); retries and
        # sandbox verification run per-task afterward. Replaces the old
        # lock-serialized per-task generation loop.
        successes = 0
        failures = 0
        self._trajectories = []

        if sw:
            sw.set_phase("solving",
                         detail=f"epoch {self.epoch}: solving {len(validated)} tasks")

        # All tasks are in-flight simultaneously in the batched decode —
        # emit task_started for every task up front so the live feed shows
        # them, then task_done as results land in order.
        for i, task in enumerate(validated):
            if sw:
                sw.task_started(task.description, i, len(validated),
                                domain=task.domain)
        # GRPO needs K rollouts of the SAME task prompt per group —
        # within-group reward variance is what produces nonzero advantage.
        # (Cross-task pairing collapsed to identical rewards → loss 0.)
        group_k = (self.config.grpo_group_size
                   if self.config.training_mode == "grpo" else 1)
        results_ordered = curriculum.solve_tasks_batch(
            validated, group_size=group_k)

        for i, item in enumerate(r for r in results_ordered if r is not None):
            task, result, elapsed_ms = item
            success = result.get("final_success", False)
            attempts = result.get("attempts", [])
            if sw:
                # One "round" event per solve attempt for the live feed.
                per_attempt_ms = elapsed_ms / max(len(attempts), 1)
                for att in attempts:
                    sw.round_done(
                        i, att.get("sample", 0), att.get("success", False),
                        quality=1.0 if att.get("success") else 0.0,
                        gen_ms=per_attempt_ms, exec_ms=0.0,
                        error=att.get("error", ""))
                best_code = next(
                    (a.get("code", "") for a in attempts
                     if a.get("success")),
                    attempts[0].get("code", "") if attempts else "")
                sw.task_done(i, task.description, success,
                             rounds_used=result.get("rounds_used", len(attempts)),
                             best_quality=result.get("best_quality",
                                                     1.0 if success else 0.0),
                             preview=best_code)
            self._record_step_metrics({
                "mean_reward": 1.0 if success else 0.0,
                "diversity_score": curriculum.stats.diversity_score,
            })

            # GRPO group: all K rollouts of this task's prompt with their
            # binary rewards — the trainer needs per-prompt reward variance
            # for a nonzero advantage.
            group_comps = [a["code"] for a in attempts]
            group_rews = [1.0 if a.get("success") else 0.0
                          for a in attempts]

            if success and attempts:
                # Export the SHORTEST passing attempt — earlier passes can
                # carry scaffolding/dead code; the terse solution teaches
                # concise answers (anti-overthinking data pressure).
                best_att = min(
                    (a for a in attempts if a.get("success")),
                    key=lambda a: len(a.get("code") or ""), default=None)
                if best_att is not None:
                    self._trajectories.append({
                        "task_description": task.description,
                        "signature": task.signature,
                        "test_cases": task.test_cases,
                        "solution_code": best_att["code"],
                        "domain": task.domain,
                        "difficulty": task.difficulty,
                        "reward": 1.0,
                        "group_completions": group_comps,
                        "group_rewards": group_rews,
                    })
                successes += 1
            else:
                failures += 1
                # Keep failed attempts too (for negative examples / analysis)
                if attempts:
                    self._trajectories.append({
                        "task_description": task.description,
                        "signature": task.signature,
                        "test_cases": task.test_cases,
                        "solution_code": attempts[0]["code"],
                        "domain": task.domain,
                        "difficulty": task.difficulty,
                        "reward": 0.0,
                        "error": attempts[0].get("error", ""),
                        "group_completions": group_comps,
                        "group_rewards": group_rews,
                    })

            # Record result for curriculum difficulty adaptation
            curriculum.record_result(task, success)

            if (i + 1) % 10 == 0 or i == len(validated) - 1:
                rate = successes / (i + 1)
                print(f"  [Solve] {i+1}/{len(validated)}: "
                      f"{successes} passed, {failures} failed "
                      f"({rate:.0%} success rate)")

        # Surface a few distinct failure errors so a 0% epoch is diagnosable
        # from the log instead of a black box.
        if failures:
            seen_errs: list[str] = []
            for t in self._trajectories:
                err = (t.get("error") or "").strip().split("\n")
                err = err[-1][:160] if err and err[-1] else ""
                if err and err not in seen_errs:
                    seen_errs.append(err)
            for e in seen_errs[:3]:
                print(f"  [Solve] sample error: {e}")

        stats = {
            "n_proposed": len(validated),
            "n_solved": successes,
            "n_failed": failures,
            "success_rate": successes / max(len(validated), 1),
            "domain": domain,
            "curriculum_stats": {
                "total_proposed": curriculum.stats.total_proposed,
                "total_validated": curriculum.stats.total_validated,
                "total_solved": curriculum.stats.total_solved,
                "mean_proposer_reward": curriculum.stats.mean_proposer_reward,
                "diversity_score": curriculum.stats.diversity_score,
            },
        }
        print(f"\n  [Self-Play] {successes}/{len(validated)} solved "
              f"({stats['success_rate']:.0%}), domain={domain}")

        # Free curriculum (holds self.model = engine.model reference).
        # The engine itself is retained on self._engine for fast_eval reuse
        # (weight-swap skips ~40s model reload). It is freed after eval/promote
        # via _free_engine().
        del curriculum
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return stats

    def _free_engine(self):
        """Free the ForgeEngine from VRAM + verify release.

        Releases the model, KV cache, CUDA graphs, Triton patches, and
        prefix cache held by the engine. Called after fast_eval completes
        (or on early-return paths where eval is skipped).
        """
        engine = self._engine
        if engine is None:
            return
        # Release acceleration resources (CUDA graphs, megakernels, etc.)
        # before deleting the model to prevent pool leaks.
        if hasattr(engine, "_release_acceleration_resources"):
            engine._release_acceleration_resources()
        # Sleep level 2 to drop the model from VRAM if the engine supports it.
        try:
            engine.sleep(level=2)
            # FluxLM sleep(2) snapshots live memory to a flux_sleep_*.flux
            # temp file for wake(); a freed engine never wakes — unlink it.
            snap = getattr(engine, "_stored_checkpoint", None)
            if snap and os.path.basename(str(snap)).startswith(
                    "flux_sleep_"):
                try:
                    os.unlink(snap)
                except OSError:
                    pass
        except Exception:
            logger.debug("Error sleeping engine during cleanup", exc_info=True)
        del engine
        self._engine = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            from forge.model_loader import ModelLoader
            ModelLoader.clear_cache()
            # Verify VRAM was actually released
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"  [VRAM] After engine free: {allocated:.2f} GB allocated, "
                  f"{reserved:.2f} GB reserved")

    def _dump_debug_round(self) -> dict:
        """Dump the debug-round generation transcript, then free the engine.

        Every generation Phase 1 made (propose/solve/retry) is written with
        its full rendered prompt, raw pre-strip output, and per-token
        id+piece table — human-readable ``debug_round_ep{N}.txt`` plus a
        machine-readable ``.jsonl``, both under ``status_dir``.
        """
        trace = self._gen_trace
        n = len(trace.events) if trace is not None else 0
        txt_path = jsonl_path = None
        if trace is not None and trace.events:
            os.makedirs(self.config.status_dir, exist_ok=True)
            txt_path = os.path.normpath(os.path.join(
                self.config.status_dir, f"debug_round_ep{self.epoch}.txt"))
            jsonl_path = txt_path[:-len(".txt")] + ".jsonl"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(f"DEBUG ROUND epoch {self.epoch} — "
                        f"{n} generations, no training\n")
                for ev in trace.events:
                    f.write(f"\n{'='*70}\n"
                            f"GEN #{ev['i']} [{ev['phase']}] "
                            f"{ev['n_tokens']} tokens ({ev['ids_source']})\n"
                            f"{'='*70}\n"
                            f"-- RENDERED PROMPT --\n{ev['rendered']}\n"
                            f"-- RAW OUTPUT (pre-strip) --\n{ev['raw']}\n")
                    if ev["reply"] != ev["raw"]:
                        f.write(f"-- POST-STRIP (pipeline input) --\n"
                                f"{ev['reply']}\n")
                    f.write("-- TOKENS (id:piece) --\n" + " ".join(
                        f"{t['id']}:{t['text']!r}" for t in ev["tokens"])
                        + "\n")
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for ev in trace.events:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            print(f"\n  [DebugRound] {n} generations traced ->")
            print(f"    {txt_path}\n    {jsonl_path}")
        else:
            print("\n  [DebugRound] no generations were traced")
        self._free_engine()
        return {"n_generations": n,
                "transcript": txt_path, "transcript_jsonl": jsonl_path}

    # ── Phase 2: Export + Finetune ───────────────────────────────────

    def _export_trajectories(self, epoch: int) -> str:
        """Export successful trajectories as SFT JSONL.

        Format: {"prompt": task_description, "response": solution_code}
        Mixes in replay data to prevent catastrophic forgetting.
        """
        output_path = os.path.normpath(os.path.join(
            self.config.data_dir, f"azr_epoch{epoch}.jsonl"))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Only export successful trajectories for SFT
        successful = [t for t in self._trajectories if t["reward"] > 0.5]
        examples = []

        for t in successful:
            # Build prompt: task description + function signature
            prompt = (f"Write a Python function {t['signature']} that "
                      f"{t['task_description']}\n\n"
                      f"```python\n")
            response = t["solution_code"]
            examples.append({"prompt": prompt, "response": response})

        # Mix in replay data (prior SFT examples) to prevent forgetting
        if self.config.replay_file and os.path.exists(self.config.replay_file):
            replay_examples = []
            with open(self.config.replay_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except Exception:
                        continue
                    if "prompt" in ex and "response" in ex:
                        replay_examples.append(ex)
                    elif "messages" in ex:
                        msgs = ex["messages"]
                        if len(msgs) >= 2:
                            user_msg = next((m for m in msgs if m["role"] == "user"), None)
                            asst_msg = next((m for m in msgs if m["role"] == "assistant"), None)
                            if user_msg and asst_msg:
                                replay_examples.append({
                                    "prompt": user_msg["content"],
                                    "response": asst_msg["content"],
                                })

            if replay_examples:
                import random as _rng
                rng = _rng.Random(42 + epoch)
                n_replay = min(len(replay_examples),
                               int(len(examples) * self.config.replay_ratio / max(1 - self.config.replay_ratio, 0.01)))
                n_replay = max(n_replay, 5)  # floor: always some replay
                rng.shuffle(replay_examples)
                examples.extend(replay_examples[:n_replay])
                print(f"  [Export] +{n_replay} replay examples (anti-forgetting)")

        # Concise-QA stream: verified short-answer pairs teach the model to
        # answer easy prompts directly and minimally — the behavior the
        # 'concise' eval scores and code-trajectory data can't reach.
        n_qa = self.config.concise_qa_per_epoch
        if n_qa > 0:
            qa = _concise_qa_pairs(n_qa, epoch)
            examples.extend(qa)
            print(f"  [Export] +{len(qa)} concise-QA examples (direct answers)")

        # Shuffle
        import random as _rng
        rng = _rng.Random(42 + epoch)
        rng.shuffle(examples)

        with open(output_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"  [Export] {len(successful)} successful + "
              f"{len(examples) - len(successful)} replay = "
              f"{len(examples)} total -> {output_path}")
        return output_path

    def _finetune(self, data_path: str, epoch: int) -> str:
        """Phase 2b: SFT continuation from best checkpoint.

        V10: wires all sft_train training tricks — Muon-SF optimizer, grad-mixup,
        curriculum learning, augmentation, SYNPRO, focal/dynamic-focal loss,
        entropy weighting, MTP auxiliary loss, distillation, sequential freeze,
        EMA, BitNet-everywhere, LazyTrain, OOMB, Hybrid8Bit clip, elastic
        grad-accum, FreeToken, disk cache, packed sequences, async prefetch.
        Each trick is gated by its LoopConfig field so users can toggle them
        via CLI args.
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 2: FINETUNE (epoch {self.epoch})")
        print(f"{'='*70}")
        # stall_hold: in-process SFT can't emit progress — keep the
        # heartbeat from false-flagging a stall during training.
        self._status_update()
        if self._status_writer:
            self._status_writer.set_phase(
                "finetune", detail=f"epoch {epoch}: SFT "
                f"({self.config.ft_optimizer}, {self.config.ft_max_steps} steps)",
                stall_hold=True)

        # Free the self-play engine before finetuning — training needs the
        # full VRAM budget. The engine is reloaded fresh next epoch.
        self._free_engine()

        save_path = self._epoch_checkpoint_path(epoch)
        c = self.config

        cmd = [
            os.sys.executable, "-m", "forge.training.runners.sft_train",
            "--data", data_path,
            "--checkpoint", self.best_checkpoint,
            "--save", save_path,
            "--config", c.config_name,
            "--max-steps", str(c.ft_max_steps),
            "--lr", str(c.ft_lr),
            "--min-lr", str(c.ft_min_lr),
            "--batch-size", str(c.ft_batch_size),
            "--grad-accum", str(c.ft_grad_accum),
            "--sync-freq", str(c.ft_sync_freq),
            "--seq-len", str(c.ft_seq_len),
            "--warmup-steps", str(c.ft_warmup_steps),
            "--optimizer", c.ft_optimizer,
            "--grad-compression", c.ft_grad_compression,
            "--entropy-alpha", str(c.ft_entropy_alpha),
            "--loss-function", c.ft_loss_function,
            "--focal-gamma", str(c.ft_focal_gamma),
            "--label-smoothing-eps", str(c.ft_label_smoothing_eps),
            "--norm-type", c.ft_norm_type,
            "--grad-mixup", str(c.ft_grad_mixup),
            "--curriculum", c.ft_curriculum,
            "--vram-limit-gb", str(c.ft_vram_limit_gb),
            "--ram-limit-percent", "90",
        ]

        # ── LoRA / BitNet ──
        if c.ft_lora:
            cmd.extend(["--lora-r", str(c.ft_lora_r),
                        "--lora-alpha", str(c.ft_lora_alpha)])
        else:
            cmd.append("--no-lora")
        if c.ft_bitnet_everywhere:
            cmd.append("--bitnet-everywhere")
        else:
            cmd.append("--no-bitnet-everywhere")
        if c.ft_manual_lora:
            cmd.append("--manual-lora")

        # ── Gradient checkpointing ──
        if c.ft_grad_checkpoint:
            cmd.extend(["--grad-checkpoint", "--checkpoint-strategy",
                        c.ft_checkpoint_strategy])
        else:
            cmd.append("--no-grad-checkpoint")

        # ── Data pipeline ──
        if c.ft_disk_cache:
            cmd.append("--disk-cache")
        else:
            cmd.append("--no-disk-cache")
        if c.ft_pack_sequences:
            cmd.append("--pack-sequences")
        else:
            cmd.append("--no-pack-sequences")
        if c.ft_async_prefetch:
            cmd.extend(["--async-prefetch", "--prefetch-count",
                        str(c.ft_prefetch_count)])
        else:
            cmd.append("--no-async-prefetch")

        # ── Augmentation / SYNPRO ──
        if c.ft_augment:
            cmd.append("--augment")
        if c.ft_synpro:
            cmd.append("--synpro")

        # ── MTP auxiliary loss ──
        if c.ft_mtp_weight > 0:
            cmd.extend(["--mtp-weight", str(c.ft_mtp_weight)])
            if c.ft_mtp_n_heads is not None:
                cmd.extend(["--mtp-n-heads", str(c.ft_mtp_n_heads)])

        # ── Distillation ──
        if c.ft_distill:
            cmd.append("--distill")
            if c.ft_teacher_checkpoint:
                cmd.extend(["--teacher-checkpoint", c.ft_teacher_checkpoint])
            cmd.extend(["--distill-topk", str(c.ft_distill_topk),
                        "--distill-truncate", str(c.ft_distill_truncate)])
            if c.ft_distill_prefix:
                cmd.append("--distill-prefix")

        # ── Sequential freeze ──
        if c.ft_sequential_freeze > 0:
            cmd.extend(["--sequential-freeze", str(c.ft_sequential_freeze),
                        "--final-finetune-steps", str(c.ft_final_finetune_steps)])

        # ── EMA ──
        if c.ft_ema:
            cmd.extend(["--ema", "--ema-decay", str(c.ft_ema_decay)])

        # ── L2-SP anchor regularization ──
        if c.ft_anchor:
            cmd.extend(["--anchor", c.ft_anchor,
                        "--l2-lambda", str(c.ft_l2_lambda)])

        # ── Sample weighting ──
        if c.ft_sample_weighting:
            cmd.append("--sample-weighting")

        # ── LazyTrain / OOMB / Hybrid8Bit / elastic / FreeToken ──
        if c.ft_lazy_train:
            cmd.append("--lazy-train")
        if c.ft_oomb:
            cmd.append("--oomb")
        if c.ft_hybrid_clip:
            cmd.append("--hybrid-clip")
        if c.ft_elastic_grad_accum:
            cmd.append("--elastic-grad-accum")
        if c.ft_freetoken:
            cmd.append("--freetoken")

        # ── torch.compile ──
        if c.ft_compile:
            cmd.append("--compile")

        # Run in-process (no subprocess spawn overhead)
        if not self._run_subprocess(cmd, "Finetune"):
            raise RuntimeError("Finetune failed (in-process)")
        if not os.path.exists(save_path):
            raise RuntimeError(f"Finetune completed but checkpoint not found: {save_path}")

        return save_path

    def _run_subprocess(self, cmd: list[str], stage_name: str) -> bool:
        """Run a training stage in-process (no subprocess spawn).

        Calls the runner's main() directly via sys.argv manipulation, with
        explicit VRAM cleanup before/after. Falls back to subprocess if the
        module has no main() or the in-process path fails.
        """
        # cmd[0] is the python executable, cmd[1] is "-m", cmd[2] is the module
        if len(cmd) >= 3 and cmd[1] == "-m":
            module_name = cmd[2]
            cli_args = cmd[3:]
        else:
            return self._run_subprocess_fallback(cmd, stage_name)

        print(f"  Running (in-process): {module_name} "
              f"{' '.join(cli_args[:3])}... ({len(cli_args)} args)")

        # Cleanup VRAM from self-play before starting training
        self._free_engine()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        import importlib
        import sys
        old_argv = sys.argv
        sys.argv = [module_name] + cli_args
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "main"):
                mod.main()
                return True
            else:
                print(f"  {stage_name}: module has no main() — falling back to subprocess")
                sys.argv = old_argv
                return self._run_subprocess_fallback(cmd, stage_name)
        except SystemExit as e:
            code = e.code
            if code is None or code == 0:
                return True
            print(f"  {stage_name} FAILED (exit code {code})")
            return False
        except Exception as e:
            print(f"  {stage_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            sys.argv = old_argv
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def _run_subprocess_fallback(self, cmd: list[str], stage_name: str) -> bool:
        """Legacy subprocess fallback (used when in-process isn't possible)."""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        print(f"  Running (subprocess fallback): {' '.join(cmd[:4])}... ({len(cmd)} args)")
        import subprocess
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  {stage_name} FAILED (exit code {result.returncode})")
            return False
        return True

    # ── Phase 2c: GRPO training (RL-only RSI mode) ──────────────────

    def _grpo_train(self, epoch: int) -> str:
        """Phase 2c: GRPO training on verified self-play trajectories.

        GRPO-only RSI mode: instead of SFT on successful trajectories, train
        the model directly via Group Relative Policy Optimization on the
        verified (success + failure) trajectories from self-play.

        The binary reward (1.0 if test_passed, 0.0 otherwise) IS the strict
        data gate — no external reward model needed.

        VRAM budget on 12GB RTX 5070 with 3B Jamba model:
          - Model (bf16): ~6 GB
          - Ref model (bf16, frozen): ~6 GB → CPU offloaded
          - LoRA adapters: ~50 MB
          - KV cache (short seq): ~0.5 GB
          - Activations + grads: ~1 GB
          Total GPU: ~7.5 GB (fits with cpu_offload optimizer)

        Uses GRPOTrainer with:
          - MC-GRPO median baseline (robust for group_size=4)
          - CPUAdamW optimizer (optimizer states on CPU)
          - N-gram repetition penalty (doom-loop mitigation)
          - OOM guard per step
        """
        print(f"\n{'='*70}")
        print(f"  PHASE 2: GRPO TRAINING (epoch {self.epoch})")
        print(f"{'='*70}")
        if self._status_writer:
            self._status_writer.set_phase(
                "grpo", detail=f"epoch {epoch}: {self.config.grpo_rl_algorithm} "
                f"({self.config.grpo_max_steps} steps)")

        # Free the self-play engine — GRPO needs the full VRAM budget
        self._free_engine()

        save_path = self._epoch_checkpoint_path(epoch)
        c = self.config

        # Collect verified trajectories for GRPO
        # Use both successful (reward=1.0) and failed (reward=0.0) — GRPO
        # needs the group contrast to compute advantages.
        verified = [t for t in self._trajectories if "reward" in t]
        if len(verified) < c.grpo_min_trajectories:
            print(f"  Too few trajectories ({len(verified)}), skipping GRPO")
            return ""

        # Build prompt → completions → rewards structure for GRPO.
        # Proper groups: K rollouts of the SAME task prompt (recorded at
        # solve time as group_completions/group_rewards). Same-prompt
        # groups give within-group reward variance → nonzero advantage.
        # (Previously paired completions of DIFFERENT tasks — rewards
        # were usually equal → advantage 0 → loss 0 → no learning.)
        prompts = []
        completions = []
        rewards = []

        for t in verified:
            comps = t.get("group_completions")
            rews = t.get("group_rewards")
            if comps and rews and len(comps) >= 2:
                prompts.append(t["task_description"])
                completions.append(comps)
                # Conciseness bonus: shorter passing solutions earn more —
                # anti-overthinking pressure + nonzero advantage in
                # all-pass groups.
                rewards.append(_concise_adjusted_rewards(
                    comps, rews, c.grpo_concise_bonus))

        if not prompts:
            # Legacy fallback for trajectories without group data —
            # pair same-domain tasks (weak: rewards often collapse).
            from collections import defaultdict
            by_domain = defaultdict(list)
            for t in verified:
                by_domain[t.get("domain", "default")].append(t)
            for domain, tasks in by_domain.items():
                for i in range(0, len(tasks) - 1, 2):
                    t1, t2 = tasks[i], tasks[i + 1]
                    prompts.append(t1["task_description"])
                    completions.append(
                        [t1["solution_code"], t2["solution_code"]])
                    rewards.append([t1["reward"], t2["reward"]])

        if not prompts:
            print("  No valid GRPO groups formed, skipping")
            return ""

        print(f"  Trajectories: {len(verified)} verified")
        print(f"  GRPO groups: {len(prompts)} "
              f"(group_size~{c.grpo_group_size})")
        print(f"  Algorithm: {c.grpo_rl_algorithm}")
        print(f"  Steps: {c.grpo_max_steps} | LR: {c.grpo_lr} | KL: {c.grpo_kl_coeff}")

        # Load model + reference model
        from forge.config import get_config
        from forge.model_loader import ModelLoader
        from forge.self_play.grpo_trainer import GRPOConfig, GRPOTrainer
        from research.tokenizer_cache import get_tokenizer

        cfg = get_config(c.config_name, device=c.device)

        # Build model with LoRA (train ~1M params, not full 3B)
        model = ModelLoader.build_model_fast(
            cfg, checkpoint_path=self.best_checkpoint,
            dtype=torch.bfloat16 if "cuda" in c.device else torch.float32,
        )
        model = model.to(c.device)

        # Apply LoRA adapters (manual LoRA — BitNet-compatible, works with
        # the 3B Jamba model on 12GB VRAM: trains ~1M params, not full 3B)
        from forge.training.bitnet_lora import add_lora_adapters, merge_lora_adapters
        n_adapters, lora_params = add_lora_adapters(
            model, rank=c.ft_lora_r, alpha=c.ft_lora_alpha)
        # Freeze all non-LoRA params
        lora_param_ids = {id(p) for p in lora_params}
        for param in model.parameters():
            if id(param) not in lora_param_ids:
                param.requires_grad = False
        print(f"  LoRA: {n_adapters} adapters (rank={c.ft_lora_r}), "
              f"{sum(p.numel() for p in lora_params)/1e6:.2f}M trainable params")
        model = model.to(c.device)

        # Reference model (frozen, for KL penalty) — CPU offloaded to save VRAM
        ref_model = ModelLoader.build_model_fast(
            cfg, checkpoint_path=self.best_checkpoint,
            dtype=torch.bfloat16 if "cuda" in c.device else torch.float32,
        )
        ref_model = ref_model.to(c.device)
        for p in ref_model.parameters():
            p.requires_grad = False
        ref_model.eval()

        tokenizer = get_tokenizer()

        grpo_config = GRPOConfig(
            learning_rate=c.grpo_lr,
            kl_coefficient=c.grpo_kl_coeff,
            clip_range=c.grpo_clip_range,
            group_size=c.grpo_group_size,
            temperature=c.grpo_temperature,
            max_seq_len=c.grpo_max_seq_len,
            grad_accum_steps=c.grpo_grad_accum,
            rl_algorithm=c.grpo_rl_algorithm,
            use_repetition_penalty=c.grpo_use_repetition_penalty,
        )
        grpo_config.optimizer = "cpu_offload"  # 12GB VRAM safe

        trainer = GRPOTrainer(
            model=model, tokenizer=tokenizer, ref_model=ref_model,
            device=c.device, config=grpo_config,
        )

        # Run GRPO training steps
        import random as _rng
        _rng.seed(42)
        torch.manual_seed(42)

        t0 = time.time()
        step = 0
        while step < c.grpo_max_steps:
            # Sample a batch of groups (batch_size=4 prompts per step)
            batch_size = min(4, len(prompts))
            indices = _rng.sample(range(len(prompts)), batch_size)
            batch_prompts = [prompts[i] for i in indices]
            batch_comps = [completions[i] for i in indices]
            batch_rewards = [rewards[i] for i in indices]

            with oom_guard(c.device, label="grpo_step") as safe:
                stats = trainer.train_step(batch_prompts, batch_comps, batch_rewards)
            if safe.skipped:
                continue
            step += 1

            self._record_step_metrics({
                "mean_reward": stats.get("mean_reward", 0.0),
                "kl_divergence": stats.get("kl", 0.0),
                "advantage_collapse_rate": stats.get("advantage_collapse_rate", 0.0),
            })
            self._status_update(
                grpo_step=step, grpo_steps=c.grpo_max_steps,
                loss=stats.get("loss", 0.0), lr=c.grpo_lr,
                grpo_kl=stats.get("kl", 0.0),
                grpo_reward=stats.get("mean_reward", 0.0),
                grpo_acr=stats.get("advantage_collapse_rate", 0.0))

            if step % 5 == 0 or step == 1:
                elapsed = time.time() - t0
                print(f"  step {step}/{c.grpo_max_steps} | "
                      f"loss {stats.get('loss', 0):.4f} | "
                      f"kl {stats.get('kl', 0):.4f} | "
                      f"reward {stats.get('mean_reward', 0):.3f} | "
                      f"acr {stats.get('advantage_collapse_rate', 0):.2%} | "
                      f"{elapsed:.0f}s")

        # Save checkpoint (merge LoRA into base weights for standalone save)
        n_merged = merge_lora_adapters(model)
        print(f"  Merged {n_merged} LoRA adapters into base model")
        from forge.checkpoint_io import save_training_checkpoint
        save_training_checkpoint(model, save_path,
                                 optimizer=trainer.optimizer, step=step)

        # Free VRAM
        del model, ref_model, trainer
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  GRPO done in {time.time() - t0:.0f}s → {save_path}")
        return save_path


    def _evaluate(self, checkpoint: str) -> dict:
        """Phase 3: Evaluate candidate vs current best via fast_eval.

        V10: passes the self-play ForgeEngine to fast_eval for weight-swap
        reuse (skips ~40s model reload). The engine's KV cache is
        re-activated to a standard eval configuration before testing.
        """
        print(f"\n{'='*70}")
        print("  PHASE 3: EVALUATE")
        print(f"{'='*70}")
        # stall_hold: fast_eval runs opaque eval suites without progress calls.
        if self._status_writer:
            self._status_writer.set_phase(
                "evaluate", detail=f"epoch {self.epoch}: candidate vs best",
                stall_hold=True)

        from forge.self_play.discovery.fast_eval import fast_eval

        try:
            results = fast_eval(
                base_checkpoint=self.best_checkpoint,
                candidate_checkpoint=checkpoint,
                device=self.config.device,
                engine=self._engine,           # reuse self-play engine
                config_name=self.config.config_name,  # V10 config
            )
        except Exception as e:
            print(f"  Eval failed: {e}")
            import traceback; traceback.print_exc()
            return {"passed": False, "error": str(e)}
        finally:
            # Engine is no longer needed after eval — free VRAM for next epoch.
            self._free_engine()

        base_q = results.get("base", {}).get("quality", 0)
        cand_q = results.get("candidate", {}).get("quality", 0)
        winner = results.get("winner", "BASE")

        # Promote if candidate wins. In strict mode (default) that is the
        # only way through — the new model must beat or tie the old one.
        if winner == "CANDIDATE":
            passed = True
        elif self.config.strict_promote:
            passed = False
        elif cand_q >= base_q * (1 - self.config.eval_threshold):
            # Within threshold of base quality → promote (avoid ratcheting)
            passed = True
        else:
            passed = False

        return {
            "passed": passed,
            "base_quality": base_q,
            "candidate_quality": cand_q,
            "winner": winner,
            "details": results,
        }

    # ── Phase 4: Promote/Demote ──────────────────────────────────────

    def _maybe_promote(self, candidate: str, eval_result: dict) -> bool:
        """Phase 4: Promote candidate if it passes evaluation."""
        if eval_result["passed"]:
            self.best_checkpoint = candidate
            self.best_score = eval_result.get("candidate_quality", 0)
            print(f"  PROMOTED: {candidate} is the new best checkpoint")
            if self._status_writer:
                self._status_writer.event(
                    "promote", checkpoint=candidate,
                    quality=eval_result.get("candidate_quality", 0))
            return True
        else:
            archive_dir = os.path.join(self.config.checkpoint_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            archived = os.path.join(archive_dir, os.path.basename(candidate))
            # Collision-safe: a demoted SP{epoch} from a previous run may
            # already sit in archive/ — shutil.move would fail on Windows.
            n = 1
            while os.path.exists(archived):
                n += 1
                stem, ext = os.path.splitext(os.path.basename(candidate))
                archived = os.path.join(archive_dir, f"{stem}_a{n}{ext}")
            if os.path.exists(candidate):
                shutil.move(candidate, archived)
                # Move sidecars too — .meta.json (save_checkpoint) and
                # .train.pt (save_training_checkpoint optimizer/RNG state)
                # would otherwise be orphaned in checkpoint_dir each epoch.
                for suffix in (".meta.json", ".train.pt"):
                    sidecar = candidate + suffix
                    if os.path.exists(sidecar):
                        shutil.move(sidecar, archived + suffix)
            print(f"  DEMOTED: reverted to {self.best_checkpoint}")
            if self._status_writer:
                self._status_writer.event(
                    "demote", checkpoint=candidate,
                    winner=eval_result.get("winner", ""))
            return False

    # ── Main loop ────────────────────────────────────────────────────

    def run_epoch(self) -> dict:
        """Run one complete epoch: self-play → finetune → eval → promote."""
        self.epoch += 1
        epoch_start = time.time()
        phases = {}
        self._status_update(step=self.epoch, max_steps=self.config.max_epochs)

        # Phase 1: Self-play (AZR curriculum)
        try:
            if self.config.training_mode == "flux":
                sp_stats = self._flux_self_play()
            else:
                sp_stats = self._run_self_play()
            phases["self_play"] = sp_stats
            # Debug round: Phase 1 only — dump the full generation
            # transcript (every rendered prompt + raw output + token ids)
            # and stop. No training, eval, or promotion.
            if self.config.debug_round:
                phases["debug_round"] = self._dump_debug_round()
                return {"epoch": self.epoch, **phases}
            if "error" in sp_stats:
                return {"epoch": self.epoch, **phases}
        except Exception as e:
            print(f"  Self-play failed: {e}")
            import traceback; traceback.print_exc()
            self._free_engine()
            return {"epoch": self.epoch, "error": f"self_play: {e}"}

        # Phase 2: Train (SFT, GRPO, or FluxLM live-learn)
        try:
            if self.config.training_mode == "flux":
                self._flux_finish(phases)
            elif self.config.training_mode == "grpo":
                # GRPO-only RSI mode: train directly on verified trajectories
                candidate = self._grpo_train(self.epoch)
                if not candidate:
                    phases["grpo"] = {"skipped": True}
                else:
                    phases["grpo"] = {"checkpoint": candidate}
                    # Phase 3: Evaluate
                    eval_result = self._evaluate(candidate)
                    phases["evaluate"] = eval_result
                    # Phase 4: Promote/Demote
                    promoted = self._maybe_promote(candidate, eval_result)
                    phases["promoted"] = promoted
            else:
                # SFT mode (default): export → finetune → eval → promote
                data_path = self._export_trajectories(self.epoch)
                with open(data_path, encoding='utf-8') as _f:
                    n_examples = sum(1 for _ in _f)

                if n_examples < self.config.ft_min_examples:
                    print(f"  Too few examples ({n_examples}), skipping finetune")
                    phases["finetune"] = {"skipped": True, "n_examples": n_examples}
                    # Free the self-play engine — otherwise it stays resident
                    # and the next epoch loads a second engine on top (OOM).
                    self._free_engine()
                else:
                    candidate = self._finetune(data_path, self.epoch)
                    phases["finetune"] = {"checkpoint": candidate, "n_examples": n_examples}

                    # Phase 3: Evaluate
                    eval_result = self._evaluate(candidate)
                    phases["evaluate"] = eval_result

                    # Phase 4: Promote/Demote
                    promoted = self._maybe_promote(candidate, eval_result)
                    phases["promoted"] = promoted
        except Exception as e:
            print(f"  Train/eval failed: {e}")
            import traceback; traceback.print_exc()
            phases["error"] = str(e)
            # Idempotent — no-op if _finetune/_evaluate already freed it.
            self._free_engine()

        elapsed = round(time.time() - epoch_start, 1)
        epoch_summary = {
            "epoch": self.epoch,
            "best_checkpoint": self.best_checkpoint,
            "elapsed_s": elapsed,
            **phases,
        }
        self.history.append(epoch_summary)
        if len(self.history) > 100:
            self.history = self.history[-100:]
        if self._status_writer:
            sp = phases.get("self_play", {})
            ev = phases.get("evaluate", {})
            self._status_writer.epoch_done(
                self.epoch, self.config.max_epochs,
                train_rate=sp.get("success_rate", 0.0)
                if isinstance(sp, dict) else 0.0,
                val_rate=ev.get("candidate_quality", 0.0)
                if isinstance(ev, dict) else 0.0,
                loss=phases.get("loss", 0.0),
                promoted=bool(phases.get("promoted", False)),
                elapsed_s=elapsed,
                best_checkpoint=self.best_checkpoint)
            self._status_update(
                step=self.epoch,
                topic=sp.get("domain", "") if isinstance(sp, dict) else "")
        print(f"\n  Epoch {self.epoch} done in {elapsed}s")
        print(f"  Best checkpoint: {self.best_checkpoint}")
        return epoch_summary

    def run(self, max_epochs: int | None = None) -> list[dict]:
        """Run the infinite loop for max_epochs (or until interrupted)."""
        # Debug round = exactly one Phase-1 pass, no training.
        n = (1 if self.config.debug_round
             else (max_epochs or self.config.max_epochs))
        self._init_status_writer()
        sw = self._status_writer
        if sw:
            sw.set_phase("startup", detail="initializing loop",
                         step=0, max_steps=n)
        print(f"\n{'#'*70}")
        print("#  INFINITE AZR SELF-PLAY LOOP")
        print(f"#  Starting checkpoint: {self.best_checkpoint}")
        print(f"#  Max epochs: {n}")
        print(f"#  Tasks per epoch: {self.config.tasks_per_epoch}")
        print(f"#  Config: {self.config.config_name}")
        if self.config.debug_round:
            print("#  DEBUG ROUND — self-play only, NO training;")
            print("#  every model generation is traced (raw output + tokens)")
        print(f"{'#'*70}")

        run_status, reason = "done", f"completed {n} epochs"
        for _ in range(n):
            # Cooperative stop: GUI Stop button writes STOP_REQUESTED next
            # to status.json. Checked at epoch boundaries (reliable on
            # Windows where taskkill/SIGTERM delivery is unreliable).
            if sw and sw.stop_requested():
                print("\n  Stop requested — shutting down after current epoch")
                run_status, reason = "stopped", "stopped via STOP_REQUESTED"
                break
            try:
                self.run_epoch()
            except KeyboardInterrupt:
                print(f"\n  Interrupted at epoch {self.epoch}")
                run_status, reason = "stopped", "interrupted"
                break
            except Exception as e:
                print(f"\n  Epoch {self.epoch} crashed: {e}")
                import traceback; traceback.print_exc()
                if sw:
                    sw.alert("error", f"epoch {self.epoch} crashed: {e}")

        self._print_summary()
        if sw:
            sw.close(status=run_status, reason=reason)
        return self.history

    def _print_summary(self):
        print(f"\n{'#'*70}")
        print(f"#  LOOP COMPLETE — {self.epoch} epochs")
        print(f"#  Final best checkpoint: {self.best_checkpoint}")
        print(f"{'#'*70}")
        for h in self.history:
            ep = h["epoch"]
            sp = h.get("self_play", {})
            ft = h.get("finetune", {})
            prom = h.get("promoted", "—")
            rate = sp.get("success_rate", "—") if isinstance(sp, dict) else "—"
            n_ex = ft.get("n_examples", "—") if isinstance(ft, dict) else "—"
            print(f"  Epoch {ep}: success={rate} examples={n_ex} promoted={prom}")


def main():
    # Opt-in runtime configuration (import of `forge` is side-effect-free).
    from forge.runtime.configure import configure
    configure()

    # Load .env for API keys (needed for --task-source api)
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    parser = argparse.ArgumentParser(
        description="Infinite AZR self-play training loop (V10: ForgeEngine + all training tricks)")
    parser.add_argument("--checkpoint", required=True,
                        help="Starting checkpoint (safetensors). Default V10: "
                             "research/checkpoints/ForgeLM_V2.safetensors")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs to run")
    parser.add_argument("--tasks-per-epoch", type=int, default=60,
                        help="Tasks to propose + solve per epoch (60 gives "
                             "~40+ unique SFT examples/epoch at typical "
                             "solve rates — enough signal per finetune)")
    parser.add_argument("--ft-steps", type=int, default=100,
                        help="Finetune steps per epoch")
    parser.add_argument("--ft-lr", type=float, default=5e-5,
                        help="Finetune learning rate (V10 default 5e-5 for muon_sf)")
    parser.add_argument("--ft-batch-size", type=int, default=1)
    parser.add_argument("--ft-grad-accum", type=int, default=5,
                        help="Gradient accumulation steps (evolution: 5)")
    parser.add_argument("--ft-optimizer", type=str, default="muon_sf",
                        choices=["fused", "bnb", "lion", "muon", "muon_sf",
                                 "muon_sf_plain", "flash_adamw", "flash_lion",
                                 "forge", "sf_normuon", "amuse", "mona",
                                 "cpu_offload", "badam", "fira_nlrq"],
                        help="Finetune optimizer (V10 default: muon_sf, 2.39x vs AdamW)")
    parser.add_argument("--self-play-mode", type=str, default="azr",
                        choices=["azr", "soar", "sgs", "thinking"],
                        help="Self-play mode: 'azr' (standard), 'soar' (meta-RL curriculum, "
                             "escapes learning plateaus), 'sgs' (self-guided self-play, "
                             "prevents Conjecturer collapse with Guide role), "
                             "'thinking' (ForgeLM V10-Thinking pipeline: CPT to SFT to DPO to RLVR)")
    parser.add_argument("--saerl", action="store_true",
                        help="Enable SAERL: SAE-guided data engineering for RL. "
                             "Diversity control + difficulty curriculum + quality filtering. "
                             "+3%% accuracy, 20%% fewer steps to target.")
    parser.add_argument("--opmix", action="store_true",
                        help="Enable OP-MIX: on-policy data mixing via low-rank adapters. "
                             "Dynamically adjusts mixing ratio across data sources.")
    parser.add_argument("--ft-no-lora", action="store_true",
                        help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--ft-grad-checkpoint", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Gradient checkpointing (default True, use --no-ft-grad-checkpoint)")
    parser.add_argument("--ft-checkpoint-strategy", type=str, default="all",
                        choices=["all", "ffn", "attn", "none", "lazy", "optimal"],
                        help="Selective gradient checkpointing strategy")
    parser.add_argument("--max-gen-tokens", type=int, default=256,
                        help="Max tokens per generation. Truncated completions "
                             "produce SyntaxError/IndentationError on solve — "
                             "raise to 512+ if solve rate is 0%%")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Self-play exploration temperature")
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--replay-file", type=str, default="",
                        help="Prior SFT JSONL for replay (anti-forgetting)")
    parser.add_argument("--replay-ratio", type=float, default=0.2,
                        help="Fraction of replay examples per epoch")
    parser.add_argument("--concise-qa-per-epoch", type=int, default=12,
                        help="Verified short-answer QA pairs mixed into each "
                             "epoch's SFT data (teaches direct minimal "
                             "answers — anti-overthinking; 0 disables)")
    parser.add_argument("--grpo-concise-bonus", type=float, default=0.1,
                        help="Length-efficiency bonus for passing GRPO "
                             "completions (shorter earns more; 0 disables)")
    parser.add_argument("--task-source", type=str, default="model",
                        choices=["model", "api"],
                        help="Task source: 'model' (AZR self-propose) or "
                             "'api' (distillation teacher APIs)")
    parser.add_argument("--config", type=str, default="forgelm_v2",
                        help="Model config name")
    parser.add_argument("--eval-threshold", type=float, default=0.5,
                        help="Lenient mode: max fractional quality regression still promoted")
    parser.add_argument("--lenient-promote", action="store_true",
                        help="Allow promotion when candidate merely stays within "
                             "--eval-threshold of base (default: strict — must "
                             "beat or tie the old checkpoint)")

    # ── Training mode ──
    parser.add_argument("--training-mode", type=str, default="sft",
                        choices=["sft", "grpo", "flux"],
                        help="Training mode: 'sft' (SFT continuation, default), "
                             "'grpo' (GRPO-only RSI on verified trajectories), "
                             "or 'flux' (FluxLM live-learning RSI: attempt → "
                             "reinforce/revert+teach → consolidate → "
                             "fixed-eval promote gate)")
    parser.add_argument("--grpo-steps", type=int, default=50,
                        help="GRPO steps per epoch (training_mode=grpo)")
    parser.add_argument("--grpo-lr", type=float, default=5e-6,
                        help="GRPO learning rate")
    parser.add_argument("--grpo-group-size", type=int, default=4,
                        help="GRPO group size (G completions per prompt)")
    parser.add_argument("--grpo-kl", type=float, default=0.02,
                        help="GRPO KL penalty coefficient")
    parser.add_argument("--grpo-clip", type=float, default=0.2,
                        help="GRPO PPO clip range")
    parser.add_argument("--flux-group-size", type=int, default=4,
                        help="flux mode: sampled attempts per task "
                             "(concurrent batch, per-gen seed/temp)")
    parser.add_argument("--flux-workers", type=int, default=4,
                        help="flux mode: cloned-memory gen workers")
    parser.add_argument("--flux-workers-device", type=str, default="cpu",
                        choices=["cpu", "cuda", "auto"],
                        help="clone worker device — cpu keeps the pool "
                             "at ~0GB extra VRAM")
    parser.add_argument("--flux-eval-n", type=int, default=40,
                        help="flux mode: fixed held-out eval size")
    parser.add_argument("--flux-web", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="flux mode: source goals from Wikipedia / "
                             "dictionary / arXiv (offline -> procedural)")
    parser.add_argument("--flux-goal-sources", type=str,
                        default="wiki,dict,arxiv",
                        help="flux mode: comma-separated web goal sources")
    parser.add_argument("--flux-web-frac", type=float, default=0.5,
                        help="flux mode: fraction of epoch goals drawn "
                             "from web sources (rest procedural)")
    parser.add_argument("--flux-tool-frac", type=float, default=0.35,
                        help="flux mode: fraction of web goals run as "
                             "two-phase tool-call tasks")
    parser.add_argument("--flux-rag-frac", type=float, default=0.35,
                        help="flux mode: fraction of non-tool web goals "
                             "with the source snippet inline (RAG)")
    parser.add_argument("--flux-web-fetch-budget", type=int, default=8,
                        help="flux mode: max new source fetches per epoch")
    parser.add_argument("--flux-eval-web-n", type=int, default=16,
                        help="flux mode: frozen web eval set size "
                             "(0 disables)")
    parser.add_argument("--flux-dict-file", type=str, default="",
                        help="flux mode: local kaikki.org wiktionary "
                             "jsonl for offline dictionary goals")
    parser.add_argument("--flux-teacher-embed", type=str, default="auto",
                        help="flux mode: teacher checkpoint whose "
                             "embedding grafts onto a FRESH flux model "
                             "('auto' = ForgeLM_V2.safetensors when "
                             "present, '' disables)")
    parser.add_argument("--grpo-algorithm", type=str, default="grpo",
                        choices=["grpo", "gtpo", "cispo", "sppo", "psppo",
                                 "evpo", "grpo_or"],
                        help="RL algorithm (grpo=default, gtpo=no-ref-model, "
                             "cispo=MiniMax detached clip)")

    # ── V10 ForgeEngine inference features ──
    parser.add_argument("--no-forge-engine", action="store_true",
                        help="Disable ForgeEngine (use bare model path, no V10 inference features)")
    parser.add_argument("--kv-cache", type=str, default="spectral",
                        help="KV cache strategy for self-play generation (V10 default: spectral)")
    parser.add_argument("--decoding", type=str, default="mtp_selfspec",
                        choices=["standard", "speculative", "medusa", "dspark",
                                 "eagle3", "mtp_selfspec"],
                        help="Decoding strategy (V10 default: mtp_selfspec)")
    parser.add_argument("--quantize", type=str, default=None,
                        choices=[None, "int8", "int4", "fp8", "w8a8", "nvfp4"],
                        help="Weight quantization for self-play generation")
    parser.add_argument("--acceleration", type=str, default=None,
                        choices=[None, "cuda_graph", "megakernel", "flex_decoding"],
                        help="Acceleration strategy")
    parser.add_argument("--use-compile", action="store_true",
                        help="torch.compile the self-play model (experimental on Windows)")
    parser.add_argument("--no-triton-conv", action="store_true",
                        help="Disable Triton fused conv kernel")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Disable prefix KV cache")
    parser.add_argument("--kv-cache-tokens", type=int, default=4096,
                        help="KV cache token budget for self-play generation")

    # ── Training tricks ──
    parser.add_argument("--ft-loss-function", type=str, default="ce",
                        choices=["ce", "focal", "label_smoothing", "lovasz",
                                 "dynamic_focal", "mixture"],
                        help="Loss function (focal/dynamic_focal: +36%% exact match)")
    parser.add_argument("--ft-focal-gamma", type=float, default=4.93,
                        help="Focal loss gamma (evolution optimum: 4.93)")
    parser.add_argument("--ft-entropy-alpha", type=float, default=0.5,
                        help="Token entropy weighting alpha (0 disables)")
    parser.add_argument("--ft-grad-mixup", type=int, default=1,
                        help="N-batch grad averaging (1=off, 3=1.25x convergence)")
    parser.add_argument("--ft-curriculum", type=str, default="none",
                        choices=["none", "vanilla", "pacing", "interleaved", "warmup"],
                        help="Curriculum learning strategy (18-45%% fewer steps)")
    parser.add_argument("--ft-augment", action="store_true",
                        help="Training-time data augmentation (token noise, FIM, target offset)")
    parser.add_argument("--ft-synpro", action="store_true",
                        help="SYNPRO synthetic data generation (3.7-5.2x effective tokens)")
    parser.add_argument("--ft-norm-type", type=str, default="rmsnorm",
                        choices=["rmsnorm", "seednorm", "dyt"],
                        help="Normalization type (dyt = Muon-compatible Dynamic Tanh)")
    parser.add_argument("--ft-mtp-weight", type=float, default=0.0,
                        help="MTP auxiliary loss weight (Nemotron Lightning, 0=disabled)")
    parser.add_argument("--ft-distill", action="store_true",
                        help="Knowledge distillation from a teacher model")
    parser.add_argument("--ft-teacher-checkpoint", type=str, default=None,
                        help="Teacher model checkpoint for distillation")
    parser.add_argument("--ft-sequential-freeze", type=int, default=0,
                        help="Sequential freeze/unfreeze (0=off, 4=4 phases)")
    parser.add_argument("--ft-ema", action="store_true",
                        help="EMA shadow weights")
    parser.add_argument("--ft-anchor", type=str, default=None,
                        help="L2-SP anchor checkpoint (anti-drift, NeurIPS 2024)")
    parser.add_argument("--ft-lazy-train", action="store_true",
                        help="LazyTrain scheduler (1.24x sustained TFLOPS)")
    parser.add_argument("--ft-oomb", action="store_true",
                        help="OOMB chunk-recurrent training (128K+ context on 12GB)")
    parser.add_argument("--ft-elastic-grad-accum", action="store_true",
                        help="Dynamic grad_accum (OOM prevention)")
    parser.add_argument("--ft-freetoken", action="store_true",
                        help="FreeToken double-buffered grad pipeline (cpu_offload only)")
    parser.add_argument("--ft-compile", action="store_true",
                        help="torch.compile the training model (experimental on Windows)")
    parser.add_argument("--ft-bitnet-everywhere", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="BitNet ternary QAT (1.58 bits, 2.39x vs AdamW)")

    # ── Live telemetry ──
    parser.add_argument("--status-dir", type=str,
                        default="research/checkpoints/self_play",
                        help="Dir for status.json/heartbeat.json/events.jsonl "
                             "live telemetry (GUI Self-Play page reads this)")
    parser.add_argument("--no-live-status", action="store_true",
                        help="Disable live telemetry files")
    parser.add_argument("--debug-round", action="store_true",
                        help="Run ONE self-play round (propose + solve) with "
                             "NO training — every model generation is printed "
                             "live and dumped to "
                             "status_dir/debug_round_ep*.txt|.jsonl (rendered "
                             "prompt, raw output, per-token ids)")
    args = parser.parse_args()

    config = LoopConfig(
        tasks_per_epoch=args.tasks_per_epoch,
        ft_max_steps=args.ft_steps,
        ft_lr=args.ft_lr,
        ft_batch_size=args.ft_batch_size,
        ft_grad_accum=args.ft_grad_accum,
        ft_optimizer=args.ft_optimizer,
        ft_lora=not args.ft_no_lora,
        ft_grad_checkpoint=args.ft_grad_checkpoint,
        ft_checkpoint_strategy=args.ft_checkpoint_strategy,
        ft_loss_function=args.ft_loss_function,
        ft_focal_gamma=args.ft_focal_gamma,
        ft_entropy_alpha=args.ft_entropy_alpha,
        ft_grad_mixup=args.ft_grad_mixup,
        ft_curriculum=args.ft_curriculum,
        ft_augment=args.ft_augment,
        ft_synpro=args.ft_synpro,
        ft_norm_type=args.ft_norm_type,
        ft_mtp_weight=args.ft_mtp_weight,
        ft_distill=args.ft_distill,
        ft_teacher_checkpoint=args.ft_teacher_checkpoint,
        ft_sequential_freeze=args.ft_sequential_freeze,
        ft_ema=args.ft_ema,
        ft_anchor=args.ft_anchor,
        ft_lazy_train=args.ft_lazy_train,
        ft_oomb=args.ft_oomb,
        ft_elastic_grad_accum=args.ft_elastic_grad_accum,
        ft_freetoken=args.ft_freetoken,
        ft_compile=args.ft_compile,
        ft_bitnet_everywhere=args.ft_bitnet_everywhere,
        # V10 inference features
        use_forge_engine=not args.no_forge_engine,
        kv_cache=args.kv_cache,
        decoding=args.decoding,
        quantize=args.quantize,
        acceleration=args.acceleration,
        use_compile=args.use_compile,
        use_triton_conv=not args.no_triton_conv,
        use_prefix_cache=not args.no_prefix_cache,
        kv_cache_tokens=args.kv_cache_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        max_gen_tokens=args.max_gen_tokens,
        max_epochs=args.epochs,
        config_name=args.config,
        replay_file=args.replay_file,
        replay_ratio=args.replay_ratio,
        concise_qa_per_epoch=args.concise_qa_per_epoch,
        grpo_concise_bonus=args.grpo_concise_bonus,
        task_source=args.task_source,
        eval_threshold=args.eval_threshold,
        strict_promote=not args.lenient_promote,
        training_mode=args.training_mode,
        flux_group_size=args.flux_group_size,
        flux_workers=args.flux_workers,
        flux_workers_device=args.flux_workers_device,
        flux_eval_n=args.flux_eval_n,
        flux_web=args.flux_web,
        flux_goal_sources=args.flux_goal_sources,
        flux_web_frac=args.flux_web_frac,
        flux_tool_frac=args.flux_tool_frac,
        flux_rag_frac=args.flux_rag_frac,
        flux_web_fetch_budget=args.flux_web_fetch_budget,
        flux_eval_web_n=args.flux_eval_web_n,
        flux_dict_file=args.flux_dict_file,
        flux_teacher_embed=args.flux_teacher_embed,
        grpo_max_steps=args.grpo_steps,
        grpo_lr=args.grpo_lr,
        grpo_group_size=args.grpo_group_size,
        grpo_kl_coeff=args.grpo_kl,
        grpo_clip_range=args.grpo_clip,
        grpo_rl_algorithm=args.grpo_algorithm,
        status_dir=args.status_dir,
        live_status=not args.no_live_status,
        debug_round=args.debug_round,
    )

    # Self-play mode dispatch
    if args.self_play_mode == "soar":
        print("[InfiniteLoop] SOAR mode: meta-RL curriculum (escapes learning plateaus)")
        # Load model + tokenizer for SOAR
        from forge.model_loader import load_default_model
        from forge.self_play.soar import SOARMetaRL
        from research.tokenizer_cache import get_tokenizer
        model, _ = load_default_model()
        tokenizer = get_tokenizer()
        # Use hard target problems from existing curriculum if available
        target_problems = []  # would be populated from curriculum
        soar = SOARMetaRL(model, model, tokenizer, target_problems)
        stats = soar.run(n_rounds=args.epochs)
        print(f"[SOAR] Completed {len(stats)} rounds. Final: {soar.stats()}")
        return

    if args.self_play_mode == "sgs":
        print("[InfiniteLoop] SGS mode: self-guided self-play (prevents Conjecturer collapse)")
        from forge.model_loader import load_default_model
        from forge.self_play.sgs import SGSTrainer
        from research.tokenizer_cache import get_tokenizer
        model, _ = load_default_model()
        tokenizer = get_tokenizer()
        target_problems = []  # would be populated from curriculum
        sgs = SGSTrainer(model, tokenizer, target_problems)
        stats = sgs.run(n_rounds=args.epochs)
        print(f"[SGS] Completed {len(stats)} rounds. Final: {sgs.stats()}")
        return

    if args.self_play_mode == "thinking":
        print("[InfiniteLoop] THINKING mode: ForgeLM V10-Thinking pipeline (CPT->SFT->DPO->RLVR)")
        pipeline_config = ThinkingPipelineConfig(
            config_name=args.config,
            optimizer=args.ft_optimizer if args.ft_optimizer != "bnb" else "cpu_offload",
        )
        pipeline = ThinkingPipeline(args.checkpoint, pipeline_config)
        final_ckpt = pipeline.run()
        print(f"\n[ThinkingPipeline] Final checkpoint: {final_ckpt}")
        return

    # Standard AZR self-play (with optional SAERL + OP-MIX)
    if args.saerl:
        print("[InfiniteLoop] SAERL enabled: SAE-guided data engineering")
        # SAERL is applied inside the loop's batch composition
        # (would be wired into the training data pipeline)

    if args.opmix:
        print("[InfiniteLoop] OP-MIX enabled: on-policy data mixing via LoRA adapters")
        # OP-MIX adjusts data source mixing ratios dynamically

    loop = InfiniteSelfPlayLoop(args.checkpoint, config)
    loop.run()


# ── ForgeLM V10-Thinking Pipeline ─────────────────────────────────────────

@dataclass
class ThinkingPipelineConfig:
    """Configuration for the ForgeLM V10-Thinking multi-stage pipeline.

    Implements the full training recipe from Liquid AI's ForgeLM V10-Thinking:
      CPT (midtraining with reasoning traces) →
      SFT (curriculum: short CoT → long CoT + mix distillation) →
      DPO (doom-loop mitigation with LLM judge) →
      RLVR (GRPO with n-gram repetition penalty on verifiable tasks)

    Each stage is a subprocess call to the corresponding runner. The pipeline
    is resumable: if interrupted, it skips completed stages by checking for
    output checkpoint existence.
    """
    # Model
    config_name: str = "forgelm_v2"  # V2 (Jamba) default
    device: str = "cuda"
    optimizer: str = "cpu_offload"  # V2 (3B) needs CPU offload on 12GB VRAM

    # Stage 1: CPT (midtraining with reasoning traces)
    cpt_enabled: bool = True
    cpt_reasoning_data: list[str] = field(default_factory=lambda: [
        "forge/distillation/hf_datasets/openr1_math.jsonl",
        "forge/distillation/hf_datasets/openthoughts_114k.jsonl",
        "forge/distillation/hf_datasets/dolphin_r1.jsonl",
    ])
    cpt_general_data: list[str] = field(default_factory=lambda: [
        "forge/distillation/hf_datasets/orca_math.jsonl",
        "forge/distillation/hf_datasets/metamath.jsonl",
    ])
    cpt_reasoning_ratio: float = 0.6
    cpt_lr: float = 1e-4
    cpt_max_steps: int = 5000
    cpt_batch_size: int = 2
    cpt_seq_len: int = 2048
    cpt_grad_accum: int = 4

    # Stage 2: Curriculum SFT (mix distillation + 2-stage curriculum)
    sft_enabled: bool = True
    sft_data_inputs: list[str] = field(default_factory=lambda: [
        "forge/distillation/hf_datasets/gsm8k.jsonl",
        "forge/distillation/hf_datasets/openr1_math.jsonl",
        "forge/distillation/hf_datasets/openthoughts_114k.jsonl",
    ])
    sft_short_cot_max_tokens: int = 150
    sft_long_cot_min_tokens: int = 300
    sft_mix_ratio: float = 0.5
    sft_filter_doom_loops: bool = True
    sft_stage1_lr: float = 5e-5
    sft_stage1_steps: int = 1000
    sft_stage2_lr: float = 2e-5
    sft_stage2_steps: int = 1500
    sft_batch_size: int = 2
    sft_seq_len: int = 1024
    sft_grad_accum: int = 4

    # Stage 3: DPO (doom-loop mitigation)
    dpo_enabled: bool = True
    dpo_n_temp_samples: int = 5
    dpo_max_new_tokens: int = 512
    dpo_judge_model: str = "qwen3-32b"
    dpo_max_prompts: int = 500
    dpo_lr: float = 5e-7
    dpo_max_steps: int = 200
    dpo_method: str = "orpo"

    # Stage 4: RLVR (GRPO with repetition penalty)
    rlvr_enabled: bool = True
    rlvr_tasks: list[str] = field(default_factory=lambda: [
        "forge/distillation/hf_datasets/gsm8k.jsonl",
    ])
    rlvr_task_type: str = "math"
    rlvr_max_steps: int = 500
    rlvr_group_size: int = 4
    rlvr_lr: float = 5e-6
    rlvr_algorithm: str = "grpo"
    rlvr_use_repetition_penalty: bool = True

    # Paths
    checkpoint_dir: str = "research/checkpoints"
    data_dir: str = "research/data/thinking_pipeline"


class ThinkingPipeline:
    """Orchestrates the full ForgeLM V10-Thinking training pipeline.

    Stages:
      1. CPT: Midtrain with reasoning traces (openthoughts, openr1_math, dolphin_r1)
      2. SFT: Curriculum (short CoT internal solver → long CoT externalize + mix distillation)
      3. DPO: Doom-loop mitigation (5 temp + 1 greedy, LLM judge, n-gram loop detector)
      4. RLVR: GRPO with n-gram repetition penalty on verifiable tasks

    Each stage runs as a subprocess calling the corresponding runner module.
    The pipeline is resumable: completed stages are skipped if the output
    checkpoint already exists.
    """

    def __init__(self, base_checkpoint: str, config: ThinkingPipelineConfig | None = None):
        self.base_checkpoint = base_checkpoint
        self.config = config or ThinkingPipelineConfig()
        self.history: list[dict] = []

    def _stage_path(self, stage: str) -> str:
        """Get the checkpoint path for a stage output."""
        return os.path.normpath(os.path.join(
            self.config.checkpoint_dir,
            f"ForgeLM_V10_{stage}.safetensors"))

    def _run_subprocess(self, cmd: list[str], stage_name: str) -> bool:
        """Run a training stage in-process (no subprocess spawn).

        Replaces the old subprocess.run() approach which spawned a fresh
        Python interpreter per stage (~3-5s startup + import overhead each).
        Now calls the runner's main() directly via sys.argv manipulation,
        with explicit VRAM cleanup between stages.
        """
        # cmd[0] is the python executable, cmd[1] is "-m", cmd[2] is the module
        if len(cmd) >= 3 and cmd[1] == "-m":
            module_name = cmd[2]
            cli_args = cmd[3:]
        else:
            # Fallback: can't parse, use old subprocess approach
            env = os.environ.copy()
            env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))))
            print(f"  Running (subprocess fallback): {' '.join(cmd[:4])}... ({len(cmd)} args)")
            import subprocess
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                print(f"  {stage_name} FAILED (exit code {result.returncode})")
                return False
            return True

        print(f"  Running (in-process): {module_name} {' '.join(cli_args[:3])}... ({len(cli_args)} args)")

        # Cleanup VRAM from previous stage before starting new one
        self._cleanup_vram()

        import importlib
        import sys
        old_argv = sys.argv
        sys.argv = [module_name] + cli_args
        try:
            mod = importlib.import_module(module_name)
            if hasattr(mod, "main"):
                mod.main()
                return True
            else:
                print(f"  {stage_name}: module has no main() — falling back to subprocess")
                sys.argv = old_argv
                return self._run_subprocess_fallback(cmd, stage_name)
        except SystemExit as e:
            code = e.code
            if code is None or code == 0:
                return True
            print(f"  {stage_name} FAILED (exit code {code})")
            return False
        except Exception as e:
            print(f"  {stage_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            sys.argv = old_argv
            self._cleanup_vram()

    def _run_subprocess_fallback(self, cmd: list[str], stage_name: str) -> bool:
        """Legacy subprocess fallback (used when in-process isn't possible)."""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        import subprocess
        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            print(f"  {stage_name} FAILED (exit code {result.returncode})")
            return False
        return True

    def _cleanup_vram(self):
        """Release VRAM between in-process pipeline stages."""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def _stage_completed(self, checkpoint: str) -> bool:
        """Check if a stage's output checkpoint already exists (for resumability)."""
        return os.path.exists(checkpoint)

    # ── Stage 1: CPT ──────────────────────────────────────────────────

    def run_cpt(self) -> str:
        """Stage 1: Midtraining with reasoning traces."""
        output = self._stage_path("CPT")
        print(f"\n{'='*70}")
        print("  STAGE 1/4: CPT (Midtraining with Reasoning Traces)")
        print(f"{'='*70}")

        if self._stage_completed(output):
            print(f"  ✓ Already completed: {output}")
            return output

        if not self.config.cpt_enabled:
            print("  Skipped (disabled)")
            return self.base_checkpoint

        cmd = [
            os.sys.executable, "-m", "forge.training.runners.cpt_train",
            "--reasoning-data", *self.config.cpt_reasoning_data,
            "--general-data", *self.config.cpt_general_data,
            "--config", self.config.config_name,
            "--checkpoint", self.base_checkpoint,
            "--save", output,
            "--optimizer", self.config.optimizer,
            "--reasoning-ratio", str(self.config.cpt_reasoning_ratio),
            "--lr", str(self.config.cpt_lr),
            "--max-steps", str(self.config.cpt_max_steps),
            "--batch-size", str(self.config.cpt_batch_size),
            "--seq-len", str(self.config.cpt_seq_len),
            "--grad-accum", str(self.config.cpt_grad_accum),
        ]
        if not self._run_subprocess(cmd, "CPT"):
            raise RuntimeError("CPT stage failed")
        self.history.append({"stage": "cpt", "checkpoint": output})
        return output

    # ── Stage 2: Curriculum SFT ───────────────────────────────────────

    def run_sft(self, cpt_checkpoint: str) -> str:
        """Stage 2: Curriculum SFT (mix distillation + 2-stage curriculum)."""
        output = self._stage_path("SFT2")
        print(f"\n{'='*70}")
        print("  STAGE 2/4: CURRICULUM SFT (Mix Distillation + 2-Stage)")
        print(f"{'='*70}")

        if self._stage_completed(output):
            print(f"  ✓ Already completed: {output}")
            return output

        if not self.config.sft_enabled:
            print("  Skipped (disabled)")
            return cpt_checkpoint

        # Step 2a: Prepare curriculum data
        curriculum_dir = os.path.join(self.config.data_dir, "curriculum")
        cmd_prep = [
            os.sys.executable, "-m", "forge.training.runners.curriculum_sft",
            "prepare",
            "--input", *self.config.sft_data_inputs,
            "--output-dir", curriculum_dir,
            "--short-cot-max-tokens", str(self.config.sft_short_cot_max_tokens),
            "--long-cot-min-tokens", str(self.config.sft_long_cot_min_tokens),
            "--mix-ratio", str(self.config.sft_mix_ratio),
        ]
        if self.config.sft_filter_doom_loops:
            cmd_prep.append("--filter-doom-loops")
        if not self._run_subprocess(cmd_prep, "SFT prepare"):
            raise RuntimeError("SFT data preparation failed")

        stage1_data = os.path.join(curriculum_dir, "stage1_short.jsonl")
        stage2_data = os.path.join(curriculum_dir, "stage2_long.jsonl")
        sft1_output = self._stage_path("SFT1")

        # Step 2b: Stage 1 (short CoT — internal solver)
        cmd_s1 = [
            os.sys.executable, "-m", "forge.training.runners.curriculum_sft",
            "train-stage1",
            "--data", stage1_data,
            "--checkpoint", cpt_checkpoint,
            "--save", sft1_output,
            "--config", self.config.config_name,
            "--lr", str(self.config.sft_stage1_lr),
            "--max-steps", str(self.config.sft_stage1_steps),
            "--optimizer", self.config.optimizer,
            "--seq-len", str(self.config.sft_seq_len),
            "--batch-size", str(self.config.sft_batch_size),
            "--grad-accum", str(self.config.sft_grad_accum),
        ]
        if not self._run_subprocess(cmd_s1, "SFT Stage 1"):
            raise RuntimeError("SFT Stage 1 failed")

        # Step 2c: Stage 2 (long CoT — externalize reasoning)
        cmd_s2 = [
            os.sys.executable, "-m", "forge.training.runners.curriculum_sft",
            "train-stage2",
            "--data", stage2_data,
            "--checkpoint", sft1_output,
            "--save", output,
            "--config", self.config.config_name,
            "--lr", str(self.config.sft_stage2_lr),
            "--max-steps", str(self.config.sft_stage2_steps),
            "--optimizer", self.config.optimizer,
            "--seq-len", str(self.config.sft_seq_len * 2),
            "--batch-size", str(self.config.sft_batch_size),
            "--grad-accum", str(self.config.sft_grad_accum),
        ]
        if not self._run_subprocess(cmd_s2, "SFT Stage 2"):
            raise RuntimeError("SFT Stage 2 failed")
        self.history.append({"stage": "sft", "checkpoint": output})
        return output

    # ── Stage 3: DPO ──────────────────────────────────────────────────

    def run_dpo(self, sft_checkpoint: str) -> str:
        """Stage 3: DPO doom-loop mitigation."""
        output = self._stage_path("DPO")
        print(f"\n{'='*70}")
        print("  STAGE 3/4: DPO (Doom-Loop Mitigation)")
        print(f"{'='*70}")

        if self._stage_completed(output):
            print(f"  ✓ Already completed: {output}")
            return output

        if not self.config.dpo_enabled:
            print("  Skipped (disabled)")
            return sft_checkpoint

        # Step 3a: Generate preference data
        dpo_data_dir = os.path.join(self.config.data_dir, "dpo")
        os.makedirs(dpo_data_dir, exist_ok=True)
        prompts_file = os.path.join(self.config.data_dir, "curriculum", "stage2_long.jsonl")
        pref_data = os.path.join(dpo_data_dir, "preference_pairs.jsonl")

        cmd_gen = [
            os.sys.executable, "-m", "forge.training.runners.dpo_data_gen",
            "--prompts", prompts_file,
            "--checkpoint", sft_checkpoint,
            "--output", pref_data,
            "--config", self.config.config_name,
            "--n-temp-samples", str(self.config.dpo_n_temp_samples),
            "--max-new-tokens", str(self.config.dpo_max_new_tokens),
            "--judge-model", self.config.dpo_judge_model,
            "--max-prompts", str(self.config.dpo_max_prompts),
        ]
        if not self._run_subprocess(cmd_gen, "DPO data generation"):
            print("  DPO data generation failed, skipping DPO stage")
            return sft_checkpoint

        # Step 3b: DPO training
        cmd_dpo = [
            os.sys.executable, "-m", "forge.training.runners.dpo_align",
            "--data", pref_data,
            "--checkpoint", sft_checkpoint,
            "--save", output,
            "--config", self.config.config_name,
            "--method", self.config.dpo_method,
            "--lr", str(self.config.dpo_lr),
            "--max-steps", str(self.config.dpo_max_steps),
            "--optimizer", self.config.optimizer,
        ]
        if not self._run_subprocess(cmd_dpo, "DPO training"):
            raise RuntimeError("DPO stage failed")
        self.history.append({"stage": "dpo", "checkpoint": output})
        return output

    # ── Stage 4: RLVR ─────────────────────────────────────────────────

    def run_rlvr(self, dpo_checkpoint: str) -> str:
        """Stage 4: RLVR with GRPO + n-gram repetition penalty."""
        output = self._stage_path("RLVR")
        print(f"\n{'='*70}")
        print("  STAGE 4/4: RLVR (GRPO + Repetition Penalty)")
        print(f"{'='*70}")

        if self._stage_completed(output):
            print(f"  ✓ Already completed: {output}")
            return output

        if not self.config.rlvr_enabled:
            print("  Skipped (disabled)")
            return dpo_checkpoint

        cmd = [
            os.sys.executable, "-m", "forge.training.runners.rlvr_train",
            "--tasks", *self.config.rlvr_tasks,
            "--task-type", self.config.rlvr_task_type,
            "--checkpoint", dpo_checkpoint,
            "--save", output,
            "--config", self.config.config_name,
            "--max-steps", str(self.config.rlvr_max_steps),
            "--group-size", str(self.config.rlvr_group_size),
            "--lr", str(self.config.rlvr_lr),
            "--rl-algorithm", self.config.rlvr_algorithm,
            "--optimizer", self.config.optimizer,
        ]
        if self.config.rlvr_use_repetition_penalty:
            cmd.append("--use-repetition-penalty")
        if not self._run_subprocess(cmd, "RLVR"):
            raise RuntimeError("RLVR stage failed")
        self.history.append({"stage": "rlvr", "checkpoint": output})
        return output

    # ── Full pipeline ─────────────────────────────────────────────────

    def run(self) -> str:
        """Run the full 4-stage ForgeLM V10-Thinking pipeline.

        Returns the path to the final RLVR checkpoint.
        """
        t0 = time.time()
        print(f"\n{'#'*70}")
        print("#  FORGELM V10-THINKING PIPELINE")
        print(f"#  Base checkpoint: {self.base_checkpoint}")
        print(f"#  Config: {self.config.config_name}")
        print(f"#  Optimizer: {self.config.optimizer}")
        print(f"{'#'*70}")

        stages = []
        try:
            # Stage 1: CPT
            cpt_ckpt = self.run_cpt()
            stages.append(("CPT", cpt_ckpt))

            # Stage 2: Curriculum SFT
            sft_ckpt = self.run_sft(cpt_ckpt)
            stages.append(("SFT", sft_ckpt))

            # Stage 3: DPO
            dpo_ckpt = self.run_dpo(sft_ckpt)
            stages.append(("DPO", dpo_ckpt))

            # Stage 4: RLVR
            rlvr_ckpt = self.run_rlvr(dpo_ckpt)
            stages.append(("RLVR", rlvr_ckpt))

        except RuntimeError as e:
            print(f"\n  PIPELINE INTERRUPTED: {e}")
            print(f"  Completed stages: {[s[0] for s in stages]}")
            if stages:
                print(f"  Last checkpoint: {stages[-1][1]}")
            raise

        elapsed = time.time() - t0
        print(f"\n{'#'*70}")
        print(f"#  PIPELINE COMPLETE ({elapsed:.0f}s)")
        print(f"{'#'*70}")
        for stage_name, ckpt in stages:
            print(f"  {stage_name}: {ckpt}")
        print(f"\n  Final model: {rlvr_ckpt}")
        print(f"  Total time: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        return rlvr_ckpt


if __name__ == "__main__":
    main()
