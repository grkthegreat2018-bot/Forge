"""Batched decoding strategy — processes multiple requests together.

Shifts matmul from GEMV (memory-bound at batch=1) to GEMM (compute-bound
at batch>1), delivering 3-5x throughput on RTX 5070 for 1.2B models.

Supports per-sequence generation settings:
  - temperature, top_p, top_k (sampling)
  - repetition_penalty
  - seed (per-sequence torch.Generator for reproducible generation)
  - stop sequences (string-based, checked after decode)
"""
import torch
import torch.nn.functional as F

from forge.engine.decoding import DecodingStrategy
from forge.model_loader import unpack_output_with_kv


class BatchedDecoding(DecodingStrategy):
    """Decode multiple prompts in a single batched forward pass.

    Finished sequences stay in the batch (masked) so KV cache dimensions
    remain stable. Sampling happens on the full batch at once.

    Per-sequence settings (temperature, top_k, repetition_penalty, seed,
    stop) are applied independently — each sequence in the batch can have
    completely different generation parameters.
    """

    def __init__(self, eos_token_id: int = 7,
                 eos_token_ids: set[int] | None = None):
        self.eos_token_id = eos_token_id
        # Callers should pass the engine's resolved EOS set — on Jamba V2 a
        # chat turn ends on <|im_end|> (519) / <|endoftext|> (2), which the
        # legacy LFM/Qwen-only default would never stop on (batch runs to
        # max_tokens every time).
        self.eos_set = (set(eos_token_ids) if eos_token_ids
                        else {7, 151643, 151645})

    @property
    def name(self) -> str:
        return "BatchedDecoding"

    def generate(self, model, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.0,
                 top_p: float = 1.0) -> torch.Tensor:
        return self.generate_batch(
            model,
            [input_ids],
            [max_new_tokens],
            [temperature],
            [top_p],
        )[0]

    def generate_batch(
        self,
        model,
        prompts: list[torch.Tensor],  # each [1, prompt_len]
        max_tokens_list: list[int],
        temperatures: list[float],
        top_ps: list[float],
        top_k_list: list[int] | None = None,
        repetition_penalty_list: list[float] | None = None,
        seed_list: list[int | None] | None = None,
        stop_list: list[list[str] | None] | None = None,
        tokenizer=None,
        processor_list: list | None = None,
    ) -> list[torch.Tensor]:
        B = len(prompts)
        if B == 0:
            return []
        device = prompts[0].device

        # Default per-sequence settings
        if top_k_list is None:
            top_k_list = [0] * B
        if repetition_penalty_list is None:
            repetition_penalty_list = [1.0] * B
        if seed_list is None:
            seed_list = [None] * B
        if stop_list is None:
            stop_list = [None] * B
        if processor_list is None:
            processor_list = [None] * B

        # Per-sequence generators for reproducible sampling
        generators = []
        for seed in seed_list:
            if seed is not None:
                gen = torch.Generator(device=device)
                gen.manual_seed(seed)
                generators.append(gen)
            else:
                generators.append(None)

        # Pad prompts to same length (LEFT-pad: zeros on left, prompt on right)
        # Left-padding is correct for causal LM generation — the last position
        # is the generation start, consistent across all sequences in the batch.
        max_prompt_len = max(p.shape[1] for p in prompts)
        padded_ids = torch.zeros(B, max_prompt_len, dtype=torch.long, device=device)
        prompt_lens = []
        for i, p in enumerate(prompts):
            L = p.shape[1]
            padded_ids[i, -L:] = p[0]  # right-align prompt (pad on left)
            prompt_lens.append(L)

        # Attention mask — attend to the rightmost L positions (the prompt)
        attn_mask = torch.zeros(B, max_prompt_len, dtype=torch.bool, device=device)
        for i, L in enumerate(prompt_lens):
            attn_mask[i, -L:] = True

        # Per-sequence state
        active = torch.ones(B, dtype=torch.bool, device=device)
        generated = [padded_ids[i:i+1, -prompt_lens[i]:].clone() for i in range(B)]
        max_tokens = max(max_tokens_list)

        # Prefill
        with torch.inference_mode():
            out = model(padded_ids, attention_mask=attn_mask, use_cache=True,
                        logits_last_only=True)
            logits, past_kv = unpack_output_with_kv(out)

        # Decode-time mask: grows by one (all-ones) column per step so pad
        # positions in the cached KV stay masked. Without this, every decode
        # step attends to pad-token KV — silent pollution for any batch with
        # unequal prompt lengths.
        cur_mask = attn_mask
        ones_col = torch.ones(B, 1, dtype=torch.bool, device=device)

        # Decode — one batched step per token
        next_ids = torch.zeros(B, 1, dtype=torch.long, device=device)
        temps_cpu = [max(t, 1e-5) for t in temperatures]
        active_cpu = torch.ones(B, dtype=torch.bool)
        eos_mask_gpu = torch.zeros(B, dtype=torch.bool, device=device)

        # Pre-compute stop token IDs per sequence
        stop_token_ids = [set() for _ in range(B)]
        if tokenizer is not None:
            for i, stops in enumerate(stop_list):
                if stops:
                    for s in stops:
                        ids = tokenizer(s, add_special_tokens=False).input_ids
                        if ids:
                            stop_token_ids[i].add(ids[-1])

        for step in range(max_tokens):
            if not active_cpu.any():
                break

            # Sample all B tokens at once from batched logits [B, 1, vocab]
            next_logits_all = logits[:, -1, :]  # [B, vocab]

            # Repetition penalty BEFORE temperature (HF ordering; also
            # applies to the all-greedy path). One index_copy per sequence
            # instead of a kernel per unique generated token.
            if any(rp != 1.0 for rp in repetition_penalty_list):
                # raw logits may be an inference tensor — clone for in-place
                next_logits_all = next_logits_all.clone()
            for i in range(B):
                if repetition_penalty_list[i] != 1.0 and active_cpu[i]:
                    uniq = generated[i][0].unique()
                    next_logits_all[i].index_copy_(
                        0, uniq,
                        next_logits_all[i].index_select(0, uniq)
                        / repetition_penalty_list[i])

            # Temperature scaling [B, vocab]
            temp_t = torch.tensor(temps_cpu, device=device, dtype=next_logits_all.dtype)
            next_logits_all = next_logits_all / temp_t.unsqueeze(1)

            # Top-k filtering (per-sequence)
            for i in range(B):
                if top_k_list[i] > 0 and active_cpu[i]:
                    k = min(top_k_list[i], next_logits_all.shape[-1])
                    topk_vals, _ = next_logits_all[i].topk(k)
                    threshold = topk_vals[-1]
                    next_logits_all[i] = torch.where(
                        next_logits_all[i] < threshold,
                        torch.full_like(next_logits_all[i], float('-inf')),
                        next_logits_all[i],
                    )

            # Top-p filtering (per-sequence) — was previously plumbed through
            # but never applied; requests asking top_p<1 got top_p=1.0.
            for i in range(B):
                if top_ps[i] < 1.0 and active_cpu[i]:
                    next_logits_all[i] = self._top_p(
                        next_logits_all[i].unsqueeze(0), top_ps[i]).squeeze(0)

            # Per-sequence logits processors (e.g. chat_loop's think-cap /
            # structural-token bans). Each gets generated-only ids —
            # same contract as the serial path's logits_processor.
            for i in range(B):
                if processor_list[i] is not None and active_cpu[i]:
                    gen_only = generated[i][0, prompt_lens[i]:].tolist()
                    next_logits_all[i] = processor_list[i](
                        next_logits_all[i].unsqueeze(0), gen_only).squeeze(0)

            # Mask finished sequences
            next_logits_all[~active] = float('-inf')

            # Greedy or sampling
            if all(t == 0 for t in temperatures):
                next_tokens = next_logits_all.argmax(-1)  # [B]
            else:
                probs = F.softmax(next_logits_all, dim=-1).clamp(min=1e-10)
                probs = torch.nan_to_num(probs, nan=1.0 / probs.shape[-1])
                # Per-sequence sampling with independent generators
                next_tokens = torch.zeros(B, dtype=torch.long, device=device)
                for i in range(B):
                    if not active_cpu[i]:
                        next_tokens[i] = 0
                        continue
                    if temps_cpu[i] == 0:
                        # Greedy for this sequence
                        next_tokens[i] = next_logits_all[i].argmax()
                    elif generators[i] is not None:
                        # Seeded sampling
                        next_tokens[i] = torch.multinomial(
                            probs[i], num_samples=1, generator=generators[i])[0]
                    else:
                        # Unseeded sampling
                        next_tokens[i] = torch.multinomial(probs[i], num_samples=1)[0]

            # Check EOS (GPU op)
            for eid in self.eos_set:
                eos_mask_gpu = eos_mask_gpu | (next_tokens == eid)
            # Check stop sequences (per-sequence)
            for i in range(B):
                if stop_token_ids[i] and active_cpu[i]:
                    if next_tokens[i].item() in stop_token_ids[i]:
                        eos_mask_gpu[i] = True
            # Single scalar sync — only transfer full tensor when EOS occurs.
            # .cpu() is a no-op returning the SAME tensor when already on CPU —
            # clone() so eos_mask_gpu.zero_() below can't wipe is_eos_cpu.
            if eos_mask_gpu.any().item():
                is_eos_cpu = eos_mask_gpu.cpu().clone()
            else:
                is_eos_cpu = torch.zeros(B, dtype=torch.bool)
            eos_mask_gpu.zero_()

            # Update active state
            active_cpu = active_cpu & ~is_eos_cpu
            active.copy_(active_cpu.to(device))

            # Append tokens
            next_ids[:, 0] = next_tokens
            for i in range(B):
                if bool(active_cpu[i]) or bool(is_eos_cpu[i]):
                    generated[i] = torch.cat([generated[i], next_ids[i:i+1, :1]], dim=-1)

            if not active_cpu.any():
                break

            # Batched forward — mask now covers cached (prompt+gen) + new col
            cur_mask = torch.cat([cur_mask, ones_col], dim=1)
            with torch.inference_mode():
                out = model(next_ids, past_key_values=past_kv,
                            attention_mask=cur_mask, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)

        return generated

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)
