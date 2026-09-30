"""Decoding strategy backends.

Pluggable decoding strategies selectable at runtime:
  - StandardDecoding: autoregressive token-by-token (baseline)
  - ExternalDraftSpeculativeDecoding: draft model + verify ("speculative" alias)
  - MedusaDecoding: parallel prediction heads (wraps medusa.py)
  - DSparkDecoding: semi-autoregressive + confidence scheduling (wraps dspark.py)
  - MTPSelfSpecDecoding: use MTP heads from checkpoint for self-speculative decoding
  - SelfSpeculativeSparse: same model as draft+target with sparse-attention draft (R39-4)

All implement the DecodingStrategy interface:
  generate(model, input_ids, max_new_tokens, temperature, top_p) -> output_ids
"""
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

from forge.model_loader import unpack_output_with_kv


def _min_k_filter_logits(logits: torch.Tensor, sensitivity: float) -> torch.Tensor:
    """Min-k semantic-cliff sampling filter (ACL 2026, Ding et al.).

    Analyzes the local shape of the sorted logit distribution to identify
    "semantic cliffs" — sharp transitions from high-confidence core tokens
    to uncertain long-tail tokens. Temperature-invariant: operates on
    relative logit dynamics, not absolute probabilities.

    Args:
        logits: (batch, vocab) logit tensor.
        sensitivity: cliff detection sensitivity in [0, 1]. Higher = more
            aggressive truncation. 0 = no filtering.

    Returns:
        Filtered logits with long-tail tokens masked to -inf.
    """
    if sensitivity <= 0:
        return logits
    # Single sort: reuse the sorted indices for the scatter mask.
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    diffs = sorted_logits[..., :-1] - sorted_logits[..., 1:]
    n = diffs.shape[-1]
    weights = torch.linspace(1.0, 0.1, n, device=logits.device, dtype=logits.dtype)
    weighted_diffs = diffs * weights
    max_decay = weighted_diffs.max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    cliff_mask = weighted_diffs > sensitivity * max_decay
    # Truncate at the rightmost sharp transition (keep tokens up to and
    # including it); fallback to the single sharpest cliff when none
    # exceed the sensitivity threshold.
    idx = torch.arange(n, device=logits.device).expand_as(cliff_mask)
    last_cliff = torch.where(
        cliff_mask, idx, torch.full_like(idx, -1)).amax(dim=-1, keepdim=True)
    cliff_pos = torch.where(
        last_cliff >= 0, last_cliff,
        weighted_diffs.argmax(dim=-1, keepdim=True))
    positions = torch.arange(sorted_logits.shape[-1], device=logits.device)
    keep = positions <= cliff_pos
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, ~keep)
    return logits.masked_fill(mask, float("-inf"))


def _dry_penalties(
    context_ids: list[int],
    last_n: int,
    allowed_length: int,
    multiplier: float,
    base: float,
) -> dict[int, float]:
    """DRY repetition penalty (llama.cpp ``sampler_dry``, p-e-w).

    Local copy of ``engine_common._dry_penalties`` — this module is kept
    import-light and cannot import engine_common without a cycle.
    Returns ``{token_id: penalty}`` for logit-space subtraction.
    """
    n = len(context_ids)
    if multiplier <= 0 or n < 2 or last_n <= 0:
        return {}
    last_idx = n - 1
    start = max(0, last_idx - last_n)
    match_len: dict[int, int] = {}
    for i in range(start, last_idx):
        if context_ids[i] != context_ids[last_idx]:
            continue
        r = 1
        while i - r >= 0 and context_ids[i - r] == context_ids[last_idx - r]:
            r += 1
        if r > allowed_length:
            nxt = context_ids[i + 1]
            if r > match_len.get(nxt, 0):
                match_len[nxt] = r
    return {
        tok: multiplier * (base ** (r - allowed_length))
        for tok, r in match_len.items()
    }


def _apply_dry(
    next_logits: torch.Tensor, penalties: dict[int, float]
) -> torch.Tensor:
    """Subtract DRY penalties from a (vocab,) or (B, vocab) logit slice."""
    if not penalties:
        return next_logits
    toks = torch.tensor(list(penalties.keys()), device=next_logits.device)
    pens = torch.tensor(
        list(penalties.values()), device=next_logits.device, dtype=next_logits.dtype
    )
    next_logits.index_copy_(-1, toks, next_logits.index_select(-1, toks) - pens)
    return next_logits


def _sample_from_logits(
    next_logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    generated_ids: list[int],
    min_p: float,
    min_k: float,
    dry_penalties: dict[int, float] | None,
    top_p_fn,
) -> torch.Tensor:
    """Shared sampling chain: rep-penalty → temp → DRY → min-p →
    min-k → top-k → top-p → multinomial (argmax at temperature==0).

    Repetition penalty precedes temperature (HF processor ordering) and
    also applies to greedy decode — penalized logits can change argmax.

    Bit-identical to the inline chain previously duplicated across
    StandardDecoding; ``top_p_fn`` is the strategy's nucleus filter
    (StandardDecoding._top_p → research.sampling_utils.top_p_filter_logits).
    """
    if repetition_penalty != 1.0 and generated_ids:
        # raw logits may be an inference tensor (model ran under
        # inference_mode) — clone so the in-place index_copy is legal
        # outside that context.
        next_logits = next_logits.clone()
        toks = torch.tensor(
            sorted(set(generated_ids[-64:])), device=next_logits.device,
            dtype=torch.long)
        next_logits.index_copy_(
            -1, toks,
            next_logits.index_select(-1, toks) / repetition_penalty)
    next_logits = next_logits / max(temperature, 1e-5)
    if temperature == 0:
        return next_logits.argmax(-1, keepdim=True)
    if dry_penalties:
        next_logits = _apply_dry(next_logits, dry_penalties)
    if min_p > 0.0:
        probs = F.softmax(next_logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True).values
        threshold = min_p * max_prob
        next_logits = torch.where(
            probs < threshold,
            torch.full_like(next_logits, float('-inf')),
            next_logits,
        )
    if min_k > 0.0:
        next_logits = _min_k_filter_logits(next_logits, min_k)
    if top_k > 0:
        indices_to_remove = next_logits < torch.topk(
            next_logits, top_k)[0][..., -1, None]
        next_logits.masked_fill_(indices_to_remove, float('-inf'))
    if top_p < 1.0:
        next_logits = top_p_fn(next_logits, top_p)
    return torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)


class DecodingStrategy(ABC):
    """Base interface for decoding strategies."""

    @abstractmethod
    def generate(self, model, input_ids: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 0.0,
                 top_p: float = 1.0,
                 top_k: int = 80,
                 repetition_penalty: float = 1.05,
                 **kwargs) -> torch.Tensor:
        """Generate tokens. Returns full sequence [1, prompt_len + gen_len].

        top_k and repetition_penalty are LFM2.5-recommended defaults; they are
        only applied when temperature > 0 (ignored for greedy decoding).
        Subclasses may accept additional kwargs (min_p, min_k, etc.) — the base
        signature accepts **kwargs so the engine can uniformly pass all sampling
        params without TypeError on strategies that don't use them.
        """
        pass

    @property
    def name(self) -> str:
        return self.__class__.__name__


class StandardDecoding(DecodingStrategy):
    """Standard autoregressive decoding with KV cache."""

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 min_p: float = 0.0, min_k: float = 0.0,
                 dry_multiplier: float = 0.0, dry_base: float = 1.75,
                 dry_allowed_length: int = 2, dry_penalty_last_n: int = 512):
        ids = input_ids.clone()
        device = input_ids.device
        # EOS detection: check model attr, config, then known defaults
        eos = getattr(model, "eos_token_id", None)
        if eos is None:
            cfg = getattr(model, "config", None)
            eos = getattr(cfg, "eos_token_id", None) if cfg else None
        # LFM2.5 <|im_end|>=7, Qwen2.5 151643/151645,
        # Jamba <|endoftext|>=2, <|im_end|>=519
        eos_set = {2, 7, 519, 151643, 151645}
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)
        # Track generated token ids for repetition penalty + degeneration.
        # DRY operates on prompt + generated context (llama.cpp semantics).
        prompt_ids = input_ids[0].tolist()
        generated_ids: list[int] = []
        # Collect generated token tensors for a single final cat (O(n) vs O(n²))
        _generated_tokens: list[torch.Tensor] = []
        # Degeneration guard: stop if same token repeats too many times.
        MAX_REPEAT = 8  # same token 8x in a row = degenerate

        # Prefill — model returns (logits, loss, presents) when use_cache=True
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)

        for _ in range(max_new_tokens):
            # DRY penalty: suppress tokens that continue a repeated
            # n-gram suffix anywhere in prompt+generated context.
            dry_pen = (
                _dry_penalties(
                    prompt_ids + generated_ids,
                    dry_penalty_last_n,
                    dry_allowed_length,
                    dry_multiplier,
                    dry_base,
                )
                if dry_multiplier > 0.0 else None
            )
            next_token = _sample_from_logits(
                logits[:, -1, :],
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
                min_p=min_p, min_k=min_k,
                dry_penalties=dry_pen,
                top_p_fn=self._top_p)

            # Single GPU->CPU sync per token: .item() brings the id over,
            # then the EOS check runs on the CPU scalar (no second sync).
            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if tok_id in eos_set:
                break

            # Degeneration guard: detect repetitive garbage and stop early.
            # Critical for base models that haven't learned to emit EOS.
            if len(generated_ids) >= MAX_REPEAT:
                if all(g == tok_id for g in generated_ids[-MAX_REPEAT:]):
                    break  # same token repeated 8x = degenerate
            if len(generated_ids) >= 20:
                recent = generated_ids[-20:]
                if len(set(recent)) / len(recent) < 0.4:
                    break  # <40% unique tokens = degenerate

            # Collect tokens for final cat (avoids O(n²) torch.cat per step)
            _generated_tokens.append(next_token)
            with torch.inference_mode():
                out = model(next_token, past_key_values=past_kv, use_cache=True)
                logits, past_kv = unpack_output_with_kv(out)

        # Expose final KV cache state for _finish_to_stop fast path
        model._forge_last_kv = past_kv
        # Single cat at the end instead of per-step (O(n) vs O(n²))
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        return ids

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)


class NGramSpeculativeDecoding(DecodingStrategy):
    """N-gram speculative decoding (prompt-lookup).

    Uses n-gram matching against the prompt to generate draft tokens —
    no separate draft model needed. Works well for code completion and
    RAG where output often repeats prompt substrings.

    Based on vLLM's prompt_lookup_speculative_decoding.
    """

    def __init__(self, ngram_size: int = 3, draft_length: int = 4):
        self.ngram_size = ngram_size
        self.draft_length = draft_length

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        device = input_ids.device
        input_ids.shape[1]
        generated = 0

        with torch.inference_mode():
            # Prefill
            out = model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)

        while generated < max_new_tokens:
            # Generate main token
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                main_token = next_logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                main_token = torch.multinomial(probs.view(-1), 1).unsqueeze(0)

            # N-gram lookup: find draft tokens by matching last ngram_size tokens
            draft_tokens = self._ngram_lookup(ids, main_token.squeeze())

            if len(draft_tokens) > 0:
                # Verify draft tokens in one forward pass
                draft_tensor = torch.tensor(draft_tokens, device=device).unsqueeze(0)
                verify_ids = torch.cat([main_token, draft_tensor], dim=-1)
                with torch.inference_mode():
                    out = model(verify_ids, past_key_values=past_kv, use_cache=True)
                    verify_logits, past_kv = unpack_output_with_kv(out)

                # Accept tokens that match
                accepted = 0
                main_token.squeeze()
                for i, draft_tok in enumerate(draft_tokens):
                    pred_tok = verify_logits[0, i, :].argmax()
                    if pred_tok.item() == draft_tok:
                        accepted += 1
                        torch.tensor([draft_tok], device=device)
                    else:
                        break

                # Append accepted + 1 (the token predicted at the mismatch)
                new_tokens = [main_token.squeeze().item()] + draft_tokens[:accepted]
                if accepted < len(draft_tokens):
                    # Mismatch — use the model's prediction at the mismatch point
                    new_tokens.append(verify_logits[0, accepted, :].argmax().item())

                for tok in new_tokens:
                    ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)
                    generated += 1
                    if generated >= max_new_tokens:
                        break

                # Get logits for next iteration (last position)
                if accepted < len(draft_tokens):
                    logits = verify_logits[:, accepted:accepted+1, :]
                else:
                    logits = verify_logits[:, -1:, :]
            else:
                # No draft — standard step
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                with torch.inference_mode():
                    out = model(main_token, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)

        return ids

    def _ngram_lookup(self, ids: torch.Tensor, last_token: int) -> list[int]:
        """Find draft tokens by n-gram matching against the sequence."""
        seq = ids[0].cpu().tolist() + [last_token]
        if len(seq) < self.ngram_size + 1:
            return []
        ngram = seq[-self.ngram_size:]
        # Search for this ngram earlier in the sequence
        for i in range(len(seq) - self.ngram_size - 1, self.ngram_size - 2, -1):
            if seq[i:i+self.ngram_size] == ngram:
                # Return up to draft_length tokens after the match
                start = i + self.ngram_size
                end = min(start + self.draft_length, len(seq))
                return seq[start:end]
        return []


class ExternalDraftSpeculativeDecoding(DecodingStrategy):
    """Speculative decoding with an external draft model.

    The draft model is a smaller/faster model that proposes tokens.
    The main model verifies them in a single forward pass.
    """

    def __init__(self, draft_model, draft_length: int = 4):
        self.draft_model = draft_model
        self.draft_length = draft_length

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        device = input_ids.device
        generated = 0

        with torch.inference_mode():
            # Prefill both models
            out = model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)
            draft_out = self.draft_model(ids, use_cache=True)
            draft_logits, draft_past_kv = unpack_output_with_kv(draft_out)

        while generated < max_new_tokens:
            # Draft model proposes k tokens
            draft_tokens = []
            cur_logits = draft_logits[:, -1, :]
            cur_ids = ids
            for _ in range(self.draft_length):
                if temperature == 0:
                    tok = cur_logits.argmax(-1, keepdim=True)
                else:
                    probs = F.softmax(cur_logits / max(temperature, 1e-5), dim=-1)
                    tok = torch.multinomial(probs.view(-1), 1).unsqueeze(0)
                draft_tokens.append(tok.squeeze().item())
                cur_ids = torch.cat([cur_ids, tok], dim=-1)
                with torch.inference_mode():
                    draft_out = self.draft_model(tok, past_key_values=draft_past_kv, use_cache=True)
                    draft_logits, draft_past_kv = unpack_output_with_kv(draft_out)

            # Main model generates next token
            main_token = (logits[:, -1, :] / max(temperature, 1e-5)).argmax(-1, keepdim=True)

            if draft_tokens:
                # Verify draft in one forward pass
                draft_tensor = torch.tensor(draft_tokens, device=device).unsqueeze(0)
                verify_ids = torch.cat([main_token, draft_tensor], dim=-1)
                with torch.inference_mode():
                    out = model(verify_ids, past_key_values=past_kv, use_cache=True)
                    verify_logits, past_kv = unpack_output_with_kv(out)

                # Accept matching prefix
                accepted = 0
                new_tokens = [main_token.squeeze().item()]
                for i, draft_tok in enumerate(draft_tokens):
                    pred_tok = verify_logits[0, i, :].argmax()
                    if pred_tok.item() == draft_tok:
                        accepted += 1
                        new_tokens.append(draft_tok)
                    else:
                        new_tokens.append(pred_tok.item())
                        break

                for tok in new_tokens:
                    ids = torch.cat([ids, torch.tensor([[tok]], device=device)], dim=-1)
                    generated += 1
                    if generated >= max_new_tokens:
                        break

                logits = verify_logits[:, -1:, :]
            else:
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                with torch.inference_mode():
                    out = model(main_token, past_key_values=past_kv, use_cache=True)
                    logits, past_kv = unpack_output_with_kv(out)

        return ids


class MedusaDecoding(DecodingStrategy):
    """Medusa parallel prediction heads."""

    def __init__(self, medusa_heads):
        self.medusa = medusa_heads

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        from forge.decoding.medusa import medusa_generate
        return medusa_generate(
            model, self.medusa, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, device=str(input_ids.device),
        )


class DSparkDecoding(DecodingStrategy):
    """DSpark semi-autoregressive speculative decoding."""

    def __init__(self, dspark_head=None):
        self.dspark = dspark_head

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        from forge.decoding.dspark import dspark_generate
        return dspark_generate(
            model, self.dspark, input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature, device=str(input_ids.device),
        )


class Eagle3Decoding(DecodingStrategy):
    """EAGLE-3 feature-level speculative decoding.

    Uses a lightweight draft head that operates on the target model's
    multi-layer hidden states (low/mid/high) to predict draft tokens.
    No separate draft model needed — the head is attached to the target.

    Args:
        eagle_head: trained Eagle3Head module (optional, can be loaded later)
        draft_length: number of tokens to draft per iteration (default 4)
    """

    def __init__(self, eagle_head=None, draft_length: int = 4):
        self.eagle_head = eagle_head
        self.draft_length = draft_length

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        if self.eagle_head is None:
            # Fallback to standard if no head loaded
            return StandardDecoding().generate(
                model, input_ids, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p,
                top_k=top_k, repetition_penalty=repetition_penalty,
                **kwargs,
            )
        # eagle3_generate takes a prompt string; here we work with token ids
        # so we use the internal generation loop directly.
        return _eagle_generate_from_ids(
            model, self.eagle_head, input_ids,
            max_new_tokens=max_new_tokens,
            draft_length=self.draft_length,
            temperature=temperature, top_k=top_k,
            repetition_penalty=repetition_penalty,
            device=str(input_ids.device),
        )


class MTPSelfSpecDecoding(DecodingStrategy):
    """Self-speculative decoding using MTP heads from KeyStack checkpoint.

    The XP model checkpoint already contains MTP heads (mtp_head.heads.0-3.weight)
    initialized from the LM head. These predict tokens at positions t+1, t+2,
    t+3, t+4 in parallel. We use them as a draft model for self-speculative
    decoding — no separate draft model needed.

    Flow:
    1. Main model generates token t (with full attention)
    2. MTP heads predict draft tokens t+1..t+k (parallel, no attention)
    3. Main model verifies all k+1 tokens in ONE forward pass (EAGLE-style
       tree verification — a linear chain's tree mask is the standard causal
       mask, so no custom mask is needed). Predictions at all k positions are
       compared against the drafts in parallel; the first mismatch is found
       and the accepted prefix is kept. KV cache is rolled back by slicing
       the returned presents tensors to the accepted length.
    4. Accept matching prefix, reject and resample at first mismatch

    This is "free" speculative decoding — the MTP heads are already in the
    checkpoint and add zero inference cost for the draft phase. The batch
    verification makes it actually faster than standard decoding (one forward
    pass verifies k drafts, vs k forward passes for one-at-a-time).

    Evolution-discovered defaults: 7 draft tokens (k=7), 0.95 acceptance
    threshold, 0.10 draft_model_ratio → 5.37x speedup over standard decoding.
    These values were found by evolution search on RTX 5070 + LFM2.5-1.2B
    and represent the sweet spot between draft accuracy and verification
    overhead.
    """

    def __init__(self, k=7, mtp_module=None, acceptance_threshold=0.95,
                 draft_model_ratio=0.10, replay_ssm_cache=None):
        self.k = k
        self.mtp = mtp_module  # Optional: pre-loaded MTP module
        self.acceptance_threshold = acceptance_threshold
        self.draft_model_ratio = draft_model_ratio
        self.replay_ssm = replay_ssm_cache  # ReplaySSMCache for SSM rollback

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        eos = getattr(model, "eos_token_id", None)

        # Get MTP module from model if not provided
        mtp = self.mtp
        if mtp is None and hasattr(model, "mtp_head"):
            mtp = model.mtp_head
        if mtp is None:
            # No MTP heads — fall back to standard
            return StandardDecoding().generate(
                model, input_ids, max_new_tokens, temperature, top_p,
                top_k=top_k, repetition_penalty=repetition_penalty,
                **kwargs)

        # Prefill — need both KV cache (presents) and hidden states for MTP
        with torch.inference_mode():
            out = model(ids, use_cache=True, return_hidden=True)
            # Returns (logits, loss, presents, hidden) when use_cache + return_hidden
            logits = out[0]
            past_kv = out[2]  # presents
            hidden = out[3]   # hidden states

        generated = 0
        while generated < max_new_tokens:
            # Step 1: Main model predicts token t (using KV cache, single token)
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                main_token = next_logits.argmax(-1, keepdim=True)
            else:
                main_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            if eos and (main_token == eos).any().item():
                ids = torch.cat([ids, main_token], dim=-1)
                break

            # ReplaySSM: checkpoint before drafting so we can rollback
            # the SSM state efficiently if the draft is rejected.
            ssm_checkpoint = None
            if self.replay_ssm is not None:
                ssm_checkpoint = self.replay_ssm.checkpoint()

            # Step 2: MTP heads predict draft tokens (parallel, from last hidden)
            if hidden is not None:
                with torch.inference_mode():
                    mtp_out = mtp(hidden[:, -1:, :])  # Use last hidden state
                draft_tokens = [main_token]  # [1, 1]
                if isinstance(mtp_out, (list, tuple)):
                    for head_out in mtp_out[:self.k]:
                        draft_tok = head_out.argmax(-1)[:, -1:]  # [B, 1]
                        draft_tokens.append(draft_tok)
                else:
                    draft_tokens.append(mtp_out.argmax(-1)[:, -1:])
            else:
                draft_tokens = [main_token]

            # Step 3: Batch-verify ALL draft tokens in ONE forward pass
            # (EAGLE-style tree verification for a linear chain — the tree
            # mask for a chain is just the standard causal mask, so no custom
            # mask is needed). This replaces the previous one-at-a-time loop
            # which defeated the purpose of speculative decoding (each token
            # did a full forward pass, making it slower than standard decoding).
            #
            # verify_seq = [main_token, draft_1, ..., draft_k]  (length n_draft)
            # logits[:, i, :] predicts the token after position i.
            # We compare the model's prediction at position i against draft_{i+1}
            # for all i in parallel, find the first mismatch, accept the prefix,
            # and resample at the mismatch.
            n_draft = len(draft_tokens) - 1  # number of MTP-drafted tokens
            if n_draft == 0:
                # No drafts — just accept the main token.
                ids = torch.cat([ids, main_token], dim=-1)
                generated += 1
                # logits already set for the next iteration.
                continue

            verify_seq = torch.cat(draft_tokens, dim=-1)  # [B, n_draft+1]
            past_len = past_kv[0][0].shape[-2] if past_kv is not None else 0
            with torch.inference_mode():
                out = model(verify_seq, past_key_values=past_kv,
                            use_cache=True, return_hidden=True)
                verify_logits = out[0]   # [B, n_draft+1, V]
                new_past_kv = out[2]     # list of (k, v) with len past_len + n_draft + 1
                verify_hidden = out[3]   # [B, n_draft+1, d]

            # Predictions for positions 0..n_draft-1 (what should follow each
            # accepted token) vs the drafted tokens at 1..n_draft.
            preds = verify_logits[:, :n_draft, :].argmax(-1)        # [B, n_draft]
            draft_stack = torch.cat(draft_tokens[1:], dim=-1)       # [B, n_draft]
            matches = (preds == draft_stack)                        # [B, n_draft]

            # First mismatch index (per batch). For B=1 this is a scalar.
            # n_accepted = number of accepted drafts (main_token + first n_accepted drafts).
            # We take the per-batch min of the first-mismatch index so the
            # accepted prefix is valid for every sequence in the batch.
            # first_false[i] = index of first False in matches[i], or n_draft if all True.
            not_match = ~matches
            # argmax returns first True index; if row is all False, argmax returns 0.
            any_mismatch = not_match.any(dim=-1)                    # [B]
            first_mismatch = torch.where(
                any_mismatch,
                not_match.float().argmax(dim=-1),
                torch.full_like(any_mismatch, n_draft, dtype=torch.long),
            )                                                       # [B]
            n_accepted = int(first_mismatch.min().item())           # accept the common prefix

            # KV cache rollback: keep only past_len + n_accepted + 1 entries
            # (the accepted prefix). The new_past_kv tensors have length
            # past_len + n_draft + 1; slice off the rejected tail.
            keep_len = past_len + n_accepted + 1
            past_kv = []
            for layer_kv in new_past_kv:
                k, v = layer_kv
                past_kv.append((k[:, :, :keep_len, :], v[:, :, :keep_len, :]))

            # ReplaySSM: if some drafts were rejected, rollback the SSM
            # input cache to the checkpoint so the next reconstruct_state
            # replays only the accepted inputs.  When all drafts are
            # accepted, no rollback is needed.
            if (self.replay_ssm is not None and ssm_checkpoint is not None
                    and n_accepted < n_draft):
                self.replay_ssm.rollback(ssm_checkpoint + n_accepted + 1)

            # Accepted tokens: main_token + drafts 1..n_accepted.
            accepted = verify_seq[:, :n_accepted + 1]
            ids = torch.cat([ids, accepted], dim=-1)
            generated += n_accepted + 1

            # Set up logits + hidden for the NEXT iteration:
            #  - If a mismatch occurred at index j=n_accepted, the model's
            #    prediction at position j (verify_logits[:, j, :]) is the
            #    corrected next token — use it as the next main_token's logits.
            #  - If all drafts accepted, use the prediction after the last
            #    draft (verify_logits[:, n_draft, :]).
            next_idx = min(n_accepted, n_draft)
            logits = verify_logits[:, next_idx:next_idx + 1, :]    # [B, 1, V]
            hidden = verify_hidden[:, next_idx:next_idx + 1, :]    # [B, 1, d]

            if eos:
                # GPU-side EOS check on the accepted prefix (single sync).
                if (accepted == eos).any().item():
                    break

        return ids


class SelfSpeculativeSparse(DecodingStrategy):
    """Self-speculative decoding with sparse-attention draft (R39-4).

    The **same model** serves as both draft and target — no separate
    draft model is needed, so there is **zero extra memory**.

    Flow per iteration:
      1. **Draft phase**: generate ``draft_len`` tokens autoregressively
         using *sparse attention* (top-``sparse_k`` KV positions).  The
         model forward is called with ``sparse_k=<int>`` so attention
         layers can limit the KV cache to the top-k most relevant
         positions.  This is faster than full attention because the
         attention computation is O(seq_len * sparse_k) instead of
         O(seq_len²).
      2. **Verify phase**: run the model with **full attention** on the
         entire draft sequence in a single forward pass.  The full-
         attention logits at each position tell us what the *correct*
         next token is.
      3. **Accept**: compare draft tokens against the full-attention
         predictions.  Accept the longest matching prefix.  Resample at
         the first mismatch using the full-attention logits.
      4. **KV selection feedback**: the verification attention scores
         from the accepted positions are reused to select which KV
         positions to keep for the next draft round's sparse attention.

    Because verification uses full attention, the output is **lossless**
    — every accepted token is exactly what standard full-attention
    decoding would produce.  The sparse draft only affects *speed*, not
    *quality*.

    Args:
        draft_len: number of tokens to draft per iteration (default 4).
        sparse_k: number of top KV positions to attend to during the
            draft phase (default 64).  Larger = more accurate draft but
            slower; smaller = faster but lower acceptance rate.
    """

    def __init__(self, draft_len: int = 4, sparse_k: int = 64,
                 replay_ssm_cache=None):
        self.draft_len = draft_len
        self.sparse_k = sparse_k
        self.replay_ssm = replay_ssm_cache  # ReplaySSMCache for SSM rollback
        # Acceptance statistics (updated during generate).
        self.acceptance_rate: float = 0.0
        self._total_drafts: int = 0
        self._total_accepted: int = 0
        # KV selection scores from the last verification pass — reused
        # for sparse attention position selection in the next draft round.
        self._kv_scores: torch.Tensor | None = None

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0,
                 top_k=80, repetition_penalty=1.05,
                 **kwargs):
        ids = input_ids.clone()
        device = input_ids.device
        eos = getattr(model, "eos_token_id", None)
        eos_set = {2, 7, 519, 151643, 151645}
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)
        torch.tensor(list(eos_set), device=device)

        # Prefill with full attention.
        with torch.inference_mode():
            out = model(ids, use_cache=True)
            logits, past_kv = unpack_output_with_kv(out)

        generated = 0
        _generated_tokens: list[torch.Tensor] = []

        while generated < max_new_tokens:
            # ── Step 1: Base token from full-attention logits ──────────
            next_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if temperature == 0:
                base_token = next_logits.argmax(-1, keepdim=True)
            else:
                base_token = torch.multinomial(
                    F.softmax(next_logits, dim=-1), num_samples=1)

            # ReplaySSM: checkpoint before drafting so we can rollback
            # the SSM state efficiently if the draft is rejected.
            ssm_checkpoint = None
            if self.replay_ssm is not None:
                ssm_checkpoint = self.replay_ssm.checkpoint()

            # ── Step 2: Draft phase — sparse attention ────────────────
            # Generate draft_len tokens autoregressively with sparse
            # attention (top-k KV positions).  The model forward receives
            # sparse_k so attention layers can limit the KV window.
            draft_tokens = [base_token]
            draft_kv = past_kv
            for _ in range(self.draft_len):
                with torch.inference_mode():
                    draft_out = model(
                        draft_tokens[-1],
                        past_key_values=draft_kv,
                        use_cache=True,
                        sparse_k=self.sparse_k,
                    )
                    draft_logits_i, draft_kv = unpack_output_with_kv(draft_out)
                draft_tok = draft_logits_i[:, -1, :].argmax(-1, keepdim=True)
                draft_tokens.append(draft_tok)

            n_draft = len(draft_tokens) - 1  # number of drafted tokens

            # ── Step 3: Verify phase — full attention ─────────────────
            # Run the model with full attention on the entire draft
            # sequence in ONE forward pass.  The draft KV cache (sparse)
            # is discarded — we re-process from the original full KV.
            verify_seq = torch.cat(draft_tokens, dim=-1)  # [B, n_draft+1]
            past_len = past_kv[0][0].shape[-2] if past_kv is not None else 0
            with torch.inference_mode():
                verify_out = model(
                    verify_seq,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                verify_logits, new_past_kv = unpack_output_with_kv(verify_out)

            # ── Step 4: Compare draft vs full-attention predictions ───
            # verify_logits[:, i, :] predicts the token after position i.
            # We compare against draft_tokens[i+1] for i in [0, n_draft).
            preds = verify_logits[:, :n_draft, :].argmax(-1)     # [B, n_draft]
            draft_stack = torch.cat(draft_tokens[1:], dim=-1)    # [B, n_draft]
            matches = (preds == draft_stack)                     # [B, n_draft]

            # First mismatch index (per batch).
            not_match = ~matches
            any_mismatch = not_match.any(dim=-1)
            first_mismatch = torch.where(
                any_mismatch,
                not_match.float().argmax(dim=-1),
                torch.full_like(any_mismatch, n_draft, dtype=torch.long),
            )
            n_accepted = int(first_mismatch.min().item())

            # ── Acceptance statistics ────────────────────────────────
            self._total_drafts += n_draft
            self._total_accepted += n_accepted
            self.acceptance_rate = (
                self._total_accepted / max(self._total_drafts, 1)
            )

            # ── Step 5: KV cache rollback to accepted prefix ─────────
            keep_len = past_len + n_accepted + 1
            past_kv = []
            for layer_kv in new_past_kv:
                k, v = layer_kv
                past_kv.append(
                    (k[:, :, :keep_len, :], v[:, :, :keep_len, :]))

            # ReplaySSM: if some drafts were rejected, rollback the SSM
            # input cache to the checkpoint + accepted prefix so the next
            # reconstruct_state replays only the accepted inputs.
            if (self.replay_ssm is not None and ssm_checkpoint is not None
                    and n_accepted < n_draft):
                self.replay_ssm.rollback(ssm_checkpoint + n_accepted + 1)

            # ── Step 6: Accept tokens ────────────────────────────────
            accepted = verify_seq[:, :n_accepted + 1]
            _generated_tokens.append(accepted)
            generated += n_accepted + 1

            # Check EOS in accepted tokens.
            if any(t.item() in eos_set for t in accepted.flatten()):
                break

            # ── Set up logits for next iteration ─────────────────────
            # If mismatch at index j=n_accepted, verify_logits[:, j, :]
            # is the corrected next token's logits.
            # If all accepted, use the prediction after the last draft.
            next_idx = min(n_accepted, n_draft)
            logits = verify_logits[:, next_idx:next_idx + 1, :]

        # Single cat at the end (O(n) vs O(n²)).
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        # Truncate to max_new_tokens (speculative acceptance can overshoot).
        max_len = input_ids.shape[1] + max_new_tokens
        if ids.shape[1] > max_len:
            ids = ids[:, :max_len]
        return ids


class DoLaDecoding(DecodingStrategy):
    """DoLa self-contrastive decoding (Chuang et al., ICLR 2024).

    Decoding by Contrasting Layers: the next-token distribution is
    ``softmax(log_softmax(final) − log_softmax(early))`` restricted to the
    final distribution's top-``candidate_top_k`` tokens. Amplifies the
    factual signal concentrated in late layers against shallow-layer
    surface priors — no auxiliary model or training required.

    On the ForgeLM hybrid, contrasting a Mamba block's hidden state
    against the final logits also isolates the attention layers' recall
    contribution (``early_layer`` accepts any block index; the block
    outputs are uniform hidden states regardless of block type).

    Args:
        early_layer: fixed early-layer block index, or ``None`` for
            per-step auto-selection (max JSD vs final distribution — the
            paper's "dynamic premature layer" choice) over
            ``early_candidates``.
        early_candidates: layer indices for auto-selection; default
            ``{n/4, n/2, 3n/4}`` clamped to valid block indices.
        candidate_top_k: restrict the contrasted distribution to the
            top-k tokens by final-layer probability (paper's candidate
            set; 0 = score the whole vocab).
    """

    def __init__(self, early_layer: int | None = None,
                 early_candidates: list[int] | None = None,
                 candidate_top_k: int = 64):
        self.early_layer = early_layer
        self.early_candidates = early_candidates
        self.candidate_top_k = candidate_top_k

    @staticmethod
    def _early_logits(model, hidden: torch.Tensor) -> torch.Tensor:
        """Project a block-output hidden state through ln_f + lm head."""
        h = hidden[:, -1, :]
        ln_f = getattr(model, "ln_f", None)
        if ln_f is not None:
            h = ln_f(h)
        return model.head(h)

    @staticmethod
    def _jsd(p: torch.Tensor, q: torch.Tensor) -> float:
        """Jensen-Shannon divergence between two probability rows."""
        m = 0.5 * (p + q)
        log_m = m.clamp_min(1e-12).log()
        kl_pm = (p * (p.clamp_min(1e-12).log() - log_m)).sum(-1)
        kl_qm = (q * (q.clamp_min(1e-12).log() - log_m)).sum(-1)
        return float((0.5 * (kl_pm + kl_qm)).mean().item())

    @torch.no_grad()
    def _contrast_logits(self, model, final_logits, hidden_list,
                         candidates) -> torch.Tensor:
        """DoLa contrasted logits for the last position (fp32).

        Runs under no_grad (NOT inference_mode): the hidden_list inputs
        are inference tensors produced during the cached forwards — model
        params (requires_grad=True) would save them for backward outside
        a no-grad scope. Outputs stay normal tensors so the caller's
        in-place sampling ops still work.
        """
        p_final = F.softmax(final_logits.float(), dim=-1)
        if len(candidates) > 1:
            # Dynamic premature-layer selection: argmax JSD vs final.
            early_logits, best_jsd = None, -1.0
            for layer_idx in candidates:
                e_logits = self._early_logits(model, hidden_list[layer_idx])
                jsd = self._jsd(p_final, F.softmax(e_logits.float(), dim=-1))
                if jsd > best_jsd:
                    best_jsd, early_logits = jsd, e_logits
        else:
            early_logits = self._early_logits(
                model, hidden_list[candidates[0]])

        contrasted = (F.log_softmax(final_logits.float(), dim=-1)
                      - F.log_softmax(early_logits.float(), dim=-1))
        if self.candidate_top_k > 0:
            k = min(self.candidate_top_k, contrasted.shape[-1])
            top_idx = p_final.topk(k, dim=-1).indices
            mask = torch.ones_like(contrasted, dtype=torch.bool)
            mask.scatter_(-1, top_idx, False)
            contrasted = contrasted.masked_fill(mask, float("-inf"))
        return contrasted

    def generate(self, model, input_ids, max_new_tokens=100,
                 temperature=0.0, top_p=1.0, top_k=80,
                 repetition_penalty=1.05,
                 min_p: float = 0.0, min_k: float = 0.0,
                 dry_multiplier: float = 0.0, dry_base: float = 1.75,
                 dry_allowed_length: int = 2, dry_penalty_last_n: int = 512):
        ids = input_ids.clone()
        eos = getattr(model, "eos_token_id", None)
        if eos is None:
            cfg = getattr(model, "config", None)
            eos = getattr(cfg, "eos_token_id", None) if cfg else None
        eos_set = {2, 7, 519, 151643, 151645}
        if isinstance(eos, (list, tuple, set, frozenset)):
            eos_set.update(eos)
        elif eos is not None:
            eos_set.add(eos)

        prompt_ids = input_ids[0].tolist()
        generated_ids: list[int] = []
        _generated_tokens: list[torch.Tensor] = []
        MAX_REPEAT = 8

        n_layers = len(getattr(model, "blocks", [])) or getattr(
            getattr(model, "config", None), "n_layers", 0)
        if self.early_layer is not None:
            candidates = [max(0, min(int(self.early_layer), n_layers - 1))]
        else:
            if self.early_candidates is not None:
                candidates = list(self.early_candidates)
            else:
                candidates = sorted({
                    max(0, min(int(f), n_layers - 1))
                    for f in (n_layers // 4, n_layers // 2,
                              3 * n_layers // 4)})
            candidates = [c for c in candidates if 0 <= c < n_layers] or [0]

        # Prefill — return_hidden_states yields (logits, loss, kv, hs_list).
        with torch.inference_mode():
            out = model(ids, use_cache=True, return_hidden_states=True)
            logits, past_kv = unpack_output_with_kv(out)
            hidden_list = out[3]

        for _ in range(max_new_tokens):
            next_logits = self._contrast_logits(
                model, logits[:, -1, :], hidden_list, candidates)
            dry_pen = (
                _dry_penalties(
                    prompt_ids + generated_ids, dry_penalty_last_n,
                    dry_allowed_length, dry_multiplier, dry_base)
                if dry_multiplier > 0.0 else None
            )
            next_token = _sample_from_logits(
                next_logits,
                temperature=temperature, top_p=top_p, top_k=top_k,
                repetition_penalty=repetition_penalty,
                generated_ids=generated_ids,
                min_p=min_p, min_k=min_k,
                dry_penalties=dry_pen,
                top_p_fn=self._top_p)

            tok_id = next_token.item()
            generated_ids.append(tok_id)
            if tok_id in eos_set:
                break
            if len(generated_ids) >= MAX_REPEAT:
                if all(g == tok_id for g in generated_ids[-MAX_REPEAT:]):
                    break
            if len(generated_ids) >= 20:
                recent = generated_ids[-20:]
                if len(set(recent)) / len(recent) < 0.4:
                    break

            _generated_tokens.append(next_token)
            with torch.inference_mode():
                out = model(next_token, past_key_values=past_kv,
                            use_cache=True, return_hidden_states=True)
                logits, past_kv = unpack_output_with_kv(out)
                hidden_list = out[3]

        model._forge_last_kv = past_kv
        if _generated_tokens:
            ids = torch.cat([ids] + _generated_tokens, dim=-1)
        return ids

    def _top_p(self, logits, top_p):
        from research.sampling_utils import top_p_filter_logits
        return top_p_filter_logits(logits, top_p)


def build_decoding(strategy: str = "standard", **kwargs) -> DecodingStrategy:
    """Factory: build decoding strategy by name."""
    strategies = {
        "standard": StandardDecoding,
        "speculative": ExternalDraftSpeculativeDecoding,
        "ngram_speculative": NGramSpeculativeDecoding,
        "external_draft_speculative": ExternalDraftSpeculativeDecoding,
        "medusa": MedusaDecoding,
        "dspark": DSparkDecoding,
        "eagle3": Eagle3Decoding,
        "mtp_selfspec": MTPSelfSpecDecoding,
        "self_speculative_sparse": SelfSpeculativeSparse,
        "dola": DoLaDecoding,
        "batched": None,  # set below to avoid circular import
    }
    if strategy == "batched":
        from forge.engine.batched_decoding import BatchedDecoding
        return BatchedDecoding(**kwargs)
    if strategy == "uno":
        from forge.decoding.uno import NgramProposer, UnoDecoding
        kwargs.setdefault("proposer", NgramProposer())
        return UnoDecoding(**kwargs)
    cls = strategies.get(strategy, StandardDecoding)
    return cls(**kwargs)


@torch.inference_mode()
def _eagle_generate_from_ids(
    model, head, input_ids, max_new_tokens=100, draft_length=4,
    temperature=0.0, top_k=0, repetition_penalty=1.0, device="cuda",
):
    """EAGLE-3 generation from token ids (used by Eagle3Decoding strategy)."""
    from forge.decoding.eagle import extract_hidden_states

    model.eval()
    head.eval()

    extract_layers = [head.low_layer, head.mid_layer, head.high_layer]
    eos_id = getattr(model, 'eos_token_id', None) or 7

    # Prefill
    hidden_list, final_hidden, presents = extract_hidden_states(
        model, input_ids, extract_layers, use_cache=True,
    )
    fused = head.fuse_hidden_states(hidden_list)
    target_logits = model.head(final_hidden)
    last_logits = target_logits[:, -1, :]

    if temperature <= 0:
        next_token = last_logits.argmax(dim=-1, keepdim=True)
    else:
        l = last_logits / temperature
        if top_k > 0:
            idx_rm = l < torch.topk(l, top_k)[0][..., -1, None]
            l.masked_fill_(idx_rm, float('-inf'))
        probs = F.softmax(l, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    generated = [next_token.item()]
    _eagle_tokens: list[torch.Tensor] = [next_token]  # collect for final cat

    while len(generated) < max_new_tokens:
        if generated[-1] == eos_id:
            break

        # Draft tokens autoregressively
        draft_tokens = []
        cur_fused = fused[:, -1:, :]
        cur_token = next_token
        for _ in range(draft_length):
            draft_tok, _ = head.predict_next(cur_fused, cur_token, temperature=temperature, top_k=top_k)
            draft_tokens.append(draft_tok.item())
            cur_token = draft_tok

        # Verify with target
        draft_tensor = torch.tensor([draft_tokens], device=device, dtype=input_ids.dtype)
        verify_hidden, verify_final, presents = extract_hidden_states(
            model, draft_tensor, extract_layers,
            past_key_values=presents, use_cache=True,
        )
        verify_logits = model.head(verify_final)

        # Accept longest prefix
        n_accepted = 0
        for i, dt in enumerate(draft_tokens):
            target_tok = verify_logits[:, i, :].argmax(dim=-1).item()
            if target_tok == dt:
                n_accepted += 1
            else:
                break

        for i in range(n_accepted):
            generated.append(draft_tokens[i])
            if len(generated) >= max_new_tokens:
                break

        if n_accepted < draft_length:
            next_token = verify_logits[:, n_accepted, :].argmax(dim=-1, keepdim=True)
        else:
            next_token = verify_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        fused = head.fuse_hidden_states(verify_hidden)
        if next_token.item() == eos_id:
            break
        generated.append(next_token.item())
        _eagle_tokens.append(next_token)

    # Single cat at the end instead of per-step (O(n) vs O(n²))
    ids = torch.cat([input_ids] + _eagle_tokens, dim=1)
    return ids[:, input_ids.shape[1]:]
