"""Packed cell layout for STRATA info blocks.

One cell = 8B header + M x u32 follower words, stored as three
parallel tensors so every op is a plain vectorized gather:

    tag   int32[S]    : high 32 bits of the probe key, forced nonzero
                        (0 = empty).  The slot index already carries
                        ~log2(S) key bits, so a 32-bit tag keeps the
                        false-merge rate ~S * 2^-32 - negligible.
    meta  int32[S]    : nfoll:u8 | wtag:u8<<8 | flags:u8<<16 | pad:u8<<24
    foll  int32[S,M]  : one u32 per follower =
                        tok:u16 | cnt_q:u8<<16 | aux:u8<<24
                        aux = src_tag_lo4 | reserved_hi4  (provenance)

Empty follower word = -1 (0xFFFFFFFF).  Row totals are DERIVED -
sum of decoded counts - never stored, so a quantized count can
never drift from its own row total.

Counts are log-quantized: q = round(log2(1+c)*16), giving ~4.4%
relative steps over 0..~63k.  Decode = one 256-entry LUT gather.
"""
from __future__ import annotations

import torch

# ---- count quantization -------------------------------------------------

QSCALE = 16  # log2 subdivisions per octave -> ~4.4% relative steps


def count_lut(device=None) -> torch.Tensor:
    """q -> count decode table (256 x f32)."""
    return torch.expm1(
        torch.arange(256, dtype=torch.float32, device=device)
        * (0.6931471805599453 / QSCALE))


def quantize_counts(c: torch.Tensor) -> torch.Tensor:
    """count -> q in [0,255] (f32 in -> int32 out), log-domain."""
    q = torch.log2(c.clamp_min(0.0) + 1.0) * QSCALE
    return q.round().clamp(0, 255).to(torch.int32)


def decode_counts(q: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    return lut[(q.long() & 0xFF)]


# ---- follower word packing ----------------------------------------------

EMPTY_FOLL = -1  # int32 0xFFFFFFFF


def pack_follower(tok: torch.Tensor, cnt_q: torch.Tensor,
                  aux: torch.Tensor) -> torch.Tensor:
    """(tok u16, cnt_q u8, aux u8) -> int32 word."""
    w = ((tok.long() & 0xFFFF)
         | ((cnt_q.long() & 0xFF) << 16)
         | ((aux.long() & 0xFF) << 24))
    # values with tok=0xFFFF & cnt=0xFF & aux=0xFF would alias EMPTY_FOLL;
    # aux top nibble is reserved so callers can't produce it
    return w.to(torch.int32)


def foll_tok(foll: torch.Tensor) -> torch.Tensor:
    return foll & 0xFFFF


def foll_cntq(foll: torch.Tensor) -> torch.Tensor:
    return (foll >> 16) & 0xFF


def foll_aux(foll: torch.Tensor) -> torch.Tensor:
    return (foll >> 24) & 0xFF


def foll_valid(foll: torch.Tensor) -> torch.Tensor:
    return foll != EMPTY_FOLL


# ---- meta word packing ----------------------------------------------------


def pack_meta(nfoll: torch.Tensor, wtag: int, flags: int = 0) -> torch.Tensor:
    return ((nfoll.long() & 0xFF)
            | ((wtag & 0xFF) << 8)
            | ((flags & 0xFF) << 16)).to(torch.int32)


def meta_nfoll(meta: torch.Tensor) -> torch.Tensor:
    return meta & 0xFF


def meta_wtag(meta: torch.Tensor) -> torch.Tensor:
    return (meta >> 8) & 0xFF


def meta_flags(meta: torch.Tensor) -> torch.Tensor:
    return (meta >> 16) & 0xFF


# ---- key helpers ------------------------------------------------------------

_CELL_BYTES = 8  # tag + meta


def cell_bytes(M: int, raw: bool = False) -> int:
    return _CELL_BYTES + 4 * M + (8 if raw else 0)
