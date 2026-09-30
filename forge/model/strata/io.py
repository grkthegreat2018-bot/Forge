""".strata container - disk layout IS the memory layout.

    [8B magic "STRATA01"][u64 header_len][JSON header][pad->64B][slabs]

Header JSON carries the full manifest (block labels, tags, config)
plus a tensor index: {name -> {off, len, dtype, shape}}.  Slabs are
raw little-endian tensor bytes, each 64B-aligned so a load is one
memmap + per-tensor views - no deserialize, no per-key seek.  Same
trick as checkpoint_io's pipelined loader, minus the safetensors tax.

Mutability rule: memmap views are read-only, so a loaded-for-mutation
store copies (memcpy - still GB/s).  ``frozen=True`` keeps the views
for sealed/read-only inspection loads.

Used for all three artifact types:
  weights  .strata  (BlockStore slabs)
  params   .iq      (ParamSet tensors)
  knobs    .cond    (JSON only - no slabs)
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch

MAGIC = b"STRATA01"
_ALIGN = 64

_DT = {torch.int32: np.int32, torch.int64: np.int64,
       torch.float32: np.float32, torch.float16: np.float16,
       torch.uint8: np.uint8, torch.int16: np.int16}
_NP2T = {np.dtype(np.int32): torch.int32, np.dtype(np.int64): torch.int64,
         np.dtype(np.float32): torch.float32,
         np.dtype(np.float16): torch.float16,
         np.dtype(np.uint8): torch.uint8, np.dtype(np.int16): torch.int16}


def save_container(path, tensors: dict, meta: dict) -> int:
    """Write header+slabs.  Returns file size in bytes."""
    path = Path(path)
    recs, blobs, off = [], [], 0
    for name, t in tensors.items():
        t = t.detach().contiguous().cpu()
        if t.dtype == torch.bfloat16:
            np_dt, arr = "bf16", t.view(torch.uint8).numpy()
        else:
            np_dt = _DT[t.dtype]
            arr = t.numpy().view(np_dt)
        b = arr.tobytes()
        pad = (-off) % _ALIGN
        if pad:
            blobs.append(b"\0" * pad)
            off += pad
        recs.append({"name": name, "off": off, "len": len(b),
                     "dtype": np_dt if np_dt == "bf16"
                     else np.dtype(np_dt).name,
                     "shape": list(t.shape)})
        blobs.append(b)
        off += len(b)
    hdr = json.dumps({"meta": meta, "tensors": recs},
                     separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(b"\0" * ((-len(hdr)) % _ALIGN))
        for b in blobs:
            f.write(b)
    return path.stat().st_size


def load_container(path, device: str | None = None,
                   copy: bool = False):
    """-> (meta, {name: tensor}).

    copy=False: zero-copy memmap views (read-only).  copy=True: host
    memcpy (writable).  device="cuda": one H2D upload per tensor.
    """
    path = Path(path)
    with open(path, "rb") as f:
        if f.read(8) != MAGIC:
            raise ValueError(f"{path}: not a .strata container")
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    base = 16 + hlen + ((-hlen) % _ALIGN)
    size = path.stat().st_size
    buf = np.memmap(path, dtype=np.uint8, mode="r", shape=(size,))
    out = {}
    for r in hdr["tensors"]:
        slab = buf[base + r["off"]: base + r["off"] + r["len"]]
        if r["dtype"] == "bf16":
            arr = np.frombuffer(slab, dtype=np.uint8)
            t = torch.from_numpy(
                arr.reshape(r["shape"])).view(torch.bfloat16)
        else:
            np_dt = np.dtype(r["dtype"])
            arr = np.frombuffer(slab, dtype=np_dt).reshape(r["shape"])
            t = torch.from_numpy(arr)
        if copy:
            t = t.clone()
        if device is not None:
            t = t.to(device, non_blocking=True)
        out[r["name"]] = t
    return hdr["meta"], out


def save_store(path, store, extra_meta: dict | None = None) -> int:
    meta = {"kind": "weights", "store": store.manifest()}
    if extra_meta:
        meta.update(extra_meta)
    return save_container(path, store.tensors(), meta)


def load_store(path, device: str = "cpu", frozen: bool = False):
    """frozen=True keeps zero-copy memmap cells (read-only - sealed
    blocks/inspection only).  Otherwise copies for mutation."""
    from .blocks import BlockStore
    meta, tensors = load_container(
        path, device=(device if device != "cpu" else None),
        copy=(device == "cpu" and not frozen))
    return BlockStore.from_manifest(meta["store"], tensors, dev=device)
