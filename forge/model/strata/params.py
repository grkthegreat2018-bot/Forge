"""ParamSet - the IQ layer (.iq).

Named tensors only - the small fixed compute machinery that turns
cell votes into logits (mixer weights, fingerprint table, optional
dense channel).  Seeded tensors (e.g. the _R randproj fingerprint
bank) are stored as {"seed": ...} derivations, not bytes - a 4 MB
table costs 8 bytes on disk and regenerates bit-exact.

Behavior rule: params decide HOW info is used, never WHAT is stored.
Swapping the .iq file upgrades the brain without touching memory.
"""
from __future__ import annotations

from collections import OrderedDict

import torch

from .io import save_container, load_container


class ParamSet:
    def __init__(self):
        self.tensors: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.derived: dict = {}   # name -> {"seed":int,"shape":[],
                                  #         "dtype":str}
        self.version: int = 1

    def set(self, name: str, t: torch.Tensor) -> None:
        self.tensors[name] = t.detach()

    def set_seeded(self, name: str, shape, seed: int,
                   dtype: str = "float32") -> torch.Tensor:
        self.derived[name] = {"seed": int(seed), "shape": list(shape),
                              "dtype": dtype}
        return self._derive(name)

    def _derive(self, name: str) -> torch.Tensor:
        d = self.derived[name]
        g = torch.Generator().manual_seed(d["seed"])
        t = torch.randn(d["shape"], generator=g)
        if d["dtype"] == "int8":
            t = (t * 24).round().clamp(-127, 127).to(torch.int8)
        return t

    def get(self, name: str, device=None) -> torch.Tensor:
        if name in self.tensors:
            t = self.tensors[name]
        elif name in self.derived:
            t = self._derive(name)
        else:
            raise KeyError(name)
        return t.to(device) if device is not None else t

    def names(self) -> list:
        return list(self.tensors) + [n for n in self.derived
                                     if n not in self.tensors]

    # -- io -----------------------------------------------------------------

    def save(self, path) -> int:
        meta = {"kind": "params", "version": self.version,
                "derived": self.derived}
        return save_container(path, self.tensors, meta)

    @classmethod
    def load(cls, path, device: str | None = None) -> "ParamSet":
        meta, tensors = load_container(path, device=device)
        p = cls()
        p.version = meta.get("version", 1)
        p.derived = meta.get("derived", {})
        p.tensors = OrderedDict(tensors)
        return p
