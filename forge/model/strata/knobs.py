"""KnobSet - the conditioning layer (.cond).

Flat named-scalar registry + overlay presets.  Every behavioral
threshold/gate/lift is a knob, so conditioning changes NEVER touch
weights or params: ingest writes cells, predict reads knobs.

Layers: base values + a stack of named overlays (personality presets,
per-request tweaks).  Resolution order: topmost overlay that defines
the name wins, else base.  Overlays are the unit of A/B swapping.
"""
from __future__ import annotations

import json
from pathlib import Path


class KnobSet:
    def __init__(self, base: dict | None = None):
        self.base: dict = dict(base or {})
        self.overlays: list[tuple[str, dict]] = []

    # -- resolution ---------------------------------------------------------

    def get(self, name: str, default=None):
        for _, ov in reversed(self.overlays):
            if name in ov:
                return ov[name]
        return self.base.get(name, default)

    def __getitem__(self, name: str):
        v = self.get(name, None)
        if v is None:
            raise KeyError(name)
        return v

    def set(self, name: str, value) -> None:
        self.base[name] = value

    def resolve(self) -> dict:
        out = dict(self.base)
        for _, ov in self.overlays:
            out.update(ov)
        return out

    # -- overlays ------------------------------------------------------------

    def push(self, name: str, values: dict) -> None:
        self.overlays.append((name, dict(values)))

    def pop(self, name: str | None = None) -> dict | None:
        if not self.overlays:
            return None
        if name is None:
            return self.overlays.pop()[1]
        for i in range(len(self.overlays) - 1, -1, -1):
            if self.overlays[i][0] == name:
                return self.overlays.pop(i)[1]
        return None

    # -- io -------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {"kind": "knobs", "base": self.base,
                "overlays": [[n, v] for n, v in self.overlays]}

    @classmethod
    def from_dict(cls, d: dict) -> "KnobSet":
        k = cls(d["base"])
        k.overlays = [(n, v) for n, v in d.get("overlays", [])]
        return k

    def save(self, path) -> int:
        p = Path(path)
        p.write_text(json.dumps(self.to_dict(), separators=(",", ":")))
        return p.stat().st_size

    @classmethod
    def load(cls, path) -> "KnobSet":
        return cls.from_dict(json.loads(Path(path).read_text()))


# Default knob surface (documents the contract for the model layer).
DEFAULT_KNOBS = {
    "thin_tot": 2.0,          # cell mass that counts as real evidence
    "surprise_floor": 0.05,   # minimum write magnitude
    "uni_escape": 0.12,       # arithmetic unigram escape mass
    "mem_bias": 1.0,          # recall-vs-gen dial
    "support_thresh": 1.5,    # sampled tok must beat this * p_uni
    "temp": 1.0,
    "top_p": 1.0,
    "top_k": 0,
}
