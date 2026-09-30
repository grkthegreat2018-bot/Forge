"""Base domain interface. Each domain defines a search space + evaluator.

To add a new search domain (quant, kernel, training, boot, etc.),
subclass BaseDomain and implement:
  - name(): domain name
  - output_dim(): number of continuous params generators produce
  - decode(params): map [0,1] params → concrete config dict
  - encode(config): map config dict → [0,1] params (for context)
  - evaluate(config): run the real evaluation, return {score, behavioral, metadata}
  - behavioral_dims(): MAP-Elites grid spec
  - discrete_choices(): optional discrete params for template generator
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class BaseDomain(ABC):
    """Abstract base for search domains."""

    def __init__(self):
        import torch
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        self._device = value

    def _t(self, *args, **kwargs):
        """Create a tensor on this domain's device."""
        import torch
        kwargs.setdefault('device', self._device)
        return torch.tensor(*args, **kwargs)

    def _randn(self, *args, **kwargs):
        """Create randn tensor on this domain's device."""
        import torch
        kwargs.setdefault('device', self._device)
        return torch.randn(*args, **kwargs)

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def output_dim(self) -> int:
        """Number of continuous [0,1] params the generators produce."""

    @abstractmethod
    def decode(self, params: torch.Tensor) -> dict[str, Any]:
        """Map [0,1] params → concrete config dict."""

    @abstractmethod
    def encode(self, config: dict[str, Any]) -> torch.Tensor:
        """Map config dict → [0,1] params (for context/surrogate)."""

    @abstractmethod
    def evaluate(self, config: dict[str, Any]) -> dict:
        """Evaluate a config. Return {score, behavioral, metadata}.
        score: float, higher = better
        behavioral: tuple of floats (for MAP-Elites grid)
        metadata: dict with any extra info
        """

    @abstractmethod
    def behavioral_dims(self) -> list[tuple[str, int, float, float]]:
        """MAP-Elites grid: [(name, n_bins, min, max), ...]"""

    def discrete_choices(self) -> dict[str, list] | None:
        """Optional discrete params for template generator."""
        return None

    def seed_configs(self) -> list[dict[str, Any]]:
        """Known-good configs to pre-evaluate before search begins.
        Override in subclass to bootstrap the surrogate + archive.
        """
        return []

    def to_cpu(self) -> BaseDomain:
        """Create a CPU-side copy of this domain for parallel evaluation.

        Override in subclasses that hold GPU tensors. The CPU copy should
        have all tensors moved to CPU so it can run in a separate process
        without touching the GPU.
        """
        return self


# ── Domain registry ──
# Import all domain modules so they're available via DOMAINS dict.
import importlib
import inspect


def _discover_domains() -> dict[str, type[BaseDomain]]:
    """Auto-discover all BaseDomain subclasses across all domain modules."""
    modules = [
        "synthetic", "quant", "kv_eviction", "sparse_attn", "hqe_kv", "kara",
        "attention_domains", "kv_domains", "quant_domains",
        "training_domains", "memory_domains", "decoding_domains", "arch_domains",
        "random_task_domain", "math_prog_domain",
    ]
    registry = {}
    for mod_name in modules:
        try:
            mod = importlib.import_module(f".{mod_name}", package=__name__)
            for name, obj in inspect.getmembers(mod, inspect.isclass):
                if (issubclass(obj, BaseDomain) and obj is not BaseDomain
                        and obj.__module__ == mod.__name__):
                    registry[obj.__name__] = obj
        except ImportError:
            pass
    return registry


def _discover_json_domains() -> dict[str, type[BaseDomain]]:
    """Auto-discover JSON-specified domains from configs/domains/*.json.

    Each JSON spec becomes a JSONSpecDomain subclass entry in the registry,
    keyed by a CamelCase class name (e.g. "w8a8_quant" -> "W8a8Quant").
    This allows JSON domains to be used everywhere a Python domain class is
    expected (run_evolve.py, rescore_db.py, etc.) without any code changes.
    """
    from ..domain_spec import JSONSpecDomain, list_specs
    registry = {}
    for spec_name in list_specs():
        try:
            cls_name = "".join(w.capitalize() for w in spec_name.split("_"))
            # Build a dynamic subclass whose __init__ binds the spec_name
            def _make_init(sn):
                def _init(self, *args, **kwargs):
                    kwargs.pop("seq_len", None)
                    kwargs.pop("seed", None)
                    kwargs.pop("device", None)
                    JSONSpecDomain.__init__(self, spec_name=sn, **kwargs)
                return _init
            cls = type(cls_name, (JSONSpecDomain,), {"__init__": _make_init(spec_name)})
            registry[cls_name] = cls
        except Exception:
            continue
    return registry

DOMAINS: dict[str, type[BaseDomain]] = _discover_domains()
# Merge in JSON-specified domains (they override Python classes with the same
# CamelCase name — JSON is the canonical source of truth when both exist)
DOMAINS.update(_discover_json_domains())

def get_domain(name: str, **kwargs) -> BaseDomain:
    """Instantiate a domain by class name."""
    if name not in DOMAINS:
        raise KeyError(f"Unknown domain '{name}'. Available: {list(DOMAINS.keys())}")
    return DOMAINS[name](**kwargs)

def list_domains() -> list[str]:
    """List all available domain class names."""
    return sorted(DOMAINS.keys())
