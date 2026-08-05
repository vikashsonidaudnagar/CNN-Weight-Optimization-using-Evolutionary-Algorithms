"""
optimizers/__init__.py

Registry mapping method names to their run() functions, so run_experiment.py
and compare_all.py can dispatch on a plain string ("adam" | "ga" | "de" |
"mcts") without importing each module by hand.
"""

from __future__ import annotations

from . import adam_baseline, genetic_algorithm, differential_evolution, mcts
from .base import OptConfig, RunResult

REGISTRY = {
    "adam": adam_baseline.run,
    "ga": genetic_algorithm.run,
    "de": differential_evolution.run,
    "mcts": mcts.run,
}


def available_methods():
    return sorted(REGISTRY.keys())


def run_method(cfg: OptConfig) -> RunResult:
    if cfg.method not in REGISTRY:
        raise ValueError(f"Unknown method {cfg.method!r}. Available: {available_methods()}")
    return REGISTRY[cfg.method](cfg)


__all__ = ["OptConfig", "RunResult", "REGISTRY", "available_methods", "run_method"]
