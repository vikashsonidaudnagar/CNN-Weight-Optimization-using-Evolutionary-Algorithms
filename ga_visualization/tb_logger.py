"""
ga_visualization/tb_logger.py

Thin wrapper around torch.utils.tensorboard.SummaryWriter so the rest of
the codebase doesn't need to import TensorBoard directly, and so it fails
gracefully (with a log message, not a crash) if the tensorboard package
isn't installed in a given environment.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from .utils import shannon_entropy, to_probabilities, logger

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TB = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_TB = False


class GATensorBoardLogger:
    """Call `log_generation` once per GA generation. Writes:
      - scalar best / mean / worst / std / entropy of fitness
      - a histogram of the raw per-member fitness values
    """

    def __init__(self, log_dir):
        self.enabled = _HAS_TB
        if self.enabled:
            self.writer = SummaryWriter(log_dir=str(log_dir))
        else:
            logger.info("tensorboard not installed; skipping TB logging")

    def log_generation(self, generation: int, fitness: List[float], best_so_far: float) -> None:
        if not self.enabled:
            return
        arr = np.asarray(fitness, dtype=np.float64)
        self.writer.add_scalar("fitness/best", arr.max(), generation)
        self.writer.add_scalar("fitness/mean", arr.mean(), generation)
        self.writer.add_scalar("fitness/worst", arr.min(), generation)
        self.writer.add_scalar("fitness/std", arr.std(), generation)
        self.writer.add_scalar("fitness/entropy_bits", shannon_entropy(to_probabilities(arr)), generation)
        self.writer.add_scalar("fitness/best_so_far", best_so_far, generation)
        self.writer.add_histogram("fitness/distribution", arr, generation)

    def close(self) -> None:
        if self.enabled:
            self.writer.close()
