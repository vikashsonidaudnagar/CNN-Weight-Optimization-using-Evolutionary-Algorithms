"""
optimizers/base.py

Shared building blocks used by every optimizer variant (Adam baseline, GA,
DE, MCTS): a trainable Individual wrapper, a common OptConfig, and a
HistoryWriter that emits the *same* CSV schema for every method. Keeping
the schema identical is what lets ga_visualization's existing
evolution_plots.py (accuracy convergence / diversity / fitness histograms)
work unmodified regardless of which optimizer produced the run, and is
what lets compare_all.py overlay all four methods' convergence curves on
one axis.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import DatasetMeta
from models import WeightOptCNN


@dataclass
class OptConfig:
    dataset: str
    method: str                 # "adam" | "ga" | "de" | "mcts"
    population_size: int = 10   # ignored by "adam" (always 1)
    generations: int = 20
    epochs_per_gen: int = 1
    batch_size: int = 256
    num_workers: int = 4
    lr: float = 1e-3
    data_dir: str = "./data"
    output_dir: str = "./runs"
    seed: int = 42
    device: str = "cpu"

    # GA-specific
    mutation_rate: float = 0.1
    # DE-specific (DE/rand/1/bin)
    de_F: float = 0.5           # differential weight
    de_CR: float = 0.7          # crossover probability
    # MCTS-specific (UCB1-over-perturbation-arms formulation; see mcts.py)
    mcts_arms: int = 4
    mcts_c: float = 1.4


class Individual:
    """One model + its own Adam optimizer. Every method uses this for the
    local gradient-refinement step it performs each generation, so all
    four methods get exactly the same amount of gradient-descent budget
    and differ only in how population weights are recombined/perturbed
    between those local-training steps."""

    def __init__(self, meta: DatasetMeta, cfg: OptConfig):
        self.cfg = cfg
        self.model = WeightOptCNN(meta).to(cfg.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.last_fitness: float = -1.0

    def train_one_epoch(self, loader: DataLoader) -> None:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        for xb, yb in loader:
            xb = xb.to(self.cfg.device, non_blocking=True)
            yb = yb.to(self.cfg.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            loss = criterion(self.model(xb), yb)
            loss.backward()
            self.optimizer.step()

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0
        for xb, yb in loader:
            xb = xb.to(self.cfg.device, non_blocking=True)
            yb = yb.to(self.cfg.device, non_blocking=True)
            preds = self.model(xb).argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        acc = correct / total
        self.last_fitness = acc
        return acc

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.model.load_state_dict(state)


class HistoryWriter:
    """Common CSV schema for every method:
    generation, member_0..member_{P-1}, best_so_far, elapsed_sec
    (P=1 for the Adam baseline)."""

    def __init__(self, path: Path, population_size: int):
        self.path = path
        self._file = open(path, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["generation"] + [f"member_{i}" for i in range(population_size)]
            + ["best_so_far", "elapsed_sec"]
        )

    def write_row(self, generation: int, fitness: List[float], best_so_far: float, elapsed: float) -> None:
        self._writer.writerow([generation] + list(fitness) + [best_so_far, elapsed])
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@dataclass
class RunResult:
    method: str
    dataset: str
    best_acc: float
    history_csv: Path
    best_model_path: Path
    output_dir: Path
    total_time_sec: float
