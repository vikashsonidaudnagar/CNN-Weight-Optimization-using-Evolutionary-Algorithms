"""
optimizers/adam_baseline.py

Plain-gradient-descent baseline: a single model trained with Adam for
generations * epochs_per_gen total epochs, no population, no weight
recombination. This is the reference point GA/DE/MCTS get compared
against -- if they can't beat this, the extra evolutionary machinery
isn't buying anything for that dataset.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from datasets import build_dataloaders
from .base import Individual, HistoryWriter, OptConfig, RunResult


def run(cfg: OptConfig) -> RunResult:
    torch.manual_seed(cfg.seed)
    train_loader, test_loader, meta = build_dataloaders(
        cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.device)

    out_dir = Path(cfg.output_dir) / f"{cfg.dataset}_adam"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryWriter(out_dir / "history.csv", population_size=1)

    individual = Individual(meta, cfg)
    best_acc, best_state = -1.0, None
    t_start = time.time()

    for gen in range(cfg.generations):
        t0 = time.time()
        for _ in range(cfg.epochs_per_gen):
            individual.train_one_epoch(train_loader)
        acc = individual.evaluate(test_loader)
        if acc > best_acc:
            best_acc = acc
            best_state = individual.state_dict()
            torch.save(best_state, out_dir / "best_model.pt")
        history.write_row(gen, [acc], best_acc, time.time() - t0)
        print(f"[adam/{cfg.dataset}] gen {gen + 1}/{cfg.generations} "
              f"acc={acc:.4f} best={best_acc:.4f}")

    history.close()
    return RunResult("adam", cfg.dataset, best_acc, out_dir / "history.csv",
                      out_dir / "best_model.pt", out_dir, time.time() - t_start)
