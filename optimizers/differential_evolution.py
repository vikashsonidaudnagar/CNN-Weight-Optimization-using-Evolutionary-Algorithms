"""
optimizers/differential_evolution.py

DE/rand/1/bin applied directly to network weights, hybridized with local
gradient training so the comparison against GA/MCTS/Adam is apples-to-
apples on gradient-descent budget. Each generation, for target x_i, three
other distinct population members a, b, c are drawn and a mutant vector is
formed elementwise per parameter tensor:

    v = a + F * (b - c)

Binomial crossover mixes v into x_i per-parameter with probability CR to
produce a trial u_i. u_i is evaluated *before* any local training; if it's
at least as good as x_i's current fitness it replaces x_i (classic DE's
greedy/elitist selection), otherwise x_i is kept. Either way, the surviving
individual then gets its epochs_per_gen of local Adam training, same as
every other method.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from datasets import build_dataloaders
from .base import Individual, HistoryWriter, OptConfig, RunResult


def _de_trial_state(target: Dict[str, torch.Tensor], a: Dict[str, torch.Tensor],
                     b: Dict[str, torch.Tensor], c: Dict[str, torch.Tensor],
                     F: float, CR: float) -> Dict[str, torch.Tensor]:
    trial = {}
    for key in target.keys():
        mutant = a[key] + F * (b[key] - c[key])
        mask = (torch.rand_like(target[key].float()) < CR)
        trial[key] = torch.where(mask, mutant, target[key])
    return trial


def run(cfg: OptConfig) -> RunResult:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_loader, test_loader, meta = build_dataloaders(
        cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.device)

    out_dir = Path(cfg.output_dir) / f"{cfg.dataset}_de"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryWriter(out_dir / "history.csv", cfg.population_size)

    population = [Individual(meta, cfg) for _ in range(cfg.population_size)]
    best_acc, best_state = -1.0, None
    t_start = time.time()

    # seed fitness with an initial evaluation so generation 0's DE
    # comparisons (trial vs. incumbent) have something to compare against
    for member in population:
        member.evaluate(test_loader)

    # scratch model used only to *score* a candidate trial vector before
    # deciding whether it's worth the current generation's training budget
    scratch = Individual(meta, cfg)

    for gen in range(cfg.generations):
        t0 = time.time()
        pop_idx = list(range(len(population)))
        states = [m.state_dict() for m in population]

        for i, member in enumerate(population):
            a_idx, b_idx, c_idx = random.sample([j for j in pop_idx if j != i], 3)
            trial_state = _de_trial_state(states[i], states[a_idx], states[b_idx],
                                           states[c_idx], cfg.de_F, cfg.de_CR)

            scratch.load_state(trial_state)
            trial_acc_raw = scratch.evaluate(test_loader)
            if trial_acc_raw >= member.last_fitness:
                member.load_state(trial_state)

            for _ in range(cfg.epochs_per_gen):
                member.train_one_epoch(train_loader)

        fitness = [member.evaluate(test_loader) for member in population]
        best_idx = int(np.argmax(fitness))
        if fitness[best_idx] > best_acc:
            best_acc = fitness[best_idx]
            best_state = population[best_idx].state_dict()
            torch.save(best_state, out_dir / "best_model.pt")

        history.write_row(gen, fitness, best_acc, time.time() - t0)
        print(f"[de/{cfg.dataset}] gen {gen + 1}/{cfg.generations} "
              f"best={fitness[best_idx]:.4f} best_ever={best_acc:.4f}")

    history.close()
    return RunResult("de", cfg.dataset, best_acc, out_dir / "history.csv",
                      out_dir / "best_model.pt", out_dir, time.time() - t_start)
