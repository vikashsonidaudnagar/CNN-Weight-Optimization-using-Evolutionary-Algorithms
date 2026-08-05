"""
optimizers/genetic_algorithm.py

Population-based weight evolution: fitness-proportionate selection,
single-point crossover on the flattened list of parameter tensors, and
random uniform-reinitialization mutation, with elitism (the current best
individual is carried over untouched each generation so the population
can never regress below its best-so-far). Same algorithm as the original
ga_weight_opti_torch.py, generalized to any dataset via
DatasetMeta/WeightOptCNN instead of a hardcoded MNIST-only CNN.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from datasets import build_dataloaders
from .base import Individual, HistoryWriter, OptConfig, RunResult


def _selection_probs(fitness: List[float]) -> np.ndarray:
    total = sum(fitness)
    if total <= 0:
        return np.full(len(fitness), 1.0 / len(fitness))
    return np.array(fitness) / total


def _mutate(individual: Individual, mutation_rate: float) -> None:
    with torch.no_grad():
        for param in individual.model.parameters():
            if random.random() < mutation_rate:
                param.copy_(torch.empty_like(param).uniform_(-1, 1))


def run(cfg: OptConfig) -> RunResult:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_loader, test_loader, meta = build_dataloaders(
        cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.device)

    out_dir = Path(cfg.output_dir) / f"{cfg.dataset}_ga"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryWriter(out_dir / "history.csv", cfg.population_size)

    population = [Individual(meta, cfg) for _ in range(cfg.population_size)]
    best_acc, best_state = -1.0, None
    t_start = time.time()

    for gen in range(cfg.generations):
        t0 = time.time()
        for member in population:
            for _ in range(cfg.epochs_per_gen):
                member.train_one_epoch(train_loader)

        fitness = [member.evaluate(test_loader) for member in population]
        best_idx = int(np.argmax(fitness))
        if fitness[best_idx] > best_acc:
            best_acc = fitness[best_idx]
            best_state = population[best_idx].state_dict()
            torch.save(best_state, out_dir / "best_model.pt")

        history.write_row(gen, fitness, best_acc, time.time() - t0)
        print(f"[ga/{cfg.dataset}] gen {gen + 1}/{cfg.generations} "
              f"best={fitness[best_idx]:.4f} best_ever={best_acc:.4f}")

        if gen != cfg.generations - 1:
            probs = _selection_probs(fitness)
            pop_idx = list(range(len(population)))
            children_states = []
            for i in range(len(population)):
                if i == best_idx:
                    children_states.append(population[i].state_dict())
                    continue
                p1, p2 = np.random.choice(pop_idx, size=2, p=probs)
                s1, s2 = population[p1].state_dict(), population[p2].state_dict()
                keys = list(s1.keys())
                mid = np.random.randint(1, len(keys))
                child = {k: (s1[k] if idx < mid else s2[k]) for idx, k in enumerate(keys)}
                children_states.append(child)

            for i, member in enumerate(population):
                member.load_state(children_states[i])
            for i, member in enumerate(population):
                if i != best_idx:
                    _mutate(member, cfg.mutation_rate)

    history.close()
    return RunResult("ga", cfg.dataset, best_acc, out_dir / "history.csv",
                      out_dir / "best_model.pt", out_dir, time.time() - t_start)
