"""
optimizers/mcts.py

IMPORTANT CAVEAT: weight optimization doesn't have the discrete, sequential
move structure classic Monte Carlo Tree Search was designed for (there's no
natural "board state" or turn sequence over a network's continuous weight
space). What's implemented here is an explicitly *simplified*, UCB1-driven
adaptation, not textbook MCTS with multi-level tree expansion:

  - Tree structure: one level deep. Root = an individual's current weights.
    Children = "apply perturbation at scale s" for s in a fixed ladder of
    candidate scales (the "arms").
  - Selection: each generation, the arm to commit to is chosen by UCB1,
    balancing arms with a high historical reward against arms that haven't
    been tried much yet:

        UCB(arm) = mean_reward(arm) + c * sqrt(ln(total_visits) / visits(arm))

  - Expansion + rollout: the chosen perturbation is applied, then the
    individual gets the *same* local-training budget (epochs_per_gen of
    Adam) every other method in this suite gets, so the comparison isn't
    distorted by giving one method more or less gradient-descent compute.
  - Backup: the resulting post-training test accuracy updates that arm's
    running (visit_count, mean_reward) statistics.

Each population member keeps its own independent arm statistics, so over
generations different individuals can specialize on different perturbation
scales (e.g. one settles into "scale 0 = pure local fine-tuning" once its
weights are already good, another keeps exploring scale 0.25 if its early
rewards were poor).
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from datasets import build_dataloaders
from .base import Individual, HistoryWriter, OptConfig, RunResult

# ladder of perturbation scales: std-dev of additive Gaussian noise applied
# to each parameter tensor, expressed as a fraction of that tensor's own
# std-dev. 0.0 is "no perturbation, just keep training" -- always available
# as an arm so MCTS can never do worse than plain local gradient descent.
_ARM_SCALES = [0.0, 0.01, 0.05, 0.1, 0.25, 0.5]


class _ArmStats:
    def __init__(self) -> None:
        self.visits = 0
        self.mean_reward = 0.0

    def update(self, reward: float) -> None:
        self.visits += 1
        self.mean_reward += (reward - self.mean_reward) / self.visits

    def ucb(self, total_visits: int, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.mean_reward + c * math.sqrt(math.log(max(total_visits, 1)) / self.visits)


def _perturb_state(state: Dict[str, torch.Tensor], scale: float) -> Dict[str, torch.Tensor]:
    if scale == 0.0:
        return {k: v.clone() for k, v in state.items()}
    out = {}
    for key, tensor in state.items():
        noise = torch.randn_like(tensor) * tensor.std() * scale
        out[key] = tensor + noise
    return out


def run(cfg: OptConfig) -> RunResult:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    train_loader, test_loader, meta = build_dataloaders(
        cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.device)

    out_dir = Path(cfg.output_dir) / f"{cfg.dataset}_mcts"
    out_dir.mkdir(parents=True, exist_ok=True)
    history = HistoryWriter(out_dir / "history.csv", cfg.population_size)

    n_arms = min(cfg.mcts_arms, len(_ARM_SCALES))
    arm_scales = _ARM_SCALES[:n_arms]

    population = [Individual(meta, cfg) for _ in range(cfg.population_size)]
    arm_stats: List[List[_ArmStats]] = [[_ArmStats() for _ in arm_scales] for _ in population]
    best_acc, best_state = -1.0, None
    t_start = time.time()

    for gen in range(cfg.generations):
        t0 = time.time()
        for i, member in enumerate(population):
            stats = arm_stats[i]
            total_visits = sum(s.visits for s in stats)
            arm_idx = int(np.argmax([s.ucb(total_visits, cfg.mcts_c) for s in stats]))
            scale = arm_scales[arm_idx]

            perturbed_state = _perturb_state(member.state_dict(), scale)
            member.load_state(perturbed_state)
            for _ in range(cfg.epochs_per_gen):
                member.train_one_epoch(train_loader)
            reward = member.evaluate(test_loader)

            stats[arm_idx].update(reward)

        fitness = [m.last_fitness for m in population]
        best_idx = int(np.argmax(fitness))
        if fitness[best_idx] > best_acc:
            best_acc = fitness[best_idx]
            best_state = population[best_idx].state_dict()
            torch.save(best_state, out_dir / "best_model.pt")

        history.write_row(gen, fitness, best_acc, time.time() - t0)
        print(f"[mcts/{cfg.dataset}] gen {gen + 1}/{cfg.generations} "
              f"best={fitness[best_idx]:.4f} best_ever={best_acc:.4f}")

    history.close()
    return RunResult("mcts", cfg.dataset, best_acc, out_dir / "history.csv",
                      out_dir / "best_model.pt", out_dir, time.time() - t_start)
