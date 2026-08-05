"""
ga_visualization/evolution_plots.py

Everything derived purely from the GA's per-generation bookkeeping
(history.csv, written by GeneticAlgorithm.run()) -- no model or test data
needed here, just the fitness numbers themselves.

history.csv columns: generation, member_0 ... member_{P-1}, best_so_far
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import ensure_dir, save_fig, shannon_entropy, to_probabilities, logger

try:
    import imageio.v2 as imageio
    _HAS_IMAGEIO = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_IMAGEIO = False


def _member_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("member_")]


# ---------------------------------------------------------------------------
# Best / mean / worst accuracy convergence
# ---------------------------------------------------------------------------
def plot_accuracy_convergence(history_csv, output_dir) -> None:
    df = pd.read_csv(history_csv)
    member_cols = _member_columns(df)
    fitness = df[member_cols].to_numpy()

    best = fitness.max(axis=1)
    mean = fitness.mean(axis=1)
    worst = fitness.min(axis=1)
    gens = df["generation"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gens, best, label="best", color="forestgreen", lw=2)
    ax.plot(gens, mean, label="mean", color="steelblue", lw=2)
    ax.plot(gens, worst, label="worst", color="firebrick", lw=2)
    ax.plot(gens, df["best_so_far"], label="best-so-far (elite)", color="black",
            lw=1.5, linestyle="--")
    ax.fill_between(gens, worst, best, color="steelblue", alpha=0.1)
    ax.set_xlabel("Generation"); ax.set_ylabel("Test accuracy")
    ax.set_title("GA accuracy convergence")
    ax.legend()
    save_fig(fig, "accuracy_convergence", output_dir)


# ---------------------------------------------------------------------------
# Population diversity: std / variance / entropy per generation
# ---------------------------------------------------------------------------
def plot_population_diversity(history_csv, output_dir) -> None:
    df = pd.read_csv(history_csv)
    member_cols = _member_columns(df)
    fitness = df[member_cols].to_numpy()
    gens = df["generation"].to_numpy()

    std = fitness.std(axis=1)
    var = fitness.var(axis=1)
    entropy = np.array([shannon_entropy(to_probabilities(row)) for row in fitness])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(gens, std, color="darkorange", marker="o", ms=3)
    axes[0].set_title("Std. dev of fitness"); axes[0].set_xlabel("Generation")

    axes[1].plot(gens, var, color="purple", marker="o", ms=3)
    axes[1].set_title("Variance of fitness"); axes[1].set_xlabel("Generation")

    axes[2].plot(gens, entropy, color="teal", marker="o", ms=3)
    axes[2].set_title("Shannon entropy of fitness dist. (bits)")
    axes[2].set_xlabel("Generation")

    fig.suptitle("Population diversity across generations")
    fig.tight_layout()
    save_fig(fig, "population_diversity", output_dir)


# ---------------------------------------------------------------------------
# Fitness distribution histogram, one per generation, as a small-multiple grid
# ---------------------------------------------------------------------------
def plot_fitness_histograms(history_csv, output_dir, max_panels: int = 20) -> None:
    df = pd.read_csv(history_csv)
    member_cols = _member_columns(df)
    fitness = df[member_cols].to_numpy()
    n_gens = fitness.shape[0]

    # if there are more generations than we want panels for, subsample evenly
    gen_idxs = np.linspace(0, n_gens - 1, min(n_gens, max_panels)).astype(int)
    gen_idxs = sorted(set(gen_idxs.tolist()))

    n_cols = 5
    n_rows = int(np.ceil(len(gen_idxs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.6, n_rows * 2.2))
    axes = np.atleast_2d(axes)

    x_min, x_max = fitness.min(), fitness.max()
    for panel_i, gen_i in enumerate(gen_idxs):
        ax = axes[panel_i // n_cols, panel_i % n_cols]
        ax.hist(fitness[gen_i], bins=10, range=(x_min, x_max), color="slateblue")
        ax.set_title(f"gen {int(df['generation'][gen_i])}", fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(len(gen_idxs), n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")

    fig.suptitle("Per-generation fitness distribution")
    fig.tight_layout()
    save_fig(fig, "fitness_histograms_by_generation", output_dir)


# ---------------------------------------------------------------------------
# Optional: animated GIF of the convergence plot building up generation by
# generation, useful for talks / supplementary material.
# ---------------------------------------------------------------------------
def make_evolution_gif(history_csv, output_dir, fps: int = 4) -> None:
    if not _HAS_IMAGEIO:
        logger.info("imageio not installed, skipping evolution GIF")
        return

    df = pd.read_csv(history_csv)
    member_cols = _member_columns(df)
    fitness = df[member_cols].to_numpy()
    gens = df["generation"].to_numpy()
    best, mean, worst = fitness.max(axis=1), fitness.mean(axis=1), fitness.min(axis=1)
    y_min, y_max = fitness.min(), fitness.max()

    out_dir = ensure_dir(output_dir)
    frame_paths = []
    for t in range(1, len(gens) + 1):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(gens[:t], best[:t], label="best", color="forestgreen")
        ax.plot(gens[:t], mean[:t], label="mean", color="steelblue")
        ax.plot(gens[:t], worst[:t], label="worst", color="firebrick")
        ax.set_xlim(gens.min(), gens.max())
        ax.set_ylim(max(0, y_min - 0.05), min(1, y_max + 0.05))
        ax.set_xlabel("Generation"); ax.set_ylabel("Test accuracy")
        ax.set_title(f"GA evolution (generation {gens[t - 1]})")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()
        frame_path = out_dir / f"_frame_{t:03d}.png"
        fig.savefig(frame_path, dpi=120)
        plt.close(fig)
        frame_paths.append(frame_path)

    gif_path = out_dir / "evolution.gif"
    with imageio.get_writer(gif_path, mode="I", fps=fps) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
    for frame_path in frame_paths:
        frame_path.unlink()
    logger.info(f"saved {gif_path.name}")
