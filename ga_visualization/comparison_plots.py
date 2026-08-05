"""
ga_visualization/comparison_plots.py

Cross-method, cross-dataset comparison plots consumed by compare_all.py:
  - a grouped bar chart (dataset on the x-axis, one bar per method),
    reproducing the shape of a results table (like the Adam/GA/DE/MCTS
    table you're benchmarking against) as a figure
  - one convergence-overlay plot per dataset, showing best-accuracy-so-far
    for every method on the same axes, so you can see which one wins and
    how fast it gets there
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .utils import save_fig

METHOD_COLORS = {"adam": "#888888", "ga": "#4C72B0", "de": "#55A868", "mcts": "#C44E52"}
METHOD_LABELS = {"adam": "Adam", "ga": "GA", "de": "DE", "mcts": "MCTS"}


def plot_comparison_bar_chart(pivot: pd.DataFrame, output_dir) -> None:
    """pivot: rows = dataset name, columns = method, values = best test
    accuracy in percent (as produced by compare_all.py)."""
    datasets = list(pivot.index)
    methods = list(pivot.columns)
    n_methods = max(len(methods), 1)
    x = np.arange(len(datasets))
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(1.8 * len(datasets) + 2, 5))
    for i, method in enumerate(methods):
        values = pivot[method].to_numpy()
        offset = (i - (n_methods - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=METHOD_LABELS.get(method, method),
                       color=METHOD_COLORS.get(method, None))
        ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)

    ax.set_xticks(x); ax.set_xticklabels(datasets)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Optimizer comparison across datasets")
    ax.legend()
    top = float(np.nanmax(pivot.to_numpy())) if pivot.size else 100.0
    ax.set_ylim(0, min(100, top * 1.15))
    save_fig(fig, "comparison_bar_chart", output_dir)


def plot_convergence_overlay(df: pd.DataFrame, output_dir) -> None:
    """df: one row per (dataset, method) run, with a 'history_csv' column
    pointing at that run's per-generation history (as produced by
    optimizers/base.py's HistoryWriter -- same schema for every method)."""
    for dataset, group in df.groupby("dataset"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for _, row in group.iterrows():
            hist = pd.read_csv(row["history_csv"])
            ax.plot(hist["generation"], hist["best_so_far"] * 100,
                    label=METHOD_LABELS.get(row["method"], row["method"]),
                    color=METHOD_COLORS.get(row["method"], None), lw=2)
        ax.set_xlabel("Generation"); ax.set_ylabel("Best test accuracy so far (%)")
        ax.set_title(f"Convergence comparison \u2014 {dataset}")
        ax.legend()
        save_fig(fig, f"convergence_overlay_{dataset}", output_dir)
