"""
compare_all.py

Runs every method (adam, ga, de, mcts) against every requested dataset,
collects each run's best test accuracy into one comparison table (CSV +
Markdown, shaped like the Adam/GA/DE/MCTS table you're benchmarking
against), and renders a grouped bar chart plus a per-dataset convergence
overlay so all four methods can be read off the same axes.

This is a lot of compute (methods x datasets x generations x population),
so the defaults are deliberately small. Bump them up once you've confirmed
the whole pipeline runs end-to-end on your machine:

    python compare_all.py --generations 5 --population-size 6          # smoke test
    python compare_all.py --generations 20 --population-size 10        # real run
    python compare_all.py --datasets mnist cifar10 --methods ga de     # subset
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch

from datasets import available_datasets
from optimizers import OptConfig, available_methods, run_method
from ga_visualization.comparison_plots import plot_comparison_bar_chart, plot_convergence_overlay


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run every method x every dataset and compare")
    p.add_argument("--datasets", nargs="+", default=available_datasets(), choices=available_datasets())
    p.add_argument("--methods", nargs="+", default=available_methods(), choices=available_methods())
    p.add_argument("--population-size", type=int, default=8)
    p.add_argument("--generations", type=int, default=15)
    p.add_argument("--epochs-per-gen", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", None])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  datasets={args.datasets}  methods={args.methods}")

    results = []
    for dataset in args.datasets:
        for method in args.methods:
            print(f"\n=== {dataset} / {method} ===")
            cfg = OptConfig(
                dataset=dataset, method=method,
                population_size=1 if method == "adam" else args.population_size,
                generations=args.generations, epochs_per_gen=args.epochs_per_gen,
                batch_size=args.batch_size, num_workers=args.num_workers, lr=args.lr,
                data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed,
                device=device,
            )
            t0 = time.time()
            result = run_method(cfg)
            elapsed = time.time() - t0
            print(f"--> best_acc={result.best_acc:.4f}  ({elapsed:.1f}s)")
            results.append({
                "dataset": dataset,
                "method": method,
                "best_acc": result.best_acc,
                "best_acc_pct": round(result.best_acc * 100, 2),
                "time_sec": round(elapsed, 1),
                "history_csv": str(result.history_csv),
            })

    df = pd.DataFrame(results)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "comparison_results.csv", index=False)

    method_order = [m for m in available_methods() if m in df["method"].unique()]
    pivot = df.pivot(index="dataset", columns="method", values="best_acc_pct")
    pivot = pivot.reindex(columns=method_order)
    pivot.to_csv(out_dir / "comparison_table.csv")

    print("\n=== Comparison table (test accuracy, %) ===")
    print(pivot.to_string())

    try:
        (out_dir / "comparison_table.md").write_text(pivot.to_markdown())
    except ImportError:
        # pivot.to_markdown() needs the optional 'tabulate' package; fall
        # back to plain CSV text so this never blocks the run
        (out_dir / "comparison_table.md").write_text(pivot.to_string())

    plot_comparison_bar_chart(pivot, out_dir)
    plot_convergence_overlay(df, out_dir)

    print(f"\nSaved to {out_dir}/:")
    print("  comparison_results.csv, comparison_table.csv, comparison_table.md")
    print("  comparison_bar_chart.png/.pdf")
    print("  convergence_overlay_<dataset>.png/.pdf (one per dataset)")


if __name__ == "__main__":
    main()
