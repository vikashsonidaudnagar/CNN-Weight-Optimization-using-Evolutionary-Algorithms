"""
run_experiment.py

Single (dataset, method) run. Trains, writes runs/<dataset>_<method>/, and
(unless --no-report) builds the full ga_visualization report against that
run's best model.

    python run_experiment.py --dataset cifar10 --method de --generations 15
    python run_experiment.py --dataset fashion_mnist --method mcts --population-size 8
    python run_experiment.py --dataset mnist --method adam --generations 15
"""

from __future__ import annotations

import argparse

import torch

from datasets import available_datasets, build_dataloaders, class_names as get_class_names
from optimizers import OptConfig, available_methods, run_method


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-run trainer: one dataset x one optimizer")
    p.add_argument("--dataset", required=True, choices=available_datasets())
    p.add_argument("--method", required=True, choices=available_methods())
    p.add_argument("--population-size", type=int, default=10)
    p.add_argument("--generations", type=int, default=20)
    p.add_argument("--epochs-per-gen", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--output-dir", type=str, default="./runs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None, choices=["cuda", "cpu", None])

    p.add_argument("--mutation-rate", type=float, default=0.1, help="GA only")
    p.add_argument("--de-F", type=float, default=0.5, help="DE only: differential weight")
    p.add_argument("--de-CR", type=float, default=0.7, help="DE only: crossover probability")
    p.add_argument("--mcts-arms", type=int, default=4, help="MCTS only: number of perturbation arms")
    p.add_argument("--mcts-c", type=float, default=1.4, help="MCTS only: UCB1 exploration constant")

    p.add_argument("--no-report", action="store_true",
                    help="skip the ga_visualization report after training")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OptConfig(
        dataset=args.dataset, method=args.method,
        population_size=1 if args.method == "adam" else args.population_size,
        generations=args.generations, epochs_per_gen=args.epochs_per_gen,
        batch_size=args.batch_size, num_workers=args.num_workers, lr=args.lr,
        data_dir=args.data_dir, output_dir=args.output_dir, seed=args.seed, device=device,
        mutation_rate=args.mutation_rate, de_F=args.de_F, de_CR=args.de_CR,
        mcts_arms=args.mcts_arms, mcts_c=args.mcts_c,
    )

    print(f"dataset={cfg.dataset} method={cfg.method} device={cfg.device}")
    result = run_method(cfg)
    print(f"Done. best_acc={result.best_acc:.4f} time={result.total_time_sec:.1f}s -> {result.output_dir}")

    if not args.no_report:
        from models import WeightOptCNN
        from ga_visualization import generate_full_report

        _, test_loader, meta = build_dataloaders(
            cfg.dataset, cfg.data_dir, cfg.batch_size, cfg.num_workers, cfg.device)

        print("generating visualization report...")
        generate_full_report(
            model=WeightOptCNN(meta),
            best_model_path=result.best_model_path,
            history_csv=result.history_csv,
            test_loader=test_loader,
            device=cfg.device,
            output_dir=result.output_dir / "report",
            class_names=get_class_names(cfg.dataset),
        )
        print(f"report written to {result.output_dir / 'report'}")


if __name__ == "__main__":
    main()
