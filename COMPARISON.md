# Multi-optimizer, multi-dataset comparison suite

Extends the original `ga_weight_opti_torch.py` (kept as-is, MNIST-only, GA-only)
with a general framework that runs **four weight-optimization strategies**
against **three datasets** and produces a comparison table + plots like the
one you're benchmarking against.

## The four methods

| Method | File | What it does |
|---|---|---|
| `adam`  | `optimizers/adam_baseline.py` | Plain gradient descent, single model, no population. The reference point. |
| `ga`    | `optimizers/genetic_algorithm.py` | Fitness-proportionate selection + single-point crossover + random mutation, with elitism. Same algorithm as your original script. |
| `de`    | `optimizers/differential_evolution.py` | DE/rand/1/bin applied directly to weight tensors: mutant = a + F·(b − c), binomial crossover, greedy replacement. |
| `mcts`  | `optimizers/mcts.py` | **Simplified/adapted** MCTS: UCB1 bandit selection over a ladder of weight-perturbation scales (one-level tree). See the caveat below. |

**Fairness of the comparison**: every method (except `adam`, which is
single-model by design) gets a population, and every individual in that
population gets exactly `epochs_per_gen` epochs of local Adam training per
generation, regardless of method. The methods differ *only* in how they
recombine/perturb weights between those local-training steps — so a
higher final accuracy for DE or MCTS reflects the recombination strategy,
not a hidden compute advantage.

**On the MCTS caveat**: weight optimization doesn't have the discrete,
sequential-move structure classic MCTS was built for (chess, Go, etc. have
a clear tree of legal moves; a network's weight space doesn't). What's
implemented is a defensible but simplified adaptation — a one-level
"tree" where the arms are perturbation scales, selected via UCB1, with the
local-training epoch playing the role of rollout. This is documented in
detail at the top of `optimizers/mcts.py`. If your comparison needs
textbook MCTS with proper multi-level expansion, that's a substantially
larger undertaking — happy to scope it separately if this simplified
version doesn't fit your needs.

## The three datasets

`mnist`, `fashion_mnist`, `cifar10` — registered in `datasets.py`. All
three have 10 classes, so the same `WeightOptCNN` head works everywhere;
input channels/resolution differ (1×28×28 vs 3×32×32) and `models.py`
handles that via `nn.AdaptiveAvgPool2d((6, 6))` after the conv stack, so
the flatten size never needs to be hand-recomputed per dataset.

## File layout (new files)

```
datasets.py                     # dataset registry: mnist / fashion_mnist / cifar10
models.py                       # WeightOptCNN, dataset-agnostic architecture
optimizers/
├── __init__.py                 # REGISTRY + run_method() dispatcher
├── base.py                     # OptConfig, Individual, HistoryWriter (shared by all 4 methods)
├── adam_baseline.py            # "adam"
├── genetic_algorithm.py        # "ga"
├── differential_evolution.py   # "de"
└── mcts.py                     # "mcts" (see caveat above)
run_experiment.py                # single (dataset, method) run
compare_all.py                   # every method x every dataset + comparison report
ga_visualization/
└── comparison_plots.py          # grouped bar chart + convergence overlay (new)
```

The existing `ga_visualization/` report suite (confusion matrix, ROC/PR,
Grad-CAM, PCA/t-SNE, weight histograms, galleries, etc.) is reused
unmodified — it now also accepts an optional `class_names` list so
Fashion-MNIST/CIFAR-10 plots show real labels ("Sneaker", "airplane")
instead of digits.

## Usage

### One dataset, one method

```bash
python run_experiment.py --dataset cifar10 --method de --generations 15 --population-size 8
python run_experiment.py --dataset fashion_mnist --method mcts
python run_experiment.py --dataset mnist --method adam --generations 15
```

Writes to `runs/<dataset>_<method>/`:
```
runs/cifar10_de/
├── best_model.pt
├── history.csv                 # generation, member_0..N, best_so_far, elapsed_sec
└── report/                     # full ga_visualization report (unless --no-report)
```

### Everything at once — the actual comparison

```bash
# smoke test first (fast, confirms the pipeline runs end to end)
python compare_all.py --generations 3 --population-size 4

# real run
python compare_all.py --generations 20 --population-size 10

# subset, e.g. only GA vs DE on two datasets
python compare_all.py --datasets mnist cifar10 --methods ga de
```

Writes to `runs/`:
```
runs/
├── comparison_results.csv        # one row per (dataset, method) run
├── comparison_table.csv          # pivoted: rows=dataset, cols=method, values=accuracy %
├── comparison_table.md           # same, as a Markdown table
├── comparison_bar_chart.png/.pdf # grouped bars, like your reference table as a figure
├── convergence_overlay_mnist.png/.pdf
├── convergence_overlay_fashion_mnist.png/.pdf
├── convergence_overlay_cifar10.png/.pdf
└── <dataset>_<method>/           # each individual run's own outputs (as above)
```

`comparison_table.csv` is exactly the shape of the table you posted —
open it after a run and you'll have your own Adam/GA/DE/MCTS × dataset
grid, filled in with real numbers from your machine instead of the
reference values.

## Compute budget — read this before a long run

`compare_all.py` runs **4 methods × 3 datasets = 12 full training runs**
by default. Each one trains `population_size` models (1 for Adam) for
`generations × epochs_per_gen` epochs. With the defaults
(`population_size=8`, `generations=15`), that's roughly
`(1 + 3×8) × 15 = 375` model-epochs per dataset, ×3 datasets ≈ 1,125
model-epochs total for GA/DE/MCTS combined, plus Adam's 45. On CIFAR-10
in particular, expect this to take a while on CPU — a GPU is strongly
recommended for the full comparison. Start with a small smoke test
(`--generations 3 --population-size 4`) to confirm everything runs before
committing to a long unattended job.

## New dependencies

Same as the visualization suite: `scikit-learn`, `pandas`, and optionally
`imageio` + `tensorboard`. `pivot.to_markdown()` in `compare_all.py` also
wants the optional `tabulate` package — if it's missing, the script falls
back to writing plain text instead of crashing.

```bash
pip install scikit-learn pandas imageio tensorboard tabulate
```

## Verification note

I don't have `torch`/CUDA available in this sandbox, so I could not run an
actual end-to-end training pass here. Every file was checked with
`python -m py_compile` (catches syntax errors, bad indentation, etc.), and
I traced the data flow by hand (dataset → Individual → optimizer →
HistoryWriter → ga_visualization). Please run the smoke test
(`python compare_all.py --generations 2 --population-size 3`) on your
machine before a long unattended job, in case a torch/torchvision-version
quirk surfaces that a static check can't catch.
