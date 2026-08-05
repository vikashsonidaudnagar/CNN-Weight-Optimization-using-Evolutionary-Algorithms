CNN Weight Optimization using Evolutionary Algorithms

A PyTorch framework for training Convolutional Neural Networks (CNNs) using multiple optimization strategies and benchmarking their performance across several image classification datasets.

The framework provides a unified comparison between gradient-based optimization and population-based evolutionary algorithms, generating publication-quality tables and figures automatically.

Features
Multiple optimization algorithms
Adam
Genetic Algorithm (GA)
Differential Evolution (DE)
Monte Carlo Tree Search (MCTS)

Multiple datasets
MNIST
Fashion-MNIST
CIFAR-10

Automatic benchmarking
Performance comparison tables
Convergence analysis
Publication-quality visualizations

Repository Structure
.
├── data/
├── ga_visualization/
│   └── comparison_plots.py
├── optimizers/
│   ├── adam_baseline.py
│   ├── genetic_algorithm.py
│   ├── differential_evolution.py
│   ├── mcts.py
│   └── base.py
├── datasets.py
├── models.py
├── run_experiment.py
├── compare_all.py
├── README.md
└── requirements.txt

Optimization Methods
Method	Description
Adam	Standard gradient-based optimizer
Genetic Algorithm	Selection, crossover, mutation and elitism
Differential Evolution	DE/rand/1/bin with greedy replacement
Monte Carlo Tree Search	UCB-based weight perturbation search
Datasets
Dataset	Classes
MNIST	10
Fashion-MNIST	10
CIFAR-10	10
Installation
git clone https://github.com/yourusername/CNN-Optimizer-Comparison.git

cd CNN-Optimizer-Comparison

pip install -r requirements.txt
Running a Single Experiment
python run_experiment.py \
    --dataset mnist \
    --method ga

Example:

python run_experiment.py \
    --dataset cifar10 \
    --method de \
    --generations 20 \
    --population-size 10
Benchmark All Optimizers

Run all optimizers across all datasets

python compare_all.py

Quick smoke test

python compare_all.py \
    --generations 5 \
    --population-size 4

Compare selected methods

python compare_all.py \
    --datasets mnist cifar10 \
    --methods ga de
Output

After execution the framework automatically generates

runs/

comparison_results.csv

comparison_table.csv

comparison_table.md

comparison_bar_chart.png

comparison_bar_chart.pdf

convergence_overlay_*.png

best_model.pt

history.csv
Workflow
Dataset
   │
   ▼
Load CNN
   │
   ▼
Select Optimizer
   │
   ▼
Training
   │
   ▼
Evaluation
   │
   ▼
Best Accuracy
   │
   ▼
Comparison Table
   │
   ▼
Bar Charts & Convergence Plots
Example Comparison
Dataset	Adam	GA	DE	MCTS
MNIST	99.3	99.1	99.0	99.2
Fashion-MNIST	92.5	91.9	91.6	92.2
CIFAR-10	84.7	83.9	83.2	84.1

(Replace these values with your experimental results.)

Dependencies
Python 3.11+
PyTorch
Torchvision
NumPy
Pandas
Matplotlib
Scikit-learn
TensorBoard (optional)
ImageIO (optional)

Install:

pip install torch torchvision pandas matplotlib scikit-learn tensorboard imageio tabulateCNN Weight Optimization using Evolutionary Algorithms


Scikit-learn
TensorBoard (optional)
ImageIO (optional)

Install:

pip install torch torchvision pandas matplotlib scikit-learn tensorboard imageio tabulate
