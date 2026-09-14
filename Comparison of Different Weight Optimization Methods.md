# Comparison of Different Weight Optimization Methods

In this project, I extended my original `ga_weight_opti_torch.py` program so that I could compare different weight optimization methods on different datasets. The original program was mainly designed for MNIST and used only the Genetic Algorithm. I kept that original implementation as it is and built a more general framework around it.

My main aim was to compare four methods:

- Adam
- Genetic Algorithm (GA)
- Differential Evolution (DE)
- MCTS

I also wanted to test these methods on three datasets:

- MNIST
- Fashion-MNIST
- CIFAR-10

Since all three datasets have 10 classes, I was able to use the same basic CNN architecture and change the input configuration according to the dataset.

## 1. Optimization Methods I Used

### Adam

I used Adam as the baseline method. Unlike the other methods, Adam does not maintain a population of models. It simply trains one model using gradient-based optimization.

I included Adam because it gives me a reference point for comparing the population-based methods. This helps me see whether the extra search performed by GA, DE, or MCTS actually gives better results than normal neural-network training.

### Genetic Algorithm

The Genetic Algorithm is based on the approach I used in my original project.

I maintain a population of neural-network models and calculate the fitness of each individual. The better-performing individuals have a higher chance of being selected.

I then use:

- fitness-proportionate selection
- single-point crossover
- random mutation
- elitism

After producing the new population, I perform local Adam training on the individuals.

This means that the GA is not replacing gradient-based learning completely. Instead, I use the genetic algorithm to change the weights between local training stages.

### Differential Evolution

For Differential Evolution, I directly operate on the neural-network weight tensors.

The basic mutation operation is:

```text
mutant = a + F × (b - c)
```

Here, `a`, `b`, and `c` are different individuals from the population.

After generating the mutant, I use binomial crossover to create a candidate solution. I then compare the candidate with the original individual and keep the better one.

I used the DE/rand/1/bin strategy for this implementation.

### MCTS

I also included an MCTS-based method, but there is an important difference from traditional MCTS.

Normal MCTS is generally used for problems where actions happen step by step and form a tree, such as chess or Go. Neural-network weights do not naturally have this type of structure.

Because of this, I used a simplified version of MCTS.

In my implementation, I treat different weight-perturbation scales as the available choices, or arms. UCB1 is then used to decide which perturbation scale should be explored.

The local training step acts somewhat like the rollout/evaluation stage.

So this should not be considered a full textbook implementation of MCTS. It is an adaptation of the idea of MCTS to the weight-optimization problem.

## 2. Making the Comparison Fair

I wanted the comparison between GA, DE, and MCTS to be as fair as possible.

For each generation, every individual in the population gets the same number of local Adam training epochs. This value is controlled by `epochs_per_gen`.

For example, if I set:

```text
epochs_per_gen = 2
```

then every individual receives two epochs of local Adam training during each generation, regardless of whether the optimizer is GA, DE, or MCTS.

The main difference between the population-based methods is therefore how they modify or recombine the weights between these training stages.

Adam is slightly different because it is intentionally a single-model baseline.

## 3. Datasets

I added three datasets to the project:

```text
MNIST
Fashion-MNIST
CIFAR-10
```

I registered them in `datasets.py`.

MNIST and Fashion-MNIST have images with one channel and a resolution of 28×28 pixels. CIFAR-10 has three channels and images of size 32×32.

To avoid having to manually calculate a different flatten size for every dataset, I used:

```python
nn.AdaptiveAvgPool2d((6, 6))
```

in my CNN architecture.

This allows the convolutional part of the model to produce a consistent feature size before the fully connected layers.

## 4. CNN Model

I created the common model in `models.py` and called it `WeightOptCNN`.

The model is designed to work with the different datasets rather than being tied only to MNIST.

The main difference between the datasets is the number of input channels. MNIST and Fashion-MNIST use one channel, while CIFAR-10 uses three.

The final classifier still has 10 output classes because all three datasets contain ten classes.

I also added support for passing the actual class names to the visualization code.

For example, instead of showing only class numbers, CIFAR-10 results can show labels such as:

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

Similarly, Fashion-MNIST can display labels such as `Sneaker`, `Shirt`, and `Trouser`.

## 5. Files I Added

The main structure of my project is:

```text
datasets.py
models.py

optimizers/
├── __init__.py
├── base.py
├── adam_baseline.py
├── genetic_algorithm.py
├── differential_evolution.py
└── mcts.py

run_experiment.py
compare_all.py

ga_visualization/
└── comparison_plots.py
```

I kept the existing visualization files and added the comparison plotting functionality separately.

The shared optimizer functionality is placed in `optimizers/base.py`.

This contains the common classes and functionality such as:

- `OptConfig`
- `Individual`
- `HistoryWriter`

The optimizer registry is handled through `optimizers/__init__.py`, so I can select an optimizer using its name rather than writing separate experiment code for every method.

## 6. Running One Experiment

I can run a single experiment by specifying the dataset and method.

For example, to run DE on CIFAR-10:

```bash
python run_experiment.py --dataset cifar10 --method de --generations 15 --population-size 8
```

For MCTS on Fashion-MNIST:

```bash
python run_experiment.py --dataset fashion_mnist --method mcts
```

For the Adam baseline on MNIST:

```bash
python run_experiment.py --dataset mnist --method adam --generations 15
```

The results are saved inside a directory based on the dataset and method.

For example:

```text
runs/cifar10_de/
├── best_model.pt
├── history.csv
└── report/
```

The trained best model is saved as `best_model.pt`.

The training progress is stored in `history.csv`, and the visualization results are stored inside the `report` directory.

## 7. Running the Complete Comparison

Instead of running every experiment manually, I created `compare_all.py`.

For testing the complete pipeline, I first use a small configuration:

```bash
python compare_all.py --generations 3 --population-size 4
```

Once that works correctly, I can use a larger configuration such as:

```bash
python compare_all.py --generations 20 --population-size 10
```

I can also compare only selected methods and datasets.

For example:

```bash
python compare_all.py --datasets mnist cifar10 --methods ga de
```

This is useful when I want to test only a particular part of the comparison instead of running all twelve combinations.

## 8. Comparison Results

After running the experiments, I get a set of files inside the `runs` directory.

The main comparison files are:

```text
comparison_results.csv
comparison_table.csv
comparison_table.md
comparison_bar_chart.png
comparison_bar_chart.pdf
```

I also get convergence plots for each dataset.

The comparison table has the following format:

| Dataset | Adam | GA | DE | MCTS |
|---|---:|---:|---:|---:|
| MNIST | Accuracy | Accuracy | Accuracy | Accuracy |
| Fashion-MNIST | Accuracy | Accuracy | Accuracy | Accuracy |
| CIFAR-10 | Accuracy | Accuracy | Accuracy | Accuracy |

The actual accuracy values are filled in after I run the experiments.

This is useful because I can directly compare the four methods on the same dataset.

## 9. Plots

I added `comparison_plots.py` to generate plots for the overall comparison.

One of the plots is a grouped bar chart. It shows the final accuracy of the different methods for each dataset.

I also generate convergence plots.

For example:

```text
convergence_overlay_mnist.png
convergence_overlay_fashion_mnist.png
convergence_overlay_cifar10.png
```

These plots allow me to see how the different methods improve over the generations.

I think the convergence plots are useful because looking only at the final accuracy does not show how quickly a method reached that result.

## 10. Existing Visualization Reports

I reused the visualization and reporting code from my existing project instead of creating a completely separate reporting system.

The existing report contains things such as:

- confusion matrix
- ROC and precision-recall curves
- Grad-CAM
- PCA/t-SNE
- weight histograms
- image galleries

I also modified the reporting interface so that I can optionally provide the class names.

This is especially useful for Fashion-MNIST and CIFAR-10 because the plots can show the actual class names instead of only numbers.

## 11. Computational Requirements

The complete experiment can take a significant amount of time.

There are four methods and three datasets, so running everything means:

```text
4 methods × 3 datasets = 12 runs
```

The population-based methods are more expensive because they train several models during every generation.

For example, with a population size of 8 and 15 generations, the approximate number of model-epochs per dataset is:

```text
(1 + 3 × 8) × 15 = 375
```

The three datasets together therefore require roughly:

```text
375 × 3 = 1125 model-epochs
```

for the population-based methods, in addition to the Adam runs.

CIFAR-10 will take considerably longer than MNIST, especially if I run everything on a CPU.

For the complete comparison, using a GPU is preferable.

## 12. Smoke Test

Before starting a long experiment, I use a small smoke test:

```bash
python compare_all.py --generations 2 --population-size 3
```

The purpose of this test is not to get meaningful accuracy results. I use it to make sure that the complete pipeline works.

The basic flow is:

```text
Dataset
   ↓
CNN Model
   ↓
Population
   ↓
Optimizer
   ↓
Local Adam Training
   ↓
Fitness Evaluation
   ↓
History
   ↓
Best Model
   ↓
Visualization
   ↓
Comparison Results
```

If this small test completes successfully, I can then increase the number of generations and population size.

## 13. Dependencies

The project requires the usual PyTorch and torchvision packages for training and datasets.

For the visualization and comparison functionality, I use:

```bash
pip install scikit-learn pandas imageio tensorboard tabulate
```

`tabulate` is used when converting the Pandas comparison table into Markdown.

If it is not available, the comparison script has a fallback so that the results can still be written without causing the whole experiment to fail.

## 14. Things That Can Affect the Results

The accuracy results will not necessarily be identical every time I run the experiments.

Some of the factors that can affect the results are:

- random initialization
- random mutation
- crossover
- data-loader ordering
- learning rate
- population size
- number of generations
- number of local training epochs
- PyTorch version
- torchvision version
- CPU/GPU hardware

Because of this, I consider the numbers generated on my machine to be experimental results rather than fixed benchmark values.

If I want a more reliable comparison later, I can run every method multiple times using different random seeds and report the mean and standard deviation.

## 15. Limitation of the MCTS Method

The main limitation of the MCTS implementation is that neural-network weight optimization does not naturally form a traditional search tree.

For this reason, my implementation uses a simplified one-level version where the available choices are different weight-perturbation scales.

UCB1 decides which scale to use based on the results obtained so far.

This makes the method useful for comparing the idea of MCTS-style exploration with the other optimizers, but it should not be described as a full traditional MCTS implementation.

A proper multi-level MCTS approach for neural-network weight optimization would require a different and much larger search design.

## 16. Conclusion

With this extension, I can now compare Adam, Genetic Algorithm, Differential Evolution, and the simplified MCTS approach using the same general CNN framework.

I can also run the comparison on MNIST, Fashion-MNIST, and CIFAR-10 without changing the main experiment code.

The most useful output for my comparison is the final accuracy table together with the convergence plots. The table shows which method performs better at the end, while the convergence plots show how the methods behave during training.

The results in the comparison files will come from my actual runs, so I can use them to make the final comparison instead of relying on fixed reference numbers.