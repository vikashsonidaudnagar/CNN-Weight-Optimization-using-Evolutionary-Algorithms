"""
ga_visualization

A self-contained reporting package for ga_weight_opti_torch.py. After a GA
run finishes (best_model.pt + history.csv on disk), call:

    from ga_visualization import generate_full_report
    generate_full_report(
        model=my_mnist_cnn_instance,          # architecture only; weights loaded from disk
        best_model_path="ga_run/best_model.pt",
        history_csv="ga_run/history.csv",
        test_loader=test_loader,
        device="cuda",
        output_dir="ga_run/report",
    )

and every figure listed in the module docstrings of visualization.py and
evolution_plots.py is generated and written to `output_dir` as matched
PNG (300dpi) + PDF (600dpi) pairs, plus a couple of plain-text/GIF
artifacts (classification_report.txt, evolution.gif).

Nothing in this package requires network access or GPU; everything runs
on CPU (slower for t-SNE/Grad-CAM on the full test set, which is why
embeddings are subsampled -- see feature_extractor.extract_embeddings).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch

from .utils import ensure_dir, logger
from .feature_extractor import extract_embeddings, full_test_predictions
from .visualization import (
    plot_feature_maps,
    plot_confusion_matrix,
    save_classification_report,
    plot_roc_curves,
    plot_pr_curves,
    plot_embedding_2d,
    plot_weight_histograms,
    plot_gradcam_gallery,
    plot_misclassified_gallery,
    plot_correct_gallery,
)
from .evolution_plots import (
    plot_accuracy_convergence,
    plot_population_diversity,
    plot_fitness_histograms,
    make_evolution_gif,
)
from .tb_logger import GATensorBoardLogger

__all__ = ["generate_full_report", "GATensorBoardLogger"]


def generate_full_report(
    model: torch.nn.Module,
    best_model_path,
    history_csv,
    test_loader,
    device: str,
    output_dir,
    embedding_subsample: int = 2000,
    gradcam_examples: int = 8,
    gallery_size: int = 25,
    make_gif: bool = True,
) -> None:
    """Run every visualization in the package against one finished GA run.

    `model` must be an *untrained/architecture-only* instance of the same
    class used during the GA run (e.g. MnistCNN()); its weights are
    overwritten in-place from `best_model_path`.
    """
    out_dir = ensure_dir(output_dir)
    logger.info(f"writing full visualization report to {out_dir}")

    # -- load the best individual found by the GA -------------------------
    state_dict = torch.load(best_model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # -- evolution-only plots (no model needed, but grouped here for one-call use) --
    plot_accuracy_convergence(history_csv, out_dir)
    plot_population_diversity(history_csv, out_dir)
    plot_fitness_histograms(history_csv, out_dir)
    if make_gif:
        make_evolution_gif(history_csv, out_dir)

    # -- full-test-set inference, used by several plots below -------------
    images, y_true, y_pred, y_score = full_test_predictions(model, test_loader, device)

    plot_confusion_matrix(y_true, y_pred, out_dir)
    save_classification_report(y_true, y_pred, out_dir)
    plot_roc_curves(y_true, y_score, out_dir)
    plot_pr_curves(y_true, y_score, out_dir)
    plot_misclassified_gallery(images, y_true, y_pred, out_dir, n=gallery_size)
    plot_correct_gallery(images, y_true, y_pred, out_dir, n=gallery_size)

    # -- weight distributions ---------------------------------------------
    plot_weight_histograms(model, out_dir)

    # -- feature maps for one representative sample ------------------------
    sample_image = torch.from_numpy(images[0])  # (1, 28, 28)
    plot_feature_maps(model, sample_image, device, out_dir)

    # -- Grad-CAM gallery on a handful of test images -----------------------
    plot_gradcam_gallery(model, images[:gradcam_examples], y_true[:gradcam_examples],
                          device, out_dir, n=gradcam_examples)

    # -- PCA / t-SNE on fc1 embeddings (subsampled for speed) ---------------
    embeddings, emb_true, _ = extract_embeddings(model, test_loader, device,
                                                  max_samples=embedding_subsample)
    plot_embedding_2d(embeddings, emb_true, out_dir, method="pca")
    plot_embedding_2d(embeddings, emb_true, out_dir, method="tsne")

    logger.info(f"report complete: {out_dir}")
