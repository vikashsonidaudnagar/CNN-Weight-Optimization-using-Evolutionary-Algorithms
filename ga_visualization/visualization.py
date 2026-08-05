"""
ga_visualization/visualization.py

Model-evaluation visualizations: everything that needs a trained model
and/or a batch of test predictions, as opposed to evolution_plots.py which
only needs the GA's per-generation fitness history.

Every plotting function here takes plain numpy arrays / a model + a
DataLoader, and writes PNG+PDF via utils.save_fig -- nothing returns a
figure, so these can be called as a simple "generate my report" pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from .utils import ensure_dir, save_fig, logger
from .feature_extractor import ActivationExtractor, GradCAM

CLASS_NAMES = [str(i) for i in range(10)]


# ---------------------------------------------------------------------------
# 1. Feature maps (conv1 / conv2 activations for one sample image)
# ---------------------------------------------------------------------------
def plot_feature_maps(model, sample_image: torch.Tensor, device: str, output_dir) -> None:
    """sample_image: a single (1, 28, 28) tensor (no batch dim)."""
    model.eval()
    extractor = ActivationExtractor(model)
    x = sample_image.unsqueeze(0).to(device)
    with torch.no_grad():
        model(x)

    for layer_name in ("conv1", "conv2"):
        acts = extractor.activations[layer_name][0].cpu().numpy()  # (C, H, W)
        n_channels = acts.shape[0]
        n_cols = 8
        n_rows = int(np.ceil(n_channels / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.3, n_rows * 1.3))
        axes = np.atleast_2d(axes)
        for i in range(n_rows * n_cols):
            ax = axes[i // n_cols, i % n_cols]
            ax.axis("off")
            if i < n_channels:
                ax.imshow(acts[i], cmap="viridis")
                ax.set_title(f"ch{i}", fontsize=6)
        fig.suptitle(f"{layer_name} activations")
        save_fig(fig, f"feature_maps_{layer_name}", output_dir)

    extractor.remove()


# ---------------------------------------------------------------------------
# 2. Confusion matrix (raw + normalized)
# ---------------------------------------------------------------------------
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir) -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)

    for cm_data, normalize, name in [(cm, False, "confusion_matrix_counts"),
                                      (cm_norm, True, "confusion_matrix_normalized")]:
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm_data, cmap="Blues")
        ax.set_xticks(range(10)); ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks(range(10)); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
        ax.set_title("Confusion matrix" + (" (normalized)" if normalize else " (counts)"))
        fmt = "{:.2f}" if normalize else "{:d}"
        thresh = cm_data.max() / 2.0
        for i in range(cm_data.shape[0]):
            for j in range(cm_data.shape[1]):
                val = cm_data[i, j]
                text = fmt.format(val) if normalize else fmt.format(int(val))
                ax.text(j, i, text, ha="center", va="center", fontsize=7,
                        color="white" if val > thresh else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        save_fig(fig, name, output_dir)


# ---------------------------------------------------------------------------
# 3. Classification report (precision / recall / F1 per class)
# ---------------------------------------------------------------------------
def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray, output_dir) -> None:
    out_dir = ensure_dir(output_dir)
    report_str = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    (out_dir / "classification_report.txt").write_text(report_str)
    logger.info("saved classification_report.txt")

    report_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES,
                                         digits=4, output_dict=True)
    metrics = ["precision", "recall", "f1-score"]
    data = np.array([[report_dict[c][m] for m in metrics] for c in CLASS_NAMES])

    fig, ax = plt.subplots(figsize=(5, 7))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(metrics))); ax.set_xticklabels(metrics)
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
    ax.set_ylabel("class")
    ax.set_title("Per-class precision / recall / F1")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if data[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    save_fig(fig, "classification_report_heatmap", output_dir)


# ---------------------------------------------------------------------------
# 4 & 5. ROC and Precision-Recall curves, one-vs-rest over all 10 classes
# ---------------------------------------------------------------------------
def _one_hot(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((len(y_true), num_classes), dtype=np.int32)
    out[np.arange(len(y_true)), y_true] = 1
    return out


def plot_roc_curves(y_true: np.ndarray, y_score: np.ndarray, output_dir, num_classes: int = 10) -> None:
    y_bin = _one_hot(y_true, num_classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(num_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, c], y_score[:, c])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.2, label=f"class {c} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="chance")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves (one-vs-rest)")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    save_fig(fig, "roc_curves", output_dir)


def plot_pr_curves(y_true: np.ndarray, y_score: np.ndarray, output_dir, num_classes: int = 10) -> None:
    y_bin = _one_hot(y_true, num_classes)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in range(num_classes):
        precision, recall, _ = precision_recall_curve(y_bin[:, c], y_score[:, c])
        ap = average_precision_score(y_bin[:, c], y_score[:, c])
        ax.plot(recall, precision, lw=1.2, label=f"class {c} (AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (one-vs-rest)")
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    save_fig(fig, "precision_recall_curves", output_dir)


# ---------------------------------------------------------------------------
# 6. PCA / t-SNE embedding plots
# ---------------------------------------------------------------------------
def plot_embedding_2d(embeddings: np.ndarray, labels: np.ndarray, output_dir, method: str = "pca") -> None:
    method = method.lower()
    if method == "pca":
        reducer = PCA(n_components=2, random_state=42)
        coords = reducer.fit_transform(embeddings)
        title = "PCA of fc1 embeddings"
        fname = "embedding_pca"
    elif method == "tsne":
        n = embeddings.shape[0]
        perplexity = min(30, max(5, n // 20))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, init="pca")
        coords = reducer.fit_transform(embeddings)
        title = "t-SNE of fc1 embeddings"
        fname = "embedding_tsne"
    else:
        raise ValueError(f"unknown method {method!r}")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=8, alpha=0.8)
    legend = ax.legend(*scatter.legend_elements(), title="digit", fontsize=7,
                        loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.add_artist(legend)
    ax.set_title(title)
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    save_fig(fig, fname, output_dir)


# ---------------------------------------------------------------------------
# 7. Weight-distribution histograms per layer
# ---------------------------------------------------------------------------
def plot_weight_histograms(model, output_dir) -> None:
    named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    n = len(named_params)
    n_cols = 3
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)
    for idx, (name, param) in enumerate(named_params):
        ax = axes[idx // n_cols, idx % n_cols]
        data = param.detach().cpu().numpy().ravel()
        ax.hist(data, bins=50, color="steelblue")
        ax.set_title(name, fontsize=8)
        ax.tick_params(labelsize=6)
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")
    fig.suptitle("Weight distributions by layer")
    fig.tight_layout()
    save_fig(fig, "weight_histograms", output_dir)


# ---------------------------------------------------------------------------
# 8. Grad-CAM gallery
# ---------------------------------------------------------------------------
def plot_gradcam_gallery(model, images: np.ndarray, labels: np.ndarray, device: str,
                          output_dir, n: int = 8) -> None:
    """images: (N, 1, 28, 28) float32 numpy array, already in [0, 1]."""
    n = min(n, len(images))
    cam_engine = GradCAM(model, model.conv2)

    fig, axes = plt.subplots(2, n, figsize=(n * 1.6, 3.4))
    for i in range(n):
        x = torch.from_numpy(images[i:i + 1]).to(device)
        label = int(labels[i])
        heatmap = cam_engine(x, class_idx=label)

        img = images[i, 0]
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"digit {label}", fontsize=8)

        axes[1, i].imshow(img, cmap="gray")
        axes[1, i].imshow(heatmap, cmap="jet", alpha=0.5)
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("input", fontsize=8)
    fig.suptitle("Grad-CAM: input (top) vs. class-activation overlay (bottom)")
    cam_engine.remove()
    save_fig(fig, "gradcam_gallery", output_dir)


# ---------------------------------------------------------------------------
# 9. Misclassified / correctly-classified prediction galleries
# ---------------------------------------------------------------------------
def _prediction_gallery(images: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
                         mask: np.ndarray, output_dir, name: str, n: int = 25) -> None:
    idxs = np.flatnonzero(mask)
    if len(idxs) == 0:
        logger.info(f"no samples for gallery '{name}', skipping")
        return
    idxs = idxs[:n]
    n_cols = 5
    n_rows = int(np.ceil(len(idxs) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.6, n_rows * 1.8))
    axes = np.atleast_2d(axes)
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        ax.axis("off")
        if i < len(idxs):
            idx = idxs[i]
            ax.imshow(images[idx, 0], cmap="gray")
            ax.set_title(f"T:{y_true[idx]} P:{y_pred[idx]}", fontsize=8,
                         color="green" if y_true[idx] == y_pred[idx] else "red")
    fig.suptitle(name.replace("_", " ").title())
    save_fig(fig, name, output_dir)


def plot_misclassified_gallery(images, y_true, y_pred, output_dir, n: int = 25) -> None:
    _prediction_gallery(images, y_true, y_pred, y_true != y_pred, output_dir,
                         "misclassified_gallery", n)


def plot_correct_gallery(images, y_true, y_pred, output_dir, n: int = 25) -> None:
    _prediction_gallery(images, y_true, y_pred, y_true == y_pred, output_dir,
                         "correct_predictions_gallery", n)
