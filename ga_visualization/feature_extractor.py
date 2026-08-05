"""
ga_visualization/feature_extractor.py

Everything that needs a forward (or forward+backward) hook into the model:
  - grabbing intermediate conv activations for feature-map plots
  - grabbing the fc1 embedding (the 100-d representation right before the
    final classification layer) for PCA / t-SNE
  - Grad-CAM, which needs both the activations of the last conv layer and
    the gradient of the class score w.r.t. those activations

The model architecture (MnistCNN in ga_weight_opti_torch.py) is fixed and
small, so a couple of named forward hooks are enough; nothing here assumes
anything beyond "the model has attributes conv1, conv2, fc1, fc2".
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class ActivationExtractor:
    """Registers forward hooks on conv1 / conv2 / fc1 and caches whatever
    passed through them on the most recent forward pass. Use as:

        extractor = ActivationExtractor(model)
        with torch.no_grad():
            model(batch)
        acts = extractor.activations   # dict of tensors
        extractor.remove()
    """

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles = []
        self._handles.append(model.conv1.register_forward_hook(self._save("conv1")))
        self._handles.append(model.conv2.register_forward_hook(self._save("conv2")))
        self._handles.append(model.fc1.register_forward_hook(self._save("fc1")))

    def _save(self, key: str):
        def hook(_module, _inp, output):
            self.activations[key] = output.detach()
        return hook

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
    max_samples: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run `model` over `loader` and collect, for up to `max_samples` test
    images: the fc1 (100-d) embedding, the true label, and the predicted
    label. Embeddings feed PCA / t-SNE; labels drive coloring and the
    misclassified/correct-prediction galleries.

    Returns
    -------
    embeddings : (N, 100) float32 array
    y_true     : (N,) int array
    y_pred     : (N,) int array
    """
    model.eval()
    extractor = ActivationExtractor(model)

    all_emb: List[np.ndarray] = []
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    seen = 0

    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        preds = logits.argmax(dim=1).cpu().numpy()
        emb = extractor.activations["fc1"].cpu().numpy()

        all_emb.append(emb)
        all_true.append(yb.numpy())
        all_pred.append(preds)
        seen += xb.size(0)
        if seen >= max_samples:
            break

    extractor.remove()
    embeddings = np.concatenate(all_emb, axis=0)[:max_samples]
    y_true = np.concatenate(all_true, axis=0)[:max_samples]
    y_pred = np.concatenate(all_pred, axis=0)[:max_samples]
    return embeddings, y_true, y_pred


@torch.no_grad()
def full_test_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the *entire* test set through `model` once. Needed for confusion
    matrix / classification report / ROC / PR curves, which should reflect
    the full test set rather than a subsample.

    Returns
    -------
    images  : (N, 1, 28, 28) float32 array (kept for the prediction galleries)
    y_true  : (N,) int array
    y_pred  : (N,) int array
    y_score : (N, 10) float32 array of softmax probabilities
    """
    model.eval()
    images_l, true_l, pred_l, score_l = [], [], [], []

    for xb, yb in loader:
        xb_dev = xb.to(device)
        logits = model(xb_dev)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        images_l.append(xb.numpy())
        true_l.append(yb.numpy())
        pred_l.append(preds.cpu().numpy())
        score_l.append(probs.cpu().numpy())

    images = np.concatenate(images_l, axis=0)
    y_true = np.concatenate(true_l, axis=0)
    y_pred = np.concatenate(pred_l, axis=0)
    y_score = np.concatenate(score_l, axis=0)
    return images, y_true, y_pred, y_score


class GradCAM:
    """Minimal Grad-CAM implementation targeting `conv2`, the last
    convolutional layer in MnistCNN. Grad-CAM highlights which spatial
    regions of the input most influenced the model's chosen class, giving
    a rough "feature importance" heatmap without needing any extra
    architecture (works on any CNN with a final conv layer before pooling).
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._fwd_handle = target_layer.register_forward_hook(self._forward_hook)
        self._bwd_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, _module, _inp, output):
        self.activations = output

    def _backward_hook(self, _module, _grad_in, grad_out):
        self.gradients = grad_out[0]

    def remove(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        """x: a single-image batch, shape (1, 1, 28, 28). Returns a
        (28, 28) heatmap normalized to [0, 1], upsampled to input resolution.
        """
        self.model.eval()
        logits = self.model(x)
        score = logits[0, class_idx]
        self.model.zero_grad(set_to_none=True)
        score.backward()

        # global-average-pool the gradients over spatial dims -> channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # (1, 1, h, w)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        return cam
