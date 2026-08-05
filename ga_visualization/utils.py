"""
ga_visualization/utils.py

Small shared helpers used by every other module in the package:
  - consistent, publication-quality figure saving (PNG @ 300dpi + PDF @ 600dpi)
  - directory bootstrapping
  - a couple of numeric helpers (entropy, safe-softmax) used by more than
    one plotting module so they aren't duplicated.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Union

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("ga_viz")

PathLike = Union[str, Path]


def ensure_dir(path: PathLike) -> Path:
    """Create `path` (and parents) if missing, return it as a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_fig(fig, name: str, output_dir: PathLike, png_dpi: int = 300, pdf_dpi: int = 600) -> None:
    """Save a matplotlib figure as both PNG (for quick viewing) and a
    high-resolution vector-friendly PDF (for publication), then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    name : str
        File stem, no extension (e.g. "confusion_matrix").
    output_dir : str | Path
        Directory the files are written into; created if it does not exist.
    """
    import matplotlib.pyplot as plt  # local import keeps module import light

    out_dir = ensure_dir(output_dir)
    png_path = out_dir / f"{name}.png"
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(png_path, dpi=png_dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=pdf_dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved {png_path.name} + {pdf_path.name}")


def shannon_entropy(probabilities: Iterable[float]) -> float:
    """Shannon entropy (base-2, in bits) of a discrete probability vector.
    Used to track population diversity across GA generations: a population
    whose fitness is concentrated on one individual has low entropy, a
    population whose fitness is spread evenly has high entropy.
    """
    p = np.asarray(list(probabilities), dtype=np.float64)
    p = p[p > 0]  # 0 * log(0) := 0, so just drop zero-mass entries
    if p.sum() <= 0:
        return 0.0
    p = p / p.sum()
    return float(-np.sum(p * np.log2(p)))


def to_probabilities(values: Iterable[float]) -> np.ndarray:
    """Normalize a non-negative vector of values (e.g. accuracies) into a
    probability distribution, guarding against an all-zero vector."""
    v = np.asarray(list(values), dtype=np.float64)
    v = np.clip(v, a_min=0.0, a_max=None)
    total = v.sum()
    if total <= 0:
        return np.full_like(v, 1.0 / len(v))
    return v / total
