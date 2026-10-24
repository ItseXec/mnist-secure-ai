"""Shared helpers for attacks and defenses: ART wrapper, charts, image grids."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import ScaledTranslation  # noqa: E402
from art.estimators.classification import PyTorchClassifier  # noqa: E402
from torch import nn  # noqa: E402

from src.utils import ROOT  # noqa: E402

DATASETS_DIR = os.path.join(ROOT, "datasets")

# Categorical slots 1-4 of the validated reference palette (dataviz skill).
# Identity is never colour-alone: every series also gets a marker and a label.
SERIES = [("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D")]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def art_classifier(model):
    """Wrap a trained PyTorch model so ART attacks can query it (white-box)."""
    model.eval()
    return PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                             input_shape=(1, 28, 28), nb_classes=10,
                             clip_values=(0.0, 1.0), device_type="cpu")


def style_axes(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def pct_label(value):
    """Percentage text that never rounds 99.99 up to 100.0 or 0.01 down to 0.0."""
    if 99.95 <= value < 100 or 0 < value < 0.05:
        return f"{value:.2f}%"
    return f"{value:.1f}%"


def label_line_ends(ax, ends, min_gap=5.0):
    """Write each line's final value at its right end, nudging labels apart so that
    lines ending at the same value (e.g. two models both at 0%) stay readable.
    ends: list of (x, y, text) in data coordinates."""
    shift = ax.transData + ScaledTranslation(8 / 72, 0, ax.figure.dpi_scale_trans)
    floor = ax.get_ylim()[0] + 0.6 * min_gap   # keep the lowest label off the x axis
    previous = None
    for x, y, text in sorted(ends, key=lambda e: e[1]):
        y_text = max(y, floor) if previous is None else max(y, previous + min_gap)
        previous = y_text
        ax.text(x, y_text, text, transform=shift, va="center", fontsize=9, color=TEXT,
                clip_on=False)


def image_grid(rows, path, title):
    """rows: list of (row_label, images (N,1,28,28), captions list)."""
    n_cols = len(rows[0][1])
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(1.3 * n_cols + 1.4, 1.5 * len(rows) + 0.6))
    for r, (label, images, captions) in enumerate(rows):
        for c in range(n_cols):
            ax = axes[r, c]
            ax.imshow(images[c, 0], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(captions[c], fontsize=8, color=TEXT)
        axes[r, 0].set_ylabel(label, fontsize=9, color=TEXT, rotation=0, ha="right", va="center")
    fig.suptitle(title, color=TEXT)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
