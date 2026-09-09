"""Plotting utilities for training curves and evaluation diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


def plot_loss_curve(losses: list[float], title: str, output_path: str | Path) -> None:
    """Plot a training loss curve and save it to `output_path`."""
    epochs = range(1, len(losses) + 1)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, losses, marker="o" if len(losses) <= 20 else None)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    if len(losses) <= 20:
        # MaxNLocator(integer=True) doesn't reliably produce integer ticks when the
        # autoscaled view range is under 1 unit wide (e.g. a single-epoch run), so
        # pin ticks explicitly for short curves rather than relying on autoscaling.
        ax.set_xticks(list(epochs))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_pred_vs_true(pred: np.ndarray, true: np.ndarray, title: str, output_path: str | Path) -> None:
    """Scatter plot of predicted vs. true values (flattened across all dims/cells)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(true.ravel(), pred.ravel(), s=4, alpha=0.3)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metric_comparison(results: dict[str, dict[str, float]], title: str, output_path: str | Path) -> None:
    """Grouped bar chart comparing metrics across models."""
    model_names = list(results)
    metric_names = list(next(iter(results.values())))

    x = np.arange(len(model_names))
    width = 0.8 / len(metric_names)

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, metric in enumerate(metric_names):
        values = [results[model][metric] for model in model_names]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
    ax.set_xticklabels(model_names)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
