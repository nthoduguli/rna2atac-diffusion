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
    ax.set_xticklabels(model_names, rotation=30, ha="right")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# Reference-palette tokens (light surface): one hue for models, muted gray for baselines/controls.
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_BASELINE = "#c3c2b7"
_MODEL_COLOR = "#2a78d6"
_CONTROL_COLOR = "#b5b3ab"


def plot_results_summary(
    results: dict[str, dict[str, float]],
    rows: list[tuple[str, str, bool]],
    panels: list[tuple[str, str]],
    title: str,
    output_path: str | Path,
) -> None:
    """Small multiples of horizontal bars: one panel per metric, one row per model.

    Args:
        results: model name -> metric name -> value.
        rows: (model name, display label, is_control) in top-to-bottom order; controls and
            baselines are drawn in muted gray so the trained models stand out.
        panels: (metric name, panel title) in left-to-right order. All panels share one
            0-1 axis so bar lengths compare across panels.
        title: Figure title.
        output_path: Where to save the PNG.
    """
    plt.rcParams.update({"font.size": 9, "text.color": _TEXT_PRIMARY})
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.0 * len(panels) + 1.6, 0.42 * len(rows) + 1.2), sharey=True, facecolor=_SURFACE
    )
    y = np.arange(len(rows))[::-1]
    for i, (ax, (metric, panel_title)) in enumerate(zip(np.atleast_1d(axes), panels, strict=True)):
        ax.set_facecolor(_SURFACE)
        values = [results[name][metric] for name, _, _ in rows]
        colors = [_CONTROL_COLOR if is_control else _MODEL_COLOR for _, _, is_control in rows]
        ax.barh(y, values, height=0.55, color=colors, edgecolor=_SURFACE, linewidth=2)
        for yi, value in zip(y, values, strict=True):
            ax.text(
                max(value, 0) + 0.02,
                yi,
                f"{0.0 if abs(value) < 0.005 else value:.2f}",
                va="center",
                ha="left",
                color=_TEXT_SECONDARY,
            )
        ax.set_xlim(0, 1.12)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.xaxis.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(panel_title, loc="left", color=_TEXT_PRIMARY, fontsize=9.5)
        ax.tick_params(colors=_MUTED, length=0)
        for side in ("top", "right", "bottom"):
            ax.spines[side].set_visible(False)
        ax.spines["left"].set_color(_BASELINE)
        if i == 0:
            ax.set_yticks(y)
            ax.set_yticklabels([label for _, label, _ in rows], color=_TEXT_PRIMARY)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=_TEXT_PRIMARY)
    fig.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, facecolor=_SURFACE)
    plt.close(fig)
