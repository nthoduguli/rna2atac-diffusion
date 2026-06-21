"""Evaluation metrics for predicted vs. true ATAC representations/profiles."""

from __future__ import annotations

import numpy as np


def pearson_r(pred: np.ndarray, true: np.ndarray) -> float:
    """Mean per-sample Pearson correlation between predicted and true vectors."""
    pred = pred - pred.mean(axis=1, keepdims=True)
    true = true - true.mean(axis=1, keepdims=True)
    num = (pred * true).sum(axis=1)
    denom = np.sqrt((pred**2).sum(axis=1) * (true**2).sum(axis=1))
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(num / denom))


def r_squared(pred: np.ndarray, true: np.ndarray) -> float:
    """Coefficient of determination (R^2), aggregated over all elements."""
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - true.mean(axis=0, keepdims=True)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def weighted_mse(pred: np.ndarray, true: np.ndarray, weights: np.ndarray | None = None) -> float:
    """MSE weighted per-feature, e.g. by peak accessibility frequency."""
    se = (pred - true) ** 2
    if weights is None:
        return float(se.mean())
    weights = weights / weights.sum()
    return float((se * weights[None, :]).sum(axis=1).mean())


def evaluate_all(pred: np.ndarray, true: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    """Compute the full evaluation suite (Pearson r, R^2, weighted MSE)."""
    return {
        "pearson_r": pearson_r(pred, true),
        "r2": r_squared(pred, true),
        "weighted_mse": weighted_mse(pred, true, weights),
    }
