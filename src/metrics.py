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


def within_group_pearson(pred: np.ndarray, true: np.ndarray, groups: np.ndarray) -> float:
    """Correlation of predicted and true variation *within* each group (e.g. cell type).

    For each group, each feature is centered on that group's own mean (separately for
    predictions and truth), and Pearson r is computed across that group's cells, then
    averaged over features. Groups are combined weighted by cell count. Predicting a
    constant per group (e.g. the cell-type-mean baseline, grouped by cell type) scores
    exactly 0, so anything above 0 is signal beyond group identity.
    """
    total, n_cells = 0.0, 0
    for group in np.unique(groups):
        mask = groups == group
        if mask.sum() < 3:
            continue
        p = pred[mask].astype(np.float64)
        t = true[mask].astype(np.float64)
        p -= p.mean(axis=0, keepdims=True)
        t -= t.mean(axis=0, keepdims=True)
        p_norm = np.sqrt((p**2).sum(axis=0))
        t_norm = np.sqrt((t**2).sum(axis=0))
        # A constant prediction carries no within-type signal: score it as r = 0. The
        # tolerance catches float rounding left over from centering a constant column,
        # which would otherwise correlate with the truth by chance.
        varying = (p_norm > 1e-6 * (np.abs(pred[mask]).max(axis=0) + 1e-12) * np.sqrt(mask.sum())) & (t_norm > 0)
        r = np.divide((p * t).sum(axis=0), p_norm * t_norm, out=np.zeros(p.shape[1]), where=varying)
        total += float(r.mean()) * int(mask.sum())
        n_cells += int(mask.sum())
    return total / n_cells if n_cells else 0.0


def evaluate_all(
    pred: np.ndarray,
    true: np.ndarray,
    cell_types: np.ndarray,
    donors: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the full evaluation suite (Pearson r, within-group r, R^2, weighted MSE).

    `within_cell_type_r` removes cell identity. If `donors` is given,
    `within_cell_type_donor_r` also removes donor (and so donor-level disease status)
    by grouping on (cell type, donor), leaving only cell-state variation within a donor.
    """
    metrics = {
        "pearson_r": pearson_r(pred, true),
        "within_cell_type_r": within_group_pearson(pred, true, cell_types),
    }
    if donors is not None:
        groups = np.char.add(np.char.add(cell_types.astype(str), "|"), donors.astype(str))
        metrics["within_cell_type_donor_r"] = within_group_pearson(pred, true, groups)
    metrics["r2"] = r_squared(pred, true)
    metrics["weighted_mse"] = weighted_mse(pred, true, weights)
    return metrics
