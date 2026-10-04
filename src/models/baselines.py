"""Baseline models for RNA -> ATAC-representation prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


class CellTypeMeanBaseline:
    """Predicts the training-set mean ATAC representation of each cell's type.

    The strongest cheap baseline for RNA -> ATAC: cell identity explains most of the
    signal, so a model has to beat this to show it uses cell state beyond cell type.
    Cell types unseen in training (e.g. the held-out type) fall back to the global mean.
    """

    def __init__(self) -> None:
        self.cell_types_: np.ndarray | None = None
        self.means_: np.ndarray | None = None
        self.global_mean_: np.ndarray | None = None

    def fit(self, cell_types: np.ndarray, target: np.ndarray) -> CellTypeMeanBaseline:
        self.cell_types_ = np.unique(cell_types).astype(str)
        self.means_ = np.stack([target[cell_types == ct].mean(axis=0) for ct in self.cell_types_])
        self.global_mean_ = target.mean(axis=0)
        return self

    def predict(self, cell_types: np.ndarray) -> np.ndarray:
        if self.means_ is None:
            raise RuntimeError("CellTypeMeanBaseline.fit must be called before predict.")  # noqa: TRY003
        lookup = dict(zip(self.cell_types_, self.means_, strict=True))
        return np.stack([lookup.get(str(ct), self.global_mean_) for ct in cell_types])

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            np.savez(f, cell_types=self.cell_types_, means=self.means_, global_mean=self.global_mean_)

    @classmethod
    def load(cls, path: str | Path) -> CellTypeMeanBaseline:
        baseline = cls()
        with np.load(path) as saved:
            baseline.cell_types_ = saved["cell_types"]
            baseline.means_ = saved["means"]
            baseline.global_mean_ = saved["global_mean"]
        return baseline


class MLPBaseline(nn.Module):
    """Simple 2-layer MLP mapping scVI cell-state embeddings to ATAC-representation embeddings."""

    def __init__(self, cond_dim: int, target_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        return self.net(cond)

    def loss(self, cond: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(self(cond), target)
