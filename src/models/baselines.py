"""Baseline models for RNA -> ATAC-representation prediction."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class MeanBaseline:
    """Predicts the training-set mean ATAC representation regardless of input."""

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None

    def fit(self, cond: np.ndarray, target: np.ndarray) -> MeanBaseline:
        self.mean_ = target.mean(axis=0)
        return self

    def predict(self, cond: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("MeanBaseline.fit must be called before predict.")  # noqa: TRY003
        return np.tile(self.mean_, (cond.shape[0], 1))


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
