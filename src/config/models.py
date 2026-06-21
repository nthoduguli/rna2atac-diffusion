from typing import Any

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    version: int
    disable_existing_loggers: bool
    formatters: dict[str, Any]
    handlers: dict[str, Any]
    loggers: dict[str, Any]
    root: dict[str, Any]


class HyperparametersConfig(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int


class DataConfig(BaseModel):
    rna_path: str = "data/rna.h5ad"
    atac_path: str = "data/atac.h5ad"
    held_out_cell_types: list[str] = []
    val_fraction: float = 0.1
    seed: int = 0


class ScVIConfig(BaseModel):
    n_latent: int = 30
    max_epochs: int = 400
    batch_key: str | None = "batch"


class PeakVIConfig(BaseModel):
    n_latent: int = 30
    max_epochs: int = 400
    batch_key: str | None = "batch"


class DiffusionConfig(BaseModel):
    n_timesteps: int = 1000
    hidden_dim: int = 256
    n_layers: int = 4
    time_dim: int = 64
    max_epochs: int = 200
    learning_rate: float = 1e-4


class BaselineConfig(BaseModel):
    mlp_hidden_dim: int = 256
    mlp_max_epochs: int = 100
    mlp_learning_rate: float = 1e-3


class CheckpointConfig(BaseModel):
    output_dir: str = "checkpoints"
    scvi_dir: str = "checkpoints/scvi"
    peakvi_dir: str = "checkpoints/peakvi"
    diffusion_path: str = "checkpoints/diffusion.pt"
    mlp_baseline_path: str = "checkpoints/mlp_baseline.pt"
    mean_baseline_path: str = "checkpoints/mean_baseline.npy"


class FiguresConfig(BaseModel):
    output_dir: str = "figures"
