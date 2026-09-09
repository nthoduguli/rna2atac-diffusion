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


class PeakFilterConfig(BaseModel):
    reference_atac_path: str = "data/raw/atac_reference_atlas.h5ad"
    output_path: str = "data/raw/atac_reference_atlas_filtered.h5ad"
    cell_type_col: str = "cell_type"
    min_cell_fraction: float = 0.05
    batch_size: int = 50_000


class PseudobulkConfig(BaseModel):
    query_atac_path: str = "data/raw/atac.h5ad"
    reference_atac_path: str = "data/raw/atac_reference_atlas.h5ad"
    output_path: str = "data/processed/paired/atac.h5ad"
    query_rna_path: str = "data/raw/rna.h5ad"
    rna_output_path: str = "data/processed/paired/rna.h5ad"
    cell_type_col: str = "cell_type"
    embedding_key: str = "X_lsi_harmony"
    k: int = 100
    batch_size: int = 2000


class TestSampleConfig(BaseModel):
    output_rna_path: str = "data/raw/rna_sample.h5ad"
    output_atac_path: str = "data/raw/atac_sample.h5ad"
    total_budget: int = 10_000
    min_per_type: int = 10
    seed: int = 42


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
