"""Synthetic-data smoke test exercising the full train/eval/inference pipeline."""

from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("anndata")

from src.config import Config  # noqa: E402
from src.config.models import (  # noqa: E402
    BaselineConfig,
    CheckpointConfig,
    DataConfig,
    DiffusionConfig,
    HyperparametersConfig,
    LoggingConfig,
    PeakVIConfig,
    ScVIConfig,
)
from src.data import load_paired_data, split_by_cell_type  # noqa: E402
from src.metrics import evaluate_all, pearson_r  # noqa: E402
from src.models import ConditionalDDPM, MeanBaseline, MLPBaseline  # noqa: E402


def _make_synthetic_adata(n_obs: int, n_vars: int, cell_types: list[str], seed: int) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    x = rng.poisson(lam=2.0, size=(n_obs, n_vars)).astype(np.float32)
    obs = {
        "cell_type": rng.choice(cell_types, size=n_obs),
        "batch": rng.choice(["batch_0", "batch_1"], size=n_obs),
    }
    adata = ad.AnnData(X=x, obs=obs)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    return adata


@pytest.fixture
def synthetic_data(tmp_path):
    cell_types = ["excitatory", "inhibitory", "astrocyte", "microglia"]
    rna = _make_synthetic_adata(n_obs=200, n_vars=50, cell_types=cell_types, seed=0)
    atac = _make_synthetic_adata(n_obs=200, n_vars=80, cell_types=cell_types, seed=1)
    atac.obs_names = rna.obs_names
    atac.obs["cell_type"] = rna.obs["cell_type"].to_numpy()
    atac.obs["batch"] = rna.obs["batch"].to_numpy()

    rna_path = tmp_path / "rna.h5ad"
    atac_path = tmp_path / "atac.h5ad"
    rna.write_h5ad(rna_path)
    atac.write_h5ad(atac_path)
    return rna_path, atac_path


def _make_config(tmp_path, rna_path, atac_path) -> Config:
    return Config(
        logging=LoggingConfig(
            version=1,
            disable_existing_loggers=False,
            formatters={},
            handlers={},
            loggers={},
            root={"handlers": [], "level": "WARNING"},
        ),
        hyperparameters=HyperparametersConfig(learning_rate=1e-3, batch_size=32, num_epochs=1),
        data=DataConfig(
            rna_path=str(rna_path),
            atac_path=str(atac_path),
            held_out_cell_types=["microglia"],
            val_fraction=0.1,
            seed=0,
        ),
        scvi=ScVIConfig(n_latent=5, max_epochs=1, batch_key="batch"),
        peakvi=PeakVIConfig(n_latent=5, max_epochs=1, batch_key="batch"),
        diffusion=DiffusionConfig(
            n_timesteps=10, hidden_dim=16, n_layers=1, time_dim=8, max_epochs=1, learning_rate=1e-3
        ),
        baseline=BaselineConfig(mlp_hidden_dim=16, mlp_max_epochs=1, mlp_learning_rate=1e-3),
        checkpoints=CheckpointConfig(
            output_dir=str(tmp_path / "checkpoints"),
            scvi_dir=str(tmp_path / "checkpoints" / "scvi"),
            peakvi_dir=str(tmp_path / "checkpoints" / "peakvi"),
            diffusion_path=str(tmp_path / "checkpoints" / "diffusion.pt"),
            mlp_baseline_path=str(tmp_path / "checkpoints" / "mlp_baseline.pt"),
            mean_baseline_path=str(tmp_path / "checkpoints" / "mean_baseline.npy"),
        ),
    )


def test_split_by_cell_type(synthetic_data):
    rna_path, atac_path = synthetic_data
    data = load_paired_data(rna_path, atac_path)
    splits = split_by_cell_type(data, held_out_cell_types=["microglia"], val_fraction=0.1, seed=0)

    assert set(splits) == {"train", "val", "test", "held_out"}
    assert len(splits["held_out"]) > 0
    assert (data.rna.obs["cell_type"].to_numpy()[splits["held_out"]] == "microglia").all()
    for name in ("train", "val", "test"):
        assert (data.rna.obs["cell_type"].to_numpy()[splits[name]] != "microglia").all()


def test_diffusion_loss_and_sample_shapes():
    torch = pytest.importorskip("torch")

    cond_dim, target_dim, batch_size = 5, 8, 16
    model = ConditionalDDPM(target_dim=target_dim, cond_dim=cond_dim, n_timesteps=10, hidden_dim=16, n_layers=1)

    cond = torch.randn(batch_size, cond_dim)
    target = torch.randn(batch_size, target_dim)

    loss = model.loss(target, cond)
    assert loss.ndim == 0
    assert loss.item() >= 0

    samples = model.sample(cond, target_dim=target_dim)
    assert samples.shape == (batch_size, target_dim)


def test_baselines_fit_predict_shapes():
    torch = pytest.importorskip("torch")

    cond = np.random.default_rng(0).normal(size=(20, 5)).astype(np.float32)
    target = np.random.default_rng(1).normal(size=(20, 8)).astype(np.float32)

    mean_baseline = MeanBaseline().fit(cond, target)
    mean_pred = mean_baseline.predict(cond)
    assert mean_pred.shape == target.shape
    assert np.allclose(mean_pred[0], mean_pred[1])

    mlp = MLPBaseline(cond_dim=5, target_dim=8, hidden_dim=16)
    cond_t = torch.as_tensor(cond)
    target_t = torch.as_tensor(target)
    loss = mlp.loss(cond_t, target_t)
    assert loss.item() >= 0


def test_metrics_identity():
    x = np.random.default_rng(0).normal(size=(10, 4))
    assert pytest.approx(pearson_r(x, x), rel=1e-5) == 1.0

    metrics = evaluate_all(x, x)
    assert pytest.approx(metrics["pearson_r"], rel=1e-5) == 1.0
    assert pytest.approx(metrics["r2"], rel=1e-5) == 1.0
    assert metrics["weighted_mse"] == pytest.approx(0.0)


@pytest.mark.slow
def test_full_pipeline_smoke(synthetic_data, tmp_path):
    """End-to-end train -> eval -> inference smoke test on synthetic data with tiny configs."""
    pytest.importorskip("scvi")

    rna_path, atac_path = synthetic_data
    config = _make_config(tmp_path, rna_path, atac_path)

    from src.pipeline import evaluate, run_inference, train

    train(config)

    results = evaluate(config, split="test")
    assert set(results) == {"mean_baseline", "mlp_baseline", "diffusion"}
    for metrics in results.values():
        assert set(metrics) == {"pearson_r", "r2", "weighted_mse"}

    generated = run_inference(config, rna_path)
    assert generated.shape[0] == 200
