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
    FiguresConfig,
    HyperparametersConfig,
    LoggingConfig,
    PeakVIConfig,
    ScVIConfig,
)
from src.data import load_paired_data, split_by_donor  # noqa: E402
from src.metrics import evaluate_all, pearson_r, within_group_pearson  # noqa: E402
from src.models import CellTypeMeanBaseline, ConditionalDDPM, MLPBaseline  # noqa: E402


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
    # 10 donors, condition constant per donor (even donors AD, odd donors nonAD).
    donors = np.random.default_rng(2).integers(0, 10, size=200)
    rna.obs["donor"] = donors
    rna.obs["condition"] = np.where(donors % 2 == 0, "AD", "nonAD")

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
            donor_col="donor",
            condition_col="condition",
            n_val_donors=2,
            n_test_donors=2,
            seed=0,
        ),
        scvi=ScVIConfig(n_latent=5, max_epochs=1, batch_key="batch"),
        peakvi=PeakVIConfig(n_latent=5, max_epochs=1, batch_key="batch"),
        diffusion=DiffusionConfig(
            n_timesteps=10, hidden_dim=16, n_layers=1, time_dim=8, max_epochs=1, learning_rate=1e-3, n_eval_samples=2
        ),
        baseline=BaselineConfig(mlp_hidden_dim=16, mlp_max_epochs=1, mlp_learning_rate=1e-3),
        checkpoints=CheckpointConfig(
            output_dir=str(tmp_path / "checkpoints"),
            scvi_dir=str(tmp_path / "checkpoints" / "scvi"),
            peakvi_dir=str(tmp_path / "checkpoints" / "peakvi"),
            diffusion_path=str(tmp_path / "checkpoints" / "diffusion.pt"),
            mlp_baseline_path=str(tmp_path / "checkpoints" / "mlp_baseline.pt"),
            mean_baseline_path=str(tmp_path / "checkpoints" / "celltype_mean_baseline.npz"),
        ),
        figures=FiguresConfig(output_dir=str(tmp_path / "figures")),
    )


def _split(data, seed=0):
    return split_by_donor(
        data,
        donor_col="donor",
        held_out_cell_types=["microglia"],
        n_val_donors=2,
        n_test_donors=2,
        condition_col="condition",
        seed=seed,
    )


def test_split_by_donor_holds_out_cell_type(synthetic_data):
    data = load_paired_data(*synthetic_data)
    splits = _split(data)

    assert set(splits) == {"train", "val", "test", "held_out"}
    cell_types = data.rna.obs["cell_type"].to_numpy()
    assert len(splits["held_out"]) > 0
    assert (cell_types[splits["held_out"]] == "microglia").all()
    for name in ("train", "val", "test"):
        assert (cell_types[splits[name]] != "microglia").all()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_split_by_donor_has_no_donor_overlap(synthetic_data, seed):
    data = load_paired_data(*synthetic_data)
    splits = _split(data, seed=seed)
    donors = data.rna.obs["donor"].to_numpy()

    split_donors = {name: set(donors[splits[name]]) for name in ("train", "val", "test")}
    assert not split_donors["train"] & split_donors["test"]
    assert not split_donors["train"] & split_donors["val"]
    assert not split_donors["val"] & split_donors["test"]
    assert len(split_donors["test"]) == 2
    assert len(split_donors["val"]) == 2
    # Every non-held-out cell lands in exactly one split.
    all_idx = np.concatenate([splits[name] for name in splits])
    assert len(all_idx) == len(np.unique(all_idx)) == data.rna.n_obs


def test_split_by_donor_balances_condition(synthetic_data):
    data = load_paired_data(*synthetic_data)
    splits = _split(data)
    obs = data.rna.obs.iloc[splits["test"]]
    conditions = obs.drop_duplicates("donor")["condition"]
    assert sorted(conditions) == ["AD", "nonAD"]


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


def test_baselines_fit_predict_shapes(tmp_path):
    torch = pytest.importorskip("torch")

    rng = np.random.default_rng(0)
    cond = rng.normal(size=(20, 5)).astype(np.float32)
    target = rng.normal(size=(20, 8)).astype(np.float32)
    cell_types = np.array(["a", "b"] * 10)

    baseline = CellTypeMeanBaseline().fit(cell_types, target)
    baseline.save(tmp_path / "baseline.npz")
    baseline = CellTypeMeanBaseline.load(tmp_path / "baseline.npz")
    pred = baseline.predict(np.array(["a", "b", "unseen"]))
    assert pred.shape == (3, 8)
    assert np.allclose(pred[0], target[cell_types == "a"].mean(axis=0), atol=1e-6)
    assert np.allclose(pred[2], target.mean(axis=0), atol=1e-6)

    mlp = MLPBaseline(cond_dim=5, target_dim=8, hidden_dim=16)
    loss = mlp.loss(torch.as_tensor(cond), torch.as_tensor(target))
    assert loss.item() >= 0


def test_metrics_identity():
    x = np.random.default_rng(0).normal(size=(10, 4))
    assert pytest.approx(pearson_r(x, x), rel=1e-5) == 1.0

    metrics = evaluate_all(x, x, cell_types=np.array(["a"] * 5 + ["b"] * 5))
    assert pytest.approx(metrics["pearson_r"], rel=1e-5) == 1.0
    assert pytest.approx(metrics["within_cell_type_r"], rel=1e-5) == 1.0
    assert "within_cell_type_donor_r" not in metrics

    donors = np.array([0, 1] * 5)
    metrics = evaluate_all(x, x, cell_types=np.array(["a"] * 5 + ["b"] * 5), donors=donors)
    assert pytest.approx(metrics["within_cell_type_donor_r"], rel=1e-5) == 1.0
    assert pytest.approx(metrics["r2"], rel=1e-5) == 1.0
    assert metrics["weighted_mse"] == pytest.approx(0.0)


def test_within_group_pearson_ignores_group_means():
    rng = np.random.default_rng(0)
    cell_types = np.array(["a"] * 50 + ["b"] * 50)
    type_means = np.where(cell_types[:, None] == "a", 5.0, -5.0) * np.ones((100, 4))
    true = type_means + rng.normal(size=(100, 4))

    # Predicting each type's mean perfectly scores high per-cell r but 0 within type.
    assert within_group_pearson(type_means, true, cell_types) == 0.0
    assert within_group_pearson(true, true, cell_types) == pytest.approx(1.0)
    assert abs(within_group_pearson(type_means + rng.normal(size=(100, 4)), true, cell_types)) < 0.3


def test_batch_key_check_rejects_per_donor_batches(synthetic_data, tmp_path):
    from src.pipeline import _check_batch_keys_cover_splits

    data = load_paired_data(*synthetic_data)
    config = _make_config(tmp_path, *synthetic_data)
    splits = _split(data)
    _check_batch_keys_cover_splits(config, data, splits)  # shared "batch" column: fine

    data.atac.obs["library"] = data.rna.obs["donor"].astype(str).to_numpy()
    config.peakvi.batch_key = "library"
    with pytest.raises(ValueError, match="absent from train"):
        _check_batch_keys_cover_splits(config, data, splits)


@pytest.mark.slow
def test_full_pipeline_smoke(synthetic_data, tmp_path):
    """End-to-end train -> eval -> inference smoke test on synthetic data with tiny configs."""
    pytest.importorskip("scvi")

    rna_path, atac_path = synthetic_data
    config = _make_config(tmp_path, rna_path, atac_path)

    from src.pipeline import evaluate, run_inference, train

    train(config)

    results = evaluate(config, split="test")
    assert set(results) == {
        "celltype_mean",
        "mlp",
        "mlp_shuffled_cond",
        "diffusion",
        "diffusion_1sample",
        "diffusion_shuffled_cond",
    }
    for metrics in results.values():
        assert set(metrics) == {"pearson_r", "within_cell_type_r", "within_cell_type_donor_r", "r2", "weighted_mse"}
    assert results["celltype_mean"]["within_cell_type_r"] == 0.0
    assert (tmp_path / "figures" / "predictions_test.npz").exists()

    from src.pipeline import summarize_results

    table = summarize_results(config, split="test")
    assert table.count("\n") == 2 + 6
    assert (tmp_path / "figures" / "results_test.png").exists()

    generated = run_inference(config, rna_path)
    assert generated.shape[0] == 200
