"""Synthetic-data tests for KNN pseudobulking, RNA standardization, and stratified sampling."""

from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("anndata")
pytest.importorskip("sklearn")
pytest.importorskip("h5py")

from src.data.pseudobulk import (  # noqa: E402
    compute_knn_neighbors,
    run_pseudobulking,
    standardize_rna_cell_type,
    stratified_downsample_paired,
)

CELL_TYPES = ["Oli", "Exc", "Inh", "OPC"]


def _make_atac(n_obs: int, n_vars: int, cell_types: list[str], seed: int, embedding_dim: int = 4) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    x = rng.poisson(lam=1.0, size=(n_obs, n_vars)).astype(np.float32)
    obs = {"cell_type_label": rng.choice(cell_types, size=n_obs), "batch": rng.choice(["b0", "b1"], size=n_obs)}
    adata = ad.AnnData(X=x, obs=obs)
    adata.obs_names = [f"cell_{seed}_{i}" for i in range(n_obs)]
    # Cluster the embedding by cell type so same-type nearest neighbors are meaningful.
    type_offsets = {ct: rng.normal(scale=10.0, size=embedding_dim) for ct in cell_types}
    embedding = np.stack([type_offsets[ct] for ct in adata.obs["cell_type_label"]])
    embedding += rng.normal(scale=0.1, size=embedding.shape)
    adata.obsm["X_lsi_harmony"] = embedding.astype(np.float32)
    return adata


@pytest.fixture
def reference_atac():
    return _make_atac(n_obs=200, n_vars=30, cell_types=CELL_TYPES, seed=1)


@pytest.fixture
def query_atac():
    return _make_atac(n_obs=20, n_vars=30, cell_types=CELL_TYPES, seed=2)


def test_compute_knn_neighbors_same_cell_type(query_atac, reference_atac):
    k = 5
    neighbor_indices = compute_knn_neighbors(query_atac, reference_atac, "cell_type_label", k=k)

    assert set(neighbor_indices) == set(query_atac.obs_names)
    reference_types = reference_atac.obs["cell_type_label"].to_numpy()
    for cell_id, indices in neighbor_indices.items():
        assert len(indices) == k
        query_type = query_atac.obs.loc[cell_id, "cell_type_label"]
        assert (reference_types[indices] == query_type).all()


def test_run_pseudobulking_roundtrip(tmp_path, query_atac, reference_atac):
    query_path = tmp_path / "query_atac.h5ad"
    reference_path = tmp_path / "reference_atac.h5ad"
    output_path = tmp_path / "pseudobulk_atac.h5ad"
    query_atac.write_h5ad(query_path)
    reference_atac.write_h5ad(reference_path)

    run_pseudobulking(
        query_atac_path=query_path,
        reference_atac_path=reference_path,
        output_path=output_path,
        cell_type_col="cell_type_label",
        k=5,
        batch_size=7,
    )

    result = ad.read_h5ad(output_path)
    assert result.n_obs == query_atac.n_obs
    assert result.n_vars == reference_atac.n_vars
    assert list(result.obs_names) == list(query_atac.obs_names)
    assert "cell_type" in result.obs.columns
    assert (result.obs["cell_type"].to_numpy() == query_atac.obs["cell_type_label"].to_numpy()).all()
    # Each pseudobulked row sums k>=1 nonnegative reference rows, so should be >= any single one.
    assert (result.X.toarray() >= 0).all()
    assert result.X.sum() > 0


def test_run_pseudobulking_writes_64bit_indptr(tmp_path, query_atac, reference_atac):
    """Regression test: a 32 bit on-disk indptr overflows once enough dense pseudobulked rows
    accumulate (hit in a real ~10k-cell run against a ~1M-peak reference atlas), so the output
    must be created with a 64 bit indptr regardless of how small this synthetic case is."""
    query_path = tmp_path / "query_atac.h5ad"
    reference_path = tmp_path / "reference_atac.h5ad"
    output_path = tmp_path / "pseudobulk_atac.h5ad"
    query_atac.write_h5ad(query_path)
    reference_atac.write_h5ad(reference_path)

    run_pseudobulking(
        query_atac_path=query_path,
        reference_atac_path=reference_path,
        output_path=output_path,
        cell_type_col="cell_type_label",
        k=5,
        batch_size=7,
    )

    h5py = pytest.importorskip("h5py")
    with h5py.File(output_path, "r") as f:
        assert f["X"]["indptr"].dtype == np.int64
        assert f["X"]["indices"].dtype == np.int64


def test_standardize_rna_cell_type(tmp_path):
    rna = _make_atac(n_obs=15, n_vars=10, cell_types=CELL_TYPES, seed=3)
    rna_path = tmp_path / "rna.h5ad"
    output_path = tmp_path / "rna_standardized.h5ad"
    rna.write_h5ad(rna_path)

    standardize_rna_cell_type(rna_path, output_path, cell_type_col="cell_type_label")

    result = ad.read_h5ad(output_path)
    assert "cell_type" in result.obs.columns
    assert (result.obs["cell_type"].to_numpy() == rna.obs["cell_type_label"].astype(str).to_numpy()).all()


def test_stratified_downsample_paired(tmp_path):
    rna = _make_atac(n_obs=500, n_vars=10, cell_types=CELL_TYPES, seed=4)
    atac = _make_atac(n_obs=500, n_vars=20, cell_types=CELL_TYPES, seed=4)
    atac.obs_names = rna.obs_names
    atac.obs["cell_type_label"] = rna.obs["cell_type_label"].to_numpy()

    rna_path, atac_path = tmp_path / "rna.h5ad", tmp_path / "atac.h5ad"
    output_rna_path, output_atac_path = tmp_path / "rna_sample.h5ad", tmp_path / "atac_sample.h5ad"
    rna.write_h5ad(rna_path)
    atac.write_h5ad(atac_path)

    stratified_downsample_paired(
        rna_path=rna_path,
        atac_path=atac_path,
        output_rna_path=output_rna_path,
        output_atac_path=output_atac_path,
        cell_type_col="cell_type_label",
        total_budget=100,
        min_per_type=5,
        seed=0,
    )

    sampled_rna = ad.read_h5ad(output_rna_path)
    sampled_atac = ad.read_h5ad(output_atac_path)
    assert sampled_rna.n_obs == sampled_atac.n_obs
    assert list(sampled_rna.obs_names) == list(sampled_atac.obs_names)
    assert sampled_rna.n_obs <= 100
    counts = sampled_atac.obs["cell_type_label"].value_counts()
    for cell_type in CELL_TYPES:
        assert counts.get(cell_type, 0) >= min(5, (atac.obs["cell_type_label"] == cell_type).sum())
