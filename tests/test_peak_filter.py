"""Synthetic-data tests for per-cell-type peak-prevalence filtering."""

from __future__ import annotations

import numpy as np
import pytest

ad = pytest.importorskip("anndata")
sp = pytest.importorskip("scipy.sparse")
pd = pytest.importorskip("pandas")

from src.data.peak_filter import compute_peak_prevalence_by_cell_type, filter_peaks_by_prevalence  # noqa: E402

# 100 cells: 10 rare_C, 60 common_A, 20 common_B, 10 unlabeled (NaN cell_type).
N_OBS = 100
RARE_C = slice(0, 10)
COMMON_A = slice(10, 70)
COMMON_B = slice(70, 90)
UNLABELED = slice(90, 100)
MIN_CELL_FRACTION = 0.2


def _make_reference() -> ad.AnnData:
    """5 peaks with deliberately chosen per-type accessibility so filtering has an exact,
    hand-computable answer at min_cell_fraction=0.2:

    - peak_everywhere: accessible in all cells -> kept (passes in every type).
    - peak_rare_marker: accessible in 9/10 rare_C cells (90%), nowhere else -> pooled
      prevalence is only 9%, but per-type prevalence within rare_C (90%) clears the 20%
      bar, so this must be KEPT under per-cell-type logic (and would be wrongly dropped
      by a pooled population-wide threshold, since 9% < 20%).
    - peak_weak_everywhere: accessible in 10% of every labeled type -> below 20% in every
      type -> dropped.
    - peak_unlabeled_only: accessible only in the 10 unlabeled (NaN cell_type) cells ->
      zero count in every labeled type -> dropped.
    - peak_common_marker: accessible in 50% of common_A -> kept via common_A alone.
    """
    n_peaks = 5
    x = np.zeros((N_OBS, n_peaks), dtype=np.float32)
    x[0:9, 0] = 1  # peak_rare_marker: 9/10 rare_C cells
    x[10:16, 1] = 1  # peak_weak_everywhere: 6/60 common_A (10%)
    x[70:72, 1] = 1  # peak_weak_everywhere: 2/20 common_B (10%)
    x[0:1, 1] = 1  # peak_weak_everywhere: 1/10 rare_C (10%)
    x[90:98, 2] = 1  # peak_unlabeled_only: 8/10 unlabeled cells
    x[10:40, 3] = 1  # peak_common_marker: 30/60 common_A (50%)
    x[:, 4] = 1  # peak_everywhere: all cells

    adata = ad.AnnData(X=sp.csr_matrix(x))
    adata.obs_names = [f"cell_{i}" for i in range(N_OBS)]
    adata.var_names = [
        "peak_rare_marker",
        "peak_weak_everywhere",
        "peak_unlabeled_only",
        "peak_common_marker",
        "peak_everywhere",
    ]
    cell_type = np.array([None] * N_OBS, dtype=object)
    cell_type[RARE_C] = "rare_C"
    cell_type[COMMON_A] = "common_A"
    cell_type[COMMON_B] = "common_B"
    # UNLABELED stays None -> NaN cell_type, as in the real reference atlas.
    adata.obs["cell_type"] = pd.Categorical(cell_type)
    # A precomputed embedding, as in the real reference atlas: unaffected by which
    # peaks (columns of X) survive filtering, so it must be carried over unchanged.
    rng = np.random.default_rng(0)
    adata.obsm["X_lsi_harmony"] = rng.random((N_OBS, 6)).astype(np.float32)
    return adata


def test_compute_peak_prevalence_by_cell_type_excludes_unlabeled(tmp_path):
    reference = _make_reference()
    path = tmp_path / "reference.h5ad"
    reference.write_h5ad(path)

    backed = ad.read_h5ad(path, backed="r")
    counts = compute_peak_prevalence_by_cell_type(backed, "cell_type", batch_size=13)

    assert set(counts) == {"rare_C", "common_A", "common_B"}  # unlabeled excluded
    peak_idx = {name: i for i, name in enumerate(reference.var_names)}
    assert counts["rare_C"][peak_idx["peak_rare_marker"]] == 9
    assert counts["common_A"][peak_idx["peak_common_marker"]] == 30
    assert counts["rare_C"][peak_idx["peak_unlabeled_only"]] == 0
    assert counts["common_A"][peak_idx["peak_unlabeled_only"]] == 0
    assert counts["common_B"][peak_idx["peak_unlabeled_only"]] == 0


def test_filter_peaks_keeps_rare_cell_type_marker(tmp_path):
    """The key regression case: a peak specific to a rare cell type must survive filtering
    even though its pooled (population-wide) prevalence is well under the threshold."""
    reference = _make_reference()
    reference_path = tmp_path / "reference.h5ad"
    output_path = tmp_path / "filtered.h5ad"
    reference.write_h5ad(reference_path)

    filter_peaks_by_prevalence(
        reference_atac_path=reference_path,
        output_path=output_path,
        cell_type_col="cell_type",
        min_cell_fraction=MIN_CELL_FRACTION,
        batch_size=17,
    )

    result = ad.read_h5ad(output_path)
    kept = set(result.var_names)
    assert kept == {"peak_rare_marker", "peak_common_marker", "peak_everywhere"}
    assert "peak_weak_everywhere" not in kept
    assert "peak_unlabeled_only" not in kept
    assert result.n_obs == N_OBS

    # Regression: obsm (e.g. the LSI embedding pseudobulk.compute_knn_neighbors needs)
    # must survive filtering unchanged -- it's a per-cell embedding, not per-peak.
    assert "X_lsi_harmony" in result.obsm
    np.testing.assert_array_equal(result.obsm["X_lsi_harmony"], reference.obsm["X_lsi_harmony"])
