"""Peak-prevalence filtering for snATAC-seq reference atlases.

Follows scvi-tools' documented PeakVI preprocessing recommendation (filter peaks
accessible in fewer than a minimum fraction of cells, via `sc.pp.filter_genes`;
see https://docs.scvi-tools.org/en/stable/tutorials/notebooks/use_cases/preprocessing.html)
-- peaks rarely accessible in any cell contribute little signal but dominate the
memory and compute cost of a PeakVI model whose input/output width is `n_peaks`.

Prevalence is computed *per cell type*, not pooled across the whole population: a peak
accessible in 100% of a cell type that's only e.g. 5% of the population has a pooled
prevalence of ~5%, indistinguishable from a peak that's weakly, uniformly accessible
everywhere. Thresholding per cell type and keeping a peak if it clears the bar in *any*
type preserves cell-type-specific marker peaks regardless of how rare that type is.

This filters the reference atlas *before* `pseudobulk.run_pseudobulking`, not after:
`run_pseudobulking` takes its output `var` (and therefore final peak space) from the
reference atlas, and summing k neighbor cells per output row densifies the matrix
substantially, so filtering upstream keeps that densification from ever happening
over the full, unfiltered peak set.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from src.data.sparse_io import write_streamed_sparse_h5ad

logger = logging.getLogger(__name__)


def compute_peak_prevalence_by_cell_type(
    reference: ad.AnnData, cell_type_col: str, batch_size: int = 50_000
) -> dict[str, np.ndarray]:
    """Count, per peak and per cell type, the number of cells (of that type) in which
    it's accessible (nonzero).

    Computed only over cells with a non-null `cell_type_col` label: in a real reference
    atlas, a large fraction of cells may have no cell-type annotation at all (e.g. ~81%
    in the atlas this project uses) and are excluded here -- they're also never selected
    as pseudobulking KNN donors, since `compute_knn_neighbors` only matches cells against
    known type labels.

    Streams through `reference` in contiguous row-chunks, so this works on a backed
    (out-of-core) AnnData without materializing the full matrix. Contiguous chunk reads
    are fast on HDF5, unlike the scattered per-cell-neighbor reads
    `pseudobulk.run_pseudobulking` needs (see its docstring) -- this only has to scan
    the reference once, sequentially.
    """
    if reference.X is None:
        raise ValueError("Reference AnnData has no X matrix.")  # noqa: TRY003

    n_peaks = reference.n_vars
    cell_types = pd.DataFrame(reference.obs)[cell_type_col].to_numpy()
    known_types = sorted({cell_type for cell_type in cell_types if pd.notna(cell_type)})
    counts = {cell_type: np.zeros(n_peaks, dtype=np.int64) for cell_type in known_types}

    for start in tqdm(range(0, reference.n_obs, batch_size), desc="Counting peak prevalence by cell type"):
        chunk = reference.X[start : start + batch_size]
        chunk_types = cell_types[start : start + batch_size]
        for cell_type in known_types:
            mask = chunk_types == cell_type
            if mask.any():
                counts[cell_type] += np.bincount(chunk[mask].indices, minlength=n_peaks)

    return counts


def _filtered_chunks(reference_x, n_obs: int, keep_mask: np.ndarray, batch_size: int) -> Iterator[sp.csr_matrix]:
    for start in tqdm(range(0, n_obs, batch_size), desc="Writing peak-filtered reference"):
        chunk = reference_x[start : start + batch_size]
        yield chunk[:, keep_mask].tocsr()


def filter_peaks_by_prevalence(
    reference_atac_path: str | Path,
    output_path: str | Path,
    cell_type_col: str,
    min_cell_fraction: float = 0.05,
    batch_size: int = 50_000,
) -> None:
    """Write a copy of a reference ATAC atlas restricted to peaks accessible in at least
    `min_cell_fraction` of cells of at least one cell type (scvi-tools' documented PeakVI
    preprocessing recommendation, using their example default threshold of 5% -- validate
    this against the target dataset's own accessibility distribution rather than assuming
    it transfers). See module docstring for why this is per-cell-type, not pooled.

    Streams both the prevalence count and the filtered write in row-chunks, so this
    runs on a reference atlas far too large to load into memory at once.
    """
    reference = ad.read_h5ad(reference_atac_path, backed="r")
    n_peaks_before = reference.n_vars
    cell_types = pd.DataFrame(reference.obs)[cell_type_col]

    counts_by_type = compute_peak_prevalence_by_cell_type(reference, cell_type_col, batch_size=batch_size)
    keep_mask = np.zeros(n_peaks_before, dtype=bool)
    for cell_type, counts in counts_by_type.items():
        n_cells_of_type = int((cell_types == cell_type).sum())
        min_cells = int(n_cells_of_type * min_cell_fraction)
        type_keep = counts >= min_cells
        keep_mask |= type_keep
        logger.info(
            "%s: %d cells, min_cells=%d, %d peaks pass threshold for this type alone.",
            cell_type,
            n_cells_of_type,
            min_cells,
            int(type_keep.sum()),
        )

    n_peaks_after = int(keep_mask.sum())
    logger.info(
        "Keeping %d/%d peaks (prevalent in >= %.1f%% of at least one cell type).",
        n_peaks_after,
        n_peaks_before,
        min_cell_fraction * 100,
    )

    write_streamed_sparse_h5ad(
        output_path,
        obs=pd.DataFrame(reference.obs).copy(),
        var=pd.DataFrame(reference.var)[keep_mask].copy(),
        row_chunks=_filtered_chunks(reference.X, reference.n_obs, keep_mask, batch_size),
        obsm=dict(reference.obsm),
    )

    logger.info(
        "Wrote peak-filtered reference atlas (%d cells x %d peaks) to %s",
        reference.n_obs,
        n_peaks_after,
        output_path,
    )
