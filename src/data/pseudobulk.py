"""KNN-based pseudobulking of snATAC-seq accessibility tracks.

Smooths each cell's ATAC track by summing it with the tracks of its k nearest
same-cell-type neighbors, drawn from a larger reference ATAC atlas rather than
the (comparatively small, per-donor-sparse) paired multiome cohort itself.
Neighbors are found per cell type in a precomputed embedding (e.g. harmonized
LSI), so a cell only borrows accessibility from same-type reference cells.

Because smoothing is applied per query cell rather than per cell-type group,
the result preserves the paired cohort's 1:1 cell-level correspondence with
the matching snRNA-seq profiles -- unlike cell-type-level pseudobulk
aggregation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from src.data.sparse_io import write_streamed_sparse_h5ad

logger = logging.getLogger(__name__)


def compute_knn_neighbors(
    query: ad.AnnData,
    reference: ad.AnnData,
    cell_type_col: str,
    embedding_key: str = "X_lsi_harmony",
    k: int = 100,
) -> dict[str, np.ndarray]:
    """Find each query cell's k nearest same-cell-type neighbors in `reference`.

    Neighbors are searched independently per cell type in `embedding_key` space
    (present in both AnnData objects' `.obsm`), so a query cell only borrows
    accessibility from reference cells of the same type.

    Returns:
        Mapping from query cell barcode to an array of integer row indices into
        `reference`.
    """
    neighbor_indices: dict[str, np.ndarray] = {}
    reference_types = reference.obs[cell_type_col].to_numpy()
    reference_embedding = reference.obsm[embedding_key]

    for cell_type in tqdm(query.obs[cell_type_col].unique(), desc="Building KNN per cell type"):
        reference_mask = reference_types == cell_type
        reference_indices = np.where(reference_mask)[0]
        query_mask = (query.obs[cell_type_col] == cell_type).to_numpy()
        n_query_cells = int(query_mask.sum())

        if len(reference_indices) == 0:
            logger.warning("No reference cells of type %r; dropping %d query cells.", cell_type, n_query_cells)
            continue

        query_embedding = query.obsm[embedding_key][query_mask]
        query_ids = query.obs_names[query_mask]

        n_neighbors = min(k, len(reference_indices))
        nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=-1)
        nn.fit(reference_embedding[reference_mask])
        _, local_indices = nn.kneighbors(query_embedding)
        global_indices = reference_indices[local_indices]

        for i, cell_id in enumerate(query_ids):
            neighbor_indices[cell_id] = global_indices[i]

    return neighbor_indices


def _sum_neighbor_tracks(reference_x, neighbor_indices: list[np.ndarray]) -> sp.csr_matrix:
    """Sum reference rows for each cell's neighbor set into one sparse matrix block."""
    unique_neighbors = np.unique(np.concatenate(neighbor_indices))
    neighbor_data = reference_x[unique_neighbors]
    position = {idx: i for i, idx in enumerate(unique_neighbors)}

    rows = []
    for indices in neighbor_indices:
        local_positions = [position[i] for i in indices]
        rows.append(sp.csr_matrix(neighbor_data[local_positions].sum(axis=0)))
    return sp.vstack(rows).tocsr()


def _pseudobulk_chunks(
    reference_x, query_order: list[str], neighbor_indices: dict[str, np.ndarray], batch_size: int
) -> Iterator[sp.csr_matrix]:
    for start in tqdm(range(0, len(query_order), batch_size), desc="Pseudobulking"):
        batch_ids = query_order[start : start + batch_size]
        batch_neighbors = [neighbor_indices[cell_id] for cell_id in batch_ids]
        yield _sum_neighbor_tracks(reference_x, batch_neighbors)


def run_pseudobulking(
    query_atac_path: str | Path,
    reference_atac_path: str | Path,
    output_path: str | Path,
    cell_type_col: str,
    embedding_key: str = "X_lsi_harmony",
    k: int = 100,
    batch_size: int = 2000,
) -> None:
    """Pseudobulk a paired-multiome ATAC AnnData via KNN smoothing against a reference atlas.

    Writes an `.h5ad` to `output_path` with the same cell order as `query_atac_path`
    (so it stays 1:1-paired with the corresponding snRNA-seq file, for
    `PairedMultiomeData`/`load_paired_data`) and `var` taken from the reference atlas.
    Streams the (potentially very large) output to disk in `batch_size` chunks so peak
    memory stays bounded regardless of cohort or atlas size.

    Args:
        query_atac_path: Paired multiome ATAC `.h5ad` to smooth (loaded fully in memory;
            expected to be small enough to do so).
        reference_atac_path: Large reference ATAC atlas `.h5ad`, opened backed, used only
            as the pool of neighbors to borrow accessibility from.
        output_path: Destination `.h5ad` path.
        cell_type_col: `obs` column present in both files, used to restrict KNN search to
            same-cell-type neighbors. Written to the output as `obs["cell_type"]`, matching
            `PairedMultiomeData`'s expectations.
        embedding_key: `obsm` key present in both files to search neighbors in.
        k: Number of neighbors to sum per query cell.
        batch_size: Number of query cells processed per streamed write.
    """
    query = ad.read_h5ad(query_atac_path)
    reference = ad.read_h5ad(reference_atac_path, backed="r")

    neighbor_indices = compute_knn_neighbors(query, reference, cell_type_col, embedding_key, k)
    query_order = [cell_id for cell_id in query.obs_names if cell_id in neighbor_indices]
    n_dropped = query.n_obs - len(query_order)
    if n_dropped > 0:
        logger.warning("Dropping %d query cells with no same-cell-type reference neighbors.", n_dropped)

    obs = pd.DataFrame(query.obs).loc[query_order].copy()
    obs["cell_type"] = obs[cell_type_col].astype(str)
    n_peaks = reference.n_vars

    write_streamed_sparse_h5ad(
        output_path,
        obs=obs,
        var=pd.DataFrame(reference.var).copy(),
        row_chunks=_pseudobulk_chunks(reference.X, query_order, neighbor_indices, batch_size),
    )

    logger.info("Wrote pseudobulked ATAC (%d cells x %d peaks) to %s", len(query_order), n_peaks, output_path)


def standardize_rna_cell_type(
    rna_path: str | Path,
    output_path: str | Path,
    cell_type_col: str,
) -> None:
    """Copy a raw paired snRNA-seq AnnData, aliasing `cell_type_col` to `obs["cell_type"]`.

    `PairedMultiomeData`/`load_paired_data` require a literal `obs["cell_type"]` column;
    raw RNA files here instead carry the cell-type annotation under a differently-named
    column (e.g. the same `Class_Jan15_2024` used for the ATAC KNN grouping). This makes
    the RNA file pipeline-ready without renaming the original column in place.
    """
    rna = ad.read_h5ad(rna_path)
    rna.obs["cell_type"] = rna.obs[cell_type_col].astype(str)
    rna.write_h5ad(output_path)
    logger.info("Wrote RNA with standardized cell_type column (%d cells) to %s", rna.n_obs, output_path)


def stratified_downsample_paired(
    rna_path: str | Path,
    atac_path: str | Path,
    output_rna_path: str | Path,
    output_atac_path: str | Path,
    cell_type_col: str,
    total_budget: int = 10_000,
    min_per_type: int = 10,
    seed: int = 42,
) -> None:
    """Downsample a paired RNA/ATAC cohort to `total_budget` cells, for fast pipeline testing
    on real (rather than synthetic) data before committing to a full-scale run.

    Every cell type gets at least `min_per_type` cells (if available); the remaining budget is
    allocated proportional to sqrt(cell-type count), so rare types aren't drowned out by common
    ones. RNA and ATAC are subset to the same sampled cells, assuming they already share
    `obs_names` (unlike `run_pseudobulking`, this does not do any KNN smoothing or reference
    atlas lookup -- it is purely for cheaply exercising the rest of the pipeline).
    """
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)

    counts = atac.obs[cell_type_col].value_counts()
    tier1 = {cell_type: min(n, min_per_type) for cell_type, n in counts.items()}
    remaining_budget = total_budget - sum(tier1.values())

    sqrt_counts = np.sqrt(counts.to_numpy())
    sqrt_proportions = sqrt_counts / sqrt_counts.sum()

    allocation = dict(tier1)
    for i, cell_type in enumerate(counts.index):
        additional = int(remaining_budget * sqrt_proportions[i])
        allocation[cell_type] += min(additional, counts[cell_type] - tier1[cell_type])

    rng = np.random.default_rng(seed)
    sampled_cells: list[str] = []
    for cell_type, n_sample in allocation.items():
        pool = atac.obs_names[atac.obs[cell_type_col] == cell_type].tolist()
        sampled_cells.extend(rng.choice(pool, size=n_sample, replace=False).tolist())

    atac[sampled_cells].copy().write_h5ad(output_atac_path)
    rna[sampled_cells].copy().write_h5ad(output_rna_path)
    logger.info("Wrote stratified sample of %d cells to %s / %s", len(sampled_cells), output_rna_path, output_atac_path)
