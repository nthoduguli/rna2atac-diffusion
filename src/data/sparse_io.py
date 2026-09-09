"""Streaming writer for large sparse `.h5ad` `X` matrices.

Shared by `pseudobulk.py` and `peak_filter.py`: both need to write cell-by-peak
matrices too large to hold fully in memory, one row-chunk at a time.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, cast

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from anndata.io import sparse_dataset, write_elem
except ImportError:  # anndata < 0.11
    from anndata.experimental import sparse_dataset, write_elem


class _AppendableSparseDataset(Protocol):
    """Backed sparse dataset supporting incremental writes.

    `anndata`'s public `CSRDataset`/`CSCDataset` stubs omit `.append()` even though the
    concrete objects `sparse_dataset()` returns implement it; `cast` to this Protocol below
    to document the real interface without fighting the incomplete upstream stub.
    """

    def append(self, sparse_matrix: sp.spmatrix) -> None: ...


def write_streamed_sparse_h5ad(
    output_path: str | Path,
    obs: pd.DataFrame,
    var: pd.DataFrame,
    row_chunks: Iterable[sp.csr_matrix],
    obsm: dict[str, np.ndarray] | None = None,
) -> None:
    """Write `obs`/`var` plus a CSR matrix streamed in row-chunks to a valid `.h5ad`.

    Initializes the on-disk `X` with a 64 bit indptr/indices from the start: scipy sizes
    a freshly constructed *empty* matrix's indptr/indices dtype for its current (zero)
    size, defaulting to int32, but large ATAC matrices commonly exceed int32's ~2.1
    billion cumulative-nnz limit once enough chunks are appended -- anndata's incremental
    writer then raises OverflowError partway through, having already written most of a
    very large file. Appending each `row_chunks` block keeps peak memory bounded to one
    chunk at a time regardless of total matrix size.

    `obsm`, if given, is written as-is (not streamed): per-cell embeddings like a
    precomputed LSI are unaffected by which columns of `X` end up in `row_chunks`
    (e.g. after peak filtering), so callers that need `.obsm` downstream (like
    `pseudobulk.compute_knn_neighbors`'s `embedding_key` lookup) must pass it through
    explicitly here -- it is not carried over automatically.
    """
    n_peaks = var.shape[0]
    empty_x = sp.csr_matrix((0, n_peaks), dtype="float32")
    empty_x.indptr = empty_x.indptr.astype(np.int64)
    empty_x.indices = empty_x.indices.astype(np.int64)

    with h5py.File(output_path, "w") as f:
        write_elem(f, "X", empty_x)
        write_elem(f, "obs", obs)
        write_elem(f, "var", var)
        if obsm:
            write_elem(f, "obsm", dict(obsm))
        x_dataset = cast(_AppendableSparseDataset, sparse_dataset(f["X"]))
        for chunk in row_chunks:
            x_dataset.append(chunk)
