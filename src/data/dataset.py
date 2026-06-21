"""Loading and splitting of paired snRNA-seq / snATAC-seq data (GSE174367, Morabito et al. 2021)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PairedMultiomeData:
    """Container for paired snRNA-seq and snATAC-seq AnnData objects.

    The two modalities are assumed to share an `obs` index of cell barcodes and
    a `cell_type` column used for the held-out-cell-type split.
    """

    rna: ad.AnnData
    atac: ad.AnnData

    def __post_init__(self) -> None:
        if self.rna.n_obs != self.atac.n_obs or not (self.rna.obs_names == self.atac.obs_names).all():
            shared = self.rna.obs_names.intersection(self.atac.obs_names)
            if len(shared) == 0:
                raise ValueError("RNA and ATAC AnnData objects share no cell barcodes.")  # noqa: TRY003
            logger.info("Restricting RNA/ATAC AnnData objects to %d shared cells.", len(shared))
            self.rna = self.rna[shared].copy()
            self.atac = self.atac[shared].copy()

        if "cell_type" not in self.rna.obs.columns:
            raise ValueError("RNA AnnData is missing the required `obs['cell_type']` annotation.")  # noqa: TRY003

    @property
    def cell_types(self) -> list[str]:
        return sorted(self.rna.obs["cell_type"].unique().tolist())


def load_paired_data(rna_path: str | Path, atac_path: str | Path) -> PairedMultiomeData:
    """Load paired snRNA-seq and snATAC-seq AnnData objects from disk.

    Args:
        rna_path: Path to an `.h5ad` file containing snRNA-seq counts with
            a `cell_type` column in `obs`.
        atac_path: Path to an `.h5ad` file containing snATAC-seq peak counts
            (or binned accessibility) sharing cell barcodes with `rna_path`.
    """
    rna = ad.read_h5ad(rna_path)
    atac = ad.read_h5ad(atac_path)
    return PairedMultiomeData(rna=rna, atac=atac)


def split_by_cell_type(
    data: PairedMultiomeData,
    held_out_cell_types: list[str],
    val_fraction: float = 0.1,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split cells into train/val/test/held-out partitions.

    Cells belonging to `held_out_cell_types` are entirely excluded from training
    and validation and placed in the `held_out` split, following the prior work's
    strategy of evaluating generalization to unseen cell states (and avoiding
    KNN-pseudobulking leakage between train and test cells of the same type).

    The remaining cells are randomly split into `train`, `val`, and `test`
    (test takes the same fraction as val).

    Returns:
        A dict mapping split name -> array of integer obs indices into `data`.
    """
    cell_types = data.rna.obs["cell_type"].to_numpy()
    held_out_mask = np.isin(cell_types, held_out_cell_types)

    rng = np.random.default_rng(seed)
    remaining_idx = np.where(~held_out_mask)[0]
    rng.shuffle(remaining_idx)

    n = len(remaining_idx)
    n_val = int(n * val_fraction)
    n_test = int(n * val_fraction)

    splits = {
        "val": remaining_idx[:n_val],
        "test": remaining_idx[n_val : n_val + n_test],
        "train": remaining_idx[n_val + n_test :],
        "held_out": np.where(held_out_mask)[0],
    }

    for name, idx in splits.items():
        logger.info("Split '%s' contains %d cells.", name, len(idx))

    return splits
