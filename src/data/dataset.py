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


def split_by_donor(
    data: PairedMultiomeData,
    donor_col: str,
    held_out_cell_types: list[str],
    n_val_donors: int,
    n_test_donors: int,
    condition_col: str | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Split cells into train/val/test/held-out partitions.

    Cells belonging to `held_out_cell_types` are entirely excluded from training
    and validation and placed in the `held_out` split, to evaluate generalization
    to an unseen cell type.

    The remaining cells are split by donor, so every cell (and brain region) of a
    donor lands in exactly one of `train`, `val`, `test`. Cells from the same donor
    share structure, so a random cell split would inflate test performance. If
    `condition_col` is given (e.g. AD vs. nonAD, constant per donor), donors are
    shuffled within each condition and interleaved before being dealt out, so
    val and test stay balanced across conditions.

    Returns:
        A dict mapping split name -> array of integer obs indices into `data`.
    """
    obs = data.rna.obs
    held_out_mask = np.isin(obs["cell_type"].to_numpy(), held_out_cell_types)
    donors = obs[donor_col].to_numpy()

    if condition_col is None:
        strata = [np.unique(donors[~held_out_mask])]
    else:
        donor_conditions = obs.loc[~held_out_mask, [donor_col, condition_col]].drop_duplicates()
        if donor_conditions[donor_col].duplicated().any():
            raise ValueError(f"`{condition_col}` is not constant within each `{donor_col}`.")  # noqa: TRY003
        strata = [
            np.sort(group[donor_col].to_numpy())
            for _, group in sorted(donor_conditions.groupby(condition_col, observed=True), key=lambda kv: str(kv[0]))
        ]

    rng = np.random.default_rng(seed)
    for stratum in strata:
        rng.shuffle(stratum)
    # Interleave strata: [c0_d0, c1_d0, c0_d1, c1_d1, ...], so any prefix is balanced.
    max_len = max(len(stratum) for stratum in strata)
    ordered = [stratum[i] for i in range(max_len) for stratum in strata if i < len(stratum)]

    if n_val_donors + n_test_donors >= len(ordered):
        raise ValueError(f"Need more than {n_val_donors + n_test_donors} donors, found {len(ordered)}.")  # noqa: TRY003
    test_donors = ordered[:n_test_donors]
    val_donors = ordered[n_test_donors : n_test_donors + n_val_donors]

    in_test = np.isin(donors, test_donors) & ~held_out_mask
    in_val = np.isin(donors, val_donors) & ~held_out_mask
    splits = {
        "val": np.where(in_val)[0],
        "test": np.where(in_test)[0],
        "train": np.where(~in_val & ~in_test & ~held_out_mask)[0],
        "held_out": np.where(held_out_mask)[0],
    }

    for name, idx in splits.items():
        logger.info("Split '%s' contains %d cells from %d donors.", name, len(idx), len(np.unique(donors[idx])))

    return splits
