"""Cell-state and ATAC-representation embedders built on scvi-tools.

- `ScVIEmbedder` wraps scVI to produce the conditioning cell-state latent from snRNA-seq.
- `PeakVIEmbedder` wraps peakVI to produce the continuous diffusion target latent
  from snATAC-seq, and decode generated latents back to accessibility tracks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import anndata as ad
import numpy as np
import scvi

logger = logging.getLogger(__name__)


class ScVIEmbedder:
    """Thin wrapper around `scvi.model.SCVI` producing cell-state latent embeddings."""

    def __init__(self, n_latent: int = 30, **model_kwargs) -> None:
        self.n_latent = n_latent
        self.model_kwargs = model_kwargs
        self.model: scvi.model.SCVI | None = None

    def fit(self, adata: ad.AnnData, max_epochs: int = 400, batch_key: str | None = "batch") -> None:
        adata = adata.copy()
        scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
        self.model = scvi.model.SCVI(adata, n_latent=self.n_latent, **self.model_kwargs)
        self.model.train(max_epochs=max_epochs)
        self._train_adata = adata

    def embed(self, adata: ad.AnnData | None = None) -> np.ndarray:
        """Return the scVI latent representation (cell-state embedding)."""
        if self.model is None:
            raise RuntimeError("ScVIEmbedder.fit must be called before embed.")  # noqa: TRY003
        target = adata if adata is not None else self._train_adata
        return self.model.get_latent_representation(target)

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("ScVIEmbedder.fit must be called before save.")  # noqa: TRY003
        self.model.save(str(path), overwrite=True)

    @classmethod
    def load(cls, path: str | Path, adata: ad.AnnData) -> ScVIEmbedder:
        adata = adata.copy()
        embedder = cls()
        embedder.model = scvi.model.SCVI.load(str(path), adata=adata)
        embedder._train_adata = adata
        return embedder


class PeakVIEmbedder:
    """Thin wrapper around `scvi.model.PEAKVI` producing the ATAC diffusion target latent.

    The encoder output is used as the continuous diffusion target; the decoder
    maps generated latents back to interpretable per-peak accessibility scores.
    """

    def __init__(self, n_latent: int = 30, **model_kwargs) -> None:
        self.n_latent = n_latent
        self.model_kwargs = model_kwargs
        self.model: scvi.model.PEAKVI | None = None

    def fit(self, adata: ad.AnnData, max_epochs: int = 400, batch_key: str | None = "batch") -> None:
        adata = adata.copy()
        scvi.model.PEAKVI.setup_anndata(adata, batch_key=batch_key)
        self.model = scvi.model.PEAKVI(adata, n_latent=self.n_latent, **self.model_kwargs)
        self.model.train(max_epochs=max_epochs)
        self._train_adata = adata

    def embed(self, adata: ad.AnnData | None = None) -> np.ndarray:
        """Return the peakVI latent representation (diffusion target)."""
        if self.model is None:
            raise RuntimeError("PeakVIEmbedder.fit must be called before embed.")  # noqa: TRY003
        target = adata if adata is not None else self._train_adata
        return self.model.get_latent_representation(target)

    def decode(self, latents: np.ndarray) -> np.ndarray:
        """Decode latent embeddings back to accessibility probabilities per peak.

        Implemented via the decoder of the underlying peakVI module, run directly
        on the supplied latent codes (bypassing the encoder).
        """
        if self.model is None:
            raise RuntimeError("PeakVIEmbedder.fit must be called before decode.")  # noqa: TRY003
        import torch

        module = self.model.module
        module.eval()
        z = torch.as_tensor(latents, dtype=torch.float32)
        # z_decoder expects one categorical-covariate index tensor per entry in
        # n_cat_list (e.g. batch); default to category 0 for all generated cells.
        cat_covs = [torch.zeros(z.shape[0], 1, dtype=torch.long) for _ in module.z_decoder.px_decoder.n_cat_list]
        with torch.no_grad():
            px = module.z_decoder(z, *cat_covs)
        return px.detach().cpu().numpy()

    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise RuntimeError("PeakVIEmbedder.fit must be called before save.")  # noqa: TRY003
        self.model.save(str(path), overwrite=True)

    @classmethod
    def load(cls, path: str | Path, adata: ad.AnnData) -> PeakVIEmbedder:
        adata = adata.copy()
        embedder = cls()
        embedder.model = scvi.model.PEAKVI.load(str(path), adata=adata)
        embedder._train_adata = adata
        return embedder
