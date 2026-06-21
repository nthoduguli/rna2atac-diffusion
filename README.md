# rna2atac-diffusion

snRNA-seq-conditioned generation of snATAC-seq chromatin accessibility profiles in Alzheimer's
Disease, using a diffusion model conditioned on cell-state embeddings.

## Overview

Given a cell's snRNA-seq profile, this project generates a plausible corresponding snATAC-seq
chromatin accessibility profile via a conditional denoising diffusion probabilistic model (DDPM).

- **Cell-state embeddings**: [scVI](https://docs.scvi-tools.org/) is trained on snRNA-seq to
  produce the conditioning cell-state latent.
- **ATAC representation / diffusion target**: [peakVI](https://docs.scvi-tools.org/) is trained
  on snATAC-seq to embed chromatin accessibility into a continuous latent space. This latent is
  the diffusion target. The peakVI decoder maps generated latents back to per-peak accessibility
  scores for evaluation. (Open design question: if peakVI-embedding diffusion underperforms,
  this may be revisited in favor of diffusion directly over raw/binned accessibility tracks —
  hence the repo name is agnostic to this choice.)
- **Diffusion model**: a conditional DDPM denoiser, conditioned on the scVI cell-state embedding
  and a timestep embedding, predicts the standard Gaussian noise added to the peakVI-embedding.
  Gaussian noise is appropriate here because the diffusion target is continuous (unlike raw ATAC
  fragment counts, for which a negative-binomial noise model would be more natural).

## Dataset

[Morabito et al. 2021 (Nature Genetics)](https://www.nature.com/articles/s41588-021-00894-z),
[GEO GSE174367](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE174367) — paired snRNA-seq
and snATAC-seq from prefrontal cortex of AD and control donors. Fully open access, no DUA
required.

## Train / val / test / held-out splitting

Following the held-out-cell-type strategy from prior work, all cells of one or more cell types
(configurable via `data.held_out_cell_types`, adapted to GSE174367's cell-type annotations) are
withheld entirely from training and validation, and evaluated separately as the `held_out` split.
This avoids leakage from KNN-pseudobulking across train/test cells of the same type. The
remaining cells are randomly partitioned into `train`, `val`, and `test`.

## Baselines

- **Mean baseline**: always predicts the training-set mean ATAC-representation embedding.
- **MLP baseline**: a simple 2-layer MLP mapping the scVI cell-state embedding to the
  peakVI-embedding.

## CLI

```bash
uv run cli train [--config path/to/config.yaml]
uv run cli eval --split {val,test,held_out} [--config path/to/config.yaml]
uv run cli inference path/to/rna.h5ad --output generated_atac.npy [--config path/to/config.yaml]
```

- `train` fits scVI, peakVI, the conditional diffusion model, and both baselines, saving
  checkpoints to the paths in `checkpoints` (see `conf/default.config.yaml`).
- `eval` computes Pearson r, R², and weighted MSE for the diffusion model and baselines on the
  requested split.
- `inference` generates an ATAC accessibility profile for new snRNA-seq input using trained
  checkpoints.

## Related work

- **scButterfly / JAMIE** (dual-VAE cross-modal translation): regress one modality from another
  under an MSE objective. Because the true RNA→ATAC mapping is one-to-many and stochastic
  (multiple chromatin states are consistent with a given transcriptional state), MSE-based
  regression mode-averages and collapses predictions to point estimates, losing the diversity of
  plausible accessibility profiles.
- **ATAC-Diff** (Huang et al., 2024): a GMM-latent diffusion model for ATAC generation, but not
  conditioned on RNA — it generates ATAC profiles unconditionally/within-modality rather than
  translating across modalities.
- **scDiffusion-X** (2026): the closest existing analog, combining multi-omics diffusion with
  cross-attention for cross-modal translation. This project differentiates itself with a
  focused, AD-specific approach operating in peakVI-embedding space, evaluated with an explicit
  held-out-cell-type generalization split.

Diffusion's key advantage over dual-VAE regression approaches is its suitability for one-to-many,
stochastic cross-modal mappings: rather than regressing to a single point estimate (and
collapsing toward the conditional mean, as observed in prior pilot work), a conditional DDPM can
sample diverse, plausible ATAC profiles consistent with a given RNA cell state.
