"""Train / eval / inference pipelines for snRNA-seq-conditioned snATAC-seq diffusion generation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.data import load_paired_data, split_by_cell_type
from src.metrics import evaluate_all
from src.models import ConditionalDDPM, MeanBaseline, MLPBaseline, PeakVIEmbedder, ScVIEmbedder
from src.visualization import plot_loss_curve, plot_metric_comparison, plot_pred_vs_true

logger = logging.getLogger(__name__)


def train(config: Config) -> None:
    """Train scVI, peakVI, the conditional diffusion model, and baselines.

    Steps:
        1. Load paired snRNA-seq/snATAC-seq data and split by (held-out) cell type.
        2. Fit scVI on snRNA-seq to obtain cell-state conditioning embeddings.
        3. Fit peakVI on snATAC-seq to obtain the continuous diffusion target embeddings.
        4. Train the always-predict-the-mean and 2-layer-MLP baselines.
        5. Train the conditional DDPM denoiser on (cond, target) embedding pairs.
    """
    data = load_paired_data(config.data.rna_path, config.data.atac_path)
    splits = split_by_cell_type(
        data,
        held_out_cell_types=config.data.held_out_cell_types,
        val_fraction=config.data.val_fraction,
        seed=config.data.seed,
    )
    train_idx = splits["train"]

    checkpoints = config.checkpoints
    figures_dir = Path(config.figures.output_dir)
    Path(checkpoints.output_dir).mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Training scVI on snRNA-seq (cell-state embeddings)...")
    scvi_embedder = ScVIEmbedder(n_latent=config.scvi.n_latent)
    scvi_embedder.fit(data.rna[train_idx], max_epochs=config.scvi.max_epochs, batch_key=config.scvi.batch_key)
    scvi_embedder.save(checkpoints.scvi_dir)

    logger.info("Training peakVI on snATAC-seq (diffusion target embeddings)...")
    peakvi_embedder = PeakVIEmbedder(n_latent=config.peakvi.n_latent)
    peakvi_embedder.fit(data.atac[train_idx], max_epochs=config.peakvi.max_epochs, batch_key=config.peakvi.batch_key)
    peakvi_embedder.save(checkpoints.peakvi_dir)

    cond = scvi_embedder.embed(data.rna[train_idx])
    target = peakvi_embedder.embed(data.atac[train_idx])

    logger.info("Fitting always-predict-the-mean baseline...")
    mean_baseline = MeanBaseline().fit(cond, target)
    np.save(checkpoints.mean_baseline_path, mean_baseline.mean_)

    logger.info("Training 2-layer MLP baseline...")
    mlp = MLPBaseline(cond_dim=cond.shape[1], target_dim=target.shape[1], hidden_dim=config.baseline.mlp_hidden_dim)
    mlp_losses = _train_mlp(
        mlp, cond, target, max_epochs=config.baseline.mlp_max_epochs, lr=config.baseline.mlp_learning_rate
    )
    torch.save(mlp.state_dict(), checkpoints.mlp_baseline_path)
    plot_loss_curve(mlp_losses, "MLP baseline training loss", figures_dir / "mlp_baseline_loss.png")

    logger.info("Training conditional diffusion model...")
    diffusion = ConditionalDDPM(
        target_dim=target.shape[1],
        cond_dim=cond.shape[1],
        n_timesteps=config.diffusion.n_timesteps,
        hidden_dim=config.diffusion.hidden_dim,
        n_layers=config.diffusion.n_layers,
        time_dim=config.diffusion.time_dim,
    )
    diffusion_losses = _train_diffusion(
        diffusion,
        cond,
        target,
        max_epochs=config.diffusion.max_epochs,
        batch_size=config.hyperparameters.batch_size,
        lr=config.diffusion.learning_rate,
    )
    torch.save(diffusion.state_dict(), checkpoints.diffusion_path)
    plot_loss_curve(diffusion_losses, "Diffusion training loss", figures_dir / "diffusion_loss.png")

    logger.info("Training complete. Checkpoints saved to %s", checkpoints.output_dir)


def _train_mlp(mlp: MLPBaseline, cond: np.ndarray, target: np.ndarray, max_epochs: int, lr: float) -> list[float]:
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    cond_t = torch.as_tensor(cond, dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.float32)
    losses = []
    for epoch in range(max_epochs):
        optimizer.zero_grad()
        loss = mlp.loss(cond_t, target_t)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if epoch % max(1, max_epochs // 10) == 0:
            logger.info("MLP baseline epoch %d/%d - loss %.4f", epoch, max_epochs, loss.item())
    return losses


def _train_diffusion(
    diffusion: ConditionalDDPM, cond: np.ndarray, target: np.ndarray, max_epochs: int, batch_size: int, lr: float
) -> list[float]:
    optimizer = torch.optim.Adam(diffusion.parameters(), lr=lr)
    cond_t = torch.as_tensor(cond, dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.float32)
    n = cond_t.shape[0]

    losses = []
    for epoch in range(max_epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = diffusion.loss(target_t[idx], cond_t[idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)
        epoch_loss = total_loss / n
        losses.append(epoch_loss)
        if epoch % max(1, max_epochs // 10) == 0:
            logger.info("Diffusion epoch %d/%d - loss %.4f", epoch, max_epochs, epoch_loss)
    return losses


def evaluate(config: Config, split: str = "test") -> dict[str, dict[str, float]]:
    """Evaluate the diffusion model and baselines on the given split.

    Args:
        config: Pipeline configuration with paths to data and trained checkpoints.
        split: One of "val", "test", or "held_out".

    Returns:
        A dict mapping model name -> dict of metric name -> value.
    """
    data = load_paired_data(config.data.rna_path, config.data.atac_path)
    splits = split_by_cell_type(
        data,
        held_out_cell_types=config.data.held_out_cell_types,
        val_fraction=config.data.val_fraction,
        seed=config.data.seed,
    )
    idx = splits[split]
    checkpoints = config.checkpoints
    figures_dir = Path(config.figures.output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scvi_embedder = ScVIEmbedder.load(checkpoints.scvi_dir, adata=data.rna[idx])
    peakvi_embedder = PeakVIEmbedder.load(checkpoints.peakvi_dir, adata=data.atac[idx])

    cond = scvi_embedder.embed()
    target = peakvi_embedder.embed()
    cond_t = torch.as_tensor(cond, dtype=torch.float32)

    results: dict[str, dict[str, float]] = {}

    mean_baseline = MeanBaseline()
    mean_baseline.mean_ = np.load(checkpoints.mean_baseline_path)
    mean_pred = mean_baseline.predict(cond)
    results["mean_baseline"] = evaluate_all(mean_pred, target)

    mlp = MLPBaseline(cond_dim=cond.shape[1], target_dim=target.shape[1], hidden_dim=config.baseline.mlp_hidden_dim)
    mlp.load_state_dict(torch.load(checkpoints.mlp_baseline_path))
    mlp.eval()
    with torch.no_grad():
        mlp_pred = mlp(cond_t).numpy()
    results["mlp_baseline"] = evaluate_all(mlp_pred, target)

    diffusion = ConditionalDDPM(
        target_dim=target.shape[1],
        cond_dim=cond.shape[1],
        n_timesteps=config.diffusion.n_timesteps,
        hidden_dim=config.diffusion.hidden_dim,
        n_layers=config.diffusion.n_layers,
        time_dim=config.diffusion.time_dim,
    )
    diffusion.load_state_dict(torch.load(checkpoints.diffusion_path))
    diffusion.eval()
    diffusion_pred = diffusion.sample(cond_t, target_dim=target.shape[1]).numpy()
    results["diffusion"] = evaluate_all(diffusion_pred, target)

    for name, metrics in results.items():
        logger.info("[%s] %s: %s", split, name, metrics)

    predictions = {"mean_baseline": mean_pred, "mlp_baseline": mlp_pred, "diffusion": diffusion_pred}
    for name, pred in predictions.items():
        plot_pred_vs_true(
            pred, target, f"{name} ({split}): predicted vs. true", figures_dir / f"{name}_{split}_pred_vs_true.png"
        )
    plot_metric_comparison(results, f"Model comparison ({split})", figures_dir / f"metric_comparison_{split}.png")

    return results


def run_inference(config: Config, rna_path: str | Path) -> np.ndarray:
    """Generate ATAC accessibility tracks for new snRNA-seq input.

    Args:
        config: Pipeline configuration with paths to trained checkpoints.
        rna_path: Path to an `.h5ad` file with snRNA-seq counts to condition on.

    Returns:
        Array of shape (n_cells, n_peaks) with generated accessibility scores.
    """
    import anndata as ad

    rna = ad.read_h5ad(rna_path)
    checkpoints = config.checkpoints

    scvi_embedder = ScVIEmbedder.load(checkpoints.scvi_dir, adata=rna)
    cond = scvi_embedder.embed()
    cond_t = torch.as_tensor(cond, dtype=torch.float32)

    target_dim = config.peakvi.n_latent
    diffusion = ConditionalDDPM(
        target_dim=target_dim,
        cond_dim=cond.shape[1],
        n_timesteps=config.diffusion.n_timesteps,
        hidden_dim=config.diffusion.hidden_dim,
        n_layers=config.diffusion.n_layers,
        time_dim=config.diffusion.time_dim,
    )
    diffusion.load_state_dict(torch.load(checkpoints.diffusion_path))
    diffusion.eval()

    generated_latents = diffusion.sample(cond_t, target_dim=target_dim).numpy()

    peakvi_embedder = PeakVIEmbedder()
    # peakVI requires an ATAC AnnData to construct the model wrapper for decoding;
    # callers should pass a checkpoint trained with a compatible reference dataset.
    import anndata as ad_atac

    reference_atac = ad_atac.read_h5ad(config.data.atac_path)
    peakvi_embedder = PeakVIEmbedder.load(checkpoints.peakvi_dir, adata=reference_atac)

    return peakvi_embedder.decode(generated_latents)
