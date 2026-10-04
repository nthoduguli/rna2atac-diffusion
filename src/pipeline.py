"""Train / eval / inference pipelines for snRNA-seq-conditioned snATAC-seq diffusion generation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from src.config import Config
from src.data import PairedMultiomeData, load_paired_data, split_by_donor
from src.metrics import evaluate_all
from src.models import CellTypeMeanBaseline, ConditionalDDPM, MLPBaseline, PeakVIEmbedder, ScVIEmbedder
from src.visualization import plot_loss_curve, plot_metric_comparison, plot_pred_vs_true, plot_results_summary

logger = logging.getLogger(__name__)


def _seed_everything(seed: int) -> None:
    import scvi

    np.random.seed(seed)
    torch.manual_seed(seed)
    scvi.settings.seed = seed


def _load_and_split(config: Config) -> tuple[PairedMultiomeData, dict[str, np.ndarray]]:
    data = load_paired_data(config.data.rna_path, config.data.atac_path)
    splits = split_by_donor(
        data,
        donor_col=config.data.donor_col,
        held_out_cell_types=config.data.held_out_cell_types,
        n_val_donors=config.data.n_val_donors,
        n_test_donors=config.data.n_test_donors,
        condition_col=config.data.condition_col,
        seed=config.data.seed,
    )
    return data, splits


def _check_batch_keys_cover_splits(config: Config, data: PairedMultiomeData, splits: dict[str, np.ndarray]) -> None:
    """Fail fast if any eval split has a batch category that training never sees.

    scVI/peakVI can't encode cells from an unseen batch category, so with a donor split a
    per-donor batch key (e.g. a sequencing-library ID) would only fail at eval time,
    after a full training run.
    """
    for adata, batch_key in ((data.rna, config.scvi.batch_key), (data.atac, config.peakvi.batch_key)):
        if batch_key is None:
            continue
        values = adata.obs[batch_key].astype(str).to_numpy()
        train_values = set(values[splits["train"]])
        for name, idx in splits.items():
            unseen = sorted(set(values[idx]) - train_values)
            if unseen:
                raise ValueError(  # noqa: TRY003
                    f"batch_key {batch_key!r} has categories in split {name!r} that are absent from train "
                    f"(e.g. {unseen[:3]}); use a batch key shared across donors, such as brain region."
                )


def _shuffle_within_cell_type(cond: np.ndarray, cell_types: np.ndarray, seed: int) -> np.ndarray:
    """Give each cell the conditioning embedding of a random other cell of the same type.

    Keeps cell-type identity in the conditioning but destroys any finer cell-state signal:
    a model that really uses RNA state should lose its within-cell-type performance.
    """
    rng = np.random.default_rng(seed)
    shuffled = cond.copy()
    for cell_type in np.unique(cell_types):
        idx = np.where(cell_types == cell_type)[0]
        shuffled[idx] = cond[rng.permutation(idx)]
    return shuffled


def train(config: Config) -> None:
    """Train scVI, peakVI, the conditional diffusion model, and baselines.

    Steps:
        1. Load paired snRNA-seq/snATAC-seq data and split by donor (plus a held-out cell type).
        2. Fit scVI on snRNA-seq to obtain cell-state conditioning embeddings.
        3. Fit peakVI on snATAC-seq to obtain the continuous diffusion target embeddings.
        4. Train the cell-type-mean and 2-layer-MLP baselines.
        5. Train the conditional DDPM denoiser on (cond, target) embedding pairs.
    """
    _seed_everything(config.data.seed)
    data, splits = _load_and_split(config)
    _check_batch_keys_cover_splits(config, data, splits)
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

    logger.info("Fitting cell-type-mean baseline...")
    train_cell_types = data.rna.obs["cell_type"].to_numpy()[train_idx]
    CellTypeMeanBaseline().fit(train_cell_types, target).save(checkpoints.mean_baseline_path)

    logger.info("Training 2-layer MLP baseline...")
    mlp = MLPBaseline(cond_dim=cond.shape[1], target_dim=target.shape[1], hidden_dim=config.baseline.mlp_hidden_dim)
    mlp_losses = _train_mlp(
        mlp, cond, target, max_epochs=config.baseline.mlp_max_epochs, lr=config.baseline.mlp_learning_rate
    )
    torch.save(mlp.state_dict(), checkpoints.mlp_baseline_path)
    plot_loss_curve(mlp_losses, "MLP baseline training loss", figures_dir / "mlp_baseline_loss.png")

    logger.info("Training conditional diffusion model...")
    # Clip the sampler's reparametrized x0 estimate to a few standard deviations of the
    # real training target's scale, rather than a fixed constant -- keeps the bound
    # meaningful regardless of the embedding dimensionality/scale in use.
    x0_clip = float(target.std() * 5)
    diffusion = ConditionalDDPM(
        target_dim=target.shape[1],
        cond_dim=cond.shape[1],
        n_timesteps=config.diffusion.n_timesteps,
        x0_clip=x0_clip,
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

    Besides each model, scores a shuffled-conditioning control for the MLP and diffusion
    model (conditioning shuffled within cell type, see `_shuffle_within_cell_type`), and a
    single-sample diffusion prediction alongside the default mean of
    `config.diffusion.n_eval_samples` samples. Saves all predictions to
    `<figures.output_dir>/predictions_<split>.npz`.

    Args:
        config: Pipeline configuration with paths to data and trained checkpoints.
        split: One of "val", "test", or "held_out".

    Returns:
        A dict mapping model name -> dict of metric name -> value.
    """
    _seed_everything(config.data.seed)
    data, splits = _load_and_split(config)
    idx = splits[split]
    checkpoints = config.checkpoints
    figures_dir = Path(config.figures.output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    scvi_embedder = ScVIEmbedder.load(checkpoints.scvi_dir, adata=data.rna[idx])
    peakvi_embedder = PeakVIEmbedder.load(checkpoints.peakvi_dir, adata=data.atac[idx])

    cond = scvi_embedder.embed()
    target = peakvi_embedder.embed()
    obs = data.rna.obs.iloc[idx]
    cell_types = obs["cell_type"].to_numpy().astype(str)
    shuffled_cond = _shuffle_within_cell_type(cond, cell_types, seed=config.data.seed)

    predictions: dict[str, np.ndarray] = {}
    predictions["celltype_mean"] = CellTypeMeanBaseline.load(checkpoints.mean_baseline_path).predict(cell_types)

    mlp = MLPBaseline(cond_dim=cond.shape[1], target_dim=target.shape[1], hidden_dim=config.baseline.mlp_hidden_dim)
    mlp.load_state_dict(torch.load(checkpoints.mlp_baseline_path))
    mlp.eval()
    with torch.no_grad():
        predictions["mlp"] = mlp(torch.as_tensor(cond, dtype=torch.float32)).numpy()
        predictions["mlp_shuffled_cond"] = mlp(torch.as_tensor(shuffled_cond, dtype=torch.float32)).numpy()

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

    def sample_mean(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c_t = torch.as_tensor(c, dtype=torch.float32)
        samples = np.stack([
            diffusion.sample(c_t, target_dim=target.shape[1]).numpy() for _ in range(config.diffusion.n_eval_samples)
        ])
        return samples.mean(axis=0), samples[0]

    predictions["diffusion"], predictions["diffusion_1sample"] = sample_mean(cond)
    predictions["diffusion_shuffled_cond"], _ = sample_mean(shuffled_cond)

    donors = obs[config.data.donor_col].to_numpy()
    results = {name: evaluate_all(pred, target, cell_types, donors) for name, pred in predictions.items()}
    for name, metrics in results.items():
        logger.info("[%s] %s: %s", split, name, metrics)

    np.savez(
        figures_dir / f"predictions_{split}.npz",
        target=target,
        cell_type=cell_types,
        donor=donors,
        **predictions,
    )
    for name in ("celltype_mean", "mlp", "diffusion"):
        plot_pred_vs_true(
            predictions[name],
            target,
            f"{name} ({split}): predicted vs. true",
            figures_dir / f"{name}_{split}_pred_vs_true.png",
        )
    correlation_results = {
        name: {metric: results[name][metric] for metric in ("pearson_r", "within_cell_type_r")} for name in results
    }
    plot_metric_comparison(
        correlation_results, f"Model comparison ({split})", figures_dir / f"metric_comparison_{split}.png"
    )

    return results


# Rows and panels of the summary figure / table (see `summarize_results`).
SUMMARY_ROWS = [
    ("celltype_mean", "Cell-type mean", True),
    ("mlp", "MLP", False),
    ("diffusion", "Diffusion (mean of samples)", False),
    ("diffusion_1sample", "Diffusion (1 sample)", False),
    ("mlp_shuffled_cond", "MLP, shuffled RNA", True),
    ("diffusion_shuffled_cond", "Diffusion, shuffled RNA", True),
]
SUMMARY_PANELS = [
    ("pearson_r", "Per-cell r"),
    ("within_cell_type_r", "Within cell type r"),
    ("within_cell_type_donor_r", "Within cell type + donor r"),
]


def summarize_results(config: Config, split: str = "test") -> str:
    """Rebuild the summary figure and a markdown results table from saved predictions.

    Reads `<figures.output_dir>/predictions_<split>.npz` (written by `evaluate`), recomputes
    metrics with the same `evaluate_all` used by `evaluate`, and writes
    `results_<split>.png` and `results_<split>.md` next to it. Does not load any models.

    Returns:
        The markdown table.
    """
    figures_dir = Path(config.figures.output_dir)
    with np.load(figures_dir / f"predictions_{split}.npz", allow_pickle=True) as saved:
        target, cell_types, donors = saved["target"], saved["cell_type"], saved["donor"]
        results = {
            name: evaluate_all(saved[name], target, cell_types, donors) for name, _, _ in SUMMARY_ROWS if name in saved
        }
    rows = [row for row in SUMMARY_ROWS if row[0] in results]

    n_donors = len(np.unique(donors))
    plot_results_summary(
        results,
        rows,
        SUMMARY_PANELS,
        f"RNA \u2192 ATAC (peakVI latent), {split} split: {len(target):,} cells, {n_donors} donors",
        figures_dir / f"results_{split}.png",
    )

    header = "| Model | " + " | ".join(title for _, title in SUMMARY_PANELS) + " | R\u00b2 |"
    lines = [header, "|" + "---|" * (len(SUMMARY_PANELS) + 2)]
    for name, label, _ in rows:
        values = [results[name][metric] for metric, _ in SUMMARY_PANELS] + [results[name]["r2"]]
        lines.append(f"| {label} | " + " | ".join(f"{0.0 if abs(v) < 0.005 else v:.2f}" for v in values) + " |")
    table = "\n".join(lines) + "\n"
    (figures_dir / f"results_{split}.md").write_text(table)
    return table


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
