import logging
from pathlib import Path

import numpy as np
import typer

from src.config import conf, load_config
from src.data import (
    filter_peaks_by_prevalence,
    run_pseudobulking,
    standardize_rna_cell_type,
    stratified_downsample_paired,
)

logger = logging.getLogger(__name__)
cli = typer.Typer(help="snRNA-seq-conditioned snATAC-seq diffusion generation CLI.")


@cli.callback()
def callback() -> None:
    """Inject the logging configuration into the logging system."""
    import logging.config

    from src.config import conf

    logging.config.dictConfig(conf.logging.model_dump())


@cli.command("filter-peaks")
def filter_peaks_cmd(
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Filter the reference ATAC atlas to peaks accessible in a minimum fraction of cells (run before `pseudobulk`)."""
    config = load_config(config_path) if config_path is not None else conf
    pf = config.peak_filter
    filter_peaks_by_prevalence(
        reference_atac_path=pf.reference_atac_path,
        cell_type_col=pf.cell_type_col,
        output_path=pf.output_path,
        min_cell_fraction=pf.min_cell_fraction,
        batch_size=pf.batch_size,
    )
    typer.echo(f"Wrote peak-filtered reference atlas to {pf.output_path}")


@cli.command("pseudobulk")
def pseudobulk_cmd(
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """KNN-pseudobulk the raw paired ATAC/RNA data into pipeline-ready inputs (run before `train`)."""
    config = load_config(config_path) if config_path is not None else conf
    pb = config.pseudobulk
    run_pseudobulking(
        query_atac_path=pb.query_atac_path,
        reference_atac_path=pb.reference_atac_path,
        output_path=pb.output_path,
        cell_type_col=pb.cell_type_col,
        embedding_key=pb.embedding_key,
        k=pb.k,
        batch_size=pb.batch_size,
        donor_col=pb.donor_col,
    )
    typer.echo(f"Wrote pseudobulked ATAC to {pb.output_path}")

    standardize_rna_cell_type(pb.query_rna_path, pb.rna_output_path, pb.cell_type_col)
    typer.echo(f"Wrote pipeline-ready RNA to {pb.rna_output_path}")


@cli.command("sample")
def sample_cmd(
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Draw a small stratified sample of the raw paired cohort, for fast pipeline testing on real data."""
    config = load_config(config_path) if config_path is not None else conf
    pb, sample = config.pseudobulk, config.sample
    stratified_downsample_paired(
        rna_path=pb.query_rna_path,
        atac_path=pb.query_atac_path,
        output_rna_path=sample.output_rna_path,
        output_atac_path=sample.output_atac_path,
        cell_type_col=pb.cell_type_col,
        total_budget=sample.total_budget,
        min_per_type=sample.min_per_type,
        seed=sample.seed,
    )
    typer.echo(f"Wrote stratified sample to {sample.output_rna_path} / {sample.output_atac_path}")


@cli.command("train")
def train_cmd(
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Train scVI, peakVI, the conditional diffusion model, and baselines."""
    from src.pipeline import train

    config = load_config(config_path) if config_path is not None else conf
    train(config)


@cli.command("eval")
def eval_cmd(
    split: str = typer.Option("test", help="Split to evaluate: 'val', 'test', or 'held_out'."),
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Compute Pearson r, R^2, and weighted MSE for the diffusion model and baselines."""
    from src.pipeline import evaluate

    config = load_config(config_path) if config_path is not None else conf
    results = evaluate(config, split=split)
    for model_name, metrics in results.items():
        typer.echo(f"{model_name}: {metrics}")


@cli.command("results")
def results_cmd(
    split: str = typer.Option("test", help="Split whose saved predictions to summarize."),
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Rebuild the results figure and markdown table from predictions saved by `eval`."""
    from src.pipeline import summarize_results

    config = load_config(config_path) if config_path is not None else conf
    typer.echo(summarize_results(config, split=split))


@cli.command("inference")
def inference_cmd(
    rna_path: Path = typer.Argument(..., help="Path to an .h5ad file with snRNA-seq input to condition on."),  # noqa: B008
    output_path: Path = typer.Option("generated_atac.npy", "--output", help="Path to save generated ATAC tracks."),  # noqa: B008
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Generate ATAC accessibility tracks for new snRNA-seq input using trained checkpoints."""
    from src.pipeline import run_inference

    config = load_config(config_path) if config_path is not None else conf
    generated = run_inference(config, rna_path)
    np.save(output_path, generated)
    typer.echo(f"Saved generated ATAC tracks of shape {generated.shape} to {output_path}")
