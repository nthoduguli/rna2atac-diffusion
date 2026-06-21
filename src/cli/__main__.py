import logging
from pathlib import Path

import numpy as np
import typer

from src.config import conf, load_config
from src.pipeline import evaluate, run_inference, train

logger = logging.getLogger(__name__)
cli = typer.Typer(help="snRNA-seq-conditioned snATAC-seq diffusion generation CLI.")


@cli.callback()
def callback() -> None:
    """Inject the logging configuration into the logging system."""
    import logging.config

    from src.config import conf

    logging.config.dictConfig(conf.logging.model_dump())


@cli.command("train")
def train_cmd(
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Train scVI, peakVI, the conditional diffusion model, and baselines."""
    config = load_config(config_path) if config_path is not None else conf
    train(config)


@cli.command("eval")
def eval_cmd(
    split: str = typer.Option("test", help="Split to evaluate: 'val', 'test', or 'held_out'."),
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Compute Pearson r, R^2, and weighted MSE for the diffusion model and baselines."""
    config = load_config(config_path) if config_path is not None else conf
    results = evaluate(config, split=split)
    for model_name, metrics in results.items():
        typer.echo(f"{model_name}: {metrics}")


@cli.command("inference")
def inference_cmd(
    rna_path: Path = typer.Argument(..., help="Path to an .h5ad file with snRNA-seq input to condition on."),  # noqa: B008
    output_path: Path = typer.Option("generated_atac.npy", "--output", help="Path to save generated ATAC tracks."),  # noqa: B008
    config_path: Path = typer.Option(None, "--config", help="Path to a YAML config file (overrides defaults)."),  # noqa: B008
) -> None:
    """Generate ATAC accessibility tracks for new snRNA-seq input using trained checkpoints."""
    config = load_config(config_path) if config_path is not None else conf
    generated = run_inference(config, rna_path)
    np.save(output_path, generated)
    typer.echo(f"Saved generated ATAC tracks of shape {generated.shape} to {output_path}")
