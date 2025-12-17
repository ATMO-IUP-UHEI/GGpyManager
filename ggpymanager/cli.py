import logging
from enum import Enum
from pathlib import Path

import click

import ggpymanager as ggp
from ggpymanager import Catalog
from ggpymanager.utils.logging import set_logger

logger = logging.getLogger(__name__)


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    help="Logging level: DEBUG, INFO, WARNING, ERROR or CRITICAL",
)
@click.pass_context
def main(ctx, log_level):
    """GGPyManager CLI"""
    # Configure logging for the CLI using the project's logging util
    set_logger(log_level)
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level


class ModelType(str, Enum):
    GRAMM = "gramm"
    GRAL = "gral"
    NONE = "none"


@main.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option(
    "--model",
    "-f",
    type=click.Choice([m.value for m in ModelType]),
    default=ModelType.NONE.value,
)
def status(directory, model):
    """Get the status of the directory"""
    logger.info("Processing {directory} with status")
    if directory == ".":
        directory = Path.cwd()
    if model == "none":
        if "gramm" in Path(directory).name:
            model = "gramm"
        elif "gral" in Path(directory).name:
            model = "gral"
        else:
            raise ValueError(
                "Could not determine model from directory name. "
                "Please specify using --model option to select either 'gramm' or 'gral."
            )
    Catalog(directory, model=model)


@main.command()
@click.argument("config_filename", type=click.Path(exists=True))
def match(config_filename):
    """
    Match GRAMM/GRAL wind fields to observations using file paths from CONFIG_FILENAME.
    """
    logging.info(
        f"Match GRAMM/GRAL wind fields to observations using file paths from "
        f"{config_filename}."
    )
    logging.info(f"Loading configuration from {config_filename}")
    config = ggp.io.read_project_yaml_file(config_filename)
    return ggp.cli_functions.generate_matching_loss_file(config)


@main.command()
@click.argument("config_filename", type=click.Path(exists=True))
def timeseries(config_filename):
    """
    Generate time series data using file paths from CONFIG_FILENAME.
    """
    logging.info(f"Generate time series plots using file paths from {config_filename}.")
    logging.info(f"Loading configuration from {config_filename}")
    config = ggp.io.read_project_yaml_file(config_filename)
    return ggp.cli_functions.generate_timeseries(config)


if __name__ == "__main__":
    main()
