from pathlib import Path

import click
import logging

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


@main.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--model", "-f", help="Model, either 'gramm' or 'gral", default="none")
@click.option("--flag2", is_flag=True, help="Boolean flag")
def status(directory, model, flag2):
    """Get the status of the directory"""
    # click.echo(f"Processing {directory} with status")
    logger.info("Processing %s with status", directory)
    # Your logic here
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
@click.argument("filename")
@click.option("--output", "-o", help="Output file")
def process2(filename, output):
    """Second processing command"""
    logger.info("Processing %s with process2", filename)
    # Your logic here


if __name__ == "__main__":
    main()
