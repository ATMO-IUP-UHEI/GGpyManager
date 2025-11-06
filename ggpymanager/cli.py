# src/my_package/cli.py
import logging
from pathlib import Path

import click

from ggpymanager import Catalog

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)


@click.group()
def main():
    """My Package CLI tool"""
    pass


@main.command()
@click.argument("directory", type=click.Path(exists=True), default=".")
@click.option("--model", "-f", help="Model, either 'gramm' or 'gral", default="none")
@click.option("--flag2", is_flag=True, help="Boolean flag")
def status(directory, model, flag2):
    """Get the status of the directory"""
    click.echo(f"Processing {directory} with status")
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
    click.echo(f"Processing {filename} with process2")
    # Your logic here


if __name__ == "__main__":
    main()
