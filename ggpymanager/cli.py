import logging
from pathlib import Path

import click
import xarray as xr

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
@click.argument("config_filename", type=click.Path(exists=True))
def match(config_filename):
    """
    Match GRAMM/GRAL wind fields to observations using file paths from CONFIG_FILENAME.
    """
    click.echo(
        f"Match GRAMM/GRAL wind fields to observations using file paths from "
        f"{config_filename}."
    )

    logging.info(f"Loading configuration from {config_filename}")
    config = ggp.io.read_project_yaml_file(config_filename)
    matching_path = Path(config["output_path"]) / "matching_loss.nc"
    if matching_path.exists():
        click.echo(
            f"Matching loss file already exists at {matching_path}, not overwriting."
        )
        return

    click.echo("Loading meteorological measurements and model data...")
    fp = config["gramm_meteo_path"] + "/meteo.nc"
    logging.info(f"Loading GRAMM meteo from {fp}")
    gramm_meteo = xr.open_dataset(fp)
    fp = config["gral_meteo_path"] + "/meteo.nc"
    logging.info(f"Loading GRAL meteo from {fp}")
    gral_meteo = xr.open_dataset(fp)
    # Fix wrong variable names in GRAL meteo file
    if "ux" in gral_meteo.variables and "vy" in gral_meteo.variables:
        gral_meteo = gral_meteo.rename({"ux": "u", "vy": "v"})
    meteo = xr.open_dataset(config["meteo_path"] + "/meteo.nc")

    click.echo("Constructing meteorological measurement mask...")
    model_selection = {
        "gramm": gramm_meteo,
        "gral": gral_meteo,
    }
    model_meteo = xr.concat(
        [
            model_selection[m].sel(station=s)
            for s, m in config["matching"]["stations"].items()
        ],
        dim="station",
        coords="minimal",
        compat="override",
    )
    meteo_measurements = meteo.sel(
        station=model_meteo["station"],
        time=slice(config["matching"]["time_start"], config["matching"]["time_end"]),
    )
    click.echo("Computing matching loss...")
    losses = []
    for filter in [False, True]:
        for loss_type in ["rmse", "regularized", "compound"]:
            loss = ggp.analysis.compute_matching_loss(
                meteo_measurements["u_wind"],
                meteo_measurements["v_wind"],
                model_meteo["u"].T,
                model_meteo["v"].T,
                matching=loss_type,
                filter=filter,
                synoptic_wind_speed=gral_meteo["speed"] if filter else None,
                global_radiation=(
                    meteo_measurements["global_radiation"].mean("station")
                    if filter
                    else None
                ),
                stab_class_catalog=gral_meteo["stab_class"] if filter else None,
            )
            loss = loss.expand_dims("loss")
            loss["loss"] = [f"{loss_type} - filter: {filter}"]
            losses.append(loss)
    matching_loss = xr.concat(losses, dim="loss")
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    click.echo(f"Saving matching loss to {matching_path}")
    matching_loss.to_netcdf(matching_path)


if __name__ == "__main__":
    main()
