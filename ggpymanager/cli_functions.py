import logging
from pathlib import Path

import xarray as xr

import ggpymanager as ggp


def generate_matching_loss_file(config):
    matching_path = Path(config["output_path"]) / "matching_loss.nc"
    if matching_path.exists():
        logging.info(
            f"Matching loss file already exists at {matching_path}, not overwriting."
        )
        return

    logging.info("Loading meteorological measurements and model data...")
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

    logging.info("Constructing meteorological measurement mask...")
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
    logging.info("Computing matching loss...")
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
            loss = loss.expand_dims("loss_type")
            loss.name = "matching_loss"
            loss["loss_type"] = [f"{loss_type} - filter: {filter}"]
            losses.append(loss)
    matching_loss = xr.concat(losses, dim="loss_type")
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving matching loss to {matching_path}")
    matching_loss.to_netcdf(matching_path)
