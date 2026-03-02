import logging
from pathlib import Path

import xarray as xr
from dask.diagnostics.progress import ProgressBar

import ggpymanager as ggp


def generate_matching_loss_file(config):
    matching_path = Path(config["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME
    if matching_path.exists():
        logging.info(
            f"Matching loss file already exists at {matching_path}, not overwriting."
        )
        return

    model_meteo = ggp.load("model_meteo", config)

    fp = config["gral_meteo_path"] + "/meteo.nc"
    logging.info(f"Loading GRAL meteo from {fp}")
    gral_meteo = xr.open_dataset(fp)
    meteo = xr.open_dataset(config["meteo_path"] + "/meteo.nc")
    meteo_measurements = meteo.sel(
        station=model_meteo["station"],
        time=slice(config["matching"]["time_start"], config["matching"]["time_end"]),
    )
    logging.info("Computing matching loss...")
    losses = []
    if "loss_types" in config["matching"]:
        loss_types = config["matching"]["loss_types"]
        logging.info(f"Using specified loss types: {loss_types}")
    else:
        loss_types = ["rmse", "regularized", "compound"]
        logging.info(f"No loss types specified in config, using default: {loss_types}")
    for filter in [False, True]:
        for loss_type in loss_types:
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
            loss = loss.astype("float32")
            losses.append(loss)
    matching_loss = xr.concat(losses, dim="loss_type")
    assert isinstance(
        matching_loss, xr.DataArray
    ), f"Variable 'matching_loss' is not a DataArray, got {type(matching_loss)}"
    matching_loss = matching_loss.to_dataset(name="matching_loss")  # type: ignore
    n_stations_per_time = (
        meteo_measurements["u_wind"].notnull() & meteo_measurements["v_wind"].notnull()
    ).sum("station")
    matching_loss["n_stations_per_time"] = n_stations_per_time
    matching_loss["n_stations_per_time"].attrs = {
        "long_name": "Number of stations with valid measurements per time step",
    }
    matching_loss["sim_id"].attrs = {"long_name": "Simulation ID", "units": "1"}
    matching_loss.attrs["title"] = (
        "Matching loss between GRAMM/GRAL simulations and meteorological measurements"
    )
    matching_path.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Saving matching loss to {matching_path}")
    ggp.io.writers.save_netcdf_with_cf_check(matching_loss, matching_path)


def generate_timeseries(config):
    concentration_timeseries_path = (
        Path(config["output_path"]) / ggp.config.CONCENTRATION_TIMESERIES_FILE_NAME
    )
    gramm_meteo_timeseries_path = (
        Path(config["output_path"]) / ggp.config.GRAMM_METEO_TIMESERIES_FILE_NAME
    )
    gral_meteo_timeseries_path = (
        Path(config["output_path"]) / ggp.config.GRAL_METEO_TIMESERIES_FILE_NAME
    )
    # TODO: Refactor to make this check for each file separately and only skip
    # generation for files that already exist, instead of skipping all if any file
    # exists. Also add error handling for missing input files or variables.
    if (
        concentration_timeseries_path.exists()
        and gramm_meteo_timeseries_path.exists()
        and gral_meteo_timeseries_path.exists()
    ):
        logging.info(
            f"Concentration and meteo timeseries files already exist at "
            f"{concentration_timeseries_path}, {gramm_meteo_timeseries_path}, and "
            f"{gral_meteo_timeseries_path}, not overwriting."
        )
        return
    matching_path = Path(config["output_path"]) / ggp.config.MATCHING_LOSS_FILE_NAME
    logging.info(f"Opening matching loss from {matching_path}")
    matching_loss = xr.load_dataset(matching_path)
    fp = Path(config["gramm_meteo_path"]) / "meteo.nc"
    logging.info(f"Opening GRAMM meteo from {fp}")
    gramm_meteo = xr.open_mfdataset(fp)
    fp = Path(config["gral_meteo_path"]) / "meteo.nc"
    logging.info(f"Opening GRAL meteo from {fp}")
    gral_meteo = xr.open_mfdataset(fp)
    fp = Path(config["gral_co2_path"]) / "co2.nc"
    logging.info(f"Opening GRAL concentration from {fp}")
    gral_concentration = xr.open_mfdataset(fp)["concentration"]
    fp = config["temporal_profiles_path"]
    logging.info(f"Loading temporal profiles from {fp}")
    temporal_factor = xr.load_dataset(fp)["temporal"]
    fp = config["source_groups_path"]
    logging.info(f"Loading source groups from {fp}")
    source_groups = xr.load_dataset(fp)

    matching_period = (
        matching_loss["time"][[0, -1]].dt.strftime("%Y-%m-%d %H:%M").values
    )
    flux_temporal_facter_period = (
        temporal_factor["time"][[0, -1]].dt.strftime("%Y-%m-%d %H:%M").values
    )
    factors_available = (flux_temporal_facter_period[0] <= matching_period[0]) and (
        flux_temporal_facter_period[1] >= matching_period[1]
    )

    matching_period = {"start": matching_period[0], "end": matching_period[1]}
    flux_temporal_facter_period = {
        "start": flux_temporal_facter_period[0],
        "end": flux_temporal_facter_period[1],
    }
    logging.info(f"Matched timeseries period: {matching_period}")
    logging.info(f"Flux temporal factor period: {flux_temporal_facter_period}")
    assert factors_available, (
        "Temporal factors do not cover the matching period. "
        "Please provide temporal factors that cover "
        f"{matching_period['start']} to {matching_period['end']}."
    )
    logging.info("Generating concentration time series data...")
    n_best_sims = config["matching"].get("n_best_simulations", 1)
    logging.info(f"Selecting {n_best_sims} best matching simulation IDs.")
    sim_ids = xr.concat(
        [
            ggp.analysis.get_sim_ids(
                matching_loss.matching_loss.sel(loss_type=lt), n_best=n_best_sims
            )
            for lt in matching_loss.loss_type.values
        ],
        dim="loss_type",
    )
    logging.info("Calculating concentration time series...")
    k = (
        (gral_concentration.groupby(source_groups["type"]).sum())
        .astype("float32")
        .compute()
    ).chunk("auto")
    logging.info(f"GRAL concentration chunk sizes: {k.chunks}")
    k = k.sel(sim_id=sim_ids)
    # Convert to ppm
    pressure = ggp.load("pressure", config).pressure
    temperature = ggp.load("temperature", config).temperature
    k = ggp.utils.ugm3_to_ppm(
        k,
        "co2",
        P_local=pressure,
        T_local=temperature,
    )
    f = temporal_factor.astype("float32")
    # Old method with source groups - commented out
    # k = gral_concentration.sel(sim_id=sim_ids).astype("float32")
    # f = temporal_factor.sel(type=source_groups["type"]).astype("float32")
    concentration_timeseries = (k * f).to_dataset(name="co2_timeseries")
    logging.info("Calculating loss difference to best simulation...")
    loss = matching_loss.matching_loss.sel(sim_id=sim_ids)
    concentration_timeseries.coords["loss_diff"] = loss - loss.sel(best_sim_id=0)
    concentration_timeseries["loss_diff"].attrs = {
        "long_name": "Difference in matching loss to best simulation",
        "units": loss.attrs.get("units", ""),
    }
    logging.info(f"Saving concentration timeseries to {concentration_timeseries_path}")
    if not concentration_timeseries_path.exists():
        with ProgressBar():
            ggp.io.writers.save_netcdf_with_cf_check(
                concentration_timeseries,
                concentration_timeseries_path,
                ignore_tests=True,
            )

    if not gramm_meteo_timeseries_path.exists():
        logging.info("Generating GRAMM meteo time series data...")
        gramm_meteo_timeseries = gramm_meteo.sel(
            sim_id=sim_ids.sel(best_sim_id=0)
        ).astype("float32")
        logging.info(f"Saving GRAMM meteo timeseries to {gramm_meteo_timeseries_path}")
        ggp.io.writers.save_netcdf_with_cf_check(
            gramm_meteo_timeseries, gramm_meteo_timeseries_path, ignore_tests=True
        )

    if not gral_meteo_timeseries_path.exists():
        logging.info("Generating GRAL meteo time series data...")
        gral_meteo_timeseries = gral_meteo.sel(
            sim_id=sim_ids.sel(best_sim_id=0)
        ).astype("float32")
        logging.info(f"Saving GRAL meteo timeseries to {gral_meteo_timeseries_path}")
        ggp.io.writers.save_netcdf_with_cf_check(
            gral_meteo_timeseries, gral_meteo_timeseries_path, ignore_tests=True
        )
