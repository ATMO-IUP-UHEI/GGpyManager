import logging
from pathlib import Path

import xarray as xr

from ggpymanager.config import (
    BACKGROUND_CO2_FILE_NAME,
    CONCENTRATION_TIMESERIES_FILE_NAME,
    GRAL_METEO_TIMESERIES_FILE_NAME,
    GRAMM_METEO_TIMESERIES_FILE_NAME,
    MATCHING_LOSS_FILE_NAME,
    STATUS_LOG_FILE_NAME,
)

_METEO_PREPROCESS = {
    "gramm_meteo_catalog": "gramm",
    "gramm_meteo_timeseries": "gramm",
    "gral_meteo_catalog": "gral",
    "gral_meteo_timeseries": "gral",
}


def preprocess_gral_meteo(ds: xr.Dataset) -> xr.Dataset:
    """Post-load processing for GRAL meteorological data.

    Renames variables to canonical names and adds derived wind variables.
    The renaming from ``'ux'``/``'vy'`` → ``'u'``/``'v'`` compensates for a
    known naming issue in the server-side NetCDF that will be corrected in a
    future update, making this step obsolete.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset as returned by ``ggpy.load("gral_meteo_catalog", config)``.

    Returns
    -------
    xr.Dataset
        Processed dataset with variables: ``u``, ``v``,
        ``synoptic_wind_direction``, ``synoptic_wind_speed``, ``stab_class``,
        ``wind_speed``, ``wind_direction``.
    """
    if "ux" in ds.variables and "vy" in ds.variables:
        logging.warning(
            "'ux' and 'vy' variable names found in GRAL meteo data. "
            "Renaming to 'u' and 'v'. This compensates for a known server-side "
            "naming issue that will be fixed in a future update."
        )
        ds = ds.rename({"ux": "u", "vy": "v"})
    ds = ds.rename(
        {"direction": "synoptic_wind_direction", "speed": "synoptic_wind_speed"}
    )
    from ggpymanager import processing

    ds["wind_speed"] = processing.wind_speed_from_vector(ds["u"], ds["v"])
    ds["wind_direction"] = processing.direction_from_vector(ds["u"], ds["v"])
    return ds


def preprocess_gramm_meteo(ds: xr.Dataset) -> xr.Dataset:
    """Post-load processing for GRAMM meteorological data.

    Drops the spurious ``'speed'`` variable and adds derived wind variables.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset as returned by ``ggpy.load("gramm_meteo_catalog", config)``.

    Returns
    -------
    xr.Dataset
        Processed dataset with variables: ``u``, ``v``, ``wind_speed``,
        ``wind_direction``.
    """
    if "ux" in ds.variables and "vy" in ds.variables:
        logging.warning(
            "'ux' and 'vy' variable names found in GRAL meteo data. "
            "Renaming to 'u' and 'v'. This compensates for a known server-side "
            "naming issue that will be fixed in a future update."
        )
        ds = ds.rename({"ux": "u", "vy": "v"})
    if "speed" in ds.variables:
        logging.warning(
            "'speed' variable found in GRAMM meteo data. Dropping it. "
            "This compensates for a known server-side issue that will be "
            "fixed in a future update."
        )
        ds = ds.drop_vars("speed")
    from ggpymanager import processing

    ds["wind_speed"] = processing.wind_speed_from_vector(ds["u"], ds["v"])
    ds["wind_direction"] = processing.direction_from_vector(ds["u"], ds["v"])
    if "sim_id" in ds.dims and "station" in ds.dims:
        ds = ds.transpose("sim_id", "station")
    return ds


def _combine_gramm_gral_meteo(
    gramm_meteo: xr.Dataset, gral_meteo: xr.Dataset, config: dict
) -> xr.Dataset:
    model_selection = {"gramm": gramm_meteo, "gral": gral_meteo}
    meteo_at_station = []
    for s, m in config["matching"]["stations"].items():
        logging.info(f"Selecting model data for station {s} from model {m}")
        meteo_at_station.append(model_selection[m].sel(station=s))
    result = xr.concat(
        meteo_at_station, dim="station", coords="different", compat="equals"
    )
    assert isinstance(result, xr.Dataset)
    return result


def load(data_name: str, config: dict) -> xr.Dataset:
    """Load a dataset based on the provided data name and configuration.

    Parameters
    ----------
    data_name : str
        The name of the dataset to load. Must be one of:

        Measurements:
          - "temperature"
          - "pressure"
          - "meteo_measurements"
          - "co2_measurements"

        Catalog status:
          - "gramm_status"
          - "gral_status"

        Model inputs:
          - "gral_buildings"
          - "gramm_terrain"
          - "gral_terrain"
          - "gramm_landcover"
          - "source_groups"
          - "temporal_profiles"
          - "gramm_meteo_catalog"
          - "gral_meteo_catalog"
          - "gral_co2_catalog"
          - "model_meteo_catalog"

        Outputs (created by ggpy CLI commands):
          - "matching_loss"
          - "concentration_timeseries"
          - "gramm_meteo_timeseries"
          - "gral_meteo_timeseries"
          - "model_meteo_timeseries"
          - "background_co2"

    config : dict
        The configuration dictionary containing paths to the datasets.

    Returns
    -------
    xr.Dataset
        The loaded dataset corresponding to the provided data name.
        For meteo keys (``gramm_meteo_catalog``, ``gramm_meteo_timeseries``,
        ``gral_meteo_catalog``, ``gral_meteo_timeseries``), preprocessing is
        applied automatically via :func:`preprocess_gramm_meteo` or
        :func:`preprocess_gral_meteo`.
    """
    if data_name == "model_meteo_catalog":
        gramm_meteo = load("gramm_meteo_catalog", config)
        gral_meteo = load("gral_meteo_catalog", config)
        return _combine_gramm_gral_meteo(gramm_meteo, gral_meteo, config)
    if data_name == "model_meteo_timeseries":
        gramm_meteo = load("gramm_meteo_timeseries", config)
        gral_meteo = load("gral_meteo_timeseries", config)
        return _combine_gramm_gral_meteo(gramm_meteo, gral_meteo, config)
    c = config
    file_paths = {
        # Measurements
        "temperature": Path(c["meteo_path"]) / "temperature.nc",
        "pressure": Path(c["meteo_path"]) / "pressure.nc",
        "meteo_measurements": Path(c["meteo_path"]) / "meteo.nc",
        "co2_measurements": Path(c["co2_measurements_path"]),
        # Catalog
        "gramm_status": Path(c["domain"]["gramm"]["conf_path"]) / STATUS_LOG_FILE_NAME,
        "gral_status": Path(c["domain"]["gral"]["conf_path"]) / STATUS_LOG_FILE_NAME,
        # Model inputs
        "gral_buildings": Path(c["data_path"]) / "Buildings" / "buildings.nc",
        "gramm_terrain": Path(c["data_path"]) / "Terrain" / "gramm_terrain.nc",
        "gral_terrain": Path(c["data_path"]) / "Terrain" / "gral_terrain.nc",
        "gramm_landcover": Path(c["data_path"])
        / "Landcover"
        / "gramm_UrbanAtlas_landcover.nc",
        "source_groups": Path(c["source_groups_path"]),
        "temporal_profiles": Path(c["temporal_profiles_path"]),
        "gramm_meteo_catalog": Path(c["gramm_meteo_path"]) / "meteo.nc",
        "gral_meteo_catalog": Path(c["gral_meteo_path"]) / "meteo.nc",
        "gral_co2_catalog": Path(c["gral_co2_path"]) / "co2.nc",
        # Outputs
        "matching_loss": Path(c["output_path"]) / MATCHING_LOSS_FILE_NAME,
        "concentration_timeseries": Path(c["output_path"])
        / CONCENTRATION_TIMESERIES_FILE_NAME,
        "gramm_meteo_timeseries": Path(c["output_path"])
        / GRAMM_METEO_TIMESERIES_FILE_NAME,
        "gral_meteo_timeseries": Path(c["output_path"])
        / GRAL_METEO_TIMESERIES_FILE_NAME,
        "background_co2": Path(c["output_path"]) / BACKGROUND_CO2_FILE_NAME,
    }
    logging.info(f"Opening {data_name} from {file_paths[data_name]}")
    ds = xr.open_mfdataset(file_paths[data_name])
    if data_name in _METEO_PREPROCESS:
        if _METEO_PREPROCESS[data_name] == "gramm":
            ds = preprocess_gramm_meteo(ds)
        else:
            ds = preprocess_gral_meteo(ds)
    return ds
