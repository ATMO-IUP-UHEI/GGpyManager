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
          - "source_groups"
          - "temporal_profiles"
          - "gramm_meteo_raw"
          - "gral_meteo_raw"
          - "gral_co2_raw"

        Outputs (created by ggpy CLI commands):
          - "matching_loss"
          - "concentration_timeseries"
          - "gramm_meteo_timeseries"
          - "gral_meteo_timeseries"
          - "background_co2"

    config : dict
        The configuration dictionary containing paths to the datasets.

    Returns
    -------
    xr.Dataset
        The loaded dataset corresponding to the provided data name.
    """
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
        "source_groups": Path(c["source_groups_path"]),
        "temporal_profiles": Path(c["temporal_profiles_path"]),
        "gramm_meteo_raw": Path(c["gramm_meteo_path"]) / "meteo.nc",
        "gral_meteo_raw": Path(c["gral_meteo_path"]) / "meteo.nc",
        "gral_co2_raw": Path(c["gral_co2_path"]) / "co2.nc",
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
    return xr.open_mfdataset(file_paths[data_name])
