import xarray as xr
from pathlib import Path

from ggpymanager.config import (
    STATUS_LOG_FILE_NAME,
    MATCHING_LOSS_FILE_NAME,
    CONCENTRATION_TIMESERIES_FILE_NAME,
    GRAMM_METEO_TIMESERIES_FILE_NAME,
    GRAL_METEO_TIMESERIES_FILE_NAME,
)


def load(data_name: str, config: dict) -> xr.Dataset:
    file_paths = {
        # Measurements
        "temperature": Path(config["data_path"])
        / config["meteo_path"]
        / "temperature.nc",
        "pressure": Path(config["data_path"]) / config["meteo_path"] / "pressure.nc",
        # Catalog
        "gramm_status": Path(config["domain"]["gramm"]["conf_path"])
        / STATUS_LOG_FILE_NAME,
        "gral_status": Path(config["domain"]["gral"]["conf_path"])
        / STATUS_LOG_FILE_NAME,
        # Outputs
        "matching_loss": Path(config["output_path"]) / MATCHING_LOSS_FILE_NAME,
        "concentration_timeseries": Path(config["gral_co2_path"])
        / CONCENTRATION_TIMESERIES_FILE_NAME,
        "gramm_meteo_timeseries": Path(config["output_path"])
        / GRAMM_METEO_TIMESERIES_FILE_NAME,
        "gral_meteo_timeseries": Path(config["output_path"])
        / GRAL_METEO_TIMESERIES_FILE_NAME,
    }
    return xr.open_dataset(file_paths[data_name])
