import logging
from pathlib import Path

import xarray as xr

from ggpymanager.config import (CONCENTRATION_TIMESERIES_FILE_NAME,
                                GRAL_METEO_TIMESERIES_FILE_NAME,
                                GRAMM_METEO_TIMESERIES_FILE_NAME,
                                MATCHING_LOSS_FILE_NAME, STATUS_LOG_FILE_NAME)


def load(data_name: str, config: dict) -> xr.Dataset:
    c = config
    file_paths = {
        # Measurements
        "temperature": Path(c["data_path"]) / c["meteo_path"] / "temperature.nc",
        "pressure": Path(c["data_path"]) / c["meteo_path"] / "pressure.nc",
        # Catalog
        "gramm_status": Path(c["domain"]["gramm"]["conf_path"]) / STATUS_LOG_FILE_NAME,
        "gral_status": Path(c["domain"]["gral"]["conf_path"]) / STATUS_LOG_FILE_NAME,
        # Outputs
        "matching_loss": Path(c["output_path"]) / MATCHING_LOSS_FILE_NAME,
        "concentration_timeseries": Path(c["gral_co2_path"])
        / CONCENTRATION_TIMESERIES_FILE_NAME,
        "gramm_meteo_timeseries": Path(c["output_path"])
        / GRAMM_METEO_TIMESERIES_FILE_NAME,
        "gral_meteo_timeseries": Path(c["output_path"])
        / GRAL_METEO_TIMESERIES_FILE_NAME,
    }
    logging.info(f"Opening {data_name} from {file_paths[data_name]}")
    return xr.open_dataset(file_paths[data_name])
