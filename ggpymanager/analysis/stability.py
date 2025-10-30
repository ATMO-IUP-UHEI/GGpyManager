"""Atmospheric stability class filtering and selection."""

import logging

import numpy as np
import xarray as xr

from ggpymanager.utils.decorators import check_docstring_dims
from ggpymanager.io.readers import load_catalog_filter


@check_docstring_dims
def get_allowed_stability_class(radiation, wind_speed, stab_class_catalog):
    """Get allowed stability classes based on radiation and wind speed.
    
    Filters simulation catalog entries based on atmospheric stability class
    compatibility with observed meteorological conditions.

    Parameters
    ----------
    radiation : xr.DataArray (time)
        Radiation data with time dimension.
    wind_speed : xr.DataArray (sim_id)
        Wind speed data with sim_id dimension.
    stab_class_catalog : xr.DataArray (sim_id)
        Stability class data with sim_id dimension.

    Returns
    -------
    stability_class_mask : xr.DataArray (time, sim_id)
        Binary array with one-hot encoding for the allowed stability classes
        at each time step.
    """
    radiation_index = xr.zeros_like(radiation, dtype=int)
    wind_speed_index = xr.zeros_like(wind_speed, dtype=int)
    catalog_filter = load_catalog_filter()
    min_rads = catalog_filter.columns.get_level_values(1).astype(float)[::-1]

    # Select bin for radiation for each time step
    for i, min_rad in enumerate(min_rads):
        above_rad_threshold = radiation >= min_rad
        radiation_index[above_rad_threshold] = i
        logging.info(
            "Radiation larger {:>5} W/m²: {:>4} entries".format(
                min_rad, above_rad_threshold.sum().values
            )
        )

    # Select bin for wind speed for each time step
    for i, min_wind_speed in enumerate(catalog_filter.index):
        above_wind_threshold = wind_speed >= float(min_wind_speed)
        wind_speed_index[above_wind_threshold] = i
        logging.info(
            "Wind speed larger {} m/s: {:>4} entries".format(
                min_wind_speed, above_wind_threshold.sum().values
            )
        )

    # Get stability class(es) for each time step (dims: sim_id, time)
    allowed_stab_classes = catalog_filter.values[:, radiation_index.values][
        wind_speed_index.values
    ].astype(str)
    allowed_stab_classes = xr.DataArray(
        allowed_stab_classes,
        dims=["sim_id", "time"],
        coords={
            "sim_id": wind_speed.sim_id,
            "time": radiation.time,
        },
    )

    # Convert stab_class_catalog to string
    stab_class_as_str = stab_class_catalog.astype(str)
    stab_index = ["A", "B", "C", "D", "E", "F", "G"]
    for stab, index in zip(stab_index, range(1, 8)):
        is_index = stab_class_catalog == index
        stab_class_as_str[is_index] = stab

    # Create empty mask
    stability_class_mask = xr.DataArray(
        np.zeros((len(radiation), len(wind_speed)), dtype=bool),
        dims=["time", "sim_id"],
        coords={
            "time": radiation.time,
            "sim_id": wind_speed.sim_id,
        },
    )

    # Check if stability class is allowed
    for i, stab in enumerate(stab_index):
        is_stab = stab_class_as_str == stab
        is_allowed_stability = np.strings.find(allowed_stab_classes, stab) >= 0
        stability_class_mask[dict(sim_id=is_stab)] = is_allowed_stability[
            dict(sim_id=is_stab)
        ]  # type: ignore
    return stability_class_mask
