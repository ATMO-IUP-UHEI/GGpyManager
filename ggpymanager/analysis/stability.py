"""Atmospheric stability class filtering and selection."""

import logging

import numpy as np
import xarray as xr

from ggpymanager.utils.decorators import check_docstring_dims
from ggpymanager.io.readers import load_catalog_filter

logger = logging.getLogger(__name__)


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
    min_rads = catalog_filter.columns.get_level_values(1).astype(float)

    radiation_index.data = np.digitize(
        radiation,
        bins=min_rads.values,
    )
    logger.info(radiation_index.to_pandas().value_counts().sort_index())

    wind_speed_index.data = (
        np.digitize(
            wind_speed,
            bins=catalog_filter.index.values.astype(float),
        )
        - 1
    )  # Adjust for zero-based indexing
    logger.info(wind_speed_index.to_pandas().value_counts().sort_index())

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

    logger.info(stab_class_as_str.to_pandas().value_counts().sort_index())
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
