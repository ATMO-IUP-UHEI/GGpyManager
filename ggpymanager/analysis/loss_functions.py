"""Loss functions for wind field matching and validation."""

import logging

import numpy as np
import xarray as xr

from ggpymanager.utils.decorators import check_docstring_dims
from ggpymanager.processing.wind import (
    direction_from_vector,
    wind_speed_from_vector,
)


@check_docstring_dims
def rmse_loss(u, v, u_model, v_model):
    """Compute RMSE loss between measured and modeled wind fields.

    Input for N hours, S stations, and M catalog entries.

    Parameters
    ----------
    u : xr.DataArray (time, station)
        Hourly wind speed in x-direction from measurements.
    v : xr.DataArray (time, station)
        Hourly wind speed in y-direction from measurements.
    u_model : xr.DataArray (sim_id, station)
        Hourly wind speed in x-direction from catalog.
    v_model : xr.DataArray (sim_id, station)
        Hourly wind speed in y-direction from catalog.

    Returns
    -------
    rmse : xr.DataArray (time, sim_id)
        Root mean squared error for each hour and catalog entry.
    """
    return np.sqrt(((u - u_model) ** 2 + (v - v_model) ** 2).mean(dim="station"))


@check_docstring_dims
def regularized_loss(u, v, u_model, v_model):
    """Compute regularized loss with temporal and spatial weighting.

    Input for N hours, S stations, and M catalog entries.

    Loss function based on: Berchet, Antoine, Katrin Zink, Clive Muller,
    Dietmar Oettl, Juerg Brunner, Lukas Emmenegger, and Dominik Brunner. 2017.
    'A Cost-Effective Method for Simulating City-Wide Air Flow and Pollutant
    Dispersion at Building Resolving Scale'. Atmospheric Environment 158:181–96.
    https://doi.org/10.1016/j.atmosenv.2017.03.030.

    Parameters
    ----------
    u : xr.DataArray (time, station)
        Hourly wind speed in x-direction from measurements.
    v : xr.DataArray (time, station)
        Hourly wind speed in y-direction from measurements.
    u_model : xr.DataArray (sim_id, station)
        Hourly wind speed in x-direction from catalog.
    v_model : xr.DataArray (sim_id, station)
        Hourly wind speed in y-direction from catalog.

    Returns
    -------
    regularized_loss : xr.DataArray (time, sim_id)
        Regularized loss for each hour and catalog entry.
    """
    # Create station weights
    wind_speed = np.sqrt(u**2 + v**2)
    wind_speed_min = 2  # m/s
    std_wind_speed = wind_speed.std(dim="time")
    station_weight = std_wind_speed * wind_speed.clip(wind_speed_min)

    # Correlation between hours
    sigma = 1.0  # correlation length in hours
    compute_range = 3  # hours

    wind_speed_difference = np.sqrt((u - u_model) ** 2 + (v - v_model) ** 2)
    loss_per_hour = (wind_speed_difference / station_weight).mean(dim="station")
    time_difference = np.arange(-compute_range, compute_range + 1)
    window = xr.DataArray(
        np.exp(-(time_difference**2) / (sigma**2)),
        dims="window",
        coords={"window": time_difference},
    )
    loss = loss_per_hour.rolling(
        time=2 * compute_range + 1,
        min_periods=1,
        center=True,
    ).construct("window")
    loss = loss * window
    loss = loss.sum(dim="window") / window.sum()
    return loss


@check_docstring_dims
def compound_loss(u, v, u_model, v_model, lambda_=0.7):
    """Compute compound loss combining direction and speed differences.

    Input for N hours, S stations, and M catalog entries.

    Parameters
    ----------
    u : xr.DataArray (time, station)
        Hourly wind speed in x-direction from measurements.
    v : xr.DataArray (time, station)
        Hourly wind speed in y-direction from measurements.
    u_model : xr.DataArray (sim_id, station)
        Hourly wind speed in x-direction from catalog.
    v_model : xr.DataArray (sim_id, station)
        Hourly wind speed in y-direction from catalog.
    lambda_ : float, optional
        Weight for direction component (0-1). Default is 0.7.

    Returns
    -------
    compound_loss : xr.DataArray (time, sim_id)
        Compound loss for each hour and catalog entry.
    """
    direction = direction_from_vector(u, v)
    wind_speed = wind_speed_from_vector(u, v)
    direction_model = direction_from_vector(u_model, v_model)
    wind_speed_model = wind_speed_from_vector(u_model, v_model)

    # Difference in direction
    direction_difference = np.abs(direction - direction_model)
    direction_difference = np.minimum(direction_difference, 360 - direction_difference)
    direction_difference = direction_difference / 180

    # Difference in wind speed
    wind_speed_difference = np.abs(wind_speed - wind_speed_model)
    weight = 1 / wind_speed.clip(wind_speed.median()).mean(dim="time")
    wind_speed_difference = wind_speed_difference * weight
    wind_speed_difference = wind_speed_difference / wind_speed_difference.max()

    compound_loss = (
        lambda_ * direction_difference + (1 - lambda_) * wind_speed_difference
    ).sum(dim="station")
    return compound_loss


@check_docstring_dims
def compute_matching_loss(
    u: xr.DataArray,
    v: xr.DataArray,
    u_model: xr.DataArray,
    v_model: xr.DataArray,
    matching: str = "rmse",
    filter: bool = False,
    synoptic_wind_speed=None,
    global_radiation=None,
    stab_class_catalog=None,
):
    """Compute matching loss between observations and model catalog.

    Input for N hours, S stations, and M catalog entries.

    Parameters
    ----------
    u : xr.DataArray (time, station)
        Hourly wind speed in x-direction from measurements.
    v : xr.DataArray (time, station)
        Hourly wind speed in y-direction from measurements.
    u_model : xr.DataArray (sim_id, station)
        Hourly wind speed in x-direction from catalog.
    v_model : xr.DataArray (sim_id, station)
        Hourly wind speed in y-direction from catalog.
    matching : str, optional
        Matching loss function. Default is 'rmse'.
        Options: 'rmse', 'regularized', 'compound'.
    filter : bool, optional
        Whether to filter by stability class. Default is False.
    synoptic_wind_speed : xr.DataArray, optional (sim_id)
        Synoptic wind speed data with sim_id dimension (required for filtering).
    global_radiation : xr.DataArray, optional (time)
        Global radiation data with time dimension (required for filtering).
    stab_class_catalog : xr.DataArray, optional (sim_id)
        Stability class data with sim_id dimension (required for filtering).

    Returns
    -------
    matching_loss : xr.DataArray (time, sim_id)
        Matching loss for each hour and catalog entry.
    """
    # Import here to avoid circular dependency
    from ggpymanager.analysis.stability import get_allowed_stability_class

    logging.info(f"Computing matching with {matching}...")
    loss_funcs = {
        "rmse": rmse_loss,
        "regularized": regularized_loss,
        "compound": compound_loss,
    }
    # Compute matching loss
    matching_loss = loss_funcs[matching](u, v, u_model, v_model)
    # Filter results
    if filter:
        stab_mask = get_allowed_stability_class(
            global_radiation, synoptic_wind_speed, stab_class_catalog
        )
        matching_loss = matching_loss.where(stab_mask)
        logging.info(
            "Filtered {:.1f} % of the results.".format(stab_mask.mean().values * 100)
        )

    # Add metadata
    matching_loss.name = f"{matching}_loss"
    matching_loss.attrs["matching"] = matching
    matching_loss.attrs["filter"] = str(filter)
    matching_loss.attrs["long_name"] = f"{matching} loss"
    matching_loss.attrs["units"] = ""
    return matching_loss
