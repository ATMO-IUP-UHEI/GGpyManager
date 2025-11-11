"""Concentrtation utilities."""

import logging
from typing import Any, Dict, Literal

import geopandas as gpd
import numpy as np
import shapely.geometry
import xarray as xr

from ggpymanager.utils.decorators import check_docstring_dims


@check_docstring_dims
def convert_locations_to_grid(
    x: xr.DataArray,
    y: xr.DataArray,
    height: xr.DataArray,
    x_grid: xr.DataArray,
    y_grid: xr.DataArray,
    concentration_levels: xr.DataArray,
) -> xr.Dataset:
    """Convert locations in crs to indices of the concentration grid.

    Parameters
    ----------
    x : xr.DataArray (station)
        x coordinates of measurement stations.
    y : xr.DataArray (station)
        y coordinates of measurement stations.
    height : xr.DataArray (station)
        z coordinates of measurement stations.
    x_grid : xr.DataArray (x)
        x coordinates of the concentration grid.
    y_grid : xr.DataArray (y)
        y coordinates of the concentration grid.
    concentration_levels : xr.DataArray (z)
        Height levels of the concentration grid.
    Returns
    -------
    grid_locations : xr.Dataset (station)
        Dataset containing the x_id, y_id, and z_id for each station.
    """
    assert "station" in x.coords
    logging.info("Generate x, y, and z ids for each station")
    x_id = abs(x_grid - x).argmin(dim="x")
    y_id = abs(y_grid - y).argmin(dim="y")
    z_id = abs(concentration_levels - height).argmin(dim="z")
    mask = (
        (x >= x_grid.min())
        & (x <= x_grid.max())
        & (y >= y_grid.min())
        & (y <= y_grid.max())
    )
    x_id = x_id.where(mask, drop=True).astype(int)
    y_id = y_id.where(mask, drop=True).astype(int)
    z_id = z_id.where(mask, drop=True).astype(int)
    logging.info(
        f"{mask.sum().values}/{len(mask)} of stations are inside the GRAL domain. "
        "Dropping the rest."
    )
    grid_locations = xr.Dataset(
        coords={
            "x_id": x_id,
            "y_id": y_id,
            "z_id": z_id,
        },
    )
    return grid_locations


# Get the x_id, y_id, and z_id
# If it is too close to a building print a warning and move one layer up
# If this does not fix the issue
# Create a config file for user input to shift the locations of measurements stations

# Read the concentration at the location and safe to the netcdf

# def get_representative_positions(x, y, height, surface_elevation, concentration_levels):
