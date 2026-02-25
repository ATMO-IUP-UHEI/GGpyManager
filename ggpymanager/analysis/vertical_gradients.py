"""The code contains functions to compute vertical gradients for the matching. This is
currently not used."""

import logging

import xarray as xr


def compute_normalized_vertical_gradient(
    data_array: xr.DataArray, vertical_coord_name: str
) -> xr.DataArray:
    """Compute the normalized vertical gradient of a data array.

    Groups the data by the specified vertical coordinate, computes the mean
    vertical profile, normalizes it, and returns the total gradient as a
    scalar (sum of finite differences along the vertical axis).

    Parameters
    ----------
    data_array : xr.DataArray
        Input data array containing a vertical coordinate.
    vertical_coord_name : str
        Name of the vertical coordinate to group by (e.g. ``"height"`` or
        ``"altitude"``).

    Returns
    -------
    xr.DataArray
        Scalar-like array containing the sum of normalized finite differences
        along the vertical coordinate.
    """
    logging.info(
        f"Computing normalized vertical gradient for {data_array.name} grouped by "
        f"{vertical_coord_name}"
    )
    vcn = vertical_coord_name
    # Group by vertical coordinate and compute mean profile
    p = data_array.load().groupby(vcn).mean()
    # Substract min and normalize by mean
    p = p - p.min()
    normalized = p / p.mean(vcn)
    # Compute total gradient
    gradient = normalized.diff(vcn).sum(vcn)
    return gradient
