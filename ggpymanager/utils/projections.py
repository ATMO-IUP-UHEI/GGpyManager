"""Coordinate reference system and projection utilities."""

import rasterio


def get_centered_custom_projection(
    center_lat: float, center_lon: float
) -> rasterio.CRS:
    """Create a custom transverse Mercator projection centered on given coordinates.

    Parameters
    ----------
    center_lat : float
        Latitude of the center point in degrees.
    center_lon : float
        Longitude of the center point in degrees.

    Returns
    -------
    rasterio.CRS
        Custom transverse Mercator CRS centered on the input coordinates.
    """
    custom_proj = rasterio.CRS.from_proj4(
        f"+proj=tmerc +lat_0={center_lat} +lon_0={center_lon} "
        "+k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
    )
    return custom_proj
