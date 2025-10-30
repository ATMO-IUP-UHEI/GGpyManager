"""Wind calculation utilities."""

import numpy as np
from typing import Any


def direction_from_vector(ux: Any, vy: Any) -> Any:
    """Calculate the wind direction from the vector components.

    Parameters
    ----------
    ux : array_like
        Eastward component of the wind vector.
    vy : array_like
        Northward component of the wind vector.

    Returns
    -------
    direction : array_like
        Wind direction in degrees (meteorological convention:
        direction wind comes from).
    """
    return (90 - np.rad2deg(np.arctan2(-vy, -ux))) % 360


def wind_speed_from_vector(ux: Any, vy: Any) -> Any:
    """Calculate the wind speed from the vector components.

    Parameters
    ----------
    ux : array_like
        Eastward component of the wind vector.
    vy : array_like
        Northward component of the wind vector.

    Returns
    -------
    wind_speed : array_like
        Wind speed in m/s.
    """
    return np.sqrt(ux**2 + vy**2)


def vector_from_direction_and_speed(direction: Any, speed: Any) -> tuple[Any, Any]:
    """Calculate the vector components from the wind direction and speed.

    Parameters
    ----------
    direction : array_like
        Wind direction in degrees (meteorological convention).
    speed : array_like
        Wind speed in m/s.

    Returns
    -------
    ux : array_like
        Eastward component of the wind vector.
    vy : array_like
        Northward component of the wind vector.
    """
    # Convert the direction into radians
    rad = np.deg2rad(direction)
    # Calculate the vector components
    ux = -speed * np.sin(rad)  # Eastward component
    vy = -speed * np.cos(rad)  # Northward component
    return ux, vy
