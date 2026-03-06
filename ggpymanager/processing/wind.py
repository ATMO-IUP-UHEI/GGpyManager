"""Wind calculation utilities."""

import numpy as np
from typing import Any


def direction_from_compass(compass: str) -> float:
    """Convert compass direction string to degrees.

    Parameters
    ----------
    compass : str
        Compass direction (e.g., "N", "NE", "SSE").
        Case-insensitive.

    Returns
    -------
    direction : float
        Wind direction in degrees (meteorological convention).

    Raises
    ------
    ValueError
        If the compass direction is not recognized.
    """
    compass_map = {
        "N": 0,
        "NNE": 22.5,
        "NE": 45,
        "ENE": 67.5,
        "E": 90,
        "ESE": 112.5,
        "SE": 135,
        "SSE": 157.5,
        "S": 180,
        "SSW": 202.5,
        "SW": 225,
        "WSW": 247.5,
        "W": 270,
        "WNW": 292.5,
        "NW": 315,
        "NNW": 337.5,
    }

    compass_upper = compass.upper().strip()
    if compass_upper not in compass_map:
        raise ValueError(f"Unknown compass direction: {compass}")

    return compass_map[compass_upper]


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


def circular_mean(angles: Any) -> Any:
    """Calculate the circular mean of a set of angles.

    Parameters
    ----------
    angles : array_like
        Array of angles in degrees.

    Returns
    -------
    mean_angle : array_like
        Circular mean angle in degrees.
    """
    sin_sum = np.sum(np.sin(np.deg2rad(angles)), axis=0)
    cos_sum = np.sum(np.cos(np.deg2rad(angles)), axis=0)
    mean_angle = np.rad2deg(np.arctan2(sin_sum, cos_sum)) % 360
    return mean_angle


def circular_diff(angle1: Any, angle2: Any) -> Any:
    """Calculate the circular difference between two angles.

    Parameters
    ----------
    angle1 : array_like
        First angle in degrees.
    angle2 : array_like
        Second angle in degrees.

    Returns
    -------
    diff : array_like
        Circular difference in degrees, in the range [-180, 180].
    """
    diff = (angle1 - angle2 + 180) % 360 - 180
    return diff
