"""Unit conversion utilities."""

import numpy as np
import xarray as xr
from numpy.typing import ArrayLike


def ugm3_to_ppm(
    ugm3: float | ArrayLike | xr.DataArray,
    gas: str,
    P_local: float | ArrayLike | xr.DataArray | None = None,
    T_local: float | ArrayLike | xr.DataArray | None = None,
    P0: float | ArrayLike | xr.DataArray | None = None,
    T0: float | ArrayLike | xr.DataArray | None = None,
    h: float | ArrayLike | xr.DataArray | None = None,
) -> float | np.ndarray | xr.DataArray:
    """Convert concentration from µg/m³ to ppm for CH4 or CO2.

    Parameters
    ----------
    ugm3 : float | ArrayLike | xr.DataArray
        Concentration in µg/m³. Can be a scalar, numpy array, or xarray DataArray.
    gas : str
        Gas species, either 'CO2' or 'CH4' (case-insensitive).
    P_local : float | ArrayLike | xr.DataArray, optional
        Local pressure in Pa. Can be a scalar, numpy array, or xarray DataArray.
    T_local : float | ArrayLike | xr.DataArray, optional
        Local temperature in K. Can be a scalar, numpy array, or xarray DataArray.
    P0 : float | ArrayLike | xr.DataArray, optional
        Ground-level pressure in Pa. Can be a scalar, numpy array, or
        xarray DataArray.
    T0 : float | ArrayLike | xr.DataArray, optional
        Ground-level temperature in K. Can be a scalar, numpy array, or
        xarray DataArray.
    h : float | ArrayLike | xr.DataArray, optional
        Height above ground in m. Can be a scalar, numpy array, or
        xarray DataArray.

    Returns
    -------
    ppm : float | np.ndarray | xr.DataArray
        Concentration in ppm (volume/volume). Returns the same type as input ugm3.

    Raises
    ------
    ValueError
        If gas is not 'CO2' or 'CH4', or if neither (P_local, T_local)
        nor (P0, T0, h) are provided.

    Notes
    -----
    The conversion uses the ideal gas law:
        ppm = (ugm3 * R * T) / (M * P) * 1e-3

    If local conditions (P_local, T_local) are not provided, they are
    calculated from ground conditions using the isothermal barometric formula.
    """
    # Determine molar mass based on gas species
    gas_upper = gas.upper()
    if gas_upper == "CO2":
        M = 44.01e-3  # kg/mol
    elif gas_upper == "CH4":
        M = 16.04e-3  # kg/mol
    else:
        raise ValueError(f"Gas must be 'CO2' or 'CH4', got '{gas}'")

    R = 8.314462618  # J/(mol K) - Universal gas constant

    # Determine local pressure and temperature
    if P_local is not None and T_local is not None:
        P = P_local
        T = T_local
    elif P0 is not None and T0 is not None and h is not None:
        # Calculate using isothermal barometric formula
        M_air = 28.965e-3  # kg/mol - Molar mass of dry air
        g = 9.80665  # m/s² - Standard gravity
        P = P0 * np.exp(-M_air * g * h / (R * T0))
        T = T0
    elif (
        P_local is None and T_local is None and P0 is None and T0 is None and h is None
    ):
        # Standard conditions
        P = 101325.0  # Pa
        T = 298.15  # K (25°C)
    else:
        raise ValueError(
            "Must provide either (P_local, T_local) or (P0, T0, h) for conversion"
        )

    # Convert µg/m³ to ppm using ideal gas law
    ppm = ugm3 * R * T / (M * P) * 1e-3

    if isinstance(ppm, xr.DataArray):
        ppm.attrs["units"] = "ppm"
    return ppm
