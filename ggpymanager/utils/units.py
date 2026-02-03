"""Unit conversion utilities."""

from typing import overload
import logging

import numpy as np
import xarray as xr
from numpy.typing import NDArray


@overload
def ugm3_to_ppm(
    ugm3: xr.DataArray,
    gas: str,
    P_local: float | NDArray[np.floating] | xr.DataArray | None = None,
    T_local: float | NDArray[np.floating] | xr.DataArray | None = None,
    P0: float | NDArray[np.floating] | xr.DataArray | None = None,
    T0: float | NDArray[np.floating] | xr.DataArray | None = None,
    h: float | NDArray[np.floating] | xr.DataArray | None = None,
) -> xr.DataArray: ...


@overload
def ugm3_to_ppm(
    ugm3: float | NDArray[np.floating],
    gas: str,
    P_local: float | NDArray[np.floating] | xr.DataArray | None = None,
    T_local: float | NDArray[np.floating] | xr.DataArray | None = None,
    P0: float | NDArray[np.floating] | xr.DataArray | None = None,
    T0: float | NDArray[np.floating] | xr.DataArray | None = None,
    h: float | NDArray[np.floating] | xr.DataArray | None = None,
) -> NDArray[np.floating]: ...


def ugm3_to_ppm(
    ugm3: float | NDArray[np.floating] | xr.DataArray,
    gas: str,
    P_local: float | NDArray[np.floating] | xr.DataArray | None = None,
    T_local: float | NDArray[np.floating] | xr.DataArray | None = None,
    P0: float | NDArray[np.floating] | xr.DataArray | None = None,
    T0: float | NDArray[np.floating] | xr.DataArray | None = None,
    h: float | NDArray[np.floating] | xr.DataArray | None = None,
) -> NDArray[np.floating] | xr.DataArray:
    """Convert concentration from µg/m³ to ppm for CH4 or CO2.

    Parameters
    ----------
    ugm3 : float | NDArray[np.floating] | xr.DataArray
        Concentration in µg/m³. Can be a scalar, numpy array, or xarray DataArray.
    gas : str
        Gas species, either 'CO2' or 'CH4' (case-insensitive).
    P_local : float | NDArray[np.floating] | xr.DataArray, optional
        Local pressure in Pa. Can be a scalar, numpy array, or xarray DataArray.
    T_local : float | NDArray[np.floating] | xr.DataArray, optional
        Local temperature in K. Can be a scalar, numpy array, or xarray DataArray.
    P0 : float | NDArray[np.floating] | xr.DataArray, optional
        Ground-level pressure in Pa. Can be a scalar, numpy array, or
        xarray DataArray.
    T0 : float | NDArray[np.floating] | xr.DataArray, optional
        Ground-level temperature in K. Can be a scalar, numpy array, or
        xarray DataArray.
    h : float | NDArray[np.floating] | xr.DataArray, optional
        Height above ground in m. Can be a scalar, numpy array, or
        xarray DataArray.

    Returns
    -------
    ppm : NDArray[np.floating] | xr.DataArray
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
        logging.info("Using provided local pressure and temperature for conversion.")
        P = P_local
        T = T_local
    elif P0 is not None and T0 is not None and h is not None:
        logging.info(
            "Calculating local pressure and temperature using "
            "isothermal barometric formula."
        )
        # Calculate using isothermal barometric formula
        M_air = 28.965e-3  # kg/mol - Molar mass of dry air
        g = 9.80665  # m/s² - Standard gravity
        P = P0 * np.exp(-M_air * g * h / (R * T0))
        T = T0
    elif (
        P_local is None and T_local is None and P0 is None and T0 is None and h is None
    ):
        logging.info("Using standard conditions for conversion.")
        # Standard conditions
        P = 101325.0  # Pa
        T = 298.15  # K (25°C)
    else:
        raise ValueError(
            "Must provide either (P_local, T_local) or (P0, T0, h) for conversion"
        )

    # Convert µg/m³ to ppm using ideal gas law
    ppm = ugm3 * R * T / (M * P) * 1e-3

    # Ensure return type matches overload signatures
    if not isinstance(ppm, xr.DataArray):
        ppm = np.asarray(ppm, dtype=np.float64)
    else:
        logging.info("Setting attributes for output DataArray.")
        ppm.attrs["long_name"] = f"{gas_upper} mixing ratio"
        ppm.attrs["units"] = "ppm"

    check_ppm_range(ppm)
    return ppm


def check_ppm_range(ppm):
    if np.min(ppm) < 0:
        logging.warning("Converted ppm values contain negative values.")
    if np.min(ppm) < -100:
        logging.error("Converted ppm values contain values less than -100 ppm.")
    if np.max(ppm) > 100:
        logging.warning("Converted ppm values contain values greater than 100 ppm.")
    if np.max(ppm) > 1000:
        logging.error("Converted ppm values contain values greater than 1,000 ppm.")
