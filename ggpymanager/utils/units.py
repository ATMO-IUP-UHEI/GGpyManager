"""Unit conversion utilities."""

import numpy as np


def ugm3_to_ppm(
    ugm3: float,
    gas: str,
    P_local: float | None = None,
    T_local: float | None = None,
    P0: float | None = None,
    T0: float | None = None,
    h: float | None = None,
) -> float:
    """Convert concentration from µg/m³ to ppm for CH4 or CO2.

    Parameters
    ----------
    ugm3 : float
        Concentration in µg/m³.
    gas : str
        Gas species, either 'CO2' or 'CH4' (case-insensitive).
    P_local : float, optional
        Local pressure in Pa.
    T_local : float, optional
        Local temperature in K.
    P0 : float, optional
        Ground-level pressure in Pa.
    T0 : float, optional
        Ground-level temperature in K.
    h : float, optional
        Height above ground in m.

    Returns
    -------
    ppm : float
        Concentration in ppm (volume/volume).

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

    return ppm
