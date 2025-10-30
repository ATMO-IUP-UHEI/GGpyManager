"""Land use processing and conversion utilities."""

import xarray as xr

from ggpymanager.io.readers import load_corine_lookup_table


def convert_to_gramm_landuse_variables(corine_data: xr.DataArray) -> xr.Dataset:
    """Convert CORINE land cover data to GRAMM landuse variables.

    Parameters
    ----------
    corine_data : xr.DataArray
        CORINE land cover classification data.

    Returns
    -------
    xr.Dataset
        Dataset with GRAMM landuse variables (RHOB, ALAMBDA, Z0, FW, EPSG, ALBEDO).

    Raises
    ------
    AssertionError
        If converted values are outside expected ranges.
    """
    corine_lookup = load_corine_lookup_table()
    converter = {
        "RHOB": lambda x: x["Heat conductivity [W/m/K]"]
        / x["Thermal diffusivity [m²/s]"]
        / 900,
        "ALAMBDA": lambda x: x["Heat conductivity [W/m/K]"],
        "Z0": lambda x: x["roughness length [m]"],
        "FW": lambda x: x["Soil moisture"],
        "EPSG": lambda x: x["emissivity"],
        "ALBEDO": lambda x: x["albedo"],
    }
    grid_data_stack = corine_data.stack(n=("x", "y"))
    landuse = {}
    for key in converter.keys():
        data = converter[key](corine_lookup.loc[grid_data_stack.values]).values
        landuse[key] = ("n", data)

    xds = xr.Dataset(landuse, coords=grid_data_stack.coords)
    xds = xds.unstack("n")
    attrs = {
        "RHOB": [
            "kg/m^3",
            "Soil density",
            "soil_density",
            "Soil density is calculated as heat conductivity divided by thermal "
            "diffusivity divided by the specific heat capacity = 900 J/(kg·K).",
        ],
        "ALAMBDA": [
            "W/m/K",
            "heat conductivity",
            "heat_conductivity",
            "",
        ],
        "Z0": [
            "m",
            "Aerodynamic surface roughness",
            "surface_roughness",
            "Aerodynamic surface roughness",
        ],
        "FW": [
            "1",
            "Specific soil moisture parameter",
            "specific_soil_moisture",
            "Specific soil moisture parameter for water is 1",
        ],
        "EPSG": ["1", "Surface emissivity", "surface_emissivity", ""],
        "ALBEDO": ["1", "Surface albedo", "surface_albedo", ""],
    }
    for var in attrs.keys():
        xds[var].attrs.update(
            dict(
                zip(
                    ["unit", "long_name", "standard_name", "description"],
                    attrs[var],
                )
            )
        )
    # Check that values are in normal range
    limits = {
        "RHOB": (0, 5000),
        "ALAMBDA": (0, 5),
        "Z0": (0, 2),
        "FW": (0, 1),
        "EPSG": (0, 1),
        "ALBEDO": (0, 1),
    }
    for var, (min_val, max_val) in limits.items():
        assert (
            xds[var].min() >= min_val
        ), f"{var} min value is below limit of {xds[var].min().values} < {min_val}"
        assert (
            xds[var].max() <= max_val
        ), f"{var} max value is above limit of {xds[var].max().values} > {max_val}"
    return xds
