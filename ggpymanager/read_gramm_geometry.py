"""
Module Name: read_gramm_geometry.py

Description: 
    Convert to `ggeom.asc` file to an xarray dataset.


Author:
    Robert Maiwald
    Institute of Environmental Physics, Heidelberg University, Germany

Contact:
    robert.maiwald@uni-heidelberg.de

Date:
    2025-01-23
"""

import numpy as np
import xarray as xr


def read_ggeom_file(filepath: str) -> xr.Dataset:
    """
    Reads a ggeom.asc file and returns an xarray Dataset with the parsed geometry.

    Parameters
    ----------
    filepath : str
        Path to ggeom.asc file.

    Returns
    -------
    xarray.Dataset
        Dataset containing arrays for Nx, Ny, Nz, and geometry fields.
    """
    with open(filepath, "r") as f:
        tokens = f.read().split()

    # Parse header
    nx, ny, nz = map(int, tokens[:3])

    variable_sizes_types_dimensions = {
        "NX": (1, int, ()),
        "NY": (1, int, ()),
        "NZ": (1, int, ()),
        "X": (nx + 1, np.float64, ("x_staggered",)),
        "Y": (ny + 1, np.float64, ("y_staggered",)),
        "Z": (nz + 1, np.float64, ("z_staggered",)),
        "AH": (ny * nx, np.float64, ("y", "x")),
        "VOL": (nz * ny * nx, np.float64, ("z", "y", "x")),
        "AREAX": (nz * ny * (nx + 1), np.float64, ("z", "y", "x_staggered")),
        "AREAY": (nz * (ny + 1) * nx, np.float64, ("z", "y_staggered", "x")),
        "AREAZX": ((nz + 1) * ny * nx, np.float64, ("z_staggered", "y", "x")),
        "AREAZY": ((nz + 1) * ny * nx, np.float64, ("z_staggered", "y", "x")),
        "AREAZ": ((nz + 1) * ny * nx, np.float64, ("z_staggered", "y", "x")),
        "ZSP": (nz * ny * nx, np.float64, ("z", "y", "x")),
        "DDX": (nx, np.float64, ("x",)),
        "DDY": (ny, np.float64, ("y",)),
        "ZAX": (nx, np.float64, ("x",)),
        "ZAY": (ny, np.float64, ("y",)),
        "IKOOA": (1, np.float64, ()),
        "JKOOA": (1, np.float64, ()),
        "WINKEL": (1, np.float64, ()),
        "AHE": (
            (nz + 1) * (ny + 1) * (nx + 1),
            np.float64,
            ("z_staggered", "y_staggered", "x_staggered"),
        ),
    }

    vars = {}
    idx = 0
    for var, (size, dtype, _) in variable_sizes_types_dimensions.items():
        vars[var] = np.array(tokens[idx : idx + size], dtype=dtype)
        idx += size

    dimensions = {
        "x": nx,
        "y": ny,
        "z": nz,
        "x_staggered": nx + 1,
        "y_staggered": ny + 1,
        "z_staggered": nz + 1,
        "": None,
    }
    ds = xr.Dataset(
        {
            var: (dims, vars[var].reshape([dimensions[dim] for dim in dims]))
            for var, (_, _, dims) in variable_sizes_types_dimensions.items()
        },
        coords={
            "x": vars["X"][:-1] + vars["DDX"] / 2,
            "x_staggered": vars["X"],
            "y": vars["Y"][:-1] + vars["DDY"] / 2,
            "y_staggered": vars["Y"],
        },
    )
    # Add attributes to data_vars
    variable_long_names_units = {
        "NX": ("number of cells in x-direction", ""),
        "NY": ("number of cells in y-direction", ""),
        "NZ": ("number of cells in z-direction", ""),
        "X": ("Distance of grid cell centre from eastern domain border", "m"),
        "Y": ("Distance of grid cell centre from southern domain border", "m"),
        "Z": ("Temporary array for generating the terrain-following grid", "m"),
        "AH": ("Height of the surface", "m"),
        "VOL": ("Volume of grid cells", "m^3"),
        "AREAX": ("Area of the grid cell in x-direction", "m^2"),
        "AREAY": ("Area of the grid cell in y-direction", "m^2"),
        "AREAZX": (
            "Projection of the ground area of the grid cell in x-direction",
            "m^2",
        ),
        "AREAZY": (
            "Projection of the ground area of the grid cell in y-direction",
            "m^2",
        ),
        "AREAZ": ("Bottom area of the grid cell", "m^2"),
        "ZSP": ("Height of the centre point of each grid cell", "m"),
        "DDX": ("Horizontal grid size in x-direction", "m"),
        "DDY": ("Horizontal grid size in y-direction", "m"),
        "ZAX": ("Distance between neighbouring grid cells in x-direction", "m"),
        "ZAY": ("Distance between neighbouring grid cells in y-direction", "m"),
        "IKOOA": ("Western border of model domain", "m"),
        "JKOOA": ("Southern border of model domain", "m"),
        "WINKEL": ("... (not used anymore)", ""),
        "AHE": ("Heights of the corner points of each grid cell", "m"),
    }

    for var, (long_name, units) in variable_long_names_units.items():
        ds[var].attrs["long_name"] = long_name
        ds[var].attrs["units"] = units

    return ds


def main():
    """Main function."""
    print(__doc__)


if __name__ == "__main__":
    main()
