"""Domain geometry and grid creation utilities."""

import logging
from typing import Any, Dict, Literal

import geopandas as gpd
import numpy as np
import shapely.geometry
import xarray as xr


def create_domain_geometry(
    name: Literal["gramm", "gral"], config: Dict[str, Any]
) -> gpd.GeoDataFrame:
    """Create a GeoDataFrame representing the domain area from configuration.

    Parameters
    ----------
    name : Literal["gramm", "gral"]
        Name of the domain, either "gramm" or "gral".
    config : Dict[str, Any]
        Configuration dictionary containing domain specifications with bbox
        coordinates and CRS information.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing a single polygon geometry representing
        the domain area.
    """
    domain_area = gpd.GeoDataFrame(
        geometry=[shapely.geometry.box(*config["domain"][name]["bbox"].values())],
        crs=config["domain"]["crs"],
    )

    return domain_area


def create_domain_grid(
    name: Literal["gramm", "gral"], config: Dict[str, Any]
) -> xr.Dataset:
    """Create a GRAMM model grid based on configuration parameters.

    Parameters
    ----------
    name : Literal["gramm", "gral"]
        Name of the domain, either "gramm" or "gral".
    config : Dict[str, Any]
        Configuration dictionary containing domain specifications including bbox,
        grid spacing (dx, dy), and coordinate reference system.

    Returns
    -------
    xr.Dataset
        xarray Dataset containing grid coordinates and placeholder variables for
        both regular and staggered grids.

    Raises
    ------
    AssertionError
        If bbox coordinates are invalid (x0 >= x1 or y0 >= y1).
    """
    minx = config["domain"][name]["bbox"]["x0"]
    maxx = config["domain"][name]["bbox"]["x1"]
    miny = config["domain"][name]["bbox"]["y0"]
    maxy = config["domain"][name]["bbox"]["y1"]
    assert minx < maxx, "Invalid bbox: x0 must be less than x1"
    assert miny < maxy, "Invalid bbox: y0 must be less than y1"

    dx = config["domain"][name]["dx"]
    dy = config["domain"][name]["dy"]

    logging.info(
        f"Creating {name} grid with width {maxx - minx} m "
        f"and height {maxy - miny} m"
    )
    # Create a grid for the domain
    x_stag_coords = np.arange(minx, maxx + dx, dx)
    y_stag_coords = np.arange(miny, maxy + dy, dy)
    x_coords = x_stag_coords[:-1] + dx / 2
    y_coords = y_stag_coords[:-1] + dy / 2
    grid = xr.Dataset(
        data_vars={
            "grid_placeholder": (
                ("y", "x"),
                np.zeros((len(y_coords), len(x_coords))),
            ),
            "grid_placeholder_stag": (
                ("y_stag", "x_stag"),
                np.zeros((len(y_stag_coords), len(x_stag_coords))),
            ),
        },
        coords={
            "y": y_coords,
            "x": x_coords,
            "y_stag": y_stag_coords,
            "x_stag": x_stag_coords,
        },
    )
    grid = grid.rio.write_crs(config["domain"]["crs"])
    return grid


def gradient(elevation: xr.DataArray) -> xr.DataArray:
    """
    Calculate the gradient magnitude of elevation data.

    Parameters
    ----------
    elevation : xr.DataArray
        2D elevation data with 'x' and 'y' dimensions.

    Returns
    -------
    xr.DataArray
        Gradient magnitude array.
    """
    return xr.DataArray(np.hypot(elevation.diff("x"), elevation.diff("y")))


def smooth_elevation(
    elevation: xr.DataArray, n_grid_cells: int = 5, n_grid_cells_border: int = 10
) -> xr.DataArray:
    """
    Smooth elevation data by applying border smoothing and gradient smoothing.

    Parameters
    ----------
    elevation : xr.DataArray
        2D elevation data to be smoothed.
    n_grid_cells : int, optional
        Number of grid cells for rolling mean smoothing, by default 5.
    n_grid_cells_border : int, optional
        Number of grid cells for border smoothing, by default 10.

    Returns
    -------
    xr.DataArray
        Smoothed elevation data.
    """
    logging.info(
        f"Max gradient before smoothing: {gradient(elevation).max().values:.1f} m"
    )

    # Smooth borders
    boundary_mask = xr.ones_like(elevation)
    boundary_mask[
        {
            "x": slice(1, -1),
            "y": slice(1, -1),
        }
    ] = 0
    min_border_elevation = elevation.where(boundary_mask).min()
    logging.info(f"Min elevation at the boundaries: {min_border_elevation.values:.1f}")

    weights = xr.zeros_like(elevation)
    for i in range(1, n_grid_cells_border + 1):
        weights[
            {
                "x": slice(i, -i),
                "y": slice(i, -i),
            }
        ] = (
            i / n_grid_cells_border
        )
    # Interpolate linearly between minimum elevation
    elevation = elevation * weights + min_border_elevation * (1 - weights)

    # Smooth gradients
    elevation = elevation.rolling({"x": n_grid_cells, "y": n_grid_cells}).mean()
    logging.info(
        f"Max gradient after smoothing: {gradient(elevation).max().values:.1f} m"
    )

    # Fill NaNs
    for dim in ["x", "y"]:
        elevation = elevation.interpolate_na(
            method="nearest", dim=dim, fill_value="extrapolate"
        )
    return elevation


def simpsons_rule(array: np.ndarray, axis: tuple[int]) -> np.ndarray:
    """
    Apply Simpson's rule for 2D integration on a 2x2 DataArray.

    Parameters
    ----------
    array : np.ndarray
        m x n x 2 x 2 array.

    Returns
    -------
    np.ndarray
        Result of Simpson's rule integration.
    """
    assert len(axis) == 2, "Not two axis in reduce function."
    assert array.shape[axis[0]] == 2, "Shape of first axis is not 2."
    assert array.shape[axis[1]] == 2, "Shape of second axis is not 2."
    assert len(array.shape) == 5, "Shape of array is not (nz, ny, nx, 2, 2)"
    # Apply Simpson's rule
    weights = np.array([[2, 1], [1, 2]]).reshape((1, 1, 1, 2, 2))
    reduced = np.sum(weights * array, axis=axis) / 6
    assert reduced.shape == (array.shape[0], array.shape[1], array.shape[2]), (
        f"Reduced shape {reduced.shape} is not "
        f"(nz, ny, nx) = {(array.shape[0], array.shape[1], array.shape[2])}"
    )
    return reduced


def interp_stag_dim(da: xr.DataArray, how_dict: Dict[str, str]) -> xr.DataArray:
    """
    Interpolate staggered dimensions in a DataArray.

    Parameters
    ----------
    da : xr.DataArray
        Input data array with staggered dimensions.
    how_dict : Dict[str, str]
        Dictionary mapping dimension names to interpolation methods.
        Valid dimensions: 'z_stag', 'y_stag', 'x_stag', 'yx_stag'.
        Valid methods: 'diff', 'mean', 'simpsons'.

    Returns
    -------
    xr.DataArray
        Interpolated data array.
    """
    for dim, how in how_dict.items():
        assert dim in [
            "z_stag",
            "y_stag",
            "x_stag",
            "yx_stag",
        ], f"{dim} is not in [z_stag, y_stag, x_stag, yx_stag]"
        assert (dim != "yx_stag") or (
            how == "simpsons"
        ), "yx_stag can only be used with how='simpsons'."
        assert how in [
            "diff",
            "mean",
            "simpsons",
        ], f"{how} is not in [diff, mean, simpsons]"
        if how == "diff":
            da = da.diff(dim)
            if dim in da.coords:
                da = da.drop_vars(dim)
            da = da.rename({dim: dim[0]})
        elif how == "mean":
            da = da.rolling({dim: 2}).mean().dropna(dim)
            if dim in da.coords:
                da = da.drop_vars(dim)
            da = da.rename({dim: dim[0]})
        elif how == "simpsons":
            da = da.rolling({"y_stag": 2, "x_stag": 2}).reduce(simpsons_rule)
            da = da.isel(y_stag=slice(1, None), x_stag=slice(1, None))
            for dim in ["y_stag", "x_stag"]:
                if dim in da.coords:
                    da = da.drop_vars(dim)
                da = da.rename({dim: dim[0]})
    return da


def create_geometry_variable_specs(nx: int, ny: int, nz: int) -> dict:
    """
    Create variable specifications for GRAMM geometry file parsing.

    Parameters
    ----------
    nx : int
        Number of grid cells in x-direction.
    ny : int
        Number of grid cells in y-direction.
    nz : int
        Number of grid cells in z-direction.

    Returns
    -------
    dict
        Dictionary containing variable specifications with size, dtype, and dimensions
        for each variable in the GRAMM geometry file format.
    """
    variable_sizes_types_dimensions = {
        "NX": (1, int, ()),
        "NY": (1, int, ()),
        "NZ": (1, int, ()),
        "X": (nx + 1, np.float64, ("x_stag",)),
        "Y": (ny + 1, np.float64, ("y_stag",)),
        "Z": (nz + 1, np.float64, ("z_stag",)),
        "AH": (ny * nx, np.float64, ("y", "x")),
        "VOL": (nz * ny * nx, np.float64, ("z", "y", "x")),
        "AREAX": (nz * ny * (nx + 1), np.float64, ("z", "y", "x_stag")),
        "AREAY": (nz * (ny + 1) * nx, np.float64, ("z", "y_stag", "x")),
        "AREAZX": ((nz + 1) * ny * nx, np.float64, ("z_stag", "y", "x")),
        "AREAZY": ((nz + 1) * ny * nx, np.float64, ("z_stag", "y", "x")),
        "AREAZ": ((nz + 1) * ny * nx, np.float64, ("z_stag", "y", "x")),
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
            ("z_stag", "y_stag", "x_stag"),
        ),
    }

    return variable_sizes_types_dimensions


def test_dataset(geom: xr.Dataset, nx: int, ny: int, nz: int) -> None:
    """
    Test if dataset has correct variable sizes, dtypes, and dimensions.

    Parameters
    ----------
    geom : xr.Dataset
        Dataset to test.
    nx : int
        Number of grid cells in x-direction.
    ny : int
        Number of grid cells in y-direction.
    nz : int
        Number of grid cells in z-direction.

    Raises
    ------
    AssertionError
        If any variable has incorrect size, dtype, or dimensions.
    """
    for var, specs in create_geometry_variable_specs(nx, ny, nz).items():
        size = np.array(geom[var]).size
        assert size == specs[0], f"{var} size is {size} instead of {specs[0]}"

        dtype = geom[var].dtype
        assert dtype == specs[1], f"{var} dtype is {dtype} instead of {specs[1]}"
        if dtype == np.float64:
            assert geom[var].isnull().sum() == 0, f"{var} contains NaNs"

        dims = geom[var].dims
        assert dims == specs[2], f"{var} dims is {dims} instead of {specs[2]}"


def create_ggeom_dataset(
    elevation: xr.DataArray,
    nz: int,
    z0: float,
    vert_stretching: float,
) -> xr.Dataset:
    """
    Create a GRAMM geometry dataset from elevation data.

    Parameters
    ----------
    elevation : xr.DataArray
        2D elevation data with 'x' and 'y' coordinates.
    nz : int
        Number of vertical grid cells.
    z0 : float
        Height of the first vertical grid cell.
    vert_stretching : float
        Vertical stretching factor for grid cells.

    Returns
    -------
    xr.Dataset
        GRAMM geometry dataset with all required variables.
    """
    geom = xr.Dataset(
        coords={
            "x": elevation.x,
            "y": elevation.y,
        }
    )
    nx = elevation.x.size
    ny = elevation.y.size
    geom["NX"] = ([], nx)
    geom["NY"] = ([], ny)
    geom["NZ"] = ([], nz)

    dx = np.diff(elevation.x)[0]
    assert (dx == np.diff(elevation.x)).all(), "Grid not equally spaced in x-direction."
    dy = np.diff(elevation.y)[0]
    assert (dy == np.diff(elevation.y)).all(), "Grid not equally spaced in y-direction."
    geom["X"] = (["x_stag"], np.arange(nx + 1) * dx)
    geom["Y"] = (["y_stag"], np.arange(ny + 1) * dy)

    xmin = geom.x.min()
    ymin = geom.y.min()
    geom = geom.assign_coords(
        {
            "x_stag": xmin - dx / 2 + geom["X"],
            "y_stag": ymin - dy / 2 + geom["Y"],
        }
    )

    geom["DDX"] = (["x"], np.ones(nx) * dx)
    geom["DDY"] = (["y"], np.ones(ny) * dy)

    geom["ZAX"] = (["x"], np.ones(nx) * dx)
    geom["ZAY"] = (["y"], np.ones(ny) * dy)

    # Interpolate to staggered grid
    geom["AHE"] = (
        ["z_stag", "y_stag", "x_stag"],
        np.full((nz + 1, ny + 1, nx + 1), np.nan),
    )

    interp_elevation = elevation.rename(x="x_stag", y="y_stag").interp_like(
        geom["AHE"][{"z_stag": 0}], method="linear"
    )
    interp_elevation = interp_elevation.interpolate_na(
        dim="x_stag", method="nearest", fill_value="extrapolate"
    )
    interp_elevation = interp_elevation.interpolate_na(
        dim="y_stag", method="nearest", fill_value="extrapolate"
    )
    geom["AHE"][{"z_stag": 0}] = interp_elevation

    min_elevation = geom["AHE"][{"z_stag": 0}].min()
    max_elevation = geom["AHE"][{"z_stag": 0}].max()
    cell_height = z0 * vert_stretching ** np.arange(nz)
    max_column_height = np.sum(cell_height)

    column_height = max_column_height - (geom["AHE"][{"z_stag": 0}] - min_elevation)
    relative_column_height = column_height / max_column_height
    assert relative_column_height.max() == 1
    for idz_stag in range(1, len(geom.z_stag)):
        idz = idz_stag - 1
        lower_points = geom["AHE"][{"z_stag": idz_stag - 1}]
        geom["AHE"][{"z_stag": idz_stag}] = (
            lower_points + cell_height[idz] * relative_column_height
        )

    geom["Z"] = (["z_stag"], np.append([0], np.cumsum(cell_height)) + min_elevation)
    logging.info(
        f"Elevation minimum {min_elevation:.2f} m and maximum {max_elevation:.2f} m."
    )
    logging.info(f"Grid maximum {geom['Z'].max().values:.2f} m")

    geom["AH"] = elevation

    # AREAX calculation (x-faces)
    areax = interp_stag_dim(geom["AHE"], {"z_stag": "diff", "y_stag": "mean"}) * dy
    geom["AREAX"] = areax

    # AREAY calculation (y-faces)
    areay = interp_stag_dim(geom["AHE"], {"z_stag": "diff", "x_stag": "mean"}) * dx
    geom["AREAY"] = areay

    # AREAZX calculation (z-faces projected on x-faces)
    areazx = interp_stag_dim(geom["AHE"], {"x_stag": "diff", "y_stag": "mean"}) * dy
    geom["AREAZX"] = areazx

    # AREAZY calculation (z-faces projected on y-faces)
    areazy = interp_stag_dim(geom["AHE"], {"y_stag": "diff", "x_stag": "mean"}) * dx
    geom["AREAZY"] = areazy

    # Calculate bottom area of the grid cell
    areaz = np.sqrt(dx**2 * dy**2 + areazx**2 + areazy**2)
    geom["AREAZ"] = areaz

    # Calculate the volume of the grid cell
    # Use 2-d Simpson's rule
    vol = (
        interp_stag_dim(geom["AHE"], {"z_stag": "diff", "yx_stag": "simpsons"})
        * dx
        * dy
    )
    geom["VOL"] = vol

    # Calculate the height of the centre point of each grid cell
    zsp = interp_stag_dim(
        geom["AHE"], {"z_stag": "mean", "y_stag": "mean", "x_stag": "mean"}
    )
    geom["ZSP"] = zsp

    # Western border of model domain
    geom["IKOOA"] = ([], geom.x_stag[0].values)
    # Eastern border of model domain
    geom["JKOOA"] = ([], geom.y_stag[0].values)
    geom["WINKEL"] = ([], 0.0)
    # Reorder data variables
    keys = create_geometry_variable_specs(nx, ny, nz).keys()
    geom = geom[keys]
    test_dataset(geom, nx, ny, nz)
    return geom
