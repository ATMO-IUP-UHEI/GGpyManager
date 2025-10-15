"""
Utility functions partly adapted from Ivo Suter.
"""

import inspect
import logging
import re
import struct
import zipfile
from dataclasses import dataclass, field
from functools import wraps
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from scipy import sparse


def set_logger(level="INFO"):
    """
    Set up the logging configuration.

    Parameters
    ----------
    level : str, optional
        Logging level. Default is 'INFO'.
    """
    logging.basicConfig(
        level=getattr(
            logging, level.upper(), logging.INFO
        ),  # Use the function argument
        format="%(levelname)s: %(message)s",
        force=True,  # Force reset of logging settings
    )
    logging.info("Logger set up.")


def check_docstring_dims(func):
    """
    Decorator to check if the dimensions of xr.DataArray arguments and return values
    match the docstring specification.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Retrieve function signature and bound arguments
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Retrieve docstring
        docstring = inspect.getdoc(func)
        if not docstring:
            return func(*args, **kwargs)  # Skip check if no docstring

        # Extract expected input dimensions
        param_pattern = re.compile(r"(\w+)\s*:\s*xr\.DataArray\s*\((.*?)\)")
        expected_dims = {
            match[0]: tuple(match[1].split(", "))
            for match in param_pattern.findall(docstring)
        }

        # Validate input arguments
        for arg_name, expected_dim in expected_dims.items():
            if arg_name in bound_args.arguments:
                arg_value = bound_args.arguments[arg_name]
                if isinstance(arg_value, xr.DataArray):
                    actual_dim = arg_value.dims
                    assert (
                        actual_dim == expected_dim
                    ), f"Argument '{arg_name}' expected dimensions {expected_dim}, "
                    f"but got {actual_dim}."

        # Execute the function
        result = func(*args, **kwargs)

        # Extract expected output dimensions
        return_pattern = re.compile(
            r"Returns\s*\n[-]+\n\s*(\w+)\s*:\s*xr\.DataArray\s*\((.*?)\)"
        )
        match = return_pattern.search(docstring)
        if match:
            return_var, expected_return_dim = match.groups()
            expected_return_dim = tuple(expected_return_dim.split(", "))

            # Validate return value
            if isinstance(result, xr.DataArray):
                actual_return_dim = result.dims
                assert actual_return_dim == expected_return_dim, (
                    f"Function '{func.__name__}' "
                    f"expected return dimensions {expected_return_dim}, "
                    f"but got {actual_return_dim}."
                )

        return result  # Return function output as usual

    return wrapper


def direction_from_vector(ux, vy):
    """
    Calculate the wind direction from the vector components.

    Parameters
    ----------
    ux : array_like
        Eastward component of the wind vector.
    vy : array_like
        Northward component of the wind vector.

    Returns
    -------
    direction : array_like
        Wind direction in degrees.
    """
    return (90 - np.rad2deg(np.arctan2(-vy, -ux))) % 360


def wind_speed_from_vector(ux, vy):
    """
    Calculate the wind speed from the vector components.

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


def vector_from_direction_and_speed(direction, speed):
    """
    Calculate the vector components from the wind direction and speed.

    Parameters
    ----------
    direction : array_like
        Wind direction in degrees.
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


def create_domain_geometry(
    name: Literal["gramm", "gral"], config: Dict[str, Any]
) -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame representing the domain area from configuration.

    Parameters
    ----------
    name : Literal["gramm", "gral"]
        Name of the domain, either "gramm" or "gral".
    config : Dict[str, Any]
        Configuration dictionary containing domain specifications with bbox coordinates
        and CRS information.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame containing a single polygon geometry representing the domain area.
    """
    domain_area = gpd.GeoDataFrame(
        geometry=[shapely.geometry.box(*config["domain"][name]["bbox"].values())],
        crs=config["domain"]["crs"],
    )

    return domain_area


def create_domain_grid(
    name: Literal["gramm", "gral"], config: Dict[str, Any]
) -> xr.Dataset:
    """
    Create a GRAMM model grid based on configuration parameters.

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
        f"Creating {name} grid with width {maxx - minx} m and height {maxy - miny} m"
    )
    # Create a grid for the domain
    x_stag_coords = np.arange(minx, maxx + dx, dx)
    y_stag_coords = np.arange(miny, maxy + dy, dy)
    x_coords = x_stag_coords[:-1] + dx / 2
    y_coords = y_stag_coords[:-1] + dy / 2
    grid = xr.Dataset(
        data_vars={
            "grid_placeholder": (("y", "x"), np.zeros((len(y_coords), len(x_coords)))),
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


@dataclass
class GRAMM:
    """
    All constants related to the GRAMM domain.

    Parameters
    ----------
    nx : int
        Number of cells in x-direction.
    ny : int
        Number of cells in y-direction.
    nz : int
        Number of cells in z-direction.
    dx
        Cell width in x-direction in [m].
    dy
        Cell width in y-direction in [m].
    xmin
        x-coordinate of the left-lower (south-west) corner of the GRAMM domain in
        Gauß-Krueger coordinates.
    ymin
        y-coordinate of the left-lower (south-west) corner of the GRAMM domain in
        Gauß-Krueger coordinates.
    """

    nx: int = 200
    ny: int = 201
    nz: int = 22
    xmin: int = 3466700
    ymin: int = 5465000
    xmax: int = 3486700
    ymax: int = 5485000
    dx: float = (xmax - xmin) / nx
    dy: float = (ymax - ymin) / ny
    xmesh, ymesh = np.meshgrid(xmin + dx * np.arange(nx), ymin + dy * np.arange(ny))
    xcmesh, ycmesh = np.meshgrid(
        xmin + dx * np.arange(nx + 1), ymin + dy * np.arange(ny + 1)
    )


@dataclass
class GRAL:
    """
    All constants related to the GRAL domain.

    Parameters
    ----------
    nx : int
        Number of cells in x-direction.
    ny : int
        Number of cells in y-direction.
    nz : int
        Number of cells in z-direction.
    dx
        Cell width in x-direction in [m].
    dy
        Cell width in y-direction in [m].
    xmin
        x-coordinate of the left-lower (south-west) corner of the GRAL domain in
        Gauß-Krueger coordinates.
    ymin
        y-coordinate of the left-lower (south-west) corner of the GRAL domain in
        Gauß-Krueger coordinates.
    """

    nx: int = 1227
    ny: int = 1232
    nz: int = 400
    dx: float = 10.0
    dy: float = 10.0
    xmin: int = 3471259
    ymin: int = 5468979
    xmesh, ymesh = np.meshgrid(xmin + dx * np.arange(nx), ymin + dy * np.arange(ny))
    xcmesh, ycmesh = np.meshgrid(
        xmin + dx * np.arange(nx + 1), ymin + dy * np.arange(ny + 1)
    )


def read_gral_config(gral_geb_path: Path, in_dat_path: Path) -> dict:
    with (gral_geb_path).open("r") as f:
        lines = f.readlines()
    lines = [line.split("!")[0].strip() for line in lines]
    config = {
        "dx": float(lines[0]),
        "dy": float(lines[1]),
        "dz0": float(lines[2].split(",")[0]),
        "stretching_factor": float(lines[2].split(",")[1]),
        "nx": int(lines[3]),
        "ny": int(lines[4]),
        "n_horizontal_slices_concentration": int(lines[5]),
        "source_groups": [int(number) for number in lines[6].split(",")],
        "west_border": float(lines[7]),
        "east_border": float(lines[8]),
        "south_border": float(lines[9]),
        "north_border": float(lines[10]),
    }
    with (in_dat_path).open("r") as f:
        lines = f.readlines()
    lines = [line.split("!")[0].strip() for line in lines]
    config |= {
        "particle_number": int(lines[0]),
        "dispersion_time": float(lines[1]),
        "steady_state": bool(int(lines[2])),
        "horizontal_slices_concentration": [
            float(number) for number in lines[9].split(",")
        ],
        "vertical_grid_spacing_concentration": float(lines[10]),
    }
    return config


LANDUSE_VARS = ["RHOB", "ALAMBDA", "Z0", "FW", "EPSG", "ALBEDO"]


def read_landuse(path, shape):
    landuse_data = {}
    with open(path) as f:
        for line, name in zip(f, LANDUSE_VARS):
            landuse_data[name] = np.array([float(d) for d in line.split()]).reshape(
                shape
            )
    return landuse_data


def write_landuse(path, landuse_data: xr.Dataset):
    with open(path, "w") as f:
        for name in LANDUSE_VARS:
            arr = landuse_data[name].values.reshape(-1)
            line = " ".join(str(x) for x in arr)
            f.write(line + "\n")


def read_topography(path):
    f = open(path, "r")
    data = f.readlines()
    f.close()
    tmp = data[1].split()
    topo_ind = 0
    topo = np.zeros((GRAMM.nx, GRAMM.ny), np.float64)
    for j in range(GRAMM.ny):
        for i in range(GRAMM.nx):
            topo[i, j] = float(tmp[topo_ind])
            topo_ind += 1

    # Z Grid
    tmp = data[8].split()
    zgrid = np.zeros([GRAMM.nx, GRAMM.ny, GRAMM.nz], np.float64)
    ind = 0
    for k in range(GRAMM.nz):
        for j in range(GRAMM.ny):
            for i in range(GRAMM.nx):
                zgrid[i, j, k] = float(tmp[ind])
                ind += 1

    return topo, zgrid


def read_gramm_windfield(path):
    with path.open("rb") as f:
        data = f.read()

    nheader = 20  # header, ni,nj,nz,gridsize -> 4*signed integer (=4*4) + float (4)
    header, nx, ny, nz, dx = struct.unpack("<iiiif", data[:nheader])

    dt = np.dtype(np.short)

    info = np.frombuffer(data[: nheader - 4], dtype=np.int32)
    # grid_size = np.frombuffer(data[nheader - 4 : nheader], dtype=np.float32)
    datarr = np.frombuffer(data[nheader:], dtype=dt)
    datarr = np.reshape(datarr, [nx, ny, nz, 3])
    wind_u = datarr[:, :, :, 0] * 0.01
    wind_v = datarr[:, :, :, 1] * 0.01
    wind_w = datarr[:, :, :, 2] * 0.01
    # umag = np.hypot(wind_u[:, :, :], wind_v[:, :, :])

    return info, wind_u, wind_v, wind_w


def read_buildings(path, GRAL=GRAL):
    data = np.genfromtxt(path, delimiter=",")

    buildings = np.zeros((GRAL.nx, GRAL.ny))

    x = data[:, 0]
    y = data[:, 1]

    idx = ((x - GRAL.xmin) / GRAL.dx).astype(int)
    idy = ((y - GRAL.ymin) / GRAL.dy).astype(int)

    buildings[idx, idy] = data[:, 3]
    return buildings


def write_buildings_file(path, building_height: xr.DataArray) -> None:
    stacked = building_height.stack(position=("x", "y"))
    stacked = stacked[~stacked.isnull()]
    logging.info(f"Writing building file to {path}...")
    with open(path, "w") as f:
        for x, y, h in zip(stacked.x.values, stacked.y.values, stacked.values):
            f.write(f"{x},{y},0,{h}\n")


def read_gral_geometries(
    path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads GRAL geometry data from a binary file and returns surface elevation (ahk),
    surface index (kkart), building height (bui_height), and orography (oro).

    Parameters
    ----------
    path : str
        Path to the binary geometry file.

    Returns
    -------
    ahk : np.ndarray
        2D array of surface elevations.
    kkart : np.ndarray
        2D array of surface indices (int).
    bui_height : np.ndarray
        2D array of building heights.
    oro : np.ndarray
        2D array of orography (surface elevation minus building height).
    """
    with open(path, mode="rb") as binfile:
        byte_list = binfile.read()

    nheader = 32
    header = byte_list[:nheader]
    nz, ny, nx, ikooagral, jkooagral, dzk, stretch, ahmin = struct.unpack(
        "iiiiifff", header
    )

    # Read geometry buffer efficiently
    blub = byte_list[nheader:]

    # Each data record: float32, int32, float32 (12 bytes per cell)
    num_cells = (nx + 1) * (ny + 1)
    arr = np.frombuffer(
        blub,
        dtype=[("ahk", "f4"), ("kkart", "i4"), ("bui_height", "f4")],
        count=num_cells,
    )
    arr_reshaped = arr.reshape((nx + 1, ny + 1))

    # Remove zero-padding at upper bounds (last row and column)
    ahk = arr_reshaped[:-1, :-1]["ahk"]
    kkart = arr_reshaped[:-1, :-1]["kkart"]
    bui_height = arr_reshaped[:-1, :-1]["bui_height"]
    oro = ahk - bui_height

    return ahk, kkart, bui_height, oro


def read_gral_windfield(path):
    # Read file data
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path, "r") as gff:
            filename = gff.namelist()[0]  # Get first file directly
            byte_list = gff.read(filename)
    else:
        with open(path, mode="rb") as binfile:
            byte_list = binfile.read()

    # Parse header
    nheader = 32
    nz, ny, nx, direction, speed, akla, dxy, h = struct.unpack(
        "iiiffifi", byte_list[:nheader]
    )
    direction = 270.0 - np.rad2deg(direction)

    # Parse data directly with proper shape
    count = (nx + 1) * (ny + 1) * (nz + 1) * 3
    data = np.frombuffer(byte_list, dtype=np.int16, count=count, offset=nheader)

    # Reshape and slice in one operation
    data = data.reshape(nx + 1, ny + 1, nz + 1, 3)

    # Use views instead of copies, multiply in-place
    ux = data[:-1, :-1, 1:, 0].astype(np.float32) * 0.01
    vy = data[:-1, :-1, 1:, 1].astype(np.float32) * 0.01
    wz = data[:-1, :-1, 1:, 2].astype(np.float32) * 0.01

    return ux, vy, wz


def read_con_file(path, GRAL=GRAL):
    with path.open("rb") as f:
        data = f.read()

    if len(data) <= 4:
        return None

    # Define structured dtype for 'iif' format
    dt = np.dtype([("x", "<i4"), ("y", "<i4"), ("val", "<f4")])
    datarr = np.frombuffer(data[4:], dtype=dt)

    con = np.zeros((GRAL.nx, GRAL.ny), dtype=np.float32)

    # Compute indices directly
    idx = ((datarr["x"] - GRAL.xmin) / GRAL.dx + 0.5).astype(int)
    idy = ((datarr["y"] - GRAL.ymin) / GRAL.dy + 0.5).astype(int)

    con[idx, idy] = datarr["val"]

    return con


def con_file_as_sparse_matrix(path, GRAL=GRAL):
    con = read_con_file(path, GRAL)
    return sparse.csr_matrix(con)


def read_gral_concentration(path):
    con_dict = {}
    conc_file = np.load(path, allow_pickle=True)
    for key in conc_file:
        con_matrix = conc_file[key].item().toarray()
        con_dict[key] = con_matrix
    return con_dict
    # con_dict = {}
    # with np.load(path, allow_pickle=True, mmap_mode="r") as conc_file:
    #     for key in conc_file:
    #         con_matrix = conc_file[key].all().toarray()
    #         con_dict[key] = con_matrix
    # return con_dict


@check_docstring_dims
def get_allowed_stability_class(radiation, wind_speed, stab_class_catalog):
    """
    Get allowed stability classes for each time step based on radiation and wind speed.

    Parameters
    ----------
    radiation : xr.DataArray (time)
        Radiation data with time dimension.
    wind_speed : xr.DataArray (sim_id)
        Wind speed data with sim_id dimension.
    stab_class_catalog : xr.DataArray (sim_id)
        Stability class data with sim_id dimension.

    Returns
    -------
    stability_class_mask : xr.DataArray (time, sim_id)
        Binary array with one-hot encoding for the allowed stability classes at each
        time step.
    """

    radiation_index = xr.zeros_like(radiation, dtype=int)
    wind_speed_index = xr.zeros_like(wind_speed, dtype=int)
    catalog_filter = load_catalog_filter()
    min_rads = catalog_filter.columns.get_level_values(1).astype(float)[::-1]

    # Select bin for radiation for each time step
    for i, min_rad in enumerate(min_rads):
        above_rad_threshold = radiation >= min_rad
        radiation_index[above_rad_threshold] = i
        logging.info(
            "Radiation larger {:>5} W/m²: {:>4} entries".format(
                min_rad, above_rad_threshold.sum().values
            )
        )

    # Select bin for wind speed for each time step
    for i, min_wind_speed in enumerate(catalog_filter.index):
        above_wind_threshold = wind_speed >= float(min_wind_speed)
        wind_speed_index[above_wind_threshold] = i
        logging.info(
            "Wind speed larger {} m/s: {:>4} entries".format(
                min_wind_speed, above_wind_threshold.sum().values
            )
        )

    # Get stability class(es) for each time step (dims: sim_id, time)
    allowed_stab_classes = catalog_filter.values[:, radiation_index.values][
        wind_speed_index.values
    ].astype(str)
    allowed_stab_classes = xr.DataArray(
        allowed_stab_classes,
        dims=["sim_id", "time"],
        coords={
            "sim_id": wind_speed.sim_id,
            "time": radiation.time,
        },
    )

    # Convert stab_class_catalog to string
    stab_class_as_str = stab_class_catalog.astype(str)
    stab_index = ["A", "B", "C", "D", "E", "F", "G"]
    for stab, index in zip(stab_index, range(1, 8)):
        is_index = stab_class_catalog == index
        stab_class_as_str[is_index] = stab

    # Create empty mask
    stability_class_mask = xr.DataArray(
        np.zeros((len(radiation), len(wind_speed)), dtype=bool),
        dims=["time", "sim_id"],
        coords={
            "time": radiation.time,
            "sim_id": wind_speed.sim_id,
        },
    )

    # Check if stability class is allowed
    for i, stab in enumerate(stab_index):
        is_stab = stab_class_as_str == stab
        is_allowed_stability = np.strings.find(allowed_stab_classes, stab) >= 0
        stability_class_mask[dict(sim_id=is_stab)] = is_allowed_stability[
            dict(sim_id=is_stab)
        ]  # type: ignore
    return stability_class_mask


def load_catalog_filter():
    file_path = str(
        resources.files("ggpymanager.data").joinpath("catalogue_filter.csv")
    )
    return pd.read_csv(
        file_path,
        comment="#",
        header=[0, 1],
        index_col=0,
    )


@check_docstring_dims
def rmse_loss(u, v, u_model, v_model):
    """
    Input for N hours, S stations, and M catalog entries.

    Parameters
    ----------
    u : xr.DataArray (time, station)
        Hourly wind speed in x-direction from measurements.
    v : xr.DataArray (time, station)
        Hourly wind speed in y-direction from measurements.
    u_model : xr.DataArray (sim_id, station)
        Hourly wind speed in x-direction from catalog.
    v_model : xr.DataArray (sim_id, station)
        Hourly wind speed in y-direction from catalog.

    Returns
    -------
    rmse : xr.DataArray (time, sim_id)
        Root mean squared error for each hour and catalog entry.
    """
    return np.sqrt(((u - u_model) ** 2 + (v - v_model) ** 2).mean(dim="station"))


@check_docstring_dims
def regularized_loss(u, v, u_model, v_model):
    """
    Input for N hours, S stations, and M catalog entries.

    Loss function based on: Berchet, Antoine, Katrin Zink, Clive Muller, Dietmar Oettl,
    Juerg Brunner, Lukas Emmenegger, and Dominik Brunner. 2017. ‘A Cost-Effective Method
    for Simulating City-Wide Air Flow and Pollutant Dispersion at Building Resolving
    Scale’. Atmospheric Environment 158:181–96.
    https://doi.org/10.1016/j.atmosenv.2017.03.030.


    Parameters
    ----------
    u : xr.DataArray (time, station)
        Hourly wind speed in x-direction from measurements.
    v : xr.DataArray (time, station)
        Hourly wind speed in y-direction from measurements.
    u_model : xr.DataArray (sim_id, station)
        Hourly wind speed in x-direction from catalog.
    v_model : xr.DataArray (sim_id, station)
        Hourly wind speed in y-direction from catalog.

    Returns
    -------
    regularized_loss : xr.DataArray (time, sim_id)
        Regularized loss for each hour and catalog entry.
    """
    # Create station weights
    wind_speed = np.sqrt(u**2 + v**2)
    wind_speed_min = 2  # m/s
    std_wind_speed = wind_speed.std(dim="time")
    station_weight = std_wind_speed * wind_speed.clip(wind_speed_min)

    # Correlation between hours
    sigma = 1.0  # correlation length in hours
    compute_range = 3  # hours

    wind_speed_difference = np.sqrt((u - u_model) ** 2 + (v - v_model) ** 2)
    loss_per_hour = (wind_speed_difference / station_weight).mean(dim="station")
    time_difference = np.arange(-compute_range, compute_range + 1)
    window = xr.DataArray(
        np.exp(-(time_difference**2) / (sigma**2)),
        dims="window",
        coords={"window": time_difference},
    )
    loss = loss_per_hour.rolling(
        time=2 * compute_range + 1,
        min_periods=1,
        center=True,
    ).construct("window")
    loss = loss * window
    loss = loss.sum(dim="window") / window.sum()
    return loss


@check_docstring_dims
def compound_loss(u, v, u_model, v_model, lambda_=0.7):
    """
    Input for N hours, S stations, and M catalog entries.

    Loss function based on:

    Parameters
    ----------
    u : xr.DataArray (time, station)
        Hourly wind speed in x-direction from measurements.
    v : xr.DataArray (time, station)
        Hourly wind speed in y-direction from measurements.
    u_model : xr.DataArray (sim_id, station)
        Hourly wind speed in x-direction from catalog.
    v_model : xr.DataArray (sim_id, station)
        Hourly wind speed in y-direction from catalog.
    lambda_ : float, optional
        Regularization parameter. Default is 0.7.

    Returns
    -------
    compound_loss : xr.DataArray (time, sim_id)
        Compound loss for each hour and catalog entry.
    """
    direction = direction_from_vector(u, v)
    wind_speed = wind_speed_from_vector(u, v)
    direction_model = direction_from_vector(u_model, v_model)
    wind_speed_model = wind_speed_from_vector(u_model, v_model)

    # Difference in direction
    direction_difference = np.abs(direction - direction_model)
    direction_difference = np.minimum(direction_difference, 360 - direction_difference)
    direction_difference = direction_difference / 180

    # Difference in wind speed
    wind_speed_difference = np.abs(wind_speed - wind_speed_model)
    weight = 1 / wind_speed.clip(wind_speed.median()).mean(dim="time")
    wind_speed_difference = wind_speed_difference * weight
    wind_speed_difference = wind_speed_difference / wind_speed_difference.max()

    compound_loss = (
        lambda_ * direction_difference + (1 - lambda_) * wind_speed_difference
    ).sum(dim="station")
    return compound_loss


@check_docstring_dims
def compute_matching_loss(
    u,
    v,
    u_model,
    v_model,
    matching="rmse",
    filter=False,
    synoptic_wind_speed=None,
    global_radiation=None,
    stab_class_catalog=None,
):
    """
    Input for N hours, S stations, and M catalog entries.

    Parameters
    ----------
    u : xr.DataArray (time, station)
        Hourly wind speed in x-direction from measurements.
    v : xr.DataArray (time, station)
        Hourly wind speed in y-direction from measurements.
    u_model : xr.DataArray (sim_id, station)
        Hourly wind speed in x-direction from catalog.
    v_model : xr.DataArray (sim_id, station)
        Hourly wind speed in y-direction from catalog.
    matching : str, optional
        Matching loss function. Default is 'rmse'. Options are 'rmse', 'regularized',
        and 'compound'.
    filter : bool, optional
        Whether to filter the matching results. Default is False.
    synoptic_wind_speed : xr.DataArray, optional
        Synoptic wind speed data with sim_id dimension required for filtering.
    global_radiation : xr.DataArray, optional
        Global radiation data with time dimension required for filtering.
    stab_class_catalog : xr.DataArray, optional
        Stability class data with sim_id dimension required for filtering.

    Returns
    -------
    matching_loss : xr.DataArray (time, sim_id)
        Matching loss for each hour and catalog entry.
    """
    logging.info(f"Computing matching with {matching}...")
    loss_funcs = {
        "rmse": rmse_loss,
        "regularized": regularized_loss,
        "compound": compound_loss,
    }
    # Compute matching loss
    matching_loss = loss_funcs[matching](u, v, u_model, v_model)
    # Filter results
    if filter:
        stab_mask = get_allowed_stability_class(
            global_radiation, synoptic_wind_speed, stab_class_catalog
        )
        matching_loss = matching_loss.where(stab_mask)
        logging.info(
            "Filtered {:.1f} % of the results.".format(stab_mask.mean().values * 100)
        )

    # Add metadata
    matching_loss.name = f"{matching}_loss"
    matching_loss.attrs["matching"] = matching
    matching_loss.attrs["filter"] = str(filter)
    matching_loss.attrs["long_name"] = f"{matching} loss"
    matching_loss.attrs["units"] = ""
    return matching_loss


@dataclass
class GRALLogMetadata:
    version: Optional[str] = None
    plattform: Optional[str] = None
    dotnet_version: Optional[str] = None
    n_source_groups: Optional[int] = None
    ggeom_file_read: bool = False
    gral_topofile_read: bool = False
    building_file_read: bool = False
    n_horizontal_slices: Optional[int] = None
    point_emissions_read: bool = False
    n_point_emissions: Optional[int] = None
    total_point_emissions: Optional[float] = None
    line_emissions_read: bool = False
    n_line_emissions: Optional[int] = None
    total_line_emissions: Optional[float] = None
    area_emissions_read: bool = False
    n_area_emissions: Optional[int] = None
    total_area_emissions: Optional[float] = None
    advection_computated: bool = False
    numerical_stabilities: List[float] = field(default_factory=list)
    obukhov_length: Optional[float] = None
    boundary_layer_height: Optional[float] = None
    friction_velocity: Optional[float] = None
    init_wind_speed: Optional[float] = None
    init_direction: Optional[float] = None
    init_stability_class: Optional[float] = None
    gramm_wind_speed: Optional[float] = None
    gramm_direction: Optional[float] = None
    gramm_stability_class: Optional[float] = None
    total_simulation_time: Optional[float] = None
    dispersion_time: Optional[float] = None
    flow_field_time: Optional[float] = None


def parse_emission_data(l_iter, has_parentheses=False):
    """Helper function to parse emission data."""
    n_line = next(l_iter).split(":")[1]
    if has_parentheses:
        n_emissions = int(n_line.split("(")[0])
    else:
        n_emissions = int(n_line)

    total_line = next(l_iter).split(":")[1]
    total_emissions = float(total_line.split("(")[0])

    return n_emissions, total_emissions


def parse_meteo_data(line):
    """Helper function to parse meteorological data."""
    parts = line.split(":")
    return {
        "wind_speed": float(parts[2].split()[0]),
        "direction": float(parts[3].split()[0]),
        "stability_class": float(parts[4].split()[0]),
    }


def filter_lines(raw_lines):
    lines = []
    for line in raw_lines:
        line = line.strip().lstrip("0: ")
        line = line.strip("|<>+- ")
        if line != "":
            lines.append(line)
    return lines


def read_gral_stdout(path: str) -> GRALLogMetadata:
    with Path(path).open() as f:
        raw_lines = f.readlines()

    lines = filter_lines(raw_lines)
    lm = GRALLogMetadata()

    for i, l in enumerate(lines):
        l_iter = iter(lines[i + 1 :])

        match True:
            case _ if "VERSION" in l:
                lm.version = l
                lm.plattform = next(l_iter)
                lm.dotnet_version = next(l_iter)

            case _ if "Source group count:" in l:
                lm.n_source_groups = int(l.split(":")[1])

            case _ if "Reading GRAMM orography" in l:
                lm.ggeom_file_read = True

            case _ if "Reading GRAL_topofile" in l:
                lm.gral_topofile_read = True

            case _ if "Reading building file" in l:
                lm.building_file_read = True

            case _ if "Total number of horizontal slices for concentration grid:" in l:
                n = int(l.split(":")[1])
                lm.n_horizontal_slices = n
                slice_height = []
                for j in range(n):
                    slice_height.append(int(next(l_iter).split(":")[1]))

            case _ if "Reading file point.dat" in l:
                lm.point_emissions_read = True
                lm.n_point_emissions, lm.total_point_emissions = parse_emission_data(
                    l_iter, has_parentheses=False
                )

            case _ if "Reading file line.dat" in l:
                lm.line_emissions_read = True
                lm.n_line_emissions, lm.total_line_emissions = parse_emission_data(
                    l_iter, has_parentheses=True
                )

            case _ if "Reading file cadastre.dat" in l:
                lm.area_emissions_read = True
                lm.n_area_emissions, lm.total_area_emissions = parse_emission_data(
                    l_iter, has_parentheses=False
                )

            case _ if "ADVECTION" in l:
                lm.advection_computated = True
                lm.numerical_stabilities = []
                next_l = next(l_iter)
                while next_l.startswith("ITERATION"):
                    lm.numerical_stabilities.append(float(next_l.split(":")[1]))
                    next_l = next(l_iter)

            case _ if "Obukhov length" in l:
                lm.obukhov_length = float(l.split(":")[1])

            case _ if "Friction velocity" in l:
                lm.friction_velocity = float(l.split(":")[1])

            case _ if "Boundary layer height" in l:  # Fixed duplicate "Obukhov length"
                lm.boundary_layer_height = float(l.split(":")[1])

            case _ if "Init meteo:" in l:
                meteo_data = parse_meteo_data(l)
                lm.init_wind_speed = meteo_data["wind_speed"]
                lm.init_direction = meteo_data["direction"]
                lm.init_stability_class = meteo_data["stability_class"]

            case _ if "GRAMM meteo:" in l:
                meteo_data = parse_meteo_data(l)
                lm.gramm_wind_speed = meteo_data["wind_speed"]
                lm.gramm_direction = meteo_data["direction"]
                lm.gramm_stability_class = meteo_data["stability_class"]

            case _ if "Total simulation time" in l:
                lm.total_simulation_time = float(l.split(":")[1])
                lm.dispersion_time = float(next(l_iter).split(":")[1])
                lm.flow_field_time = float(next(l_iter).split(":")[1])

    return lm


@dataclass
class GRAMMLogMetadata:
    version: Optional[str] = None
    plattform: Optional[str] = None
    dotnet_version: Optional[str] = None
    n_processors: Optional[int] = None
    ggeom_file_read: bool = False
    min_elevation: Optional[float] = None
    max_elevation: Optional[float] = None
    init_wind_speed: Optional[float] = None
    init_direction: Optional[float] = None
    u_component: Optional[float] = None
    v_component: Optional[float] = None
    init_stability_class: Optional[float] = None
    init_obukhov_length: Optional[float] = None
    roughness_length: Optional[float] = None
    init_boundary_layer_height: Optional[float] = None
    friction_velocity: Optional[float] = None
    simulation_attempt: List[int] = field(default_factory=list)
    simulation_time: List[float] = field(default_factory=list)
    simulation_timestep: List[float] = field(default_factory=list)
    simulation_divergence: List[float] = field(default_factory=list)


def read_gramm_stdout(path: str) -> GRAMMLogMetadata:
    with Path(path).open() as f:
        raw_lines = f.readlines()

    lines = filter_lines(raw_lines)
    lm = GRAMMLogMetadata()

    for i, l in enumerate(lines):
        l_iter = iter(lines[i + 1 :])

        match True:
            case _ if "VERSION" in l:
                lm.version = l
                lm.plattform = next(l_iter)
                lm.dotnet_version = next(l_iter)

            case _ if "maximum degree of parallelism" in l:
                lm.n_processors = int(l.split(":")[1].split()[0])

            case _ if "Reading ggeom.asc" in l:
                lm.ggeom_file_read = True
                lm.min_elevation = float(l.split(":")[1].split()[0].rstrip("m"))
                lm.max_elevation = float(l.split(":")[1].split()[1].rstrip("m"))

            case _ if "Wind direction" in l:
                lm.init_direction = float(l.split(":")[1])

            case _ if "Wind speed" in l:
                lm.init_wind_speed = float(l.split(":")[1].rstrip("m/s"))

            case _ if "U-component" in l:
                lm.u_component = float(l.split(":")[1].rstrip("m/s"))

            case _ if "V-component" in l:
                lm.v_component = float(l.split(":")[1].rstrip("m/s"))

            case _ if "Stability class" in l:
                lm.init_stability_class = float(l.split(":")[1])

            case _ if "Obukhov length" in l:
                lm.init_obukhov_length = float(l.split(":")[1].rstrip("m"))

            case _ if "Roughness length" in l:
                lm.roughness_length = float(l.split(":")[1].rstrip("m"))

            case _ if "Boundary-Layer height" in l:
                lm.init_boundary_layer_height = float(l.split(":")[1].rstrip("m"))

            case _ if "Friction velocity" in l:
                lm.friction_velocity = float(l.split(":")[1].rstrip("m/s"))

            case _ if "WEATHER-SIT." in l:
                next_l = next(l_iter)
                lm.simulation_attempt.append(int(next_l.split()[0].split("/")[1]))
                lm.simulation_time.append(float(next_l.split()[1]))
                lm.simulation_timestep.append(float(next_l.split()[2]))
                lm.simulation_divergence.append(float(next_l.split()[5]))
    return lm


def read_esri_ascii(path: str | Path) -> tuple[np.ndarray, dict]:
    """
    Reads an ESRI ASCII raster file and returns the data as a numpy array and metadata.

    Parameters
    ----------
    path : str or Path
        Path to the ESRI ASCII file.

    Returns
    -------
    data : np.ndarray
        2D array of raster values.
    metadata : dict
        Dictionary containing header information (ncols, nrows, xllcorner, yllcorner,
        cellsize, NODATA_value).
    """
    with open(path, "r") as f:
        header = {}
        for _ in range(6):
            line = f.readline()
            key, value = line.strip().split()
            header[key.lower()] = float(value) if "." in value else int(value)
        data = np.loadtxt(f)
        assert data.shape == (header["nrows"], header["ncols"])
    return data, header


def write_esri_ascii(path: str | Path, data: xr.DataArray) -> None:
    if Path(path).exists():
        logging.warning(f"File {path} already exists!")
        raise FileExistsError
    ncols, nrows = data.sizes["x"], data.sizes["y"]
    xllcorner = data.x.min().values - 5
    yllcorner = data.y.min().values - 5
    cellsize = data.x.diff("x").mean().values
    nodata_value = -9999
    header = (
        "NCOLS {}\n"
        "NROWS {}\n"
        "XLLCORNER {}\n"
        "YLLCORNER {}\n"
        "CELLSIZE {}\n"
        "NODATA_VALUE {}\n"
    ).format(ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value)
    with open(path, "w") as f:
        f.write(header)
        for i in range(nrows):
            # Start backwards to have the origin at the lower left corner
            idy = nrows - i - 1
            f.write(" ".join(data[dict(y=idy)].values.astype(int).astype(str)))
            f.write("\n")


def load_corine_lookup_table() -> pd.DataFrame:
    file_path = str(resources.files("ggpymanager.data").joinpath("gramm_corine.csv"))
    return pd.read_csv(
        file_path,
        comment="#",
        index_col=0,
    )


def convert_to_gramm_landuse_variables(corine_data: xr.DataArray) -> xr.Dataset:
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
            dict(zip(["unit", "long_name", "standard_name", "description"], attrs[var]))
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


def write_point_dat(
    path, x, y, z, flux, exit_velocity, stack_diameter, exit_temperature, source_group
) -> None:
    if Path(path).exists():
        logging.warning(f"File {path} already exists!")
        raise FileExistsError
    assert all(exit_temperature > 273.15), "Exit temperature must be in Kelvin!"
    assert all(flux != 0), "Flux must be non-zero!"
    assert all(stack_diameter > 0), "Stack diameter must be positive!"
    assert all(exit_velocity > 0), "Exit velocity must be positive!"
    assert source_group.min() > 0, "Source group index is 1-based!"
    assert (
        len(x)
        == len(y)
        == len(z)
        == len(flux)
        == len(exit_velocity)
        == len(stack_diameter)
        == len(exit_temperature)
        == len(source_group)
    ), "All input arrays must have the same length!"
    file_str = "---\n"
    columns = [
        "x",
        "y",
        "z",
        "Emission [kg/h]",
        "-",
        "-",
        "-",
        "exit-velocity [m/s]",
        "stack-diameter [m]",
        "exit-temperature [K]",
        "source group",
    ]
    file_str += ", ".join(columns) + "\n"
    for i in range(len(x)):
        var_list = [
            x[i],
            y[i],
            z[i],
            flux[i],
            0,
            0,
            0,
            exit_velocity[i],
            stack_diameter[i],
            exit_temperature[i],
            source_group[i],
        ]
        file_str += ", ".join([str(np.round(var, 1)) for var in var_list]) + "\n"
    with open(path, "w") as f:
        f.write(file_str)


def write_cadastre_dat(path, x, y, z, dx, dy, dz, flux, source_group) -> None:
    if Path(path).exists():
        logging.warning(f"File {path} already exists!")
        raise FileExistsError
    assert all(flux >= 0), "Flux must be positive!"
    assert source_group.min() > 0, "Source group index is 1-based!"
    assert (
        len(x) == len(y) == len(z) == len(flux) == len(source_group)
    ), "All input arrays must have the same length!"
    df = pd.DataFrame(
        {
            "x": x,
            "y": y,
            "z": z,
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "Emission [kg/h]": flux,
            "-": np.zeros(len(flux)),
            "--": np.zeros(len(flux)),
            "---": np.zeros(len(flux)),
            "source group": source_group,
            "deposition data": np.zeros(len(flux)),
        }
    )
    logging.info(f"Total flux: {df['Emission [kg/h]'].sum()} kg/h")
    logging.info(f"Total number of cadastre entries: {len(df)}")
    df = df[df["Emission [kg/h]"] > 0]
    logging.info(f"Number of cadastre entries after removing zero fluxes: {len(df)}")
    logging.info(f"Writing cadastre file to {path}...")
    df.to_csv(path, index=False, sep=",", mode="w")
