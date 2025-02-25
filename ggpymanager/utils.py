"""
Utility functions partly adapted from Ivo Suter.
"""

import inspect
import logging
import re
import struct
import zipfile
from dataclasses import dataclass
from functools import wraps
from importlib import resources

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse


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
                    ), f"Argument '{arg_name}' expected dimensions {expected_dim}, but got {actual_dim}."

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
                    f"Function '{func.__name__}' expected return dimensions {expected_return_dim}, "
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


def read_landuse(path, GRAMM=GRAMM):
    f = open(path, "r")
    data = f.readlines()
    f.close()

    thermalcondu = data[
        0
    ].split()  # it is actually heatcondu/thermalcondu/900  ->  thermalcondu=1/(data[0]*900/data[1])
    thermalcondu = [float(i) for i in thermalcondu]

    heatcondu = data[1].split()
    heatcondu = [float(i) for i in heatcondu]

    thermalcondu = np.divide(1, np.divide(thermalcondu, heatcondu) * 900)
    # thermalcondu = [round_sig(i) for i in thermalcondu]

    roughness = data[2].split()
    roughness = [float(i) for i in roughness]

    moisture = data[3].split()
    moisture = [float(i) for i in moisture]

    emiss = data[4].split()
    emiss = [float(i) for i in emiss]

    albedo = data[5].split()
    albedo = [float(i) for i in albedo]

    landuse_ind = 0
    thermalcondum = np.zeros((GRAMM.nx, GRAMM.ny), float)
    heatcondum = np.zeros((GRAMM.nx, GRAMM.ny), float)
    roughnessm = np.zeros((GRAMM.nx, GRAMM.ny), float)
    moisturem = np.zeros((GRAMM.nx, GRAMM.ny), float)
    emissm = np.zeros((GRAMM.nx, GRAMM.ny), float)
    albedom = np.zeros((GRAMM.nx, GRAMM.ny), float)

    for i in range(GRAMM.ny):
        for j in range(GRAMM.nx):
            thermalcondum[j, i] = float(thermalcondu[landuse_ind])
            heatcondum[j, i] = float(heatcondu[landuse_ind])
            roughnessm[j, i] = float(roughness[landuse_ind])
            moisturem[j, i] = float(moisture[landuse_ind])
            emissm[j, i] = float(emiss[landuse_ind])
            albedom[j, i] = float(albedo[landuse_ind])
            landuse_ind += 1

    return thermalcondum, heatcondum, roughnessm, moisturem, emissm, albedom


def read_topography(path):
    f = open(path, "r")
    data = f.readlines()
    f.close()
    tmp = data[1].split()
    topo_ind = 0
    topo = np.zeros((GRAMM.nx, GRAMM.ny), np.float)
    for j in range(GRAMM.ny):
        for i in range(GRAMM.nx):
            topo[i, j] = float(tmp[topo_ind])
            topo_ind += 1

    # Z Grid
    tmp = data[8].split()
    zgrid = np.zeros([GRAMM.nx, GRAMM.ny, GRAMM.nz], np.float)
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
    grid_size = np.frombuffer(data[nheader - 4 : nheader], dtype=np.float32)
    datarr = np.frombuffer(data[nheader:], dtype=dt)
    datarr = np.reshape(datarr, [nx, ny, nz, 3])
    wind_u = datarr[:, :, :, 0] * 0.01
    wind_v = datarr[:, :, :, 1] * 0.01
    wind_w = datarr[:, :, :, 2] * 0.01
    umag = np.hypot(wind_u[:, :, :], wind_v[:, :, :])

    return info, wind_u, wind_v


def read_buildings(path, GRAL=GRAL):
    data = np.genfromtxt(path, delimiter=",")

    buildings = np.zeros((GRAL.nx, GRAL.ny))

    x = data[:, 0]
    y = data[:, 1]

    idx = ((x - GRAL.xmin) / GRAL.dx).astype(int)
    idy = ((y - GRAL.ymin) / GRAL.dy).astype(int)

    buildings[idx, idy] = data[:, 3]
    return buildings


def read_gral_geometries(path):
    with open(path, mode="rb") as binfile:
        byte_list = binfile.read()
        binfile.close()

    nheader = 32
    header = byte_list[:nheader]
    nz, ny, nx, ikooagral, jkooagral, dzk, stretch, ahmin = struct.unpack(
        "iiiiifff", header
    )
    # print(dzk, stretch)
    blub = byte_list[nheader:]
    # float and int32 -> 4byte each
    # somehow the array is padded with 0? Therefore it is 1 cell bigger in x- and y-dimension
    datarr = np.zeros([nx + 1, ny + 1, 3])
    c = 0
    for i in range(nx + 1):
        for j in range(ny + 1):
            datarr[i, j, 0] = np.frombuffer(blub[c : (c + 4)], dtype=np.float32)
            datarr[i, j, 1] = np.frombuffer(blub[(c + 4) : (c + 8)], dtype=np.int32)
            datarr[i, j, 2] = np.frombuffer(blub[(c + 8) : (c + 12)], dtype=np.float32)
            c += 12

    # remove the padding with zeroes at both ends
    ahk = datarr[:-1, :-1, 0]  # surface elevation
    kkart = datarr[:-1, :-1, 1].astype(int)  # index of gral surface
    bui_height = datarr[:-1, :-1, 2]  # building height
    oro = ahk - bui_height  # orography / topography (without buildings!)

    return ahk, kkart, bui_height, oro


def read_gral_windfield(path):
    if zipfile.is_zipfile(path):
        gff = zipfile.ZipFile(path, "r")

        for filename in gff.namelist():
            byte_list = gff.read(filename)
            gff.close()

    else:
        with open(path, mode="rb") as binfile:
            byte_list = binfile.read()
            binfile.close()

    nheader = 32
    header = byte_list[:nheader]
    nz, ny, nx, direction, speed, akla, dxy, h = struct.unpack("iiiffifi", header)
    # convert direction to degree (strange, but seems to work)
    direction = 270.0 - np.rad2deg(direction)

    dt = np.dtype(np.short)

    count = (nx + 1) * (ny + 1) * (nz + 1) * 3

    # data = np.fromstring(byte_list[nheader:len(byte_list)], dtype=dt, count=count, sep='')
    data = np.frombuffer(byte_list[nheader:], dtype=dt, count=count)

    data = np.reshape(data, [nx + 1, ny + 1, nz + 1, 3])
    # Cut out zeros at the border
    ux = data[:-1, :-1, 1:, 0] * 0.01
    vy = data[:-1, :-1, 1:, 1] * 0.01
    wz = data[:-1, :-1, 1:, 2] * 0.01

    # print("done loading GRAL flowfields")
    return ux, vy, wz


def read_con_file(path, GRAL=GRAL):
    with path.open("rb") as f:
        data = f.read()
    # Check if empty
    if len(data) <= 4:
        return -1

    header = struct.unpack("i", data[:4])
    data_list = list(struct.iter_unpack("iif", data[4:]))
    datarr = np.array(data_list)
    con = np.zeros((GRAL.nx, GRAL.ny))

    x = datarr[:, 0]
    y = datarr[:, 1]

    idx = ((x - GRAL.xmin) / GRAL.dx).astype(int)
    idy = ((y - GRAL.ymin) / GRAL.dy).astype(int)

    con[idx, idy] = datarr[:, 2]

    return con


def con_file_as_sparse_matrix(path, GRAL=GRAL):
    con = read_con_file(path, GRAL)
    return sparse.csr_matrix(con)


def read_gral_concentration(path):
    con_dict = {}
    conc_file = np.load(path, allow_pickle=True)
    for key in conc_file:
        con_matrix = conc_file[key].all().toarray()
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

    # Select bin for wind speed for each time step
    for i, min_wind_speed in enumerate(catalog_filter.index):
        above_wind_threshold = wind_speed >= float(min_wind_speed)
        wind_speed_index[above_wind_threshold] = i

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
        ]
    return stability_class_mask


def load_catalog_filter():
    file_path = resources.files("ggpymanager.data").joinpath("catalogue_filter.csv")
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
        Synoptic wind speed data with sim_id dimension.
    global_radiation : xr.DataArray, optional
        Global radiation data with time dimension.
    stab_class_catalog : xr.DataArray, optional
        Stability class data with sim_id dimension.

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

    return matching_loss
