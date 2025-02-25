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
    Decorator to check if the dimensions of xr.DataArray arguments match the docstring specification.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Retrieve function signature
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Retrieve docstring
        docstring = inspect.getdoc(func)
        if not docstring:
            return func(*args, **kwargs)  # Skip check if no docstring

        # Extract expected dimensions from docstring
        pattern = re.compile(r"(\w+)\s*:\s*xr\.DataArray\s*\((.*?)\)")
        expected_dims = {
            match[0]: tuple(match[1].split(", "))
            for match in pattern.findall(docstring)
        }

        # Validate dimensions for each xr.DataArray argument
        for arg_name, expected_dim in expected_dims.items():
            if arg_name in bound_args.arguments:
                arg_value = bound_args.arguments[arg_name]
                if isinstance(arg_value, xr.DataArray):
                    actual_dim = arg_value.dims
                    assert (
                        actual_dim == expected_dim
                    ), f"Argument '{arg_name}' expected dimensions {expected_dim}, but got {actual_dim}."

        return func(*args, **kwargs)

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
