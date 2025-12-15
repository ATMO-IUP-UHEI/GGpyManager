"""File readers for GRAMM/GRAL model input and output files."""

import struct
import zipfile
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import sparse

# Constants
LANDUSE_VARS = ["RHOB", "ALAMBDA", "Z0", "FW", "EPSG", "ALBEDO"]


def read_gral_config(gral_geb_path: Path, in_dat_path: Path) -> dict:
    """Read GRAL configuration from .geb and .dat files.

    Parameters
    ----------
    gral_geb_path : Path
        Path to the GRAL.geb file.
    in_dat_path : Path
        Path to the in.dat file.

    Returns
    -------
    dict
        Dictionary containing GRAL configuration parameters.
    """
    with gral_geb_path.open("r") as f:
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
    with in_dat_path.open("r") as f:
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


def read_landuse(path: str | Path, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    """Read GRAMM landuse file.

    Parameters
    ----------
    path : str | Path
        Path to the landuse file.
    shape : tuple[int, int]
        Shape of the landuse grid (nx, ny).

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with landuse variables as keys and arrays as values.
    """
    landuse_data = {}
    with open(path) as f:
        for line, name in zip(f, LANDUSE_VARS):
            landuse_data[name] = np.array([float(d) for d in line.split()]).reshape(
                shape
            )
    return landuse_data


def read_topography(path: str | Path, GRAMM: Any) -> tuple[np.ndarray, np.ndarray]:
    """Read GRAMM topography file.

    Parameters
    ----------
    path : str | Path
        Path to the topography file.
    GRAMM : object
        GRAMM configuration object with nx, ny, nz attributes.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Topography data and z-grid data.
    """
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


def read_gramm_windfield(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read GRAMM wind field file.

    Parameters
    ----------
    path : Path
        Path to the GRAMM wind field file (.wnd).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Header info, u wind component, v wind component, w wind component in m/s.
    """
    with path.open("rb") as f:
        data = f.read()

    nheader = 20  # header, ni,nj,nz,gridsize -> 4*signed integer (=4*4) + float (4)
    header, nx, ny, nz, dx = struct.unpack("<iiiif", data[:nheader])

    dt = np.dtype(np.short)

    info = np.frombuffer(data[: nheader - 4], dtype=np.int32)
    datarr = np.frombuffer(data[nheader:], dtype=dt)
    datarr = np.reshape(datarr, [nx, ny, nz, 3])
    wind_u = datarr[:, :, :, 0] / 100
    wind_v = datarr[:, :, :, 1] / 100
    wind_w = datarr[:, :, :, 2] / 100

    return info, wind_u, wind_v, wind_w


def read_buildings(path: str | Path, GRAL: Any) -> np.ndarray:
    """Read GRAL buildings file.

    Parameters
    ----------
    path : str | Path
        Path to the buildings.dat file.
    GRAL : object
        GRAL configuration object with nx, ny, dx, dy, xmin, ymin attributes.

    Returns
    -------
    np.ndarray
        2D array of building heights.
    """
    data = np.genfromtxt(path, delimiter=",")

    buildings = np.zeros((GRAL.nx, GRAL.ny))

    x = data[:, 0]
    y = data[:, 1]

    idx = ((x - GRAL.xmin) / GRAL.dx).astype(int)
    idy = ((y - GRAL.ymin) / GRAL.dy).astype(int)

    buildings[idx, idy] = data[:, 3]
    return buildings


def read_gral_geometries(
    path: str | Path, as_xarray: bool, dx: float, dy: float
) -> xr.Dataset | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read GRAL geometry data from a binary file.

    Returns surface elevation (ahk), surface index (kkart),
    building height (bui_height), and orography (oro).

    Parameters
    ----------
    path : str | Path
        Path to the binary geometry file.
    as_xarray : bool, optional
        If True, return as xarray Dataset. Default is False.
    dx : float
        Cell size in x-direction.
    dy : float
        Cell size in y-direction.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | xr.Dataset
        If as_xarray is False, returns tuple of (ahk, kkart, bui_height, oro) arrays.
        If as_xarray is True, returns xarray Dataset with these variables.
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
    if not as_xarray:
        return ahk, kkart, bui_height, oro
    else:
        geom = xr.Dataset(
            {
                "ahk": (("x", "y"), ahk),
                "kkart": (("x", "y"), kkart),
                "bui_height": (("x", "y"), bui_height),
                "oro": (("x", "y"), oro),
            },
            coords={
                "x": (np.arange(0, nx) + 0.5) * dx + ikooagral,
                "y": (np.arange(0, ny) + 0.5) * dy + jkooagral,
            },
            attrs={
                "nz": nz,
                "dzk": dzk,
                "stretching_factor": stretch,
                "ahmin": ahmin,
                "description": f"GRAL geometries read from {path}.",
            },
        )
        geom["ahk"].attrs = {"units": "m", "long_name": "Surface elevation"}
        geom["kkart"].attrs = {"units": "-", "long_name": "Surface index"}
        geom["bui_height"].attrs = {"units": "m", "long_name": "Building height"}
        geom["oro"].attrs = {"units": "m", "long_name": "Orography"}
        # Add the header info as attributes
        geom.attrs.update(
            {
                "nz": nz,
                "ny": ny,
                "nx": nx,
                "ikooagral": ikooagral,
                "jkooagral": jkooagral,
                "dzk": dzk,
                "stretch": stretch,
                "ahmin": ahmin,
            }
        )
        return geom


def read_gral_windfield(
    path: Path, as_xarray: bool = False, config: dict | None = None
) -> xr.Dataset | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read GRAL wind field file (.gff).

    Parameters
    ----------
    path : Path
        Path to the GRAL wind field file.
    as_xarray : bool, optional
        If True, return as xarray Dataset. Default is False.
    config : dict | None, optional
        GRAL configuration dictionary, required if as_xarray is True.

    Returns
    -------
    xr.Dataset | tuple[np.ndarray, np.ndarray, np.ndarray]
        If as_xarray is True, returns Dataset with wind components.
        Otherwise, returns tuple of (ux, vy, wz) arrays.

    Raises
    ------
    ValueError
        If as_xarray is True but config is None.
    """
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
    nz, ny, nx, direction, speed, stab_class, dxy, h = struct.unpack(
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

    if not as_xarray:
        return ux, vy, wz
    else:
        if config is None:
            raise ValueError("Config must be provided when as_xarray is True.")
        x = np.arange(0, nx) * config["dx"] + config["west_border"] + config["dx"] / 2
        y = np.arange(0, ny) * config["dy"] + config["south_border"] + config["dy"] / 2

        z_ids = np.zeros(nz)
        z_ids[1:] = np.arange(0, nz - 1)
        dz = config["dz0"] * (config["stretching_factor"] ** z_ids)
        z = np.cumsum(dz) - dz / 2

        wind = xr.Dataset(
            {
                "ux": (("x", "y", "z"), ux),
                "vy": (("x", "y", "z"), vy),
                "wz": (("x", "y", "z"), wz),
            },
            coords={"x": x, "y": y, "z": z},
            attrs={
                "speed": speed,
                "direction": direction,
                "stab_class": stab_class,
                "dxy": dxy,
                "h": h,
                "description": f"Wind fields from GRAL read from {path.name}.",
            },
        )
        wind["ux"].attrs = {"units": "m/s", "long_name": "Eastward wind component"}
        wind["vy"].attrs = {"units": "m/s", "long_name": "Northward wind component"}
        wind["wz"].attrs = {"units": "m/s", "long_name": "Vertical wind component"}
        return wind


def read_con_file(path: Path, GRAL: Any) -> np.ndarray | None:
    """Read GRAL concentration file.

    Parameters
    ----------
    path : Path
        Path to the concentration file.
    GRAL : object
        GRAL configuration object with nx, ny, dx, dy, xmin, ymin attributes.

    Returns
    -------
    np.ndarray | None
        2D concentration array, or None if file is too small.
    """
    with path.open("rb") as f:
        data = f.read()

    if len(data) <= 4:
        return None

    # Define structured dtype for 'iif' format
    dt = np.dtype([("x", "<i4"), ("y", "<i4"), ("val", "<f4")])
    datarr = np.frombuffer(data[4:], dtype=dt)

    con = np.zeros((GRAL.nx, GRAL.ny), dtype=np.float32)

    # Compute indices directly
    idx = ((datarr["x"] - GRAL.xmin) / GRAL.dx - 0.5).astype(int)
    idy = ((datarr["y"] - GRAL.ymin) / GRAL.dy - 0.5).astype(int)

    con[idx, idy] = datarr["val"]

    return con


def con_file_as_sparse_matrix(path: Path, GRAL: Any) -> sparse.csr_matrix:
    """Read GRAL concentration file as sparse matrix.

    Parameters
    ----------
    path : Path
        Path to the concentration file.
    GRAL : object
        GRAL configuration object with nx, ny, dx, dy, xmin, ymin attributes.

    Returns
    -------
    sparse.csr_matrix
        Sparse matrix representation of concentration data.
    """
    con = read_con_file(path, GRAL)
    return sparse.csr_matrix(con)


def read_gral_concentration(path: str | Path) -> dict[str, np.ndarray]:
    """Read GRAL concentration from .npz file.

    Parameters
    ----------
    path : str | Path
        Path to the .npz file containing concentration data.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with concentration matrices.
    """
    con_dict = {}
    conc_file = np.load(path, allow_pickle=True)
    for key in conc_file:
        con_matrix = conc_file[key].item().toarray()
        con_dict[key] = con_matrix
    return con_dict


def read_esri_ascii(path: str | Path) -> tuple[np.ndarray, dict]:
    """Read an ESRI ASCII raster file.

    Returns the data as a numpy array and metadata.

    Parameters
    ----------
    path : str | Path
        Path to the ESRI ASCII file.

    Returns
    -------
    data : np.ndarray
        2D array of raster values.
    metadata : dict
        Dictionary containing header information (ncols, nrows, xllcorner,
        yllcorner, cellsize, NODATA_value).
    """
    with open(path, "r") as f:
        header = {}
        for _ in range(6):
            line = f.readline()
            key, value = line.strip().split()
            header[key.lower()] = float(value) if "." in value else int(value)
        data = np.loadtxt(f)
        assert data.shape == (header["nrows"], header["ncols"])
    # ESRI ASCII files are stored from top to bottom, so we need to flip the array
    data = np.flipud(data)
    return data, header


def load_corine_lookup_table() -> pd.DataFrame:
    """Load CORINE land cover lookup table.

    Returns
    -------
    pd.DataFrame
        CORINE lookup table with land use parameters.
    """
    file_path = str(resources.files("ggpymanager.data").joinpath("gramm_corine.csv"))
    return pd.read_csv(
        file_path,
        comment="#",
        index_col=0,
    )


def load_catalog_filter() -> pd.DataFrame:
    """Load catalog filter for stability class selection.

    Returns
    -------
    pd.DataFrame
        Catalog filter DataFrame with multi-level columns.
    """
    file_path = str(
        resources.files("ggpymanager.data").joinpath("catalogue_filter.csv")
    )
    return pd.read_csv(
        file_path,
        comment="#",
        header=[0, 1],
        index_col=0,
    )


def read_ggeom_file(file_path: str | Path) -> xr.Dataset:
    """
    Reads a ggeom.asc file and returns an xarray Dataset with the parsed geometry.

    Parameters
    ----------
    file_path : str | Path
        Path to ggeom.asc file.

    Returns
    -------
    xr.Dataset
        Dataset containing arrays for Nx, Ny, Nz, and geometry fields.
    """
    from ..processing.geometry import create_geometry_variable_specs

    with open(file_path, "r") as f:
        tokens = f.read().split()

    # Parse header
    nx, ny, nz = map(int, tokens[:3])

    variable_sizes_types_dimensions = create_geometry_variable_specs(nx, ny, nz)

    vars = {}
    idx = 0
    for var, (size, dtype, _) in variable_sizes_types_dimensions.items():
        vars[var] = np.array(tokens[idx : (idx + size)], dtype=dtype)
        idx += size

    dimensions = {
        "x": nx,
        "y": ny,
        "z": nz,
        "x_stag": nx + 1,
        "y_stag": ny + 1,
        "z_stag": nz + 1,
        "": None,
    }
    ds = xr.Dataset(
        {
            var: (dims, vars[var].reshape([dimensions[dim] for dim in dims]))
            for var, (_, _, dims) in variable_sizes_types_dimensions.items()
        },
        coords={
            "x": vars["X"][:-1] + vars["DDX"] / 2,
            "x_stag": vars["X"],
            "y": vars["Y"][:-1] + vars["DDY"] / 2,
            "y_stag": vars["Y"],
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
