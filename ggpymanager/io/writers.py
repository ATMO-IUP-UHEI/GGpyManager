"""File writers for GRAMM/GRAL model input files."""

import logging
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from compliance_checker.runner import CheckSuite, ComplianceChecker

# Constants
LANDUSE_VARS = ["RHOB", "ALAMBDA", "Z0", "FW", "EPSG", "ALBEDO"]


def write_landuse(path: str | Path, landuse_data: xr.Dataset) -> None:
    """Write GRAMM landuse file.

    Parameters
    ----------
    path : str | Path
        Path to the output landuse file.
    landuse_data : xr.Dataset
        Dataset containing landuse variables.
    """
    with open(path, "w") as f:
        for name in LANDUSE_VARS:
            arr = landuse_data[name].values.reshape(-1)
            line = " ".join(str(x) for x in arr)
            f.write(line + "\n")


def write_buildings_file(path: str | Path, building_height: xr.DataArray) -> None:
    """Write GRAL buildings file.

    Parameters
    ----------
    path : str | Path
        Path to the output buildings.dat file.
    building_height : xr.DataArray
        2D DataArray with building heights.
    """
    stacked = building_height.stack(position=("x", "y"))
    stacked = stacked[~stacked.isnull()]
    logging.info(f"Writing building file to {path}...")
    with open(path, "w") as f:
        for x, y, h in zip(stacked.x.values, stacked.y.values, stacked.values):
            f.write(f"{x},{y},0,{h}\n")


def write_esri_ascii(path: str | Path, data: xr.DataArray) -> None:
    """Write an ESRI ASCII raster file.

    Parameters
    ----------
    path : str | Path
        Path to the output ESRI ASCII file.
    data : xr.DataArray
        2D DataArray to write.

    Raises
    ------
    FileExistsError
        If the file already exists.
    """
    if Path(path).exists():
        logging.warning(f"File {path} already exists!")
        raise FileExistsError
    ncols, nrows = data.sizes["x"], data.sizes["y"]
    cellsize = data.x.diff("x").mean().values
    xllcorner = data.x.min().values - cellsize / 2
    yllcorner = data.y.min().values - cellsize / 2
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


def write_point_dat(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    flux: np.ndarray,
    exit_velocity: np.ndarray,
    stack_diameter: np.ndarray,
    exit_temperature: np.ndarray,
    source_group: np.ndarray,
) -> None:
    """Write GRAL point emissions file.

    Parameters
    ----------
    path : str | Path
        Path to the output point.dat file.
    x : np.ndarray
        X coordinates of point sources.
    y : np.ndarray
        Y coordinates of point sources.
    z : np.ndarray
        Z coordinates of point sources.
    flux : np.ndarray
        Emission flux in kg/h.
    exit_velocity : np.ndarray
        Exit velocity in m/s.
    stack_diameter : np.ndarray
        Stack diameter in m.
    exit_temperature : np.ndarray
        Exit temperature in K.
    source_group : np.ndarray
        Source group indices (1-based).

    Raises
    ------
    FileExistsError
        If the file already exists.
    AssertionError
        If validation checks fail.
    """
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


def write_cadastre_dat(
    path: str | Path,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    dz: np.ndarray,
    flux: np.ndarray,
    source_group: np.ndarray,
) -> None:
    """Write GRAL cadastre (area source) emissions file.

    Parameters
    ----------
    path : str | Path
        Path to the output cadastre.dat file.
    x : np.ndarray
        X coordinates of area sources.
    y : np.ndarray
        Y coordinates of area sources.
    z : np.ndarray
        Z coordinates of area sources.
    dx : np.ndarray
        X extent of area sources.
    dy : np.ndarray
        Y extent of area sources.
    dz : np.ndarray
        Z extent of area sources.
    flux : np.ndarray
        Emission flux in kg/h.
    source_group : np.ndarray
        Source group indices (1-based).

    Raises
    ------
    FileExistsError
        If the file already exists.
    AssertionError
        If validation checks fail.
    """
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


def num_to_str(num: float) -> str:
    """
    Convert a number to a string with trailing zeros removed.

    Parameters
    ----------
    num : float
        Number to convert.

    Returns
    -------
    str
        String representation with trailing zeros removed.
    """
    return format(num, ".2f").rstrip("0").rstrip(".")


def data_to_str(data: np.ndarray | xr.DataArray) -> str:
    """
    Convert an array to a string.

    Parameters
    ----------
    data : np.ndarray | xr.DataArray
        Array to convert to a string.

    Returns
    -------
    str
        String representation of the array.
    """
    array = np.array(data)
    return " ".join([num_to_str(num) for num in array.ravel()])


def write_ggeom_file(geom: xr.Dataset, file_path: str | Path) -> None:
    """
    Write a GRAMM geometry dataset to a ggeom.asc file.

    Parameters
    ----------
    geom : xr.Dataset
        Geometry dataset to write.
    file_path : str | Path
        Path where to write the file.

    Raises
    ------
    AssertionError
        If file already exists or dataset validation fails.
    """
    from ..processing.geometry import test_dataset

    logging.info(f"Writing ggeom.asc file for GRAMM to {file_path}.")
    assert not Path(file_path).exists(), f"File path {file_path} already exists."
    test_dataset(geom, int(geom["NX"]), int(geom["NY"]), int(geom["NZ"]))
    with open(file_path, "w") as file:
        line = f"{int(geom["NX"])} {int(geom["NY"])} {int(geom["NZ"])} "
        line += data_to_str(geom["X"]) + " "
        line += data_to_str(geom["Y"]) + " "
        line += data_to_str(geom["Z"]) + " "
        file.write(line + "\n")

        line = data_to_str(geom["AH"])
        file.write(line + "\n")

        line = data_to_str(geom["VOL"])
        file.write(line + "\n")

        line = data_to_str(geom["AREAX"])
        file.write(line + "\n")

        line = data_to_str(geom["AREAY"])
        file.write(line + "\n")

        line = data_to_str(geom["AREAZX"])
        file.write(line + "\n")

        line = data_to_str(geom["AREAZY"])
        file.write(line + "\n")

        line = data_to_str(geom["AREAZ"])
        file.write(line + "\n")

        line = data_to_str(geom["ZSP"])
        file.write(line + "\n")

        line = data_to_str(geom["DDX"])
        file.write(line + "\n")

        line = data_to_str(geom["DDY"])
        file.write(line + "\n")

        line = data_to_str(geom["ZAX"])
        file.write(line + "\n")

        line = data_to_str(geom["ZAY"])
        file.write(line + "\n")

        line = f"{int(geom["IKOOA"])} {int(geom["JKOOA"])} {int(geom["WINKEL"])}"
        file.write(line + "\n")

        line = data_to_str(geom["AHE"])
        file.write(line + "\n")


def save_netcdf_with_cf_check(
    dataset: xr.Dataset,
    path: str | Path,
    ignore_tests: bool = False,
    **to_netcdf_kwargs,
) -> bool:
    """
    Save a dataset to netCDF only if it passes CF compliance checks.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to save
    path : str | Path
        Output path for the netCDF file
    ignore_tests : bool, optional
        If True, skip CF compliance checks and save the file directly. Default is False.
    **to_netcdf_kwargs
        Additional arguments passed to dataset.to_netcdf()

    Returns
    -------
    bool
        True if file was saved successfully, False if CF check failed
    """
    assert isinstance(dataset, xr.Dataset), "Input must be an xarray Dataset."

    if ignore_tests:
        logging.warning("Ignoring CF compliance checks and saving file directly.")

    path = Path(path)
    # Save to temporary file first
    with tempfile.NamedTemporaryFile(suffix="." + str(path.name), delete=False) as tmp:
        tmp_path = tmp.name
        # Determine unlimited dimensions if not provided
        prepare_netcdf_dataset(dataset, to_netcdf_kwargs)
        dataset.to_netcdf(tmp_path, **to_netcdf_kwargs)

    try:
        # Check if CRS information is present
        if any(v in dataset.dims for v in ["x", "y", "lat", "lon"]):
            # Assert that ds.rio.crs is not None
            if not hasattr(dataset, "rio") or dataset.rio.crs is None:
                assert False, (
                    "Dataset has spatial coordinates but no CRS information. "
                    "Please set the CRS using dataset.rio.write_crs()."
                )

        # Run CF checker
        check_suite = CheckSuite()
        check_suite.load_all_available_checkers()
        return_value, errors = ComplianceChecker.run_checker(
            tmp_path, checker_names=["cf:1.11"], verbose=2, criteria="strict"
        )

        # Only fail on actual CF compliance issues (return_value == 0 means failure)
        # The 'errors' variable captures internal checker exceptions (e.g., when
        # checking monotonicity of string coordinates), which are not CF violations.
        # Log a warning if there were checker exceptions but tests passed.
        if errors and return_value:
            logging.warning(
                f"CF checker encountered internal exceptions but all tests passed. "
                f"File will be saved: {path}"
            )

        if (not return_value) and (not ignore_tests):
            logging.error(f"File NOT saved due to CF compliance errors: {path}")
            Path(tmp_path).unlink()  # Delete temp file
            return False
        else:
            # Move temp file to final destination
            Path(tmp_path).rename(path)
            logging.info(f"File saved successfully: {path}")
            return True

    except Exception as e:
        logging.error(f"Error during CF check: {e}")
        Path(tmp_path).unlink(missing_ok=True)
        return False


def prepare_netcdf_dataset(dataset, to_netcdf_kwargs):
    # Determine unlimited dimensions
    if "time" in dataset.dims:
        unlimited_dims = ("time",)
        logging.info("Setting 'time' as unlimited dimension for netCDF output.")
    elif "iteration" in dataset.dims:
        unlimited_dims = ("iteration",)
        logging.info("Setting 'iteration' as unlimited dimension for netCDF output.")
    else:
        unlimited_dims = None
        # Add units_metadata attribute if not present
    for var in dataset.variables:
        if dataset[var].attrs.get("standard_name", "") == "time":
            if "units_metadata" not in dataset[var].attrs:
                logging.info(f"Adding 'units_metadata' attribute to '{var}' variable.")
                dataset[var].attrs.update({"units_metadata": "leap_seconds: utc"})
    to_netcdf_kwargs.setdefault("unlimited_dims", unlimited_dims)

    # Add global attributes for CF compliance if not present
    if "Conventions" not in dataset.attrs:
        dataset.attrs["Conventions"] = "CF-1.11"
        logging.info("Adding 'Conventions' global attribute for CF compliance.")
    if "history" not in dataset.attrs:
        dataset.attrs["history"] = (
            f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        logging.info("Adding 'history' global attribute.")

    # Add long_name to "spatial_ref" variable if missing
    if "spatial_ref" in dataset.variables:
        if "long_name" not in dataset["spatial_ref"].attrs:
            dataset["spatial_ref"].attrs["long_name"] = "CRS definition"
            logging.info("Adding 'long_name' attribute to 'spatial_ref' variable.")
