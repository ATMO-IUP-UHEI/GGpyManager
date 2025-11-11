"""Concentration utilities."""

import logging
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from collections import namedtuple
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from ggpymanager.utils.decorators import check_docstring_dims
from ggpymanager import io

MIN_DISTANCE_TO_SURFACE = 3.0  # meters


@check_docstring_dims
def convert_locations_to_grid(
    x: xr.DataArray,
    y: xr.DataArray,
    height: xr.DataArray,
    x_grid: xr.DataArray,
    y_grid: xr.DataArray,
    concentration_levels: xr.DataArray,
) -> xr.Dataset:
    """Convert locations in crs to indices of the concentration grid.

    Parameters
    ----------
    x : xr.DataArray (station)
        x coordinates of measurement stations.
    y : xr.DataArray (station)
        y coordinates of measurement stations.
    height : xr.DataArray (station)
        z coordinates of measurement stations.
    x_grid : xr.DataArray (x)
        x coordinates of the concentration grid.
    y_grid : xr.DataArray (y)
        y coordinates of the concentration grid.
    concentration_levels : xr.DataArray (z)
        Height levels of the concentration grid.
    Returns
    -------
    grid_locations : xr.Dataset (station)
        Dataset containing the x_id, y_id, and z_id for each station.
    """
    assert "station" in x.coords
    logging.info("Generate x, y, and z ids for each station")
    x_id = xr.DataArray(abs(x_grid - x).argmin(dim="x"))
    y_id = xr.DataArray(abs(y_grid - y).argmin(dim="y"))
    z_id = xr.DataArray(abs(concentration_levels - height).argmin(dim="z"))
    mask = (
        (x >= x_grid.min())
        & (x <= x_grid.max())
        & (y >= y_grid.min())
        & (y <= y_grid.max())
    )
    x_id = x_id.where(mask, drop=True).astype(int)
    y_id = y_id.where(mask, drop=True).astype(int)
    z_id = z_id.where(mask, drop=True).astype(int)
    logging.info(
        f"{mask.sum().values}/{len(mask)} of stations are inside the GRAL domain. "
        "Dropping the rest."
    )
    grid_locations = xr.Dataset(
        coords={
            "x_id": x_id,
            "y_id": y_id,
            "z_id": z_id,
        },
    )
    return grid_locations


def adjust_station_x_position(
    geometry: xr.Dataset,
    min_distance_to_surface: float,
    model_stations: xr.Dataset,
    i: int,
    station_name: str,
) -> None:
    """Adjust station position in x-direction to maintain minimum distance to surface.

    Parameters
    ----------
    geometry : xr.Dataset
        Dataset containing building heights and grid coordinates.
    min_distance_to_surface : float
        Minimum required distance to surface in meters.
    model_stations : xr.Dataset (station)
        Dataset containing station positions and attributes.
    i : int
        Station index.
    station_name : str
        Name of the station being adjusted.
    """
    for dx in [-1, 1]:
        new_x_id = model_stations["x_id"].isel(station=i) + dx
        new_building_height = geometry["bui_height"].isel(
            x=new_x_id, y=model_stations["y_id"].isel(station=i)
        )
        new_height_margin = (
            model_stations["model_height"].isel(station=i) - new_building_height
        )
        if new_height_margin >= min_distance_to_surface:
            model_stations["x_id"][dict(station=i)] = new_x_id
            model_stations["building_height"][dict(station=i)] = new_building_height
            model_stations["height_margin"][dict(station=i)] = new_height_margin
            logging.info(f"Moved {station_name} by {dx} in x-direction")
            break


def move_station_in_y_direction(
    geometry: xr.Dataset,
    min_distance_to_surface: float,
    model_stations: xr.Dataset,
    i: int,
    station_name: str,
) -> None:
    """Move station in y-direction to maintain minimum distance to surface.

    Parameters
    ----------
    geometry : xr.Dataset
        Dataset containing building heights and grid coordinates.
    min_distance_to_surface : float
        Minimum required distance to surface in meters.
    model_stations : xr.Dataset
        Dataset containing station positions and attributes.
    i : int
        Station index.
    station_name : str
        Name of the station being moved.
    """
    for dy in [-1, 1]:
        new_y_id = model_stations["y_id"].isel(station=i) + dy
        new_building_height = geometry["bui_height"].isel(
            x=model_stations["x_id"].isel(station=i), y=new_y_id
        )
        new_height_margin = (
            model_stations["model_height"].isel(station=i) - new_building_height
        )
        if new_height_margin >= min_distance_to_surface:
            model_stations["y_id"][dict(station=i)] = new_y_id
            model_stations["building_height"][dict(station=i)] = new_building_height
            model_stations["height_margin"][dict(station=i)] = new_height_margin
            logging.info(f"Moved {station_name} by {dy} in y-direction")
            break


@check_docstring_dims
def get_measurement_locations_in_model(
    measurements: xr.Dataset,
    geometry: xr.Dataset,
    height_levels: xr.DataArray,
    min_distance_to_surface: float = MIN_DISTANCE_TO_SURFACE,
) -> xr.Dataset:
    """
    Find the x, y, and z index for each measurement site corresponding to the GRAL
    concentration grid. Account for buildings and the surface to find a good
    representation by selecting neighboring grid cells.

    Parameters
    ----------
    measurements : xr.Dataset
        Dataset containing measurement station coordinates and heights.
    geometry : xr.Dataset
        Dataset containing building heights and grid coordinates.
    height_levels : xr.DataArray
        Height levels of the concentration grid.
    min_distance_to_surface : float, optional
        Minimum required distance to surface in meters, by default
        MIN_DISTANCE_TO_SURFACE.

    Returns
    -------
    model_stations : xr.Dataset (station)
        Dataset containing the x_id, y_id, z_id, and height information for each
        station.
    """
    model_stations = convert_locations_to_grid(
        measurements["x"],
        measurements["y"],
        measurements["height"],
        geometry["x"],
        geometry["y"],
        height_levels,
    )
    model_stations["building_height"] = geometry["bui_height"].isel(
        x=model_stations["x_id"], y=model_stations["y_id"]
    )
    model_stations["model_height"] = height_levels[model_stations["z_id"]]
    model_stations["height_margin"] = (
        model_stations["model_height"] - model_stations["building_height"]
    )
    for i in range(len(model_stations["station"])):
        while model_stations["height_margin"].isel(station=i) < min_distance_to_surface:
            station_name = str(model_stations["station"].isel(station=i).values)
            height_margin = model_stations["height_margin"].isel(station=i).values

            logging.info(f"{station_name} has height margin of {height_margin:.0f}")

            # Move around
            if (
                model_stations["height_margin"].isel(station=i)
                < min_distance_to_surface
            ):
                adjust_station_x_position(
                    geometry, min_distance_to_surface, model_stations, i, station_name
                )
            if (
                model_stations["height_margin"].isel(station=i)
                < min_distance_to_surface
            ):
                move_station_in_y_direction(
                    geometry, min_distance_to_surface, model_stations, i, station_name
                )
            if (
                model_stations["height_margin"].isel(station=i)
                < min_distance_to_surface
            ):
                # Try to increase level by one
                if model_stations["z_id"].isel(station=i) >= len(height_levels) - 1:
                    logging.info("Already at highest level, no fix found.")
                    break
                new_z_id = model_stations["z_id"].isel(station=i) + 1
                model_stations["z_id"][dict(station=i)] = new_z_id
                model_stations["model_height"][dict(station=i)] = height_levels[
                    new_z_id
                ]
                model_stations["height_margin"][dict(station=i)] = model_stations[
                    "model_height"
                ].isel(station=i) - model_stations["building_height"].isel(station=i)
                logging.info(f"Moved {station_name} by 1 height level up")
    return model_stations


@check_docstring_dims
def create_figure_of_locations(
    model_stations: xr.Dataset,
    geometry: xr.Dataset,
    height_levels: xr.DataArray,
    figure_save_path: str,
) -> None:
    """Create a figure showing the measurement locations in the model grid.

    Parameters
    ----------
    model_stations : xr.Dataset (station)
        Dataset containing the x_id, y_id, and z_id for each station.
    geometry : xr.Dataset
        Dataset containing building heights and grid coordinates.
    height_levels : xr.DataArray
        Height levels of the concentration grid.
    figure_save_path : str
        Path where the figure will be saved.
    """
    logging.info("Creating figure of measurement locations")
    n_cols = 3
    n_rows = len(model_stations["station"]) // n_cols + 1
    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(5 * n_cols, 4 * n_rows)
    )

    plot_idx = 0
    for i in range(len(model_stations["station"])):
        s = model_stations[{"station": i}]
        ax = axes.flat[plot_idx]
        margin = 10
        x_slice = slice(s["x_id"].values - margin, s["x_id"].values + margin)
        y_slice = slice(s["y_id"].values - margin, s["y_id"].values + margin)

        x, y = geometry["x"][s["x_id"]], geometry["y"][s["y_id"]]

        surface_distance = (
            geometry["bui_height"][x_slice, y_slice] - height_levels[s["z_id"]].values
        )
        surface_distance.attrs = {
            "long_name": "Distance to Surface or Buildings",
            "units": "m",
        }
        vmax = np.abs(surface_distance).max()
        vmin = -vmax
        surface_distance.plot(
            x="x",
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cmap="coolwarm",
        )  # type: ignore

        ax.scatter(
            x,
            y,
            s=100,
            color=("black" if s["height_margin"] >= MIN_DISTANCE_TO_SURFACE else "red"),
            label=f"Station: {s["height_margin"]:.1f} m buffer",
        )

        ax.set_aspect("equal")
        ax.legend()
        ax.set_title(
            f"Station number {i}: {s["station"].values}\n"
            f"Height: {s["z_id"].values} | {height_levels[s["z_id"]].values} m agl"
        )
        ax.set_xticks([])
        ax.set_yticks([])
        plot_idx += 1

    # Hide any unused subplots
    for idx in range(plot_idx, len(axes.flat)):
        axes.flat[idx].set_visible(False)

    plt.tight_layout()
    logging.info(f"Saving figure to {figure_save_path}")
    fig.savefig(figure_save_path)
    plt.close(fig)


@check_docstring_dims
def generate_empty_dataset(
    geometry: xr.Dataset,
    source_groups: List[int],
    sim_indices: List[int],
    model_stations: xr.Dataset,
) -> xr.Dataset:
    """Generate an empty xarray Dataset for GRAL concentration results.

    Parameters
    ----------
    geometry : xr.Dataset
        Dataset containing building heights and grid coordinates.
    source_groups : list of int
        List of source group identifiers.
    sim_indices : list of int
        List of simulation indices.
    model_stations : xr.Dataset (station)
        Dataset containing the x_id, y_id, and z_id for each station.

    Returns
    -------
    gral_con : xr.Dataset
        Dataset containing empty concentration arrays with proper coordinates and
        attributes.
    """
    n_sims = len(sim_indices)
    n_stations = len(model_stations["station"])
    gral_con = xr.Dataset(
        {
            "concentration": (
                (
                    "sim_id",
                    "source_group",
                    "station",
                ),
                np.nan * np.zeros((n_sims, len(source_groups), n_stations)),
            ),
        },
        coords={
            "sim_id": sim_indices,
            "source_group": source_groups,
            "station": model_stations["station"],
            "x": geometry["x"][model_stations["x_id"]],
            "y": geometry["y"][model_stations["y_id"]],
            "z": model_stations["model_height"],
        },
    )

    # Add attributes
    gral_con["concentration"].attrs = {
        "units": "mu g m-3",
        "long_name": "Concentration",
    }
    gral_con["sim_id"].attrs = {
        "long_name": "Simulation ID",
    }
    gral_con["source_group"].attrs = {
        "long_name": "Source Group",
    }
    gral_con["station"].attrs = {
        "long_name": "Measurement Station",
    }
    gral_con["x"].attrs = {
        "long_name": "x coordinate of measurement station",
        "units": "m",
    }
    gral_con["y"].attrs = {
        "long_name": "y coordinate of measurement station",
        "units": "m",
    }
    gral_con["z"].attrs = {
        "long_name": "Height above ground level of measurement station",
        "units": "m",
    }

    return gral_con


def load_concentration(
    dir_path: Path,
    x_ids: np.ndarray,
    y_ids: np.ndarray,
    z_ids: np.ndarray,
    config: Dict[str, Any],
    gral_config: tuple,
) -> np.ndarray:
    """Load concentration data from GRAL output files.

    Parameters
    ----------
    dir_path : Path
        Directory path containing GRAL output files.
    x_ids : np.ndarray
        Array of x indices for stations.
    y_ids : np.ndarray
        Array of y indices for stations.
    z_ids : np.ndarray
        Array of z indices for stations.
    config : dict
        Configuration dictionary containing source groups.
    gral_config : tuple
        Named tuple containing GRAL grid configuration.

    Returns
    -------
    data : np.ndarray
        Array of concentration values with shape (n_source_groups, n_stations).
    """
    data = np.zeros((len(config["source_groups"]), len(x_ids)), dtype=np.float32)
    for z_id in np.unique(z_ids):
        mask = z_ids == z_id
        height_index = z_id + 1
        for i, source_group in enumerate(config["source_groups"]):
            file_path = dir_path / f"00001-{height_index}{source_group:03}.con"
            arr = io.read_con_file(file_path, gral_config)
            if arr is not None:
                data[i, mask] = arr[x_ids[mask], y_ids[mask]]
    return data


def pool_data(
    geometry: xr.Dataset,
    x_ids: xr.DataArray,
    y_ids: xr.DataArray,
    z_ids: xr.DataArray,
    dir_list: List[Path],
    n_processes: int,
    config: Dict[str, Any],
) -> List[np.ndarray]:
    """Load concentration data in parallel using thread pool.

    Parameters
    ----------
    geometry : xr.Dataset
        Dataset containing grid coordinates and dimensions.
    x_ids : xr.DataArray
        Array of x indices for stations.
    y_ids : xr.DataArray
        Array of y indices for stations.
    z_ids : xr.DataArray
        Array of z indices for stations.
    dir_list : list of Path
        List of directory paths containing GRAL output files.
    n_processes : int
        Number of worker threads for parallel processing.
    config : dict
        Configuration dictionary containing source groups.

    Returns
    -------
    data : list of np.ndarray
        List of concentration arrays, one per simulation.
    """
    gral_config = {
        "nx": geometry.sizes["x"],
        "ny": geometry.sizes["y"],
        "dx": geometry["x"][1].values - geometry.x[0].values,
        "dy": geometry["y"][1].values - geometry.y[0].values,
        "xmin": geometry["x"].min().item(),
        "ymin": geometry["y"].min().item(),
    }
    gral_config = namedtuple("gral_config", gral_config.keys())(**gral_config)
    func = partial(
        load_concentration,
        x_ids=x_ids.values,
        y_ids=y_ids.values,
        z_ids=z_ids.values,
        config=config,
        gral_config=gral_config,
    )
    with ThreadPoolExecutor(max_workers=n_processes) as ex:
        data = list(tqdm(ex.map(func, dir_list), total=len(dir_list)))

    return data


def process_concentration_from_model(
    geometry: xr.Dataset,
    model_stations: xr.Dataset,
    sim_path: Path,
    gral_concentration_output_path: Path,
    config: Dict[str, Any],
    n_processes: int = 1,
    batch_size: int = 10,
) -> None:
    """Process GRAL concentration output files and extract data at measurement stations.

    Parameters
    ----------
    geometry : xr.Dataset
        Dataset containing building heights and grid coordinates.
    model_stations : xr.Dataset
        Dataset containing station positions and grid indices.
    sim_path : Path
        Path to directory containing simulation subdirectories.
    gral_concentration_output_path : Path
        Path where processed concentration dataset will be saved.
    config : dict
        Configuration dictionary containing source groups and other settings.
    n_processes : int, optional
        Number of worker threads for parallel processing, by default 1.
    batch_size : int, optional
        Number of simulations to process in each batch, by default 10.
    """
    logging.info(f"Looking for simulation directories in {sim_path}")
    dir_list = sorted([d for d in sim_path.iterdir() if d.is_dir()])
    logging.info(f"Found {len(dir_list)} directories.")
    sim_indices = [int(f.name.removeprefix("sim_")) for f in dir_list]
    if gral_concentration_output_path.exists():
        logging.info(
            "Loading existing GRAL concentration dataset from:\n"
            f"{gral_concentration_output_path}"
        )
        con = xr.load_dataset(gral_concentration_output_path)
    else:
        logging.info("Generating empty GRAL concentration dataset")
        con = generate_empty_dataset(
            geometry, config["source_groups"], sim_indices, model_stations
        )

    # Check how much data is missing
    missing = con["concentration"].isnull().any(("station", "source_group"))
    logging.info(
        f"Missing data for {missing.sum().item()}/{len(con['sim_id'])} simulations"
    )
    missing_dir_list = [dir_list[i] for i in range(len(dir_list)) if missing[i]]
    if len(missing_dir_list) == 0:
        logging.info("No missing data found. Exiting.")
        return
    logging.info(
        f"Processing data in batches of {batch_size} using {n_processes} threads"
    )
    for batch_start in range(0, len(missing_dir_list), batch_size):
        timer = datetime.now()
        batch_end = min(batch_start + batch_size, len(missing_dir_list))
        batch_dirs = missing_dir_list[batch_start:batch_end]
        batch_sim_indices = sim_indices[batch_start:batch_end]

        logging.info(
            f"Processing simulations {batch_start + 1} to {batch_end} "
            f"out of {len(missing_dir_list)}"
        )

        data = pool_data(
            geometry,
            model_stations["x_id"],
            model_stations["y_id"],
            model_stations["z_id"],
            batch_dirs,
            n_processes,
            config,
        )
        for i, sim_index in enumerate(batch_sim_indices):
            con["concentration"].loc[dict(sim_id=sim_index)] = data[i]
        elapsed = datetime.now() - timer
        logging.info(
            f"Processed batch in {elapsed.total_seconds():.1f} seconds "
            f"({(batch_end - batch_start) / elapsed.total_seconds():.2f} sims/sec)"
        )
        # Save intermediate result
        logging.info(f"Saving intermediate result to {gral_concentration_output_path}")
        con.to_netcdf(gral_concentration_output_path)
