"""Module to manage the GRAMM/GRAL catalog.

The module checks consistency of the simulations and can create a list of simulations
which can be started to efficiently use the available computation power.
"""

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from tqdm import tqdm

import ggpymanager.config as CONFIG
from ggpymanager import io, models, processing

logger = logging.getLogger(__name__)


def _get_dir_size(sim_dir: Path) -> int:
    """Get disk space for a single directory.

    Parameters
    ----------
    sim_dir : Path
        Path to the simulation directory.

    Returns
    -------
    int
        Size in bytes, or 0 on error.
    """
    try:
        result = subprocess.run(
            ["du", "-sk", str(sim_dir)],
            capture_output=True,
            text=True,
            check=True,
        )
        size_kb = int(result.stdout.split()[0])
        return size_kb * 1024
    except (subprocess.CalledProcessError, ValueError, IndexError) as e:
        logger.warning(f"Error calculating disk space for {sim_dir}: {e}")
        return 0


def _process_wind_gradient(p: Path, model: str, config: dict) -> xr.DataArray:
    """Process wind field for a single simulation.

    Parameters
    ----------
    p : Path
        Path to simulation directory.
    model : str
        Model type ("gramm" or "gral").
    config : dict
        GRAL configuration dictionary.

    Returns
    -------
    xr.DataArray
        Mean wind speed vertical profile, or NaN if wind file doesn't exist.
    """
    p_wind = p / CONFIG.WIND_FILE_EXTENSION[model]
    if not p_wind.exists():
        logger.warning(f"Wind file does not exist: {p_wind}")
        return xr.DataArray(np.nan, dims=["sim_id"])

    wind = io.read_gral_windfield(p_wind, as_xarray=True, config=config)
    if not isinstance(wind, xr.Dataset):
        logger.warning(f"Failed to read wind field from file: {p_wind}")
        return xr.DataArray(np.nan, dims=["sim_id"])

    wind_speed = processing.wind_speed_from_vector(wind["ux"], wind["vy"])
    return wind_speed.mean(dim=["x", "y"])


def _process_concentration_gradient(p: Path, config: dict) -> xr.DataArray:
    """Process concentration fields for a single simulation.

    Parameters
    ----------
    p : Path
        Path to simulation directory.
    config : dict
        GRAL configuration dictionary.

    Returns
    -------
    xr.DataArray
        Concentration vertical profile.
    """
    nz = config["n_horizontal_slices_concentration"]
    con = np.zeros((nz,), dtype=np.float64)

    for i in range(nz):
        path_list = list(p.glob(f"*-{i+1}*.con"))
        for path in path_list:
            con[i] += io.read_con_file_mean(path, GRAL=config)

    return xr.DataArray(
        con,
        dims=["vertical_level"],
        coords={"vertical_level": config["horizontal_slices_concentration"]},
    )


class Catalog:
    """Catalog manager for GRAMM/GRAL simulations.

    Parameters
    ----------
    catalog_path : str | Path
        Path to the catalog directory.
    model : Literal["gramm", "gral"]
        Model type, either "gramm" or "gral".

    Raises
    ------
    ValueError
        If model is not "gramm" or "gral".
    """

    def __init__(self, catalog_path: str | Path, model: Literal["gramm", "gral"]):
        if model not in ["gramm", "gral"]:
            raise ValueError(f"Model must be 'gramm' or 'gral', got {model}")
        self.model = model
        self.catalog_path = Path(catalog_path)
        logger.info(f"Scanning catalog {self.catalog_path}")
        self._check_input_files()
        self._check_simulations()
        self._check_status_log()

    def _check_input_files(self) -> None:
        """Check if all required input files are present in the config directory."""
        self.config_path = self.catalog_path / CONFIG.CONFIG_PATH
        logger.info(f"Checking input files in directory: {self.config_path}")
        self.input_files = CONFIG.INPUT_FILES[self.model]
        missing_files = {k: [] for k in self.input_files.keys()}
        for k in self.input_files:
            for file in self.input_files[k]:
                file_path = self.config_path / file
                if not file_path.exists():
                    missing_files[k].append(file)

        if missing_files["required"]:
            logger.warning(
                "The following required input files are missing in "
                f"{self.config_path}:"
            )
            for file in missing_files["required"]:
                logger.warning(f" - {file}")
        elif missing_files["optional"]:
            logger.info(
                "The following optional input files are missing in "
                f"{self.config_path}:"
            )
            for file in missing_files["optional"]:
                logger.info(f" - {file}")
        else:
            logger.info("All required input files are present.")

    def _check_simulations(self) -> None:
        """Scan simulation directory and count completed simulations and wind files."""
        self.simulation_path = self.catalog_path / CONFIG.SIMULATION_PATH
        logger.info(f"Checking simulations in directory: {self.simulation_path}")

        self.simulation_entries = []
        self.sim_ids = []
        self.n_completed_simulations = 0
        self.n_wind_files = 0

        if not self.simulation_path.exists():
            logger.warning(f"Simulation path does not exist: {self.simulation_path}")
            return

        for sim_dir in tqdm(sorted(self.simulation_path.iterdir())):
            if not (sim_dir.is_dir() and sim_dir.name.startswith("sim_")):
                continue

            self.simulation_entries.append(sim_dir)
            sim_id = int(sim_dir.name.lstrip("sim_"))
            self.sim_ids.append(sim_id)

            # Check for completed simulation
            if self._is_simulation_completed(sim_dir):
                self.n_completed_simulations += 1

            # Check for wind field file
            wind_file = sim_dir / CONFIG.WIND_FILE_EXTENSION[self.model]
            if wind_file.exists():
                self.n_wind_files += 1

        logger.info(
            f"Found {len(self.simulation_entries)} simulation entries in the catalog."
        )
        logger.info(
            f"{self.n_completed_simulations} simulations are marked as completed."
        )
        logger.info(f"{self.n_wind_files} simulations have computed wind fields.")

    def _is_simulation_completed(self, sim_dir: Path) -> bool:
        """Check if a simulation is marked as completed based on stdout file.

        Parameters
        ----------
        sim_dir : Path
            Path to the simulation directory.

        Returns
        -------
        bool
            True if the simulation is completed, False otherwise.
        """
        stdout_file = sim_dir / CONFIG.STD_OUT_FILE_NAME[self.model]
        if not stdout_file.exists():
            return False

        try:
            with open(stdout_file, "r") as f:
                stdout_content = f.read()
            return (
                CONFIG.STD_OUT_STRING_FOR_COMPLETED_SIMULATION[self.model]
                in stdout_content
            )
        except (IOError, OSError) as e:
            logger.warning(f"Error reading stdout file {stdout_file}: {e}")
            return False

    def _check_status_log(self) -> None:
        """Check for existing status log or create a new one."""
        self.status_log_path = self.catalog_path / CONFIG.STATUS_LOG_FILE_NAME
        if self.status_log_path.exists():
            logger.info(f"Status log found at {self.status_log_path}")
            ds = xr.load_dataset(self.status_log_path)
            logger.info("Status log loaded successfully.")
        else:
            logger.info("No status log found. Creating a new one.")
            data = self._get_summary()
            ds = self._create_status_log(data)
            io.writers.save_netcdf_with_cf_check(ds, self.status_log_path)
            logger.info(f"Status log created at {self.status_log_path}")
        logger.info(ds)

    def _get_summary(self) -> dict:
        """Parse stdout files and extract summary data.

        Returns
        -------
        dict
            Dictionary containing parsed log metadata for all simulations.
        """
        stdout_files = [
            sim_dir / CONFIG.STD_OUT_FILE_NAME[self.model]
            for sim_dir in self.simulation_entries
        ]
        logger.info(f"Parsing {self.model} stdout files for summary.")
        logs = []
        for file in tqdm(stdout_files):
            if file.exists():
                logs.append(
                    io.read_gramm_stdout(str(file))
                    if self.model == "gramm"
                    else io.read_gral_stdout(str(file))
                )
            else:
                logs.append(
                    models.GRAMMLogMetadata()
                    if self.model == "gramm"
                    else models.GRALLogMetadata()
                )

        data = {}
        for log in logs:
            for key in log.__dataclass_fields__.keys():
                var = getattr(log, key)
                data[key] = data.get(key, []) + [var]

        return data

    def _calculate_disk_space(self) -> list:
        """Calculate disk space used by each simulation directory.

        Returns
        -------
        list
            List of disk space in bytes for each simulation directory.
        """
        # Use multiprocessing to parallelize du calls
        disk_space = [0] * len(self.simulation_entries)
        with ProcessPoolExecutor() as executor:
            # Submit all tasks
            future_to_idx = {
                executor.submit(_get_dir_size, sim_dir): idx
                for idx, sim_dir in enumerate(self.simulation_entries)
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_idx),
                total=len(self.simulation_entries),
                desc="Calculating disk space",
            ):
                idx = future_to_idx[future]
                disk_space[idx] = future.result()

        return disk_space

    def _create_status_log(self, data: dict) -> xr.Dataset:
        """Create an xarray Dataset from parsed log data.

        Parameters
        ----------
        data : dict
            Dictionary containing parsed log metadata.

        Returns
        -------
        xr.Dataset
            Dataset containing simulation status information.
        """
        # Get maximum number of time steps across all simulations
        n_time_steps = 0
        for key in data.keys():
            if isinstance(data[key][0], list):
                for lst in data[key]:
                    n_time_steps = max(n_time_steps, len(lst))

        ds = xr.Dataset(
            coords={"sim_id": self.sim_ids, "iteration": np.arange(n_time_steps)}
        )

        # Add metadata to coordinates
        ds["sim_id"].attrs = {
            "long_name": "Simulation ID",
            "description": "Unique identifier for each simulation",
        }
        ds["iteration"].attrs = {
            "long_name": "Iteration step",
            "description": "Iteration step index for time-varying simulation "
            "parameters",
        }

        # Define metadata for each variable
        variable_metadata = {
            "version": {
                "long_name": "Model version",
                "description": "GRAMM/GRAL version string",
            },
            "plattform": {
                "long_name": "Platform",
                "description": "Operating system platform",
            },
            "dotnet_version": {
                "long_name": ".NET version",
                "description": ".NET runtime version",
            },
            "n_processors": {
                "long_name": "Number of processors",
                "description": "Number of CPU cores used",
            },
            "ggeom_file_read": {
                "long_name": "Geometry file read status",
                "description": "Whether geometry file was successfully read",
            },
            "min_elevation": {
                "long_name": "Minimum elevation",
                "units": "m",
                "description": "Minimum terrain elevation in domain",
            },
            "max_elevation": {
                "long_name": "Maximum elevation",
                "units": "m",
                "description": "Maximum terrain elevation in domain",
            },
            "init_wind_speed": {
                "long_name": "Initial wind speed",
                "units": "m s-1",
                "description": "Initial wind speed for simulation",
            },
            "init_direction": {
                "long_name": "Initial wind direction",
                "units": "degree",
                "description": "Initial wind direction",
            },
            "u_component": {
                "long_name": "U wind component",
                "units": "m s-1",
                "description": "Zonal wind component",
            },
            "v_component": {
                "long_name": "V wind component",
                "units": "m s-1",
                "description": "Meridional wind component",
            },
            "init_stability_class": {
                "long_name": "Initial stability class",
                "description": "Atmospheric stability class",
            },
            "init_obukhov_length": {
                "long_name": "Initial Obukhov length",
                "units": "m",
                "description": "Initial Obukhov length for stability",
            },
            "roughness_length": {
                "long_name": "Surface roughness length",
                "units": "m",
                "description": "Aerodynamic roughness length",
            },
            "init_boundary_layer_height": {
                "long_name": "Initial boundary layer height",
                "units": "m",
                "description": "Initial height of atmospheric boundary layer",
            },
            "friction_velocity": {
                "long_name": "Friction velocity",
                "units": "m s-1",
                "description": "Surface friction velocity",
            },
            "simulation_attempt": {
                "long_name": "Simulation attempt number",
                "description": "Iteration attempt number",
            },
            "simulation_time": {
                "long_name": "Simulation time",
                "units": "s",
                "description": "Total simulation time",
            },
            "simulation_timestep": {
                "long_name": "Simulation timestep",
                "units": "s",
                "description": "Time step size",
            },
            "simulation_divergence": {
                "long_name": "Simulation divergence",
                "description": "Maximum divergence value",
            },
        }

        for key, values in data.items():
            if isinstance(values[0], list):
                arr = np.full((len(values), n_time_steps), np.nan)
                for i, lst in enumerate(values):
                    arr[i, : len(lst)] = lst
                ds[key] = (("sim_id", "iteration"), arr)
            else:
                ds[key] = (("sim_id",), values)

            # Add metadata if available
            if key in variable_metadata:
                ds[key].attrs = variable_metadata[key]

        for key in ds.data_vars:
            if ds[key].dtype == np.float64:
                ds[key] = ds[key].astype(np.float32)

        # Add disk space variable
        logger.info("Calculating disk space usage for each simulation directory.")
        disk_space = self._calculate_disk_space()
        ds["disk_space_bytes"] = (("sim_id",), disk_space)
        ds["disk_space_bytes"].attrs = {
            "long_name": "Disk space used by simulation directory",
            "units": "bytes",
        }

        # Calculate total disk space
        total_disk_space_bytes = sum(disk_space)
        total_disk_space_gb = total_disk_space_bytes / (1024**3)

        ds.attrs = {
            "Conventions": "CF-1.11",
            "title": f"{self.model.capitalize()} simulation logs",
            "description": (
                f"Parsed {self.model.capitalize()} log files from multiple simulations."
            ),
            "history": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "disk_usage_summary": (
                f"Total disk space used: {total_disk_space_gb:.2f} "
                f"GiB ({total_disk_space_bytes} bytes)."
            ),
            "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "simulation_path": self.simulation_path.as_posix(),
            "model": self.model,
        }
        return ds

    def load_config_files(self) -> None:
        if self.model == "gramm":
            raise NotImplementedError("GRIMM config loading not implemented yet.")
        elif self.model == "gral":
            self.config = io.read_gral_config(
                gral_geb_path=self.catalog_path / CONFIG.CONFIG_PATH / "GRAL.geb",
                in_dat_path=self.catalog_path / CONFIG.CONFIG_PATH / "in.dat",
            )

    def compute_vertical_gradients(self, n_processes: int) -> xr.Dataset:
        logger.info("Load config files")
        self.load_config_files()

        logger.info("Open catalog status log")
        ds_status = xr.load_dataset(self.status_log_path)

        if "wind_speed_vertical_gradient" in ds_status.data_vars:
            logger.info("Vertical wind speed gradients already computed in status log.")
        else:
            logger.info(f"Processing wind gradients using {n_processes} threads.")
            with ThreadPoolExecutor(max_workers=n_processes) as executor:
                futures = {
                    executor.submit(
                        _process_wind_gradient, p, self.model, self.config
                    ): p
                    for p in self.simulation_entries
                }
                gradients = [
                    future.result()
                    for future in tqdm(
                        as_completed(futures),
                        total=len(self.simulation_entries),
                        desc="Wind gradients",
                    )
                ]
            ds_status["wind_speed_vertical_gradient"] = xr.concat(
                gradients, dim="sim_id", join="outer"  # type: ignore
            )  # type: ignore
            logger.info("Saving updated status log with vertical gradients.")
            io.writers.save_netcdf_with_cf_check(ds_status, self.status_log_path)

        # Process concentration gradients with incremental saving
        if "concentration_vertical_profile" not in ds_status.data_vars:
            # Initialize the variable if it doesn't exist
            nz = self.config["n_horizontal_slices_concentration"]
            vertical_coords = self.config["horizontal_slices_concentration"]
            empty_profile = xr.DataArray(
                np.full((len(self.sim_ids), nz), np.nan, dtype=np.float32),
                dims=["sim_id", "vertical_level"],
                coords={
                    "sim_id": self.sim_ids,
                    "vertical_level": vertical_coords,
                },
            )
            ds_status["concentration_vertical_profile"] = empty_profile
            io.writers.save_netcdf_with_cf_check(ds_status, self.status_log_path)
            logger.info("Initialized concentration_vertical_profile variable.")

        # Find which simulations still need processing
        already_computed = (
            ~ds_status["concentration_vertical_profile"].isel(vertical_level=0).isnull()
        )
        sim_ids_to_process = [
            sim_id
            for sim_id, is_computed in zip(self.sim_ids, already_computed.values)
            if not is_computed
        ]

        if not sim_ids_to_process:
            logger.info("All concentration gradients already computed.")
        else:
            logger.info(
                f"Processing {len(sim_ids_to_process)} remaining concentration "
                f"gradients using {n_processes} processes (out of {len(self.sim_ids)} "
                f"total)."
            )

            # Get simulation paths for the simulations to process
            sim_entries_to_process = [
                self.simulation_entries[self.sim_ids.index(sim_id)]
                for sim_id in sim_ids_to_process
            ]

            # Process in batches of 64
            batch_size = 32
            for batch_start in range(0, len(sim_ids_to_process), batch_size):
                batch_end = min(batch_start + batch_size, len(sim_ids_to_process))
                batch_sim_ids = sim_ids_to_process[batch_start:batch_end]
                batch_sim_entries = sim_entries_to_process[batch_start:batch_end]

                logger.info(
                    f"Processing batch {batch_start // batch_size + 1} "
                    f"({batch_start + 1}-{batch_end} of {len(sim_ids_to_process)})"
                )

                # Process this batch
                with ThreadPoolExecutor(max_workers=n_processes) as executor:
                    futures = {
                        executor.submit(
                            _process_concentration_gradient, p, self.config
                        ): i
                        for i, p in enumerate(batch_sim_entries)
                    }
                    gradients = [None] * len(batch_sim_entries)
                    for future in tqdm(
                        as_completed(futures),
                        total=len(batch_sim_entries),
                        desc=f"Batch {batch_start // batch_size + 1}",
                    ):
                        idx = futures[future]
                        gradients[idx] = future.result()  # type: ignore

                # Update the dataset with the batch results (vectorized operation)
                logger.info("Updating status log with batch results.")
                # Stack gradients into a single DataArray for efficient batch update
                batch_gradients = xr.concat(gradients, dim="sim_id")  # type: ignore
                batch_gradients["sim_id"] = batch_sim_ids

                # Use vectorized indexing instead of loop
                for i, sim_id in enumerate(batch_sim_ids):
                    sim_idx = self.sim_ids.index(sim_id)
                    ds_status["concentration_vertical_profile"].values[sim_idx, :] = (
                        batch_gradients.values[i, :]
                    )

                # Save intermediate results
                logger.info(
                    f"Saving intermediate results "
                    f"(batch {batch_start // batch_size + 1})"
                )
                io.writers.save_netcdf_with_cf_check(ds_status, self.status_log_path)

            logger.info("All concentration gradients computed and saved.")

        return ds_status
