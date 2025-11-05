"""Module to manage the GRAMM/GRAL catalog.

The module checks consistency of the simulations and can create a list of simulations
which can be started to efficiently use the available computation power.
"""

import logging
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import xarray as xr
from tqdm import tqdm

import ggpymanager.config as CONFIG
from ggpymanager import io, models


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
        logging.warning(f"Error calculating disk space for {sim_dir}: {e}")
        return 0


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
        logging.info(f"Scanning catalog {self.catalog_path}")
        self._check_input_files()
        self._check_simulations()

        self._check_status_log()

    def _check_input_files(self) -> None:
        """Check if all required input files are present in the config directory."""
        self.config_path = self.catalog_path / CONFIG.CONFIG_PATH
        logging.info(f"Checking input files in directory: {self.config_path}")
        self.input_files = CONFIG.INPUT_FILES[self.model]
        missing_files = []
        for file in tqdm(self.input_files):
            file_path = self.config_path / file
            if not file_path.exists():
                missing_files.append(file)
        if missing_files:
            logging.warning(
                "The following input files are missing in "
                f"{self.config_path}: {missing_files}"
            )
        else:
            logging.info("All required input files are present.")

    def _check_simulations(self) -> None:
        """Scan simulation directory and count completed simulations and wind files."""
        self.simulation_path = self.catalog_path / CONFIG.SIMULATION_PATH
        logging.info(f"Checking simulations in directory: {self.simulation_path}")

        self.simulation_entries = []
        self.sim_ids = []
        self.n_completed_simulations = 0
        self.n_wind_files = 0

        if not self.simulation_path.exists():
            logging.warning(f"Simulation path does not exist: {self.simulation_path}")
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

        logging.info(
            f"Found {len(self.simulation_entries)} simulation entries in the catalog."
        )
        logging.info(
            f"{self.n_completed_simulations} simulations are marked as completed."
        )
        logging.info(f"{self.n_wind_files} simulations have computed wind fields.")

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
            logging.warning(f"Error reading stdout file {stdout_file}: {e}")
            return False

    def _check_status_log(self) -> None:
        """Check for existing status log or create a new one."""
        self.status_log_path = self.catalog_path / CONFIG.STATUS_LOG_FILE_NAME
        if self.status_log_path.exists():
            logging.info(f"Status log found at {self.status_log_path}")
            ds = xr.load_dataset(self.status_log_path)
            logging.info("Status log loaded successfully.")
        else:
            logging.info("No status log found. Creating a new one.")
            data = self._get_summary()
            ds = self._create_status_log(data)
            ds.to_netcdf(self.status_log_path)
            logging.info(f"Status log created at {self.status_log_path}")
        logging.info(ds)

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
        logging.info(f"Parsing {self.model} stdout files for summary.")
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
            coords={"sim_id": self.sim_ids, "time": np.arange(n_time_steps)}
        )
        for key, values in data.items():
            if isinstance(values[0], list):
                arr = np.full((len(values), n_time_steps), np.nan)
                for i, lst in enumerate(values):
                    arr[i, : len(lst)] = lst
                ds[key] = (("sim_id", "time"), arr)
            else:
                ds[key] = (("sim_id",), values)

        # Add disk space variable
        logging.info("Calculating disk space usage for each simulation directory.")
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
            "title": f"{self.model.capitalize()} simulation logs",
            "description": (
                f"Parsed {self.model.capitalize()} log files from multiple simulations."
            ),
            "disk_usage_summary": (
                f"Total disk space used: {total_disk_space_gb:.2f} "
                f"GiB ({total_disk_space_bytes} bytes)."
            ),
            "created_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "simulation_path": self.simulation_path.as_posix(),
            "model": self.model,
        }
        return ds
