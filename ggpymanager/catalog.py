"""Modul to manage the GRAMM/GRAL catalog.

The modul checks consistency of the simulations and can create a list of simulations
which can be started to efficiently use the available computation power.
"""

import time
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import ggpymanager.config as CONFIG
from typing import Literal


class Catalog:
    def __init__(self, catalog_path: str | Path, model: Literal["gramm", "gral"]):
        if model not in ["gramm", "gral"]:
            raise ValueError(f"Model must be 'gramm' or 'gral', got {model}")
        self.model = model
        self.catalog_path = Path(catalog_path)
        logging.info(f"Scannin catalog {self.catalog_path}")
        self._check_input_files()
        self._check_simulations()

        self._check_status_log()

    def _check_input_files(self):
        self.config_path = self.catalog_path / CONFIG.CONFIG_PATH
        logging.info(f"Checking input files in directory {self.config_path}")
        self.input_files = CONFIG.INPUT_FILES[self.model]
        missing_files = []
        for file in self.input_files:
            file_path = self.config_path / file
            if not file_path.exists():
                missing_files.append(file)
        if missing_files:
            logging.info(
                "The following input files are missing in "
                f"{self.config_path}: {missing_files}"
            )
        else:
            logging.info("All required input files are present.")

    def _check_simulations(self):
        self.simulation_path = self.catalog_path / CONFIG.SIMUATION_PATH
        logging.info(f"Checking simulations in directory {self.simulation_path}")

        if not self.simulation_path.exists():
            logging.warning(f"Simulation path does not exist: {self.simulation_path}")
            self.simulation_entries = []
            self.n_completed_simulations = 0
            self.n_wind_files = 0
            return

        self.simulation_entries = []
        self.n_completed_simulations = 0
        self.n_wind_files = 0

        for sim_dir in self.simulation_path.iterdir():
            if not (sim_dir.is_dir() and sim_dir.name.startswith("sim_")):
                continue

            self.simulation_entries.append(sim_dir)

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
        """Check if a simulation is marked as completed based on stdout file."""
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

    def _check_status_log(self):
        self.status_log_path = self.catalog_path / "status_log.yaml"
        if self.status_log_path.exists():
            logging.info(f"Status log found at {self.status_log_path}.")
        else:
            logging.info("No status log found.")
