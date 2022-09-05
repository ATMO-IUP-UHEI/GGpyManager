"""Modul to manage the GRAMM/GRAL catalog.

The modul checks consistency of the simulations and can create a list of simulations
which can be started to efficiently use the available computation power.
"""

import time
from pathlib import Path

from .simulation import Simulation, Status

N_HEADER = 2  # Number of header lines in the "meteopgt.all" file


class Catalog:
    """
    Class to manage the catalog of GRAMM/GRAL.

    For every GRAMM situation one GRAL situation can be computed which
    results in as many concentration maps as there are specified source
    groups.

    Parameters
    ----------
    catalog_path : like `pathlib.Path`
        Path to the GRAMM simulation results (see Notes).
    sim_path : like `pathlib.Path`
        Path to the GRAL simulation directory (see Notes).
    read_only : bool, optional
        Flag if the catalog should only be read.

    Attributes
    ----------
    catalog_path : like `pathlib.Path`
        Path to the GRAMM simulation results (see Notes).
    sim_path : like `pathlib.Path`
        Path to the GRAL simulation directory (see Notes).
    config_path : like `pathlib.Path`
        Path to the config directory inside the GRAMM directory (see Notes).
    meteopgt : list of str
        List of the lines of the file "meteopgt.all".
    simulations : list of Simulations
        List with all GRAL simulations associated with this catalog (see Notes).

    Methods
    -------
    get_status()
    read_dir()
    add_simulations(n_simulations)
    create_link_list(meteo_number)
    run_simulations(n_simulations)
    wait_for_simulations()
    collect_simulations()

    Notes
    -----
    Path structure:

    - `catalog_path`

        - "config"

            - [config files]
        - "meteopgt.all"
        - [GRAMM ".scl" and ".wnd" files]

    - `sim_path`

        - ["sim_group_####"]

            - [data from GRAL simulation ####]

    """

    def __init__(self, catalog_path, sim_path, read_only=True):
        self.catalog_path = Path(catalog_path)
        self.sim_path = Path(sim_path)
        self.read_only = read_only

        self.config_path = self.catalog_path / "config"
        self.meteopgt = self.get_meteopgt()

        self.total_sim = len(self.meteopgt) - N_HEADER
        self.simulations = []

    def get_meteopgt(self):
        """
        Generate a list of lines from the "meteopgt.all" file.

        Returns
        -------
        meteopgt: list of str
            Lines from the "meteopgt.all" file.
        """
        with open(self.catalog_path / "meteopgt.all") as file:
            meteopgt_text = file.read()
        meteopgt = meteopgt_text.split("\n")
        # Drop trailing empty lines
        for i in range(len(meteopgt)):
            if meteopgt[-i] == "":
                meteopgt.pop(-i)
            else:
                break
        return meteopgt

    def get_info(self):
        """
        Gives information about the catalog.

        Returns
        -------
        info : dict
            Dictionary with information about the GRAMM and the GRAL simulations.
        """
        info = {
            "Total wind situations":self.total_sim,
            "Missing GRAMM simulations":self.get_gramm_missing(),
            "Missing GRAL simulations":self.get_gral_missing(),
        }
        return info

    def get_missing_files(self, search_path, suffix):
        """
        Finds missing files for each meteo situations from the given suffix in the
        search_path and one step of recursion.

        Parameters
        ----------
        search_path : Path
            Path to look for files.
        suffix : str
            Suffix of the filetype to look for.

        Returns
        -------
        missing_list : list of int
            List of meteo_numbers for which the file with the suffix is not present.
        """
        missing_list = [i + 1 for i in range(self.total_sim)]
        for path in search_path.iterdir():
            # Search in subdirectories
            if path.is_dir():
                for subpath in path.iterdir():
                    if subpath.suffix == suffix:
                        meteo_number = int(path.stem)
                        missing_list.remove(meteo_number)
                        break
            elif path.suffix == suffix:
                meteo_number = int(path.stem)
                missing_list.remove(meteo_number)
        return missing_list

    def get_gramm_missing(self):
        gramm_missing_list = self.get_missing_files(
            search_path=self.catalog_path, suffix=".wnd"
        )
        return gramm_missing_list

    def get_gral_missing(self):
        gral_missing_list = self.get_missing_files(
            search_path=self.sim_path, suffix=".gff"
        )
        return gral_missing_list

    def add_simulations(self, n_simulations=None):
        _, _, gral_missing_list = self.read_dir()
        meteo_list = gral_missing_list

        if n_simulations != None:
            meteo_list = meteo_list[:n_simulations]

        for meteo_number in meteo_list:
            sim_sub_path = self.sim_path / "{:05}_sim".format(meteo_number)
            link_target_path_list, link_name_list = self.create_link_list(meteo_number)

            meteo_id = meteo_number - 1
            n_line = meteo_id + N_HEADER
            meteo_text = "\n".join(
                self.meteopgt[:N_HEADER] + self.meteopgt[n_line : n_line + 1]
            )

            self.simulations.append(
                Simulation(
                    self.catalog_path,
                    self.config_path,
                    sim_sub_path,
                    link_target_path_list,
                    link_name_list,
                    meteo_text,
                )
            )

    def create_link_list(self, meteo_number):
        link_target_path_list = [target for target in self.config_path.iterdir()]
        link_name_list = [target.name for target in link_target_path_list]

        for suffix in [".wnd", ".scl"]:
            target_name = "{:05}{}".format(meteo_number, suffix)
            link_target_path_list.append(self.catalog_path / target_name)
            link_name_list.append(
                "{:05}{}".format(1, suffix)
            )  # Use only one meteo situation per simulation
        return link_target_path_list, link_name_list

    def run_simulations(self, n_simulations=None):
        # todo: make more readable and cleaner and connect with
        # 'wait_for_simulations'

        if self.read_only:
            print(
                "Catalog is set to read-only. If you wish to run simulations, please \
initialize with 'read_only = False'."
            )
            return -1

        n_initialized = 0
        for sim in self.simulations:
            if sim.get_status() == Status.init:
                n_initialized += 1

        if n_simulations == None:
            n_simulations = n_initialized

        # Limit to 5 parallel processes as 12 threads are used per simulation
        # on a server with 72 threads. The number of parallel processes should not be
        # higher than 5 as the RAM of the server is not sufficient.
        n_limit = 5
        start = time.perf_counter()
        while True:
            n_running = 0
            n_queue = 0
            for sim in self.simulations:
                if sim.get_status() == Status.running:
                    n_running += 1
                if sim.get_status() == Status.init:
                    n_queue += 1

            if n_queue == 0 and n_running == 0:
                break

            for sim in self.simulations:
                if n_running >= n_limit or (n_initialized - n_queue) >= n_simulations:
                    break
                if sim.get_status() == Status.init:
                    sim.run()
                    n_running += 1
                    n_queue -= 1

            time.sleep(2)
            current_time = time.perf_counter() - start
            print(
                "Queue size: {} Currently running: {} Time running: {} min \r".format(
                    n_queue, n_running, int(current_time) // 60
                )
            )

    def wait_for_simulations(self):
        print("Wait with me until all simulations are finished.")
        start = time.perf_counter()
        while True:
            counter = len(self.simulations)
            for sim in self.simulations:
                # print("{}\r".format(STATUS_NAMES[sim.get_status()]))
                if sim.get_status() != Status.running:
                    counter -= 1

            current_time = time.perf_counter() - start
            # To do: add display for hours and minutes
            print(
                "{} Simulations still running for {}s. \r".format(
                    counter, int(current_time)
                ),
                end="",
            )
            if counter == 0:
                break
            time.sleep(2)

    def collect_simulations(self):
        print("I am collecting the results of the simulations!")

    def __del__(self):
        if not self.read_only:
            self.wait_for_simulations()
            self.collect_simulations()
