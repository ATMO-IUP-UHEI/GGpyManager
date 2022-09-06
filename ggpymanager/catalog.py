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

    def __init__(self, catalog_path, config_path, sim_path, read_only=True):
        self.catalog_path = Path(catalog_path)
        self.sim_path = Path(sim_path)
        self.read_only = read_only

        self.config_path = config_path
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
        for i in range(1, len(meteopgt) + 1):
            if meteopgt[-i] == "":
                limit = -i
            else:
                break
        meteopgt = meteopgt[:limit]
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
            "Total wind situations": self.total_sim,
            "Missing GRAMM simulations": self.get_gramm_missing(),
            "Missing GRAL simulations": self.get_gral_missing(),
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

    def init_simulations(self, n_limit=None):
        """
        Initialize the missing GRAL simulations in the simulation directory. The number
        of new simulations can be limited for performance or testing reasons.

        Parameters
        ----------
        n_limit : in, optional
            Limit of simulations to initialize, by default None
        """
        gral_missing_list = self.get_gral_missing()

        # Only add a limited amount of simulations
        if n_limit is not None:
            gral_missing_list = gral_missing_list[:n_limit]

        for meteo_number in gral_missing_list:
            # Create a subpath from the `meteo_number`
            sim_sub_path = self.sim_path / "{:05}_sim".format(meteo_number)
            link_target_path_list, link_name_list = self.create_link_list(meteo_number)
            meteo_text = self.create_meteo_text(meteo_number)

            self.simulations.append(
                Simulation(
                    self.catalog_path,
                    sim_sub_path,
                    link_target_path_list,
                    link_name_list,
                    meteo_text,
                    self.read_only,
                )
            )

    def create_link_list(self, meteo_number):
        """
        Generate a link list for a given meteo situation.

        Parameters
        ----------
        meteo_number : int
            Meteo situation referenced by the number.

        Returns
        -------
        link_target_path_list : list of Path
            List of the targets which need to be linked for GRAL as input.
        link_name_list : list of str
            List of the file names for the links in the simulation directory.
        """
        link_target_path_list, link_name_list = self.create_link_list_config()
        lists_wind = self.create_link_list_wind(meteo_number)
        link_target_path_list += lists_wind[0]
        link_name_list += lists_wind[1]
        return link_target_path_list, link_name_list

    def create_link_list_config(self):
        link_target_path_list = [target for target in self.config_path.iterdir()]
        link_name_list = [target.name for target in link_target_path_list]
        return link_target_path_list, link_name_list

    def create_link_list_wind(self, meteo_number):
        link_target_path_list = []
        link_name_list = []
        for suffix in [".wnd", ".scl"]:
            target_name = "{:05}{}".format(meteo_number, suffix)
            link_target_path_list.append(self.catalog_path / target_name)
            link_name_list.append(
                "{:05}{}".format(1, suffix)
            )  # Use only one meteo situation per simulation
        return link_target_path_list, link_name_list

    def create_meteo_text(self, meteo_number):
        meteo_id = meteo_number - 1
        n_line = meteo_id + N_HEADER
        meteo_text = "\n".join(
            self.meteopgt[:N_HEADER] + self.meteopgt[n_line : n_line + 1]
        )
        return meteo_text

    def get_simulations(self, status=None):
        """
        Returns all simulations with the specified status.

        Parameters
        ----------
        status : State, optional
            Status of the simulation, by default None

        Returns
        -------
        simulations : list of Simulation
            List of simulations with the specified state.
        """
        if status is None:
            simulations = self.simulations
        else:
            simulations = []
            for simulation in self.simulations:
                if simulation.status == status:
                    simulations.append(simulation)
        return simulations

    def print_runtime_info(self, start, len_queue, len_parallel_simulations):
        current_time = time.time() - start
        days = current_time // (24 * 60 * 60)
        hours = current_time // (60 * 60)
        min = current_time // 60
        print(
            "Queue size: {} Currently running: {} running since {} days {}:{}\r".format(
                len_queue,
                len_parallel_simulations,
                days,
                hours,
                min,
            )
        )

    def run_simulations(self, n_limit=None):
        if self.read_only:
            print(
                "Catalog is set to read-only. If you wish to run simulations, please"
                + "initialize with 'read_only = False'."
            )
            return -1

        queue = self.get_simulations(Status.init)

        # Limit queue length for testing or performance reasons
        if n_limit is not None:
            queue = queue[n_limit]

        # Limit to 5 parallel processes as 12 threads are used per simulation
        # on a server with 72 threads. The number of parallel processes should not be
        # higher than 5 as the RAM of the server is not sufficient.
        n_parallel_max = 5
        parallel_simulations = []
        start = time.time()
        while len(queue) > 0:
            # Start new simulation if space is available
            if len(parallel_simulations) < n_parallel_max:
                new_simulation = queue.pop()
                new_simulation.run()
                parallel_simulations.append(new_simulation)
            # Throw out terminated simulations
            for simulation in parallel_simulations:
                if not simulation.get_status() == Status.running:
                    parallel_simulations.remove(simulation)
            # Display info
            time.sleep(2)
            self.print_runtime_info(start, len(queue), len(parallel_simulations))            

    def wait_for_simulations(self):
        # Count running simulations
        parallel_simulations = []
        for simulation in self.simulations:
            if simulation.status == Status.running:
                parallel_simulations.append(simulation)

        if len(parallel_simulations) > 0:
            print("Wait with me until all simulations are finished.")
            len_queue = 0 # No queue left, only running simulations
            start = time.time()
            while len(parallel_simulations) > 0:
                # Throw out terminated simulations
                for simulation in parallel_simulations:
                    if not simulation.get_status() == Status.running:
                        parallel_simulations.remove(simulation)
                # Display info
                time.sleep(2)
                self.print_runtime_info(start, len_queue, len(parallel_simulations))

    def collect_simulations(self):
        # print("I am collecting the results of the simulations!")
        pass

    def __del__(self):
        if not self.read_only:
            self.wait_for_simulations()
            self.collect_simulations()
