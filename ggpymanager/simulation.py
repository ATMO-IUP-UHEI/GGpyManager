"""Module for the management of one GRAL simulation.

The module contains the class `Simulation` which can be used to initialize, run, and 
analyse one simulation run of a GRAL simulation
"""
from subprocess import Popen, PIPE
from dataclasses import dataclass


class State:
    """
    Class to define states for comparison and verbose output.

    Examples
    --------
    >>> small_error = State("small error", -1)
    >>> big_error = State("big error", -1)
    >>> small_error == big_error
    True
    >>> print(small_error)
    small error
    """

    def __init__(self, name, value) -> None:
        self.name = name
        self.value = value

    def __eq__(self, __o: object) -> bool:
        if self.value == __o.value:
            return True
        else:
            return False

    def __repr__(self) -> str:
        return self.name


@dataclass
class Status:
    """
    Status names for the simulation.
    """

    init: State = State("init", 0)
    running: State = State("running", 1)
    finished: State = State("finished", 2)
    cleaned: State = State("cleaned", 3)
    error: State = State("error", -1)


class Simulation:
    """
    Manages the configuration and run of a single GRAL simulation.

    From the paths, all given files are linked and the folder structure of the
    simulation is created. After the simulation is initialized, it can be started and
    polled through the class.

    Parameters
    ----------
    catalog_path : pathlib Path
    sim_sub_path : pathlib Path
    link_target_path_list : list of pathlib Path
    link_name_list : list of str
    meteo_text : str

    Attributes
    ----------
    catalog_path : pathlib Path
    sim_sub_path : pathlib Path
    link_target_path_list : list of pathlib Path
    link_name_list : list of str
    meteo_text : str
    logfile_path : pathlib Path
    error_logfile_path : pathlib Path

    Methods
    -------
    run()
    get_status()
    get_paths(suffix=None)    
    """

    def __init__(
        self,
        catalog_path,
        sim_sub_path,
        link_target_path_list,
        link_name_list,
        meteo_text,
        read_only=True,
    ):
        # Init variables
        self.catalog_path = catalog_path
        self.sim_sub_path = sim_sub_path

        self.link_target_path_list = link_target_path_list
        self.link_name_list = link_name_list
        self.meteo_text = meteo_text

        self.logfile_path = self.sim_sub_path / "{}_log.txt".format(
            self.sim_sub_path.name
        )
        self.error_logfile_path = self.sim_sub_path / "{}_error_log.txt".format(
            self.sim_sub_path.name
        )

        # Not implemented yet
        self.file_list_for_storage = []

        # Read or create simulation directory
        if self.sim_sub_path.exists():
            self.read_dir()
        elif not read_only:
            self.setup_input()

    def test_for_init(self):
        # Check if meteo file is identical
        meteo_file_path = self.sim_sub_path / "meteopgt.all"
        if meteo_file_path.exists():
            with open(meteo_file_path, "r") as meteo_f:
                if meteo_f.read() != self.meteo_text:
                    self.status = State('"meteopgt.all" not identical', -1)
                    return -1
        else:
            self.status = State('"meteopgt.all" not in folder', -1)
            return -1
        # Check if all links and log-files exist
        for p in self.sim_sub_path.iterdir():
            if p.is_symlink():
                try:
                    self.link_name_list.remove(p.name)
                    self.link_target_path_list.remove(p.readlink())
                except:
                    pass
                    # Cannot raise error if unknown link is present, because 
                    # wind files from a previous GRAL run could be linked as well.
                    # self.status = State("Unknown symlink {}".format(p), -1)
                    # return -1
        if len(self.link_name_list) != 0 or len(self.link_target_path_list) != 0:
            self.status = State(
                name="Links missing {} or {}".format(
                    self.link_name_list, self.link_target_path_list
                ),
                value=-1,
            )
            return -1
        if not self.logfile_path.exists() or not self.error_logfile_path.exists():
            self.status = State("Logfile or error logfile not found", -1)
            return -1

    def test_for_running(self):
        # Running check needs to be implemented (maybe with PID check)
        return 0

    def test_for_finished(self):
        # Check if log file is written
        if self.logfile_path.stat().st_size > 0:
            log_last_line_finished = (
                "GRAL simulations finished. Press any key to continue..."
            )
            try:
                with self.logfile_path.open() as logfile:
                    log = logfile.read()
                    assert log.split("\n")[-2] == log_last_line_finished
                self.status = Status.finished
            except:
                self.status = State("Gral error", -1)
                return -1

        # The error logfile is also written, when the "Press any key..." input
        # is wrong.
        """ # Check if error file is written
        if self.error_logfile_path.stat().st_size > 0:
            status = -2 """

    def read_dir(self):
        """
        Read the simulation directory and the check the state of the directory.
        """
        # Asssume initialized
        self.status = Status.init
        self.test_for_init()
        if not self.status == Status.error:
            self.test_for_running()
        if not self.status == Status.error:
            self.test_for_finished()

    def setup_input(self):
        """
        Creates the directory for the simulation and links all input files from the
        `catalog_path`.
        """
        # Create directory
        self.sim_sub_path.mkdir()
        # Link input files
        for n, t in zip(self.link_name_list, self.link_target_path_list):
            link_path = self.sim_sub_path / n
            link_path.symlink_to(t)
        # Create new single line "meteopgt.all" file.
        with open(self.sim_sub_path / "meteopgt.all", "w") as meteo_f:
            meteo_f.write(self.meteo_text)
        # Create the logfiles
        self.logfile_path.touch()
        self.error_logfile_path.touch()
        # Set new status
        self.status = Status.init

    def run(self):
        """
        If the simulation is initialized, the GRAL run is started.
        """
        self.test_for_init()
        self.test_for_running()
        self.test_for_finished()
        if self.status == Status.init:
            # Open logfiles
            self.logfile = self.logfile_path.open("w")
            self.error_logfile = self.error_logfile_path.open("w")

            compile_exe = "/mnt/data/users/svardag/dotnetcore/version_2_1_0/dotnet"
            self.process = Popen(
                [compile_exe, "GRAL_exe.dll"],
                stdin=PIPE,
                stdout=self.logfile,
                stderr=self.error_logfile,
                cwd=self.sim_sub_path,
            )
            # Write ENTER into the pipeline to terminate GRAL after finishing the run.
            self.process.stdin.write(b"\n")

            self.status = Status.running

    def get_status(self):
        """
        Get the current status of the simulation.

        Returns
        -------
        status: State
            Current status of the simulation.
        """
        # Check if already finished
        if self.status == Status.running:
            # If process terminates, send enter to close:
            # self.process.communicate("\n") # Already in run
            if self.process.poll() != None:
                # Close logfiles
                self.logfile.close()
                self.error_logfile.close()
                self.status = Status.finished
        return self.status

    def get_paths(self, suffix=None):
        """
        Return a list of paths in the simulation directory. The paths can be filtered by
        their suffix

        Parameters
        ----------
        suffix : str, optional
            Suffix of the paths, by default None

        Returns
        -------
        path_list : list of  str
            All paths with the specified suffix.
        """
        path_list = []
        for p in self.sim_sub_path.iterdir():
            if suffix is None:
                path_list.append(p)
            elif p.suffix == suffix:
                path_list.append(p)
        return path_list
        

    def __repr__(self):
        return "Sim {} status: {}.".format(
            self.sim_sub_path.name, self.status
        )

    def clean_simulation_path(self):
        pass
