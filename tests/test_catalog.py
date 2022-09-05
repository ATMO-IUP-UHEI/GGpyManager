import time

from context import data_path
from context import Simulation, Status
from context import Catalog


# Minimal working example
def create_input(data_path):
    """
    Create input for a minimum working simulation.

    Parameters
    ----------
    get_data_path : Path
        Tempory directory generator.

    Returns
    -------
    input: list
        Parameters for the initialization of `Simulation`.
    """
    catalog_path = data_path / "catalog"
    catalog_path.mkdir()
    sim_path = data_path / "sim"
    sim_path.mkdir()

    # Create fake input files and directories
    config_path = catalog_path / "config"
    config_path.mkdir()
    meteo_path = catalog_path / "meteopgt.all"
    meteo_text = "header1\nheader2\nmeteo1\n\n\n"
    with open(meteo_path, "w") as file:
        file.write(meteo_text)

    # link_target_path_list = []
    # targets = ["A.all", "B.con", "C.test"]
    # for target in targets:
    #     target_path = data_path / target
    #     target_path.touch()
    #     link_target_path_list.append(target_path)
    # link_name_list = targets
    return [catalog_path, sim_path]


def test_catalog(data_path):
    catalog = Catalog(*create_input(data_path), read_only=False)
    catalog.get_info()
    # simulation.run()
    # time.sleep(1)
    # assert simulation.get_status() == Status.finished
