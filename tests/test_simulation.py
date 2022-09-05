import time

from context import data_path
from context import Simulation, Status


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
    sim_sub_path = data_path / "sim"
    # sim_sub_path.mkdir()
    config_path = data_path / "config"
    config_path.mkdir()

    # Create fake input files
    link_target_path_list = []
    targets = ["A.all", "B.con", "C.test"]
    for target in targets:
        target_path = data_path / target
        target_path.touch()
        link_target_path_list.append(target_path)
    link_name_list = targets

    meteo_text = "ABC"  # Just some string
    return [
        catalog_path,
        sim_sub_path,
        link_target_path_list,
        link_name_list,
        meteo_text,
    ]


def test_simulation(data_path):
    simulation = Simulation(*create_input(data_path))
    simulation.get_status()
    simulation.run()
    time.sleep(1)
    assert simulation.get_status() == Status.finished