import time

from context import data_path
from context import Simulation, Status, State
from context import Catalog, N_HEADER

# Additional inputs
meteo_text = "header1\nheader2\nmeteo1\n\n\n"
n_meteo = 1

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
    info = catalog.get_info()
    assert info["Total wind situations"] == n_meteo
    assert info["Missing GRAMM simulations"] == [i + 1 for i in range(n_meteo)]
    assert info["Missing GRAL simulations"] == [i + 1 for i in range(n_meteo)]
    catalog.init_simulations()
    # Test init
    for simulation in catalog.simulations:
        assert isinstance(simulation.status, State)
    catalog.run_simulations()

