"""Pytest configuration and shared fixtures.

This file contains fixtures that are available to all test modules.
"""

import shutil
from pathlib import Path
from typing import Literal

import pytest

import ggpymanager.config as CONFIG


@pytest.fixture
def tmp_catalog_dir(tmp_path):
    """Create a temporary catalog directory structure.

    Parameters
    ----------
    tmp_path : Path
        Pytest's built-in temporary directory fixture.

    Yields
    ------
    Path
        Path to the temporary catalog directory.

    Examples
    --------
    >>> def test_something(tmp_catalog_dir):
    ...     assert tmp_catalog_dir.exists()
    ...     config_dir = tmp_catalog_dir / "config"
    ...     assert config_dir.exists()
    """
    catalog_dir = tmp_path / "test_catalog"
    catalog_dir.mkdir()

    # Create config directory
    config_dir = catalog_dir / CONFIG.CONFIG_PATH
    config_dir.mkdir(parents=True)

    # Create simulations directory
    sim_dir = catalog_dir / CONFIG.SIMULATION_PATH
    sim_dir.mkdir(parents=True)

    yield catalog_dir

    # Cleanup happens automatically with tmp_path


@pytest.fixture
def gramm_config_files(tmp_catalog_dir):
    """Create minimal GRAMM configuration files.

    Parameters
    ----------
    tmp_catalog_dir : Path
        Temporary catalog directory from tmp_catalog_dir fixture.

    Yields
    ------
    Path
        Path to the config directory with GRAMM files.
    """
    config_dir = tmp_catalog_dir / CONFIG.CONFIG_PATH

    # Create minimal versions of required GRAMM input files
    gramm_input_files = (
        CONFIG.INPUT_FILES["gramm"]["required"]
        + CONFIG.INPUT_FILES["gramm"]["optional"]
    )
    for file_name in gramm_input_files:
        file_path = config_dir / file_name
        # Create empty or minimal content files
        if file_name.endswith(".dat") or file_name.endswith(".txt"):
            file_path.write_text("# Minimal test file\n")
        elif file_name.endswith(".asc"):
            # Create minimal ASCII grid
            file_path.write_text(
                "ncols 10\n"
                "nrows 10\n"
                "xllcorner 0\n"
                "yllcorner 0\n"
                "cellsize 10\n"
                "NODATA_value -9999\n" + "0 " * 10 + "\n" * 10
            )
        elif file_name.endswith(".all"):
            # Minimal meteopgt.all format
            file_path.write_text("01.01.2020 00:00 0.0 0.0 0.0 0.0 0.0\n")
        elif file_name.endswith(".geb"):
            # Minimal geometry file
            file_path.write_text("10 10 10\n0 0 0\n100 100 100\n")
        else:
            file_path.write_text("")

    yield config_dir


@pytest.fixture
def gral_config_files(tmp_catalog_dir):
    """Create minimal GRAL configuration files.

    Parameters
    ----------
    tmp_catalog_dir : Path
        Temporary catalog directory from tmp_catalog_dir fixture.

    Yields
    ------
    Path
        Path to the config directory with GRAL files.
    """
    config_dir = tmp_catalog_dir / CONFIG.CONFIG_PATH

    # Create minimal versions of required GRAL input files
    for file_name in CONFIG.INPUT_FILES["gral"]:
        file_path = config_dir / file_name
        # Create empty or minimal content files
        if file_name.endswith(".dat") or file_name.endswith(".txt"):
            file_path.write_text("# Minimal test file\n")
        elif file_name.endswith(".asc"):
            # Create minimal ASCII grid
            file_path.write_text(
                "ncols 10\n"
                "nrows 10\n"
                "xllcorner 0\n"
                "yllcorner 0\n"
                "cellsize 10\n"
                "NODATA_value -9999\n" + "0 " * 10 + "\n" * 10
            )
        elif file_name.endswith(".all"):
            # Minimal meteopgt.all format
            file_path.write_text("01.01.2020 00:00 0.0 0.0 0.0 0.0 0.0\n")
        elif file_name.endswith(".geb"):
            # Minimal geometry file
            file_path.write_text("10 10 10\n0 0 0\n100 100 100\n")
        else:
            file_path.write_text("")

    yield config_dir


@pytest.fixture
def simulation_entry(tmp_catalog_dir, request):
    """Create a simulation entry in the catalog.

    Parameters
    ----------
    tmp_catalog_dir : Path
        Temporary catalog directory.
    request : FixtureRequest
        Pytest request object for parameterization.
        Use @pytest.mark.parametrize to pass sim_id and completed status.

    Yields
    ------
    Path
        Path to the simulation entry directory.

    Examples
    --------
    >>> @pytest.mark.parametrize("sim_id,completed", [(1, True), (2, False)])
    >>> def test_sim(tmp_catalog_dir, simulation_entry, sim_id, completed):
    ...     # simulation_entry will be created with specified parameters
    ...     pass
    """
    # Get parameters from test or use defaults
    sim_id = getattr(request, "param", {}).get("sim_id", 1)
    completed = getattr(request, "param", {}).get("completed", False)
    model = getattr(request, "param", {}).get("model", "gramm")

    sim_dir = (
        tmp_catalog_dir
        / CONFIG.SIMULATION_PATH
        / CONFIG.CATALOG_ENTRY_PATH_FORMATTER.format(sim_id=sim_id)
    )
    sim_dir.mkdir(parents=True)

    # Create log file
    log_file = sim_dir / CONFIG.STD_OUT_FILE_NAME[model]
    log_content = ""
    if completed:
        log_content = CONFIG.STD_OUT_STRING_FOR_COMPLETED_SIMULATION[model]
    log_file.write_text(log_content)

    # Create wind file if completed
    if completed:
        wind_file = sim_dir / CONFIG.WIND_FILE_EXTENSION[model]
        wind_file.write_text("# Wind field data\n")

    yield sim_dir


@pytest.fixture(params=["gramm", "gral"])
def model_type(request) -> Literal["gramm", "gral"]:
    """Parametrized fixture for model types.

    Automatically runs tests for both GRAMM and GRAL models.

    Yields
    ------
    str
        Model type: "gramm" or "gral"

    Examples
    --------
    >>> def test_both_models(model_type):
    ...     # This test will run twice: once for "gramm", once for "gral"
    ...     assert model_type in ["gramm", "gral"]
    """
    return request.param


@pytest.fixture
def sample_data_file(tmp_path):
    """Create a sample data file for testing I/O operations.

    Parameters
    ----------
    tmp_path : Path
        Pytest's built-in temporary directory fixture.

    Yields
    ------
    Path
        Path to the sample data file.
    """
    data_file = tmp_path / "sample_data.txt"
    data_file.write_text("Sample data content\n" * 10)
    yield data_file


@pytest.fixture
def gramm_asset_catalog(tmp_path):
    """Create a temporary catalog directory populated with GRAMM assets.

    Copies files from tests/assets/gramm_catalog to a temporary directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest's built-in temporary directory fixture.

    Yields
    ------
    Path
        Path to the temporary catalog directory.
    """
    assets_dir = Path(__file__).parent / "assets" / "gramm_catalog"
    if not assets_dir.exists():
        pytest.skip("GRAMM assets not found in tests/assets/gramm_catalog")

    catalog_dir = tmp_path / "gramm_catalog"
    shutil.copytree(assets_dir, catalog_dir)

    yield catalog_dir


# Hooks for better test output
def pytest_configure(config):
    """Configure pytest with custom settings."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'integration' marker to tests with 'integration' in name
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        # Add 'unit' marker to other tests
        else:
            item.add_marker(pytest.mark.unit)
