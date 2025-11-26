"""Template for testing the Catalog class with comprehensive examples.

This module demonstrates best practices for testing the Catalog class
using pytest fixtures, parametrization, and coverage.

To run these tests:
    pytest tests/test_catalog_template.py -v
    pytest tests/test_catalog_template.py -v --cov=ggpymanager.models.catalog
    pytest tests/test_catalog_template.py -v -k "test_initialization"
    pytest -m "unit" -v
"""

import logging
from pathlib import Path

import pytest

import ggpymanager.config as CONFIG
from ggpymanager.models.catalog import Catalog


class TestCatalogInitialization:
    """Tests for Catalog initialization and setup."""

    def test_initialization_with_gramm(self, tmp_catalog_dir, gramm_config_files):
        """Test Catalog initialization with GRAMM model.

        Parameters
        ----------
        tmp_catalog_dir : Path
            Temporary catalog directory fixture.
        gramm_config_files : Path
            GRAMM configuration files fixture.
        """
        # Act
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Assert
        assert catalog.model == "gramm"
        assert catalog.catalog_path == tmp_catalog_dir
        assert catalog.config_path == tmp_catalog_dir / CONFIG.CONFIG_PATH
        assert catalog.simulation_path == tmp_catalog_dir / CONFIG.SIMULATION_PATH

    def test_initialization_with_gral(self, tmp_catalog_dir, gral_config_files):
        """Test Catalog initialization with GRAL model."""
        # Act
        catalog = Catalog(tmp_catalog_dir, model="gral")

        # Assert
        assert catalog.model == "gral"
        assert catalog.catalog_path == tmp_catalog_dir

    def test_initialization_invalid_model(self, tmp_catalog_dir, gramm_config_files):
        """Test that initialization fails with invalid model type."""
        # Act & Assert
        with pytest.raises(ValueError, match="Model must be 'gramm' or 'gral'"):
            Catalog(tmp_catalog_dir, model="invalid")

    @pytest.mark.parametrize("model", ["gramm", "gral"])
    def test_initialization_both_models_parametrized(
        self, tmp_catalog_dir, gramm_config_files, gral_config_files, model
    ):
        """Test initialization for both models using parametrization."""
        # Act
        catalog = Catalog(tmp_catalog_dir, model=model)

        # Assert
        assert catalog.model == model


class TestCatalogInputFiles:
    """Tests for checking input files."""

    def test_all_input_files_present_gramm(self, tmp_catalog_dir, gramm_config_files):
        """Test that all required GRAMM input files are detected."""
        # Act
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Assert
        assert catalog.input_files == CONFIG.INPUT_FILES["gramm"]
        # Verify all files exist
        for file_name in catalog.input_files:
            file_path = catalog.config_path / file_name
            assert file_path.exists(), f"File {file_name} should exist"

    def test_missing_input_files_warning(
        self, tmp_catalog_dir, gramm_config_files, caplog
    ):
        """Test that missing input files trigger a warning."""
        # Arrange - Remove one file
        config_dir = tmp_catalog_dir / CONFIG.CONFIG_PATH
        (config_dir / "meteopgt.all").unlink()

        # Act
        with caplog.at_level(logging.WARNING):
            catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Assert
        assert "missing" in caplog.text.lower()
        assert "meteopgt.all" in caplog.text

    def test_no_warning_when_all_files_present(
        self, tmp_catalog_dir, gramm_config_files, caplog
    ):
        """Test that no warning is logged when all files are present."""
        # Act
        with caplog.at_level(logging.WARNING):
            catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Assert - No warnings about missing files
        assert "missing" not in caplog.text.lower()


class TestCatalogSimulations:
    """Tests for simulation scanning and management."""

    def test_empty_simulation_directory(self, tmp_catalog_dir, gramm_config_files):
        """Test catalog with no simulations."""
        # Act
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Assert
        assert len(catalog.simulation_entries) == 0
        assert len(catalog.sim_ids) == 0

    @pytest.mark.parametrize(
        "sim_params",
        [
            {"sim_id": 1, "completed": True, "model": "gramm"},
            {"sim_id": 2, "completed": False, "model": "gramm"},
        ],
        indirect=["sim_params"],
    )
    def test_simulation_detection(
        self, tmp_catalog_dir, gramm_config_files, sim_params
    ):
        """Test detection of simulation entries.

        Note: This is a template - you'll need to implement the
        simulation_entry fixture properly or create simulations manually.
        """
        # The simulation_entry fixture would create the simulation
        # For now, demonstrate the structure
        sim_id = sim_params["sim_id"]
        completed = sim_params["completed"]

        # Manually create simulation for this template
        sim_dir = (
            tmp_catalog_dir
            / CONFIG.SIMULATION_PATH
            / CONFIG.CATALOG_ENTRY_PATH_FORMATTER.format(sim_id=sim_id)
        )
        sim_dir.mkdir(parents=True)

        log_file = sim_dir / CONFIG.STD_OUT_FILE_NAME["gramm"]
        if completed:
            log_file.write_text(CONFIG.STD_OUT_STRING_FOR_COMPLETED_SIMULATION["gramm"])
            wind_file = sim_dir / CONFIG.WIND_FILE_EXTENSION["gramm"]
            wind_file.write_text("# Wind data")
        else:
            log_file.write_text("")

        # Act
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Assert
        assert sim_id in catalog.sim_ids
        # Add more specific assertions based on your Catalog implementation


class TestCatalogPaths:
    """Tests for path handling."""

    def test_catalog_path_as_string(self, tmp_catalog_dir, gramm_config_files):
        """Test that catalog accepts path as string."""
        # Act
        catalog = Catalog(str(tmp_catalog_dir), model="gramm")

        # Assert
        assert isinstance(catalog.catalog_path, Path)
        assert catalog.catalog_path == tmp_catalog_dir

    def test_catalog_path_as_path_object(self, tmp_catalog_dir, gramm_config_files):
        """Test that catalog accepts path as Path object."""
        # Act
        catalog = Catalog(Path(tmp_catalog_dir), model="gramm")

        # Assert
        assert isinstance(catalog.catalog_path, Path)


class TestCatalogIntegration:
    """Integration tests for Catalog."""

    @pytest.mark.integration
    def test_full_catalog_workflow(self, tmp_catalog_dir, gramm_config_files):
        """Test complete catalog workflow from initialization to analysis.

        This is an integration test that tests multiple components together.
        """
        # Arrange - Create multiple simulations
        sim_dir = tmp_catalog_dir / CONFIG.SIMULATION_PATH
        for sim_id in range(1, 4):
            entry_dir = sim_dir / CONFIG.CATALOG_ENTRY_PATH_FORMATTER.format(
                sim_id=sim_id
            )
            entry_dir.mkdir(parents=True)

            # Create completed simulation
            log_file = entry_dir / CONFIG.STD_OUT_FILE_NAME["gramm"]
            log_file.write_text(CONFIG.STD_OUT_STRING_FOR_COMPLETED_SIMULATION["gramm"])
            wind_file = entry_dir / CONFIG.WIND_FILE_EXTENSION["gramm"]
            wind_file.write_text("# Wind data")

        # Act
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Assert
        assert len(catalog.sim_ids) == 3
        assert all(i in catalog.sim_ids for i in [1, 2, 3])


# Example of how to use fixtures in parametrize
@pytest.fixture
def sim_params(request, tmp_catalog_dir):
    """Fixture to create simulations with parameters."""
    return request.param


# Fixtures for specific test scenarios
@pytest.fixture
def catalog_with_simulations(tmp_catalog_dir, gramm_config_files):
    """Create a catalog with pre-populated simulations for testing."""
    # Create 5 completed simulations
    sim_dir = tmp_catalog_dir / CONFIG.SIMULATION_PATH
    for sim_id in range(1, 6):
        entry_dir = sim_dir / CONFIG.CATALOG_ENTRY_PATH_FORMATTER.format(sim_id=sim_id)
        entry_dir.mkdir(parents=True)

        log_file = entry_dir / CONFIG.STD_OUT_FILE_NAME["gramm"]
        log_file.write_text(CONFIG.STD_OUT_STRING_FOR_COMPLETED_SIMULATION["gramm"])
        wind_file = entry_dir / CONFIG.WIND_FILE_EXTENSION["gramm"]
        wind_file.write_text("# Wind data")

    return Catalog(tmp_catalog_dir, model="gramm")


def test_using_prepared_catalog(catalog_with_simulations):
    """Example test using a pre-prepared catalog fixture."""
    assert len(catalog_with_simulations.sim_ids) == 5


# Performance tests
@pytest.mark.slow
def test_large_catalog_performance(tmp_catalog_dir, gramm_config_files):
    """Test catalog performance with many simulations.

    Marked as 'slow' - run with: pytest -m slow
    Skip with: pytest -m "not slow"
    """
    # Create 100 simulations
    sim_dir = tmp_catalog_dir / CONFIG.SIMULATION_PATH
    for sim_id in range(1, 101):
        entry_dir = sim_dir / CONFIG.CATALOG_ENTRY_PATH_FORMATTER.format(sim_id=sim_id)
        entry_dir.mkdir(parents=True)

    # Act & Assert - should complete in reasonable time
    catalog = Catalog(tmp_catalog_dir, model="gramm")
    assert len(catalog.simulation_entries) == 100
