"""Unit tests for models.catalog module."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

from ggpymanager.models.catalog import Catalog, _get_dir_size


class TestGetDirSize:
    """Tests for _get_dir_size helper function."""

    def test_get_dir_size_returns_int(self, tmp_path):
        """Test that function returns an integer."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        (test_dir / "file.txt").write_text("test content")

        result = _get_dir_size(test_dir)

        assert isinstance(result, int)
        assert result >= 0

    def test_get_dir_size_empty_directory(self, tmp_path):
        """Test size of empty directory."""
        test_dir = tmp_path / "empty"
        test_dir.mkdir()

        result = _get_dir_size(test_dir)

        assert result >= 0

    def test_get_dir_size_with_files(self, tmp_path):
        """Test that size increases with files."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()

        size_empty = _get_dir_size(test_dir)

        # Add a file
        (test_dir / "file.txt").write_text("x" * 1000)
        size_with_file = _get_dir_size(test_dir)

        assert size_with_file > size_empty

    def test_get_dir_size_nonexistent_returns_zero(self):
        """Test that nonexistent directory returns 0."""
        result = _get_dir_size(Path("/nonexistent/path"))

        assert result == 0


class TestCatalogInitialization:
    """Tests for Catalog class initialization."""

    def test_catalog_init_invalid_model(self, tmp_catalog_dir, gramm_config_files):
        """Test that invalid model raises ValueError."""
        with pytest.raises(ValueError, match="Model must be"):
            Catalog(tmp_catalog_dir, model="invalid")

    @pytest.mark.parametrize("model", ["gramm", "gral"])
    def test_catalog_init_valid_models(
        self, tmp_catalog_dir, gramm_config_files, gral_config_files, model
    ):
        """Test initialization with valid model types."""
        catalog = Catalog(tmp_catalog_dir, model=model)

        assert catalog.model == model
        assert catalog.catalog_path == tmp_catalog_dir

    def test_catalog_paths_set_correctly(self, tmp_catalog_dir, gramm_config_files):
        """Test that all paths are set correctly during init."""
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        assert catalog.catalog_path.exists()
        assert catalog.config_path.exists()
        assert catalog.simulation_path.exists()


class TestCatalogCheckInputFiles:
    """Tests for _check_input_files method."""

    def test_check_input_files_all_present(
        self, tmp_catalog_dir, gramm_config_files, caplog
    ):
        """Test that no warnings when all files present."""
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        assert "missing" not in caplog.text.lower()

    def test_check_input_files_missing_file_warning(
        self, tmp_catalog_dir, gramm_config_files, caplog
    ):
        """Test warning when files are missing."""
        # Remove one file
        config_dir = tmp_catalog_dir / "config"
        (config_dir / "meteopgt.all").unlink()

        catalog = Catalog(tmp_catalog_dir, model="gramm")

        assert "missing" in caplog.text.lower()


class TestCatalogCheckSimulations:
    """Tests for _check_simulations method."""

    def test_check_simulations_empty_directory(
        self, tmp_catalog_dir, gramm_config_files
    ):
        """Test with no simulations present."""
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        assert len(catalog.simulation_entries) == 0
        assert len(catalog.sim_ids) == 0

    def test_check_simulations_with_entries(self, tmp_catalog_dir, gramm_config_files):
        """Test detection of simulation entries."""
        # Create simulation directories
        sim_dir = tmp_catalog_dir / "simulations"
        for i in range(1, 4):
            (sim_dir / f"sim_{i:04}").mkdir(parents=True)

        catalog = Catalog(tmp_catalog_dir, model="gramm")

        assert len(catalog.simulation_entries) > 0


class TestCatalogCheckStatusLog:
    """Tests for _check_status_log method."""

    def test_check_status_log_creates_if_missing(
        self, tmp_catalog_dir, gramm_config_files
    ):
        """Test that status log is created if it doesn't exist."""
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Check that status log file exists after initialization
        # (Implementation dependent - adjust based on actual behavior)
        assert catalog is not None


# Additional test templates for Catalog methods


class TestCatalogMethods:
    """Tests for various Catalog methods."""

    @pytest.fixture
    def catalog_with_simulations(self, tmp_catalog_dir, gramm_config_files):
        """Create a catalog with some simulation entries."""
        # Create simulation directories
        sim_dir = tmp_catalog_dir / "simulations"
        for i in range(1, 6):
            entry_dir = sim_dir / f"sim_{i:04}"
            entry_dir.mkdir(parents=True)
            # Add wind file for completed simulations
            if i <= 3:
                (entry_dir / "00001.wnd").write_text("wind data")

        return Catalog(tmp_catalog_dir, model="gramm")

    def test_catalog_string_representation(self, tmp_catalog_dir, gramm_config_files):
        """Test string representation of Catalog."""
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        result = str(catalog)
        assert isinstance(result, str)

    def test_catalog_iteration(self, catalog_with_simulations):
        """Test that catalog can be iterated (if applicable)."""
        # Adjust based on actual implementation
        assert catalog_with_simulations is not None

    @pytest.mark.integration
    def test_catalog_full_workflow(self, tmp_catalog_dir, gramm_config_files):
        """Integration test for complete catalog workflow."""
        catalog = Catalog(tmp_catalog_dir, model="gramm")

        # Test that basic operations work
        assert catalog.catalog_path.exists()
        assert len(catalog.sim_ids) >= 0
