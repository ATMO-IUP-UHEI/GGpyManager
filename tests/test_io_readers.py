"""Unit tests for io.readers module."""

import numpy as np
import pytest
import xarray as xr
from pathlib import Path

from ggpymanager.io import readers


class TestReadGRALConfig:
    """Tests for read_gral_config function."""

    @pytest.fixture
    def sample_gral_geb_file(self, tmp_path):
        """Create a sample GRAL.geb file."""
        geb_file = tmp_path / "GRAL.geb"
        geb_content = """10.0 ! dx
10.0 ! dy
5.0,1.1 ! dz0, stretching_factor
100 ! nx
100 ! ny
5 ! n_horizontal_slices
1,2,3 ! source_groups
0.0 ! west_border
1000.0 ! east_border
0.0 ! south_border
1000.0 ! north_border
"""
        geb_file.write_text(geb_content)
        return geb_file

    @pytest.fixture
    def sample_in_dat_file(self, tmp_path):
        """Create a sample in.dat file."""
        dat_file = tmp_path / "in.dat"
        dat_content = """10000 ! particle_number
3600.0 ! dispersion_time
1 ! steady_state
0
0
0
0
0
0
0.0,5.0,10.0 ! horizontal_slices
1.0 ! vertical_grid_spacing
"""
        dat_file.write_text(dat_content)
        return dat_file

    def test_read_gral_config_returns_dict(
        self, sample_gral_geb_file, sample_in_dat_file
    ):
        """Test that function returns a dictionary."""
        result = readers.read_gral_config(sample_gral_geb_file, sample_in_dat_file)

        assert isinstance(result, dict)

    def test_read_gral_config_parses_geometry(
        self, sample_gral_geb_file, sample_in_dat_file
    ):
        """Test that geometry parameters are correctly parsed."""
        result = readers.read_gral_config(sample_gral_geb_file, sample_in_dat_file)

        assert result["dx"] == 10.0
        assert result["dy"] == 10.0
        assert result["nx"] == 100
        assert result["ny"] == 100

    def test_read_gral_config_parses_vertical_grid(
        self, sample_gral_geb_file, sample_in_dat_file
    ):
        """Test that vertical grid parameters are parsed."""
        result = readers.read_gral_config(sample_gral_geb_file, sample_in_dat_file)

        assert result["dz0"] == 5.0
        assert result["stretching_factor"] == 1.1

    def test_read_gral_config_parses_source_groups(
        self, sample_gral_geb_file, sample_in_dat_file
    ):
        """Test that source groups are parsed as list."""
        result = readers.read_gral_config(sample_gral_geb_file, sample_in_dat_file)

        assert result["source_groups"] == [1, 2, 3]


class TestReadLanduse:
    """Tests for read_landuse function."""

    @pytest.fixture
    def sample_landuse_file(self, tmp_path):
        """Create a sample landuse file."""
        landuse_file = tmp_path / "landuse.asc"
        # Create 3x3 grid for 6 variables
        content = ""
        for var in readers.LANDUSE_VARS:
            content += " ".join(["1.0"] * 9) + "\n"
        landuse_file.write_text(content)
        return landuse_file

    def test_read_landuse_returns_dict(self, sample_landuse_file):
        """Test that function returns a dictionary."""
        result = readers.read_landuse(sample_landuse_file, shape=(3, 3))

        assert isinstance(result, dict)

    def test_read_landuse_has_all_variables(self, sample_landuse_file):
        """Test that all landuse variables are present."""
        result = readers.read_landuse(sample_landuse_file, shape=(3, 3))

        for var in readers.LANDUSE_VARS:
            assert var in result

    def test_read_landuse_correct_shape(self, sample_landuse_file):
        """Test that arrays have correct shape."""
        result = readers.read_landuse(sample_landuse_file, shape=(3, 3))

        for var in readers.LANDUSE_VARS:
            assert result[var].shape == (3, 3)

    def test_read_landuse_numeric_values(self, sample_landuse_file):
        """Test that values are numeric."""
        result = readers.read_landuse(sample_landuse_file, shape=(3, 3))

        for var in readers.LANDUSE_VARS:
            assert np.issubdtype(result[var].dtype, np.number)


class TestReadTopography:
    """Tests for read_topography function."""

    @pytest.fixture
    def mock_gramm_config(self):
        """Create a mock GRAMM configuration object."""

        class MockGRAMM:
            nx = 10
            ny = 10
            nz = 5

        return MockGRAMM()

    @pytest.fixture
    def sample_topo_file(self, tmp_path, mock_gramm_config):
        """Create a sample topography file."""
        topo_file = tmp_path / "ggeom.asc"
        # Line 1: header
        content = "GRAMM geometry file\n"
        # Line 2: topography (nx * ny values)
        topo_values = " ".join(
            ["100.0"] * (mock_gramm_config.nx * mock_gramm_config.ny)
        )
        content += topo_values + "\n"
        # Add more lines as needed for complete format
        for _ in range(6):
            content += "header line\n"
        # Line 9: z-grid (nx * ny * nz values)
        zgrid_values = " ".join(
            ["10.0"]
            * (mock_gramm_config.nx * mock_gramm_config.ny * mock_gramm_config.nz)
        )
        content += zgrid_values + "\n"

        topo_file.write_text(content)
        return topo_file

    def test_read_topography_returns_tuple(self, sample_topo_file, mock_gramm_config):
        """Test that function returns a tuple of two arrays."""
        result = readers.read_topography(sample_topo_file, mock_gramm_config)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_read_topography_topo_shape(self, sample_topo_file, mock_gramm_config):
        """Test that topography has correct 2D shape."""
        topo, zgrid = readers.read_topography(sample_topo_file, mock_gramm_config)

        assert topo.shape == (mock_gramm_config.nx, mock_gramm_config.ny)

    def test_read_topography_zgrid_shape(self, sample_topo_file, mock_gramm_config):
        """Test that z-grid has correct 3D shape."""
        topo, zgrid = readers.read_topography(sample_topo_file, mock_gramm_config)

        assert zgrid.shape == (
            mock_gramm_config.nx,
            mock_gramm_config.ny,
            mock_gramm_config.nz,
        )


class TestLoadCatalogFilter:
    """Tests for load_catalog_filter function."""

    def test_load_catalog_filter_returns_dataframe(self):
        """Test that function returns a pandas DataFrame."""
        result = readers.load_catalog_filter()

        assert hasattr(result, "columns")  # pandas DataFrame-like
        assert hasattr(result, "index")

    def test_load_catalog_filter_has_data(self):
        """Test that catalog filter contains data."""
        result = readers.load_catalog_filter()

        assert len(result) > 0
        assert len(result.columns) > 0


# Additional test templates for other reader functions


class TestReadGRAMMWindfield:
    """Tests for read_gramm_windfield function (if exists)."""

    @pytest.mark.skip(reason="Requires binary file format specification")
    def test_read_gramm_windfield_placeholder(self):
        """Placeholder for GRAMM windfield reading tests."""
        pass


class TestReadGRALConcentration:
    """Tests for read_gral_concentration function (if exists)."""

    @pytest.mark.skip(reason="Requires binary file format specification")
    def test_read_gral_concentration_placeholder(self):
        """Placeholder for GRAL concentration reading tests."""
        pass
