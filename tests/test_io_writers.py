"""Unit tests for io.writers module."""

import numpy as np
import pytest
import xarray as xr
from pathlib import Path

from ggpymanager.io import writers


class TestWriteLanduse:
    """Tests for write_landuse function."""

    @pytest.fixture
    def sample_landuse_data(self):
        """Create sample landuse data."""
        data_vars = {}
        for var in writers.LANDUSE_VARS:
            data_vars[var] = (["y", "x"], np.ones((5, 5)))

        return xr.Dataset(data_vars, coords={"x": np.arange(5), "y": np.arange(5)})

    def test_write_landuse_creates_file(self, tmp_path, sample_landuse_data):
        """Test that function creates output file."""
        output_file = tmp_path / "landuse.asc"
        writers.write_landuse(output_file, sample_landuse_data)

        assert output_file.exists()

    def test_write_landuse_correct_number_of_lines(self, tmp_path, sample_landuse_data):
        """Test that output has correct number of lines (one per variable)."""
        output_file = tmp_path / "landuse.asc"
        writers.write_landuse(output_file, sample_landuse_data)

        with open(output_file) as f:
            lines = f.readlines()

        assert len(lines) == len(writers.LANDUSE_VARS)

    def test_write_landuse_correct_values_per_line(self, tmp_path, sample_landuse_data):
        """Test that each line has correct number of values."""
        output_file = tmp_path / "landuse.asc"
        writers.write_landuse(output_file, sample_landuse_data)

        with open(output_file) as f:
            lines = f.readlines()

        for line in lines:
            values = line.strip().split()
            assert len(values) == 25  # 5x5 grid

    def test_write_landuse_roundtrip(self, tmp_path, sample_landuse_data):
        """Test that written file can be read back correctly."""
        from ggpymanager.io.readers import read_landuse

        output_file = tmp_path / "landuse.asc"
        writers.write_landuse(output_file, sample_landuse_data)

        # Read it back
        result = read_landuse(output_file, shape=(5, 5))

        for var in writers.LANDUSE_VARS:
            np.testing.assert_array_almost_equal(
                result[var], sample_landuse_data[var].values
            )


class TestWriteBuildingsFile:
    """Tests for write_buildings_file function."""

    @pytest.fixture
    def sample_building_height(self):
        """Create sample building height data."""
        data = np.full((10, 10), np.nan)
        # Add some buildings
        data[2:5, 3:6] = 20.0  # Building 1
        data[7:9, 7:9] = 15.0  # Building 2

        return xr.DataArray(
            data,
            dims=["y", "x"],
            coords={"x": np.arange(10) * 10.0, "y": np.arange(10) * 10.0},
        )

    def test_write_buildings_file_creates_file(self, tmp_path, sample_building_height):
        """Test that function creates output file."""
        output_file = tmp_path / "buildings.dat"
        writers.write_buildings_file(output_file, sample_building_height)

        assert output_file.exists()

    def test_write_buildings_file_only_nonzero(self, tmp_path, sample_building_height):
        """Test that only non-NaN buildings are written."""
        output_file = tmp_path / "buildings.dat"
        writers.write_buildings_file(output_file, sample_building_height)

        with open(output_file) as f:
            lines = f.readlines()

        # Should only write entries where height is not NaN
        non_nan_count = (~np.isnan(sample_building_height.values)).sum()
        assert len(lines) == non_nan_count

    def test_write_buildings_file_format(self, tmp_path, sample_building_height):
        """Test that output format is correct (x,y,z,height)."""
        output_file = tmp_path / "buildings.dat"
        writers.write_buildings_file(output_file, sample_building_height)

        with open(output_file) as f:
            first_line = f.readline().strip()

        # Format should be: x,y,0,height
        parts = first_line.split(",")
        assert len(parts) == 4
        assert parts[2] == "0"  # z-coordinate should be 0


class TestWriteESRIAscii:
    """Tests for write_esri_ascii function."""

    @pytest.fixture
    def sample_raster_data(self):
        """Create sample raster data."""
        x = np.arange(0, 100, 10)
        y = np.arange(0, 100, 10)
        data = np.random.rand(10, 10) * 100

        return xr.DataArray(data, dims=["y", "x"], coords={"x": x, "y": y})

    def test_write_esri_ascii_creates_file(self, tmp_path, sample_raster_data):
        """Test that function creates output file."""
        output_file = tmp_path / "raster.asc"
        writers.write_esri_ascii(output_file, sample_raster_data)

        assert output_file.exists()

    def test_write_esri_ascii_has_header(self, tmp_path, sample_raster_data):
        """Test that output file has ESRI ASCII header."""
        output_file = tmp_path / "raster.asc"
        writers.write_esri_ascii(output_file, sample_raster_data)

        with open(output_file) as f:
            lines = f.readlines()

        # Check header keywords
        assert "NCOLS" in lines[0]
        assert "NROWS" in lines[1]
        assert "XLLCORNER" in lines[2]
        assert "YLLCORNER" in lines[3]
        assert "CELLSIZE" in lines[4]
        assert "NODATA_VALUE" in lines[5]

    def test_write_esri_ascii_correct_ncols_nrows(self, tmp_path, sample_raster_data):
        """Test that header has correct dimensions."""
        output_file = tmp_path / "raster.asc"
        writers.write_esri_ascii(output_file, sample_raster_data)

        with open(output_file) as f:
            lines = f.readlines()

        ncols = int(lines[0].split()[1])
        nrows = int(lines[1].split()[1])

        assert ncols == sample_raster_data.sizes["x"]
        assert nrows == sample_raster_data.sizes["y"]

    def test_write_esri_ascii_file_exists_error(self, tmp_path, sample_raster_data):
        """Test that function raises error if file exists."""
        output_file = tmp_path / "raster.asc"
        output_file.touch()  # Create empty file

        with pytest.raises(FileExistsError):
            writers.write_esri_ascii(output_file, sample_raster_data)


class TestWritePointDat:
    """Tests for write_point_dat function."""

    @pytest.fixture
    def sample_emission_data(self):
        """Create sample point emission data."""
        n_sources = 5
        return {
            "x": np.random.rand(n_sources) * 1000,
            "y": np.random.rand(n_sources) * 1000,
            "z": np.ones(n_sources) * 10,
            "flux": np.ones(n_sources) * 0.5,
            "exit_velocity": np.ones(n_sources) * 1.5,
            "stack_diameter": np.ones(n_sources) * 0.5,
            "exit_temperature": np.ones(n_sources) * 350,
            "source_group": np.array([1, 1, 2, 2, 3]),
        }

    def test_write_point_dat_creates_file(self, tmp_path, sample_emission_data):
        """Test that function creates output file."""
        output_file = tmp_path / "point.dat"
        writers.write_point_dat(output_file, **sample_emission_data)

        assert output_file.exists()

    def test_write_point_dat_correct_number_of_lines(
        self, tmp_path, sample_emission_data
    ):
        """Test that output has correct number of lines."""
        output_file = tmp_path / "point.dat"
        writers.write_point_dat(output_file, **sample_emission_data)

        with open(output_file) as f:
            lines = f.readlines()

        # Should have header + one line per source
        assert len(lines) >= len(sample_emission_data["x"])

    def test_write_point_dat_array_length_mismatch(self, tmp_path):
        """Test that function handles mismatched array lengths."""
        with pytest.raises((ValueError, AssertionError)):
            writers.write_point_dat(
                tmp_path / "point.dat",
                x=np.array([1, 2]),
                y=np.array([1, 2, 3]),  # Wrong length
                z=np.array([1, 2]),
                flux=np.array([1, 2]),
                exit_velocity=np.array([1, 2]),
                stack_diameter=np.array([1, 2]),
                exit_temperature=np.array([1, 2]),
                source_group=np.array([1, 2]),
            )
