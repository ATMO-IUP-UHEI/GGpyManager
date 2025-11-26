"""Unit tests for processing.geometry module."""

import numpy as np
import pytest
import xarray as xr

from ggpymanager.processing import geometry


class TestCreateDomainGeometry:
    """Tests for create_domain_geometry function."""

    @pytest.fixture
    def sample_config_gramm(self):
        """Create sample GRAMM configuration."""
        return {
            "domain": {
                "crs": "EPSG:25832",  # UTM zone 32N
                "gramm": {"bbox": {"x0": 0, "x1": 10000, "y0": 0, "y1": 10000}},
            }
        }

    @pytest.fixture
    def sample_config_custom_crs(self):
        """Create sample configuration with custom CRS."""
        return {
            "domain": {
                "crs": {"center_lat": 49.0, "center_lon": 8.5},
                "gramm": {"bbox": {"x0": -5000, "x1": 5000, "y0": -5000, "y1": 5000}},
            }
        }

    def test_create_domain_geometry_returns_geodataframe(self, sample_config_gramm):
        """Test that function returns a GeoDataFrame."""
        result = geometry.create_domain_geometry("gramm", sample_config_gramm)

        assert hasattr(result, "geometry")
        assert len(result) == 1

    def test_create_domain_geometry_bbox_coordinates(self, sample_config_gramm):
        """Test that geometry has correct bounding box."""
        result = geometry.create_domain_geometry("gramm", sample_config_gramm)
        bounds = result.geometry.bounds.iloc[0]

        assert bounds["minx"] == 0
        assert bounds["maxx"] == 10000
        assert bounds["miny"] == 0
        assert bounds["maxy"] == 10000

    def test_create_domain_geometry_crs_set(self, sample_config_gramm):
        """Test that CRS is properly set."""
        result = geometry.create_domain_geometry("gramm", sample_config_gramm)

        assert result.crs is not None
        assert result.crs == "EPSG:25832"

    def test_create_domain_geometry_custom_crs(self, sample_config_custom_crs):
        """Test domain creation with custom projection."""
        result = geometry.create_domain_geometry("gramm", sample_config_custom_crs)

        assert result.crs is not None
        assert len(result) == 1


class TestCreateDomainGrid:
    """Tests for create_domain_grid function."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for grid."""
        return {
            "domain": {
                "crs": "EPSG:25832",
                "gramm": {
                    "bbox": {"x0": 0, "x1": 1000, "y0": 0, "y1": 1000},
                    "dx": 100,
                    "dy": 100,
                },
            }
        }

    def test_create_domain_grid_returns_dataset(self, sample_config):
        """Test that function returns an xarray Dataset."""
        result = geometry.create_domain_grid("gramm", sample_config)

        assert isinstance(result, xr.Dataset)

    def test_create_domain_grid_dimensions(self, sample_config):
        """Test that grid has correct dimensions."""
        result = geometry.create_domain_grid("gramm", sample_config)

        assert "x" in result.dims
        assert "y" in result.dims
        assert "x_stag" in result.dims
        assert "y_stag" in result.dims

    def test_create_domain_grid_coordinates(self, sample_config):
        """Test that grid coordinates are correct."""
        result = geometry.create_domain_grid("gramm", sample_config)

        # Regular grid: cell centers
        assert len(result.x) == 10  # (1000 - 0) / 100
        assert len(result.y) == 10

        # Staggered grid: cell edges
        assert len(result.x_stag) == 11
        assert len(result.y_stag) == 11

    def test_create_domain_grid_coordinate_values(self, sample_config):
        """Test that coordinate values are correct."""
        result = geometry.create_domain_grid("gramm", sample_config)

        # First regular coordinate should be at cell center (dx/2)
        np.testing.assert_almost_equal(result.x.values[0], 50)
        np.testing.assert_almost_equal(result.y.values[0], 50)

        # First staggered coordinate should be at edge
        np.testing.assert_almost_equal(result.x_stag.values[0], 0)
        np.testing.assert_almost_equal(result.y_stag.values[0], 0)

    def test_create_domain_grid_invalid_bbox(self):
        """Test that invalid bbox raises assertion error."""
        invalid_config = {
            "domain": {
                "crs": "EPSG:25832",
                "gramm": {
                    "bbox": {"x0": 1000, "x1": 0, "y0": 0, "y1": 1000},  # x0 > x1
                    "dx": 100,
                    "dy": 100,
                },
            }
        }

        with pytest.raises(AssertionError, match="Invalid bbox"):
            geometry.create_domain_grid("gramm", invalid_config)

    def test_create_domain_grid_has_crs(self, sample_config):
        """Test that grid has CRS information."""
        result = geometry.create_domain_grid("gramm", sample_config)

        assert result.rio.crs is not None


class TestGradient:
    """Tests for gradient function."""

    @pytest.fixture
    def sample_elevation(self):
        """Create sample elevation data."""
        x = np.arange(10)
        y = np.arange(10)
        # Create a simple slope
        elevation_data = np.outer(np.ones(10), np.arange(10))
        return xr.DataArray(elevation_data, dims=["y", "x"], coords={"x": x, "y": y})

    def test_gradient_dimensions(self, sample_elevation):
        """Test that gradient has correct dimensions."""
        result = geometry.gradient(sample_elevation)

        # Gradient reduces each dimension by 1
        assert "x" in result.dims or "y" in result.dims

    def test_gradient_positive(self, sample_elevation):
        """Test that gradient magnitude is non-negative."""
        result = geometry.gradient(sample_elevation)

        assert (result >= 0).all()

    def test_gradient_flat_terrain(self):
        """Test gradient on flat terrain."""
        flat_elevation = xr.DataArray(
            np.ones((10, 10)),
            dims=["y", "x"],
            coords={"x": np.arange(10), "y": np.arange(10)},
        )
        result = geometry.gradient(flat_elevation)

        # Gradient of flat terrain should be zero (or very close)
        assert np.allclose(result.values, 0)


class TestSmoothElevation:
    """Tests for smooth_elevation function."""

    @pytest.fixture
    def sample_elevation(self):
        """Create sample elevation data with noise."""
        np.random.seed(42)
        x = np.arange(50)
        y = np.arange(50)
        # Base elevation + noise
        base = np.outer(np.linspace(0, 100, 50), np.ones(50))
        noise = np.random.randn(50, 50) * 5
        return xr.DataArray(base + noise, dims=["y", "x"], coords={"x": x, "y": y})

    def test_smooth_elevation_reduces_gradient(self, sample_elevation):
        """Test that smoothing reduces maximum gradient."""
        gradient_before = geometry.gradient(sample_elevation).max().values

        smoothed = geometry.smooth_elevation(sample_elevation, n_grid_cells=5)
        gradient_after = geometry.gradient(smoothed).max().values

        assert gradient_after <= gradient_before

    def test_smooth_elevation_preserves_shape(self, sample_elevation):
        """Test that smoothing preserves data shape."""
        smoothed = geometry.smooth_elevation(sample_elevation)

        assert smoothed.shape == sample_elevation.shape
        assert smoothed.dims == sample_elevation.dims

    def test_smooth_elevation_no_nan(self, sample_elevation):
        """Test that smoothed elevation contains no NaN values."""
        smoothed = geometry.smooth_elevation(sample_elevation)

        assert not smoothed.isnull().any()

    @pytest.mark.parametrize("n_cells", [3, 5, 7])
    def test_smooth_elevation_different_window_sizes(self, sample_elevation, n_cells):
        """Test smoothing with different window sizes."""
        smoothed = geometry.smooth_elevation(sample_elevation, n_grid_cells=n_cells)

        assert smoothed.shape == sample_elevation.shape
