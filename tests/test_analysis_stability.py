"""Unit tests for analysis.stability module."""

import numpy as np
import pytest
import xarray as xr

from ggpymanager.analysis import stability


class TestGetAllowedStabilityClass:
    """Tests for get_allowed_stability_class function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample meteorological data for testing."""
        np.random.seed(42)
        time = np.arange(24)
        sim_id = [1, 2, 3, 4, 5]

        radiation = xr.DataArray(
            np.random.rand(24) * 800,  # Radiation between 0-800 W/m²
            dims=["time"],
            coords={"time": time},
        )
        wind_speed = xr.DataArray(
            np.random.rand(5) * 10 + 2,  # Wind speed between 2-12 m/s
            dims=["sim_id"],
            coords={"sim_id": sim_id},
        )
        stab_class_catalog = xr.DataArray(
            [1, 2, 3, 4, 5],  # Stability classes A-E
            dims=["sim_id"],
            coords={"sim_id": sim_id},
        )
        return radiation, wind_speed, stab_class_catalog

    def test_get_allowed_stability_class_dimensions(self, sample_data):
        """Test that function returns correct dimensions."""
        radiation, wind_speed, stab_class_catalog = sample_data

        result = stability.get_allowed_stability_class(
            radiation, wind_speed, stab_class_catalog
        )

        assert result.dims == ("time", "sim_id")
        assert len(result.time) == len(radiation)
        assert len(result.sim_id) == len(wind_speed)

    def test_get_allowed_stability_class_binary_mask(self, sample_data):
        """Test that result is a binary mask (only 0s and 1s)."""
        radiation, wind_speed, stab_class_catalog = sample_data

        result = stability.get_allowed_stability_class(
            radiation, wind_speed, stab_class_catalog
        )

        assert result.dtype == bool
        assert set(result.values.flatten()).issubset({True, False})

    def test_get_allowed_stability_class_at_least_one_allowed(self, sample_data):
        """Test that at least one stability class is allowed per timestep."""
        radiation, wind_speed, stab_class_catalog = sample_data

        result = stability.get_allowed_stability_class(
            radiation, wind_speed, stab_class_catalog
        )

        # Each timestep should have at least one allowed stability class
        assert (result.sum(dim="sim_id") > 0).all()

    def test_get_allowed_stability_class_high_radiation(self):
        """Test stability class filtering with high radiation (daytime)."""
        radiation = xr.DataArray([800.0], dims=["time"])
        wind_speed = xr.DataArray([3.0, 4.0, 5.0], dims=["sim_id"])
        stab_class_catalog = xr.DataArray([1, 2, 3], dims=["sim_id"])  # A, B, C

        result = stability.get_allowed_stability_class(
            radiation, wind_speed, stab_class_catalog
        )

        # With high radiation, unstable classes (A, B) should be more likely
        assert result.any()

    def test_get_allowed_stability_class_low_radiation(self):
        """Test stability class filtering with low radiation (nighttime)."""
        radiation = xr.DataArray([0.0], dims=["time"])
        wind_speed = xr.DataArray([3.0, 4.0, 5.0], dims=["sim_id"])
        stab_class_catalog = xr.DataArray([4, 5, 6], dims=["sim_id"])  # D, E, F

        result = stability.get_allowed_stability_class(
            radiation, wind_speed, stab_class_catalog
        )

        # With low radiation, stable classes (E, F) should be more likely
        assert result.any()

    def test_get_allowed_stability_class_no_nan(self, sample_data):
        """Test that result contains no NaN values."""
        radiation, wind_speed, stab_class_catalog = sample_data

        result = stability.get_allowed_stability_class(
            radiation, wind_speed, stab_class_catalog
        )

        assert not result.isnull().any()
