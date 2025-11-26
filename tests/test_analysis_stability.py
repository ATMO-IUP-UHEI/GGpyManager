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
        #ƒ Systematic sampling: 3 radiation levels (time dimension)
        # 7 wind speeds between 0 and 7 m/s and 7 stability classes (1-7)
        # Create a sim_id dimension containing all combinations (7*7 = 49)
        radiations = np.array([0.0, 400.0, 800.0])
        time = np.arange(len(radiations))

        wind_speeds = np.linspace(0, 7, 7)
        stab_classes = np.arange(1, 8)

        # Build all combinations of (wind_speed, stab_class)
        wind_grid = np.repeat(wind_speeds, len(stab_classes))
        stab_grid = np.tile(stab_classes, len(wind_speeds))

        sim_id = list(range(1, len(wind_grid) + 1))

        radiation = xr.DataArray(
            radiations,
            dims=["time"],
            coords={"time": time},
        )
        wind_speed = xr.DataArray(
            wind_grid,
            dims=["sim_id"],
            coords={"sim_id": sim_id},
        )
        stab_class_catalog = xr.DataArray(
            stab_grid,
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


    def test_get_allowed_stability_class_no_nan(self, sample_data):
        """Test that result contains no NaN values."""
        radiation, wind_speed, stab_class_catalog = sample_data

        result = stability.get_allowed_stability_class(
            radiation, wind_speed, stab_class_catalog
        )

        assert not result.isnull().any()
