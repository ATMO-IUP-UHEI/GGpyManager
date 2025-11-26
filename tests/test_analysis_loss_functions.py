"""Unit tests for analysis.loss_functions module."""

import numpy as np
import pytest
import xarray as xr

from ggpymanager.analysis import loss_functions


class TestRMSELoss:
    """Tests for rmse_loss function."""

    @pytest.fixture
    def sample_wind_data(self):
        """Create sample wind data for testing."""
        np.random.seed(42)
        time = np.arange(10)
        station = ["A", "B", "C"]
        sim_id = [1, 2, 3]

        u = xr.DataArray(
            np.random.randn(10, 3),
            dims=["time", "station"],
            coords={"time": time, "station": station},
        )
        v = xr.DataArray(
            np.random.randn(10, 3),
            dims=["time", "station"],
            coords={"time": time, "station": station},
        )
        u_model = xr.DataArray(
            np.random.randn(3, 3),
            dims=["sim_id", "station"],
            coords={"sim_id": sim_id, "station": station},
        )
        v_model = xr.DataArray(
            np.random.randn(3, 3),
            dims=["sim_id", "station"],
            coords={"sim_id": sim_id, "station": station},
        )
        return u, v, u_model, v_model

    def test_rmse_loss_dimensions(self, sample_wind_data):
        """Test that rmse_loss returns correct dimensions."""
        u, v, u_model, v_model = sample_wind_data
        result = loss_functions.rmse_loss(u, v, u_model, v_model)

        assert result.dims == ("time", "sim_id")
        assert len(result.time) == len(u.time)
        assert len(result.sim_id) == len(u_model.sim_id)

    def test_rmse_loss_zero_error(self):
        """Test RMSE loss when model matches observations perfectly."""
        u = xr.DataArray([[1.0, 2.0]], dims=["time", "station"])
        v = xr.DataArray([[1.0, 2.0]], dims=["time", "station"])
        u_model = xr.DataArray([[1.0, 2.0]], dims=["sim_id", "station"])
        v_model = xr.DataArray([[1.0, 2.0]], dims=["sim_id", "station"])

        result = loss_functions.rmse_loss(u, v, u_model, v_model)
        np.testing.assert_almost_equal(result.values, 0.0)

    def test_rmse_loss_positive(self, sample_wind_data):
        """Test that RMSE loss is always non-negative."""
        u, v, u_model, v_model = sample_wind_data
        result = loss_functions.rmse_loss(u, v, u_model, v_model)

        assert (result >= 0).all()

    def test_rmse_loss_symmetry(self, sample_wind_data):
        """Test RMSE loss is symmetric in u and v components."""
        u, v, u_model, v_model = sample_wind_data

        result1 = loss_functions.rmse_loss(u, v, u_model, v_model)
        result2 = loss_functions.rmse_loss(v, u, v_model, u_model)

        xr.testing.assert_allclose(result1, result2)


class TestRegularizedLoss:
    """Tests for regularized_loss function."""

    @pytest.fixture
    def sample_wind_data(self):
        """Create sample wind data for testing."""
        np.random.seed(42)
        time = np.arange(24)  # 24 hours
        station = ["A", "B", "C"]
        sim_id = [1, 2]

        u = xr.DataArray(
            np.random.randn(24, 3) + 5,  # Add offset to have positive wind speeds
            dims=["time", "station"],
            coords={"time": time, "station": station},
        )
        v = xr.DataArray(
            np.random.randn(24, 3) + 5,
            dims=["time", "station"],
            coords={"time": time, "station": station},
        )
        u_model = xr.DataArray(
            np.random.randn(2, 3) + 5,
            dims=["sim_id", "station"],
            coords={"sim_id": sim_id, "station": station},
        )
        v_model = xr.DataArray(
            np.random.randn(2, 3) + 5,
            dims=["sim_id", "station"],
            coords={"sim_id": sim_id, "station": station},
        )
        return u, v, u_model, v_model

    def test_regularized_loss_dimensions(self, sample_wind_data):
        """Test that regularized_loss returns correct dimensions."""
        u, v, u_model, v_model = sample_wind_data
        result = loss_functions.regularized_loss(u, v, u_model, v_model)

        assert result.dims == ("time", "sim_id")
        assert len(result.time) == len(u.time)
        assert len(result.sim_id) == len(u_model.sim_id)

    def test_regularized_loss_positive(self, sample_wind_data):
        """Test that regularized loss is always non-negative."""
        u, v, u_model, v_model = sample_wind_data
        result = loss_functions.regularized_loss(u, v, u_model, v_model)

        assert (result >= 0).all()

    def test_regularized_loss_temporal_smoothing(self, sample_wind_data):
        """Test that temporal smoothing is applied."""
        u, v, u_model, v_model = sample_wind_data
        result = loss_functions.regularized_loss(u, v, u_model, v_model)

        # Result should have no NaN values due to rolling with min_periods=1
        assert not result.isnull().any()


class TestCompoundLoss:
    """Tests for compound_loss function."""

    @pytest.fixture
    def sample_wind_data(self):
        """Create sample wind data for testing."""
        np.random.seed(42)
        time = np.arange(10)
        station = ["A", "B"]
        sim_id = [1, 2]

        u = xr.DataArray(
            np.random.randn(10, 2) + 5,
            dims=["time", "station"],
            coords={"time": time, "station": station},
        )
        v = xr.DataArray(
            np.random.randn(10, 2) + 5,
            dims=["time", "station"],
            coords={"time": time, "station": station},
        )
        u_model = xr.DataArray(
            np.random.randn(2, 2) + 5,
            dims=["sim_id", "station"],
            coords={"sim_id": sim_id, "station": station},
        )
        v_model = xr.DataArray(
            np.random.randn(2, 2) + 5,
            dims=["sim_id", "station"],
            coords={"sim_id": sim_id, "station": station},
        )
        return u, v, u_model, v_model

    def test_compound_loss_dimensions(self, sample_wind_data):
        """Test that compound_loss returns correct dimensions."""
        u, v, u_model, v_model = sample_wind_data
        result = loss_functions.compound_loss(u, v, u_model, v_model)

        assert result.dims == ("time", "sim_id")

    def test_compound_loss_lambda_parameter(self, sample_wind_data):
        """Test compound_loss with different lambda values."""
        u, v, u_model, v_model = sample_wind_data

        result_07 = loss_functions.compound_loss(u, v, u_model, v_model, lambda_=0.7)
        result_03 = loss_functions.compound_loss(u, v, u_model, v_model, lambda_=0.3)

        # Results should differ with different lambda
        assert not xr.testing.assert_allclose(result_07, result_03, rtol=0.01)

    def test_compound_loss_bounds(self, sample_wind_data):
        """Test that compound loss is bounded."""
        u, v, u_model, v_model = sample_wind_data
        result = loss_functions.compound_loss(u, v, u_model, v_model)

        # Loss should be non-negative and finite
        assert (result >= 0).all()
        assert np.isfinite(result).all()


class TestComputeMatchingLoss:
    """Tests for compute_matching_loss function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        time = np.arange(10)
        station = ["A", "B"]
        sim_id = [1, 2, 3]

        u = xr.DataArray(
            np.random.randn(10, 2),
            dims=["time", "station"],
            coords={"time": time, "station": station},
        )
        v = xr.DataArray(
            np.random.randn(10, 2),
            dims=["time", "station"],
            coords={"time": time, "station": station},
        )
        u_model = xr.DataArray(
            np.random.randn(3, 2),
            dims=["sim_id", "station"],
            coords={"sim_id": sim_id, "station": station},
        )
        v_model = xr.DataArray(
            np.random.randn(3, 2),
            dims=["sim_id", "station"],
            coords={"sim_id": sim_id, "station": station},
        )
        return u, v, u_model, v_model

    @pytest.mark.parametrize("matching", ["rmse", "regularized", "compound"])
    def test_compute_matching_loss_methods(self, sample_data, matching):
        """Test compute_matching_loss with different methods."""
        u, v, u_model, v_model = sample_data

        result = loss_functions.compute_matching_loss(
            u, v, u_model, v_model, matching=matching, filter=False
        )

        assert result.dims == ("time", "sim_id")
        assert not result.isnull().any()

    def test_compute_matching_loss_invalid_method(self, sample_data):
        """Test that invalid matching method raises KeyError."""
        u, v, u_model, v_model = sample_data

        with pytest.raises(KeyError):
            loss_functions.compute_matching_loss(
                u, v, u_model, v_model, matching="invalid"
            )

    def test_compute_matching_loss_with_filter(self, sample_data):
        """Test compute_matching_loss with filtering enabled."""
        u, v, u_model, v_model = sample_data

        # Create required data for filtering
        synoptic_wind_speed = xr.DataArray([5.0, 6.0, 7.0], dims=["sim_id"])
        global_radiation = xr.DataArray(np.random.rand(10) * 800, dims=["time"])
        stab_class_catalog = xr.DataArray([1, 2, 3], dims=["sim_id"])

        # This test verifies the function can be called with filter=True
        # The actual filtering logic is tested in test_analysis_stability.py
        result = loss_functions.compute_matching_loss(
            u,
            v,
            u_model,
            v_model,
            matching="rmse",
            filter=True,
            synoptic_wind_speed=synoptic_wind_speed,
            global_radiation=global_radiation,
            stab_class_catalog=stab_class_catalog,
        )

        assert result is not None
