"""Unit tests for cli_functions module.

Tests for the CLI functions that generate matching loss files,
timeseries data, and other output files with CF compliance.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

import ggpymanager as ggp
from ggpymanager import cli_functions


class TestGenerateMatchingLossFile:
    """Tests for generate_matching_loss_file function."""

    @pytest.fixture
    def sample_gramm_meteo(self):
        """Create sample GRAMM meteo dataset."""
        n_sims = 10
        n_stations = 3
        return xr.Dataset(
            {
                "u": (
                    ["sim_id", "station"],
                    np.random.randn(n_sims, n_stations).astype("float32"),
                    {"long_name": "U wind component", "units": "m s-1"},
                ),
                "v": (
                    ["sim_id", "station"],
                    np.random.randn(n_sims, n_stations).astype("float32"),
                    {"long_name": "V wind component", "units": "m s-1"},
                ),
                "speed": (
                    ["sim_id"],
                    np.random.rand(n_sims).astype("float32") * 10,
                    {"long_name": "Wind speed", "units": "m s-1"},
                ),
                "stab_class": (
                    ["sim_id"],
                    np.random.randint(1, 7, n_sims),
                    {"long_name": "Stability class"},
                ),
            },
            coords={
                "sim_id": np.arange(1, n_sims + 1),
                "station": ["station_1", "station_2", "station_3"],
            },
            attrs={"Conventions": "CF-1.11", "history": "Created for testing"},
        )

    @pytest.fixture
    def sample_gral_meteo(self):
        """Create sample GRAL meteo dataset."""
        n_sims = 10
        n_stations = 3
        return xr.Dataset(
            {
                "u": (
                    ["sim_id", "station"],
                    np.random.randn(n_sims, n_stations).astype("float32"),
                    {"long_name": "U wind component", "units": "m s-1"},
                ),
                "v": (
                    ["sim_id", "station"],
                    np.random.randn(n_sims, n_stations).astype("float32"),
                    {"long_name": "V wind component", "units": "m s-1"},
                ),
                "speed": (
                    ["sim_id"],
                    np.random.rand(n_sims).astype("float32") * 10,
                    {"long_name": "Wind speed", "units": "m s-1"},
                ),
                "stab_class": (
                    ["sim_id"],
                    np.random.randint(1, 7, n_sims),
                    {"long_name": "Stability class"},
                ),
            },
            coords={
                "sim_id": np.arange(1, n_sims + 1),
                "station": ["station_1", "station_2", "station_3"],
            },
            attrs={"Conventions": "CF-1.11", "history": "Created for testing"},
        )

    @pytest.fixture
    def sample_meteo_measurements(self):
        """Create sample meteorological measurements."""
        n_times = 24
        n_stations = 3
        times = np.arange(
            "2020-01-01", "2020-01-02", dtype="datetime64[h]"
        )
        return xr.Dataset(
            {
                "u_wind": (
                    ["time", "station"],
                    np.random.randn(n_times, n_stations).astype("float32"),
                    {"long_name": "U wind component", "units": "m s-1"},
                ),
                "v_wind": (
                    ["time", "station"],
                    np.random.randn(n_times, n_stations).astype("float32"),
                    {"long_name": "V wind component", "units": "m s-1"},
                ),
                "global_radiation": (
                    ["time", "station"],
                    np.random.rand(n_times, n_stations).astype("float32") * 800,
                    {"long_name": "Global radiation", "units": "W m-2"},
                ),
            },
            coords={
                "time": times,
                "station": ["station_1", "station_2", "station_3"],
            },
            attrs={"Conventions": "CF-1.11", "history": "Created for testing"},
        )

    @pytest.fixture
    def matching_config(self, tmp_path):
        """Create a sample config for matching loss generation."""
        output_path = tmp_path / "output"
        output_path.mkdir()
        return {
            "output_path": str(output_path),
            "gramm_meteo_path": str(tmp_path / "gramm"),
            "gral_meteo_path": str(tmp_path / "gral"),
            "meteo_path": str(tmp_path / "meteo"),
            "matching": {
                "stations": {
                    "station_1": "gramm",
                    "station_2": "gral",
                    "station_3": "gramm",
                },
                "time_start": "2020-01-01",
                "time_end": "2020-01-01 23:00",
            },
        }

    def test_matching_loss_file_not_overwritten_if_exists(
        self, tmp_path, matching_config, caplog
    ):
        """Test that existing matching loss file is not overwritten."""
        # Create existing file
        output_path = Path(matching_config["output_path"])
        matching_file = output_path / ggp.config.MATCHING_LOSS_FILE_NAME
        matching_file.write_text("existing data")

        cli_functions.generate_matching_loss_file(matching_config)

        assert "already exists" in caplog.text
        assert matching_file.read_text() == "existing data"


class TestSaveNetcdfWithCfCheck:
    """Tests for save_netcdf_with_cf_check function."""

    @pytest.fixture
    def cf_compliant_dataset(self):
        """Create a CF-compliant dataset."""
        return xr.Dataset(
            {
                "wind_speed": (
                    ["x", "y"],
                    np.random.randn(10, 10).astype("float32"),
                    {
                        "long_name": "Wind speed",
                        "units": "m s-1",
                    },
                ),
            },
            coords={
                "x": (
                    ["x"],
                    np.arange(10),
                    {"long_name": "X coordinate", "units": "m"},
                ),
                "y": (
                    ["y"],
                    np.arange(10),
                    {"long_name": "Y coordinate", "units": "m"},
                ),
            },
            attrs={
                "Conventions": "CF-1.11",
                "title": "Test dataset",
                "history": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            },
        )

    @pytest.fixture
    def cf_compliant_dataset_with_iteration(self):
        """Create a CF-compliant dataset with iteration dimension."""
        n_iter = 5
        n_sims = 3
        return xr.Dataset(
            {
                "simulation_time": (
                    ["sim_id", "iteration"],
                    np.random.rand(n_sims, n_iter).astype("float32") * 100,
                    {
                        "long_name": "Simulation time",
                        "units": "s",
                        "description": "Total simulation time",
                    },
                ),
                "divergence": (
                    ["sim_id", "iteration"],
                    np.random.rand(n_sims, n_iter).astype("float32") * 0.01,
                    {
                        "long_name": "Simulation divergence",
                        "description": "Maximum divergence value",
                    },
                ),
            },
            coords={
                "sim_id": (
                    ["sim_id"],
                    np.arange(1, n_sims + 1),
                    {
                        "long_name": "Simulation ID",
                        "description": "Unique identifier for each simulation",
                    },
                ),
                "iteration": (
                    ["iteration"],
                    np.arange(n_iter),
                    {
                        "long_name": "Iteration step",
                        "description": "Iteration step index",
                    },
                ),
            },
            attrs={
                "Conventions": "CF-1.11",
                "title": "Simulation status log",
                "history": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            },
        )

    def test_save_cf_compliant_dataset_succeeds(self, tmp_path, cf_compliant_dataset):
        """Test that a CF-compliant dataset is saved successfully."""
        output_path = tmp_path / "output.nc"

        result = ggp.io.writers.save_netcdf_with_cf_check(
            cf_compliant_dataset, output_path
        )

        assert result is True
        assert output_path.exists()

    def test_save_cf_compliant_dataset_with_iteration_succeeds(
        self, tmp_path, cf_compliant_dataset_with_iteration
    ):
        """Test that a CF-compliant dataset with iteration dimension is saved."""
        output_path = tmp_path / "status_log.nc"

        result = ggp.io.writers.save_netcdf_with_cf_check(
            cf_compliant_dataset_with_iteration, output_path
        )

        assert result is True
        assert output_path.exists()

        # Verify the file can be read back
        ds = xr.open_dataset(output_path)
        assert "iteration" in ds.dims
        assert "sim_id" in ds.dims
        ds.close()

    def test_saved_dataset_has_correct_attributes(
        self, tmp_path, cf_compliant_dataset_with_iteration
    ):
        """Test that saved dataset preserves CF attributes."""
        output_path = tmp_path / "status_log.nc"

        ggp.io.writers.save_netcdf_with_cf_check(
            cf_compliant_dataset_with_iteration, output_path
        )

        ds = xr.open_dataset(output_path)

        # Check global attributes
        assert ds.attrs["Conventions"] == "CF-1.11"
        assert "history" in ds.attrs

        # Check variable attributes
        assert ds["simulation_time"].attrs["long_name"] == "Simulation time"
        assert ds["simulation_time"].attrs["units"] == "s"

        # Check coordinate attributes
        assert ds["sim_id"].attrs["long_name"] == "Simulation ID"
        assert ds["iteration"].attrs["long_name"] == "Iteration step"

        ds.close()

    def test_non_cf_compliant_dataset_fails(self, tmp_path):
        """Test that a non-CF-compliant dataset fails to save."""
        # Create dataset without required CF attributes
        non_compliant_ds = xr.Dataset(
            {
                "data": (["time"], np.random.randn(10)),
            },
            coords={
                "time": np.arange(10),  # Named 'time' but no standard_name
            },
        )

        output_path = tmp_path / "non_compliant.nc"

        result = ggp.io.writers.save_netcdf_with_cf_check(non_compliant_ds, output_path)

        assert result is False
        assert not output_path.exists()

    def test_iteration_dimension_set_as_unlimited(
        self, tmp_path, cf_compliant_dataset_with_iteration
    ):
        """Test that iteration dimension is set as unlimited."""
        output_path = tmp_path / "status_log.nc"

        ggp.io.writers.save_netcdf_with_cf_check(
            cf_compliant_dataset_with_iteration, output_path
        )

        # Check unlimited dimension using netCDF4
        import netCDF4

        nc = netCDF4.Dataset(output_path, "r")
        assert nc.dimensions["iteration"].isunlimited()
        nc.close()


class TestStatusLogCfCompliance:
    """Tests for status log CF compliance in Catalog."""

    @pytest.fixture
    def gramm_asset_catalog_for_cf(self, gramm_asset_catalog):
        """Provide the gramm asset catalog fixture."""
        return gramm_asset_catalog

    def test_status_log_is_cf_compliant(self, gramm_asset_catalog_for_cf):
        """Test that the status log created by Catalog is CF compliant."""
        from ggpymanager.models.catalog import Catalog

        catalog = Catalog(gramm_asset_catalog_for_cf, model="gramm")

        status_log_path = gramm_asset_catalog_for_cf / ggp.config.STATUS_LOG_FILE_NAME
        assert status_log_path.exists()

        # Load and verify CF attributes
        ds = xr.open_dataset(status_log_path)

        # Check global attributes
        assert ds.attrs.get("Conventions") == "CF-1.11"
        assert "history" in ds.attrs
        assert "title" in ds.attrs

        # Check that iteration is used instead of time
        assert "iteration" in ds.dims
        assert "time" not in ds.dims

        # Check coordinate attributes
        assert "long_name" in ds["sim_id"].attrs
        assert "long_name" in ds["iteration"].attrs

        # Check variable attributes have long_name
        for var in ds.data_vars:
            assert "long_name" in ds[var].attrs, f"Variable '{var}' missing long_name"

        ds.close()

    def test_status_log_variable_metadata(self, gramm_asset_catalog_for_cf):
        """Test that status log variables have proper metadata."""
        from ggpymanager.models.catalog import Catalog

        catalog = Catalog(gramm_asset_catalog_for_cf, model="gramm")

        status_log_path = gramm_asset_catalog_for_cf / ggp.config.STATUS_LOG_FILE_NAME
        ds = xr.open_dataset(status_log_path)

        # Check specific expected variables and their metadata
        expected_vars_with_units = {
            "simulation_time": "s",
            "simulation_timestep": "s",
            "disk_space_bytes": "bytes",
        }

        for var, expected_unit in expected_vars_with_units.items():
            if var in ds.data_vars:
                assert ds[var].attrs.get("units") == expected_unit, (
                    f"Variable '{var}' has incorrect units: "
                    f"expected '{expected_unit}', got '{ds[var].attrs.get('units')}'"
                )

        ds.close()


class TestOutputDatasetCfCompliance:
    """Tests for CF compliance of output datasets in CLI functions."""

    @pytest.fixture
    def sample_numeric_dataset(self):
        """Create a sample CF-compliant dataset with numeric coordinates only."""
        n_times = 24
        n_sims = 10
        ds = xr.Dataset(
            {
                "matching_loss": (
                    ["time_index", "sim_id"],
                    np.random.rand(n_times, n_sims).astype("float32"),
                    {
                        "long_name": "Matching loss",
                        "description": "Loss function value for wind matching",
                    },
                ),
                "n_stations_per_time": (
                    ["time_index"],
                    np.full(n_times, 3, dtype="int32"),
                    {
                        "long_name": "Number of stations with valid measurements per time step",
                    },
                ),
            },
            coords={
                "time_index": (
                    ["time_index"],
                    np.arange(n_times),
                    {
                        "long_name": "Time index",
                        "description": "Index for time dimension",
                    },
                ),
                "sim_id": (
                    ["sim_id"],
                    np.arange(1, n_sims + 1),
                    {"long_name": "Simulation ID"},
                ),
            },
            attrs={
                "Conventions": "CF-1.11",
                "title": "Matching loss data",
                "history": f"Created on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            },
        )
        return ds

    def test_numeric_dataset_is_cf_compliant(
        self, tmp_path, sample_numeric_dataset
    ):
        """Test that a properly formatted numeric dataset passes CF check."""
        output_path = tmp_path / "matching_loss.nc"

        result = ggp.io.writers.save_netcdf_with_cf_check(
            sample_numeric_dataset, output_path
        )

        assert result is True
        assert output_path.exists()

    def test_dataset_variables_have_metadata(
        self, tmp_path, sample_numeric_dataset
    ):
        """Test that dataset variables have proper metadata."""
        output_path = tmp_path / "matching_loss.nc"

        ggp.io.writers.save_netcdf_with_cf_check(
            sample_numeric_dataset, output_path
        )

        ds = xr.open_dataset(output_path)
        
        # Check that data variables have long_name
        for var in ds.data_vars:
            assert "long_name" in ds[var].attrs, f"Variable '{var}' missing long_name"

        # Check that coordinates have long_name
        for coord in ds.coords:
            assert "long_name" in ds[coord].attrs, f"Coordinate '{coord}' missing long_name"
            
        ds.close()
