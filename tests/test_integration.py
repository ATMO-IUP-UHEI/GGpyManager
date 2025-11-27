import pytest
from ggpymanager.models.catalog import Catalog
import ggpymanager.config as CONFIG


@pytest.mark.integration
def test_gramm_catalog_integration(gramm_asset_catalog):
    """Test full integration with a GRAMM catalog using assets."""
    catalog_path = gramm_asset_catalog

    # Initialize Catalog
    catalog = Catalog(catalog_path, model="gramm")

    # Check if simulations were found
    assert len(catalog.simulation_entries) == 1
    assert catalog.n_completed_simulations == 1
    assert catalog.n_wind_files == 1

    # Check status log creation
    status_log_path = catalog_path / CONFIG.STATUS_LOG_FILE_NAME
    assert status_log_path.exists()

    # Verify status log content (basic check)
    import xarray as xr

    ds = xr.load_dataset(status_log_path)
    assert "sim_id" in ds.coords
    assert len(ds.sim_id) == 1
    assert ds.sim_id[0] == 1
    assert "disk_space_bytes" in ds

    # Check if input files were verified
    # (logs would show this, but we can check internal state if exposed)
    # The _check_input_files method is called in __init__,
    # so if no exception was raised, it passed.
    # We can check if the config path is correct
    assert catalog.config_path == catalog_path / CONFIG.CONFIG_PATH
