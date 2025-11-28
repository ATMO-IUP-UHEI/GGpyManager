"""Unit tests for CLI commands and logging configuration."""

from click.testing import CliRunner

from ggpymanager.cli import main


def test_status_cmd_logs(tmp_catalog_dir, gramm_config_files):
    """Ensure that the status command prints logs when run via CLI."""
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["--log-level", "INFO", "status", "--model", "gramm", str(tmp_catalog_dir)],
    )
    assert result.exit_code == 0
    # Ensure the Catalog scans the catalog and logs something
    assert "Scanning catalog" in result.output
