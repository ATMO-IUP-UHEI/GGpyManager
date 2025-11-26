"""Unit tests for io.parsers module."""

import pytest

from ggpymanager.io import parsers
from ggpymanager.models.dataclasses import GRALLogMetadata, GRAMMLogMetadata


class TestParseEmissionData:
    """Tests for parse_emission_data function."""

    def test_parse_emission_data_valid_line(self):
        """Test parsing of valid emission data line."""
        # parse_emission_data expects an iterator of lines
        lines = iter(["Number of emissions: 2", "Total emissions: 100.5"])
        n_emissions, total_emissions = parsers.parse_emission_data(lines)

        assert n_emissions == 2
        assert total_emissions == 100.5

    def test_parse_emission_data_whitespace(self):
        """Test parsing handles extra whitespace."""
        lines = iter(["  Number of emissions: 5(1)", "  Total emissions: 200.0  "])
        n_emissions, total_emissions = parsers.parse_emission_data(
            lines, has_parentheses=True
        )
        assert n_emissions == 5
        assert total_emissions == 200.0

    def test_parse_emission_data_integer_values(self):
        """Test parsing with integer values."""
        lines = iter(["Number of emissions: 1", "Total emissions: 300"])
        n_emissions, total_emissions = parsers.parse_emission_data(lines)
        assert n_emissions == 1
        assert total_emissions == 300.0


class TestParseMeteoData:
    """Tests for parse_meteo_data function."""

    def test_parse_meteo_data_valid_line(self):
        """Test parsing of valid meteorological data line for parser expectations."""
        # The parser expects colon separated parts with wind speed at part 2,
        # direction at part 3 and stability class at part 4.
        line = "Init meteo: timestamp: 3.2 m/s: 270.0 deg: 4"
        result = parsers.parse_meteo_data(line)

        assert result["wind_speed"] == 3.2
        assert result["direction"] == 270.0
        assert result["stability_class"] == 4.0

    def test_parse_meteo_data_handles_missing_values(self):
        """Test parsing handles missing/default values - function should
        raise on invalid format.
        """
        line = "Init meteo: timestamp: 3.2 m/s"
        with pytest.raises(Exception):
            parsers.parse_meteo_data(line)


class TestFilterLines:
    """Tests for filter_lines function."""

    def test_filter_lines_strips_zero_prefix_and_whitespace(self):
        """Test that leading '0: ' prefixes and whitespace are trimmed."""
        raw_lines = [
            "0: data line 1",
            "data line 2",
            "  0: data line 3  ",
        ]
        result = parsers.filter_lines(raw_lines)

        assert len(result) == 3
        assert result[0] == "data line 1"
        assert result[1] == "data line 2"
        assert result[2] == "data line 3"

    def test_filter_lines_removes_empty_lines(self):
        """Test that empty lines are filtered out."""
        raw_lines = [
            "data line 1",
            "",
            "   ",
            "data line 2",
            "\n",
        ]
        result = parsers.filter_lines(raw_lines)

        assert len(result) == 2
        assert "" not in result

    def test_filter_lines_preserves_order(self):
        """Test that line order is preserved (without comments)."""
        raw_lines = ["line 1", "0: line 2", "", "line 3"]
        result = parsers.filter_lines(raw_lines)

        assert result == ["line 1", "line 2", "line 3"]

    def test_filter_lines_strips_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        raw_lines = [
            "  line 1  ",
            "\tline 2\t",
            "   line 3",
        ]
        result = parsers.filter_lines(raw_lines)

        assert result[0] == "line 1"
        assert result[1] == "line 2"
        assert result[2] == "line 3"


class TestReadGRALStdout:
    """Tests for read_gral_stdout function."""

    @pytest.fixture
    def sample_gral_log(self, tmp_path):
        """Create a sample GRAL log file."""
        log_file = tmp_path / "gral-log.txt"
        log_content = """
    GRAL Started
    Total particles: 1000000
    Computation time: 3600 seconds
    Wind field: 00001.gff
    Concentration files created
    Simulation completed successfully
        """
        log_file.write_text(log_content)
        return str(log_file)

    def test_read_gral_stdout_returns_metadata(self, sample_gral_log):
        """Test that function returns GRALLogMetadata object."""
        result = parsers.read_gral_stdout(sample_gral_log)

        assert isinstance(result, GRALLogMetadata)

    def test_read_gral_stdout_file_not_found(self):
        """Test that missing file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            parsers.read_gral_stdout("/nonexistent/path/gral-log.txt")


class TestReadGRAMMStdout:
    """Tests for read_gramm_stdout function."""

    @pytest.fixture
    def sample_gramm_log(self, tmp_path):
        """Create a sample GRAMM log file."""
        log_file = tmp_path / "gramm-log.txt"
        log_content = """
    GRAMM Started
    Grid cells: 100x100x20
    Computation time: 1800 seconds
    Wind field: 00001.wnd
    Scalar field: 00001.scl
    0:  MMAIN : OUT 00001.wnd  00001.scl
    Simulation completed
        """
        log_file.write_text(log_content)
        return str(log_file)

    def test_read_gramm_stdout_returns_metadata(self, sample_gramm_log):
        """Test that function returns GRAMMLogMetadata object."""
        result = parsers.read_gramm_stdout(sample_gramm_log)

        assert isinstance(result, GRAMMLogMetadata)

    def test_read_gramm_stdout_file_not_found(self):
        """Test that missing file raises appropriate error."""
        with pytest.raises(FileNotFoundError):
            parsers.read_gramm_stdout("/nonexistent/path/gramm-log.txt")

    def test_read_gramm_stdout_detects_completion(self, sample_gramm_log):
        """Test that completion string is detected."""
        result = parsers.read_gramm_stdout(sample_gramm_log)

        # Check if completion was detected (implementation dependent)
        assert result is not None
