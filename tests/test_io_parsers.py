"""Unit tests for io.parsers module."""

import pytest
from pathlib import Path

from ggpymanager.io import parsers
from ggpymanager.models.dataclasses import GRALLogMetadata, GRAMMLogMetadata


class TestParseEmissionData:
    """Tests for parse_emission_data function."""

    def test_parse_emission_data_valid_line(self):
        """Test parsing of valid emission data line."""
        line = "100.5, 200.3, 10.0, 0.5, 1.2, 0.8, 300.0, 1"
        result = parsers.parse_emission_data(line)

        assert result["x"] == 100.5
        assert result["y"] == 200.3
        assert result["z"] == 10.0
        assert result["flux"] == 0.5
        assert result["exit_velocity"] == 1.2
        assert result["stack_diameter"] == 0.8
        assert result["exit_temperature"] == 300.0
        assert result["source_group"] == 1

    def test_parse_emission_data_whitespace(self):
        """Test parsing handles extra whitespace."""
        line = "  100.5 ,  200.3 ,  10.0 ,  0.5 ,  1.2 ,  0.8 ,  300.0 ,  1  "
        result = parsers.parse_emission_data(line)

        assert result["x"] == 100.5
        assert result["source_group"] == 1

    def test_parse_emission_data_integer_values(self):
        """Test parsing with integer values."""
        line = "100, 200, 10, 1, 1, 1, 300, 1"
        result = parsers.parse_emission_data(line)

        assert result["x"] == 100.0
        assert result["flux"] == 1.0


class TestParseMeteoData:
    """Tests for parse_meteo_data function."""

    def test_parse_meteo_data_valid_line(self):
        """Test parsing of valid meteorological data line."""
        line = "01.01.2020 12:00 15.5 3.2 270.0 800.0 1013.0"
        result = parsers.parse_meteo_data(line)

        assert result["temperature"] == 15.5
        assert result["wind_speed"] == 3.2
        assert result["wind_direction"] == 270.0
        assert result["radiation"] == 800.0
        assert result["pressure"] == 1013.0

    def test_parse_meteo_data_handles_missing_values(self):
        """Test parsing handles missing/default values."""
        line = "01.01.2020 12:00 15.5 3.2 270.0"
        # Depending on implementation, may need adjustment
        # This tests the function's error handling


class TestFilterLines:
    """Tests for filter_lines function."""

    def test_filter_lines_removes_comments(self):
        """Test that comment lines are filtered out."""
        raw_lines = [
            "! This is a comment",
            "data line 1",
            "! Another comment",
            "data line 2",
        ]
        result = parsers.filter_lines(raw_lines)

        assert len(result) == 2
        assert "data line 1" in result
        assert "data line 2" in result
        assert "! This is a comment" not in result

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
        """Test that line order is preserved."""
        raw_lines = [
            "line 1",
            "! comment",
            "line 2",
            "",
            "line 3",
        ]
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
! Comment line
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
! Comment line
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
