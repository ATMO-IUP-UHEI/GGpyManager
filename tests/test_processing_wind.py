"""Unit tests for processing.wind module."""

import numpy as np
import pytest
import xarray as xr

from ggpymanager.processing import wind


# Centralized test cases: direction_label -> (ux, vy, direction_degrees)
# Wind vector components (ux, vy) and corresponding meteorological direction
# (direction wind is coming from, in degrees clockwise from north)
WIND_TEST_CASES = {
    "N": (0, -1, 0),
    "NE": (-1, -1, 45),
    "E": (-1, 0, 90),
    "SE": (-1, 1, 135),
    "S": (0, 1, 180),
    "SW": (1, 1, 225),
    "W": (1, 0, 270),
    "NW": (1, -1, 315),
}


class TestDirectionFromCompass:
    """Tests for direction_from_compass function."""

    @pytest.mark.parametrize(
        "compass,expected",
        [(label, direction) for label, (_, _, direction) in WIND_TEST_CASES.items()]
        + [("NNE", 22.5), ("ENE", 67.5)],  # Additional inter-cardinal directions
    )
    def test_direction_from_compass_cardinal_directions(self, compass, expected):
        """Test conversion of cardinal and inter-cardinal directions."""
        result = wind.direction_from_compass(compass)
        assert result == expected

    def test_direction_from_compass_case_insensitive(self):
        """Test that function is case-insensitive."""
        # Use test cases from WIND_TEST_CASES
        n_direction = WIND_TEST_CASES["N"][2]
        ne_direction = WIND_TEST_CASES["NE"][2]

        assert wind.direction_from_compass("n") == n_direction
        assert wind.direction_from_compass("N") == n_direction
        assert wind.direction_from_compass("Ne") == ne_direction

    def test_direction_from_compass_with_whitespace(self):
        """Test that function handles whitespace."""
        # Use test cases from WIND_TEST_CASES
        n_direction = WIND_TEST_CASES["N"][2]
        ne_direction = WIND_TEST_CASES["NE"][2]

        assert wind.direction_from_compass(" N ") == n_direction
        assert wind.direction_from_compass("  NE  ") == ne_direction

    def test_direction_from_compass_invalid(self):
        """Test that invalid direction raises ValueError."""
        with pytest.raises(ValueError, match="Unknown compass direction"):
            wind.direction_from_compass("INVALID")

        with pytest.raises(ValueError):
            wind.direction_from_compass("X")


class TestDirectionFromVector:
    """Tests for direction_from_vector function."""

    @pytest.mark.parametrize(
        "label,ux,vy,expected_direction",
        [
            (label, ux, vy, direction)
            for label, (ux, vy, direction) in WIND_TEST_CASES.items()
        ],
    )
    def test_direction_from_vector_all_directions(
        self, label, ux, vy, expected_direction
    ):
        """Test direction calculation for all cardinal and inter-cardinal directions."""
        result = wind.direction_from_vector(ux, vy)
        np.testing.assert_almost_equal(
            result,
            expected_direction,
            decimal=10,
            err_msg=f"Failed for {label} direction",
        )

    def test_direction_from_vector_array(self):
        """Test with numpy arrays using WIND_TEST_CASES."""
        # Use S, W, N, E from test cases
        test_labels = ["S", "W", "N", "E"]
        ux = np.array([WIND_TEST_CASES[label][0] for label in test_labels])
        vy = np.array([WIND_TEST_CASES[label][1] for label in test_labels])
        expected = np.array([WIND_TEST_CASES[label][2] for label in test_labels])

        result = wind.direction_from_vector(ux, vy)
        np.testing.assert_array_almost_equal(result, expected)

    def test_direction_from_vector_xarray(self):
        """Test with xarray DataArrays using WIND_TEST_CASES."""
        # Use S and W from test cases
        s_ux, s_vy, s_dir = WIND_TEST_CASES["S"]
        w_ux, w_vy, w_dir = WIND_TEST_CASES["W"]

        ux = xr.DataArray([s_ux, w_ux], dims=["time"])
        vy = xr.DataArray([s_vy, w_vy], dims=["time"])
        result = wind.direction_from_vector(ux, vy)

        assert isinstance(result, xr.DataArray)
        np.testing.assert_almost_equal(result.values[0], s_dir)
        np.testing.assert_almost_equal(result.values[1], w_dir)


class TestWindSpeedFromVector:
    """Tests for wind_speed_from_vector function."""

    def test_wind_speed_from_vector_zero(self):
        """Test wind speed with zero components."""
        result = wind.wind_speed_from_vector(0, 0)
        assert result == 0

    def test_wind_speed_from_vector_unit_vectors(self):
        """Test wind speed with unit vectors."""
        result = wind.wind_speed_from_vector(1, 0)
        assert result == 1
        result = wind.wind_speed_from_vector(0, 1)
        assert result == 1

    def test_wind_speed_from_vector_pythagoras(self):
        """Test wind speed calculation (Pythagorean theorem)."""
        result = wind.wind_speed_from_vector(3, 4)
        assert result == 5

    def test_wind_speed_from_vector_array(self):
        """Test with numpy arrays."""
        ux = np.array([3, 0, 4])
        vy = np.array([4, 5, 0])
        result = wind.wind_speed_from_vector(ux, vy)

        expected = np.array([5, 5, 4])
        np.testing.assert_array_almost_equal(result, expected)

    def test_wind_speed_positive(self):
        """Test that wind speed is always non-negative."""
        ux = np.array([-3, -5, 0])
        vy = np.array([4, -12, -1])
        result = wind.wind_speed_from_vector(ux, vy)

        assert (result >= 0).all()


class TestVectorFromDirectionAndSpeed:
    """Tests for vector_from_direction_and_speed function."""

    @pytest.mark.parametrize(
        "label,expected_ux,expected_vy,direction",
        [
            (label, ux, vy, direction)
            for label, (ux, vy, direction) in WIND_TEST_CASES.items()
        ],
    )
    def test_vector_from_direction_and_speed_all_directions(
        self, label, expected_ux, expected_vy, direction
    ):
        """Test vector components for all cardinal and inter-cardinal directions."""
        expected_speed = np.sqrt(expected_ux**2 + expected_vy**2)
        ux, vy = wind.vector_from_direction_and_speed(direction, expected_speed)
        np.testing.assert_almost_equal(
            ux, expected_ux, decimal=10, err_msg=f"Failed ux for {label} direction"
        )
        np.testing.assert_almost_equal(
            vy, expected_vy, decimal=10, err_msg=f"Failed vy for {label} direction"
        )

    def test_vector_from_direction_and_speed_roundtrip(self):
        """Test roundtrip conversion direction/speed -> vector -> direction/speed."""
        direction_orig = 45
        speed_orig = 10

        ux, vy = wind.vector_from_direction_and_speed(direction_orig, speed_orig)
        direction_calc = wind.direction_from_vector(ux, vy)
        speed_calc = wind.wind_speed_from_vector(ux, vy)

        np.testing.assert_almost_equal(direction_calc, direction_orig)
        np.testing.assert_almost_equal(speed_calc, speed_orig)

    def test_vector_from_direction_and_speed_array(self):
        """Test with numpy arrays using WIND_TEST_CASES."""
        # Use N, E, S, W from test cases
        test_labels = ["N", "E", "S", "W"]
        directions = np.array([WIND_TEST_CASES[label][2] for label in test_labels])
        speeds = np.array([1, 2, 3, 4])

        ux, vy = wind.vector_from_direction_and_speed(directions, speeds)

        assert len(ux) == len(directions)
        assert len(vy) == len(directions)

        # Verify the directions are correct for unit speed
        for i, label in enumerate(test_labels):
            expected_ux, expected_vy, _ = WIND_TEST_CASES[label]
            np.testing.assert_almost_equal(ux[i], expected_ux * speeds[i], decimal=10)
            np.testing.assert_almost_equal(vy[i], expected_vy * speeds[i], decimal=10)


class TestCircularMean:
    """Tests for circular_mean function."""

    def test_circular_mean_identical_angles(self):
        """Test circular mean with identical angles."""
        angles = np.array([45, 45, 45, 45])
        result = wind.circular_mean(angles)
        np.testing.assert_almost_equal(result, 45)

    def test_circular_mean_opposite_angles(self):
        """Test circular mean with angles at 0° and 180°."""
        # Should average to 90° or 270° depending on distribution
        angles = np.array([0, 180])
        result = wind.circular_mean(angles)
        # Result should be between 0 and 360
        assert 0 <= result < 360

    def test_circular_mean_wraparound(self):
        """Test circular mean handles wraparound at 360°."""
        # Angles near 360°/0°
        angles = np.array([350, 10])
        result = wind.circular_mean(angles)
        # Mean should be around 0° (or 360°)
        assert result < 30 or result > 330

    def test_circular_mean_cardinal_directions(self):
        """Test circular mean of cardinal directions using WIND_TEST_CASES."""
        # Average of N, E, S, W
        angles = np.array(
            [
                WIND_TEST_CASES["N"][2],
                WIND_TEST_CASES["E"][2],
                WIND_TEST_CASES["S"][2],
                WIND_TEST_CASES["W"][2],
            ]
        )
        result = wind.circular_mean(angles)
        assert 0 <= result < 360

    def test_circular_mean_2d_array(self):
        """Test circular mean with 2D array (averaging along axis 0)."""
        angles = np.array([[0, 90], [180, 270]])
        result = wind.circular_mean(angles)
        assert result.shape == (2,)
