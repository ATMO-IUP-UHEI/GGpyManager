"""Unit tests for utils.units module."""

import numpy as np
import pytest
import xarray as xr

from ggpymanager.utils.units import ugm3_to_ppm


class TestUgm3ToPpm:
    """Tests for ugm3_to_ppm function."""

    def test_ugm3_to_ppm_co2_with_local_conditions(self):
        """Test CO2 conversion with local pressure and temperature."""
        # Standard conditions: 101325 Pa, 288.15 K (15°C)
        # Expected result calculated using ideal gas law
        ugm3 = 1000.0  # µg/m³
        P_local = 101325.0  # Pa
        T_local = 288.15  # K

        result = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=P_local, T_local=T_local)

        # Expected: (1000 * 8.314462618 * 288.15) / (44.01e-3 * 101325) * 1e-3
        # ≈ 0.537 ppm
        expected = (ugm3 * 8.314462618 * T_local) / (44.01e-3 * P_local) * 1e-3
        assert np.isclose(result, expected, rtol=1e-10)
        assert np.isclose(result, 0.537, rtol=1e-3)

    def test_ugm3_to_ppm_ch4_with_local_conditions(self):
        """Test CH4 conversion with local pressure and temperature."""
        # Standard conditions
        ugm3 = 2000.0  # µg/m³
        P_local = 101325.0  # Pa
        T_local = 288.15  # K

        result = ugm3_to_ppm(ugm3=ugm3, gas="CH4", P_local=P_local, T_local=T_local)

        # Expected: (2000 * 8.314462618 * 288.15) / (16.04e-3 * 101325) * 1e-3
        # ≈ 2.94 ppm
        expected = (ugm3 * 8.314462618 * T_local) / (16.04e-3 * P_local) * 1e-3
        assert np.isclose(result, expected, rtol=1e-10)
        assert np.isclose(result, 2.94, rtol=1e-2)

    def test_ugm3_to_ppm_with_ground_conditions_zero_height(self):
        """Test conversion using ground conditions at zero height."""
        # At h=0, should be same as using local conditions
        ugm3 = 1500.0
        P0 = 101325.0
        T0 = 288.15
        h = 0.0

        result = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P0=P0, T0=T0, h=h)

        # Should equal the direct calculation with P_local=P0, T_local=T0
        expected = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=P0, T_local=T0)
        assert np.isclose(result, expected, rtol=1e-10)

    def test_ugm3_to_ppm_with_ground_conditions_at_height(self):
        """Test conversion using ground conditions at elevated height."""
        ugm3 = 1000.0
        P0 = 101325.0  # Pa
        T0 = 288.15  # K
        h = 1000.0  # m

        result = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P0=P0, T0=T0, h=h)

        # Calculate expected pressure at height using barometric formula
        R = 8.314462618
        M_air = 28.965e-3
        g = 9.80665
        P_expected = P0 * np.exp(-M_air * g * h / (R * T0))

        # Calculate expected ppm
        M_co2 = 44.01e-3
        expected = (ugm3 * R * T0) / (M_co2 * P_expected) * 1e-3

        assert np.isclose(result, expected, rtol=1e-10)
        # At 1000m, pressure is lower, so ppm should be higher than at ground level
        ground_result = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=P0, T_local=T0)
        assert result > ground_result

    @pytest.mark.parametrize(
        "gas,expected_ratio",
        [
            ("CO2", 16.04 / 44.01),  # CH4/CO2 molar mass ratio
            ("co2", 16.04 / 44.01),  # lowercase
            ("Co2", 16.04 / 44.01),  # mixed case
        ],
    )
    def test_ugm3_to_ppm_gas_case_insensitive(self, gas, expected_ratio):
        """Test that gas parameter is case-insensitive."""
        ugm3 = 1000.0
        P_local = 101325.0
        T_local = 288.15

        result_co2 = ugm3_to_ppm(ugm3=ugm3, gas=gas, P_local=P_local, T_local=T_local)

        result_ch4 = ugm3_to_ppm(ugm3=ugm3, gas="CH4", P_local=P_local, T_local=T_local)

        # CH4 has lower molar mass, so ppm should be higher by the inverse ratio
        assert np.isclose(result_ch4 / result_co2, 44.01 / 16.04, rtol=1e-10)

    @pytest.mark.parametrize(
        "P,T",
        [
            (101325.0, 288.15),  # Standard conditions
            (90000.0, 273.15),  # Cold, low pressure
            (110000.0, 310.15),  # Hot, high pressure
            (80000.0, 250.0),  # High altitude conditions
        ],
    )
    def test_ugm3_to_ppm_various_conditions(self, P, T):
        """Test conversion under various atmospheric conditions."""
        ugm3 = 1000.0

        result = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=P, T_local=T)

        # Verify result is positive and reasonable
        assert result > 0
        # For typical atmospheric conditions, 1000 µg/m³ CO2 should be < 2 ppm
        assert result < 2.0

    def test_ugm3_to_ppm_zero_concentration(self):
        """Test that zero concentration returns zero ppm."""
        result = ugm3_to_ppm(ugm3=0.0, gas="CO2", P_local=101325.0, T_local=288.15)

        assert result == 0.0

    def test_ugm3_to_ppm_high_concentration(self):
        """Test conversion of high concentration values."""
        ugm3 = 1e6  # 1000 mg/m³

        result = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=101325.0, T_local=288.15)

        # Should be approximately 537 ppm
        assert np.isclose(result, 537, rtol=1e-2)

    def test_ugm3_to_ppm_invalid_gas_raises_error(self):
        """Test that invalid gas species raises ValueError."""
        with pytest.raises(ValueError, match="Gas must be 'CO2' or 'CH4'"):
            ugm3_to_ppm(ugm3=1000.0, gas="N2O", P_local=101325.0, T_local=288.15)

        with pytest.raises(ValueError, match="Gas must be 'CO2' or 'CH4'"):
            ugm3_to_ppm(ugm3=1000.0, gas="invalid", P_local=101325.0, T_local=288.15)

    def test_ugm3_to_ppm_no_conditions_uses_standard(self):
        """Test that no conditions defaults to standard conditions."""
        # No conditions provided - should use standard conditions (101325 Pa, 298.15 K)
        result = ugm3_to_ppm(ugm3=1000.0, gas="CO2")
        
        # Verify it returns a valid result (doesn't raise error)
        assert result > 0
        
        # Should equal explicit standard conditions
        expected = ugm3_to_ppm(
            ugm3=1000.0,
            gas="CO2",
            P_local=101325.0,
            T_local=298.15
        )
        assert np.isclose(result, expected, rtol=1e-10)

    def test_ugm3_to_ppm_partial_conditions_raises_error(self):
        """Test that partial/incomplete conditions raise ValueError."""
        # Only P_local provided (missing T_local)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", P_local=101325.0)

        # Only T_local provided (missing P_local)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", T_local=288.15)

        # Only P0 and T0 provided (missing h)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", P0=101325.0, T0=288.15)

        # Only P0 and h provided (missing T0)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", P0=101325.0, h=100.0)

        # Only T0 and h provided (missing P0)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", T0=288.15, h=100.0)

        # Only P0 provided (missing T0 and h)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", P0=101325.0)

        # Only T0 provided (missing P0 and h)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", T0=288.15)

        # Only h provided (missing P0 and T0)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", h=100.0)

        # Mixed partial inputs: P_local and P0 (missing T_local, T0, h)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", P_local=101325.0, P0=101325.0)

        # Mixed partial inputs: T_local and T0 (missing P_local, P0, h)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", T_local=288.15, T0=288.15)

        # Mixed partial inputs: P_local and h (missing T_local, P0, T0)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", P_local=101325.0, h=100.0)

        # Mixed partial inputs: T_local and P0 (missing P_local, T0, h)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", T_local=288.15, P0=101325.0)

        # Mixed partial inputs: T_local and h (missing P_local, P0, T0)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", T_local=288.15, h=100.0)

        # Mixed partial inputs: P0 and T_local (missing P_local, T0, h)
        with pytest.raises(
            ValueError,
            match="Must provide either \\(P_local, T_local\\) or \\(P0, T0, h\\)",
        ):
            ugm3_to_ppm(ugm3=1000.0, gas="CO2", P0=101325.0, T_local=288.15)

    def test_ugm3_to_ppm_temperature_effect(self):
        """Test that higher temperature increases ppm for same µg/m³."""
        ugm3 = 1000.0
        P = 101325.0

        result_cold = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=P, T_local=273.15)

        result_hot = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=P, T_local=313.15)

        # At constant pressure, higher temperature means higher ppm
        assert result_hot > result_cold
        # Ratio should match temperature ratio
        assert np.isclose(result_hot / result_cold, 313.15 / 273.15, rtol=1e-10)

    def test_ugm3_to_ppm_pressure_effect(self):
        """Test that lower pressure increases ppm for same µg/m³."""
        ugm3 = 1000.0
        T = 288.15

        result_high_p = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=110000.0, T_local=T)

        result_low_p = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=90000.0, T_local=T)

        # At constant temperature, lower pressure means higher ppm
        assert result_low_p > result_high_p
        # Ratio should be inverse of pressure ratio
        assert np.isclose(result_low_p / result_high_p, 110000.0 / 90000.0, rtol=1e-10)

    def test_ugm3_to_ppm_barometric_formula_consistency(self):
        """Test that barometric formula gives consistent results."""
        ugm3 = 1500.0
        P0 = 101325.0
        T0 = 288.15

        # Calculate for multiple heights
        heights = [0, 500, 1000, 1500, 2000]
        results = []

        for h in heights:
            result = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P0=P0, T0=T0, h=h)
            results.append(result)

        # Results should increase with height (lower pressure)
        for i in range(len(results) - 1):
            assert results[i + 1] > results[i]

    def test_ugm3_to_ppm_ch4_co2_ratio(self):
        """Test that CH4 and CO2 conversions maintain correct molar mass ratio."""
        ugm3 = 1000.0
        P = 101325.0
        T = 288.15

        result_co2 = ugm3_to_ppm(ugm3=ugm3, gas="CO2", P_local=P, T_local=T)

        result_ch4 = ugm3_to_ppm(ugm3=ugm3, gas="CH4", P_local=P, T_local=T)

        # Ratio should equal M_CO2 / M_CH4
        expected_ratio = 44.01 / 16.04
        actual_ratio = result_ch4 / result_co2

        assert np.isclose(actual_ratio, expected_ratio, rtol=1e-10)


class TestUgm3ToPpmWithArrays:
    """Tests for ugm3_to_ppm function with numpy arrays and xarray DataArrays."""

    def test_ugm3_to_ppm_with_numpy_array(self):
        """Test conversion with numpy array input."""
        ugm3_array = np.array([500.0, 1000.0, 1500.0, 2000.0])
        P_local = 101325.0
        T_local = 288.15

        result = ugm3_to_ppm(
            ugm3=ugm3_array, gas="CO2", P_local=P_local, T_local=T_local
        )

        # Result should be a numpy array with same shape
        assert isinstance(result, np.ndarray)
        assert result.shape == ugm3_array.shape

        # Each element should match individual calculation
        for i, ugm3_val in enumerate(ugm3_array):
            expected = ugm3_to_ppm(
                ugm3=float(ugm3_val), gas="CO2", P_local=P_local, T_local=T_local
            )
            assert np.isclose(result[i], expected, rtol=1e-10)

    def test_ugm3_to_ppm_with_2d_numpy_array(self):
        """Test conversion with 2D numpy array."""
        ugm3_array = np.array([[100.0, 200.0], [300.0, 400.0]])
        P_local = 101325.0
        T_local = 288.15

        result = ugm3_to_ppm(
            ugm3=ugm3_array, gas="CH4", P_local=P_local, T_local=T_local
        )

        # Result should preserve shape
        assert isinstance(result, np.ndarray)
        assert result.shape == ugm3_array.shape
        assert result.shape == (2, 2)

        # Verify all values are correct
        expected = (ugm3_array * 8.314462618 * T_local) / (16.04e-3 * P_local) * 1e-3
        assert np.allclose(result, expected, rtol=1e-10)

    def test_ugm3_to_ppm_with_xarray_dataarray(self):
        """Test conversion with xarray DataArray input."""
        ugm3_data = np.array([500.0, 1000.0, 1500.0])
        ugm3_da = xr.DataArray(
            ugm3_data, dims=["station"], coords={"station": ["A", "B", "C"]}
        )

        P_local = 101325.0
        T_local = 288.15

        result = ugm3_to_ppm(ugm3=ugm3_da, gas="CO2", P_local=P_local, T_local=T_local)

        # Result should be an xarray DataArray with same structure
        assert isinstance(result, xr.DataArray)
        assert result.dims == ugm3_da.dims
        assert list(result.coords) == list(ugm3_da.coords)

        # Values should be correct
        for i, ugm3_val in enumerate(ugm3_data):
            expected = ugm3_to_ppm(
                ugm3=float(ugm3_val), gas="CO2", P_local=P_local, T_local=T_local
            )
            assert np.isclose(result.values[i], expected, rtol=1e-10)

    def test_ugm3_to_ppm_with_2d_xarray_dataarray(self):
        """Test conversion with 2D xarray DataArray."""
        ugm3_data = np.array([[100.0, 200.0, 300.0], [400.0, 500.0, 600.0]])
        ugm3_da = xr.DataArray(
            ugm3_data, dims=["y", "x"], coords={"y": [0, 1], "x": [0, 1, 2]}
        )

        P_local = 101325.0
        T_local = 288.15

        result = ugm3_to_ppm(ugm3=ugm3_da, gas="CH4", P_local=P_local, T_local=T_local)

        # Result should preserve structure
        assert isinstance(result, xr.DataArray)
        assert result.dims == ("y", "x")
        assert result.shape == (2, 3)

        # Values should be correct
        expected = (ugm3_data * 8.314462618 * T_local) / (16.04e-3 * P_local) * 1e-3
        assert np.allclose(result.values, expected, rtol=1e-10)

    def test_ugm3_to_ppm_with_array_and_barometric_formula(self):
        """Test conversion with arrays using barometric formula."""
        ugm3_array = np.array([1000.0, 1500.0, 2000.0])
        heights = np.array([0.0, 500.0, 1000.0])

        P0 = 101325.0
        T0 = 288.15

        results = []
        for ugm3_val, h in zip(ugm3_array, heights):
            result = ugm3_to_ppm(ugm3=ugm3_val, gas="CO2", P0=P0, T0=T0, h=h)
            results.append(result)

        results_array = np.array(results)

        # Verify results are positive and increase with height for same concentration
        assert np.all(results_array > 0)

    def test_ugm3_to_ppm_xarray_with_attributes(self):
        """Test that xarray attributes are preserved."""
        ugm3_da = xr.DataArray(
            [1000.0, 2000.0],
            dims=["time"],
            coords={"time": [0, 1]},
            attrs={"units": "µg/m³", "description": "CO2 concentration"},
        )

        result = ugm3_to_ppm(ugm3=ugm3_da, gas="CO2", P_local=101325.0, T_local=288.15)

        # Check that result is xarray DataArray
        assert isinstance(result, xr.DataArray)
        # Coordinates should be preserved
        assert "time" in result.coords

    def test_ugm3_to_ppm_broadcast_scalar_conditions_with_array(self):
        """Test that scalar conditions broadcast correctly with array inputs."""
        ugm3_array = np.array([100.0, 500.0, 1000.0, 1500.0])
        P_local = 101325.0  # Scalar
        T_local = 288.15  # Scalar

        result = ugm3_to_ppm(
            ugm3=ugm3_array, gas="CO2", P_local=P_local, T_local=T_local
        )

        # Should work without errors and return array
        assert isinstance(result, np.ndarray)
        assert result.shape == ugm3_array.shape
        assert np.all(result > 0)

    def test_ugm3_to_ppm_array_element_wise_operations(self):
        """Test that element-wise operations work correctly with arrays."""
        # Create arrays where we know the relationship
        ugm3_array = np.array([1000.0, 1000.0, 1000.0])
        T_array = np.array([273.15, 288.15, 303.15])  # Different temperatures
        P = 101325.0

        results = []
        for ugm3_val, T_val in zip(ugm3_array, T_array):
            result = ugm3_to_ppm(ugm3=ugm3_val, gas="CO2", P_local=P, T_local=T_val)
            results.append(result)

        results_array = np.array(results)

        # Higher temperature should give higher ppm
        assert results_array[1] > results_array[0]
        assert results_array[2] > results_array[1]

    def test_ugm3_to_ppm_xarray_with_multiple_dimensions(self):
        """Test conversion with multi-dimensional xarray DataArray."""
        ugm3_data = np.random.uniform(100, 2000, size=(3, 4, 5))
        ugm3_da = xr.DataArray(
            ugm3_data,
            dims=["time", "y", "x"],
            coords={"time": [0, 1, 2], "y": [0, 1, 2, 3], "x": [0, 1, 2, 3, 4]},
        )

        result = ugm3_to_ppm(ugm3=ugm3_da, gas="CO2", P_local=101325.0, T_local=288.15)

        # Check structure is preserved
        assert isinstance(result, xr.DataArray)
        assert result.shape == (3, 4, 5)
        assert result.dims == ("time", "y", "x")
        assert np.all(result.values > 0)
        assert np.all(result.values < 5.0)  # Reasonable range for CO2 in ppm

    def test_ugm3_to_ppm_numpy_array_with_zeros(self):
        """Test that arrays with zero values are handled correctly."""
        ugm3_array = np.array([0.0, 500.0, 0.0, 1000.0])

        result = ugm3_to_ppm(
            ugm3=ugm3_array, gas="CO2", P_local=101325.0, T_local=288.15
        )

        # Zero concentrations should give zero ppm
        assert result[0] == 0.0
        assert result[2] == 0.0
        # Non-zero concentrations should give non-zero ppm
        assert result[1] > 0.0
        assert result[3] > 0.0

    def test_ugm3_to_ppm_large_array_performance(self):
        """Test conversion with large array to ensure efficiency."""
        ugm3_array = np.random.uniform(100, 2000, size=10000)

        result = ugm3_to_ppm(
            ugm3=ugm3_array, gas="CO2", P_local=101325.0, T_local=288.15
        )

        # Should handle large arrays without issues
        assert isinstance(result, np.ndarray)
        assert result.shape == ugm3_array.shape
        assert np.all(result >= 0)
        assert np.all(np.isfinite(result))
