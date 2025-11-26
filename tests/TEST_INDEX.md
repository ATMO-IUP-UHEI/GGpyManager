# Test Template Index

Complete unit test templates for all GGpyManager modules.

## Overview

This directory contains comprehensive unit test templates organized by module. Each test file follows pytest conventions and includes:

- **Fixtures** for test data setup
- **Parametrized tests** for testing multiple scenarios efficiently  
- **Unit tests** for individual functions
- **Integration tests** marked with `@pytest.mark.integration`
- **Documentation** explaining what each test validates

## Test Files by Module

### Analysis Module

- **`test_analysis_loss_functions.py`** - Tests for wind field matching loss functions
  - `rmse_loss()` - Root mean square error loss
  - `regularized_loss()` - Regularized loss with temporal/spatial weighting
  - `compound_loss()` - Combined direction and speed loss
  - `compute_matching_loss()` - Main matching function with filtering

- **`test_analysis_stability.py`** - Tests for atmospheric stability class filtering
  - `get_allowed_stability_class()` - Stability class filtering based on meteorology

### Processing Module

- **`test_processing_wind.py`** - Tests for wind calculation utilities
  - `direction_from_compass()` - Convert compass directions to degrees
  - `direction_from_vector()` - Calculate wind direction from components
  - `wind_speed_from_vector()` - Calculate wind speed from components
  - `vector_from_direction_and_speed()` - Convert direction/speed to vector
  - `circular_mean()` - Circular averaging of angles

- **`test_processing_geometry.py`** - Tests for domain geometry and grids
  - `create_domain_geometry()` - Create GeoDataFrame for domain
  - `create_domain_grid()` - Create xarray grid structure
  - `gradient()` - Calculate elevation gradients
  - `smooth_elevation()` - Smooth elevation data

- **`test_processing_concentration.py`** - Tests for concentration processing (template ready)
- **`test_processing_landuse.py`** - Tests for land use conversion (template ready)

### IO Module

- **`test_io_parsers.py`** - Tests for file parsing functions
  - `parse_emission_data()` - Parse emission data lines
  - `parse_meteo_data()` - Parse meteorological data
  - `filter_lines()` - Filter comments and empty lines
  - `read_gral_stdout()` - Read GRAL log files
  - `read_gramm_stdout()` - Read GRAMM log files

- **`test_io_readers.py`** - Tests for file reading functions
  - `read_gral_config()` - Read GRAL configuration
  - `read_landuse()` - Read land use files
  - `read_topography()` - Read topography files
  - `load_catalog_filter()` - Load catalog filter data

- **`test_io_writers.py`** - Tests for file writing functions
  - `write_landuse()` - Write land use files
  - `write_buildings_file()` - Write building data
  - `write_esri_ascii()` - Write ESRI ASCII rasters
  - `write_point_dat()` - Write point emissions

### Models Module

- **`test_models_catalog.py`** - Tests for Catalog class
  - `_get_dir_size()` - Helper function for disk space
  - Catalog initialization and validation
  - Input file checking
  - Simulation scanning

- **`test_catalog_template.py`** - Comprehensive Catalog test template (reference)
- **`test_models_dataclasses.py`** - Tests for data classes (template ready)

### Utils Module

- **`test_utils_decorators.py`** - Tests for decorators
  - `check_docstring_dims()` - Dimension validation decorator

- **`test_utils_projections.py`** - Tests for projection utilities
  - `get_centered_custom_projection()` - Create custom CRS

- **`test_utils_logging.py`** - Tests for logging utilities (template ready)

## Running Tests

### Run All Tests
```bash
pytest tests/
```

### Run Specific Module
```bash
pytest tests/test_analysis_loss_functions.py -v
pytest tests/test_processing_wind.py -v
pytest tests/test_io_readers.py -v
```

### Run Specific Test Class
```bash
pytest tests/test_processing_wind.py::TestDirectionFromCompass -v
```

### Run Specific Test
```bash
pytest tests/test_processing_wind.py::TestDirectionFromCompass::test_direction_from_compass_north -v
```

### Run with Coverage
```bash
pytest --cov=ggpymanager --cov-report=html
```

### Run by Marker
```bash
# Unit tests only
pytest -m unit

# Integration tests only
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Template Pattern

Each test file follows this structure:

```python
"""Unit tests for module_name."""

import pytest
import numpy as np
import xarray as xr

from ggpymanager.module import function_name


class TestFunctionName:
    """Tests for function_name."""

    @pytest.fixture
    def sample_data(self):
        """Create sample input data."""
        # Setup test data
        return data

    def test_function_basic_case(self, sample_data):
        """Test basic functionality."""
        result = function_name(sample_data)
        
        assert result is not None
        # More assertions

    def test_function_edge_case(self):
        """Test edge case handling."""
        # Test edge cases

    def test_function_error_handling(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_name(invalid_input)

    @pytest.mark.parametrize("param,expected", [
        (value1, expected1),
        (value2, expected2),
    ])
    def test_function_parametrized(self, param, expected):
        """Test multiple scenarios."""
        result = function_name(param)
        assert result == expected
```

## Shared Fixtures

Located in `conftest.py`:

- `tmp_catalog_dir` - Temporary catalog directory structure
- `gramm_config_files` - GRAMM configuration files
- `gral_config_files` - GRAL configuration files
- `simulation_entry` - Simulation entry with parameters
- `model_type` - Parametrized model type ("gramm"/"gral")
- `sample_data_file` - Generic sample data file

## Test Coverage Goals

Target coverage by module:

- **Analysis**: 85%+
- **Processing**: 90%+
- **IO**: 85%+
- **Models**: 80%+
- **Utils**: 90%+

## Best Practices Applied

1. **AAA Pattern**: Arrange, Act, Assert structure
2. **Descriptive Names**: Test names explain what's being tested
3. **Fixtures**: Shared setup via pytest fixtures
4. **Parametrization**: Test multiple scenarios efficiently
5. **Markers**: Organize tests by type (unit/integration/slow)
6. **Isolation**: Tests don't depend on each other
7. **Temp Files**: Use `tmp_path` fixture for file operations
8. **Documentation**: Docstrings explain test purpose

## Adding New Tests

1. Create test file: `test_<module>_<submodule>.py`
2. Import module being tested
3. Create test class: `class TestFunctionName:`
4. Add fixtures for test data if needed
5. Write test methods starting with `test_`
6. Use descriptive names and docstrings
7. Add parametrize for multiple scenarios
8. Mark integration/slow tests appropriately

## CI Integration

These tests are designed to work with continuous integration:

```yaml
# .github/workflows/test.yml
- name: Run tests
  run: pytest --cov=ggpymanager --cov-report=xml --cov-report=term
```

## Coverage Report

After running with coverage:

```bash
# View in terminal
pytest --cov=ggpymanager --cov-report=term-missing

# Generate HTML report
pytest --cov=ggpymanager --cov-report=html
open htmlcov/index.html

# Generate for CI
pytest --cov=ggpymanager --cov-report=xml
```

## Troubleshooting

### Import Errors
- Ensure package is installed: `pip install -e .`
- Check you're in correct conda environment

### Missing Dependencies
- Install test requirements: `conda install pytest pytest-cov -c conda-forge`

### Fixture Not Found
- Check `conftest.py` for fixture definition
- Verify fixture name spelling

### Tests Not Discovered
- File must start with `test_`
- Functions must start with `test_`
- Classes must start with `Test`

## Further Reading

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest Fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [Parametrize](https://docs.pytest.org/en/stable/parametrize.html)
- [Coverage.py](https://coverage.readthedocs.io/)
