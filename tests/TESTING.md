# Testing Guide for GGpyManager

## Setup

### Install Test Dependencies

```bash
# Option 1: Create/update conda environment with all dependencies (recommended)
conda env create -f environment-test.yml
conda activate ggpymanager-test

# Option 2: Install testing packages in existing environment
conda install pytest pytest-cov pytest-xdist pytest-mock coverage -c conda-forge

# Option 3: Install from requirements file
conda install --file requirements-test.txt -c conda-forge

# Install package in development mode
pip install -e .
```

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_catalog_template.py

# Run specific test class
pytest tests/test_catalog_template.py::TestCatalogInitialization

# Run specific test method
pytest tests/test_catalog_template.py::TestCatalogInitialization::test_initialization_with_gramm

# Run tests matching pattern
pytest -k "test_initialization"
```

### Coverage Reports

```bash
# Run tests with coverage
pytest --cov=ggpymanager

# Generate HTML coverage report
pytest --cov=ggpymanager --cov-report=html
# Then open: htmlcov/index.html

# Show missing lines in terminal
pytest --cov=ggpymanager --cov-report=term-missing

# Generate XML report (for CI)
pytest --cov=ggpymanager --cov-report=xml

# All coverage formats at once
pytest --cov=ggpymanager --cov-report=html --cov-report=term-missing --cov-report=xml
```

### Test Selection with Markers

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only slow tests
pytest -m slow

# Skip slow tests
pytest -m "not slow"

# Run unit tests but not slow ones
pytest -m "unit and not slow"
```

### Parallel Execution

```bash
# Run tests in parallel (faster)
pytest -n auto

# Run with 4 workers
pytest -n 4
```

### Debugging Tests

```bash
# Stop on first failure
pytest -x

# Show local variables on failure
pytest --showlocals

# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Increase verbosity
pytest -vv
```

## Project Structure

```
tests/
├── conftest.py                    # Shared fixtures and configuration
├── test_catalog_template.py       # Template with examples (rename/copy)
├── test_catalog.py                # Your actual catalog tests
├── test_utils.py                  # Utility tests
└── assets/                        # Test data files
```

## Writing Tests

### Using Fixtures

```python
def test_with_temp_catalog(tmp_catalog_dir, gramm_config_files):
    """Example using shared fixtures from conftest.py"""
    catalog = Catalog(tmp_catalog_dir, model="gramm")
    assert catalog.config_path.exists()
```

### Parametrized Tests

```python
@pytest.mark.parametrize("model", ["gramm", "gral"])
def test_both_models(tmp_catalog_dir, model):
    """This test runs twice: once for gramm, once for gral"""
    # ... test code
```

### Test Organization

```python
class TestFeatureName:
    """Group related tests in a class"""
    
    def test_scenario_1(self):
        # Arrange
        # Act
        # Assert
        pass
    
    def test_scenario_2(self):
        pass
```

## Coverage Goals

- **Target**: 80%+ overall coverage
- **Focus**: Critical paths and business logic
- **Exclude**: Generated code, `__init__.py`, abstract methods

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - uses: conda-incubator/setup-miniconda@v3
        with:
          auto-update-conda: true
          python-version: '3.12'
          channels: conda-forge,defaults
          environment-file: environment-test.yml
      - run: pytest --cov=ggpymanager --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## Best Practices

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One concept per test**: Keep tests focused
3. **Descriptive names**: `test_initialization_with_invalid_model`
4. **Use fixtures**: Share setup code via conftest.py
5. **Isolate tests**: Use tmp_path, don't depend on test order
6. **Mark slow tests**: Use `@pytest.mark.slow` for long-running tests
7. **Document fixtures**: Add docstrings explaining fixture purpose

## Common Issues

### Tests Not Discovered
- Ensure files start with `test_`
- Ensure functions start with `test_`
- Check `pytest.ini` testpaths

### Import Errors
- Install package in dev mode: `pip install -e .`
- Check PYTHONPATH
- Ensure you're in the correct conda environment: `conda activate ggpymanager-test`

### Fixture Not Found
- Check `conftest.py` is in tests directory
- Verify fixture name matches usage

## Useful Commands

```bash
# List all available fixtures
pytest --fixtures

# Show available markers
pytest --markers

# Dry run (collect tests without running)
pytest --collect-only

# Show test execution times
pytest --durations=10

# Generate test report
pytest --html=report.html --self-contained-html
```
