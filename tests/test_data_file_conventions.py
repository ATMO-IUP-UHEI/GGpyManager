# """Unit tests for data/file_conventions.yml structure validation."""

# import sys
# import pytest
# import yaml

# if sys.version_info >= (3, 9):
#     from importlib.resources import files
# else:
#     from importlib_resources import files


# class TestFileConventionsStructure:
#     """Tests to enforce the structure of file_conventions.yml."""

#     @pytest.fixture
#     def file_conventions_data(self):
#         """Load file_conventions.yml data using importlib.resources."""
#         resource = files("ggpymanager.data").joinpath("file_conventions.yml")
#         with resource.open("r") as f:
#             data = yaml.safe_load(f)
#         return data

#     def test_file_conventions_exists(self):
#         """Test that file_conventions.yml exists."""
#         resource = files("ggpymanager.data").joinpath("file_conventions.yml")
#         # Check if the resource exists by trying to open it
#         with resource.open("r") as f:
#             assert f is not None

#     def test_file_conventions_is_valid_yaml(self, file_conventions_data):
#         """Test that file_conventions.yml is valid YAML."""
#         assert file_conventions_data is not None
#         assert isinstance(file_conventions_data, dict)

#     def test_file_conventions_has_required_top_level_keys(self, file_conventions_data):
#         """Test that file_conventions.yml has required top-level keys."""
#         required_keys = ["file_names", "scripts"]

#         for key in required_keys:
#             assert (
#                 key in file_conventions_data
#             ), f"Missing required top-level key: {key}"

#     def test_file_names_section_structure(self, file_conventions_data):
#         """Test that file_names section has proper structure."""
#         file_names = file_conventions_data.get("file_names")

#         assert isinstance(file_names, list), "file_names should be a list"
#         assert len(file_names) > 0, "file_names should not be empty"

#     def test_each_file_name_entry_has_required_fields(self, file_conventions_data):
#         """Test that each file_names entry has required fields."""
#         file_names = file_conventions_data.get("file_names", [])

#         for entry in file_names:
#             assert isinstance(
#                 entry, dict
#             ), "Each file_names entry should be a dictionary"
#             assert (
#                 len(entry) == 1
#             ), "Each file_names entry should have exactly one file name key"

#             # Get the file name and its configuration
#             file_name = list(entry.keys())[0]
#             file_config = entry[file_name]

#             assert isinstance(
#                 file_config, dict
#             ), f"{file_name}: configuration should be a dictionary"
#             assert (
#                 "description" in file_config
#             ), f"{file_name}: missing 'description' field"
#             assert "data_vars" in file_config, f"{file_name}: missing 'data_vars' field"

#     def test_file_name_descriptions_are_strings(self, file_conventions_data):
#         """Test that all file name descriptions are strings."""
#         file_names = file_conventions_data.get("file_names", [])

#         for entry in file_names:
#             file_name = list(entry.keys())[0]
#             file_config = entry[file_name]
#             description = file_config.get("description")

#             assert isinstance(
#                 description, str
#             ), f"{file_name}: description should be a string"
#             assert len(description) > 0, f"{file_name}: description should not be empty"

#     def test_data_vars_structure(self, file_conventions_data):
#         """Test that data_vars have proper structure."""
#         file_names = file_conventions_data.get("file_names", [])

#         for entry in file_names:
#             file_name = list(entry.keys())[0]
#             file_config = entry[file_name]
#             data_vars = file_config.get("data_vars")

#             assert isinstance(
#                 data_vars, list
#             ), f"{file_name}: data_vars should be a list"

#     def test_data_vars_have_valid_structure(self, file_conventions_data):
#         """Test that data_vars entries have valid structure when specified."""
#         file_names = file_conventions_data.get("file_names", [])

#         for entry in file_names:
#             file_name = list(entry.keys())[0]
#             file_config = entry[file_name]
#             data_vars = file_config.get("data_vars", [])

#             for var_entry in data_vars:
#                 if var_entry is None or var_entry == "...":
#                     # Allow placeholder entries
#                     continue

#                 assert isinstance(
#                     var_entry, dict
#                 ), f"{file_name}: each data_var should be a dict"
#                 assert (
#                     len(var_entry) == 1
#                 ), f"{file_name}: each data_var should have one variable name"

#                 var_name = list(var_entry.keys())[0]
#                 var_config = var_entry[var_name]

#                 if var_config is not None:
#                     assert isinstance(
#                         var_config, dict
#                     ), f"{file_name}.{var_name}: config should be dict"

#     def test_data_var_dims_format_when_present(self, file_conventions_data):
#         """Test that dims field has valid format when present."""
#         file_names = file_conventions_data.get("file_names", [])

#         for entry in file_names:
#             file_name = list(entry.keys())[0]
#             file_config = entry[file_name]
#             data_vars = file_config.get("data_vars", [])

#             for var_entry in data_vars:
#                 if var_entry is None or var_entry == "...":
#                     continue

#                 if isinstance(var_entry, dict):
#                     var_name = list(var_entry.keys())[0]
#                     var_config = var_entry[var_name]

#                     if var_config and "dims" in var_config:
#                         dims = var_config["dims"]
#                         # dims should be a tuple-like string or actual tuple/list
#                         msg = (
#                             f"{file_name}.{var_name}: "
#                             "dims should be string, list or tuple"
#                         )
#                         assert isinstance(dims, (str, list, tuple)), msg

#     def test_data_var_dtype_when_present(self, file_conventions_data):
#         """Test that dtype field has valid value when present."""
#         file_names = file_conventions_data.get("file_names", [])

#         for entry in file_names:
#             file_name = list(entry.keys())[0]
#             file_config = entry[file_name]
#             data_vars = file_config.get("data_vars", [])

#             for var_entry in data_vars:
#                 if var_entry is None or var_entry == "...":
#                     continue

#                 if isinstance(var_entry, dict):
#                     var_name = list(var_entry.keys())[0]
#                     var_config = var_entry[var_name]

#                     if var_config and "dtype" in var_config:
#                         dtype = var_config["dtype"]
#                         msg = f"{file_name}.{var_name}: dtype should be a string"
#                         assert isinstance(dtype, str), msg

#     def test_scripts_section_structure(self, file_conventions_data):
#         """Test that scripts section has proper structure."""
#         scripts = file_conventions_data.get("scripts")

#         assert isinstance(scripts, list), "scripts should be a list"

#         if len(scripts) > 0:
#             for script_entry in scripts:
#                 assert isinstance(
#                     script_entry, dict
#                 ), "Each script entry should be a dictionary"
#                 assert (
#                     len(script_entry) == 1
#                 ), "Each script entry should have one script name"

#     def test_script_entries_have_input_output(self, file_conventions_data):
#         """Test that script entries have input and output sections."""
#         scripts = file_conventions_data.get("scripts", [])

#         for script_entry in scripts:
#             if script_entry is None:
#                 continue

#             script_name = list(script_entry.keys())[0]
#             script_config = script_entry[script_name]

#             assert isinstance(
#                 script_config, list
#             ), f"Script '{script_name}': config should be a list"

#             # Check for input and output in the config
#             config_keys = []
#             for item in script_config:
#                 if isinstance(item, dict):
#                     config_keys.extend(item.keys())

#             # Both input and output should be present
#             assert (
#                 "input" in config_keys
#             ), f"Script '{script_name}': missing 'input' section"
#             assert (
#                 "output" in config_keys
#             ), f"Script '{script_name}': missing 'output' section"

#     def test_script_input_output_are_lists(self, file_conventions_data):
#         """Test that script input and output sections are lists."""
#         scripts = file_conventions_data.get("scripts", [])

#         for script_entry in scripts:
#             if script_entry is None:
#                 continue

#             script_name = list(script_entry.keys())[0]
#             script_config = script_entry[script_name]

#             for item in script_config:
#                 if isinstance(item, dict):
#                     if "input" in item:
#                         assert isinstance(
#                             item["input"], list
#                         ), f"Script '{script_name}': input should be a list"
#                     if "output" in item:
#                         assert isinstance(
#                             item["output"], list
#                         ), f"Script '{script_name}': output should be a list"

#     def test_known_file_names_are_present(self, file_conventions_data):
#         """Test that expected file names are present in the configuration."""
#         expected_files = [
#             "terrain_gramm.nc",
#             "terrain_gral.nc",
#             "buildings.nc",
#             "meteo.nc",
#             "tracer_measurements.nc",
#         ]

#         file_names = file_conventions_data.get("file_names", [])

#         # Extract all file names from the structure
#         defined_files = []
#         for entry in file_names:
#             if isinstance(entry, dict):
#                 defined_files.extend(entry.keys())

#         for expected_file in expected_files:
#             assert (
#                 expected_file in defined_files
#             ), f"Expected file '{expected_file}' not found in file_conventions.yml"

#     def test_no_duplicate_file_names(self, file_conventions_data):
#         """Test that there are no duplicate file names defined."""
#         file_names = file_conventions_data.get("file_names", [])

#         defined_files = []
#         for entry in file_names:
#             if isinstance(entry, dict):
#                 defined_files.extend(entry.keys())

#         # Check for duplicates
#         seen = set()
#         duplicates = set()
#         for file_name in defined_files:
#             if file_name in seen:
#                 duplicates.add(file_name)
#             seen.add(file_name)

#         assert len(duplicates) == 0, f"Duplicate file names found: {duplicates}"

#     def test_no_duplicate_script_names(self, file_conventions_data):
#         """Test that there are no duplicate script names defined."""
#         scripts = file_conventions_data.get("scripts", [])

#         defined_scripts = []
#         for entry in scripts:
#             if isinstance(entry, dict):
#                 defined_scripts.extend(entry.keys())

#         # Check for duplicates
#         seen = set()
#         duplicates = set()
#         for script_name in defined_scripts:
#             if script_name in seen:
#                 duplicates.add(script_name)
#             seen.add(script_name)

#         assert len(duplicates) == 0, f"Duplicate script names found: {duplicates}"
