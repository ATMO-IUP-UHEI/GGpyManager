"""
Provides the context for pytest. The relative import is necessary to test the module
without requiring an installation by the user.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pathlib import Path
import pytest


from ggpymanager import *


@pytest.fixture(scope="session")
def data_path(tmp_path_factory):
    """
    Creates a temporary data directory with pseudo-data for tests.

    Returns
    -------
    data_path : pathlib Path
        Parent of temporary data directory.
    """

    # Configure tmp data path
    data_path = tmp_path_factory.mktemp("data")
    assert isinstance(data_path, Path)
    return data_path
