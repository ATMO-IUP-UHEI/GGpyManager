"""
Provides the context for pytest. The relative import is necessary to test the module
without requiring an installation by the user.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ggpymanager import *
