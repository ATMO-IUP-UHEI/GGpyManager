from ggpymanager.catalog import Catalog
from ggpymanager import utils
from ggpymanager import config

# Expose new modular structure
from ggpymanager import io
from ggpymanager import models
from ggpymanager import processing
from ggpymanager import analysis

__all__ = [
    "Catalog",
    "utils",
    "config",
    # New modules
    "io",
    "models",
    "processing",
    "analysis",
]
