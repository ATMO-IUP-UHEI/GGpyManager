"""Analysis functions for model validation and comparison."""

from .loss_functions import (
    rmse_loss,
    regularized_loss,
    compound_loss,
    compute_matching_loss,
    get_sim_ids,
)
from .stability import get_allowed_stability_class, load_catalog_filter
from .vertical_gradients import compute_normalized_vertical_gradient

__all__ = [
    # Loss functions
    "rmse_loss",
    "regularized_loss",
    "compound_loss",
    "compute_matching_loss",
    "get_sim_ids",
    # Stability
    "get_allowed_stability_class",
    "load_catalog_filter",
    # Vertical gradients
    "compute_normalized_vertical_gradient",
]
