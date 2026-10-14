"""Raw and physics-branch feature fusion."""

import numpy as np


def fuse(raw: np.ndarray, physics: np.ndarray) -> np.ndarray:
    """Concatenate row-aligned raw and physics features."""
    raw_values = np.asarray(raw)
    physics_values = np.asarray(physics)
    if raw_values.ndim != 2 or physics_values.ndim != 2:
        raise ValueError("raw and physics representations must be 2-D")
    if raw_values.shape[0] != physics_values.shape[0]:
        raise ValueError("raw and physics representations must have the same row count")
    return np.concatenate([raw_values, physics_values], axis=1)
