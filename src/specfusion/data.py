"""Indian Pines loading for SpecFusion."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


def _required_array(path: str | Path, key: str) -> np.ndarray:
    content = loadmat(Path(path))
    if key not in content:
        raise ValueError(f"MAT file {path} is missing required key {key!r}")
    return np.asarray(content[key])


def load_indian_pines(
    data_path: str | Path, labels_path: str | Path
) -> tuple[np.ndarray, np.ndarray]:
    cube = _required_array(data_path, "indian_pines_corrected")
    labels = _required_array(labels_path, "indian_pines_gt")
    if cube.ndim != 3 or labels.ndim != 2:
        raise ValueError("cube must be 3-D and ground truth must be 2-D")
    if cube.shape[:2] != labels.shape:
        raise ValueError("cube and ground truth spatial shapes differ")
    if labels.min() < 0 or labels.max() > 16:
        raise ValueError("Indian Pines labels must be in the range 0..16")
    return cube, labels.astype(np.int16, copy=False)


def labeled_spectra(cube: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if cube.ndim != 3 or labels.ndim != 2 or cube.shape[:2] != labels.shape:
        raise ValueError("cube and labels must have matching spatial dimensions")
    flat_labels = labels.reshape(-1)
    mask = flat_labels > 0
    return cube.reshape(-1, cube.shape[-1])[mask].astype(float), flat_labels[mask]
