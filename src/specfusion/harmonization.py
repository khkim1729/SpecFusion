"""Wavelength-grid harmonization with explicit validity masks."""

import numpy as np
from scipy.interpolate import interp1d


def harmonize(
    spectra: np.ndarray, source_nm: np.ndarray, target_nm: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate spectra onto a target grid and mask extrapolated wavelengths."""
    values = np.asarray(spectra, dtype=float)
    source = np.asarray(source_nm, dtype=float).reshape(-1)
    target = np.asarray(target_nm, dtype=float).reshape(-1)
    if values.ndim != 2 or values.shape[1] != source.size:
        raise ValueError("spectra columns must align with source wavelengths")
    if source.size < 2 or not np.all(np.diff(source) > 0):
        raise ValueError("source wavelengths must be strictly increasing")
    if target.size < 1 or not np.all(np.diff(target) > 0):
        raise ValueError("target wavelengths must be strictly increasing")
    valid = (target >= source[0]) & (target <= source[-1])
    interpolator = interp1d(
        source, values, axis=1, bounds_error=False, fill_value=0.0, assume_sorted=True
    )
    return interpolator(target), valid
