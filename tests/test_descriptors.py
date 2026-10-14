import numpy as np
import pytest
import warnings

from specfusion.descriptors import compute_descriptors


def test_compute_descriptors_returns_196_unique_finite_features():
    wavelengths = np.linspace(400.0, 2500.0, 220)
    spectra = np.vstack(
        [np.linspace(0.1, 0.8, 220), np.linspace(0.8, 0.1, 220)]
    )

    values, names = compute_descriptors(spectra, wavelengths)

    assert values.shape == (2, 196)
    assert len(names) == len(set(names)) == 196
    assert np.isfinite(values).all()


def test_compute_descriptors_gates_unavailable_spectrum():
    wavelengths = np.linspace(400.0, 2500.0, 220)
    spectra = np.ones((1, 220))

    values, _ = compute_descriptors(spectra, wavelengths, mask=np.zeros(220, dtype=bool))

    np.testing.assert_array_equal(values, np.zeros((1, 196)))


def test_compute_descriptors_rejects_mask_shape_mismatch():
    with np.errstate(all="raise"), pytest.raises(ValueError, match="mask"):
        compute_descriptors(np.zeros((1, 5)), np.arange(5), mask=np.ones(4, dtype=bool))


def test_compute_descriptors_returns_finite_zeros_without_runtime_warnings():
    wavelengths = np.linspace(400.0, 2500.0, 220)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        values, _ = compute_descriptors(np.zeros((1, 220)), wavelengths)

    assert np.isfinite(values).all()
