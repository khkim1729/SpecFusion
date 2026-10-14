import numpy as np
import pytest

from specfusion.harmonization import harmonize


def test_harmonize_interpolates_inside_support_and_masks_outside():
    spectra = np.array([[0.0, 1.0, 2.0]])
    values, mask = harmonize(
        spectra,
        source_nm=np.array([400.0, 500.0, 600.0]),
        target_nm=np.array([350.0, 450.0, 550.0, 650.0]),
    )

    np.testing.assert_allclose(values, [[0.0, 0.5, 1.5, 0.0]])
    np.testing.assert_array_equal(mask, [False, True, True, False])


def test_harmonize_rejects_unsorted_or_misaligned_inputs():
    with pytest.raises(ValueError):
        harmonize(np.ones((2, 3)), np.array([400, 600, 500]), np.array([450]))
    with pytest.raises(ValueError):
        harmonize(np.ones((2, 2)), np.array([400, 500, 600]), np.array([450]))
