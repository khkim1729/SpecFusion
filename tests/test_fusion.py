import numpy as np
import pytest

from specfusion.fusion import fuse


def test_fuse_builds_416_column_representation():
    combined = fuse(np.zeros((3, 220)), np.ones((3, 196)))

    assert combined.shape == (3, 416)
    np.testing.assert_array_equal(combined[:, :220], 0)
    np.testing.assert_array_equal(combined[:, 220:], 1)


def test_fuse_rejects_row_mismatch():
    with pytest.raises(ValueError, match="row"):
        fuse(np.zeros((3, 220)), np.ones((2, 196)))
