import numpy as np
import pandas as pd

from specfusion.evaluate import evaluate_representations


def test_evaluate_representations_uses_three_aligned_feature_spaces():
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(120, 8))
    physics = rng.normal(size=(120, 5))
    labels = np.tile(np.arange(1, 5), 30)

    first = evaluate_representations(raw, physics, labels, seed=42, n_estimators=10)
    second = evaluate_representations(raw, physics, labels, seed=42, n_estimators=10)

    pd.testing.assert_frame_equal(first, second)
    assert first["representation"].tolist() == ["raw", "physics", "fusion"]
    assert first["features"].tolist() == [8, 5, 13]
    assert first[["accuracy", "macro_f1"]].to_numpy().min() >= 0
    assert first[["accuracy", "macro_f1"]].to_numpy().max() <= 1
