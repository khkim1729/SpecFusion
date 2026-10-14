"""Aligned evaluation of SpecFusion branches."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .fusion import fuse


def evaluate_representations(
    raw: np.ndarray,
    physics: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    n_estimators: int = 100,
) -> pd.DataFrame:
    """Evaluate raw, physics, and fused representations on one fixed split."""
    raw_values = np.asarray(raw, dtype=float)
    physics_values = np.asarray(physics, dtype=float)
    target = np.asarray(labels).reshape(-1)
    combined = fuse(raw_values, physics_values)
    if raw_values.shape[0] != target.size:
        raise ValueError("feature rows and labels must align")
    indices = np.arange(target.size)
    train, test = train_test_split(
        indices, test_size=0.3, random_state=seed, stratify=target
    )
    rows: list[dict[str, float | int | str]] = []
    for name, features in (
        ("raw", raw_values),
        ("physics", physics_values),
        ("fusion", combined),
    ):
        model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=seed, n_jobs=-1
        )
        model.fit(features[train], target[train])
        prediction = model.predict(features[test])
        rows.append(
            {
                "representation": name,
                "features": features.shape[1],
                "accuracy": accuracy_score(target[test], prediction),
                "macro_f1": f1_score(target[test], prediction, average="macro"),
            }
        )
    return pd.DataFrame(rows)
