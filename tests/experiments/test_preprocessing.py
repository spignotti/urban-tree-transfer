"""Unit tests for Phase 3 preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urban_tree_transfer.experiments.preprocessing import (
    encode_genus_labels,
    scale_features,
)


def test_encode_genus_labels_sorted_mapping() -> None:
    labels = pd.Series(["TILIA", "ACER", "TILIA", "BETULA"])

    encoded, label_to_idx, idx_to_label = encode_genus_labels(labels)

    assert label_to_idx["ACER"] == 0
    assert label_to_idx["BETULA"] == 1
    assert label_to_idx["TILIA"] == 2
    assert idx_to_label[0] == "ACER"
    assert len(encoded) == len(labels)


def test_encode_genus_labels_missing_values() -> None:
    labels = pd.Series(["TILIA", pd.NA])

    with pytest.raises(ValueError, match="missing values"):
        encode_genus_labels(labels)


def test_scale_features_no_leakage() -> None:
    x_train = pd.DataFrame({"a": [0.0, 2.0], "b": [1.0, 3.0]})
    x_val = pd.DataFrame({"a": [10.0, 12.0], "b": [11.0, 13.0]})

    x_train_scaled, x_val_scaled, _, scaler = scale_features(x_train, x_val=x_val)

    assert np.allclose(scaler.mean_, np.array([1.0, 2.0]))
    assert x_train_scaled.shape == (2, 2)
    assert x_val_scaled is not None
    assert np.abs(x_val_scaled.mean(axis=0)).min() > 1.0
