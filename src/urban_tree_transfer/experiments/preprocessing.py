"""Preprocessing utilities for Phase 3 experiments."""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from urban_tree_transfer.experiments.data_loading import get_feature_columns


def encode_genus_labels(
    y: pd.Series,
) -> tuple[np.ndarray, dict[str, int], dict[int, str]]:
    """Encode genus_latin labels to integers.

    Args:
        y: Series of genus_latin values.

    Returns:
        Tuple of (encoded_labels, label_to_idx, idx_to_label).

    Raises:
        ValueError: If labels contain missing values.
    """
    if y.isna().any():
        raise ValueError("genus_latin contains missing values.")

    label_encoder = LabelEncoder()
    labels_sorted = sorted(y.unique())
    label_encoder.fit(labels_sorted)
    classes_raw = label_encoder.classes_
    if classes_raw is None:
        raise ValueError("LabelEncoder classes not initialized.")
    classes = list(classes_raw)

    encoded = label_encoder.transform(y)
    label_to_idx = {label: int(idx) for idx, label in enumerate(classes)}
    idx_to_label = {int(idx): label for idx, label in enumerate(classes)}

    return encoded, label_to_idx, idx_to_label


def scale_features(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame | None = None,
    x_test: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, StandardScaler]:
    """Scale features using StandardScaler (fit on train, transform all).

    Args:
        x_train: Training features.
        x_val: Validation features (optional).
        x_test: Test features (optional).

    Returns:
        Tuple of (X_train_scaled, X_val_scaled, X_test_scaled, scaler).

    Raises:
        ValueError: If training data is empty.
    """
    if x_train.empty:
        raise ValueError("x_train is empty.")

    scaler = StandardScaler()
    x_train_scaled = cast(np.ndarray, scaler.fit_transform(x_train.to_numpy(dtype=float)))

    x_val_scaled = None
    if x_val is not None:
        x_val_scaled = cast(np.ndarray, scaler.transform(x_val.to_numpy(dtype=float)))

    x_test_scaled = None
    if x_test is not None:
        x_test_scaled = cast(np.ndarray, scaler.transform(x_test.to_numpy(dtype=float)))

    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


def prepare_data_for_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    chm_features: list[str] | None = None,
) -> dict:
    """Complete preprocessing pipeline: extract features, scale, encode labels.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        test_df: Test DataFrame.
        chm_features: Specific CHM features to use (default: all 3).

    Returns:
        Dict with feature arrays, labels, mappings, and scaler.

    Raises:
        ValueError: If required columns are missing or labels are inconsistent.
    """
    for name, df in {"train": train_df, "val": val_df, "test": test_df}.items():
        if "genus_latin" not in df.columns:
            raise ValueError(f"{name} DataFrame missing genus_latin column.")

    feature_columns = get_feature_columns(train_df, chm_features=chm_features)

    x_train = cast(pd.DataFrame, train_df[feature_columns])
    x_val = cast(pd.DataFrame, val_df[feature_columns])
    x_test = cast(pd.DataFrame, test_df[feature_columns])

    y_train_series = cast(pd.Series, train_df["genus_latin"])
    y_train_encoded, label_to_idx, idx_to_label = encode_genus_labels(y_train_series)

    def encode_split(labels: pd.Series) -> np.ndarray:
        unknown = sorted(set(labels.unique()) - set(label_to_idx.keys()))
        if unknown:
            raise ValueError(f"Unknown genus labels in split: {unknown}")
        mapped = labels.map(label_to_idx)
        if mapped.isna().any():
            raise ValueError("Failed to encode labels due to missing mappings.")
        return mapped.to_numpy(dtype=int)

    y_val_series = cast(pd.Series, val_df["genus_latin"])
    y_test_series = cast(pd.Series, test_df["genus_latin"])
    y_val_encoded = encode_split(y_val_series)
    y_test_encoded = encode_split(y_test_series)

    x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_features(
        x_train, x_val=x_val, x_test=x_test
    )

    return {
        "X_train": x_train_scaled,
        "X_val": x_val_scaled,
        "X_test": x_test_scaled,
        "y_train": y_train_encoded,
        "y_val": y_val_encoded,
        "y_test": y_test_encoded,
        "feature_names": feature_columns,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "scaler": scaler,
        "n_features": len(feature_columns),
        "n_classes": len(label_to_idx),
    }
