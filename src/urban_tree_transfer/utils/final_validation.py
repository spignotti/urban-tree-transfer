"""Final dataset validation utilities for Phase 2c outputs."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


def validate_zero_nan(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    dataset_name: str,
) -> None:
    """Assert that no NaN values remain in feature columns.

    Args:
        trees_gdf: Dataset to validate.
        feature_columns: Feature columns to check.
        dataset_name: Label for error messages.

    Raises:
        ValueError: If any NaN values remain in feature columns.
    """
    if not feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")

    missing = [col for col in feature_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Missing feature columns in {dataset_name}: {missing}")

    nan_counts = trees_gdf[feature_columns].isna().sum()
    nan_features = nan_counts[nan_counts > 0]

    if not nan_features.empty:
        details = pd.Series(nan_features).to_string()
        raise ValueError(
            f"{dataset_name}: Found NaN in feature columns:\n{details}\n"
            "All NaN should be resolved in Phase 2b."
        )
