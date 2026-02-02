"""Feature selection and redundancy removal.

This module handles:
- Correlation analysis within feature groups
- Redundant feature identification and removal
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.utils.final_validation import validate_zero_nan
from urban_tree_transfer.utils.schema_validation import validate_phase2c_output


def compute_feature_correlations(
    trees_gdf: gpd.GeoDataFrame,
    feature_groups: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """Compute correlation matrices for feature groups.

    Args:
        trees_gdf: GeoDataFrame with features.
        feature_groups: Dict mapping group name to feature list.
                       Example: {'spectral': ['B2_04', 'B3_04', ...]}

    Returns:
        Dict mapping group name to correlation DataFrame.
    """
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")

    if not feature_groups:
        raise ValueError("feature_groups must be a non-empty dict.")

    correlations: dict[str, pd.DataFrame] = {}
    for group_name, columns in feature_groups.items():
        if not columns:
            raise ValueError(f"feature_groups['{group_name}'] must be non-empty.")
        missing = [col for col in columns if col not in trees_gdf.columns]
        if missing:
            raise ValueError(f"Missing columns for group '{group_name}': {missing}")
        correlations[group_name] = pd.DataFrame(trees_gdf[columns]).corr()

    return correlations


def identify_redundant_features(
    correlation_matrices: dict[str, pd.DataFrame],
    threshold: float = 0.95,
    feature_importance: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Identify redundant feature pairs based on correlation threshold.

    For each highly correlated pair (|r| > threshold), marks one for removal.
    Selection criteria: If feature_importance provided, keep the more important
    feature; otherwise default to removing feature_b.

    Args:
        correlation_matrices: From compute_feature_correlations.
        threshold: Correlation threshold for redundancy.
        feature_importance: Optional dict of feature -> importance score.

    Returns:
        List of dicts with keys: feature_to_remove, correlated_with, correlation, reason.
    """
    if not correlation_matrices:
        raise ValueError("correlation_matrices must be a non-empty dict.")
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be between 0 and 1.")

    redundant: list[dict[str, Any]] = []
    for group_name, matrix in correlation_matrices.items():
        if matrix.empty:
            continue
        cols = list(matrix.columns)
        for i, feature_a in enumerate(cols):
            for feature_b in cols[i + 1 :]:
                corr_value = float(matrix.loc[feature_a, feature_b])
                if abs(corr_value) > threshold:
                    to_remove = feature_b
                    correlated_with = feature_a
                    if feature_importance is not None:
                        importance_a = feature_importance.get(feature_a, 0.0)
                        importance_b = feature_importance.get(feature_b, 0.0)
                        if importance_b > importance_a:
                            to_remove = feature_a
                            correlated_with = feature_b
                    redundant.append(
                        {
                            "feature_to_remove": to_remove,
                            "correlated_with": correlated_with,
                            "correlation": corr_value,
                            "reason": f"group={group_name}, |r|>{threshold}",
                        }
                    )
    return redundant


def compute_vif(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for features.

    Args:
        trees_gdf: GeoDataFrame with feature columns.
        feature_columns: Feature columns to compute VIF for.

    Returns:
        DataFrame with columns: feature, vif.
    """
    if not feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")
    missing = [col for col in feature_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    data = trees_gdf[feature_columns].dropna()
    if data.empty:
        raise ValueError("No valid rows available for VIF computation.")

    values = data.to_numpy(dtype=float)
    n_features = values.shape[1]
    vifs: list[float] = []

    for i in range(n_features):
        y = values[:, i]
        if n_features == 1:
            vifs.append(1.0)
            continue

        x = np.delete(values, i, axis=1)
        x = np.column_stack([np.ones(x.shape[0]), x])
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_pred = x @ coeffs

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

        if r_squared >= 1.0:
            vifs.append(float("inf"))
        else:
            vifs.append(float(1.0 / (1.0 - r_squared)))

    return pd.DataFrame({"feature": feature_columns, "vif": vifs})


def validate_final_preparation_output(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
) -> gpd.GeoDataFrame:
    """Validate Phase 2c output schema after final preparation steps."""
    validate_phase2c_output(trees_gdf, feature_columns)
    validate_zero_nan(trees_gdf, feature_columns, dataset_name="phase2c")
    return trees_gdf


def remove_redundant_features(
    trees_gdf: gpd.GeoDataFrame,
    features_to_remove: list[str],
) -> gpd.GeoDataFrame:
    """Remove redundant features from dataset.

    Args:
        trees_gdf: Input GeoDataFrame.
        features_to_remove: List of column names to drop.

    Returns:
        GeoDataFrame with redundant columns removed.
    """
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")
    if features_to_remove is None:
        raise ValueError("features_to_remove must not be None.")

    if not features_to_remove:
        return trees_gdf.copy()

    safe_remove = [col for col in features_to_remove if col != "geometry"]
    if len(safe_remove) < len(features_to_remove):
        warnings.warn("Skipping 'geometry' in features_to_remove.", stacklevel=2)

    missing = [col for col in safe_remove if col not in trees_gdf.columns]
    if missing:
        warnings.warn(f"Columns not found for removal: {missing}", stacklevel=2)

    existing = [col for col in safe_remove if col in trees_gdf.columns]
    return cast(gpd.GeoDataFrame, trees_gdf.drop(columns=existing).copy())
