"""Feature selection and redundancy removal.

This module handles:
- Correlation analysis within feature groups
- Redundant feature identification and removal
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import geopandas as gpd
import pandas as pd

from urban_tree_transfer.config import PROJECT_CRS


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
) -> list[dict[str, Any]]:
    """Identify redundant feature pairs based on correlation threshold.

    For each highly correlated pair (|r| > threshold), marks one for removal.
    Selection criteria: Keep the feature with higher variance.

    Args:
        correlation_matrices: From compute_feature_correlations.
        threshold: Correlation threshold for redundancy.

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
                    redundant.append(
                        {
                            "feature_to_remove": feature_b,
                            "correlated_with": feature_a,
                            "correlation": corr_value,
                            "reason": f"group={group_name}, |r|>{threshold}",
                        }
                    )
    return redundant


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
