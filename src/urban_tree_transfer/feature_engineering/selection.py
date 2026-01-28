"""Feature selection and redundancy removal.

This module handles:
- Correlation analysis within feature groups
- Redundant feature identification and removal
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd


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
    raise NotImplementedError("To be implemented in PRD 002c")


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
    raise NotImplementedError("To be implemented in PRD 002c")


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
    raise NotImplementedError("To be implemented in PRD 002c")
