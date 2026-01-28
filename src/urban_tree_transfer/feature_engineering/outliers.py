"""Outlier detection using multiple methods.

This module handles:
- Z-score outlier detection
- Mahalanobis distance outlier detection
- IQR-based outlier detection (for CHM)
- Consensus-based outlier classification
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd


def detect_zscore_outliers(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    threshold: float = 4.0,
    min_features_flagged: int = 3,
) -> pd.Series:
    """Detect outliers using Z-score method.

    Args:
        trees_gdf: Input GeoDataFrame.
        feature_columns: Columns to check for outliers.
        threshold: Z-score threshold (|z| > threshold = outlier).
        min_features_flagged: Minimum features exceeding threshold.

    Returns:
        Boolean Series indicating outlier status per tree.
    """
    raise NotImplementedError("To be implemented in PRD 002c")


def detect_mahalanobis_outliers(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    alpha: float = 0.001,
) -> pd.Series:
    """Detect outliers using Mahalanobis distance.

    Args:
        trees_gdf: Input GeoDataFrame.
        feature_columns: Columns for multivariate distance.
        alpha: Significance level for chi-squared critical value.

    Returns:
        Boolean Series indicating outlier status per tree.
    """
    raise NotImplementedError("To be implemented in PRD 002c")


def detect_iqr_outliers(
    trees_gdf: gpd.GeoDataFrame,
    column: str,
    multiplier: float = 2.0,
    group_by: str | None = "genus_latin",
) -> pd.Series:
    """Detect outliers using IQR method, optionally grouped.

    Args:
        trees_gdf: Input GeoDataFrame.
        column: Column to check (typically CHM_1m).
        multiplier: IQR multiplier for fence calculation.
        group_by: Column to group by (e.g., genus for within-genus IQR).

    Returns:
        Boolean Series indicating outlier status per tree.
    """
    raise NotImplementedError("To be implemented in PRD 002c")


def classify_outliers_by_consensus(
    zscore_outliers: pd.Series,
    mahalanobis_outliers: pd.Series,
    iqr_outliers: pd.Series,
    min_methods: int = 3,
) -> pd.DataFrame:
    """Classify outliers based on method consensus.

    Categories:
        - CRITICAL: Flagged by min_methods or more methods → remove
        - WARNING: Flagged by 1-2 methods → keep but flag
        - OK: Not flagged by any method

    Args:
        zscore_outliers: Boolean Series from Z-score detection.
        mahalanobis_outliers: Boolean Series from Mahalanobis detection.
        iqr_outliers: Boolean Series from IQR detection.
        min_methods: Minimum methods for CRITICAL classification.

    Returns:
        DataFrame with columns: zscore, mahalanobis, iqr, consensus_count, category.
    """
    raise NotImplementedError("To be implemented in PRD 002c")


def remove_outliers(
    trees_gdf: gpd.GeoDataFrame,
    outlier_classification: pd.DataFrame,
    remove_category: str = "CRITICAL",
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Remove outliers based on classification.

    Args:
        trees_gdf: Input GeoDataFrame.
        outlier_classification: From classify_outliers_by_consensus.
        remove_category: Category to remove (default: only CRITICAL).

    Returns:
        Tuple of (filtered_gdf, removal_report).
    """
    raise NotImplementedError("To be implemented in PRD 002c")
