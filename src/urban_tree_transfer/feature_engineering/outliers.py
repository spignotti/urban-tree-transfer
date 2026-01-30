"""Outlier detection using multiple methods.

This module handles:
- Z-score outlier detection
- Mahalanobis distance outlier detection
- IQR-based outlier detection (for CHM)
- Consensus-based outlier classification
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import chi2, zscore

from urban_tree_transfer.config import PROJECT_CRS


def _require_project_crs(trees_gdf: gpd.GeoDataFrame) -> None:
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")


def _validate_columns(
    trees_gdf: gpd.GeoDataFrame,
    columns: Iterable[str],
    *,
    label: str,
) -> list[str]:
    columns_list = list(columns)
    if not columns_list:
        raise ValueError(f"{label} must be a non-empty list.")
    missing = [col for col in columns_list if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return columns_list


def detect_zscore_outliers(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    z_threshold: float = 3.0,
    min_feature_count: int = 10,
) -> pd.Series:
    """Flag trees with many extreme Z-scores.

    Args:
        trees_gdf: Input trees.
        feature_columns: Features to check.
        z_threshold: Z-score absolute value threshold.
        min_feature_count: Minimum features exceeding threshold to flag tree.

    Returns:
        Boolean Series: True for outlier trees.
    """
    _require_project_crs(trees_gdf)
    feature_columns = _validate_columns(trees_gdf, feature_columns, label="feature_columns")
    if z_threshold <= 0:
        raise ValueError("z_threshold must be > 0.")
    if min_feature_count < 1:
        raise ValueError("min_feature_count must be >= 1.")

    values = trees_gdf[feature_columns]
    z_values = np.asarray(zscore(values, nan_policy="omit"))
    if z_values.ndim == 1:
        z_values = z_values.reshape(-1, 1)

    flagged = np.abs(z_values) > z_threshold
    flagged = np.where(np.isnan(z_values), False, flagged)
    counts = flagged.sum(axis=1)
    return pd.Series(counts >= min_feature_count, index=trees_gdf.index, name="outlier_zscore")


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
    _require_project_crs(trees_gdf)
    feature_columns = _validate_columns(trees_gdf, feature_columns, label="feature_columns")
    if "genus_latin" not in trees_gdf.columns:
        raise ValueError("Missing required column: genus_latin")
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be between 0 and 1.")

    flags = pd.Series(False, index=trees_gdf.index, dtype=bool, name="outlier_mahalanobis")
    critical_value = float(chi2.ppf(1 - alpha, df=len(feature_columns)))

    for _genus, group in trees_gdf.groupby("genus_latin"):
        if group.empty:
            continue
        data = group[feature_columns].to_numpy(dtype=float)
        valid_mask = ~np.isnan(data).any(axis=1)
        if valid_mask.sum() < len(feature_columns) + 2:
            continue
        valid_data = data[valid_mask]
        mean = np.mean(valid_data, axis=0)
        cov = np.cov(valid_data, rowvar=False)
        inv_cov = np.linalg.pinv(cov)

        diff = valid_data - mean
        distances = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        genus_flags = distances > critical_value

        group_index = group.index[valid_mask]
        flags.loc[group_index] = genus_flags

    return flags


def detect_iqr_outliers(
    trees_gdf: gpd.GeoDataFrame,
    height_column: str,
    multiplier: float = 1.5,
    group_by: list[str] | str | None = None,
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
    _require_project_crs(trees_gdf)
    if height_column not in trees_gdf.columns:
        raise ValueError(f"Missing required column: {height_column}")
    if multiplier <= 0:
        raise ValueError("multiplier must be > 0.")

    if group_by is None:
        group_by = ["genus_latin", "city"]
    if isinstance(group_by, str):
        group_by = [group_by]
    _validate_columns(trees_gdf, group_by, label="group_by")

    flags = pd.Series(False, index=trees_gdf.index, dtype=bool, name="outlier_iqr")
    for _, group in trees_gdf.groupby(group_by):
        values = group[height_column].astype(float)
        if values.dropna().empty:
            continue
        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        group_flags = (values < lower) | (values > upper)
        flags.loc[group.index] = group_flags.fillna(False)

    return flags


def apply_consensus_outlier_filter(
    gdf: gpd.GeoDataFrame,
    zscore_flags: pd.Series,
    mahal_flags: pd.Series,
    iqr_flags: pd.Series,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Add outlier metadata flags based on method consensus.

    NO TREES ARE REMOVED - all trees retained for ablation studies.

    Args:
        gdf: Input trees.
        zscore_flags: Z-score outlier flags.
        mahal_flags: Mahalanobis outlier flags.
        iqr_flags: IQR outlier flags.

    Returns:
        Tuple of (GeoDataFrame with new columns, statistics dict).
    """
    _require_project_crs(gdf)
    if len(gdf) == 0:
        raise ValueError("gdf must not be empty.")

    flags = {
        "outlier_zscore": zscore_flags.reindex(gdf.index).fillna(False).astype(bool),
        "outlier_mahalanobis": mahal_flags.reindex(gdf.index).fillna(False).astype(bool),
        "outlier_iqr": iqr_flags.reindex(gdf.index).fillna(False).astype(bool),
    }

    for name, series in flags.items():
        if len(series) != len(gdf):
            raise ValueError(f"Flag series length mismatch for {name}.")

    method_count = (
        flags["outlier_zscore"].astype(int)
        + flags["outlier_mahalanobis"].astype(int)
        + flags["outlier_iqr"].astype(int)
    )

    severity = pd.Series("none", index=gdf.index)
    severity.loc[method_count == 1] = "low"
    severity.loc[method_count == 2] = "medium"
    severity.loc[method_count == 3] = "high"

    result = gdf.copy()
    result["outlier_zscore"] = flags["outlier_zscore"]
    result["outlier_mahalanobis"] = flags["outlier_mahalanobis"]
    result["outlier_iqr"] = flags["outlier_iqr"]
    result["outlier_severity"] = severity
    result["outlier_method_count"] = method_count

    stats = {
        "high_count": int((severity == "high").sum()),
        "medium_count": int((severity == "medium").sum()),
        "low_count": int((severity == "low").sum()),
        "none_count": int((severity == "none").sum()),
        "per_method_counts": {
            "zscore": int(flags["outlier_zscore"].sum()),
            "mahalanobis": int(flags["outlier_mahalanobis"].sum()),
            "iqr": int(flags["outlier_iqr"].sum()),
        },
        "total_trees": len(result),
        "trees_removed": 0,
    }

    return result, stats
