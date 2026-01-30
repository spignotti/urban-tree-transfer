"""Data quality control and NaN handling.

This module handles:
- Temporal feature selection (apply month filtering)
- NaN interpolation (within-tree only - no data leakage!)
- CHM feature engineering (z-score, percentile)
- NDVI plausibility filtering
- Plant year filtering
"""

from __future__ import annotations

from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from urban_tree_transfer.config.loader import (
    get_all_s2_features,
    get_metadata_columns,
)


def filter_deciduous_genera(
    trees_gdf: gpd.GeoDataFrame,
    deciduous_genera: list[str],
) -> gpd.GeoDataFrame:
    """Filter to deciduous genera only for spectral homogeneity.

    Args:
        trees_gdf: Input trees with genus_latin column.
        deciduous_genera: List of deciduous genus names.

    Returns:
        Filtered GeoDataFrame containing only deciduous genera.
    """
    if "genus_latin" not in trees_gdf.columns:
        raise ValueError("Missing required column: genus_latin")
    if not deciduous_genera:
        raise ValueError("deciduous_genera must be a non-empty list.")

    mask = trees_gdf["genus_latin"].isin(deciduous_genera)
    removed = int((~mask).sum())
    retained = int(mask.sum())
    print(f"Deciduous filter: removed {removed}, retained {retained}.")
    return cast(gpd.GeoDataFrame, trees_gdf.loc[mask].copy())


def filter_by_plant_year(
    trees_gdf: gpd.GeoDataFrame,
    max_year: int,
    reference_year: int = 2021,
) -> gpd.GeoDataFrame:
    """Filter trees by planting year to exclude recently planted trees.

    Args:
        trees_gdf: Input trees with plant_year column.
        max_year: Maximum planting year to include.
        reference_year: Sentinel-2 imagery year for context (metadata only).

    Returns:
        Filtered GeoDataFrame with only established trees.
    """
    if "plant_year" not in trees_gdf.columns:
        raise ValueError("Missing required column: plant_year")
    if max_year > reference_year:
        raise ValueError("max_year cannot exceed reference_year.")

    mask = (trees_gdf["plant_year"] <= max_year) | (trees_gdf["plant_year"].isna())
    removed = int((~mask).sum())
    retained = int(mask.sum())
    print(f"Plant year filter (<= {max_year}): removed {removed}, retained {retained}.")
    return cast(gpd.GeoDataFrame, trees_gdf.loc[mask].copy())


def apply_temporal_selection(
    trees_gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Remove columns for non-selected months.

    Args:
        trees_gdf: GeoDataFrame with all 12 months of features.
        selected_months: Months to keep (from temporal_selection.json).
        feature_config: Loaded feature config.

    Returns:
        GeoDataFrame with only selected months' features.
    """
    if not selected_months:
        raise ValueError("selected_months must be a non-empty list.")
    if any(month < 1 or month > 12 for month in selected_months):
        raise ValueError("selected_months must be between 1 and 12.")

    s2_features = get_all_s2_features(feature_config)
    metadata = get_metadata_columns(feature_config)

    cols_to_keep = [*metadata, "CHM_1m"]
    for feature in s2_features:
        for month in selected_months:
            col_name = f"{feature}_{month:02d}"
            if col_name in trees_gdf.columns:
                cols_to_keep.append(col_name)

    available_s2_cols = [
        col
        for col in trees_gdf.columns
        if any(col.startswith(f"{feature}_") for feature in s2_features)
    ]
    features_before = len(available_s2_cols)
    features_after = len([c for c in cols_to_keep if c not in [*metadata, "CHM_1m"]])
    print(f"Temporal selection: {features_before} -> {features_after} features.")

    missing_columns = [col for col in cols_to_keep if col not in trees_gdf.columns]
    if missing_columns:
        raise ValueError(f"Missing expected columns: {missing_columns}")

    return cast(gpd.GeoDataFrame, trees_gdf.loc[:, cols_to_keep].copy())


def analyze_nan_distribution(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute NaN statistics per feature.

    Args:
        trees_gdf: Input trees.
        feature_columns: List of feature column names.

    Returns:
        DataFrame with per-feature NaN counts and percentages.
    """
    if not feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")
    missing = [col for col in feature_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    nan_counts = trees_gdf[feature_columns].isna().sum()
    nan_pct = (nan_counts / len(trees_gdf)) * 100

    df = (
        pd.DataFrame(
            {
                "feature": nan_counts.index,
                "nan_count": nan_counts.values,
                "nan_percentage": nan_pct.values,
            }
        )
        .sort_values("nan_count", ascending=False)
        .reset_index(drop=True)
    )
    return df


def filter_nan_trees(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    max_nan_months: int = 2,
) -> gpd.GeoDataFrame:
    """Remove trees with too many missing months per feature base.

    Args:
        trees_gdf: Input trees.
        feature_columns: Feature columns to check.
        max_nan_months: Maximum allowed NaN months per feature.

    Returns:
        Filtered GeoDataFrame.
    """
    if not feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")
    if max_nan_months < 0:
        raise ValueError("max_nan_months must be >= 0.")

    feature_bases = set()
    for col in feature_columns:
        if "_" in col:
            base = col.rsplit("_", 1)[0]
            feature_bases.add(base)

    mask = pd.Series(True, index=trees_gdf.index)
    for base in feature_bases:
        base_cols = [c for c in feature_columns if c.startswith(f"{base}_")]
        nan_count_per_tree = trees_gdf[base_cols].isna().sum(axis=1)
        mask &= nan_count_per_tree <= max_nan_months

    removed = int((~mask).sum())
    print(f"NaN filter: removed {removed} trees.")
    return cast(gpd.GeoDataFrame, trees_gdf.loc[mask].copy())


def interpolate_features_within_tree(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    selected_months: list[int],
    max_edge_nan_months: int = 1,
) -> gpd.GeoDataFrame:
    """Fill NaN values using within-tree temporal information only.

    ⚠️ CRITICAL: No cross-tree information is used to avoid data leakage.

    Args:
        trees_gdf: Input trees (after max_nan_months filtering).
        feature_columns: Feature columns to interpolate.
        selected_months: Selected months (for temporal index).
        max_edge_nan_months: Maximum NaN months tolerated at edges (default: 1).

    Returns:
        GeoDataFrame with interpolated values (some trees may still have NaN).

    Algorithm:
        - Interior NaNs: Linear interpolation between adjacent valid values
        - Edge NaNs (≤1 month): Forward/backward fill with nearest value
        - Edge NaNs (≥2 months): No imputation (tree marked for removal)

    Rationale:
        Each tree is processed independently to prevent data leakage between
        train/validation/test sets. Using genus or city means would leak
        information from training data into validation data.

        Edge tolerance (1 month): 87.5% data retained (7/8 months), minimal
        phenological information loss. ≥2 months edge gap indicates significant
        missing phenology (e.g., spring or autumn missing).

    References:
        - PRD 002b Section 1.6 (Within-Tree Interpolation)
        - Jönsson & Eklundh (2004): max 1-2 edge gaps for vegetation time series

    Example:
        >>> # Interior NaN: NDVI_04=0.5, NDVI_05=NaN, NDVI_06=0.7 → NDVI_05=0.6
        >>> # Edge NaN (1 month): NDVI_04=NaN, NDVI_05=0.3 → NDVI_04=0.3
        >>> # Edge NaN (≥2 months): remains NaN (tree removed in validation)
    """
    if not feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")
    if not selected_months:
        raise ValueError("selected_months must be a non-empty list.")
    if max_edge_nan_months < 0:
        raise ValueError("max_edge_nan_months must be >= 0.")

    missing = [col for col in feature_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    selected_months = sorted(set(selected_months))
    feature_bases = sorted({col.rsplit("_", 1)[0] for col in feature_columns if "_" in col})

    interpolated = trees_gdf.copy()

    for base in feature_bases:
        base_cols = [f"{base}_{month:02d}" for month in selected_months]
        missing_cols = [col for col in base_cols if col not in interpolated.columns]
        if missing_cols:
            raise ValueError(f"Missing expected temporal columns: {missing_cols}")

        values_matrix = interpolated[base_cols].to_numpy(dtype=float)
        for idx, values in enumerate(values_matrix):
            series = pd.Series(values, index=selected_months, dtype=float)
            series = series.interpolate(method="linear", limit_area="inside")

            isna = series.isna().to_numpy()
            start_edge_len = 0
            for value in isna:
                if not value:
                    break
                start_edge_len += 1

            end_edge_len = 0
            for value in isna[::-1]:
                if not value:
                    break
                end_edge_len += 1

            if start_edge_len <= max_edge_nan_months:
                series = series.ffill(limit=max_edge_nan_months)
            if end_edge_len <= max_edge_nan_months and start_edge_len <= max_edge_nan_months:
                series = series.bfill(limit=max_edge_nan_months)

            values_matrix[idx] = series.to_numpy()

        interpolated.loc[:, base_cols] = values_matrix

    remaining_nan = interpolated[feature_columns].isna().sum().sum()
    if remaining_nan > 0:
        print(
            f"Within-tree interpolation: {remaining_nan} NaN values remain "
            f"(likely edge NaN ≥{max_edge_nan_months + 1} months). "
            "These trees should be removed in validation."
        )
    else:
        print("Within-tree interpolation: all NaN values successfully filled.")

    return interpolated


def compute_chm_engineered_features(
    trees_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add CHM_1m_zscore and CHM_1m_percentile per city.

    Args:
        trees_gdf: Input trees with CHM_1m and city columns.

    Returns:
        GeoDataFrame with CHM_1m_zscore and CHM_1m_percentile columns.

    Why city-level normalization doesn't cause leakage:
        CHM is an independent variable (from LiDAR/stereo), not derived from
        other trees' spectral signatures. City-level statistics normalize
        structural differences between cities (e.g., Berlin has older/taller
        trees on average) without leaking phenological information.

    Contrast with forbidden approach:
        NDVI genus-normalization WOULD cause leakage (NDVI is dependent variable,
        genus mean would leak class information between train/val sets).

    References:
        - PRD 002b Section 1.7 (CHM Feature Engineering)
        - PRD 002_phase2_feature_engineering_overview.md Section 4.1 (Data Leakage)
    """
    if "CHM_1m" not in trees_gdf.columns:
        raise ValueError("Missing required column: CHM_1m")
    if "city" not in trees_gdf.columns:
        raise ValueError("Missing required column: city")

    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std(ddof=1)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
        return (series - series.mean()) / std

    def _percentile(series: pd.Series) -> pd.Series:
        non_null = series.dropna()
        if non_null.empty:
            return pd.Series(np.full(len(series), np.nan), index=series.index, dtype=float)
        return cast(
            pd.Series,
            series.map(
                lambda x: np.nan if pd.isna(x) else percentileofscore(non_null, x, kind="rank")
            ),
        )

    result = trees_gdf.copy()
    result["CHM_1m_zscore"] = result.groupby("city")["CHM_1m"].transform(_zscore)
    result["CHM_1m_percentile"] = result.groupby("city")["CHM_1m"].transform(_percentile)
    return result


def filter_ndvi_plausibility(
    trees_gdf: gpd.GeoDataFrame,
    ndvi_columns: list[str],
    min_threshold: float = 0.3,
) -> gpd.GeoDataFrame:
    """Remove trees with implausibly low maximum NDVI.

    Args:
        trees_gdf: Input trees.
        ndvi_columns: NDVI columns (e.g., NDVI_04, NDVI_05).
        min_threshold: Minimum max NDVI value (default: 0.3).

    Returns:
        Filtered GeoDataFrame.
    """
    if not ndvi_columns:
        raise ValueError("ndvi_columns must be a non-empty list.")
    missing = [col for col in ndvi_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Missing NDVI columns: {missing}")

    max_ndvi = trees_gdf[ndvi_columns].max(axis=1)
    mask = max_ndvi >= min_threshold
    removed = int((~mask).sum())
    retained = int(mask.sum())
    print(f"NDVI plausibility: removed {removed}, retained {retained}.")
    return cast(gpd.GeoDataFrame, trees_gdf.loc[mask].copy())
