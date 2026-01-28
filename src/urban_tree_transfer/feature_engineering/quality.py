"""Data quality control and NaN handling.

This module handles:
- Temporal feature selection (apply month filtering)
- NaN interpolation (within-tree only - no data leakage!)
- CHM feature engineering (z-score, percentile)
- NDVI plausibility filtering
- Plant year filtering
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd
import pandas as pd


def apply_temporal_selection(
    trees_gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Remove features for non-selected months.

    Args:
        trees_gdf: GeoDataFrame with all 12 months of features.
        selected_months: Months to keep (from temporal_selection.json).
        feature_config: Loaded feature config.

    Returns:
        GeoDataFrame with only selected months' features.
    """
    raise NotImplementedError("To be implemented in PRD 002b")


def interpolate_nan_values(
    trees_gdf: gpd.GeoDataFrame,
    feature_config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Interpolate NaN values using within-tree temporal interpolation.

    CRITICAL: This uses ONLY within-tree data to avoid data leakage.
    No genus means, no city means, no cross-tree statistics.

    Algorithm:
        1. For each tree, identify NaN months
        2. Interior NaNs: Linear interpolation from adjacent months
        3. Edge NaNs (1 month at start/end): Forward/backward fill
        4. Edge NaNs (≥2 months at start/end): Flag for removal
        5. Trees with >max_nan_months total NaN: Flag for removal

    Args:
        trees_gdf: GeoDataFrame with temporal features.
        feature_config: Loaded feature config (nan_handling settings).

    Returns:
        Tuple of (interpolated_gdf, removal_report).
        removal_report contains per-tree flags and reasons.
    """
    raise NotImplementedError("To be implemented in PRD 002b")


def engineer_chm_features(
    trees_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add CHM engineered features: z-score and percentile within genus.

    Features added:
        - CHM_1m_zscore: (CHM_1m - genus_mean) / genus_std
        - CHM_1m_percentile: percentile rank within genus

    Args:
        trees_gdf: GeoDataFrame with CHM_1m and genus_latin columns.

    Returns:
        GeoDataFrame with additional CHM engineered columns.
    """
    raise NotImplementedError("To be implemented in PRD 002b")


def filter_by_ndvi_plausibility(
    trees_gdf: gpd.GeoDataFrame,
    feature_config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, int]:
    """Remove trees with implausibly low NDVI values.

    A tree should have max(NDVI) >= threshold during growing season.
    Trees below threshold are likely mislocated or non-vegetated.

    Args:
        trees_gdf: GeoDataFrame with NDVI columns.
        feature_config: Loaded feature config (ndvi_plausibility settings).

    Returns:
        Tuple of (filtered_gdf, removed_count).
    """
    raise NotImplementedError("To be implemented in PRD 002b")


def filter_by_plant_year(
    trees_gdf: gpd.GeoDataFrame,
    max_plant_year: int,
) -> tuple[gpd.GeoDataFrame, int]:
    """Remove trees planted too recently for satellite visibility.

    Args:
        trees_gdf: GeoDataFrame with plant_year column.
        max_plant_year: Maximum plant year to include (from chm_assessment.json).

    Returns:
        Tuple of (filtered_gdf, removed_count).
    """
    raise NotImplementedError("To be implemented in PRD 002b")


def run_quality_pipeline(
    trees_gdf: gpd.GeoDataFrame,
    temporal_config: dict[str, Any],
    chm_config: dict[str, Any],
    feature_config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Complete data quality pipeline.

    Steps:
        1. Apply temporal selection
        2. Filter by plant year
        3. Interpolate NaN values
        4. Remove trees with too many NaNs
        5. Engineer CHM features
        6. Apply NDVI plausibility filter
        7. Generate quality report

    Args:
        trees_gdf: Input trees with extracted features.
        temporal_config: From temporal_selection.json.
        chm_config: From chm_assessment.json.
        feature_config: Loaded feature_config.yaml.

    Returns:
        Tuple of (clean_gdf, quality_report).
    """
    raise NotImplementedError("To be implemented in PRD 002b")
