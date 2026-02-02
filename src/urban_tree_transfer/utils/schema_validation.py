"""Schema validation helpers for Phase 2 pipeline outputs."""

from __future__ import annotations

from typing import Any

import geopandas as gpd

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.config.loader import (
    get_all_feature_names,
    get_chm_feature_names,
    get_metadata_columns,
    get_temporal_feature_names,
    load_feature_config,
)


def _require_project_crs(trees_gdf: gpd.GeoDataFrame) -> None:
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")


def validate_phase2a_output(
    trees_gdf: gpd.GeoDataFrame,
    feature_config: dict[str, Any] | None = None,
) -> None:
    """Validate that GeoDataFrame matches Phase 2a expected schema."""
    if feature_config is None:
        feature_config = load_feature_config()

    _require_project_crs(trees_gdf)
    if trees_gdf.empty:
        raise ValueError("Phase 2a output must not be empty.")

    metadata_columns = get_metadata_columns(feature_config)
    expected_features = get_all_feature_names(config=feature_config)
    required_columns = metadata_columns + expected_features

    missing = [col for col in required_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Phase 2a output missing columns: {missing}")


def validate_phase2b_output(
    trees_gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict[str, Any] | None = None,
) -> None:
    """Validate that GeoDataFrame matches Phase 2b expected schema."""
    if not selected_months:
        raise ValueError("selected_months must be a non-empty list.")
    if feature_config is None:
        feature_config = load_feature_config()

    _require_project_crs(trees_gdf)
    if trees_gdf.empty:
        raise ValueError("Phase 2b output must not be empty.")

    metadata_columns = get_metadata_columns(feature_config)
    chm_features = get_chm_feature_names(include_engineered=True)
    temporal_features = get_temporal_feature_names(months=selected_months, config=feature_config)
    required_columns = metadata_columns + chm_features + temporal_features

    missing = [col for col in required_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Phase 2b output missing columns: {missing}")


def validate_phase2c_output(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    feature_config: dict[str, Any] | None = None,
) -> None:
    """Validate that GeoDataFrame matches Phase 2c expected schema."""
    if not feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")
    if feature_config is None:
        feature_config = load_feature_config()

    _require_project_crs(trees_gdf)
    if trees_gdf.empty:
        raise ValueError("Phase 2c output must not be empty.")

    metadata_columns = get_metadata_columns(feature_config)
    required_columns = (
        metadata_columns
        + feature_columns
        + [
            "outlier_zscore",
            "outlier_mahalanobis",
            "outlier_iqr",
            "outlier_severity",
            "outlier_method_count",
            "block_id",
        ]
    )

    missing = [col for col in required_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Phase 2c output missing columns: {missing}")
