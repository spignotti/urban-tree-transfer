"""Data quality control and NaN handling.

This module handles:
- Temporal feature selection (apply month filtering)
- NaN interpolation (within-tree only - no data leakage!)
- CHM feature engineering (z-score, percentile)
- NDVI plausibility filtering
- Plant year filtering
"""

from __future__ import annotations

import logging
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from urban_tree_transfer.config.loader import (
    get_all_feature_names,
    get_all_s2_features,
    get_metadata_columns,
)
from urban_tree_transfer.utils.final_validation import validate_zero_nan
from urban_tree_transfer.utils.schema_validation import validate_phase2b_output

logger = logging.getLogger(__name__)


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
    max_edge_nan_months: int = 1,
    min_valid_months: int = 2,
) -> gpd.GeoDataFrame:
    """Remove trees with too many missing months per feature base.

    Removal criteria:
    1. Trees with >max_nan_months total NaN values in any feature base
    2. Trees with >max_edge_nan_months NaN values at temporal edges
    3. Trees with <min_valid_months valid values (insufficient for interpolation)

    Args:
        trees_gdf: Input trees.
        feature_columns: Feature columns to check.
        max_nan_months: Maximum allowed NaN months per feature (default: 2).
        max_edge_nan_months: Maximum allowed NaN months at edges (default: 1).
        min_valid_months: Minimum required valid months for interpolation (default: 2).

    Returns:
        Filtered GeoDataFrame.
    """
    if not feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")
    if max_nan_months < 0:
        raise ValueError("max_nan_months must be >= 0.")
    if max_edge_nan_months < 0:
        raise ValueError("max_edge_nan_months must be >= 0.")
    if min_valid_months < 1:
        raise ValueError("min_valid_months must be >= 1.")

    feature_bases = set()
    for col in feature_columns:
        if "_" not in col:
            continue
        base, suffix = col.rsplit("_", 1)
        if suffix.isdigit() and len(suffix) == 2:
            feature_bases.add(base)

    mask = pd.Series(True, index=trees_gdf.index)
    for base in feature_bases:
        base_cols = [c for c in feature_columns if c.startswith(f"{base}_")]
        if not base_cols:
            continue

        # Total NaN count per tree
        nan_count_per_tree = trees_gdf[base_cols].isna().sum(axis=1)
        valid_count_per_tree = trees_gdf[base_cols].notna().sum(axis=1)

        # Per-tree edge NaN count (first and last columns)
        first_col, last_col = base_cols[0], base_cols[-1]
        edge_nan_counts_per_tree = trees_gdf[first_col].isna().astype(int) + trees_gdf[
            last_col
        ].isna().astype(int)

        # Removal logic: (>max_nan_months total) OR (>max_edge_nan_months at edges) OR (<min_valid_months)
        mask &= (
            (nan_count_per_tree <= max_nan_months)
            & (edge_nan_counts_per_tree <= max_edge_nan_months)
            & (valid_count_per_tree >= min_valid_months)
        )

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

    interior_nan_count = 0
    for base in feature_bases:
        base_cols = [f"{base}_{month:02d}" for month in selected_months]
        if len(base_cols) <= 2:
            continue
        values = interpolated[base_cols].to_numpy()
        for row in values:
            nan_mask = np.isnan(row)
            if not nan_mask.any():
                continue
            start_edge_len = 0
            for value in nan_mask:
                if not value:
                    break
                start_edge_len += 1
            end_edge_len = 0
            for value in nan_mask[::-1]:
                if not value:
                    break
                end_edge_len += 1
            interior_mask = nan_mask.copy()
            if start_edge_len:
                interior_mask[:start_edge_len] = False
            if end_edge_len:
                interior_mask[len(row) - end_edge_len :] = False
            interior_nan_count += int(interior_mask.sum())

    if interior_nan_count > 0:
        raise RuntimeError(
            "Interior NaN interpolation failed: "
            f"{interior_nan_count} remaining (expected 0). "
            "Check interpolation logic."
        )

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
    genus_column: str = "genus_latin",
    city_column: str = "city",
) -> gpd.GeoDataFrame:
    """Engineer CHM features with genus-specific normalization per city.

    Args:
        trees_gdf: Input trees with CHM_1m column.
        genus_column: Column containing genus labels (default: genus_latin).
        city_column: Column containing city labels (default: city).

    Returns:
        GeoDataFrame with CHM_1m_zscore and CHM_1m_percentile columns.

    Notes:
        - Genera with <10 samples in a city fall back to city-level normalization.
        - This prevents unstable statistics for rare genera; fallbacks are logged.
    """
    if "CHM_1m" not in trees_gdf.columns:
        raise ValueError("Missing required column: CHM_1m")
    if genus_column not in trees_gdf.columns:
        raise ValueError(f"Missing required column: {genus_column}")
    if city_column not in trees_gdf.columns:
        raise ValueError(f"Missing required column: {city_column}")

    def _zscore(values: pd.Series, reference: pd.Series) -> pd.Series:
        std = reference.std(ddof=1)
        if std == 0 or np.isnan(std):
            return pd.Series(np.zeros(len(values)), index=values.index, dtype=float)
        mean = reference.mean()
        return (values - mean) / std

    def _percentile(values: pd.Series, reference: pd.Series) -> pd.Series:
        non_null = reference.dropna()
        if non_null.empty:
            return pd.Series(np.full(len(values), np.nan), index=values.index, dtype=float)
        result = values.map(
            lambda x: np.nan if pd.isna(x) else percentileofscore(non_null, x, kind="rank")
        )
        result = cast(pd.Series, result).clip(0.0, 100.0)
        return result

    def _value_mask(series: pd.Series, value: object) -> pd.Series:
        is_na = pd.isna(value)
        if isinstance(is_na, (bool, np.bool_)) and is_na:
            return series.isna()
        return series == value

    result = trees_gdf.copy()
    result["CHM_1m_zscore"] = np.nan
    result["CHM_1m_percentile"] = np.nan

    city_series = cast(pd.Series, result[city_column])
    genus_series = cast(pd.Series, result[genus_column])

    for city, city_group in result.groupby(city_column, dropna=False):
        city_mask = _value_mask(city_series, city)
        city_reference = result.loc[city_mask, "CHM_1m"]

        for genus, _ in city_group.groupby(genus_column, dropna=False):
            mask = city_mask & _value_mask(genus_series, genus)
            n_samples = int(mask.sum())

            if n_samples < 10:
                logger.warning(
                    "Genus %s in %s has %s samples (<10). Using city-level normalization.",
                    genus,
                    city,
                    n_samples,
                )
                reference = city_reference
            else:
                reference = result.loc[mask, "CHM_1m"]

            values = result.loc[mask, "CHM_1m"]
            result.loc[mask, "CHM_1m_zscore"] = _zscore(values, reference)
            result.loc[mask, "CHM_1m_percentile"] = _percentile(values, reference)

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


def run_quality_pipeline(
    trees_gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict[str, Any],
    max_nan_months: int = 2,
    max_edge_nan_months: int = 1,
    ndvi_min_threshold: float = 0.3,
) -> gpd.GeoDataFrame:
    """Run the core Phase 2b quality pipeline and validate output schema.

    Args:
        trees_gdf: Input Phase 2a GeoDataFrame.
        selected_months: Months to retain (from temporal_selection.json).
        feature_config: Loaded feature config.
        max_nan_months: Maximum NaN months allowed per feature base.
        max_edge_nan_months: Maximum NaN months allowed at edges.
        ndvi_min_threshold: Minimum NDVI plausibility threshold.

    Returns:
        GeoDataFrame validated against Phase 2b schema.
    """
    trees_gdf = apply_temporal_selection(trees_gdf, selected_months, feature_config)

    s2_features = get_all_s2_features(feature_config)
    s2_selected_cols = [
        f"{feature}_{month:02d}" for feature in s2_features for month in selected_months
    ]
    s2_selected_cols = [col for col in s2_selected_cols if col in trees_gdf.columns]

    feature_columns = ["CHM_1m", *s2_selected_cols]
    trees_gdf = filter_nan_trees(
        trees_gdf, feature_columns=feature_columns, max_nan_months=max_nan_months
    )
    trees_gdf = interpolate_features_within_tree(
        trees_gdf,
        feature_columns=s2_selected_cols,
        selected_months=selected_months,
        max_edge_nan_months=max_edge_nan_months,
    )
    trees_gdf = compute_chm_engineered_features(trees_gdf)

    ndvi_columns = [f"NDVI_{month:02d}" for month in selected_months]
    ndvi_columns = [col for col in ndvi_columns if col in trees_gdf.columns]
    trees_gdf = filter_ndvi_plausibility(
        trees_gdf, ndvi_columns=ndvi_columns, min_threshold=ndvi_min_threshold
    )

    validate_phase2b_output(
        trees_gdf, selected_months=selected_months, feature_config=feature_config
    )

    expected_features = get_all_feature_names(
        months=selected_months, include_chm_engineered=True, config=feature_config
    )
    validate_zero_nan(trees_gdf, expected_features, dataset_name="phase2b")

    return trees_gdf
