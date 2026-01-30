"""Feature extraction from Phase 1 outputs.

This module handles:
- Tree position correction using CHM local maxima
- CHM height extraction at tree locations
- Sentinel-2 feature extraction (bands + indices)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.windows import transform as window_transform
from shapely.geometry import Point

from urban_tree_transfer.config.loader import (
    get_all_feature_names,
    get_all_s2_features,
    get_metadata_columns,
)
from urban_tree_transfer.utils.geo import ensure_project_crs

_DEFAULT_BATCH_SIZE = 50000


def correct_tree_positions(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
    search_radius_m: float = 5.0,
    height_weight: float = 0.7,
) -> gpd.GeoDataFrame:
    """Snap tree positions to local CHM maximum within search radius.

    Algorithm:
        1. For each tree, extract CHM values within search_radius_m
        2. Find local maximum using height-weighted distance score:
           score = height * height_weight - distance * (1 - height_weight)
        3. Snap tree to pixel with highest score
        4. Flag trees that couldn't be corrected (no valid CHM data)

    Args:
        trees_gdf: Input tree point geometries.
        chm_path: Path to 1m CHM GeoTIFF.
        search_radius_m: Maximum search distance for local peak.
        height_weight: Weight for height vs distance in peak selection (0-1).

    Returns:
        GeoDataFrame with corrected geometries and 'position_corrected' column (bool).

    Note:
        Uses rasterio windowed reading for memory efficiency.
    """
    if not 0.0 <= height_weight <= 1.0:
        raise ValueError("height_weight must be between 0 and 1.")

    trees_gdf = ensure_project_crs(trees_gdf.copy())

    corrected_geometries: list[Point] = []
    position_corrected: list[bool] = []

    with rasterio.open(chm_path) as src:
        nodata_value = src.nodata

        for geom in trees_gdf.geometry:
            if geom is None or geom.is_empty:
                corrected_geometries.append(Point())
                position_corrected.append(False)
                continue

            x, y = geom.x, geom.y
            window = (
                from_bounds(
                    x - search_radius_m,
                    y - search_radius_m,
                    x + search_radius_m,
                    y + search_radius_m,
                    src.transform,
                )
                .round_offsets()
                .round_lengths()
            )

            chm_data = src.read(1, window=window, boundless=True, fill_value=nodata_value)
            if nodata_value is not None:
                chm_data = np.where(chm_data == nodata_value, np.nan, chm_data)

            if chm_data.size == 0 or np.all(np.isnan(chm_data)):
                corrected_geometries.append(cast(Point, geom))
                position_corrected.append(False)
                continue

            win_transform = window_transform(window, src.transform)
            height, width = chm_data.shape

            cols = np.arange(width, dtype=np.float32)
            rows = np.arange(height, dtype=np.float32)
            xs = win_transform.c + (cols + 0.5) * win_transform.a
            ys = win_transform.f + (rows + 0.5) * win_transform.e
            xx, yy = np.meshgrid(xs, ys)

            distances = np.sqrt((xx - x) ** 2 + (yy - y) ** 2)
            valid_mask = ~np.isnan(chm_data) & (distances <= search_radius_m)
            if not np.any(valid_mask):
                corrected_geometries.append(cast(Point, geom))
                position_corrected.append(False)
                continue

            scores = (chm_data * height_weight) - (distances * (1.0 - height_weight))
            scores = np.where(valid_mask, scores, -np.inf)

            if np.all(np.isneginf(scores)):
                corrected_geometries.append(cast(Point, geom))
                position_corrected.append(False)
                continue

            best_index = int(np.nanargmax(scores))
            best_row, best_col = np.unravel_index(best_index, chm_data.shape)
            corrected_geometries.append(
                Point(float(xx[best_row, best_col]), float(yy[best_row, best_col]))
            )
            position_corrected.append(True)

    trees_gdf["position_corrected"] = pd.Series(position_corrected, dtype=bool)
    trees_gdf = trees_gdf.set_geometry(
        gpd.GeoSeries(corrected_geometries, index=trees_gdf.index, crs=trees_gdf.crs)
    )
    return trees_gdf


def extract_chm_features(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
) -> gpd.GeoDataFrame:
    """Extract CHM_1m value at each tree location.

    This is direct point sampling at 1m resolution - NO neighborhood aggregation.
    We do NOT compute CHM_mean, CHM_max, CHM_std (contaminated in legacy pipeline).

    Args:
        trees_gdf: Tree points (should be position-corrected).
        chm_path: Path to 1m CHM GeoTIFF.

    Returns:
        GeoDataFrame with new column 'CHM_1m' (float, may contain NaN for NoData).

    Note:
        The 'height_m' column from cadastre is DIFFERENT from CHM_1m.
        Both are kept - they come from different sources.
    """
    trees_gdf = ensure_project_crs(trees_gdf.copy())
    trees_gdf["CHM_1m"] = np.nan
    valid_mask = ~trees_gdf.geometry.is_empty & trees_gdf.geometry.notnull()

    with rasterio.open(chm_path) as src:
        nodata_value = src.nodata
        valid_indices = np.where(valid_mask)[0]

        for start in range(0, len(valid_indices), _DEFAULT_BATCH_SIZE):
            batch_indices = valid_indices[start : start + _DEFAULT_BATCH_SIZE]
            batch_coords = [
                (trees_gdf.geometry.iloc[idx].x, trees_gdf.geometry.iloc[idx].y)
                for idx in batch_indices
            ]
            samples = np.array(list(src.sample(batch_coords)), dtype=np.float32).reshape(-1)
            if nodata_value is not None:
                samples = np.where(samples == nodata_value, np.nan, samples)
            chm_col_idx = trees_gdf.columns.get_loc("CHM_1m")
            trees_gdf.iloc[batch_indices, chm_col_idx] = samples
    return trees_gdf


def extract_sentinel_features(
    trees_gdf: gpd.GeoDataFrame,
    sentinel_dir: Path,
    city: str,
    year: int,
    months: list[int],
    batch_size: int = 50000,
) -> gpd.GeoDataFrame:
    """Extract Sentinel-2 bands and indices for specified months.

    Args:
        trees_gdf: Tree points (with CHM features).
        sentinel_dir: Directory containing monthly composites.
        city: City name (for filename pattern matching).
        year: Reference year (2021).
        months: List of months to extract (1-12).
        batch_size: Number of trees to process per batch (memory management).

    Returns:
        GeoDataFrame with new columns: {band}_{MM}, {index}_{MM}
        Example: 'NDVI_04' for April NDVI, 'B8_07' for July B8.

    File naming convention (from Phase 1):
        S2_{city}_{year}_{MM}_median.tif

    Band order in composites (23 bands total):
        Bands 1-10: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        Bands 11-23: NDVI, EVI, GNDVI, NDre1, NDVIre, CIre, IRECI,
                     RTVIcore, NDWI, MSI, NDII, kNDVI, VARI

    Note:
        - Uses rasterio for efficient multi-band reading
        - Preserves NoData as NaN (handled in PRD 002b)
        - Batch processing to avoid memory issues with large datasets
    """
    trees_gdf = ensure_project_crs(trees_gdf.copy())

    s2_features = get_all_s2_features()
    valid_mask = ~trees_gdf.geometry.is_empty & trees_gdf.geometry.notnull()
    valid_indices = np.where(valid_mask)[0]

    for month in sorted(months):
        month_columns = [f"{feature}_{month:02d}" for feature in s2_features]
        for col in month_columns:
            if col not in trees_gdf.columns:
                trees_gdf[col] = np.nan

        file_path = sentinel_dir / f"S2_{city}_{year}_{month:02d}_median.tif"
        if not file_path.exists():
            warnings.warn(
                f"Sentinel-2 composite missing for {city} {year}-{month:02d}: {file_path}",
                stacklevel=2,
            )
            continue

        with rasterio.open(file_path) as src:
            if src.count < len(s2_features):
                raise ValueError(
                    f"Expected {len(s2_features)} bands in {file_path}, found {src.count}."
                )
            nodata_value = src.nodata

            for start in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[start : start + batch_size]
                batch_coords = [
                    (trees_gdf.geometry.iloc[idx].x, trees_gdf.geometry.iloc[idx].y)
                    for idx in batch_indices
                ]
                samples = np.array(list(src.sample(batch_coords)), dtype=np.float32)
                if np.ma.isMaskedArray(samples):
                    samples = samples.filled(np.nan)
                if nodata_value is not None:
                    samples = np.where(samples == nodata_value, np.nan, samples)
                col_positions = trees_gdf.columns.get_indexer(month_columns)
                trees_gdf.iloc[batch_indices, col_positions] = samples[:, : len(s2_features)]

    return trees_gdf


def extract_all_features(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
    sentinel_dir: Path,
    city: str,
    feature_config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Complete feature extraction pipeline for one city.

    Processing steps:
        1. Correct tree positions (snap to CHM peaks)
        2. Extract CHM features (1m point sampling)
        3. Extract Sentinel-2 features (all 12 months)
        4. Validate output schema
        5. Generate extraction summary

    Args:
        trees_gdf: Input trees from Phase 1.
        chm_path: Path to 1m CHM raster.
        sentinel_dir: Directory with Sentinel-2 monthly composites.
        city: City name.
        feature_config: Loaded feature_config.yaml.

    Returns:
        Tuple of (trees_with_features, extraction_summary).

    Summary dict contains:
        - total_trees: Number of trees processed
        - trees_corrected: Trees with successful position correction
        - feature_completeness: Dict of feature -> completion rate
        - nan_statistics: Dict of feature -> NaN count
        - processing_timestamp: ISO timestamp
    """
    trees_gdf = ensure_project_crs(trees_gdf)

    tree_correction = feature_config.get("tree_correction", {})
    search_radius_m = float(tree_correction.get("search_radius_m", 5.0))
    height_weight = float(tree_correction.get("height_weight", 0.7))

    corrected_gdf = correct_tree_positions(
        trees_gdf=trees_gdf,
        chm_path=chm_path,
        search_radius_m=search_radius_m,
        height_weight=height_weight,
    )

    chm_gdf = extract_chm_features(corrected_gdf, chm_path=chm_path)

    temporal_cfg = feature_config.get("temporal", {})
    months = temporal_cfg.get("extraction_months", list(range(1, 13)))
    year = int(temporal_cfg.get("reference_year", 2021))

    full_gdf = extract_sentinel_features(
        trees_gdf=chm_gdf,
        sentinel_dir=sentinel_dir,
        city=city,
        year=year,
        months=months,
    )

    metadata_columns = get_metadata_columns(feature_config)
    expected_features = get_all_feature_names(months=months, config=feature_config)
    missing_features = [col for col in expected_features if col not in full_gdf.columns]

    if missing_features:
        warnings.warn(
            f"Missing expected feature columns: {missing_features}",
            stacklevel=2,
        )

    feature_columns = [col for col in expected_features if col in full_gdf.columns]
    nan_counts = full_gdf[feature_columns].isna().sum().to_dict()
    total_trees = len(full_gdf)
    feature_completeness = {
        feature: float(1.0 - (nan_counts.get(feature, 0) / total_trees))
        for feature in feature_columns
    }

    trees_corrected = (
        int(full_gdf["position_corrected"].sum()) if "position_corrected" in full_gdf.columns else 0
    )
    summary = {
        "total_trees": int(total_trees),
        "trees_corrected": trees_corrected,
        "feature_completeness": feature_completeness,
        "nan_statistics": nan_counts,
        "missing_features": missing_features,
        "metadata_columns": metadata_columns,
        "processing_timestamp": pd.Timestamp.now("UTC").isoformat(),
    }

    return full_gdf, summary
