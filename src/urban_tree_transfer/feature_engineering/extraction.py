"""Feature extraction from Phase 1 outputs.

This module handles:
- Tree position correction using CHM local maxima
- CHM height extraction at tree locations
- Sentinel-2 feature extraction (bands + indices)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd


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
    raise NotImplementedError("To be implemented in PRD 002a")


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
    raise NotImplementedError("To be implemented in PRD 002a")


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
    raise NotImplementedError("To be implemented in PRD 002a")


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
    raise NotImplementedError("To be implemented in PRD 002a")
