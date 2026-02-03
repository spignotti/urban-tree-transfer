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
from numpy.lib.stride_tricks import sliding_window_view
from rasterio.transform import xy as raster_xy
from rasterio.windows import Window, from_bounds
from shapely.geometry import Point

from urban_tree_transfer.config.loader import (
    get_all_feature_names,
    get_all_s2_features,
    get_metadata_columns,
)
from urban_tree_transfer.utils.geo import ensure_project_crs
from urban_tree_transfer.utils.schema_validation import validate_phase2a_output
from urban_tree_transfer.utils.strings import normalize_city_name

_DEFAULT_BATCH_SIZE = 50000
_DEFAULT_SAMPLING_RADIUS_M = 5.0
_DEFAULT_MIN_PEAK_HEIGHT_M = 3.0
_DEFAULT_FOOTPRINT_SIZE = 3
_DEFAULT_TILE_SIZE_PX = 512


def _compute_local_maxima_mask(
    chm_data: np.ndarray,
    footprint_size: int,
) -> np.ndarray | None:
    if footprint_size <= 0 or footprint_size % 2 == 0:
        raise ValueError("footprint_size must be a positive odd integer.")

    valid_mask = ~np.isnan(chm_data)
    if not np.any(valid_mask):
        return None

    chm_temp = np.where(valid_mask, chm_data, -np.inf)
    if footprint_size == 1:
        local_max = chm_temp
    else:
        pad = footprint_size // 2
        padded = np.pad(chm_temp, pad, mode="constant", constant_values=-np.inf)
        windows = sliding_window_view(padded, (footprint_size, footprint_size))
        local_max = windows.max(axis=(-2, -1))

    return (chm_data == local_max) & valid_mask


def _select_best_peak(
    chm_data: np.ndarray,
    center_row: int,
    center_col: int,
    pixel_size: float,
    search_radius_m: float,
    min_peak_height_m: float,
    footprint_size: int,
) -> tuple[int, int, float, float] | None:
    peak_mask = _compute_local_maxima_mask(chm_data, footprint_size)
    if peak_mask is None:
        return None

    peak_mask = peak_mask & (chm_data >= min_peak_height_m)
    if not np.any(peak_mask):
        return None

    peak_rows, peak_cols = np.where(peak_mask)
    peak_coords = np.column_stack([peak_rows, peak_cols]).astype(np.float64)
    center = np.array([center_row, center_col], dtype=np.float64)
    distances_px = np.linalg.norm(peak_coords - center, axis=1)
    distances_m = distances_px * pixel_size

    within_mask = distances_m <= search_radius_m
    if not np.any(within_mask):
        return None

    valid_rows = peak_rows[within_mask]
    valid_cols = peak_cols[within_mask]
    valid_distances = distances_m[within_mask]
    valid_heights = chm_data[valid_rows, valid_cols]

    scores = valid_distances / (valid_heights + 1.0)
    best_idx = int(np.argmin(scores))

    return (
        int(valid_rows[best_idx]),
        int(valid_cols[best_idx]),
        float(valid_distances[best_idx]),
        float(valid_heights[best_idx]),
    )


def _find_best_peak_distance(
    tree_point: Point,
    chm_path: Path,
    search_radius_m: float,
    min_peak_height_m: float,
    footprint_size: int,
) -> float | None:
    if tree_point is None or tree_point.is_empty:
        return None

    with rasterio.open(chm_path) as src:
        nodata_value = src.nodata
        x, y = tree_point.x, tree_point.y

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
        chm_data = np.where(chm_data < 0, np.nan, chm_data)

        if chm_data.size == 0 or np.all(np.isnan(chm_data)):
            return None

        row, col = src.index(x, y)
        row_in_window = int(row - window.row_off)
        col_in_window = int(col - window.col_off)
        if not (0 <= row_in_window < chm_data.shape[0] and 0 <= col_in_window < chm_data.shape[1]):
            return None

        pixel_size = float(abs(src.res[0]))
        peak = _select_best_peak(
            chm_data=chm_data,
            center_row=row_in_window,
            center_col=col_in_window,
            pixel_size=pixel_size,
            search_radius_m=search_radius_m,
            min_peak_height_m=min_peak_height_m,
            footprint_size=footprint_size,
        )
        if peak is None:
            return None

        return peak[2]


def correct_tree_positions(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
    percentile: float = 90.0,
    sample_size: int = 1000,
    sampling_radius_m: float = _DEFAULT_SAMPLING_RADIUS_M,
    min_peak_height_m: float = _DEFAULT_MIN_PEAK_HEIGHT_M,
    footprint_size: int = _DEFAULT_FOOTPRINT_SIZE,
    tile_size_px: int = _DEFAULT_TILE_SIZE_PX,
    height_weight: float | None = None,
    safety_factor: float | None = None,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Snap tree positions to local CHM peaks using adaptive radius.

    Algorithm:
        Phase 1 (Adaptive Radius Determination):
            1. Sample up to `sample_size` trees randomly
            2. For each sampled tree, find distance to best local peak within `sampling_radius_m`
            3. Compute `percentile` of these distances (e.g., P90)
            4. Set max_radius = ceil(Pxx)

        Phase 2 (Snap-to-Peak):
            5. For each tree, extract CHM window within max_radius
            6. Detect local maxima (footprint_size) above min_peak_height_m
            7. Score candidates by distance / (height + 1)
            8. Snap to peak with minimal score
            9. Store correction_distance for analysis

    Notes:
        - height_weight and safety_factor are deprecated and ignored.

    Returns:
        Tuple of (corrected_gdf, metadata_dict).
        corrected_gdf includes 'position_corrected' (bool) and 'correction_distance' (float).
    """
    trees_gdf = ensure_project_crs(trees_gdf.copy())
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100.")
    if sampling_radius_m <= 0.0:
        raise ValueError("sampling_radius_m must be > 0.")
    if min_peak_height_m < 0.0:
        raise ValueError("min_peak_height_m must be >= 0.")

    geometry = trees_gdf.geometry
    valid_mask = ~geometry.is_empty & ~geometry.isna()
    valid_indices = np.where(valid_mask)[0]
    sampled_indices = valid_indices

    if height_weight is not None or safety_factor is not None:
        warnings.warn(
            "height_weight and safety_factor are deprecated and ignored.",
            stacklevel=2,
        )

    if len(valid_indices) > 0:
        np.random.seed(42)
        sample_count = min(sample_size, len(valid_indices))
        sampled_indices = np.random.choice(valid_indices, size=sample_count, replace=False)

    sample_distances: list[float] = []
    for idx in sampled_indices:
        geom = trees_gdf.geometry.iloc[idx]
        distance = _find_best_peak_distance(
            cast(Point, geom),
            chm_path=chm_path,
            search_radius_m=sampling_radius_m,
            min_peak_height_m=min_peak_height_m,
            footprint_size=footprint_size,
        )
        if distance is not None and not np.isnan(distance):
            sample_distances.append(distance)

    if sample_distances:
        p_distance = float(np.percentile(sample_distances, percentile))
        adaptive_max_radius = float(max(1.0, np.ceil(p_distance)))
    else:
        adaptive_max_radius = float(max(1.0, np.ceil(sampling_radius_m)))
        p_distance = adaptive_max_radius

    percentile_key = (
        f"p{int(percentile)}_distance"
        if float(percentile).is_integer()
        else f"p{percentile}_distance"
    )

    total_trees = len(trees_gdf)
    corrected_geometries: list[Point | None] = [None] * total_trees
    position_corrected: list[bool] = [False] * total_trees
    correction_distances: list[float] = [0.0] * total_trees

    with rasterio.open(chm_path) as src:
        nodata_value = src.nodata
        pixel_size = float(abs(src.res[0]))
        radius_px = int(np.ceil(adaptive_max_radius / pixel_size))
        if radius_px <= 0:
            raise ValueError("Computed radius_px must be > 0.")
        if tile_size_px <= 0:
            raise ValueError("tile_size_px must be > 0.")
        tile_size_px = int(max(1, min(tile_size_px, src.width, src.height)))

        rows = np.full(len(trees_gdf), -1, dtype=int)
        cols = np.full(len(trees_gdf), -1, dtype=int)
        for idx in valid_indices:
            geom = trees_gdf.geometry.iloc[idx]
            row, col = src.index(geom.x, geom.y)
            rows[idx] = row
            cols[idx] = col

        tile_groups: dict[tuple[int, int], list[int]] = {}
        for idx in valid_indices:
            row = int(rows[idx])
            col = int(cols[idx])
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                geom = trees_gdf.geometry.iloc[idx]
                corrected_geometries[idx] = cast(Point, geom)
                position_corrected[idx] = False
                correction_distances[idx] = 0.0
                continue
            tile_row = row // tile_size_px
            tile_col = col // tile_size_px
            tile_groups.setdefault((tile_row, tile_col), []).append(idx)

        for idx, geom in enumerate(trees_gdf.geometry):
            if geom is None or geom.is_empty:
                corrected_geometries[idx] = Point()
                position_corrected[idx] = False
                correction_distances[idx] = 0.0
            elif not valid_mask.iloc[idx]:
                corrected_geometries[idx] = cast(Point, geom)
                position_corrected[idx] = False
                correction_distances[idx] = 0.0

        for (tile_row, tile_col), indices in tile_groups.items():
            row_off = tile_row * tile_size_px
            col_off = tile_col * tile_size_px
            row_off_pad = row_off - radius_px
            col_off_pad = col_off - radius_px
            window = Window.from_slices(
                (row_off_pad, row_off_pad + tile_size_px + 2 * radius_px),
                (col_off_pad, col_off_pad + tile_size_px + 2 * radius_px),
                height=src.height,
                width=src.width,
                boundless=True,
            )

            chm_tile = src.read(1, window=window, boundless=True, fill_value=nodata_value)
            if nodata_value is not None:
                chm_tile = np.where(chm_tile == nodata_value, np.nan, chm_tile)
            chm_tile = np.where(chm_tile < 0, np.nan, chm_tile)

            expected_window_size = 2 * radius_px + 1
            for idx in indices:
                geom = trees_gdf.geometry.iloc[idx]
                row = rows[idx]
                col = cols[idx]
                if row < 0 or col < 0 or row >= src.height or col >= src.width:
                    corrected_geometries[idx] = cast(Point, geom)
                    position_corrected[idx] = False
                    correction_distances[idx] = 0.0
                    continue

                row_in_tile = row - row_off_pad
                col_in_tile = col - col_off_pad
                r_start = row_in_tile - radius_px
                r_end = row_in_tile + radius_px + 1
                c_start = col_in_tile - radius_px
                c_end = col_in_tile + radius_px + 1
                chm_window = chm_tile[r_start:r_end, c_start:c_end]

                if chm_window.shape != (expected_window_size, expected_window_size):
                    padded = np.full(
                        (expected_window_size, expected_window_size),
                        np.nan,
                        dtype=chm_window.dtype,
                    )
                    r_slice = slice(max(0, -r_start), max(0, -r_start) + chm_window.shape[0])
                    c_slice = slice(max(0, -c_start), max(0, -c_start) + chm_window.shape[1])
                    padded[r_slice, c_slice] = chm_window
                    chm_window = padded

                peak = _select_best_peak(
                    chm_data=chm_window,
                    center_row=radius_px,
                    center_col=radius_px,
                    pixel_size=pixel_size,
                    search_radius_m=adaptive_max_radius,
                    min_peak_height_m=min_peak_height_m,
                    footprint_size=footprint_size,
                )

                if peak is None:
                    corrected_geometries[idx] = cast(Point, geom)
                    position_corrected[idx] = False
                    correction_distances[idx] = 0.0
                    continue

                peak_row_local, peak_col_local, snap_distance, _ = peak
                global_row = row - radius_px + peak_row_local
                global_col = col - radius_px + peak_col_local
                peak_x, peak_y = raster_xy(
                    src.transform,
                    global_row,
                    global_col,
                    offset="center",
                )
                corrected_point = Point(float(peak_x), float(peak_y))
                corrected_geometries[idx] = corrected_point
                position_corrected[idx] = True
                correction_distances[idx] = float(snap_distance)

        for idx, geom in enumerate(trees_gdf.geometry):
            if corrected_geometries[idx] is None:
                corrected_geometries[idx] = (
                    Point() if geom is None or geom.is_empty else cast(Point, geom)
                )

    trees_gdf["position_corrected"] = pd.Series(
        position_corrected, dtype=bool, index=trees_gdf.index
    )
    trees_gdf["correction_distance"] = pd.Series(
        correction_distances, dtype=float, index=trees_gdf.index
    )
    trees_gdf = trees_gdf.set_geometry(
        gpd.GeoSeries(
            [cast(Point, geom) if geom is not None else Point() for geom in corrected_geometries],
            index=trees_gdf.index,
            crs=trees_gdf.crs,
        )
    )

    metadata = {
        "adaptive_max_radius": adaptive_max_radius,
        percentile_key: p_distance,
        "sample_size": sample_size,
        "sampled_trees": len(sampled_indices),
        "sampling_radius_m": sampling_radius_m,
        "min_peak_height_m": min_peak_height_m,
        "footprint_size": footprint_size,
        "tile_size_px": tile_size_px,
    }

    return trees_gdf, metadata


def extract_chm_features(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
    batch_size: int = _DEFAULT_BATCH_SIZE,
) -> gpd.GeoDataFrame:
    """Extract CHM_1m value at each tree location.

    This is direct point sampling at 1m resolution - NO neighborhood aggregation.
    We do NOT compute CHM_mean, CHM_max, CHM_std (contaminated in legacy pipeline).

    Args:
        trees_gdf: Tree points (should be position-corrected).
        chm_path: Path to 1m CHM GeoTIFF.
        batch_size: Batch size for raster sampling.

    Returns:
        GeoDataFrame with new column 'CHM_1m' (float, may contain NaN for NoData).

    Note:
        The 'height_m' column from cadastre is DIFFERENT from CHM_1m.
        Both are kept - they come from different sources.
    """
    trees_gdf = ensure_project_crs(trees_gdf.copy())
    trees_gdf["CHM_1m"] = np.nan
    valid_mask = ~trees_gdf.geometry.is_empty & trees_gdf.geometry.notnull()

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    with rasterio.open(chm_path) as src:
        nodata_value = src.nodata
        valid_indices = np.where(valid_mask)[0]

        for start in range(0, len(valid_indices), batch_size):
            batch_indices = valid_indices[start : start + batch_size]
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
    city_key = normalize_city_name(city)
    valid_mask = ~trees_gdf.geometry.is_empty & trees_gdf.geometry.notnull()
    valid_indices = np.where(valid_mask)[0]

    months_sorted = sorted(months)
    all_month_columns = [
        f"{feature}_{month:02d}" for month in months_sorted for feature in s2_features
    ]
    s2_df = pd.DataFrame(
        np.nan,
        index=trees_gdf.index,
        columns=pd.Index(all_month_columns, dtype="object"),
        dtype=float,
    )
    overlap_cols = [col for col in all_month_columns if col in trees_gdf.columns]
    if overlap_cols:
        trees_gdf = cast(gpd.GeoDataFrame, trees_gdf.drop(columns=overlap_cols))

    for month_index, month in enumerate(months_sorted):
        file_path = sentinel_dir / f"S2_{city_key}_{year}_{month:02d}_median.tif"
        if not file_path.exists():
            title_name = city.title()
            fallback_path = sentinel_dir / f"S2_{title_name}_{year}_{month:02d}_median.tif"
            if fallback_path.exists():
                warnings.warn(
                    "Sentinel-2 composite uses non-normalized city name. "
                    f"Expected {file_path.name}, found {fallback_path.name}.",
                    stacklevel=2,
                )
                file_path = fallback_path
            else:
                warnings.warn(
                    f"Sentinel-2 composite missing for {city_key} {year}-{month:02d}: {file_path}",
                    stacklevel=2,
                )
                continue

        with rasterio.open(file_path) as src:
            if src.count != len(s2_features):
                raise ValueError(
                    f"Expected {len(s2_features)} bands in {file_path}, found {src.count}."
                )
            descriptions = list(src.descriptions or [])
            band_indexes: list[int] | None = None
            if descriptions and all(desc is not None for desc in descriptions):
                description_map = {desc: idx + 1 for idx, desc in enumerate(descriptions)}
                missing = [band for band in s2_features if band not in description_map]
                if missing:
                    raise ValueError(
                        "Sentinel-2 band descriptions missing expected bands. "
                        f"Missing={missing}, found={descriptions}."
                    )
                band_indexes = [description_map[band] for band in s2_features]
            elif descriptions:
                warnings.warn(
                    f"Band descriptions incomplete in {file_path}; assuming expected order.",
                    stacklevel=2,
                )
            nodata_value = src.nodata

            for start in range(0, len(valid_indices), batch_size):
                batch_indices = valid_indices[start : start + batch_size]
                batch_coords = [
                    (trees_gdf.geometry.iloc[idx].x, trees_gdf.geometry.iloc[idx].y)
                    for idx in batch_indices
                ]
                if band_indexes is None:
                    samples = np.array(list(src.sample(batch_coords)), dtype=np.float32)
                else:
                    samples = np.array(
                        list(src.sample(batch_coords, indexes=band_indexes)), dtype=np.float32
                    )
                if np.ma.isMaskedArray(samples):
                    samples = samples.filled(np.nan)
                if nodata_value is not None:
                    samples = np.where(samples == nodata_value, np.nan, samples)
                start_col = month_index * len(s2_features)
                end_col = start_col + len(s2_features)
                s2_df.iloc[batch_indices, start_col:end_col] = samples[:, : len(s2_features)]

    trees_gdf = cast(gpd.GeoDataFrame, pd.concat([trees_gdf, s2_df], axis=1))
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

    tree_correction = feature_config.get("tree_position_correction", {})
    percentile = float(tree_correction.get("percentile", 90.0))
    sample_size = int(tree_correction.get("sample_size", 1000))
    sampling_radius_m = float(tree_correction.get("sampling_radius_m", _DEFAULT_SAMPLING_RADIUS_M))
    min_peak_height_m = float(tree_correction.get("min_peak_height_m", _DEFAULT_MIN_PEAK_HEIGHT_M))
    footprint_size = int(tree_correction.get("footprint_size", _DEFAULT_FOOTPRINT_SIZE))
    tile_size_px = int(tree_correction.get("tile_size_px", _DEFAULT_TILE_SIZE_PX))

    corrected_gdf, correction_metadata = correct_tree_positions(
        trees_gdf=trees_gdf,
        chm_path=chm_path,
        percentile=percentile,
        sample_size=sample_size,
        sampling_radius_m=sampling_radius_m,
        min_peak_height_m=min_peak_height_m,
        footprint_size=footprint_size,
        tile_size_px=tile_size_px,
    )

    processing_cfg = feature_config.get("processing", {})
    batch_size = int(processing_cfg.get("batch_size", _DEFAULT_BATCH_SIZE))

    chm_gdf = extract_chm_features(corrected_gdf, chm_path=chm_path, batch_size=batch_size)

    temporal_cfg = feature_config.get("temporal", {})
    months = temporal_cfg.get("extraction_months", list(range(1, 13)))
    year = int(temporal_cfg.get("reference_year", 2021))

    full_gdf = extract_sentinel_features(
        trees_gdf=chm_gdf,
        sentinel_dir=sentinel_dir,
        city=city,
        year=year,
        months=months,
        batch_size=batch_size,
    )

    validate_phase2a_output(full_gdf, feature_config)

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
        "position_correction": correction_metadata,
        "processing_timestamp": pd.Timestamp.now("UTC").isoformat(),
    }

    return full_gdf, summary
