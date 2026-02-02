"""Unit tests for Phase 2 extraction functions."""

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.config.loader import get_all_s2_features, load_feature_config
from urban_tree_transfer.feature_engineering.extraction import (
    correct_tree_positions,
    extract_all_features,
    extract_chm_features,
    extract_sentinel_features,
)


def _make_trees(points: list[Point]) -> gpd.GeoDataFrame:
    n = len(points)
    return gpd.GeoDataFrame(
        {
            "tree_id": [f"T{i:03d}" for i in range(n)],
            "city": ["berlin"] * n,
            "genus_latin": ["TILIA"] * n,
            "species_latin": [pd.NA] * n,
            "genus_german": [pd.NA] * n,
            "species_german": [pd.NA] * n,
            "plant_year": pd.Series([pd.NA] * n, dtype="Int64"),
            "height_m": np.zeros(n, dtype=float),
            "tree_type": [pd.NA] * n,
            "geometry": points,
        },
        crs=PROJECT_CRS,
    )


def _write_raster(
    path: Path,
    data: np.ndarray,
    *,
    nodata: float | None = None,
    descriptions: list[str] | None = None,
) -> None:
    height, width = data.shape[-2], data.shape[-1]
    count = data.shape[0] if data.ndim == 3 else 1
    transform = from_origin(0.0, 10.0, 1.0, 1.0)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=PROJECT_CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        if count == 1:
            data_to_write = data[0] if data.ndim == 3 else data
            dst.write(data_to_write, 1)
        else:
            dst.write(data)
        if descriptions is not None:
            dst.descriptions = tuple(descriptions)


def test_correct_tree_positions_handles_empty_geometry(tmp_path: Path) -> None:
    gdf = _make_trees([Point()])
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, np.ones((2, 2), dtype=np.float32))

    result, metadata = correct_tree_positions(gdf, chm_path)

    assert bool(result.loc[0, "position_corrected"]) is False
    assert result.loc[0, "correction_distance"] == 0.0
    assert "adaptive_max_radius" in metadata


def test_correct_tree_positions_scoring_prefers_closer_peak(tmp_path: Path) -> None:
    gdf = _make_trees([Point(2.5, 7.5)])
    chm = np.zeros((5, 5), dtype=np.float32)
    chm[2, 3] = 5.0
    chm[2, 0] = 6.0
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, chm)

    result, _ = correct_tree_positions(
        gdf,
        chm_path,
        sample_size=0,
        sampling_radius_m=5.0,
        min_peak_height_m=1.0,
        footprint_size=3,
        tile_size_px=4,
    )

    corrected_point = result.geometry.iloc[0]
    assert np.isclose(corrected_point.x, 3.5)
    assert np.isclose(corrected_point.y, 7.5)


def test_correct_tree_positions_metadata_keys(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    chm = np.array([[5.0, 2.0], [1.0, 2.0]], dtype=np.float32)
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, chm)

    _, metadata = correct_tree_positions(gdf, chm_path, sample_size=1, percentile=75.0)

    assert "adaptive_max_radius" in metadata
    assert "p75_distance" in metadata


def test_correct_tree_positions_no_valid_chm(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    chm = np.full((2, 2), np.nan, dtype=np.float32)
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, chm, nodata=np.nan)

    result, _ = correct_tree_positions(gdf, chm_path, sample_size=1)

    assert bool(result.loc[0, "position_corrected"]) is False
    assert result.loc[0, "correction_distance"] == 0.0


def test_extract_chm_features_batch_processing(tmp_path: Path) -> None:
    points = [Point(0.5, 9.5), Point(1.5, 9.5), Point(0.5, 8.5)]
    gdf = _make_trees(points)
    chm = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, chm)

    result = extract_chm_features(gdf, chm_path, batch_size=2)

    assert np.isclose(result.loc[0, "CHM_1m"], 1.0)
    assert np.isclose(result.loc[1, "CHM_1m"], 2.0)
    assert np.isclose(result.loc[2, "CHM_1m"], 3.0)


def test_extract_chm_features_nodata(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    chm = np.array([[0.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, chm, nodata=0.0)

    result = extract_chm_features(gdf, chm_path)

    assert pd.isna(result.loc[0, "CHM_1m"])


def test_extract_sentinel_features_multi_month(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()

    s2_features = get_all_s2_features()
    band_count = len(s2_features)
    data = np.stack([np.full((2, 2), i + 1, dtype=np.float32) for i in range(band_count)], axis=0)

    for month in [1, 2]:
        path = sentinel_dir / f"S2_berlin_2021_{month:02d}_median.tif"
        _write_raster(path, data, descriptions=s2_features)

    result = extract_sentinel_features(gdf, sentinel_dir, city="berlin", year=2021, months=[1, 2])

    assert f"{s2_features[0]}_01" in result.columns
    assert f"{s2_features[0]}_02" in result.columns
    assert np.isclose(result.loc[0, f"{s2_features[0]}_01"], 1.0)
    assert np.isclose(result.loc[0, f"{s2_features[1]}_02"], 2.0)


def test_extract_sentinel_features_nodata(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()

    s2_features = get_all_s2_features()
    band_count = len(s2_features)
    data = np.stack([np.full((2, 2), i + 1, dtype=np.float32) for i in range(band_count)], axis=0)
    data[:, 0, 0] = 0.0
    path = sentinel_dir / "S2_berlin_2021_01_median.tif"
    _write_raster(path, data, descriptions=s2_features, nodata=0.0)

    result = extract_sentinel_features(gdf, sentinel_dir, city="berlin", year=2021, months=[1])

    assert pd.isna(result.loc[0, f"{s2_features[0]}_01"])


def test_extract_sentinel_features_band_count_validation(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()

    data = np.ones((1, 2, 2), dtype=np.float32)
    path = sentinel_dir / "S2_berlin_2021_01_median.tif"
    _write_raster(path, data)

    with pytest.raises(ValueError, match="Expected"):
        extract_sentinel_features(gdf, sentinel_dir, city="berlin", year=2021, months=[1])


def test_extract_sentinel_features_band_order_validation(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()

    s2_features = get_all_s2_features()
    band_count = len(s2_features)
    data = np.stack([np.full((2, 2), i + 1, dtype=np.float32) for i in range(band_count)], axis=0)
    wrong_order = list(reversed(s2_features))

    path = sentinel_dir / "S2_berlin_2021_01_median.tif"
    _write_raster(path, data, descriptions=wrong_order)

    with pytest.raises(ValueError, match="band order"):
        extract_sentinel_features(gdf, sentinel_dir, city="berlin", year=2021, months=[1])


def test_extract_sentinel_features_missing_file_warns(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        result = extract_sentinel_features(gdf, sentinel_dir, city="berlin", year=2021, months=[1])

    assert any("Sentinel-2 composite missing" in str(w.message) for w in records)
    s2_features = get_all_s2_features()
    assert pd.isna(result.loc[0, f"{s2_features[0]}_01"])


def test_extract_all_features_pipeline(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, np.array([[5.0, 2.0], [1.0, 2.0]], dtype=np.float32))

    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()
    s2_features = get_all_s2_features()
    band_count = len(s2_features)
    data = np.stack([np.full((2, 2), i + 1, dtype=np.float32) for i in range(band_count)], axis=0)
    for month in [1, 2]:
        path = sentinel_dir / f"S2_berlin_2021_{month:02d}_median.tif"
        _write_raster(path, data, descriptions=s2_features)

    feature_config = load_feature_config()
    feature_config["temporal"]["extraction_months"] = [1, 2]

    result, summary = extract_all_features(
        gdf,
        chm_path=chm_path,
        sentinel_dir=sentinel_dir,
        city="berlin",
        feature_config=feature_config,
    )

    assert "CHM_1m" in result.columns
    assert f"{s2_features[0]}_01" in result.columns
    assert summary["total_trees"] == 1


def test_extract_all_features_single_tree(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, np.array([[5.0, 2.0], [1.0, 2.0]], dtype=np.float32))

    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()
    s2_features = get_all_s2_features()
    band_count = len(s2_features)
    data = np.stack([np.full((2, 2), i + 1, dtype=np.float32) for i in range(band_count)], axis=0)
    path = sentinel_dir / "S2_berlin_2021_01_median.tif"
    _write_raster(path, data, descriptions=s2_features)

    feature_config = load_feature_config()
    feature_config["temporal"]["extraction_months"] = [1]

    result, _ = extract_all_features(
        gdf,
        chm_path=chm_path,
        sentinel_dir=sentinel_dir,
        city="berlin",
        feature_config=feature_config,
    )

    assert len(result) == 1


def test_extract_chm_features_raises_on_invalid_batch_size(tmp_path: Path) -> None:
    gdf = _make_trees([Point(0.5, 9.5)])
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, np.ones((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="batch_size"):
        extract_chm_features(gdf, chm_path, batch_size=0)
