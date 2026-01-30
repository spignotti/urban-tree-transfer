"""Unit tests for Phase 2 feature extraction."""

from pathlib import Path
from tempfile import TemporaryDirectory

import geopandas as gpd
import numpy as np
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


def _write_single_band_raster(path: Path, data: np.ndarray, nodata: float = -9999.0) -> None:
    transform = from_origin(0.0, float(data.shape[0]), 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=PROJECT_CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def _write_multiband_raster(path: Path, data: np.ndarray, nodata: float = -9999.0) -> None:
    transform = from_origin(0.0, float(data.shape[1]), 1.0, 1.0)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=data.shape[0],
        dtype=data.dtype,
        crs=PROJECT_CRS,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data)


def test_correct_tree_positions_weighted_score():
    with TemporaryDirectory() as tmpdir:
        chm_path = Path(tmpdir) / "chm.tif"
        data = np.zeros((5, 5), dtype=np.float32)
        data[1, 1] = 10.0
        data[1, 3] = 12.0
        _write_single_band_raster(chm_path, data)

        trees = gpd.GeoDataFrame(
            {
                "tree_id": [1, 2],
                "geometry": [Point(1.5, 3.5), Point(10.0, 10.0)],
            },
            crs=PROJECT_CRS,
        )

        corrected = correct_tree_positions(
            trees_gdf=trees,
            chm_path=chm_path,
            search_radius_m=3.0,
            height_weight=0.7,
        )

        first_geom = corrected.geometry.iloc[0]
        assert np.isclose(first_geom.x, 3.5)
        assert np.isclose(first_geom.y, 3.5)
        assert corrected["position_corrected"].iloc[0]

        second_geom = corrected.geometry.iloc[1]
        assert np.isclose(second_geom.x, 10.0)
        assert np.isclose(second_geom.y, 10.0)
        assert not corrected["position_corrected"].iloc[1]


def test_correct_tree_positions_null_geometry():
    with TemporaryDirectory() as tmpdir:
        chm_path = Path(tmpdir) / "chm.tif"
        data = np.zeros((3, 3), dtype=np.float32)
        data[1, 1] = 5.0
        _write_single_band_raster(chm_path, data)

        trees = gpd.GeoDataFrame(
            {
                "tree_id": [1, 2],
                "geometry": [Point(1.5, 1.5), None],
            },
            crs=PROJECT_CRS,
        )

        corrected = correct_tree_positions(
            trees_gdf=trees,
            chm_path=chm_path,
            search_radius_m=2.0,
            height_weight=0.7,
        )

        assert len(corrected) == 2
        assert not corrected["position_corrected"].iloc[1]


def test_extract_chm_features_point_sample():
    with TemporaryDirectory() as tmpdir:
        chm_path = Path(tmpdir) / "chm.tif"
        nodata_value = -9999.0
        data = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        data[1, 1] = nodata_value
        _write_single_band_raster(chm_path, data, nodata=nodata_value)

        trees = gpd.GeoDataFrame(
            {
                "tree_id": [1, 2],
                "height_m": [5.0, 6.0],
                "geometry": [Point(0.5, 2.5), Point(1.5, 1.5)],
            },
            crs=PROJECT_CRS,
        )

        extracted = extract_chm_features(trees, chm_path=chm_path)

        assert np.isclose(extracted.loc[0, "CHM_1m"], 1.0)
        assert np.isnan(extracted.loc[1, "CHM_1m"])
        assert np.isclose(extracted.loc[0, "height_m"], 5.0)
        assert np.isclose(extracted.loc[1, "height_m"], 6.0)


def test_extract_chm_empty_geodataframe():
    with TemporaryDirectory() as tmpdir:
        chm_path = Path(tmpdir) / "chm.tif"
        data = np.ones((2, 2), dtype=np.float32)
        _write_single_band_raster(chm_path, data)

        empty_gdf = gpd.GeoDataFrame({"tree_id": [], "geometry": []}, crs=PROJECT_CRS)
        extracted = extract_chm_features(empty_gdf, chm_path=chm_path)

        assert extracted.empty
        assert "CHM_1m" in extracted.columns


def test_extract_sentinel_features_multiband():
    with TemporaryDirectory() as tmpdir:
        sentinel_dir = Path(tmpdir)
        city = "berlin"
        year = 2021
        month = 1
        file_path = sentinel_dir / f"S2_{city}_{year}_{month:02d}_median.tif"

        s2_features = get_all_s2_features()
        band_count = len(s2_features)
        data = np.stack(
            [np.full((2, 2), idx + 1, dtype=np.float32) for idx in range(band_count)],
            axis=0,
        )
        _write_multiband_raster(file_path, data)

        trees = gpd.GeoDataFrame(
            {"tree_id": [1], "geometry": [Point(0.5, 1.5)]},
            crs=PROJECT_CRS,
        )

        extracted = extract_sentinel_features(
            trees_gdf=trees,
            sentinel_dir=sentinel_dir,
            city=city,
            year=year,
            months=[month],
            batch_size=10,
        )

        for idx, feature in enumerate(s2_features):
            col = f"{feature}_{month:02d}"
            assert col in extracted.columns
            assert np.isclose(extracted.loc[0, col], idx + 1)


def test_extract_sentinel_missing_file_warns():
    with TemporaryDirectory() as tmpdir:
        sentinel_dir = Path(tmpdir)
        trees = gpd.GeoDataFrame(
            {"tree_id": [1], "geometry": [Point(0.5, 1.5)]},
            crs=PROJECT_CRS,
        )
        month = 1

        with pytest.warns(UserWarning):
            extracted = extract_sentinel_features(
                trees_gdf=trees,
                sentinel_dir=sentinel_dir,
                city="berlin",
                year=2021,
                months=[month],
                batch_size=10,
            )

        s2_features = get_all_s2_features()
        for feature in s2_features:
            col = f"{feature}_{month:02d}"
            assert col in extracted.columns
            assert extracted[col].isna().all()


def test_extract_all_features_summary():
    with TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        chm_path = tmp_path / "chm.tif"
        sentinel_dir = tmp_path / "sentinel"
        sentinel_dir.mkdir()

        chm_data = np.ones((3, 3), dtype=np.float32)
        _write_single_band_raster(chm_path, chm_data)

        s2_features = get_all_s2_features()
        band_count = len(s2_features)
        s2_data = np.stack(
            [np.full((3, 3), idx + 1, dtype=np.float32) for idx in range(band_count)],
            axis=0,
        )
        city = "berlin"
        year = 2021
        month = 1
        s2_path = sentinel_dir / f"S2_{city}_{year}_{month:02d}_median.tif"
        _write_multiband_raster(s2_path, s2_data)

        trees = gpd.GeoDataFrame(
            {
                "tree_id": [1],
                "city": [city],
                "genus_latin": ["ACER"],
                "height_m": [10.0],
                "geometry": [Point(1.5, 1.5)],
            },
            crs=PROJECT_CRS,
        )

        feature_config = load_feature_config()
        feature_config["temporal"]["extraction_months"] = [month]
        feature_config["tree_correction"]["search_radius_m"] = 2.0
        feature_config["tree_correction"]["height_weight"] = 0.7

        result_gdf, summary = extract_all_features(
            trees_gdf=trees,
            chm_path=chm_path,
            sentinel_dir=sentinel_dir,
            city=city,
            feature_config=feature_config,
        )

        assert "CHM_1m" in result_gdf.columns
        assert f"{s2_features[0]}_{month:02d}" in result_gdf.columns
        assert summary["total_trees"] == 1
        assert summary["trees_corrected"] == 1
