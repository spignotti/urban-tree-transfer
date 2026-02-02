"""Integration tests for Phase 2 pipeline schema and metadata preservation."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.config.loader import (
    get_all_feature_names,
    get_all_s2_features,
    get_metadata_columns,
    load_feature_config,
)
from urban_tree_transfer.feature_engineering.extraction import extract_all_features
from urban_tree_transfer.feature_engineering.quality import (
    apply_temporal_selection,
    run_quality_pipeline,
)
from urban_tree_transfer.feature_engineering.selection import validate_final_preparation_output
from urban_tree_transfer.utils.final_validation import validate_zero_nan
from urban_tree_transfer.utils.json_validation import validate_temporal_selection
from urban_tree_transfer.utils.schema_validation import validate_phase2c_output


def _make_phase1_gdf(n: int = 3) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "tree_id": [f"T{i:03d}" for i in range(n)],
            "city": ["berlin"] * n,
            "genus_latin": ["TILIA"] * n,
            "species_latin": [pd.NA] * n,
            "genus_german": [pd.NA] * n,
            "species_german": [pd.NA] * n,
            "plant_year": pd.Series([2010] * n, dtype="Int64"),
            "height_m": np.linspace(10.0, 12.0, n),
            "tree_type": [pd.NA] * n,
            "geometry": [Point(0.5 + i, 9.5) for i in range(n)],
        },
        crs=PROJECT_CRS,
    )


def _write_raster(path: Path, data: np.ndarray, descriptions: list[str] | None = None) -> None:
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
    ) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            dst.write(data)
        if descriptions is not None:
            dst.descriptions = tuple(descriptions)


def test_phase2a_to_phase2b_schema(tmp_path: Path) -> None:
    phase1_gdf = _make_phase1_gdf(2)
    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, np.array([[5.0, 2.0], [1.0, 2.0]], dtype=np.float32))

    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()
    s2_features = get_all_s2_features()
    data = np.stack(
        [np.full((2, 2), i + 1, dtype=np.float32) for i in range(len(s2_features))], axis=0
    )
    for month in [1, 2]:
        _write_raster(
            sentinel_dir / f"S2_berlin_2021_{month:02d}_median.tif",
            data,
            descriptions=s2_features,
        )

    feature_config = load_feature_config()
    feature_config["temporal"]["extraction_months"] = [1, 2]

    phase2a_gdf, _ = extract_all_features(
        phase1_gdf,
        chm_path=chm_path,
        sentinel_dir=sentinel_dir,
        city="berlin",
        feature_config=feature_config,
    )

    phase2b_gdf = run_quality_pipeline(
        phase2a_gdf,
        selected_months=[1, 2],
        feature_config=feature_config,
        max_nan_months=2,
        max_edge_nan_months=1,
        ndvi_min_threshold=0.0,
    )

    assert "CHM_1m_percentile" in phase2b_gdf.columns


def test_exploratory_json_consumption(tmp_path: Path) -> None:
    payload = {"selected_months": [4, 5], "selection_method": "jm_distance"}
    json_path = tmp_path / "temporal_selection.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    config = validate_temporal_selection(json_path)
    assert config["selected_months"] == [4, 5]

    feature_config = load_feature_config()
    s2_features = get_all_s2_features(feature_config)
    gdf = _make_phase1_gdf(1)
    for col in get_metadata_columns(feature_config):
        if col not in gdf.columns:
            if col == "position_corrected":
                gdf[col] = False
            elif col == "correction_distance":
                gdf[col] = 0.0
            else:
                gdf[col] = pd.NA
    gdf["CHM_1m"] = 5.0
    for month in range(1, 13):
        for feature in s2_features:
            gdf[f"{feature}_{month:02d}"] = 0.5

    selected_months = config["selected_months"]
    result = apply_temporal_selection(gdf, selected_months, feature_config)

    assert f"{s2_features[0]}_04" in result.columns
    assert f"{s2_features[0]}_06" not in result.columns


def test_phase2c_output_schema_and_zero_nan() -> None:
    feature_config = load_feature_config()
    metadata_cols = get_metadata_columns(feature_config)
    selected_months = [4, 5, 6, 7, 8, 9, 10, 11]
    feature_columns = get_all_feature_names(
        months=selected_months, include_chm_engineered=True, config=feature_config
    )

    gdf = _make_phase1_gdf(2)
    for col in metadata_cols:
        if col not in gdf.columns:
            gdf[col] = pd.NA
    for col in feature_columns:
        gdf[col] = 1.0

    gdf["outlier_zscore"] = False
    gdf["outlier_mahalanobis"] = False
    gdf["outlier_iqr"] = False
    gdf["outlier_severity"] = "none"
    gdf["outlier_method_count"] = 0
    gdf["block_id"] = "b0"

    validate_phase2c_output(gdf, feature_columns, feature_config)
    validate_zero_nan(gdf, feature_columns, dataset_name="phase2c")
    validate_final_preparation_output(gdf, feature_columns)

    assert set(metadata_cols).issubset(gdf.columns)


def test_final_geopackage_schema_validation() -> None:
    feature_config = load_feature_config()
    metadata_cols = get_metadata_columns(feature_config)
    selected_months = [4, 5, 6, 7, 8, 9, 10, 11]
    feature_columns = get_all_feature_names(
        months=selected_months, include_chm_engineered=True, config=feature_config
    )

    gdf = _make_phase1_gdf(1)
    for col in metadata_cols:
        if col not in gdf.columns:
            gdf[col] = pd.NA
    for col in feature_columns:
        gdf[col] = 1.0

    gdf["outlier_zscore"] = False
    gdf["outlier_mahalanobis"] = False
    gdf["outlier_iqr"] = False
    gdf["outlier_severity"] = "none"
    gdf["outlier_method_count"] = 0
    gdf["block_id"] = "b0"

    outputs = [
        "berlin_train.gpkg",
        "berlin_val.gpkg",
        "berlin_test.gpkg",
        "leipzig_finetune.gpkg",
        "leipzig_test.gpkg",
        "berlin_train_filtered.gpkg",
        "berlin_val_filtered.gpkg",
        "berlin_test_filtered.gpkg",
        "leipzig_finetune_filtered.gpkg",
        "leipzig_test_filtered.gpkg",
    ]
    assert len(outputs) == 10

    for name in outputs:
        validate_phase2c_output(gdf, feature_columns, feature_config)
        validate_zero_nan(gdf, feature_columns, dataset_name=name)


def test_metadata_preservation_through_pipeline(tmp_path: Path) -> None:
    phase1_gdf = _make_phase1_gdf(2)
    metadata_cols = [
        col for col in get_metadata_columns(load_feature_config()) if col in phase1_gdf.columns
    ]

    chm_path = tmp_path / "chm.tif"
    _write_raster(chm_path, np.array([[5.0, 2.0], [1.0, 2.0]], dtype=np.float32))

    sentinel_dir = tmp_path / "s2"
    sentinel_dir.mkdir()
    s2_features = get_all_s2_features()
    data = np.stack(
        [np.full((2, 2), i + 1, dtype=np.float32) for i in range(len(s2_features))], axis=0
    )
    _write_raster(
        sentinel_dir / "S2_berlin_2021_01_median.tif",
        data,
        descriptions=s2_features,
    )

    feature_config = load_feature_config()
    feature_config["temporal"]["extraction_months"] = [1]

    phase2a_gdf, _ = extract_all_features(
        phase1_gdf,
        chm_path=chm_path,
        sentinel_dir=sentinel_dir,
        city="berlin",
        feature_config=feature_config,
    )

    phase2b_gdf = run_quality_pipeline(
        phase2a_gdf,
        selected_months=[1],
        feature_config=feature_config,
        max_nan_months=2,
        max_edge_nan_months=1,
        ndvi_min_threshold=0.0,
    )

    phase2b_gdf["outlier_zscore"] = False
    phase2b_gdf["outlier_mahalanobis"] = False
    phase2b_gdf["outlier_iqr"] = False
    phase2b_gdf["outlier_severity"] = "none"
    phase2b_gdf["outlier_method_count"] = 0
    phase2b_gdf["block_id"] = "b0"

    for col in metadata_cols:
        assert phase2b_gdf[col].equals(phase1_gdf[col])


def test_feature_count_progression() -> None:
    feature_config = load_feature_config()
    metadata_cols = get_metadata_columns(feature_config)
    all_months = list(range(1, 13))

    phase2a_features = get_all_feature_names(months=all_months, config=feature_config)
    assert len(metadata_cols) + len(phase2a_features) == 288

    selected_months = [4, 5, 6, 7, 8, 9, 10, 11]
    phase2b_features = get_all_feature_names(
        months=selected_months, include_chm_engineered=True, config=feature_config
    )
    assert len(metadata_cols) + len(phase2b_features) == 198

    final_feature_count = 170 + 3
    total_phase2c = len(metadata_cols) + final_feature_count + 5 + 1
    assert total_phase2c == 190
