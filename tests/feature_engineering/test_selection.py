"""Unit tests for feature selection functions."""

from __future__ import annotations

import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.config.loader import get_metadata_columns, load_feature_config
from urban_tree_transfer.feature_engineering.selection import (
    compute_feature_correlations,
    compute_vif,
    export_geometry_lookup,
    export_splits_to_parquet,
    identify_redundant_features,
    remove_redundant_features,
    validate_final_preparation_output,
)


def _make_gdf() -> gpd.GeoDataFrame:
    feature_config = load_feature_config()
    metadata_cols = get_metadata_columns(feature_config)
    base_data = {
        "feature_a": [1.0, 2.0, 3.0, 4.0],
        "feature_b": [2.0, 4.0, 6.0, 8.0],
        "feature_c": [1.0, 1.1, 0.9, 1.2],
        "genus_latin": ["TILIA", "TILIA", "ACER", "ACER"],
        "block_id": ["b0", "b0", "b1", "b1"],
        "geometry": [Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3)],
    }
    for col in metadata_cols:
        if col not in base_data:
            if col == "city":
                base_data[col] = ["berlin"] * 4
            elif col == "position_corrected":
                base_data[col] = [False] * 4
            elif col == "correction_distance":
                base_data[col] = [0.0] * 4
            else:
                base_data[col] = [pd.NA] * 4
    return gpd.GeoDataFrame(base_data, crs=PROJECT_CRS)


def test_compute_feature_correlations_returns_matrix() -> None:
    gdf = _make_gdf()
    groups = {"spectral": ["feature_a", "feature_b", "feature_c"]}

    result = compute_feature_correlations(gdf, groups)

    assert "spectral" in result
    assert result["spectral"].shape == (3, 3)
    assert np.isclose(result["spectral"].loc["feature_a", "feature_b"], 1.0)


def test_compute_feature_correlations_missing_columns() -> None:
    gdf = _make_gdf()
    groups = {"spectral": ["feature_a", "missing_col"]}

    with pytest.raises(ValueError, match="Missing columns"):
        compute_feature_correlations(gdf, groups)


def test_identify_redundant_features_threshold() -> None:
    gdf = _make_gdf()
    groups = {"spectral": ["feature_a", "feature_b", "feature_c"]}
    corr = compute_feature_correlations(gdf, groups)

    redundant = identify_redundant_features(corr, threshold=0.9)

    assert any(item["feature_to_remove"] in {"feature_a", "feature_b"} for item in redundant)


def test_identify_redundant_features_with_importance() -> None:
    gdf = _make_gdf()
    groups = {"spectral": ["feature_a", "feature_b"]}
    corr = compute_feature_correlations(gdf, groups)

    redundant = identify_redundant_features(
        corr, threshold=0.9, feature_importance={"feature_a": 0.9, "feature_b": 0.1}
    )

    assert redundant[0]["feature_to_remove"] == "feature_b"


def test_remove_redundant_features_drops_columns() -> None:
    gdf = _make_gdf()

    result = remove_redundant_features(gdf, ["feature_c"])

    assert "feature_c" not in result.columns


def test_remove_redundant_features_skips_geometry() -> None:
    gdf = _make_gdf()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Skipping 'geometry' in features_to_remove.")
        result = remove_redundant_features(gdf, ["geometry"])

    assert "geometry" in result.columns


def test_compute_vif_single_feature() -> None:
    gdf = _make_gdf()
    result = compute_vif(gdf, ["feature_a"])

    assert np.isclose(result.loc[0, "vif"], 1.0)


def test_compute_vif_detects_collinearity() -> None:
    gdf = _make_gdf()
    result = compute_vif(gdf, ["feature_a", "feature_b"])

    assert result["vif"].max() > 10.0


def test_validate_final_preparation_output_schema() -> None:
    gdf = _make_gdf()
    gdf["outlier_zscore"] = False
    gdf["outlier_mahalanobis"] = False
    gdf["outlier_iqr"] = False
    gdf["outlier_severity"] = "none"
    gdf["outlier_method_count"] = 0

    result = validate_final_preparation_output(gdf, ["feature_a", "feature_b", "feature_c"])

    assert len(result) == len(gdf)


def test_export_splits_to_parquet_drops_geometry(tmp_path) -> None:
    """Parquet output must not contain geometry column."""
    gdf = _make_gdf()
    splits = {"test_split": gdf}

    parquet_paths = export_splits_to_parquet(splits, tmp_path)

    df = pd.read_parquet(parquet_paths["test_split"])
    assert "geometry" not in df.columns


def test_export_splits_to_parquet_drops_metadata(tmp_path) -> None:
    """Parquet output must drop only redundant columns, keep analysis metadata."""
    gdf = _make_gdf()
    gdf["species_latin"] = ["species1"] * len(gdf)
    gdf["plant_year"] = [2020] * len(gdf)
    gdf["height_m"] = [10.0] * len(gdf)
    splits = {"test_split": gdf}

    parquet_paths = export_splits_to_parquet(splits, tmp_path)

    df = pd.read_parquet(parquet_paths["test_split"])
    # Only redundant columns should be dropped
    assert "height_m" not in df.columns
    assert "geometry" not in df.columns
    # Analysis metadata should be kept
    assert "species_latin" in df.columns
    assert "plant_year" in df.columns  # For error metrics analysis


def test_export_splits_to_parquet_preserves_features(tmp_path) -> None:
    """All ML feature values must be identical between GeoPackage and Parquet."""
    gdf = _make_gdf()
    splits = {"test_split": gdf}

    parquet_paths = export_splits_to_parquet(splits, tmp_path)

    df = pd.read_parquet(parquet_paths["test_split"])
    for col in ["feature_a", "feature_b", "feature_c"]:
        pd.testing.assert_series_equal(
            df[col].reset_index(drop=True),
            gdf[col].reset_index(drop=True),
            check_names=False,
        )


def test_export_splits_to_parquet_preserves_row_count(tmp_path) -> None:
    """Row count must match between GeoPackage and Parquet."""
    gdf = _make_gdf()
    splits = {"berlin_train": gdf, "berlin_val": gdf.iloc[:2]}

    parquet_paths = export_splits_to_parquet(splits, tmp_path)

    assert len(pd.read_parquet(parquet_paths["berlin_train"])) == len(gdf)
    assert len(pd.read_parquet(parquet_paths["berlin_val"])) == 2


def test_export_geometry_lookup_schema(tmp_path) -> None:
    """Geometry lookup must have expected columns and no duplicates per split."""
    gdf = _make_gdf()
    gdf["tree_id"] = ["T001", "T002", "T003", "T004"]
    splits = {"berlin_train": gdf, "berlin_train_filtered": gdf.iloc[:2]}

    output_path = tmp_path / "geometry_lookup.parquet"
    n_trees = export_geometry_lookup(splits, output_path)

    lookup = pd.read_parquet(output_path)
    assert set(lookup.columns) == {"tree_id", "city", "split", "filtered", "x", "y"}
    assert n_trees == 4  # All unique trees
    assert len(lookup) == 6  # Trees appear once per split (4 + 2 = 6 rows)


def test_export_geometry_lookup_coordinates(tmp_path) -> None:
    """X/Y coordinates must match original geometry."""
    gdf = gpd.GeoDataFrame(
        {
            "tree_id": ["T001", "T002"],
            "city": ["berlin", "berlin"],
            "geometry": [Point(100.5, 200.7), Point(300.1, 400.9)],
        },
        crs=PROJECT_CRS,
    )
    splits = {"berlin_train": gdf}

    output_path = tmp_path / "geometry_lookup.parquet"
    export_geometry_lookup(splits, output_path)

    lookup = pd.read_parquet(output_path)
    assert lookup.loc[lookup["tree_id"] == "T001", "x"].values[0] == 100.5
    assert lookup.loc[lookup["tree_id"] == "T001", "y"].values[0] == 200.7
    assert lookup.loc[lookup["tree_id"] == "T002", "x"].values[0] == 300.1
    assert lookup.loc[lookup["tree_id"] == "T002", "y"].values[0] == 400.9
