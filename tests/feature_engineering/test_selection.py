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
