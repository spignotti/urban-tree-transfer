"""Unit tests for feature selection functions."""

from __future__ import annotations

import geopandas as gpd
import pytest
from shapely.geometry import Point

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.feature_engineering.selection import remove_redundant_features


def test_remove_redundant_features_drops_columns():
    gdf = gpd.GeoDataFrame(
        {
            "tree_id": [1, 2, 3],
            "B8_01": [100, 200, 300],
            "B8A_01": [99, 199, 299],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs=PROJECT_CRS,
    )

    result = remove_redundant_features(gdf, ["B8A_01"])

    assert "B8A_01" not in result.columns
    assert "B8_01" in result.columns
    assert len(result) == 3


def test_remove_redundant_features_missing_columns_warns():
    gdf = gpd.GeoDataFrame(
        {
            "tree_id": [1, 2, 3],
            "B8_01": [100, 200, 300],
            "geometry": [Point(0, 0), Point(1, 1), Point(2, 2)],
        },
        crs=PROJECT_CRS,
    )

    with pytest.warns(UserWarning):
        result = remove_redundant_features(gdf, ["missing_feature"])

    assert result.columns.tolist() == gdf.columns.tolist()


def test_remove_redundant_features_preserves_metadata():
    gdf = gpd.GeoDataFrame(
        {
            "tree_id": ["T1", "T2"],
            "city": ["berlin", "berlin"],
            "B8A_01": [1.0, 2.0],
            "geometry": [Point(0, 0), Point(1, 1)],
        },
        crs=PROJECT_CRS,
    )

    result = remove_redundant_features(gdf, ["B8A_01"])

    assert "tree_id" in result.columns
    assert "city" in result.columns
