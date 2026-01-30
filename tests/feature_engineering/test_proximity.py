"""Unit tests for proximity filtering functions."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.feature_engineering.proximity import (
    analyze_genus_specific_impact,
    apply_proximity_filter,
    compute_nearest_different_genus_distance,
)


def _base_gdf(n: int, genus: str) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(
        {
            "tree_id": [f"T{i:03d}" for i in range(n)],
            "genus_latin": [genus] * n,
            "geometry": [Point(float(i * 10), 0.0) for i in range(n)],
        },
        crs=PROJECT_CRS,
    )


def test_compute_nearest_different_genus_distance_basic():
    tilia = _base_gdf(1, "TILIA")
    acer = gpd.GeoDataFrame(
        {
            "tree_id": ["T999"],
            "genus_latin": ["ACER"],
            "geometry": [Point(15.0, 0.0)],
        },
        crs=PROJECT_CRS,
    )
    gdf = pd.concat([tilia, acer], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=PROJECT_CRS)

    distances = compute_nearest_different_genus_distance(gdf)

    assert len(distances) == 2
    assert np.isclose(distances.iloc[0], 15.0, atol=0.1)
    assert np.isclose(distances.iloc[1], 15.0, atol=0.1)


def test_compute_nearest_different_genus_distance_returns_inf_when_single_genus():
    gdf = _base_gdf(3, "TILIA")

    distances = compute_nearest_different_genus_distance(gdf)

    assert len(distances) == 3
    assert all(np.isinf(distances))


def test_apply_proximity_filter_removes_close_trees():
    tilia = _base_gdf(1, "TILIA")
    acer_close = gpd.GeoDataFrame(
        {"tree_id": ["T998"], "genus_latin": ["ACER"], "geometry": [Point(5.0, 0.0)]},
        crs=PROJECT_CRS,
    )
    acer_far = gpd.GeoDataFrame(
        {"tree_id": ["T999"], "genus_latin": ["ACER"], "geometry": [Point(50.0, 0.0)]},
        crs=PROJECT_CRS,
    )
    gdf = pd.concat([tilia, acer_close, acer_far], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=PROJECT_CRS)

    filtered, stats = apply_proximity_filter(gdf, threshold_m=10.0)

    assert stats["original_count"] == 3
    assert stats["removed_count"] == 2
    assert stats["retained_count"] == 1
    assert np.isclose(stats["retention_rate"], 1 / 3, atol=0.01)
    assert len(filtered) == 1


def test_apply_proximity_filter_keeps_all_if_threshold_low():
    gdf = _base_gdf(3, "TILIA")
    acer = gpd.GeoDataFrame(
        {
            "tree_id": ["T100", "T101"],
            "genus_latin": ["ACER", "ACER"],
            "geometry": [Point(100.0, 0.0), Point(120.0, 0.0)],
        },
        crs=PROJECT_CRS,
    )
    gdf = pd.concat([gdf, acer], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=PROJECT_CRS)

    _filtered, stats = apply_proximity_filter(gdf, threshold_m=0.1)

    assert stats["removed_count"] == 0
    assert stats["retention_rate"] == 1.0


def test_analyze_genus_specific_impact_returns_correct_stats():
    tilia = _base_gdf(2, "TILIA")
    acer = gpd.GeoDataFrame(
        {
            "tree_id": ["T998", "T999"],
            "genus_latin": ["ACER", "ACER"],
            "geometry": [Point(5.0, 0.0), Point(100.0, 0.0)],
        },
        crs=PROJECT_CRS,
    )
    gdf = pd.concat([tilia, acer], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry", crs=PROJECT_CRS)

    impact = analyze_genus_specific_impact(gdf, threshold_m=10.0)

    assert len(impact) == 2
    assert set(impact["genus"]) == {"TILIA", "ACER"}
    assert "total_trees" in impact.columns
    assert "removed_trees" in impact.columns
    assert "removal_rate" in impact.columns
