"""Unit tests for spatial splits."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.feature_engineering.splits import (
    create_spatial_blocks,
    create_stratified_splits_berlin,
    create_stratified_splits_leipzig,
    validate_split_stratification,
)


def _make_blocked_gdf(city: str, block_ids: list[str]) -> gpd.GeoDataFrame:
    records = []
    for idx, block_id in enumerate(block_ids):
        for i in range(4):
            genus = "TILIA" if i % 2 == 0 else "ACER"
            records.append(
                {
                    "tree_id": f"{city}_{block_id}_{i}",
                    "city": city,
                    "genus_latin": genus,
                    "block_id": block_id,
                    "geometry": Point(float(idx * 10 + i), float(idx * 10 + i)),
                }
            )
    return gpd.GeoDataFrame(records, crs=PROJECT_CRS)


def test_create_spatial_blocks_assigns_block_ids():
    gdf = gpd.GeoDataFrame(
        {
            "tree_id": ["A1", "A2", "B1", "B2"],
            "city": ["berlin", "berlin", "leipzig", "leipzig"],
            "genus_latin": ["TILIA"] * 4,
            "geometry": [Point(0, 0), Point(5, 5), Point(100, 100), Point(110, 110)],
        },
        crs=PROJECT_CRS,
    )

    result = create_spatial_blocks(gdf, block_size_m=20.0)

    assert "block_id" in result.columns
    assert result["block_id"].isna().sum() == 0
    assert result["block_id"].str.startswith(("berlin_", "leipzig_")).all()


def test_create_stratified_splits_berlin_disjoint():
    gdf = _make_blocked_gdf("berlin", ["b0", "b1", "b2", "b3"])

    train_gdf, val_gdf, test_gdf = create_stratified_splits_berlin(
        gdf, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25, random_seed=42
    )

    validation = validate_split_stratification(
        train_gdf, val_gdf, test_gdf, split_names=["train", "val", "test"]
    )

    assert validation["spatial_overlap"] == 0
    assert max(validation["kl_divergences"].values()) < 0.01


def test_create_stratified_splits_leipzig_disjoint():
    gdf = _make_blocked_gdf("leipzig", ["l0", "l1", "l2", "l3", "l4"])

    finetune_gdf, test_gdf = create_stratified_splits_leipzig(
        gdf, finetune_ratio=0.8, test_ratio=0.2, random_seed=42
    )

    validation = validate_split_stratification(
        finetune_gdf, test_gdf, split_names=["finetune", "test"]
    )

    assert validation["spatial_overlap"] == 0
    assert validation["kl_divergences"]["finetune_vs_test"] < 0.01
