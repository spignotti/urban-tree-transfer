"""Spatial block creation and stratified splitting.

This module handles:
- Regular grid (block) creation
- Spatial assignment of trees to blocks
- Stratified train/val/test splitting with spatial disjointness
"""

from __future__ import annotations

from typing import Any

import geopandas as gpd


def create_spatial_blocks(
    boundary_gdf: gpd.GeoDataFrame,
    block_size_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Create regular grid of spatial blocks covering the boundary.

    Args:
        boundary_gdf: City boundary geometry.
        block_size_m: Block size in meters (default 500m).

    Returns:
        GeoDataFrame with block polygons and block_id column.
    """
    raise NotImplementedError("To be implemented in PRD 002c")


def assign_trees_to_blocks(
    trees_gdf: gpd.GeoDataFrame,
    blocks_gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Assign each tree to its containing spatial block.

    Args:
        trees_gdf: Tree points.
        blocks_gdf: Block polygons from create_spatial_blocks.

    Returns:
        GeoDataFrame with block_id column added.
    """
    raise NotImplementedError("To be implemented in PRD 002c")


def create_stratified_spatial_splits(
    trees_gdf: gpd.GeoDataFrame,
    split_ratios: dict[str, float],
    stratify_column: str = "genus_latin",
    random_seed: int = 42,
) -> dict[str, gpd.GeoDataFrame]:
    """Create spatially disjoint, stratified splits.

    Uses StratifiedGroupKFold with blocks as groups to ensure:
    - No spatial leakage (blocks are atomic units)
    - Proportional genus distribution in each split

    Args:
        trees_gdf: Trees with block_id column.
        split_ratios: Dict of split_name -> ratio (e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15}).
        stratify_column: Column for stratification (default: genus_latin).
        random_seed: Random seed for reproducibility.

    Returns:
        Dict mapping split name to GeoDataFrame.
    """
    raise NotImplementedError("To be implemented in PRD 002c")


def validate_splits(
    splits: dict[str, gpd.GeoDataFrame],
    stratify_column: str = "genus_latin",
) -> dict[str, Any]:
    """Validate split quality: disjointness and stratification.

    Checks:
    - No block overlap between splits
    - KL-divergence of class distribution < 0.01
    - All classes present in all splits

    Args:
        splits: Dict of split_name -> GeoDataFrame.
        stratify_column: Column used for stratification.

    Returns:
        Validation report dict.
    """
    raise NotImplementedError("To be implemented in PRD 002c")
