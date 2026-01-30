"""Mixed-genus proximity analysis and filtering.

This module handles:
- Computing nearest different-genus distances
- Applying proximity threshold filters
- Genus-specific impact analysis
"""

from __future__ import annotations

from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from urban_tree_transfer.config import PROJECT_CRS


def compute_nearest_different_genus_distance(
    trees_gdf: gpd.GeoDataFrame,
    genus_column: str = "genus_latin",
) -> pd.Series:
    """Compute nearest distance to a tree of a different genus.

    For each tree, finds the minimum distance to any tree with a different genus.
    This identifies trees at risk of Sentinel-2 pixel mixing.

    Args:
        trees_gdf: Input trees with geometry and genus information.
        genus_column: Column name containing genus labels.

    Returns:
        Series of distances (meters) with same index as input.
        Values are np.inf if no different-genus trees exist for that genus.

    Raises:
        ValueError: If GeoDataFrame has wrong CRS or missing columns.
    """
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")
    if genus_column not in trees_gdf.columns:
        raise ValueError(f"Missing required column: {genus_column}")
    if trees_gdf.empty:
        raise ValueError("trees_gdf must not be empty.")

    distances = pd.Series(index=trees_gdf.index, dtype="float64")
    genera = sorted(trees_gdf[genus_column].dropna().unique())

    for genus in genera:
        same_genus = trees_gdf[trees_gdf[genus_column] == genus]
        diff_genus = trees_gdf[trees_gdf[genus_column] != genus]

        if diff_genus.empty:
            distances.loc[same_genus.index] = np.inf
            continue

        diff_geom = diff_genus.geometry
        distances.loc[same_genus.index] = same_genus.geometry.apply(
            lambda geom, diff_geom=diff_geom: diff_geom.distance(geom).min()
        )

    return distances


def apply_proximity_filter(
    trees_gdf: gpd.GeoDataFrame,
    threshold_m: float,
    genus_column: str = "genus_latin",
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Filter trees based on nearest different-genus distance threshold.

    Removes trees closer than threshold_m to a different genus to reduce
    Sentinel-2 pixel mixing contamination.

    Args:
        trees_gdf: Input trees with geometry and genus information.
        threshold_m: Minimum distance (meters) to different genus.
                     Trees closer than this are removed.
        genus_column: Column name containing genus labels.

    Returns:
        Tuple of (filtered_gdf, stats_dict):
        - filtered_gdf: Trees with nearest_diff_genus_m >= threshold_m
        - stats_dict: Dictionary with filtering statistics:
            - original_count: Input tree count
            - removed_count: Trees filtered out
            - retained_count: Trees kept
            - retention_rate: Fraction retained (0.0 to 1.0)
            - threshold_used: Applied threshold

    Raises:
        ValueError: If threshold_m <= 0 or validation fails.
    """
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")
    if threshold_m <= 0:
        raise ValueError("threshold_m must be > 0.")
    if trees_gdf.empty:
        raise ValueError("trees_gdf must not be empty.")

    distances = compute_nearest_different_genus_distance(trees_gdf, genus_column)

    gdf_copy = trees_gdf.copy()
    gdf_copy["_nearest_diff_genus_m"] = distances

    filtered_gdf = gdf_copy[gdf_copy["_nearest_diff_genus_m"] >= threshold_m].copy()
    filtered_gdf = filtered_gdf.drop(columns=["_nearest_diff_genus_m"])

    original_count = len(trees_gdf)
    retained_count = len(filtered_gdf)
    removed_count = original_count - retained_count
    retention_rate = retained_count / original_count if original_count > 0 else 0.0

    stats = {
        "original_count": int(original_count),
        "removed_count": int(removed_count),
        "retained_count": int(retained_count),
        "retention_rate": float(retention_rate),
        "threshold_used": float(threshold_m),
    }

    return cast(gpd.GeoDataFrame, filtered_gdf), stats


def analyze_genus_specific_impact(
    trees_gdf: gpd.GeoDataFrame,
    threshold_m: float,
    genus_column: str = "genus_latin",
) -> pd.DataFrame:
    """Analyze filtering impact per genus (for validation/reporting).

    Args:
        trees_gdf: Input trees with geometry and genus information.
        threshold_m: Distance threshold (meters).
        genus_column: Column name containing genus labels.

    Returns:
        DataFrame with columns:
        - genus: Genus name
        - total_trees: Count before filtering
        - removed_trees: Count removed
        - removal_rate: Fraction removed (0.0 to 1.0)

    Raises:
        ValueError: If validation fails.
    """
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")
    if threshold_m <= 0:
        raise ValueError("threshold_m must be > 0.")
    if genus_column not in trees_gdf.columns:
        raise ValueError(f"Missing required column: {genus_column}")

    distances = compute_nearest_different_genus_distance(trees_gdf, genus_column)

    gdf_copy = trees_gdf.copy()
    gdf_copy["_removed"] = distances < threshold_m

    genus_stats = (
        gdf_copy.groupby(genus_column)["_removed"]
        .agg(total_trees="count", removed_trees="sum")
        .reset_index()
    )
    genus_stats = genus_stats.rename(columns={genus_column: "genus"})
    genus_stats["removal_rate"] = genus_stats["removed_trees"] / genus_stats["total_trees"]

    return genus_stats
