"""Shared geographic utilities for data processing."""

from __future__ import annotations

import warnings

import geopandas as gpd
from shapely.validation import make_valid

from urban_tree_transfer.config import PROJECT_CRS


def ensure_project_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate and reproject GeoDataFrame to project CRS (EPSG:25833).

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame in project CRS.

    Raises:
        ValueError: If GeoDataFrame has no CRS defined.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")

    if str(gdf.crs) == PROJECT_CRS:
        return gdf

    warnings.warn(f"Reprojecting from {gdf.crs} to {PROJECT_CRS}", stacklevel=2)
    return gdf.to_crs(PROJECT_CRS)


def buffer_boundaries(gdf: gpd.GeoDataFrame, buffer_m: float = 500.0) -> gpd.GeoDataFrame:
    """Buffer boundary geometries by specified distance.

    Args:
        gdf: Input GeoDataFrame with polygon geometries.
        buffer_m: Buffer distance in meters (default: 500m).

    Returns:
        GeoDataFrame with buffered geometries.
    """
    if gdf.empty:
        return gdf.copy()

    gdf = ensure_project_crs(gdf.copy())
    gdf["geometry"] = gdf.geometry.buffer(buffer_m)
    return gdf


def validate_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Fix invalid geometries using shapely make_valid.

    Args:
        gdf: Input GeoDataFrame.

    Returns:
        GeoDataFrame with valid geometries.
    """
    if gdf.empty:
        return gdf.copy()

    gdf = gdf.copy()
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        warnings.warn(f"Fixing {n_invalid} invalid geometries", stacklevel=2)
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)
    return gdf


def clip_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary: gpd.GeoDataFrame,
    buffer_m: float = 0.0,
) -> gpd.GeoDataFrame:
    """Clip GeoDataFrame to boundary with optional buffer.

    Args:
        gdf: Input GeoDataFrame to clip.
        boundary: Boundary GeoDataFrame.
        buffer_m: Buffer distance in meters to expand boundary before clipping.

    Returns:
        Clipped GeoDataFrame.
    """
    if gdf.empty or boundary.empty:
        return gdf.copy()

    gdf = ensure_project_crs(gdf.copy())
    boundary = ensure_project_crs(boundary.copy())

    clip_geom = boundary.geometry.unary_union
    if buffer_m > 0:
        clip_geom = clip_geom.buffer(buffer_m)

    clip_mask = gpd.GeoSeries([clip_geom], crs=PROJECT_CRS)
    return gpd.clip(gdf, clip_mask)
