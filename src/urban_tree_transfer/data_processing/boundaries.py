"""City boundary download and preparation."""

from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import geopandas as gpd
import pandas as pd
import requests
from shapely.validation import make_valid

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.config.loader import load_city_config


def validate_polygon_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate and fix polygon geometries using shapely make_valid.

    Args:
        gdf: Input GeoDataFrame with polygon geometries.

    Returns:
        GeoDataFrame with valid geometries.
    """
    if gdf.empty:
        return gdf.copy()

    gdf = gdf.copy()
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        warnings.warn(f"Fixing {n_invalid} invalid polygon geometries", stacklevel=2)
        gdf.loc[invalid_mask, "geometry"] = gdf.loc[invalid_mask, "geometry"].apply(make_valid)
    return gdf


def _largest_polygon(geom):
    if geom is None:
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda poly: poly.area)
    return geom


def download_city_boundary(city_config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Download city boundary from WFS."""
    boundary_cfg = city_config.get("boundaries", {})
    url = boundary_cfg.get("url")
    layer = boundary_cfg.get("layer")
    filter_expr = boundary_cfg.get("filter")
    if not url or not layer:
        raise ValueError("Boundary config requires url and layer.")

    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer,
        "OUTPUTFORMAT": "gml3",
    }
    if filter_expr:
        params["CQL_FILTER"] = filter_expr

    response = requests.get(f"{url}?{urlencode(params)}", timeout=60)
    response.raise_for_status()

    return gpd.read_file(BytesIO(response.content))


def clean_boundaries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Keep largest polygon per feature, validate geometries, and reproject to project CRS."""
    gdf_clean = gdf.copy()
    gdf_clean["geometry"] = gdf_clean.geometry.apply(_largest_polygon)
    gdf_clean = validate_polygon_geometries(gdf_clean)
    gdf_clean = gdf_clean.to_crs(PROJECT_CRS)
    return gdf_clean


def load_boundaries(cities: list[str], config_dir: Path) -> gpd.GeoDataFrame:
    """Load boundaries for multiple cities."""
    gdfs: list[gpd.GeoDataFrame] = []
    for city in cities:
        city_config = load_city_config(city, config_dir=config_dir)
        gdf = download_city_boundary(city_config)
        gdf = clean_boundaries(gdf)
        gdf["city"] = city_config["name"]
        gdfs.append(gdf)
    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=PROJECT_CRS)
    return combined
