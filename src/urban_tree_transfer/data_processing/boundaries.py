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
from urban_tree_transfer.utils.strings import normalize_city_name


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


def _build_ogc_filter(property_name: str, value: str, namespace: str | None = None) -> str:
    """Build an OGC XML Filter for property equality.

    Args:
        property_name: Property name to filter on.
        value: Value to match.
        namespace: Optional namespace prefix for the property (e.g., 'ave').

    Returns:
        OGC Filter XML string.
    """
    ns_decl = ""
    prop = property_name
    if namespace:
        # Common namespace URIs for German geodata
        ns_uris = {
            "ave": "http://repository.gdi-de.org/schemas/adv/produkt/alkis-vereinfacht/2.0",
        }
        if namespace in ns_uris:
            ns_decl = f' xmlns:{namespace}="{ns_uris[namespace]}"'
        prop = f"{namespace}:{property_name}"

    return f"""<Filter xmlns="http://www.opengis.net/ogc"{ns_decl}>
<PropertyIsEqualTo>
<PropertyName>{prop}</PropertyName>
<Literal>{value}</Literal>
</PropertyIsEqualTo>
</Filter>"""


def download_city_boundary(city_config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Download city boundary from WFS.

    Supports both CQL_FILTER (for services like Berlin GDI) and OGC XML Filter
    (for services like Saxony GDI that don't support CQL).

    Config options:
        url: WFS base URL
        layer: Layer/typename to request
        filter: CQL filter expression (optional)
        filter_ogc: Dict with {property, value, namespace} for OGC XML filter (optional)
        version: WFS version, defaults to "2.0.0"
    """
    boundary_cfg = city_config.get("boundaries", {})
    url = boundary_cfg.get("url")
    layer = boundary_cfg.get("layer")
    cql_filter = boundary_cfg.get("filter")
    ogc_filter_cfg = boundary_cfg.get("filter_ogc")
    version = boundary_cfg.get("version", "2.0.0")

    if not url or not layer:
        raise ValueError("Boundary config requires url and layer.")

    # Build request parameters based on WFS version
    if version.startswith("1."):
        params: dict[str, str] = {
            "SERVICE": "WFS",
            "VERSION": version,
            "REQUEST": "GetFeature",
            "TYPENAME": layer,  # WFS 1.x uses TYPENAME (singular)
        }
    else:
        params = {
            "SERVICE": "WFS",
            "VERSION": version,
            "REQUEST": "GetFeature",
            "TYPENAMES": layer,  # WFS 2.x uses TYPENAMES (plural)
            "OUTPUTFORMAT": "gml3",
        }

    # Add filter (CQL or OGC XML)
    if cql_filter:
        params["CQL_FILTER"] = cql_filter
    elif ogc_filter_cfg:
        ogc_xml = _build_ogc_filter(
            property_name=ogc_filter_cfg["property"],
            value=ogc_filter_cfg["value"],
            namespace=ogc_filter_cfg.get("namespace"),
        )
        params["FILTER"] = ogc_xml

    response = requests.get(f"{url}?{urlencode(params)}", timeout=60)
    response.raise_for_status()

    # Check for WFS exception in response
    if b"ExceptionReport" in response.content[:500]:
        raise RuntimeError(f"WFS error: {response.text[:500]}")

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
        gdf["city"] = normalize_city_name(str(city_config["name"]))
        gdfs.append(gdf)
    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=PROJECT_CRS)
    return combined
