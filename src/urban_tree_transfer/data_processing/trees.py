"""Tree cadastre download and harmonization."""

from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import pandas as pd
import requests
from requests.exceptions import ConnectionError, ConnectTimeout, ReadTimeout

from urban_tree_transfer.config import MIN_SAMPLES_PER_GENUS, PROJECT_CRS

_ALLOWED_GEOMETRIES = {"Point", "MultiPoint"}
_DEFAULT_BUFFER_M = 500.0

# Cache filename pattern for raw tree data
_CACHE_FILENAME_PATTERN = "{city}_trees_raw.gpkg"


def _fetch_wfs_feature_count(url: str, layer: str, timeout_s: int) -> int | None:
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer,
        "RESULTTYPE": "hits",
    }
    response = requests.get(url, params=params, timeout=timeout_s)
    response.raise_for_status()
    try:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(response.content)
    except ET.ParseError:
        return None
    count = root.attrib.get("numberMatched") or root.attrib.get("numberOfFeatures")
    return int(count) if count and count.isdigit() else None


def _download_wfs_layer(
    url: str,
    layer: str,
    *,
    page_size: int = 5000,
    timeout_s: int = 300,
) -> gpd.GeoDataFrame:
    base_params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer,
    }
    output_formats = [
        "application/json",
        "application/gml+xml; version=3.2",
    ]
    total_features = _fetch_wfs_feature_count(url, layer, timeout_s=timeout_s)
    offset = 0
    gdfs: list[gpd.GeoDataFrame] = []

    while True:
        if total_features is not None and offset >= total_features:
            break
        params = {
            **base_params,
            "COUNT": str(page_size),
            "STARTINDEX": str(offset),
        }
        last_error: Exception | None = None
        page_gdf: gpd.GeoDataFrame | None = None
        for output_format in output_formats:
            try:
                response = requests.get(
                    url,
                    params={**params, "OUTPUTFORMAT": output_format},
                    timeout=timeout_s,
                )
                response.raise_for_status()
                page_gdf = gpd.read_file(BytesIO(response.content))
                break
            except Exception as exc:
                last_error = exc
                page_gdf = None
        if page_gdf is None:
            raise RuntimeError(
                f"WFS layer download failed for {layer} after trying output formats."
            ) from last_error
        if page_gdf.empty:
            break
        gdfs.append(page_gdf)
        offset += len(page_gdf)
        if len(page_gdf) < page_size:
            break

    if not gdfs:
        return gpd.GeoDataFrame()
    if len(gdfs) == 1:
        return gdfs[0]
    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))


def _download_ogc_api_features(url: str, limit: int = 10000) -> gpd.GeoDataFrame:
    all_features: list[dict[str, Any]] = []
    offset = 0
    while True:
        response = requests.get(
            url,
            params={"f": "json", "limit": limit, "offset": offset},
            timeout=300,
        )
        response.raise_for_status()
        payload = response.json()
        features = payload.get("features", [])
        if not features:
            break
        all_features.extend(features)
        offset += limit
        if len(features) < limit:
            break
    payload: dict[str, Any] = {"type": "FeatureCollection", "features": all_features}
    return gpd.GeoDataFrame.from_features(cast(Any, payload))


def _find_cached_trees(city_name: str, cache_dir: Path | None) -> Path | None:
    """Find cached tree data file for a city.

    Args:
        city_name: City name (e.g., "Leipzig")
        cache_dir: Directory to search for cached files.

    Returns:
        Path to cached file if found, None otherwise.
    """
    if cache_dir is None:
        return None

    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None

    # Try common filename patterns
    patterns = [
        _CACHE_FILENAME_PATTERN.format(city=city_name.lower()),
        f"{city_name.lower()}_trees.gpkg",
        f"{city_name}_trees_raw.gpkg",
    ]

    for pattern in patterns:
        cache_path = cache_dir / pattern
        if cache_path.exists():
            return cache_path

    return None


def download_tree_cadastre(
    city_config: dict[str, Any],
    cache_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Download tree cadastre from WFS/OGC API, with fallback to cached file.

    Attempts to download from the configured WFS/OGC API endpoint. If the download
    fails due to connection timeout or network issues, falls back to a cached
    GeoPackage file if available.

    Args:
        city_config: City configuration dictionary with trees section.
        cache_dir: Optional directory containing cached raw tree data.
                   Expected filename: {city}_trees_raw.gpkg

    Returns:
        GeoDataFrame with raw tree data.

    Raises:
        ValueError: If config is invalid.
        RuntimeError: If download fails and no cache is available.
    """
    trees_cfg = city_config.get("trees", {})
    url = trees_cfg.get("url")
    layers = trees_cfg.get("layers")
    source_type = trees_cfg.get("type", "wfs")
    city_name = city_config.get("name", "unknown")

    if not url:
        raise ValueError("Tree cadastre config requires a url.")

    # Try downloading from WFS/API
    try:
        if source_type == "ogc_api_features":
            return _download_ogc_api_features(url)
        if source_type != "wfs":
            raise ValueError(f"Unsupported tree source type: {source_type}")

        if layers is None:
            raise ValueError("WFS tree download requires layers.")

        gdfs = []
        for layer in layers:
            gdf = _download_wfs_layer(url, layer)
            gdf["source_layer"] = layer
            gdfs.append(gdf)
        if len(gdfs) == 1:
            return gdfs[0]
        return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))

    except (ConnectTimeout, ConnectionError, ReadTimeout) as e:
        # Network error - try fallback to cached file
        cache_path = _find_cached_trees(city_name, cache_dir)

        if cache_path is not None:
            warnings.warn(
                f"WFS download failed for {city_name}, using cached file: {cache_path}",
                stacklevel=2,
            )
            return gpd.read_file(cache_path)

        # No cache available - re-raise with helpful message
        raise RuntimeError(
            f"WFS download failed for {city_name} and no cached file found.\n"
            f"Error: {e}\n"
            f"To create a cache file, run locally:\n"
            f"  uv run python scripts/download_leipzig_trees.py\n"
            f"Then upload to: {cache_dir or 'data/trees/raw/'}"
        ) from e


def filter_trees_to_boundary(
    gdf: gpd.GeoDataFrame,
    boundary_gdf: gpd.GeoDataFrame,
    buffer_m: float = _DEFAULT_BUFFER_M,
) -> gpd.GeoDataFrame:
    """Filter trees to those within buffered boundary.

    Args:
        gdf: Tree GeoDataFrame.
        boundary_gdf: Boundary GeoDataFrame.
        buffer_m: Buffer distance in meters to expand boundary (default: 500m).

    Returns:
        GeoDataFrame with trees within buffered boundary.
    """
    if gdf.empty or boundary_gdf.empty:
        return gdf.copy()

    gdf = gdf.copy()
    boundary = boundary_gdf.copy()

    if gdf.crs is None:
        raise ValueError("Tree GeoDataFrame has no CRS defined.")
    if boundary.crs is None:
        raise ValueError("Boundary GeoDataFrame has no CRS defined.")

    if str(gdf.crs) != PROJECT_CRS:
        gdf = gdf.to_crs(PROJECT_CRS)
    if str(boundary.crs) != PROJECT_CRS:
        boundary = boundary.to_crs(PROJECT_CRS)

    clip_geom = boundary.geometry.unary_union
    if buffer_m > 0:
        clip_geom = clip_geom.buffer(buffer_m)

    mask = gdf.geometry.within(clip_geom)
    filtered = gdf[mask].copy()

    n_removed = len(gdf) - len(filtered)
    if n_removed > 0:
        warnings.warn(f"Removed {n_removed} trees outside boundary", stacklevel=2)

    return gpd.GeoDataFrame(filtered, crs=PROJECT_CRS)


def remove_duplicate_trees(
    gdf: gpd.GeoDataFrame,
    by_id: bool = True,
    proximity_m: float | None = None,
) -> gpd.GeoDataFrame:
    """Remove duplicate trees by ID or spatial proximity.

    Args:
        gdf: Tree GeoDataFrame with tree_id column.
        by_id: Remove duplicates based on tree_id column (default: True).
        proximity_m: If set, also remove trees within this distance of each other.

    Returns:
        GeoDataFrame with duplicates removed.
    """
    if gdf.empty:
        return gdf.copy()

    gdf = gdf.copy()
    n_original = len(gdf)

    if by_id and "tree_id" in gdf.columns:
        gdf = gpd.GeoDataFrame(
            gdf.drop_duplicates(subset=["tree_id"], keep="first"),
            geometry="geometry",
            crs=gdf.crs,
        )

    if proximity_m is not None and proximity_m > 0:
        gdf = gdf.copy()
        gdf["_geom_wkt"] = gdf.geometry.apply(
            lambda g: g.buffer(proximity_m).wkt if g is not None else None
        )
        gdf = gpd.GeoDataFrame(
            gdf.drop_duplicates(subset=["_geom_wkt"], keep="first").drop(columns=["_geom_wkt"]),
            geometry="geometry",
            crs=gdf.crs,
        )

    n_removed = n_original - len(gdf)
    if n_removed > 0:
        warnings.warn(f"Removed {n_removed} duplicate trees", stacklevel=2)

    return gpd.GeoDataFrame(gdf, crs=gdf.crs)


def _normalize_genus(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text.upper() if text else None


def _normalize_species(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    parts = text.split()
    if len(parts) > 1 and parts[0][0].isupper():
        return " ".join(parts[1:]).lower()
    return text.lower()


def harmonize_trees(gdf: gpd.GeoDataFrame, city_config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Harmonize to unified schema with consistent columns and dtypes.

    Ensures identical schema for all cities:
    - Same columns in same order
    - Same dtypes
    - Consistent null handling (None for missing values)

    Args:
        gdf: Raw tree GeoDataFrame.
        city_config: City configuration with tree mapping.

    Returns:
        Harmonized GeoDataFrame with unified schema.
    """
    mapping = city_config.get("trees", {}).get("mapping")
    if not isinstance(mapping, dict):
        raise ValueError("Tree mapping is required for harmonization.")
    if any(value is not None and not isinstance(value, str) for value in mapping.values()):
        raise ValueError("Tree mapping values must be strings or null.")
    mapping = cast(dict[str, str | None], mapping)

    def _get_column(key: str) -> pd.Series:
        column = mapping.get(key)
        if not column or column not in gdf.columns:
            return pd.Series([None] * len(gdf))
        return cast(pd.Series, gdf[column])

    df = normalize_tree_geometries(gdf.copy())

    df["tree_id"] = _get_column("tree_id").astype(str)
    df["city"] = city_config["name"]
    df["genus_latin"] = _get_column("genus_latin").apply(_normalize_genus)
    df["species_latin"] = _get_column("species_latin").apply(_normalize_species)

    genus_german = _get_column("genus_german")
    species_german = _get_column("species_german")
    df["genus_german"] = genus_german.where(genus_german.notna(), None)
    df["species_german"] = species_german.where(species_german.notna(), None)

    plant_year = cast(pd.Series, pd.to_numeric(_get_column("plant_year"), errors="coerce"))
    df["plant_year"] = plant_year.astype("Int64")

    height_m = cast(pd.Series, pd.to_numeric(_get_column("height_m"), errors="coerce"))
    df["height_m"] = height_m.astype("Float64")

    tree_type = _get_column("tree_type")
    df["tree_type"] = tree_type.where(tree_type.notna(), None)

    if df.crs is None:
        raise ValueError("Tree cadastre is missing CRS information.")
    if str(df.crs) != PROJECT_CRS:
        df = df.to_crs(PROJECT_CRS)

    columns = [
        "tree_id",
        "city",
        "genus_latin",
        "species_latin",
        "genus_german",
        "species_german",
        "plant_year",
        "height_m",
        "tree_type",
        "geometry",
    ]
    return gpd.GeoDataFrame(df[columns], crs=PROJECT_CRS)


def normalize_tree_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Ensure point geometries and normalize MultiPoint to Point."""
    if "geometry" not in gdf.columns:
        raise ValueError("Tree cadastre is missing geometry column.")

    geom_types = set(gdf.geometry.type.unique())
    if not geom_types.issubset(_ALLOWED_GEOMETRIES):
        raise ValueError(f"Unsupported geometry types: {sorted(geom_types)}")

    if "MultiPoint" not in geom_types:
        return gdf

    gdf = gdf.copy()
    gdf["geometry"] = gdf.geometry.apply(
        lambda geom: geom.geoms[0] if geom is not None and geom.geom_type == "MultiPoint" else geom
    )
    return gdf


def summarize_tree_cadastre(gdf: gpd.GeoDataFrame, city_name: str) -> dict[str, Any]:
    """Collect basic metadata for a tree cadastre."""
    geometry_types: list[str] = []
    if "geometry" in gdf.columns:
        geometry_types = sorted(gdf.geometry.type.unique().tolist())

    return {
        "city": city_name,
        "records": len(gdf),
        "crs": str(gdf.crs) if gdf.crs else None,
        "geometry_types": geometry_types,
    }


def filter_viable_genera(
    gdf: gpd.GeoDataFrame, min_samples: int = MIN_SAMPLES_PER_GENUS
) -> gpd.GeoDataFrame:
    """Filter to genera with sufficient samples in all cities."""
    counts = gdf.dropna(subset=["genus_latin"]).groupby(["city", "genus_latin"]).size()
    counts = counts.unstack(fill_value=0)
    if counts.empty:
        return gpd.GeoDataFrame(gdf.iloc[0:0].copy(), crs=gdf.crs)
    viable = (counts >= min_samples).all(axis=1)
    viable_index = cast(pd.Index, counts.index[viable])
    viable_genera = [str(item) for item in viable_index.tolist()]
    filtered = gdf[gdf["genus_latin"].isin(viable_genera)].copy()
    return gpd.GeoDataFrame(filtered, crs=gdf.crs)
