"""Integration tests for external data endpoints.

These tests hit real external APIs to verify configurations are correct.
Run with: uv run nox -s test_integration

Each test fetches minimal data (1 feature or HEAD request) to verify:
- URL is reachable
- Layer/resource exists
- Filter syntax is correct
- Response schema matches expectations
"""

from __future__ import annotations

from io import BytesIO

import geopandas as gpd
import requests

from urban_tree_transfer.config import get_config_dir
from urban_tree_transfer.data_processing.boundaries import download_city_boundary
from urban_tree_transfer.data_processing.elevation import (
    _get_dataset_feed_url,
    _load_urls_from_file,
    _parse_atom_feed,
)


def _fetch_wfs_feature(url: str, layer: str, max_features: int = 1) -> gpd.GeoDataFrame:
    params = {
        "SERVICE": "WFS",
        "VERSION": "2.0.0",
        "REQUEST": "GetFeature",
        "TYPENAMES": layer,
        "OUTPUTFORMAT": "application/gml+xml; version=3.2",
        "MAXFEATURES": str(max_features),
        "COUNT": str(max_features),
    }
    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    return gpd.read_file(BytesIO(response.content))


def _assert_url_accessible(url: str) -> None:
    """Check URL is accessible. Uses GET with stream=True to avoid downloading full file."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; urban-tree-transfer/1.0)"}
    # Try HEAD first, fall back to streaming GET if HEAD fails (some servers don't support HEAD)
    response = requests.head(url, allow_redirects=True, timeout=60, headers=headers)
    if response.status_code == 401 or response.status_code == 405:
        # WebDAV servers may not support HEAD, try GET with stream
        response = requests.get(url, allow_redirects=True, timeout=60, headers=headers, stream=True)
        response.close()  # Don't download the body
    response.raise_for_status()
    assert response.status_code == 200


def test_berlin_boundary_wfs(berlin_config):
    """Verify Berlin boundary WFS returns 1 valid polygon."""
    gdf = download_city_boundary(berlin_config)
    assert len(gdf) == 1
    assert gdf.crs is not None
    assert gdf.geometry.iloc[0].is_valid


def test_leipzig_boundary_wfs(leipzig_config):
    """Verify Leipzig boundary WFS returns 1 valid polygon."""
    gdf = download_city_boundary(leipzig_config)
    assert len(gdf) == 1
    assert gdf.crs is not None
    assert gdf.geometry.iloc[0].is_valid


def test_berlin_trees_wfs(berlin_config):
    """Verify Berlin tree WFS is reachable and has expected schema."""
    trees_cfg = berlin_config["trees"]
    gdf = _fetch_wfs_feature(trees_cfg["url"], trees_cfg["layers"][0])
    assert isinstance(gdf, gpd.GeoDataFrame)
    expected_columns = {
        "gisid",
        "gattung",
        "art_bot",
        "gattung_deutsch",
        "art_dtsch",  # Note: column is art_dtsch, not art_deutsch
        "pflanzjahr",
        "baumhoehe",
    }
    assert expected_columns.issubset(set(gdf.columns))


def test_leipzig_trees_wfs(leipzig_config):
    """Verify Leipzig tree WFS is reachable and has expected schema."""
    trees_cfg = leipzig_config["trees"]
    gdf = _fetch_wfs_feature(trees_cfg["url"], trees_cfg["layers"][0])
    assert isinstance(gdf, gpd.GeoDataFrame)
    expected_columns = {
        "objectid",
        "gattung",
        "ga_lang_wiss",
        "ga_lang_deutsch",
        "pflanzjahr",
        "baumhoehe",
    }
    assert expected_columns.issubset(set(gdf.columns))


def test_berlin_dom_atom_feed(berlin_config):
    """Verify Berlin DOM Atom feed is parseable and has tile links."""
    dom_cfg = berlin_config["elevation"]["dom"]
    dataset_url = _get_dataset_feed_url(dom_cfg["url"])
    tiles = _parse_atom_feed(dataset_url)
    assert tiles
    _assert_url_accessible(tiles[0]["url"])


def test_berlin_dgm_atom_feed(berlin_config):
    """Verify Berlin DGM Atom feed works."""
    dgm_cfg = berlin_config["elevation"]["dgm"]
    try:
        dataset_url = _get_dataset_feed_url(dgm_cfg["url"])
    except ValueError:
        dataset_url = dgm_cfg["url"]
    tiles = _parse_atom_feed(dataset_url)
    assert tiles
    _assert_url_accessible(tiles[0]["url"])


def test_leipzig_elevation_urls(leipzig_config):
    """Verify Leipzig elevation URL files exist and URLs are valid."""
    dom_cfg = leipzig_config["elevation"]["dom"]
    urls_path = get_config_dir() / "cities" / dom_cfg["urls_file"]
    urls = _load_urls_from_file(urls_path)
    assert len(urls) > 0
    _assert_url_accessible(urls[0])
