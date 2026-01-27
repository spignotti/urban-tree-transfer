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
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import rasterio
import requests

from urban_tree_transfer.config import PROJECT_CRS, get_config_dir
from urban_tree_transfer.data_processing.boundaries import download_city_boundary
from urban_tree_transfer.data_processing.elevation import (
    DEFAULT_HEADERS,
    _collect_atom_zip_links,
    _get_dataset_feed_url,
    _load_urls_from_file,
    _xyz_to_geotiff,
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
    tiles = _collect_atom_zip_links(dataset_url)
    assert tiles
    _assert_url_accessible(tiles[0]["url"])


def test_berlin_dom_single_tile_download(berlin_config):
    """Download and convert a single Berlin DOM tile to verify full pipeline.

    This test verifies:
    1. ZIP file can be downloaded
    2. ZIP contains expected file format (XYZ ASCII .txt)
    3. XYZ file can be converted to GeoTIFF
    4. GeoTIFF has valid CRS and elevation data
    """
    dom_cfg = berlin_config["elevation"]["dom"]
    dataset_url = _get_dataset_feed_url(dom_cfg["url"])
    tiles = _collect_atom_zip_links(dataset_url)
    assert tiles, "No tiles found in feed"

    # Download only the first tile
    tile = tiles[0]
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "test_tile.zip"

        # Download
        response = requests.get(tile["url"], timeout=120, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

        # Extract
        with ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)
            extracted_files = zf.namelist()

        # Verify we got an XYZ file (Berlin format) or TIF
        xyz_files = [f for f in extracted_files if f.lower().endswith((".txt", ".xyz"))]
        tif_files = [f for f in extracted_files if f.lower().endswith((".tif", ".tiff"))]

        assert xyz_files or tif_files, (
            f"Expected .txt/.xyz (XYZ) or .tif files, got: {extracted_files}"
        )

        # If XYZ format, convert to GeoTIFF
        if xyz_files:
            xyz_path = tmpdir_path / xyz_files[0]
            tif_path = _xyz_to_geotiff(xyz_path)
        else:
            tif_path = tmpdir_path / tif_files[0]

        # Verify GeoTIFF is valid
        with rasterio.open(tif_path) as src:
            assert str(src.crs) == PROJECT_CRS, f"Wrong CRS: {src.crs}"
            data = src.read(1)
            # Should have some valid elevation data
            valid_mask = data != src.nodata
            assert valid_mask.sum() > 0, "No valid elevation data"
            # Berlin elevations should be roughly 30-150m
            valid_data = data[valid_mask]
            assert np.all(valid_data > 0), "Elevation values should be positive"
            assert np.all(valid_data < 500), "Elevation values unreasonably high"


def test_berlin_dgm_atom_feed(berlin_config):
    """Verify Berlin DGM Atom feed works."""
    dgm_cfg = berlin_config["elevation"]["dgm"]
    try:
        dataset_url = _get_dataset_feed_url(dgm_cfg["url"])
    except ValueError:
        dataset_url = dgm_cfg["url"]
    tiles = _collect_atom_zip_links(dataset_url)
    assert tiles
    _assert_url_accessible(tiles[0]["url"])


def test_berlin_dgm_single_tile_download(berlin_config):
    """Download and convert a single Berlin DGM tile to verify full pipeline."""
    dgm_cfg = berlin_config["elevation"]["dgm"]
    try:
        dataset_url = _get_dataset_feed_url(dgm_cfg["url"])
    except ValueError:
        dataset_url = dgm_cfg["url"]
    tiles = _collect_atom_zip_links(dataset_url)
    assert tiles, "No tiles found in feed"

    tile = tiles[0]
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "test_tile.zip"

        response = requests.get(tile["url"], timeout=120, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

        with ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)
            extracted_files = zf.namelist()

        xyz_files = [f for f in extracted_files if f.lower().endswith((".txt", ".xyz"))]
        tif_files = [f for f in extracted_files if f.lower().endswith((".tif", ".tiff"))]

        assert xyz_files or tif_files, (
            f"Expected .txt/.xyz (XYZ) or .tif files, got: {extracted_files}"
        )

        if xyz_files:
            xyz_path = tmpdir_path / xyz_files[0]
            tif_path = _xyz_to_geotiff(xyz_path)
        else:
            tif_path = tmpdir_path / tif_files[0]

        with rasterio.open(tif_path) as src:
            assert str(src.crs) == PROJECT_CRS
            data = src.read(1)
            valid_mask = data != src.nodata
            assert valid_mask.sum() > 0
            valid_data = data[valid_mask]
            # DGM (terrain) values should be lower than DOM (surface)
            assert np.all(valid_data > 0)
            assert np.all(valid_data < 300)


def test_leipzig_elevation_urls(leipzig_config):
    """Verify Leipzig elevation URL files exist and URLs are valid."""
    dom_cfg = leipzig_config["elevation"]["dom"]
    urls_path = get_config_dir() / "cities" / dom_cfg["urls_file"]
    urls = _load_urls_from_file(urls_path)
    assert len(urls) > 0
    # Note: HEAD requests fail on this WebDAV server, so we skip accessibility check here
    # The single tile download test below verifies actual downloads work


def test_leipzig_dom_single_tile_download(leipzig_config):
    """Download and verify a single Leipzig DOM tile.

    Leipzig uses GeoTIFF format (unlike Berlin's XYZ ASCII).
    """
    dom_cfg = leipzig_config["elevation"]["dom"]
    urls_path = get_config_dir() / "cities" / dom_cfg["urls_file"]
    urls = _load_urls_from_file(urls_path)
    assert urls, "No URLs found"

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "test_tile.zip"

        # Download with browser-like headers (required for Nextcloud/WebDAV)
        response = requests.get(urls[0], timeout=120, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

        with ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)
            extracted_files = zf.namelist()

        tif_files = [f for f in extracted_files if f.lower().endswith((".tif", ".tiff"))]
        assert tif_files, f"Expected .tif files, got: {extracted_files}"

        tif_path = tmpdir_path / tif_files[0]
        with rasterio.open(tif_path) as src:
            assert src.crs is not None
            data = src.read(1)
            valid_mask = data != src.nodata if src.nodata else data != 0
            assert valid_mask.sum() > 0


def test_leipzig_dgm_single_tile_download(leipzig_config):
    """Download and verify a single Leipzig DGM tile."""
    dgm_cfg = leipzig_config["elevation"]["dgm"]
    urls_path = get_config_dir() / "cities" / dgm_cfg["urls_file"]
    urls = _load_urls_from_file(urls_path)
    assert urls, "No URLs found"

    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        zip_path = tmpdir_path / "test_tile.zip"

        response = requests.get(urls[0], timeout=120, headers=DEFAULT_HEADERS)
        response.raise_for_status()
        zip_path.write_bytes(response.content)

        with ZipFile(zip_path, "r") as zf:
            zf.extractall(tmpdir_path)
            extracted_files = zf.namelist()

        tif_files = [f for f in extracted_files if f.lower().endswith((".tif", ".tiff"))]
        assert tif_files, f"Expected .tif files, got: {extracted_files}"

        tif_path = tmpdir_path / tif_files[0]
        with rasterio.open(tif_path) as src:
            assert src.crs is not None
            data = src.read(1)
            valid_mask = data != src.nodata if src.nodata else data != 0
            assert valid_mask.sum() > 0
