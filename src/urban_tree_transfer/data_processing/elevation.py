"""Elevation (DOM/DGM) download and harmonization."""

from __future__ import annotations

import json
import re
import time
import warnings
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from zipfile import ZipFile, is_zipfile

import geopandas as gpd
import numpy as np
import rasterio
import requests
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject
from shapely.geometry import box

from urban_tree_transfer.config import PROJECT_CRS, get_config_dir

DEFAULT_NODATA = -9999.0
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom", "georss": "http://www.georss.org/georss"}
DEFAULT_HEADERS = {"User-Agent": "urban-tree-transfer/1.0"}


def _get_dataset_feed_url(main_feed_url: str) -> str:
    """Parse main Atom feed to get nested dataset feed URL.

    Berlin's elevation data uses a nested Atom feed structure:
    1. Main feed contains link to dataset feed
    2. Dataset feed contains actual tile download links

    Args:
        main_feed_url: URL to the main Atom feed.

    Returns:
        URL to the dataset feed with actual tile links.
    """
    response = requests.get(main_feed_url, timeout=60, headers=DEFAULT_HEADERS)
    response.raise_for_status()

    root = ET.fromstring(response.content)

    for link in root.findall("atom:link", ATOM_NS):
        rel = link.get("rel", "")
        link_type = link.get("type", "")
        href = link.get("href", "")

        if rel == "alternate" and "atom+xml" in link_type and href:
            return href

    for entry in root.findall("atom:entry", ATOM_NS):
        for link in entry.findall("atom:link", ATOM_NS):
            rel = link.get("rel", "")
            link_type = link.get("type", "")
            href = link.get("href", "")

            if "atom+xml" in link_type and href:
                return href

    raise ValueError(f"Could not find dataset feed URL in {main_feed_url}")


def _parse_atom_feed(feed_url: str) -> list[dict[str, str]]:
    """Parse Atom feed for tile links.

    Handles Berlin's nested Atom feed structure where entries may contain
    section links pointing to actual ZIP downloads.

    Args:
        feed_url: URL to the Atom feed.

    Returns:
        List of dicts with 'title' and 'url' keys for each tile.
    """
    response = requests.get(feed_url, timeout=60, headers=DEFAULT_HEADERS)
    response.raise_for_status()

    root = ET.fromstring(response.content)
    links: list[dict[str, str]] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        title_elem = entry.find("atom:title", ATOM_NS)
        title = title_elem.text if title_elem is not None and title_elem.text else ""

        for link in entry.findall("atom:link", ATOM_NS):
            href = link.get("href", "")
            if not href:
                continue
            links.append(
                {
                    "title": title,
                    "url": href,
                    "rel": link.get("rel", ""),
                    "type": link.get("type", ""),
                }
            )

    return links


def _is_zip_link(url: str, link_type: str) -> bool:
    return url.lower().endswith(".zip") or "application/zip" in link_type.lower()


def _looks_like_feed_link(url: str, rel: str, link_type: str) -> bool:
    rel = rel.lower()
    link_type = link_type.lower()
    url_lower = url.lower()
    return (
        rel == "section"
        or "atom+xml" in link_type
        or "application/atom+xml" in link_type
        or url_lower.endswith(".xml")
        or url_lower.endswith("/")
    )


def _collect_atom_zip_links(feed_url: str, max_depth: int = 3) -> list[dict[str, str]]:
    """Collect ZIP download links from an Atom feed, following section links."""
    visited: set[str] = set()
    queue: list[tuple[str, int]] = [(feed_url, 0)]
    zip_links: list[dict[str, str]] = []

    while queue:
        current_url, depth = queue.pop(0)
        if current_url in visited:
            continue
        visited.add(current_url)

        links = _parse_atom_feed(current_url)
        for link in links:
            url = link.get("url", "")
            if not url:
                continue

            if _is_zip_link(url, link.get("type", "")):
                title = link.get("title") or Path(url).name
                zip_links.append({"title": title, "url": url})
                continue

            if depth < max_depth and _looks_like_feed_link(
                url, link.get("rel", ""), link.get("type", "")
            ):
                queue.append((url, depth + 1))

    return zip_links


def _extract_tile_coordinates(title: str) -> tuple[int, int] | None:
    """Extract km coordinates from tile title.

    Berlin elevation tiles use naming convention like 'dom1_33_XXX_YYYY'
    where XXX and YYYY are km coordinates in EPSG:25833.

    Args:
        title: Tile title from Atom feed.

    Returns:
        Tuple of (x_km, y_km) or None if parsing fails.
    """
    patterns = [
        r"(\d{3})_(\d{4})",
        r"_(\d{3})_(\d{4})",
        r"(\d+)_(\d+)(?:\.zip)?$",
    ]

    for pattern in patterns:
        match = re.search(pattern, title)
        if match:
            x_km = int(match.group(1))
            y_km = int(match.group(2))
            return (x_km, y_km)

    return None


def _filter_tiles_by_boundary(
    tiles: list[dict[str, str]],
    boundary_gdf: gpd.GeoDataFrame,
    buffer_m: float = 500.0,
) -> list[dict[str, str]]:
    """Filter tiles to those intersecting with boundary.

    Args:
        tiles: List of tile dicts with 'title' and 'url' keys.
        boundary_gdf: Boundary GeoDataFrame.
        buffer_m: Buffer distance in meters.

    Returns:
        Filtered list of tiles intersecting the buffered boundary.
    """
    if boundary_gdf.empty:
        return tiles

    boundary = boundary_gdf.to_crs(PROJECT_CRS)
    boundary_geom = boundary.geometry.unary_union
    if buffer_m > 0:
        boundary_geom = boundary_geom.buffer(buffer_m)

    filtered: list[dict[str, str]] = []
    for tile in tiles:
        coords = _extract_tile_coordinates(tile["title"])
        if coords is None:
            filtered.append(tile)
            continue

        x_km, y_km = coords
        x_min = x_km * 1000
        y_min = y_km * 1000
        x_max = x_min + 1000
        y_max = y_min + 1000
        tile_box = box(x_min, y_min, x_max, y_max)

        if tile_box.intersects(boundary_geom):
            filtered.append(tile)

    return filtered


def _download_atom_feed_tiles(
    feed_url: str,
    output_dir: Path,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    buffer_m: float = 500.0,
    progress: bool = True,
) -> list[Path]:
    """Download tiles from Berlin Atom feed.

    Args:
        feed_url: Main Atom feed URL.
        output_dir: Directory for downloaded tiles.
        boundary_gdf: Optional boundary for spatial filtering.
        buffer_m: Buffer distance for spatial filtering.

    Returns:
        List of paths to extracted raster tiles.
    """
    try:
        dataset_feed_url = _get_dataset_feed_url(feed_url)
    except ValueError:
        dataset_feed_url = feed_url

    tiles = _collect_atom_zip_links(dataset_feed_url)
    if not tiles:
        raise ValueError(f"No tiles found in Atom feed: {feed_url}")

    if boundary_gdf is not None:
        tiles = _filter_tiles_by_boundary(tiles, boundary_gdf, buffer_m)
        if not tiles:
            raise ValueError("No tiles intersect the provided boundary")

    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    total_tiles = len(tiles)
    if progress:
        print(f"Downloading {total_tiles} elevation tiles to {output_dir}...")

    for idx, tile in enumerate(tiles, start=1):
        url = tile["url"]
        filename = url.split("/")[-1]
        if not filename.endswith(".zip"):
            filename = f"{tile['title']}.zip"

        zip_path = output_dir / filename
        if progress:
            print(f"[{idx}/{total_tiles}] {filename}")
        _download_file(url, zip_path, progress_label=filename)
        if not is_zipfile(zip_path):
            raise ValueError(f"Downloaded file is not a ZIP archive: {url}")

        with ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
            for name in zf.namelist():
                if name.lower().endswith((".tif", ".tiff")):
                    extracted.append(output_dir / name)

    if not extracted:
        raise ValueError("No raster tiles extracted from Atom feed downloads.")

    return extracted


def _download_file(
    url: str,
    output_path: Path,
    progress_label: str | None = None,
    log_every_seconds: int = 30,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=300, headers=DEFAULT_HEADERS) as response:
        response.raise_for_status()
        total_bytes = response.headers.get("Content-Length")
        total_mb = (
            int(total_bytes) / (1024 * 1024) if total_bytes and total_bytes.isdigit() else None
        )
        downloaded = 0
        last_log = time.time()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    downloaded += len(chunk)
                    if progress_label and log_every_seconds > 0:
                        now = time.time()
                        if now - last_log >= log_every_seconds:
                            downloaded_mb = downloaded / (1024 * 1024)
                            if total_mb:
                                print(f"  {progress_label}: {downloaded_mb:.1f}/{total_mb:.1f} MB")
                            else:
                                print(f"  {progress_label}: {downloaded_mb:.1f} MB")
                            last_log = now
    return output_path


def _load_urls_from_file(urls_file: Path) -> list[str]:
    if not urls_file.exists():
        raise FileNotFoundError(f"Elevation URL list not found: {urls_file}")
    if urls_file.suffix.lower() == ".json":
        data = json.loads(urls_file.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list of URLs in {urls_file}")
        return [str(item) for item in data]
    return [
        line.strip() for line in urls_file.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _ensure_project_crs(raster_path: Path) -> Path:
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            warnings.warn(f"Missing CRS for {raster_path}", stacklevel=2)
            return raster_path
        if str(src.crs) == PROJECT_CRS:
            return raster_path

        warnings.warn(
            f"Reprojecting {raster_path.name} from {src.crs} to {PROJECT_CRS}",
            stacklevel=2,
        )
        transform, width, height = calculate_default_transform(
            src.crs, PROJECT_CRS, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(crs=PROJECT_CRS, transform=transform, width=width, height=height)

        reprojected_path = raster_path.with_name(f"{raster_path.stem}_reprojected.tif")
        with rasterio.open(reprojected_path, "w", **profile) as dst:
            for band_idx in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_idx),
                    destination=rasterio.band(dst, band_idx),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=PROJECT_CRS,
                    resampling=Resampling.bilinear,
                )
        return reprojected_path


def _clip_to_boundary(raster_path: Path, boundary_gdf: gpd.GeoDataFrame, buffer_m: float) -> Path:
    if boundary_gdf.empty:
        return raster_path

    with rasterio.open(raster_path) as src:
        boundary = boundary_gdf.to_crs(src.crs)
        geometry = boundary.geometry.unary_union
        if buffer_m > 0:
            geometry = gpd.GeoSeries([geometry], crs=src.crs).buffer(buffer_m).iloc[0]
        out_image, out_transform = mask(src, [geometry], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            {"height": out_image.shape[1], "width": out_image.shape[2], "transform": out_transform}
        )

    clipped_path = raster_path.with_name(f"{raster_path.stem}_clipped.tif")
    with rasterio.open(clipped_path, "w", **out_meta) as dst:
        dst.write(out_image)
    return clipped_path


def _mosaic_tiles(tile_paths: list[Path], output_path: Path) -> Path:
    datasets = [rasterio.open(path) for path in tile_paths]
    mosaic, transform = merge(datasets)
    profile = datasets[0].profile.copy()
    profile.update(
        height=mosaic.shape[1],
        width=mosaic.shape[2],
        transform=transform,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mosaic)

    for dataset in datasets:
        dataset.close()

    return output_path


def _download_zip_list(urls: list[str], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []

    for url in urls:
        filename = url.split("/")[-1]
        zip_path = output_dir / filename
        _download_file(url, zip_path)

        with ZipFile(zip_path, "r") as zf:
            zf.extractall(output_dir)
            for name in zf.namelist():
                if name.lower().endswith(".tif") or name.lower().endswith(".tiff"):
                    extracted.append(output_dir / name)

    if not extracted:
        raise ValueError("No raster tiles extracted from ZIP list.")
    return extracted


def download_elevation(
    city_config: dict[str, Any],
    data_type: str,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    buffer_m: float = 500.0,
    progress: bool = True,
) -> Path:
    """Download DOM or DGM for city."""
    elev_cfg = city_config.get("elevation", {}).get(data_type)
    if not isinstance(elev_cfg, dict):
        raise ValueError(f"Elevation config for {data_type} is missing.")

    source_type = elev_cfg.get("type")
    url = elev_cfg.get("url")

    output_dir = Path("data/elevation") / city_config["name"].lower()
    output_path = output_dir / f"{data_type}.tif"

    if source_type == "direct_download":
        if not url:
            raise ValueError(f"Elevation config for {data_type} requires a url.")
        return _download_file(url, output_path)
    if source_type == "zip_list":
        urls_file = elev_cfg.get("urls_file")
        if not urls_file:
            raise ValueError(f"Elevation config for {data_type} requires urls_file.")
        # Resolve urls_file relative to config/cities directory
        urls_path = get_config_dir() / "cities" / urls_file
        urls = _load_urls_from_file(urls_path)
        raw_dir = output_dir / f"{data_type}_tiles"
        tile_paths = _download_zip_list(urls, raw_dir)
        mosaic_path = output_dir / f"{data_type}_mosaic.tif"
        raster_path = _mosaic_tiles(tile_paths, mosaic_path)
        raster_path = _ensure_project_crs(raster_path)
        if boundary_gdf is not None:
            raster_path = _clip_to_boundary(raster_path, boundary_gdf, buffer_m=buffer_m)
        return raster_path
    if source_type == "atom_feed":
        if not url:
            raise ValueError(f"Elevation config for {data_type} requires a url.")
        raw_dir = output_dir / f"{data_type}_tiles"
        tile_paths = _download_atom_feed_tiles(url, raw_dir, boundary_gdf, buffer_m, progress)
        mosaic_path = output_dir / f"{data_type}_mosaic.tif"
        raster_path = _mosaic_tiles(tile_paths, mosaic_path)
        raster_path = _ensure_project_crs(raster_path)
        if boundary_gdf is not None:
            raster_path = _clip_to_boundary(raster_path, boundary_gdf, buffer_m=buffer_m)
        return raster_path

    raise ValueError(f"Unsupported elevation type: {source_type}")


def harmonize_elevation(dom_path: Path, dgm_path: Path, output_path: Path) -> None:
    """Harmonize elevation data to consistent format."""
    output_path.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dom_path) as dom_src:
        dom_data = dom_src.read(1).astype(np.float32)
        dom_profile = dom_src.profile.copy()

    with rasterio.open(dgm_path) as dgm_src:
        dgm_data = dgm_src.read(1).astype(np.float32)
        dgm_profile = dgm_src.profile.copy()

    dom_nodata = dom_profile.get("nodata", DEFAULT_NODATA)
    dgm_nodata = dgm_profile.get("nodata", DEFAULT_NODATA)

    dom_profile.update(
        nodata=dom_nodata,
        compress="LZW",
        tiled=True,
    )

    dgm_aligned = np.full(dom_data.shape, dgm_nodata, dtype=np.float32)
    reproject(
        source=dgm_data,
        destination=dgm_aligned,
        src_transform=dgm_profile["transform"],
        src_crs=dgm_profile.get("crs"),
        src_nodata=dgm_nodata,
        dst_transform=dom_profile["transform"],
        dst_crs=dom_profile.get("crs") or PROJECT_CRS,
        dst_nodata=dgm_nodata,
        resampling=Resampling.bilinear,
    )

    dom_out = output_path / "dom_1m.tif"
    dgm_out = output_path / "dgm_1m.tif"

    dom_profile.update(crs=PROJECT_CRS)
    dgm_profile = dom_profile.copy()

    with rasterio.open(dom_out, "w", **dom_profile) as dst:
        dst.write(dom_data, 1)

    with rasterio.open(dgm_out, "w", **dgm_profile) as dst:
        dst.write(dgm_aligned, 1)
