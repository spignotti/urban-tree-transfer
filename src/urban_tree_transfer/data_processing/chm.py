"""CHM creation and filtering."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask as rasterio_mask

from urban_tree_transfer.config import CHM_MAX_VALID, CHM_MIN_VALID, PROJECT_CRS

DEFAULT_NODATA = -9999.0


def clip_chm_to_boundary(
    chm_path: Path,
    boundary_gdf: gpd.GeoDataFrame,
    output_path: Path,
    buffer_m: float = 500.0,
) -> Path:
    """Clip CHM raster to boundary with optional buffer.

    Args:
        chm_path: Path to input CHM raster.
        boundary_gdf: Boundary GeoDataFrame.
        output_path: Path for clipped output.
        buffer_m: Buffer distance in meters (default: 500m).

    Returns:
        Path to clipped CHM raster.
    """
    if boundary_gdf.empty:
        return chm_path

    with rasterio.open(chm_path) as src:
        boundary = boundary_gdf.to_crs(src.crs if src.crs else PROJECT_CRS)
        clip_geom = boundary.geometry.unary_union

        if buffer_m > 0:
            clip_geom = clip_geom.buffer(buffer_m)

        out_image, out_transform = rasterio_mask(src, [clip_geom], crop=True)
        out_meta = src.meta.copy()
        out_meta.update(
            height=out_image.shape[1],
            width=out_image.shape[2],
            transform=out_transform,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(out_image)

    return output_path


def create_chm(dom_path: Path, dgm_path: Path, output_path: Path) -> None:
    """Create CHM from DOM and DGM."""
    with rasterio.open(dom_path) as dom_src, rasterio.open(dgm_path) as dgm_src:
        dom = dom_src.read(1).astype(np.float32)
        dgm = dgm_src.read(1).astype(np.float32)

        if dom.shape != dgm.shape:
            raise ValueError("DOM and DGM shapes do not match. Harmonize first.")

        nodata = dom_src.nodata if dom_src.nodata is not None else DEFAULT_NODATA
        dom = np.where(np.isclose(dom, nodata), np.nan, dom)
        dgm = np.where(np.isclose(dgm, dgm_src.nodata or nodata), np.nan, dgm)

        chm = dom - dgm
        chm[np.isnan(dom) | np.isnan(dgm)] = np.nan

        profile = dom_src.profile.copy()
        profile.update(dtype=rasterio.float32, nodata=nodata, compress="LZW", tiled=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(np.where(np.isnan(chm), nodata, chm).astype(np.float32), 1)


def filter_chm(
    chm_path: Path,
    output_path: Path,
    min_val: float = CHM_MIN_VALID,
    max_val: float = CHM_MAX_VALID,
) -> None:
    """Filter CHM values to valid range."""
    with rasterio.open(chm_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else DEFAULT_NODATA
        mask = np.isclose(data, nodata)
        data = np.where(mask, np.nan, data)

        data = np.where(data < min_val, 0.0, data)
        data = np.where(data > max_val, np.nan, data)

        profile = src.profile.copy()
        profile.update(dtype=rasterio.float32, nodata=nodata, compress="LZW", tiled=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.where(np.isnan(data), nodata, data).astype(np.float32), 1)
