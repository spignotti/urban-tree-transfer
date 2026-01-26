"""Sentinel-2 Google Earth Engine utilities."""

# pyright: reportPrivateImportUsage=false

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import ee
import geopandas as gpd
import pandas as pd
import rasterio

from urban_tree_transfer.config import PROJECT_CRS, SPECTRAL_BANDS, VEGETATION_INDICES

DEFAULT_BUFFER_M = 500.0


def add_vegetation_indices(image: Any) -> Any:
    """Calculate vegetation indices for Sentinel-2."""

    def get_band(band: str) -> Any:
        return image.select(band).toFloat()

    b2 = get_band("B2")
    b3 = get_band("B3")
    b4 = get_band("B4")
    b5 = get_band("B5")
    b6 = get_band("B6")
    b7 = get_band("B7")
    b8 = get_band("B8")
    b11 = get_band("B11")
    b12 = get_band("B12")

    b2_s = b2.divide(10000.0)
    b4_s = b4.divide(10000.0)
    b8_s = b8.divide(10000.0)

    ndvi = b8.subtract(b4).divide(b8.add(b4)).rename("NDVI")
    gndvi = b8.subtract(b3).divide(b8.add(b3)).rename("GNDVI")
    evi = (
        b8_s.subtract(b4_s)
        .multiply(2.5)
        .divide(b8_s.add(b4_s.multiply(6)).subtract(b2_s.multiply(7.5)).add(1))
        .rename("EVI")
    )
    vari = b3.subtract(b4).divide(b3.add(b4).subtract(b2)).rename("VARI")
    ndre1 = b8.subtract(b5).divide(b8.add(b5)).rename("NDre1")
    ndvire = b8.subtract(b6).divide(b8.add(b6)).rename("NDVIre")
    cire = b8.divide(b5).subtract(1).rename("CIre")
    ireci = b7.subtract(b4).divide(b5.divide(b6)).rename("IRECI")
    rtvicore = (
        b8.subtract(b5).multiply(100).subtract(b8.subtract(b3).multiply(10)).rename("RTVIcore")
    )
    ndwi = b8.subtract(b11).divide(b8.add(b11)).rename("NDWI")
    msi = b11.divide(b8).rename("MSI")
    ndii = b8.subtract(b12).divide(b8.add(b12)).rename("NDII")
    kndvi = b8.subtract(b4).divide(b8.add(b4)).pow(2).tanh().rename("kNDVI")

    return image.addBands(
        [
            ndvi,
            gndvi,
            evi,
            vari,
            ndre1,
            ndvire,
            cire,
            ireci,
            rtvicore,
            ndwi,
            msi,
            ndii,
            kndvi,
        ]
    )


def _mask_scl(image: Any) -> Any:
    scl = image.select("SCL")
    mask = scl.eq(4).Or(scl.eq(5))
    return image.updateMask(mask)


def _clamp_bands(image: Any) -> Any:
    return image.clamp(0, 10000)


def _geometry_to_ee(geom: Any) -> Any:
    if geom.geom_type == "Polygon":
        return ee.Geometry.Polygon([list(geom.exterior.coords)])
    if geom.geom_type == "MultiPolygon":
        polygons = [list(poly.exterior.coords) for poly in geom.geoms]
        return ee.Geometry.MultiPolygon(polygons)
    raise ValueError(f"Unsupported geometry type: {geom.geom_type}")


def create_gee_tasks(
    boundary: gpd.GeoDataFrame,
    city: str,
    year: int,
    months: list[int],
    drive_folder: str = "sentinel2_processing_stage",
    scale: int = 10,
    crs: str = PROJECT_CRS,
    buffer_m: float = DEFAULT_BUFFER_M,
) -> list[Any]:
    """Create GEE export tasks for Sentinel-2 composites.

    Args:
        boundary: Boundary GeoDataFrame.
        city: City name for task description.
        year: Year for Sentinel-2 data.
        months: List of months to export.
        drive_folder: Google Drive folder for exports.
        scale: Output resolution in meters.
        crs: Output CRS.
        buffer_m: Buffer distance in meters to expand boundary (default: 500m).

    Returns:
        List of started GEE tasks.
    """
    boundary = boundary.copy()

    if boundary.crs is not None and str(boundary.crs) != PROJECT_CRS:
        boundary = boundary.to_crs(PROJECT_CRS)

    if buffer_m > 0:
        boundary["geometry"] = boundary.geometry.buffer(buffer_m)

    boundary = boundary.to_crs("EPSG:4326")

    merged_geom = boundary.geometry.unary_union
    ee_geom = _geometry_to_ee(merged_geom)

    tasks: list[Any] = []
    for month in months:
        next_year = year + 1 if month == 12 else year
        next_month = 1 if month == 12 else month + 1
        start_date = f"{year}-{month:02d}-01"
        end_date = f"{next_year}-{next_month:02d}-01"

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(ee_geom)
            .filterDate(start_date, end_date)
            .map(_mask_scl)
            .map(_clamp_bands)
            .map(add_vegetation_indices)
        )

        median_img = collection.median().clip(ee_geom)
        out_bands = SPECTRAL_BANDS + VEGETATION_INDICES
        final_img = median_img.select(out_bands).toFloat()

        description = f"S2_{city}_{year}_{month:02d}_median"
        task = ee.batch.Export.image.toDrive(
            image=final_img,
            description=description,
            folder=drive_folder,
            region=ee_geom,
            scale=scale,
            crs=crs,
            maxPixels=1e10,
        )
        task.start()
        tasks.append(task)

    return tasks


def check_task_status(tasks: list[Any]) -> dict[str, int]:
    """Check status of running GEE tasks."""
    counts: dict[str, int] = {}
    for task in tasks:
        state = task.status().get("state", "UNKNOWN")
        counts[state] = counts.get(state, 0) + 1
    return counts


def monitor_tasks(
    tasks: list[Any],
    interval_seconds: int = 30,
    max_wait_minutes: int = 120,
) -> dict[str, list[Any]]:
    """Monitor GEE tasks until completion or timeout.

    Args:
        tasks: List of GEE tasks to monitor.
        interval_seconds: Polling interval in seconds (default: 30).
        max_wait_minutes: Maximum wait time in minutes (default: 120).

    Returns:
        Dict with 'completed', 'failed', and 'pending' task lists.
    """
    max_iterations = (max_wait_minutes * 60) // interval_seconds
    completed: list[Any] = []
    failed: list[Any] = []
    pending = list(tasks)

    for _ in range(max_iterations):
        if not pending:
            break

        still_pending: list[Any] = []
        for task in pending:
            status = task.status()
            state = status.get("state", "UNKNOWN")

            if state == "COMPLETED":
                completed.append(task)
            elif state in ("FAILED", "CANCELLED"):
                failed.append(task)
            else:
                still_pending.append(task)

        pending = still_pending
        if pending:
            time.sleep(interval_seconds)

    return {"completed": completed, "failed": failed, "pending": pending}


def validate_sentinel_raster(raster_path: Path) -> dict[str, Any]:
    """Validate a Sentinel-2 raster file.

    Args:
        raster_path: Path to the raster file.

    Returns:
        Dict with validation results including band count, bounds, and coverage.
    """
    if not raster_path.exists():
        return {"valid": False, "error": "File not found", "path": str(raster_path)}

    with rasterio.open(raster_path) as src:
        expected_bands = len(SPECTRAL_BANDS) + len(VEGETATION_INDICES)
        actual_bands = src.count

        data = src.read()
        valid_pixels = (~pd.isna(data)).sum()
        total_pixels = data.size
        coverage = valid_pixels / total_pixels if total_pixels > 0 else 0.0

        return {
            "valid": actual_bands == expected_bands and coverage > 0.5,
            "path": str(raster_path),
            "crs": str(src.crs) if src.crs else None,
            "bounds": list(src.bounds),
            "shape": list(src.shape),
            "expected_bands": expected_bands,
            "actual_bands": actual_bands,
            "coverage": round(coverage, 4),
        }


def batch_validate_sentinel(paths: list[Path]) -> pd.DataFrame:
    """Validate multiple Sentinel-2 raster files.

    Args:
        paths: List of paths to raster files.

    Returns:
        DataFrame with validation results for each file.
    """
    results = [validate_sentinel_raster(p) for p in paths]
    return pd.DataFrame(results)


def move_exports_to_destination(
    staging_dir: Path,
    destination_dir: Path,
    pattern: str = "S2_*.tif",
) -> list[Path]:
    """Move exported Sentinel-2 files from staging to destination.

    Args:
        staging_dir: Source directory (GEE export staging folder).
        destination_dir: Target directory for final files.
        pattern: Glob pattern for files to move (default: S2_*.tif).

    Returns:
        List of paths to moved files in destination.
    """
    import shutil

    destination_dir.mkdir(parents=True, exist_ok=True)

    moved_files: list[Path] = []
    for src_path in staging_dir.glob(pattern):
        dst_path = destination_dir / src_path.name
        shutil.move(str(src_path), str(dst_path))
        moved_files.append(dst_path)

    return moved_files
