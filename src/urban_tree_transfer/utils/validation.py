"""Data validation utilities for Phase 1 outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
import rasterio

from urban_tree_transfer.config import PROJECT_CRS

BOUNDARY_TOLERANCE_M = 10.0


def validate_crs(data: gpd.GeoDataFrame | Path, expected_crs: str = PROJECT_CRS) -> dict[str, Any]:
    """Validate that data has the expected CRS.

    Args:
        data: GeoDataFrame or path to raster file.
        expected_crs: Expected CRS string (default: PROJECT_CRS).

    Returns:
        Dict with validation result and details.
    """
    if isinstance(data, Path):
        if not data.exists():
            return {"valid": False, "error": "File not found", "path": str(data)}
        with rasterio.open(data) as src:
            actual_crs = str(src.crs) if src.crs else None
    else:
        actual_crs = str(data.crs) if data.crs else None

    return {
        "valid": actual_crs == expected_crs,
        "expected_crs": expected_crs,
        "actual_crs": actual_crs,
    }


def validate_within_boundary(
    data: gpd.GeoDataFrame | Path,
    boundary_gdf: gpd.GeoDataFrame,
    tolerance_m: float = BOUNDARY_TOLERANCE_M,
) -> dict[str, Any]:
    """Validate that data is within boundary (with tolerance).

    Args:
        data: GeoDataFrame or path to raster file.
        boundary_gdf: Boundary GeoDataFrame.
        tolerance_m: Tolerance in meters for boundary check.

    Returns:
        Dict with validation result and details.
    """
    if boundary_gdf.empty:
        return {"valid": True, "message": "No boundary provided"}

    boundary = boundary_gdf.to_crs(PROJECT_CRS)
    boundary_geom = boundary.geometry.unary_union.buffer(tolerance_m)

    if isinstance(data, Path):
        if not data.exists():
            return {"valid": False, "error": "File not found", "path": str(data)}
        with rasterio.open(data) as src:
            from shapely.geometry import box

            data_bounds = box(*src.bounds)
            is_within = boundary_geom.contains(data_bounds)
            return {
                "valid": is_within,
                "data_bounds": list(src.bounds),
                "boundary_bounds": list(boundary_geom.bounds),
            }
    else:
        data = data.to_crs(PROJECT_CRS)
        data_bounds = data.geometry.unary_union
        is_within = boundary_geom.contains(data_bounds)
        return {
            "valid": is_within,
            "data_bounds": list(data_bounds.bounds) if data_bounds else None,
            "boundary_bounds": list(boundary_geom.bounds),
        }


def validate_no_null_geometries(gdf: gpd.GeoDataFrame) -> dict[str, Any]:
    """Validate that GeoDataFrame has no null geometries.

    Args:
        gdf: GeoDataFrame to validate.

    Returns:
        Dict with validation result and count of null geometries.
    """
    null_count = gdf.geometry.isna().sum()
    return {
        "valid": null_count == 0,
        "null_count": int(null_count),
        "total_count": len(gdf),
    }


def validate_schema(
    gdf: gpd.GeoDataFrame,
    expected_columns: list[str],
    expected_dtypes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Validate GeoDataFrame schema matches expected columns and dtypes.

    Args:
        gdf: GeoDataFrame to validate.
        expected_columns: List of expected column names.
        expected_dtypes: Optional dict of column name to expected dtype.

    Returns:
        Dict with validation result and details.
    """
    actual_columns = list(gdf.columns)
    missing_columns = [c for c in expected_columns if c not in actual_columns]
    extra_columns = [c for c in actual_columns if c not in expected_columns]

    dtype_mismatches: list[dict[str, str]] = []
    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col in gdf.columns:
                actual_dtype = str(gdf[col].dtype)
                if actual_dtype != expected_dtype:
                    dtype_mismatches.append(
                        {"column": col, "expected": expected_dtype, "actual": actual_dtype}
                    )

    return {
        "valid": len(missing_columns) == 0 and len(dtype_mismatches) == 0,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "dtype_mismatches": dtype_mismatches,
    }


def validate_dataset(
    data: gpd.GeoDataFrame | Path,
    boundary_gdf: gpd.GeoDataFrame | None = None,
    expected_columns: list[str] | None = None,
    expected_dtypes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Comprehensive validation of a dataset.

    Args:
        data: GeoDataFrame or path to raster file.
        boundary_gdf: Optional boundary for spatial validation.
        expected_columns: Optional list of expected columns (for GeoDataFrame).
        expected_dtypes: Optional dict of expected dtypes (for GeoDataFrame).

    Returns:
        Dict with all validation results.
    """
    results: dict[str, Any] = {"overall_valid": True}

    crs_result = validate_crs(data)
    results["crs"] = crs_result
    if not crs_result["valid"]:
        results["overall_valid"] = False

    if boundary_gdf is not None and not boundary_gdf.empty:
        boundary_result = validate_within_boundary(data, boundary_gdf)
        results["within_boundary"] = boundary_result
        if not boundary_result["valid"]:
            results["overall_valid"] = False

    if isinstance(data, gpd.GeoDataFrame):
        null_geom_result = validate_no_null_geometries(data)
        results["null_geometries"] = null_geom_result
        if not null_geom_result["valid"]:
            results["overall_valid"] = False

        if expected_columns:
            schema_result = validate_schema(data, expected_columns, expected_dtypes)
            results["schema"] = schema_result
            if not schema_result["valid"]:
                results["overall_valid"] = False

    return results


def generate_validation_report(datasets: dict[str, gpd.GeoDataFrame | Path]) -> dict[str, Any]:
    """Generate comprehensive validation report for multiple datasets.

    Args:
        datasets: Dict mapping dataset names to GeoDataFrames or raster paths.

    Returns:
        Dict with validation results for each dataset.
    """
    report: dict[str, Any] = {"datasets": {}, "summary": {"valid": 0, "invalid": 0}}

    for name, data in datasets.items():
        result = validate_dataset(data)
        report["datasets"][name] = result

        if result["overall_valid"]:
            report["summary"]["valid"] += 1
        else:
            report["summary"]["invalid"] += 1

    report["summary"]["total"] = len(datasets)
    report["summary"]["all_valid"] = report["summary"]["invalid"] == 0

    return report
