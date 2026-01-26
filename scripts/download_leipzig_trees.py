#!/usr/bin/env python
"""Download Leipzig tree cadastre and save as GeoPackage for Colab fallback.

This script downloads the raw Leipzig tree data from the WFS server and saves it
locally. The resulting GeoPackage can be uploaded to Google Drive as a fallback
when the WFS server is unreachable from Colab.

Usage:
    uv run python scripts/download_leipzig_trees.py

Output:
    data/trees/raw/leipzig_trees_raw.gpkg

Upload instructions:
    Upload the output file to Google Drive:
    /content/drive/MyDrive/dev/urban-tree-transfer/data/phase_1_processing/trees/raw/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from urban_tree_transfer.config.loader import load_city_config
from urban_tree_transfer.data_processing.trees import (
    _normalize_genus,
    _normalize_species,
    download_tree_cadastre,
)


def validate_raw_data(gdf, city_config: dict) -> dict:
    """Validate that raw data has expected columns and can be processed."""
    mapping = city_config.get("trees", {}).get("mapping", {})
    results = {
        "total_records": len(gdf),
        "columns_found": list(gdf.columns),
        "mapping_validation": {},
        "sample_values": {},
        "issues": [],
    }

    # Check each mapped column exists
    for target_col, source_col in mapping.items():
        if source_col is None:
            results["mapping_validation"][target_col] = "skipped (null mapping)"
            continue

        if source_col in gdf.columns:
            results["mapping_validation"][target_col] = f"OK ({source_col})"
            # Get sample non-null values
            non_null = gdf[source_col].dropna()
            if len(non_null) > 0:
                results["sample_values"][target_col] = non_null.head(3).tolist()
        else:
            results["mapping_validation"][target_col] = f"MISSING ({source_col})"
            results["issues"].append(f"Column '{source_col}' not found in data")

    # Test genus/species normalization
    genus_col = mapping.get("genus_latin")
    if genus_col and genus_col in gdf.columns:
        sample = gdf[genus_col].dropna().head(5)
        normalized = [_normalize_genus(v) for v in sample]
        results["genus_normalization_sample"] = list(zip(sample.tolist(), normalized, strict=True))

    species_col = mapping.get("species_latin")
    if species_col and species_col in gdf.columns:
        sample = gdf[species_col].dropna().head(5)
        normalized = [_normalize_species(v) for v in sample]
        results["species_normalization_sample"] = list(
            zip(sample.tolist(), normalized, strict=True)
        )

    # Check geometry
    if "geometry" not in gdf.columns:
        results["issues"].append("No geometry column found")
    else:
        geom_types = set(gdf.geometry.type.unique())
        results["geometry_types"] = list(geom_types)
        allowed = {"Point", "MultiPoint"}
        if not geom_types.issubset(allowed):
            results["issues"].append(f"Unexpected geometry types: {geom_types - allowed}")

    # Check CRS
    if gdf.crs is None:
        results["issues"].append("No CRS defined")
    else:
        results["crs"] = str(gdf.crs)

    return results


def main() -> int:
    """Download Leipzig trees and save as GeoPackage."""
    print("=" * 60)
    print("Leipzig Tree Cadastre Download")
    print("=" * 60)

    # Load config
    print("\nLoading Leipzig configuration...")
    config = load_city_config("leipzig")
    print(f"  Source: {config['trees']['source']}")
    print(f"  URL: {config['trees']['url']}")
    print(f"  Layers: {config['trees']['layers']}")

    # Download
    print("\nDownloading from WFS (this may take a few minutes)...")
    try:
        gdf = download_tree_cadastre(config)
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        print("\nThe Leipzig WFS server may be temporarily unavailable.")
        print("Try again later or check: https://geoserver.leipzig.de/geoserver/web/")
        return 1

    print(f"  Downloaded: {len(gdf)} trees")

    # Validate
    print("\nValidating data structure...")
    validation = validate_raw_data(gdf, config)

    print(f"\n  Total records: {validation['total_records']}")
    print(f"  CRS: {validation.get('crs', 'MISSING')}")
    print(f"  Geometry types: {validation.get('geometry_types', 'N/A')}")

    print("\n  Column mapping validation:")
    for col, status in validation["mapping_validation"].items():
        print(f"    {col}: {status}")

    if validation.get("genus_normalization_sample"):
        print("\n  Genus normalization samples (raw -> normalized):")
        for raw, norm in validation["genus_normalization_sample"][:3]:
            print(f"    '{raw}' -> '{norm}'")

    if validation.get("species_normalization_sample"):
        print("\n  Species normalization samples (raw -> normalized):")
        for raw, norm in validation["species_normalization_sample"][:3]:
            print(f"    '{raw}' -> '{norm}'")

    if validation["issues"]:
        print("\n  [WARNING] Issues found:")
        for issue in validation["issues"]:
            print(f"    - {issue}")

    # Save
    output_dir = Path(__file__).parent.parent / "data" / "trees" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "leipzig_trees_raw.gpkg"

    print(f"\nSaving to: {output_path}")
    gdf.to_file(output_path, driver="GPKG")

    # File size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {size_mb:.1f} MB")

    print("\n" + "=" * 60)
    print("SUCCESS!")
    print("=" * 60)
    print("\nUpload this file to Google Drive:")
    print(f"  {output_path}")
    print("\nDrive destination:")
    print("  /content/drive/MyDrive/dev/urban-tree-transfer/data/phase_1_processing/trees/raw/")
    print("\nThe trees module will automatically use this as fallback when WFS fails.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
