#!/usr/bin/env python3
"""
Inspect Phase 2 Parquet Datasets

Quick script to analyze column structure of Phase 2 outputs and classify them
into features, metadata, outlier flags, etc.

Usage:
    python scripts/inspect_phase2_data.py [path_to_phase_2_splits]
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def find_google_drive_path():
    """Try to locate Google Drive path automatically."""
    possible_paths = [
        Path.home()
        / "Library"
        / "CloudStorage"
        / "GoogleDrive-silas.pignotti@gmail.com"
        / "MyDrive"
        / "dev"
        / "urban-tree-transfer"
        / "data"
        / "phase_2_splits",
        Path.home()
        / "Google Drive"
        / "My Drive"
        / "dev"
        / "urban-tree-transfer"
        / "data"
        / "phase_2_splits",
        Path("/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_2_splits"),  # Colab
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


def classify_columns(columns):
    """Classify columns into categories."""
    classification = {
        "identifiers": [],
        "target": [],
        "metadata_analysis": [],
        "metadata_spatial": [],
        "chm_features": [],
        "outlier_flags": [],
        "sentinel_features": [],
    }

    # Spectral bands from Phase 2 correlation removal
    spectral_bands = ["B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12"]
    # Vegetation indices from Phase 2 correlation removal
    vegetation_indices = [
        "NDVI",
        "EVI",
        "VARI",
        "NDre1",
        "NDVIre",
        "CIre",
        "IRECI",
        "RTVIcore",
        "NDWI",
        "MSI",
    ]

    for col in columns:
        if col in ["tree_id", "city", "block_id"]:
            classification["identifiers"].append(col)
        elif col == "genus_latin":
            classification["target"].append(col)
        elif col in [
            "plant_year",
            "species_latin",
            "genus_german",
            "species_german",
            "tree_type",
            "position_corrected",
            "correction_distance",
        ]:
            classification["metadata_analysis"].append(col)
        elif col in ["x", "y", "geometry"]:
            classification["metadata_spatial"].append(col)
        elif col.startswith("CHM_"):
            classification["chm_features"].append(col)
        elif col.startswith("outlier_"):
            classification["outlier_flags"].append(col)
        else:
            # Check if it's a temporal feature (band or index with month suffix)
            parts = col.split("_")
            if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 2:
                base = "_".join(parts[:-1])
                if base in spectral_bands or base in vegetation_indices:
                    classification["sentinel_features"].append(col)

    return classification


def extract_temporal_structure(sentinel_features):
    """Extract temporal feature base names and months."""
    temporal_features = {}

    # Define feature types from Phase 2
    spectral_bands = ["B2", "B3", "B4", "B5", "B8", "B8A", "B11", "B12"]

    for col in sentinel_features:
        # Check if column ends with _XX pattern (month suffix)
        parts = col.split("_")
        if len(parts) >= 2 and parts[-1].isdigit() and len(parts[-1]) == 2:
            base = "_".join(parts[:-1])
            month = int(parts[-1])
            if base not in temporal_features:
                temporal_features[base] = {
                    "months": [],
                    "type": "spectral_band" if base in spectral_bands else "vegetation_index",
                }
            temporal_features[base]["months"].append(month)

    # Sort months for each base
    for base in temporal_features:
        temporal_features[base]["months"] = sorted(temporal_features[base]["months"])

    return temporal_features


def main():
    """Main inspection logic."""
    print("=" * 80)
    print("PHASE 2 DATASET INSPECTION")
    print("=" * 80)

    # Check for command-line argument
    if len(sys.argv) > 1:
        data_dir = Path(sys.argv[1]).expanduser()
        print(f"\n📂 Using provided path: {data_dir}")
    else:
        # Try to find Google Drive path
        data_dir = find_google_drive_path()

    if data_dir is None:
        print("\n❌ Error: Google Drive path not found automatically.")
        print("\nUsage:")
        print("  python scripts/inspect_phase2_data.py <path_to_phase_2_splits>")
        print("\nExample:")
        print(
            "  python scripts/inspect_phase2_data.py ~/Library/CloudStorage/GoogleDrive-xxx/MyDrive/dev/urban-tree-transfer/data/phase_2_splits"
        )
        return

    if not data_dir.exists():
        print(f"\n❌ Error: Directory not found: {data_dir}")
        return

    print(f"✅ Found data directory: {data_dir}")

    # Find a representative parquet file
    parquet_files = list(data_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"\n❌ Error: No parquet files found in {data_dir}")
        return

    # Use berlin_train.parquet as representative (baseline variant)
    target_file = "berlin_train.parquet"
    parquet_path = data_dir / target_file

    if not parquet_path.exists():
        # Fallback to first available parquet
        parquet_path = parquet_files[0]
        target_file = parquet_path.name

    print(f"\n📊 Inspecting: {target_file}")
    print(f"   Path: {parquet_path}")

    # Load parquet
    print("\n⏳ Loading parquet file...")
    df = pd.read_parquet(parquet_path)

    print(f"✅ Loaded: {len(df):,} rows x {len(df.columns)} columns")

    # Display all columns
    print("\n" + "=" * 80)
    print("ALL COLUMNS")
    print("=" * 80)
    for i, col in enumerate(df.columns, 1):
        print(f"{i:3d}. {col}")

    # Classify columns
    print("\n" + "=" * 80)
    print("COLUMN CLASSIFICATION")
    print("=" * 80)
    classification = classify_columns(df.columns)

    for category, cols in classification.items():
        if cols:
            print(f"\n{category.upper().replace('_', ' ')} ({len(cols)}):")
            for col in sorted(cols):
                dtype = df[col].dtype
                print(f"  - {col:40s} ({dtype})")

    # Temporal structure analysis
    if classification["sentinel_features"]:
        print("\n" + "=" * 80)
        print("TEMPORAL FEATURE STRUCTURE")
        print("=" * 80)
        temporal_features = extract_temporal_structure(classification["sentinel_features"])

        print(f"\nFeature bases: {len(temporal_features)}")
        all_months = sorted({m for info in temporal_features.values() for m in info["months"]})
        print(f"Months: {all_months}")

        # Group by feature type
        spectral_bands = {
            k: v for k, v in temporal_features.items() if v["type"] == "spectral_band"
        }
        vegetation_indices = {
            k: v for k, v in temporal_features.items() if v["type"] == "vegetation_index"
        }

        print(f"\nSpectral bands ({len(spectral_bands)}):")
        for base in sorted(spectral_bands.keys()):
            months = spectral_bands[base]["months"]
            print(f"  {base:20s} → months {months}")

        print(f"\nVegetation indices ({len(vegetation_indices)}):")
        for base in sorted(vegetation_indices.keys()):
            months = vegetation_indices[base]["months"]
            print(f"  {base:20s} → months {months}")

    # Missing values check
    print("\n" + "=" * 80)
    print("DATA QUALITY")
    print("=" * 80)
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️  Missing values found:")
        print(missing[missing > 0])
    else:
        print("✅ No missing values (expected after Phase 2 QC)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("FEATURE COUNT SUMMARY")
    print("=" * 80)
    summary = {
        "total_columns": len(df.columns),
        "identifiers": len(classification["identifiers"]),
        "target_variable": len(classification["target"]),
        "metadata_analysis": len(classification["metadata_analysis"]),
        "metadata_spatial": len(classification["metadata_spatial"]),
        "chm_features": len(classification["chm_features"]),
        "outlier_flags": len(classification["outlier_flags"]),
        "sentinel_features": len(classification["sentinel_features"]),
        "temporal_bases": len(temporal_features) if classification["sentinel_features"] else 0,
    }

    for key, value in summary.items():
        print(f"  {key.replace('_', ' ').title():30s}: {value}")

    # Validation checks
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS")
    print("=" * 80)

    checks = []
    checks.append(("CHM features == 3", len(classification["chm_features"]) == 3))
    checks.append(("Target variable present", "genus_latin" in classification["target"]))
    checks.append(("Outlier flags >= 4", len(classification["outlier_flags"]) >= 4))

    if classification["sentinel_features"]:
        checks.append(("Temporal bases == 18", len(temporal_features) == 18))
        months_per_base = [len(info["months"]) for info in temporal_features.values()]
        checks.append(("All features have 8 months", all(m == 8 for m in months_per_base)))

        spectral_count = len(
            [k for k, v in temporal_features.items() if v["type"] == "spectral_band"]
        )
        indices_count = len(
            [k for k, v in temporal_features.items() if v["type"] == "vegetation_index"]
        )
        checks.append(("Spectral bands == 8", spectral_count == 8))
        checks.append(("Vegetation indices == 10", indices_count == 10))

    for check_name, result in checks:
        status = "✅" if result else "❌"
        print(f"{status} {check_name}")

    # Export JSON schema
    print("\n" + "=" * 80)
    print("EXPORT SCHEMA")
    print("=" * 80)

    schema_output = {
        "version": "1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "source_file": target_file,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "column_classification": {k: sorted(v) for k, v in classification.items()},
        "feature_counts": summary,
    }

    if classification["sentinel_features"]:
        spectral_bands = {
            k: v["months"] for k, v in temporal_features.items() if v["type"] == "spectral_band"
        }
        vegetation_indices = {
            k: v["months"] for k, v in temporal_features.items() if v["type"] == "vegetation_index"
        }

        schema_output["temporal_structure"] = {
            "total_bases": len(temporal_features),
            "spectral_bands": {
                "count": len(spectral_bands),
                "bases": sorted(spectral_bands.keys()),
            },
            "vegetation_indices": {
                "count": len(vegetation_indices),
                "bases": sorted(vegetation_indices.keys()),
            },
            "months": sorted({m for info in temporal_features.values() for m in info["months"]}),
            "months_per_base": (
                next(iter({len(info["months"]) for info in temporal_features.values()}))
                if temporal_features
                else 0
            ),
        }

    # Export to outputs/phase_3/metadata/
    output_dir = Path(__file__).parent.parent / "outputs" / "phase_3" / "metadata"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dataset_schema.json"

    output_path.write_text(json.dumps(schema_output, indent=2), encoding="utf-8")
    print(f"✅ Exported: {output_path}")

    print("\n" + "=" * 80)
    print("INSPECTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
