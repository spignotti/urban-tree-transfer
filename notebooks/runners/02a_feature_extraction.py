# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Urban Tree Transfer - Runner Notebook
#
# **Title:** Feature Extraction
#
# **Phase:** 2 - Feature Engineering
#
# **Step:** 02a - Feature Extraction
#
# **Purpose:** Extract CHM and Sentinel-2 features for Berlin and Leipzig tree datasets.
#
# **Input:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_1_processing`
#
# **Output:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_2_features`
#
# **Author:** Silas Pignotti
#
# **Created:** 2025-01-15
#
# **Updated:** 2026-03-11
#
# Runner notebooks execute data processing only: input -> processing -> output.
# No analysis, no interpretation. They should be deterministic and repeatable.

# %%
# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================
# Runtime: CPU (Standard)
# GPU: Not required
# RAM: High-RAM recommended
#
# Prerequisites:
#   - GITHUB_TOKEN in Colab Secrets (key icon in sidebar)
#   - Google Drive mounted (Cell 2)
# ============================================================================

import os
import subprocess
import sys

TEST_MODE = os.environ.get("NOTEBOOK_TEST_MODE") == "1"
IN_COLAB = "google.colab" in sys.modules
RUN_NOTEBOOK = IN_COLAB and not TEST_MODE

if RUN_NOTEBOOK:
    from google.colab import userdata

    token = userdata.get("GITHUB_TOKEN")
    if not token:
        raise ValueError(
            "GITHUB_TOKEN not found in Colab Secrets.\n"
            "1. Click the key icon in the left sidebar\n"
            "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
            "3. Toggle 'Notebook access' ON"
        )

    repo_url = f"git+https://{token}@github.com/silas-workspace/urban-tree-transfer.git"
    subprocess.run(["pip", "install", repo_url, "-q"], check=True)
    print("OK: Package installed")
else:
    print("Notebook test mode or non-Colab environment: skipping repo install.")


# %%
# ============================================================================
# 2. GOOGLE DRIVE
# ============================================================================

if RUN_NOTEBOOK:
    from google.colab import drive
    
    drive.mount("/content/drive")
    
    print("OK: Google Drive mounted")


# %%
# ============================================================================
# 3. IMPORTS
# ============================================================================

if RUN_NOTEBOOK:
    from pathlib import Path
    import json
    
    import geopandas as gpd
    import pandas as pd
    
    from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
    from urban_tree_transfer.config.loader import (
        load_feature_config,
        get_all_feature_names,
        get_metadata_columns,
    )
    from urban_tree_transfer.feature_engineering.extraction import extract_all_features
    from urban_tree_transfer.utils import (
        ExecutionLog,
        generate_validation_report,
        validate_dataset,
    )
    log = ExecutionLog("02a_feature_extraction")
    
    print("OK: Package imports complete")


# %%
if RUN_NOTEBOOK:
    # ============================================================================
    # 4. CONFIGURATION
    # ============================================================================
    
    # Large data files → Google Drive (not in repo)
    DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
    
    # Phase 2 data directories (consistent with Phase 1 structure)
    INPUT_DIR = DRIVE_DIR / "data" / "phase_1_processing"
    OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
    
    # Metadata, logs, figures inside data directory (like Phase 1)
    METADATA_DIR = OUTPUT_DIR / "metadata"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    # Cities to process (must match city column values and S2 file naming)
    CITIES = ["berlin", "leipzig"]
    
    # Create directories
    for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load feature configuration (static defaults from YAML)
    feature_config = load_feature_config()
    temporal_cfg = feature_config.get("temporal", {})
    EXTRACTION_MONTHS = temporal_cfg.get("extraction_months", list(range(1, 13)))
    REFERENCE_YEAR = int(temporal_cfg.get("reference_year", temporal_cfg.get("year", 2021)))
    
    print(f"Input (Phase 1):   {INPUT_DIR}")
    print(f"Output (Phase 2a): {OUTPUT_DIR}")
    print(f"Metadata:          {METADATA_DIR}")
    print(f"Logs:              {LOGS_DIR}")
    print(f"Cities:            {CITIES}")
    print(f"Random seed:       {RANDOM_SEED}")
    print(f"Reference year:    {REFERENCE_YEAR}")
    print(f"Months:            {EXTRACTION_MONTHS}")

# %% [markdown]
# # Feature Extraction
#
# Extract CHM and Sentinel-2 features for each city.
#

# %%
if RUN_NOTEBOOK:
    # ============================================================================
    # 5. FEATURE EXTRACTION (CHM + SENTINEL-2)
    # ============================================================================
    
    log.start_step("Feature Extraction")
    
    try:
        trees_path = INPUT_DIR / "trees" / "trees_filtered_viable.gpkg"
        if not trees_path.exists():
            raise FileNotFoundError(f"Missing input trees: {trees_path}")
    
        trees_all = gpd.read_file(trees_path)
        base_validation = validate_dataset(trees_all)
        print(f"Loaded trees: {len(trees_all):,}")
        print(f"Validation: CRS={base_validation['crs']['valid']}")
    
        metadata_cols = get_metadata_columns(feature_config)
        expected_features = get_all_feature_names(
            months=EXTRACTION_MONTHS,
            config=feature_config,
        )
    
        results = {}
        output_paths = {}
    
        for city in CITIES:
            print("=" * 60)
            print(f"Processing {city.upper()}")
            print("=" * 60)
    
            city_trees = trees_all[trees_all["city"] == city].copy()
            if city_trees.empty:
                print(f"No trees found for {city}. skipping.")
                results[city] = {"status": "skipped", "records": 0}
                continue
    
            output_path = OUTPUT_DIR / f"trees_with_features_{city}.gpkg"
            output_paths[city] = output_path
    
            if output_path.exists():
                print(f"Found existing output (skipping): {output_path}")
                gdf = gpd.read_file(output_path)
                expected_cols = metadata_cols + expected_features + ["geometry"]
                validation = validate_dataset(gdf, expected_columns=expected_cols)
                schema_valid = validation.get("schema", {}).get("valid", True)
                print(f"Validation: CRS={validation['crs']['valid']}, Schema={schema_valid}")
                results[city] = {
                    "status": "skipped",
                    "records": int(len(gdf)),
                    "output_path": str(output_path),
                }
                continue
    
            # CHM files use lowercase city names
            chm_path = INPUT_DIR / "chm" / f"CHM_1m_{city.lower()}.tif"
            if not chm_path.exists():
                raise FileNotFoundError(f"Missing CHM raster: {chm_path}")
    
            sentinel_dir = INPUT_DIR / "sentinel2"
            if not sentinel_dir.exists():
                raise FileNotFoundError(f"Missing Sentinel-2 directory: {sentinel_dir}")
    
            trees_with_features, summary = extract_all_features(
                trees_gdf=city_trees,
                chm_path=chm_path,
                sentinel_dir=sentinel_dir,
                city=city,
                feature_config=feature_config,
            )
    
            trees_with_features.to_file(output_path, driver="GPKG")
            print(f"Saved: {output_path}")
    
            expected_cols = metadata_cols + expected_features + ["geometry"]
            validation = validate_dataset(trees_with_features, expected_columns=expected_cols)
            schema_valid = validation.get("schema", {}).get("valid", True)
            print(f"Validation: CRS={validation['crs']['valid']}, Schema={schema_valid}")
    
            results[city] = {
                "status": "processed",
                "records": int(len(trees_with_features)),
                "output_path": str(output_path),
                "summary": summary,
            }
    
        total_records = sum(result.get("records", 0) for result in results.values())
        log.end_step(status="success", records=int(total_records))
    
    except Exception as e:
        log.end_step(status="error", errors=[str(e)])
        raise


# %% [markdown]
# # Summary & Validation
#
# Validate outputs, save metadata JSON, and write execution log.
#

# %%
if RUN_NOTEBOOK:
    # ============================================================================
    # SUMMARY
    # ============================================================================
    
    print("Generating validation report...")
    
    datasets_to_validate = {}
    for city in CITIES:
        output_path = OUTPUT_DIR / f"trees_with_features_{city}.gpkg"
        if output_path.exists():
            datasets_to_validate[f"trees_with_features_{city}"] = gpd.read_file(output_path)
    
    validation_report = generate_validation_report(datasets_to_validate)
    print(f"Validation summary: {validation_report['summary']}")
    
    validation_path = METADATA_DIR / "feature_extraction_validation.json"
    validation_path.write_text(
        json.dumps(validation_report, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Validation report saved: {validation_path}")
    
    # Save execution log
    log.summary()
    log_path = LOGS_DIR / f"{log.notebook}_execution.json"
    log.save(log_path)
    print(f"Execution log saved: {log_path}")
    
    # ============================================================================
    # DETAILED OUTPUT SUMMARY
    # ============================================================================
    print("=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    
    metadata_cols = get_metadata_columns(feature_config)
    expected_features = get_all_feature_names(
        months=EXTRACTION_MONTHS,
        config=feature_config,
    )
    
    summary_payload = {
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "reference_year": REFERENCE_YEAR,
        "extraction_months": EXTRACTION_MONTHS,
        "feature_count": len(expected_features),
        "metadata_columns": metadata_cols,
        "cities": results,
    }
    
    summary_path = METADATA_DIR / "feature_extraction_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Summary saved: {summary_path}")
    
    for city in CITIES:
        output_path = OUTPUT_DIR / f"trees_with_features_{city}.gpkg"
        if not output_path.exists():
            print(f"{city.upper()}: output missing")
            continue
    
        gdf = gpd.read_file(output_path)
        print(f"--- {city.upper()} ---")
        print(f"Total trees: {len(gdf):,}")
        print(f"CRS: {gdf.crs}")
    
        feature_cols = [col for col in expected_features if col in gdf.columns]
        print(f"Feature columns: {len(feature_cols)} (expected {len(expected_features)})")
    
        missing_features = [col for col in expected_features if col not in gdf.columns]
        if missing_features:
            print(f"Missing features: {len(missing_features)}")
    
        chm_nan = int(gdf["CHM_1m"].isna().sum()) if "CHM_1m" in gdf.columns else -1
        print(f"CHM_1m NaN count: {chm_nan}")
    
        if "position_corrected" in gdf.columns:
            corrected_pct = float(gdf["position_corrected"].mean() * 100.0)
            print(f"Position corrected: {corrected_pct:.2f}%")
    
        expected_cols = metadata_cols + expected_features + ["geometry"]
        validation = validate_dataset(gdf, expected_columns=expected_cols)
        schema_valid = validation.get("schema", {}).get("valid", True)
        print(f"Schema valid: {schema_valid}")
        print(f"CRS valid (EPSG:25833): {validation['crs']['valid']}")
    
    print("=" * 60)
    print("OK: NOTEBOOK COMPLETE")
    print("=" * 60)

