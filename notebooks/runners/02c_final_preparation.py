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

# %%
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================

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

    repo_url = f"git+https://{token}@github.com/SilasPignotti/urban-tree-transfer.git"
    subprocess.run(["pip", "install", repo_url, "-q"], check=True)
    print("OK: Package installed and imports complete")
else:
    print("Notebook test mode or non-Colab environment: skipping repo install.")


# %%
if RUN_NOTEBOOK:
    pass


# %%
if RUN_NOTEBOOK:
    from google.colab import drive
    
    drive.mount("/content/drive")
    print("Google Drive mounted")


# %%
if RUN_NOTEBOOK:
    from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
    from urban_tree_transfer.feature_engineering.selection import remove_redundant_features
    from urban_tree_transfer.feature_engineering.outliers import (
        detect_zscore_outliers,
        detect_mahalanobis_outliers,
        detect_iqr_outliers,
        apply_consensus_outlier_filter,
    )
    from urban_tree_transfer.feature_engineering.splits import (
        create_spatial_blocks,
        create_stratified_splits_berlin,
        create_stratified_splits_leipzig,
        validate_split_stratification,
    )
    from urban_tree_transfer.feature_engineering.proximity import (
        apply_proximity_filter,
        analyze_genus_specific_impact,
    )
    from urban_tree_transfer.utils import ExecutionLog
    
    from pathlib import Path
    from datetime import datetime, timezone
    import json
    
    import geopandas as gpd
    import pandas as pd
    import numpy as np
    
    log = ExecutionLog("02c_final_preparation")
    
    print("OK: Package imports complete")


# %%
if RUN_NOTEBOOK:
    DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
    INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
    OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_splits"
    METADATA_DIR = DRIVE_DIR / "data" / "phase_2_features" / "metadata"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"Input (Phase 2):  {INPUT_DIR}")
    print(f"Output (Splits):  {OUTPUT_DIR}")
    print(f"Metadata (JSON):  {METADATA_DIR}")
    print(f"Logs (Drive):     {LOGS_DIR}")
    print(f"Random seed:      {RANDOM_SEED}")


# %%
if RUN_NOTEBOOK:
    log.start_step("Data Loading")
    
    city_data = {}
    for city in ["berlin", "leipzig"]:
        path = INPUT_DIR / f"trees_clean_{city}.gpkg"
        print(f"Loading: {path}")
        gdf = gpd.read_file(path)
    
        if gdf.crs is None or gdf.crs.to_string() != PROJECT_CRS:
            raise ValueError(f"Expected CRS {PROJECT_CRS}, got {gdf.crs}")
    
        if "city" not in gdf.columns:
            gdf["city"] = city
    
        print(f"  {city.title()}: {len(gdf):,} trees")
        city_data[city] = gdf
    
    log.end_step(status="success", records=sum(len(gdf) for gdf in city_data.values()))

# %%
if RUN_NOTEBOOK:
    log.start_step("Load Configurations")
    
    # Define all required exploratory JSON configs
    required_jsons = [
        ("correlation_removal.json", "exp_03_correlation_analysis.ipynb"),
        ("outlier_thresholds.json", "exp_04_outlier_thresholds.ipynb"),
        ("spatial_autocorrelation.json", "exp_05_spatial_autocorrelation.ipynb"),
        ("proximity_filter.json", "exp_06_mixed_genus_proximity.ipynb"),
    ]
    
    # Check for missing JSONs
    missing_jsons = []
    for json_file, notebook in required_jsons:
        path = METADATA_DIR / json_file
        if not path.exists():
            missing_jsons.append((json_file, notebook))
    
    if missing_jsons:
        error_msg = "❌ Missing configuration files:\n"
        for json_file, notebook in missing_jsons:
            error_msg += f"   - {json_file} (run exploratory/{notebook})\n"
        error_msg += f"\nExpected location: {METADATA_DIR}/"
        raise FileNotFoundError(error_msg)
    
    # Load all configs
    configs = {}
    for json_file, _ in required_jsons:
        path = METADATA_DIR / json_file
        configs[json_file.replace(".json", "")] = json.loads(path.read_text(encoding="utf-8"))
        print(f"✓ Loaded: {json_file}")
    
    # Extract parameters
    features_to_remove = configs["correlation_removal"]["temporal_removal"]["removed_temporal_features"]
    z_threshold = configs["outlier_thresholds"]["zscore"]["threshold"]
    z_min_features = configs["outlier_thresholds"]["zscore"]["min_feature_count"]
    mahal_alpha = configs["outlier_thresholds"]["mahalanobis"]["alpha"]
    iqr_multiplier = configs["outlier_thresholds"]["iqr"]["multiplier"]
    block_size_m = configs["spatial_autocorrelation"]["recommended_block_size_m"]
    proximity_threshold_m = configs["proximity_filter"]["recommended_threshold_m"]
    
    print("\nParameters:")
    print(f"  Features to remove: {len(features_to_remove)}")
    print(f"  Z-score threshold: {z_threshold}")
    print(f"  Mahalanobis alpha: {mahal_alpha}")
    print(f"  IQR multiplier: {iqr_multiplier}")
    print(f"  Block size: {block_size_m}m")
    print(f"  Proximity threshold: {proximity_threshold_m}m")
    
    log.end_step(status="success")


# %%
if RUN_NOTEBOOK:
    log.start_step("Remove Redundant Features")
    
    for city, gdf in city_data.items():
        original_cols = len(gdf.columns)
        city_data[city] = remove_redundant_features(gdf, features_to_remove)
        final_cols = len(city_data[city].columns)
        removed = original_cols - final_cols
        print(f"{city.title()}: Removed {removed} columns ({original_cols} → {final_cols})")
    
    log.end_step(status="success")


# %%
if RUN_NOTEBOOK:
    log.start_step("Outlier Detection")
    
    outlier_stats = {}
    
    for city, gdf in city_data.items():
        print(f"{city.title()}:")
    
        # Select only numeric columns
        numeric_cols = gdf.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude non-feature numeric columns (IDs, metadata, etc.)
        exclude_numeric = {
            "tree_id", "block_id", "plant_year", 
            "correction_distance", "position_corrected"
        }
        
        # Feature columns = numeric columns that are not in exclude list
        feature_cols = [col for col in numeric_cols if col not in exclude_numeric]
        
        print(f"  Total columns: {len(gdf.columns)}")
        print(f"  Numeric columns: {len(numeric_cols)}")
        print(f"  Feature columns for outlier detection: {len(feature_cols)}")
    
        zscore_flags = detect_zscore_outliers(
            gdf,
            feature_cols,
            z_threshold=z_threshold,
            min_feature_count=z_min_features,
        )
        print(f"  Z-score flagged: {zscore_flags.sum()}")
    
        mahal_flags = detect_mahalanobis_outliers(gdf, feature_cols, alpha=mahal_alpha)
        print(f"  Mahalanobis flagged: {mahal_flags.sum()}")
    
        iqr_flags = detect_iqr_outliers(gdf, height_column="CHM_1m", multiplier=iqr_multiplier)
        print(f"  IQR flagged: {iqr_flags.sum()}")
    
        city_data[city], stats = apply_consensus_outlier_filter(
            gdf, zscore_flags, mahal_flags, iqr_flags
        )
    
        assert len(city_data[city]) == len(gdf), "ERROR: Trees were removed! Should only flag."
        assert stats["trees_removed"] == 0, "ERROR: trees_removed should be 0!"
    
        outlier_stats[city] = stats
        print(
            f"  Severity: high={stats['high_count']}, medium={stats['medium_count']}, "
            f"low={stats['low_count']}, none={stats['none_count']}"
        )
    
    log.end_step(status="success")

# %%
if RUN_NOTEBOOK:
    log.start_step("Create Spatial Blocks")
    
    for city, gdf in city_data.items():
        city_data[city] = create_spatial_blocks(gdf, block_size_m=block_size_m)
        n_blocks = city_data[city]["block_id"].nunique()
        print(f"{city.title()}: {n_blocks} blocks created (block_size={block_size_m}m)")
    
    log.end_step(status="success")


# %%
if RUN_NOTEBOOK:
    log.start_step("Create Baseline Splits")
    
    baseline_splits = {}
    
    berlin_train, berlin_val, berlin_test = create_stratified_splits_berlin(
        city_data["berlin"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=RANDOM_SEED,
    )
    
    print("Berlin baseline splits:")
    print(f"  Train: {len(berlin_train):,} trees, {berlin_train['block_id'].nunique()} blocks")
    print(f"  Val:   {len(berlin_val):,} trees, {berlin_val['block_id'].nunique()} blocks")
    print(f"  Test:  {len(berlin_test):,} trees, {berlin_test['block_id'].nunique()} blocks")
    
    baseline_splits["berlin"] = {
        "train": berlin_train,
        "val": berlin_val,
        "test": berlin_test,
    }
    
    leipzig_finetune, leipzig_test = create_stratified_splits_leipzig(
        city_data["leipzig"],
        finetune_ratio=0.80,
        test_ratio=0.20,
        random_seed=RANDOM_SEED,
    )
    
    print("Leipzig baseline splits:")
    print(
        f"  Finetune: {len(leipzig_finetune):,} trees, "
        f"{leipzig_finetune['block_id'].nunique()} blocks"
    )
    print(f"  Test:     {len(leipzig_test):,} trees, {leipzig_test['block_id'].nunique()} blocks")
    
    baseline_splits["leipzig"] = {
        "finetune": leipzig_finetune,
        "test": leipzig_test,
    }
    
    log.end_step(status="success")

# %%
if RUN_NOTEBOOK:
    log.start_step("Validate Baseline Splits")
    
    berlin_validation = validate_split_stratification(
        berlin_train, berlin_val, berlin_test,
        split_names=["train", "val", "test"],
    )
    
    print("Berlin validation:")
    print(f"  Spatial overlap: {berlin_validation['spatial_overlap']} blocks (must be 0)")
    print(f"  KL-divergence (train vs val): {berlin_validation['kl_divergences']['train_vs_val']:.6f}")
    print(f"  KL-divergence (train vs test): {berlin_validation['kl_divergences']['train_vs_test']:.6f}")
    print(f"  KL-divergence (val vs test): {berlin_validation['kl_divergences']['val_vs_test']:.6f}")
    
    assert berlin_validation["spatial_overlap"] == 0, "Berlin: Spatial overlap detected!"
    
    leipzig_validation = validate_split_stratification(
        leipzig_finetune, leipzig_test,
        split_names=["finetune", "test"],
    )
    
    print("Leipzig validation:")
    print(f"  Spatial overlap: {leipzig_validation['spatial_overlap']} blocks (must be 0)")
    print(
        f"  KL-divergence (finetune vs test): "
        f"{leipzig_validation['kl_divergences']['finetune_vs_test']:.6f}"
    )
    
    assert leipzig_validation["spatial_overlap"] == 0, "Leipzig: Spatial overlap detected!"
    
    log.end_step(status="success")


# %%
if RUN_NOTEBOOK:
    log.start_step("Create Filtered Splits")
    
    filtered_data = {}
    filter_stats = {}
    
    for city, gdf in city_data.items():
        filtered_gdf, stats = apply_proximity_filter(gdf, threshold_m=proximity_threshold_m)
        filtered_data[city] = filtered_gdf
        filter_stats[city] = stats
    
        print(f"{city.title()} filtered:")
        print(f"  Original: {stats['original_count']:,} trees")
        print(f"  Removed: {stats['removed_count']:,} trees")
        print(f"  Retained: {stats['retained_count']:,} trees ({stats['retention_rate']*100:.2f}%)")
    
    filtered_splits = {}
    
    berlin_train_filt, berlin_val_filt, berlin_test_filt = create_stratified_splits_berlin(
        filtered_data["berlin"],
        train_ratio=0.70,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=RANDOM_SEED,
    )
    
    print("Berlin filtered splits:")
    print(f"  Train: {len(berlin_train_filt):,} trees")
    print(f"  Val:   {len(berlin_val_filt):,} trees")
    print(f"  Test:  {len(berlin_test_filt):,} trees")
    
    filtered_splits["berlin"] = {
        "train": berlin_train_filt,
        "val": berlin_val_filt,
        "test": berlin_test_filt,
    }
    
    leipzig_finetune_filt, leipzig_test_filt = create_stratified_splits_leipzig(
        filtered_data["leipzig"],
        finetune_ratio=0.80,
        test_ratio=0.20,
        random_seed=RANDOM_SEED,
    )
    
    print("Leipzig filtered splits:")
    print(f"  Finetune: {len(leipzig_finetune_filt):,} trees")
    print(f"  Test:     {len(leipzig_test_filt):,} trees")
    
    filtered_splits["leipzig"] = {
        "finetune": leipzig_finetune_filt,
        "test": leipzig_test_filt,
    }
    
    log.end_step(status="success")

# %%
if RUN_NOTEBOOK:
    log.start_step("Validate Filtered Splits")
    
    berlin_validation_filt = validate_split_stratification(
        berlin_train_filt, berlin_val_filt, berlin_test_filt,
        split_names=["train", "val", "test"],
    )
    
    print("Berlin filtered validation:")
    print(f"  Spatial overlap: {berlin_validation_filt['spatial_overlap']} blocks")
    print(
        f"  KL-divergence (train vs val): "
        f"{berlin_validation_filt['kl_divergences']['train_vs_val']:.6f}"
    )
    
    assert berlin_validation_filt["spatial_overlap"] == 0, "Berlin filtered: Spatial overlap!"
    
    leipzig_validation_filt = validate_split_stratification(
        leipzig_finetune_filt, leipzig_test_filt,
        split_names=["finetune", "test"],
    )
    
    print("Leipzig filtered validation:")
    print(f"  Spatial overlap: {leipzig_validation_filt['spatial_overlap']} blocks")
    print(
        f"  KL-divergence (finetune vs test): "
        f"{leipzig_validation_filt['kl_divergences']['finetune_vs_test']:.6f}"
    )
    
    assert leipzig_validation_filt["spatial_overlap"] == 0, "Leipzig filtered: Spatial overlap!"
    
    log.end_step(status="success")


# %%
if RUN_NOTEBOOK:
    log.start_step("Export Datasets")
    
    # Drop height_m from all splits (redundant with CHM features)
    for city, splits in baseline_splits.items():
        for split_name, split_gdf in splits.items():
            if "height_m" in split_gdf.columns:
                baseline_splits[city][split_name] = split_gdf.drop(columns=["height_m"])
    
    for city, splits in filtered_splits.items():
        for split_name, split_gdf in splits.items():
            if "height_m" in split_gdf.columns:
                filtered_splits[city][split_name] = split_gdf.drop(columns=["height_m"])
    
    print("Dropped height_m from all splits (redundant with CHM features)")
    
    # Export baseline GeoPackages
    for city, splits in baseline_splits.items():
        for split_name, split_gdf in splits.items():
            filename = f"{city}_{split_name}.gpkg"
            path = OUTPUT_DIR / filename
            if path.exists():
                print(f"Found existing output: {filename} (skipping)")
                continue
            split_gdf.to_file(path, driver="GPKG")
            print(f"Saved baseline: {filename} ({len(split_gdf):,} trees)")
    
    # Export filtered GeoPackages
    for city, splits in filtered_splits.items():
        for split_name, split_gdf in splits.items():
            filename = f"{city}_{split_name}_filtered.gpkg"
            path = OUTPUT_DIR / filename
            if path.exists():
                print(f"Found existing output: {filename} (skipping)")
                continue
            split_gdf.to_file(path, driver="GPKG")
            print(f"Saved filtered: {filename} ({len(split_gdf):,} trees)")
    
    log.end_step(status="success")

# %%
if RUN_NOTEBOOK:
    log.start_step("Export Summary Metadata")
    
    summary = {
        "version": "1.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "random_seed": RANDOM_SEED,
        "configurations": {
            "block_size_m": block_size_m,
            "proximity_threshold_m": proximity_threshold_m,
            "z_threshold": z_threshold,
            "mahalanobis_alpha": mahal_alpha,
            "iqr_multiplier": iqr_multiplier,
            "features_removed": len(features_to_remove),
        },
        "outlier_stats": outlier_stats,
        "proximity_filter_stats": filter_stats,
        "baseline_splits": {
            "berlin": {
                "train": {"count": len(berlin_train), "blocks": int(berlin_train["block_id"].nunique())},
                "val": {"count": len(berlin_val), "blocks": int(berlin_val["block_id"].nunique())},
                "test": {"count": len(berlin_test), "blocks": int(berlin_test["block_id"].nunique())},
            },
            "leipzig": {
                "finetune": {
                    "count": len(leipzig_finetune),
                    "blocks": int(leipzig_finetune["block_id"].nunique()),
                },
                "test": {"count": len(leipzig_test), "blocks": int(leipzig_test["block_id"].nunique())},
            },
        },
        "filtered_splits": {
            "berlin": {
                "train": {
                    "count": len(berlin_train_filt),
                    "blocks": int(berlin_train_filt["block_id"].nunique()),
                },
                "val": {"count": len(berlin_val_filt), "blocks": int(berlin_val_filt["block_id"].nunique())},
                "test": {
                    "count": len(berlin_test_filt),
                    "blocks": int(berlin_test_filt["block_id"].nunique()),
                },
            },
            "leipzig": {
                "finetune": {
                    "count": len(leipzig_finetune_filt),
                    "blocks": int(leipzig_finetune_filt["block_id"].nunique()),
                },
                "test": {"count": len(leipzig_test_filt), "blocks": int(leipzig_test_filt["block_id"].nunique())},
            },
        },
        "validation": {
            "berlin_baseline": {
                "spatial_overlap": berlin_validation["spatial_overlap"],
                "kl_divergences": berlin_validation["kl_divergences"],
            },
            "berlin_filtered": {
                "spatial_overlap": berlin_validation_filt["spatial_overlap"],
                "kl_divergences": berlin_validation_filt["kl_divergences"],
            },
            "leipzig_baseline": {
                "spatial_overlap": leipzig_validation["spatial_overlap"],
                "kl_divergences": leipzig_validation["kl_divergences"],
            },
            "leipzig_filtered": {
                "spatial_overlap": leipzig_validation_filt["spatial_overlap"],
                "kl_divergences": leipzig_validation_filt["kl_divergences"],
            },
        },
    }
    
    summary_path = METADATA_DIR / "phase_2_final_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {summary_path}")
    
    log.end_step(status="success")

# %%
# ==============================================================================
# PARQUET EXPORT (ML-optimized format for Phase 3)
# ==============================================================================
if RUN_NOTEBOOK:
    from urban_tree_transfer.feature_engineering.selection import (
        export_splits_to_parquet,
        export_geometry_lookup,
    )
    
    log.start_step("Export Parquet Files")
    
    # Collect all splits (baseline + filtered)
    all_splits = {}
    for city, splits in baseline_splits.items():
        for split_name, split_gdf in splits.items():
            all_splits[f"{city}_{split_name}"] = split_gdf
    for city, splits in filtered_splits.items():
        for split_name, split_gdf in splits.items():
            all_splits[f"{city}_{split_name}_filtered"] = split_gdf
    
    # Export Parquet files
    parquet_paths = export_splits_to_parquet(all_splits, OUTPUT_DIR)
    print(f"Exported {len(parquet_paths)} Parquet files")
    
    # Export geometry lookup
    n_trees = export_geometry_lookup(all_splits, OUTPUT_DIR / "geometry_lookup.parquet")
    print(f"Geometry lookup: {n_trees} unique trees")
    
    # Validation: verify Parquet matches GeoPackage content
    print("\\nValidating Parquet exports...")
    for name, gdf in all_splits.items():
        pq = pd.read_parquet(OUTPUT_DIR / f"{name}.parquet")
        
        # Row count must match
        assert len(pq) == len(gdf), f"{name}: row count mismatch"
        
        # Geometry must NOT be present (moved to geometry_lookup.parquet)
        assert "geometry" not in pq.columns, f"{name}: geometry column found in Parquet"
        
        # height_m should not be in either (already dropped before GeoPackage export)
        assert "height_m" not in gdf.columns, f"{name}: height_m still in GeoDataFrame (should be dropped)"
        assert "height_m" not in pq.columns, f"{name}: height_m in Parquet (should be dropped)"
        
        # Analysis metadata must be kept
        expected_metadata = ["plant_year", "position_corrected", "correction_distance",
                            "species_latin", "genus_german", "species_german", "tree_type"]
        for col in expected_metadata:
            if col in gdf.columns:
                assert col in pq.columns, f"{name}: {col} missing (needed for analysis)"
        
        # All other columns from GeoPackage must be in Parquet (except geometry)
        feature_cols = [c for c in gdf.columns if c != "geometry"]
        missing_cols = [c for c in feature_cols if c not in pq.columns]
        assert not missing_cols, f"{name}: missing columns in Parquet: {missing_cols}"
    
    print("✅ All Parquet files validated")
    
    # Validate geometry lookup
    lookup = pd.read_parquet(OUTPUT_DIR / "geometry_lookup.parquet")
    assert set(lookup.columns) == {"tree_id", "city", "split", "filtered", "x", "y"}
    for name, gdf in all_splits.items():
        split_ids = set(gdf["tree_id"])
        lookup_ids = set(lookup[lookup["split"] == name]["tree_id"])
        assert split_ids == lookup_ids, f"{name}: tree_id mismatch in geometry lookup"
    
    print("✅ Geometry lookup validated")
    
    log.end_step()

# %%
if RUN_NOTEBOOK:
    log.summary()
    log_path = LOGS_DIR / f"{log.notebook}_execution.json"
    log.save(log_path)
    print(f"Execution log saved: {log_path}")
    
    print("=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    
    print("--- BASELINE GEOPACKAGES (5) ---")
    for city, splits in baseline_splits.items():
        for split_name, split_gdf in splits.items():
            print(f"  {city}_{split_name}.gpkg ({len(split_gdf):,} trees)")
    
    print("--- FILTERED GEOPACKAGES (5) ---")
    for city, splits in filtered_splits.items():
        for split_name, split_gdf in splits.items():
            print(f"  {city}_{split_name}_filtered.gpkg ({len(split_gdf):,} trees)")
    
    print("Total: 10 GeoPackages")
    print("Note: height_m removed (redundant with CHM features)")
    
    print("\\n--- PARQUET FILES (10 + 1 geometry lookup) ---")
    print("ML-optimized format for Phase 3:")
    print("  Same schema as GeoPackages (geometry → geometry_lookup.parquet)")
    for city, splits in baseline_splits.items():
        for split_name in splits.keys():
            print(f"  {city}_{split_name}.parquet")
    for city, splits in filtered_splits.items():
        for split_name in splits.keys():
            print(f"  {city}_{split_name}_filtered.parquet")
    print("  geometry_lookup.parquet (x/y coordinates for visualization)")
    print("Total: 10 Parquet + 1 geometry lookup")
    
    print("=" * 60)
    print("⚠️  MANUAL SYNC REQUIRED")
    print("=" * 60)
    print("Download from Google Drive and commit to Git:")
    print("1. GeoPackages: data/phase_2_splits/*.gpkg")
    print("2. Parquet files: data/phase_2_splits/*.parquet")
    print("3. Summary: outputs/phase_2/metadata/phase_2_final_summary.json")
    print("Git commands:")
    print("  git add data/phase_2_splits/*.gpkg data/phase_2_splits/*.parquet outputs/phase_2/metadata/phase_2_final_summary.json")
    print("  git commit -m 'Add Phase 2 final splits (baseline + filtered) with Parquet export'")
    print("  git push")
    
    print("=" * 60)
    print("✅ PHASE 2 COMPLETE - Ready for Phase 3")
    print("=" * 60)
