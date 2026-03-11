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
    # Mount Google Drive for data files
    from google.colab import drive
    
    drive.mount("/content/drive")
    
    print("Google Drive mounted")

# %%
if RUN_NOTEBOOK:
    # Standard library imports
    from pathlib import Path
    import json
    from datetime import datetime, timezone
    import time
    
    import geopandas as gpd
    import pandas as pd
    
    # Package imports
    from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
    from urban_tree_transfer.config.loader import (
        get_all_feature_names,
        get_all_s2_features,
        get_metadata_columns,
        get_temporal_feature_names,
        load_feature_config,
    )
    from urban_tree_transfer.feature_engineering import (
        analyze_nan_distribution,
        apply_temporal_selection,
        compute_chm_engineered_features,
        filter_by_plant_year,
        filter_deciduous_genera,
        filter_nan_trees,
        filter_ndvi_plausibility,
        interpolate_features_within_tree,
    )
    from urban_tree_transfer.utils import (
        ExecutionLog,
        generate_validation_report,
        validate_dataset,
    )
    log = ExecutionLog("02b_data_quality")
    
    print("OK: Package imports complete")


# %%
if RUN_NOTEBOOK:
    # ============================================================
    # CONFIGURATION
    # ============================================================
    
    # Large data files → Google Drive (not in repo)
    DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
    
    
    # Phase 2 data directories
    INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
    OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
    
    # Metadata & logs → Google Drive (download manually and commit to repo)
    METADATA_DIR = INPUT_DIR / "metadata"
    LOGS_DIR = INPUT_DIR / "logs"
    
    # Cities to process (must match city column values)
    CITIES = ["berlin", "leipzig"]
    
    # Create directories
    for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load feature configuration (static defaults from YAML)
    feature_config = load_feature_config()
    metadata_cols = get_metadata_columns(feature_config)
    
    temporal_cfg = feature_config.get("temporal", {})
    EXTRACTION_MONTHS = temporal_cfg.get("extraction_months", list(range(1, 13)))
    REFERENCE_YEAR = int(temporal_cfg.get("reference_year", temporal_cfg.get("year", 2021)))
    
    quality_cfg = feature_config.get("quality", {})
    nan_cfg = quality_cfg.get("nan_handling", {})
    ndvi_cfg = quality_cfg.get("ndvi_plausibility", {})
    
    MAX_NAN_MONTHS = int(nan_cfg.get("max_nan_months", 2))
    MAX_EDGE_NAN_MONTHS = int(nan_cfg.get("max_edge_nan_months", 1))
    NDVI_MIN_THRESHOLD = float(ndvi_cfg.get("min_max_ndvi", 0.3))
    
    # Load exploratory configs (must exist)
    temporal_path = METADATA_DIR / "temporal_selection.json"
    chm_path = METADATA_DIR / "chm_assessment.json"
    
    if not temporal_path.exists():
        raise FileNotFoundError(
            "Config not found. Run exp_01_temporal_analysis.ipynb first to generate "
            "temporal_selection.json"
        )
    
    if not chm_path.exists():
        raise FileNotFoundError(
            "Config not found. Run exp_02_chm_assessment.ipynb first to generate "
            "chm_assessment.json"
        )
    
    temporal_config = json.loads(temporal_path.read_text(encoding="utf-8"))
    chm_assessment = json.loads(chm_path.read_text(encoding="utf-8"))
    
    selected_months = temporal_config.get("selected_months")
    if not selected_months:
        raise ValueError("temporal_selection.json missing 'selected_months'")
    
    recommended_max_plant_year = chm_assessment.get("plant_year_analysis", {}).get("recommended_max_plant_year")
    if recommended_max_plant_year is None:
        raise ValueError("chm_assessment.json missing 'plant_year_analysis.recommended_max_plant_year'")
    
    analysis_scope = (
        chm_assessment.get("genus_inventory", {}).get("analysis_scope", "all_genera")
    )
    
    # Deciduous genera list (from feature config)
    deciduous_genera = (
        feature_config.get("genus_classification", {}).get("deciduous", [])
    )
    
    print(f"Input (Phase 2):    {INPUT_DIR}")
    print(f"Output (Phase 2):   {OUTPUT_DIR}")
    print(f"Metadata (Drive):   {METADATA_DIR}")
    print(f"Logs (Drive):       {LOGS_DIR}")
    print(f"Cities:             {CITIES}")
    print(f"Random seed:        {RANDOM_SEED}")
    print(f"Reference year:     {REFERENCE_YEAR}")
    print(f"Selected months:    {selected_months}")
    print(f"Analysis scope:     {analysis_scope}")
    print(f"Max plant year:     {recommended_max_plant_year}")
    print(f"Max NaN months:     {MAX_NAN_MONTHS}")
    print(f"Max edge NaN months:{MAX_EDGE_NAN_MONTHS}")
    print(f"NDVI min threshold: {NDVI_MIN_THRESHOLD}")

# %% [markdown]
# # Data Quality Pipeline
#
# Run the complete Phase 2b data quality pipeline in the required step order.

# %%
if RUN_NOTEBOOK:
    # ============================================================
    # SECTION: Data Quality Pipeline
    # ============================================================
    
    log.start_step("Data Quality Pipeline")
    
    execution_log_lines = []
    
    def log_exec(city: str, step: str, duration_s: float, status: str, message: str = "") -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        line = f"{timestamp}	{city}	{step}	{duration_s:.2f}	{status}	{message}"
        execution_log_lines.append(line)
    
    
    summary_payload = {
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "reference_year": REFERENCE_YEAR,
        "configs_used": {
            "temporal_selection": str(temporal_path),
            "chm_assessment": str(chm_path),
            "feature_config": str(DRIVE_DIR / "configs" / "features" / "feature_config.yaml"),
        },
        "cities": {},
    }
    
    output_paths = {
        city: OUTPUT_DIR / f"trees_clean_{city}.gpkg" for city in CITIES
    }
    
    
    try:
        for city in CITIES:
            print("=" * 60)
            print(f"PROCESSING {city.upper()}")
            print("=" * 60)
    
            city_steps = {}
    
            input_path = INPUT_DIR / f"trees_with_features_{city}.gpkg"
            output_path = output_paths[city]
    
            if output_path.exists():
                print(f"Found existing output: {output_path} (skipping city)")
                gdf = gpd.read_file(output_path)
                input_trees = None
                if input_path.exists():
                    input_trees = len(gpd.read_file(input_path))
                output_trees = len(gdf)
    
                summary_payload["cities"][city] = {
                    "input_trees": input_trees,
                    "steps": {"skipped": True},
                    "output_trees": output_trees,
                    "retention_rate": None if input_trees is None else output_trees / input_trees,
                    "output_features": len([c for c in gdf.columns if c != "geometry"]),
                    "output_path": str(output_path),
                }
                continue
    
            # --------------------------------------------------------
            # Step 1: Load Input Data
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 1: Load input data")
    
            if not input_path.exists():
                raise FileNotFoundError(f"Missing input file: {input_path}")
    
            trees_gdf = gpd.read_file(input_path)
            input_count = len(trees_gdf)
    
            expected_features = get_temporal_feature_names(
                months=EXTRACTION_MONTHS,
                config=feature_config,
            )
            expected_cols = metadata_cols + ["CHM_1m"] + expected_features + ["geometry"]
            validation = validate_dataset(trees_gdf, expected_columns=expected_cols)
            schema_valid = validation.get("schema", {}).get("valid", True)
            print(f"Loaded trees: {input_count:,}")
            print(f"Validation: CRS={validation['crs']['valid']}, Schema={schema_valid}")
    
            city_steps["01_load_input"] = {
                "input": input_count,
                "removed": 0,
                "retained": input_count,
            }
            log_exec(city, "01_load_input", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 2: Genus Filtering (Deciduous Only)
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 2: Genus filtering")
            before_count = len(trees_gdf)
    
            if analysis_scope == "deciduous_only":
                trees_gdf = filter_deciduous_genera(trees_gdf, deciduous_genera)
            else:
                print("Analysis scope allows all genera; skipping filter.")
    
            after_count = len(trees_gdf)
            city_steps["02_genus_filter"] = {
                "input": before_count,
                "removed": before_count - after_count,
                "retained": after_count,
            }
            log_exec(city, "02_genus_filter", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 3: Plant Year Filtering
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 3: Plant year filtering")
            before_count = len(trees_gdf)
    
            trees_gdf = filter_by_plant_year(
                trees_gdf,
                max_year=int(recommended_max_plant_year),
                reference_year=REFERENCE_YEAR,
            )
    
            after_count = len(trees_gdf)
            city_steps["03_plant_year_filter"] = {
                "input": before_count,
                "removed": before_count - after_count,
                "retained": after_count,
            }
            log_exec(city, "03_plant_year_filter", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 4: Temporal Feature Selection
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 4: Temporal feature selection")
    
            s2_features = get_all_s2_features(feature_config)
            available_s2_cols = [
                col
                for col in trees_gdf.columns
                if any(col.startswith(f"{feature}_") for feature in s2_features)
            ]
            features_before = len(available_s2_cols)
    
            trees_gdf = apply_temporal_selection(
                trees_gdf,
                selected_months=selected_months,
                feature_config=feature_config,
            )
    
            s2_selected_cols = [
                f"{feature}_{month:02d}"
                for feature in s2_features
                for month in selected_months
            ]
            s2_selected_cols = [col for col in s2_selected_cols if col in trees_gdf.columns]
            features_after = len(s2_selected_cols)
    
            city_steps["04_temporal_selection"] = {
                "input": features_before,
                "removed": features_before - features_after,
                "retained": features_after,
            }
            log_exec(city, "04_temporal_selection", time.perf_counter() - step_start, "success")
    
            # Feature columns for steps 5-8 (CHM_1m + selected S2)
            feature_columns = ["CHM_1m"] + s2_selected_cols
            temporal_feature_columns = s2_selected_cols
    
            # --------------------------------------------------------
            # Step 5: NaN Distribution Analysis
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 5: NaN distribution analysis")
    
            nan_stats = analyze_nan_distribution(trees_gdf, feature_columns)
            top10 = nan_stats.head(10).to_dict(orient="records")
    
            print("Top 10 features with highest NaN rates:")
            for row in top10:
                print(
                    f"  {row['feature']}: {row['nan_count']} "
                    f"({row['nan_percentage']:.2f}%)"
                )
    
            nan_stats_path = METADATA_DIR / f"nan_distribution_{city}.json"
            nan_stats_path.write_text(
                json.dumps(nan_stats.to_dict(orient="records"), indent=2),
                encoding="utf-8",
            )
            print(f"NaN stats saved: {nan_stats_path}")
    
            city_steps["05_nan_analysis"] = {
                "input": len(feature_columns),
                "removed": 0,
                "retained": len(feature_columns),
                "top10": top10,
            }
            log_exec(city, "05_nan_analysis", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 6: NaN Tree Filtering
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 6: NaN tree filtering")
            before_count = len(trees_gdf)
    
            trees_gdf = filter_nan_trees(
                trees_gdf,
                feature_columns=feature_columns,
                max_nan_months=MAX_NAN_MONTHS,
            )
    
            after_count = len(trees_gdf)
            city_steps["06_nan_filter"] = {
                "input": before_count,
                "removed": before_count - after_count,
                "retained": after_count,
            }
            log_exec(city, "06_nan_filter", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 7: Within-Tree Temporal Interpolation
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 7: Within-tree temporal interpolation")
    
            nan_before = int(trees_gdf[temporal_feature_columns].isna().sum().sum())
            trees_gdf = interpolate_features_within_tree(
                trees_gdf,
                feature_columns=temporal_feature_columns,
                selected_months=selected_months,
                max_edge_nan_months=MAX_EDGE_NAN_MONTHS,
            )
            nan_after = int(trees_gdf[temporal_feature_columns].isna().sum().sum())
    
            city_steps["07_interpolation"] = {
                "input": nan_before,
                "removed": nan_before - nan_after,
                "retained": nan_after,
            }
            log_exec(city, "07_interpolation", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 8: Validate Zero NaN
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 8: Validate zero NaN")
    
            total_nan = int(trees_gdf[feature_columns].isna().sum().sum())
            print(f"Remaining NaN count: {total_nan}")
            if total_nan != 0:
                raise ValueError(
                    "NaN values remain after interpolation. "
                    "Trees with >=2 edge NaN months should have been removed in Step 6."
                )
    
            city_steps["08_zero_nan_validation"] = {
                "input": total_nan,
                "removed": 0,
                "retained": total_nan,
            }
            log_exec(city, "08_zero_nan_validation", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 9: CHM Feature Engineering
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 9: CHM feature engineering")
    
            trees_gdf = compute_chm_engineered_features(trees_gdf)
            chm_nan = int(trees_gdf[["CHM_1m_zscore", "CHM_1m_percentile"]].isna().sum().sum())
            if chm_nan != 0:
                raise ValueError("CHM engineered features contain NaN values.")
    
            city_steps["09_chm_engineering"] = {
                "input": len(trees_gdf),
                "removed": 0,
                "retained": len(trees_gdf),
            }
            log_exec(city, "09_chm_engineering", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 10: NDVI Plausibility Filtering
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 10: NDVI plausibility filtering")
            before_count = len(trees_gdf)
    
            ndvi_columns = [f"NDVI_{month:02d}" for month in selected_months]
            ndvi_columns = [col for col in ndvi_columns if col in trees_gdf.columns]
    
            trees_gdf = filter_ndvi_plausibility(
                trees_gdf,
                ndvi_columns=ndvi_columns,
                min_threshold=NDVI_MIN_THRESHOLD,
            )
    
            after_count = len(trees_gdf)
            city_steps["10_ndvi_filter"] = {
                "input": before_count,
                "removed": before_count - after_count,
                "retained": after_count,
            }
            log_exec(city, "10_ndvi_filter", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 11: Final Validation
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 11: Final validation")
    
            expected_features_final = get_all_feature_names(
                months=selected_months,
                include_chm_engineered=True,
                config=feature_config,
            )
            expected_cols_final = metadata_cols + expected_features_final + ["geometry"]
            validation = validate_dataset(trees_gdf, expected_columns=expected_cols_final)
            schema_valid = validation.get("schema", {}).get("valid", True)
            print(f"Validation: CRS={validation['crs']['valid']}, Schema={schema_valid}")
    
            final_nan = int(trees_gdf[expected_features_final].isna().sum().sum())
            if final_nan != 0:
                raise ValueError("NaN values found in final feature set.")
    
            retention_rate = len(trees_gdf) / input_count if input_count else 0
            print(f"Retention rate: {retention_rate:.3f}")
            if retention_rate <= 0.85:
                raise ValueError("Retention rate below 0.85 threshold.")
    
            city_steps["11_final_validation"] = {
                "input": len(trees_gdf),
                "removed": 0,
                "retained": len(trees_gdf),
            }
            log_exec(city, "11_final_validation", time.perf_counter() - step_start, "success")
    
            # --------------------------------------------------------
            # Step 12: Save Output
            # --------------------------------------------------------
            step_start = time.perf_counter()
            print("Step 12: Save output")
    
            trees_gdf.to_file(output_path, driver="GPKG")
            print(f"Saved: {output_path}")
    
            city_steps["12_save_output"] = {
                "input": len(trees_gdf),
                "removed": 0,
                "retained": len(trees_gdf),
            }
            log_exec(city, "12_save_output", time.perf_counter() - step_start, "success")
    
            summary_payload["cities"][city] = {
                "input_trees": input_count,
                "steps": city_steps,
                "output_trees": len(trees_gdf),
                "retention_rate": retention_rate,
                "output_features": len(expected_features_final),
                "output_path": str(output_path),
            }
    
        total_records = sum(
            city_data.get("output_trees", 0)
            for city_data in summary_payload["cities"].values()
            if isinstance(city_data, dict)
        )
        log.end_step(status="success", records=int(total_records))
    
    except Exception as e:
        log.end_step(status="error", errors=[str(e)])
        raise

# %% [markdown]
# # Summary & Validation
#
# Validate outputs, save summary JSON, and write execution log.

# %%
if RUN_NOTEBOOK:
    # ============================================================
    # SUMMARY & VALIDATION
    # ============================================================
    
    print("Generating validation report...")
    
    datasets_to_validate = {}
    for city in CITIES:
        output_path = OUTPUT_DIR / f"trees_clean_{city}.gpkg"
        if output_path.exists():
            datasets_to_validate[f"trees_clean_{city}"] = gpd.read_file(output_path)
    
    validation_report = generate_validation_report(datasets_to_validate)
    print(f"Validation summary: {validation_report['summary']}")
    
    validation_path = METADATA_DIR / "data_quality_validation.json"
    validation_path.write_text(
        json.dumps(validation_report, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Validation report saved: {validation_path}")
    
    summary_path = METADATA_DIR / "data_quality_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Summary saved: {summary_path}")
    
    # Save execution log (structured JSON)
    log.summary()
    log_path = LOGS_DIR / f"{log.notebook}_execution.json"
    log.save(log_path)
    print(f"Execution log saved: {log_path}")
    
    # Save execution log (plain .log format)
    log_txt_path = LOGS_DIR / "02b_execution.log"
    log_txt_path.write_text("\n".join(execution_log_lines), encoding="utf-8")
    print(f"Execution log (.log) saved: {log_txt_path}")
    
    # ============================================================
    # DETAILED OUTPUT SUMMARY
    # ============================================================
    print("=" * 60)
    print("OUTPUT SUMMARY")
    print("=" * 60)
    
    for city in CITIES:
        output_path = OUTPUT_DIR / f"trees_clean_{city}.gpkg"
        if not output_path.exists():
            print(f"{city.upper()}: output missing")
            continue
    
        gdf = gpd.read_file(output_path)
        print(f"--- {city.upper()} ---")
        print(f"Total trees: {len(gdf):,}")
        print(f"CRS: {gdf.crs}")
    
        feature_cols = [col for col in gdf.columns if col not in ["geometry"]]
        print(f"Feature columns: {len(feature_cols)}")
    
        total_nan = int(gdf[feature_cols].isna().sum().sum())
        print(f"NaN values (features): {total_nan}")
    
        if "tree_id" in gdf.columns:
            duplicate_ids = gdf["tree_id"].duplicated().sum()
            print(f"Duplicate tree_ids: {int(duplicate_ids)}")
    
    print("=" * 60)
    print("NOTEBOOK COMPLETE")
    print("=" * 60)
