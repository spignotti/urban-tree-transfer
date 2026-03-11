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
#     name: python3
# ---

# %% [markdown]
# # Urban Tree Transfer - Runner Notebook
#
# **Title:** Setup Fixation
#
# **Phase:** 3 - Experiments
#
# **Step:** 03a - Setup Fixation
#
# **Purpose:** Apply finalized setup decisions and export ML-ready Phase 3 experiment datasets.
#
# **Input:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_2_splits`
#
# **Output:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_3_experiments`
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
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================================

import subprocess
from google.colab import userdata

# Get GitHub token from Colab Secrets
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        """GITHUB_TOKEN not found in Colab Secrets.
1. Click the key icon in the left sidebar
2. Add a secret named 'GITHUB_TOKEN' with your GitHub token
3. Toggle 'Notebook access' ON"""
    )

# Install package from GitHub
repo_url = f"git+https://{token}@github.com/silas-workspace/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

print("OK: Package installed")

# %%
# ============================================================================
# 2. GOOGLE DRIVE
# ============================================================================

from google.colab import drive

drive.mount("/content/drive")

print("OK: Google Drive mounted")


# %%
# ============================================================================
# 3. IMPORTS
# ============================================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
import gc

import pandas as pd

from urban_tree_transfer.config import RANDOM_SEED
from urban_tree_transfer.experiments import ablation, data_loading
from urban_tree_transfer.utils import (
    ExecutionLog,
    validate_setup_decisions,
)
warnings.filterwarnings("ignore", category=UserWarning)

log = ExecutionLog("03a_setup_fixation")

print("OK: Imports complete")


# %%
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
DATA_DIR = DRIVE_DIR / "data"
INPUT_DIR = DATA_DIR / "phase_2_splits"
OUTPUT_DIR = DATA_DIR / "phase_3_experiments"

METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create output directories
for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


print(f"Input (Phase 2 Splits):  {INPUT_DIR}")
print(f"Output (Phase 3):        {OUTPUT_DIR}")
print(f"Metadata:                {METADATA_DIR}")
print(f"Logs:                    {LOGS_DIR}")
print(f"Random seed:             {RANDOM_SEED}")
print("OK: Configuration complete")

# %%
# ============================================================================
# 5. LOAD SETUP DECISIONS
# ============================================================================

log.start_step("Load Setup Decisions")

try:
    setup_path = METADATA_DIR / "setup_decisions.json"

    if not setup_path.exists():
        raise FileNotFoundError(
            f"""setup_decisions.json not found at {setup_path}
Run exploratory notebooks exp_08 → exp_08b → exp_08c → exp_09 → exp_10 first."""
        )

    # Validate and load setup decisions
    setup = validate_setup_decisions(setup_path)

    # Check for genus selection (optional - exp_10 may not have run yet)
    has_genus_selection = "genus_selection" in setup

    print("\n" + "=" * 70)
    print("Setup Decisions Loaded")
    print("=" * 70)
    print(f"CHM Strategy:       {setup['chm_strategy']['decision']}")
    print(f"Proximity Strategy: {setup['proximity_strategy']['decision']}")
    print(f"Outlier Strategy:   {setup['outlier_strategy']['decision']}")
    print(f"Selected Features:  {len(setup['selected_features'])}")
    print(f"Genus Selection:    {'Yes' if has_genus_selection else 'No (will be applied later)'}")

    if has_genus_selection:
        genus_config = setup["genus_selection"]
        final_genera_count = len(genus_config["decision"]["final_genera_list"])
        print(f"   Final Genera:    {final_genera_count} classes")
        if genus_config["grouping_analysis"]["n_groups"] > 0:
            print(f"   Grouped Genera:  {genus_config['grouping_analysis']['n_groups']} groups")
    else:
        genus_config = None
        print("\n⚠️  Note: Genus selection not yet configured.")
        print("   Run exp_10_genus_selection_validation.ipynb to add genus filtering.")

    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 6. APPLY DECISIONS TO CREATE FILTERED DATASETS
# ============================================================================

log.start_step("Apply Setup Decisions")

try:
    # Define all city-split combinations
    city_splits = [
        ("berlin", "train"),
        ("berlin", "val"),
        ("berlin", "test"),
        ("leipzig", "finetune"),
        ("leipzig", "test"),
    ]

    results = []

    print("\n" + "=" * 70)
    print("Processing Splits")
    print("=" * 70)

    for city, split in city_splits:
        split_name = f"{city}_{split}"
        print(f"\n[{split_name}]")

        # Apply setup decisions (test split policy enforced automatically)
        df, meta = ablation.prepare_ablation_dataset(
            base_path=INPUT_DIR,
            city=city,
            split=split,
            setup_decisions=setup,
            return_metadata=True,
            optimize_memory=True,
        )

        # Fix missing German names before any filtering
        df = data_loading.fix_missing_genus_german(df)

        # Apply genus selection if available
        genus_filtering_applied = False
        genera_removed = 0
        original_len_post_setup = len(df)

        if genus_config is not None:
            viable_genera = genus_config["validation_results"]["viable_genera"]
            genus_to_final = genus_config["grouping_analysis"]["genus_to_final_mapping"]
            genus_german_mapping = genus_config["grouping_analysis"]["genus_german_mapping"]

            # Filter to viable genera
            df = df[df["genus_latin"].isin(viable_genera)].copy()
            genera_removed = original_len_post_setup - len(df)

            # Apply genus grouping (both columns)
            df["genus_latin"] = df["genus_latin"].map(genus_to_final)
            df["genus_german"] = df["genus_latin"].map(genus_german_mapping)

            if df["genus_german"].isna().any():
                missing = df[df["genus_german"].isna()]["genus_latin"].unique().tolist()
                raise ValueError(f"Missing German names after mapping in {split_name}: {missing}")

            genus_filtering_applied = True

        # Save filtered dataset
        output_path = OUTPUT_DIR / f"{split_name}.parquet"
        df.to_parquet(output_path, index=False)

        # Track results
        results.append(
            {
                "split": split_name,
                "n_samples": len(df),
                "n_features": meta["n_features"],
                "outliers_removed": meta["outliers_removed"],
                "genus_filtering": "Yes" if genus_filtering_applied else "No",
                "genera_removed": genera_removed,
                "test_policy_applied": meta["test_split_policy_applied"],
                "proximity_used": meta["proximity_strategy_used"],
            }
        )

        # Report
        print(f"  Samples: {meta['original_n_samples']:,} → {len(df):,}")
        print(f"  Features: {meta['n_features']}")
        print(f"  Outliers removed: {meta['outliers_removed']}")

        if genus_filtering_applied:
            print(f"  Genus filter: {genera_removed} samples removed")
            print("  genus_german: Mapped")
        else:
            print("  genus_german: Fixed (no genus filter)")

        if meta["test_split_policy_applied"]:
            print("  ⚠️  Test split policy: proximity override baseline")

        print(f"  ✅ Saved: {output_path.name}")

        gc.collect()

        # Memory cleanup
        del df, meta

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(results_df.to_string(index=False))

    log.end_step(status="success", records=len(results))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 7. CREATE CNN1D DATASETS (FULL FEATURES)
# ============================================================================

log.start_step("Create CNN1D Datasets")

try:
    results_cnn = []
    
    print("\n" + "=" * 70)
    print("Processing CNN1D Splits (Full Temporal Features)")
    print("=" * 70)
    
    for city, split in city_splits:
        split_name = f"{city}_{split}_cnn"
        print(f"\n[{split_name}]")
        
        # Apply setup decisions but keep ALL features (skip feature selection)
        df_cnn, meta_cnn = ablation.prepare_ablation_dataset(
            base_path=INPUT_DIR,
            city=city,
            split=split,
            setup_decisions=setup,
            return_metadata=True,
            optimize_memory=True,
            skip_feature_selection=True,  # CRITICAL: Keep full temporal sequences
        )
        
        # Fix missing German names
        df_cnn = data_loading.fix_missing_genus_german(df_cnn)
        
        # Apply genus selection if available (same logic as XGBoost)
        genus_filtering_applied = False
        genera_removed = 0
        original_len_post_setup = len(df_cnn)
        
        if genus_config is not None:
            viable_genera = genus_config["validation_results"]["viable_genera"]
            genus_to_final = genus_config["grouping_analysis"]["genus_to_final_mapping"]
            genus_german_mapping = genus_config["grouping_analysis"]["genus_german_mapping"]
            
            # Filter to viable genera
            df_cnn = df_cnn[df_cnn["genus_latin"].isin(viable_genera)].copy()
            genera_removed = original_len_post_setup - len(df_cnn)
            
            # Apply genus grouping (both columns)
            df_cnn["genus_latin"] = df_cnn["genus_latin"].map(genus_to_final)
            df_cnn["genus_german"] = df_cnn["genus_latin"].map(genus_german_mapping)
            
            if df_cnn["genus_german"].isna().any():
                missing = df_cnn[df_cnn["genus_german"].isna()]["genus_latin"].unique().tolist()
                raise ValueError(f"Missing German names after mapping in {split_name}: {missing}")
            
            genus_filtering_applied = True
        
        # Save CNN dataset
        output_path = OUTPUT_DIR / f"{split_name}.parquet"
        df_cnn.to_parquet(output_path, index=False)
        
        # Track results
        results_cnn.append(
            {
                "split": split_name,
                "n_samples": len(df_cnn),
                "n_features": meta_cnn["n_features"],
                "outliers_removed": meta_cnn["outliers_removed"],
                "genus_filtering": "Yes" if genus_filtering_applied else "No",
                "genera_removed": genera_removed,
                "test_policy_applied": meta_cnn["test_split_policy_applied"],
                "proximity_used": meta_cnn["proximity_strategy_used"],
            }
        )
        
        # Report
        print(f"  Samples: {meta_cnn['original_n_samples']:,} → {len(df_cnn):,}")
        print(f"  Features: {meta_cnn['n_features']} (full temporal)")
        print(f"  Outliers removed: {meta_cnn['outliers_removed']}")
        
        if genus_filtering_applied:
            print(f"  Genus filter: {genera_removed} samples removed")
            print("  genus_german: Mapped")
        else:
            print("  genus_german: Fixed (no genus filter)")
        
        if meta_cnn["test_split_policy_applied"]:
            print("  ⚠️  Test split policy: proximity override baseline")
        
        print(f"  ✅ Saved: {output_path.name}")
        
        gc.collect()
        
        # Memory cleanup
        del df_cnn, meta_cnn
    
    results_cnn_df = pd.DataFrame(results_cnn)
    print("\n" + "=" * 70)
    print("CNN1D Summary")
    print("=" * 70)
    print(results_cnn_df.to_string(index=False))
    
    log.end_step(status="success", records=len(results_cnn))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 8. VALIDATION
# ============================================================================

log.start_step("Validation")

try:
    print("\n" + "=" * 70)
    print("Validation Checks")
    print("=" * 70)

    validation_passed = True

    # Validate XGBoost datasets (reduced features)
    print("\n[XGBoost Datasets - Reduced Features]")
    for city, split in city_splits:
        split_name = f"{city}_{split}"
        output_path = OUTPUT_DIR / f"{split_name}.parquet"

        if not output_path.exists():
            print(f"❌ Missing: {split_name}.parquet")
            validation_passed = False
            continue

        # Load and validate
        df = pd.read_parquet(output_path)

        # Check feature count matches
        feature_cols = [c for c in df.columns if c not in ablation.get_metadata_columns(df)]
        if len(feature_cols) != len(setup["selected_features"]):
            print(
                f"❌ {split_name}: Feature count mismatch "
                f"({len(feature_cols)} != {len(setup['selected_features'])})"
            )
            validation_passed = False

        # Check for NaN values in features
        nan_count = df[feature_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"❌ {split_name}: Contains {nan_count} NaN values")
            validation_passed = False

        # Check German names present
        if df["genus_german"].isna().any():
            print(f"❌ {split_name}: Missing genus_german values")
            validation_passed = False

        # Check sample count is reasonable
        if len(df) < 100:
            print(f"⚠️  {split_name}: Very few samples ({len(df)})")

        print(f"✅ {split_name}: {len(df):,} samples, {len(feature_cols)} features, 0 NaN")

    # Validate CNN1D datasets (full features)
    print("\n[CNN1D Datasets - Full Temporal Features]")
    for city, split in city_splits:
        split_name = f"{city}_{split}_cnn"
        output_path = OUTPUT_DIR / f"{split_name}.parquet"

        if not output_path.exists():
            print(f"❌ Missing: {split_name}.parquet")
            validation_passed = False
            continue

        # Load and validate
        df = pd.read_parquet(output_path)

        # Check feature columns exist
        feature_cols = [c for c in df.columns if c not in ablation.get_metadata_columns(df)]
        
        # CNN should have MORE features than XGBoost (full temporal set)
        if len(feature_cols) <= len(setup["selected_features"]):
            print(
                f"⚠️  {split_name}: Expected more features than XGBoost "
                f"({len(feature_cols)} <= {len(setup['selected_features'])})"
            )

        # Check for NaN values in features
        nan_count = df[feature_cols].isna().sum().sum()
        if nan_count > 0:
            print(f"❌ {split_name}: Contains {nan_count} NaN values")
            validation_passed = False

        # Check German names present
        if df["genus_german"].isna().any():
            print(f"❌ {split_name}: Missing genus_german values")
            validation_passed = False

        # Check sample count matches XGBoost dataset (same samples, different features)
        xgb_path = OUTPUT_DIR / f"{city}_{split}.parquet"
        xgb_df = pd.read_parquet(xgb_path)
        if len(df) != len(xgb_df):
            print(f"⚠️  {split_name}: Sample count mismatch with XGBoost dataset ({len(df)} != {len(xgb_df)})")

        print(f"✅ {split_name}: {len(df):,} samples, {len(feature_cols)} features, 0 NaN")

    print("=" * 70)

    if validation_passed:
        print("\n✅ All validation checks passed")
        log.end_step(status="success")
    else:
        print("\n❌ Validation failed")
        log.end_step(status="error", errors=["Validation checks failed"])
        raise ValueError("Validation checks failed")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 9. SAVE EXECUTION LOG
# ============================================================================

log.start_step("Save Execution Log")

try:
    # Add summary metadata
    execution_summary = {
        "notebook": "03a_setup_fixation",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "setup_decisions": {
            "chm": setup["chm_strategy"]["decision"],
            "proximity": setup["proximity_strategy"]["decision"],
            "outlier": setup["outlier_strategy"]["decision"],
            "n_features_reduced": len(setup["selected_features"]),
        },
        "test_split_policy": "Test splits always use baseline proximity (no filtering)",
        "dual_dataset_strategy": "XGBoost (reduced features) + CNN1D (full temporal sequences)",
        "xgboost_datasets": {
            "splits_processed": len(results),
            "total_samples": sum(r["n_samples"] for r in results),
            "splits": results,
        },
        "cnn_datasets": {
            "splits_processed": len(results_cnn),
            "total_samples": sum(r["n_samples"] for r in results_cnn),
            "splits": results_cnn,
        },
    }

    # Save execution log
    log.summary()
    log_path = LOGS_DIR / f"{log.notebook}_execution.json"
    log.save(log_path)

    # Save summary
    summary_path = METADATA_DIR / "03a_summary.json"
    summary_path.write_text(json.dumps(execution_summary, indent=2))

    print(f"\n✅ Execution log saved: {log_path}")
    print(f"✅ Summary saved: {summary_path}")

    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("OK: NOTEBOOK COMPLETE")
print("=" * 70)
print("\nSetup Decisions Applied:")
print(f"  CHM:       {setup['chm_strategy']['decision']}")
print(f"  Proximity: {setup['proximity_strategy']['decision']}")
print(f"  Outlier:   {setup['outlier_strategy']['decision']}")
print(f"\nDual Dataset Strategy:")
print(f"  XGBoost:  {len(setup['selected_features'])} features (reduced)")
print(f"  CNN1D:    {results_cnn[0]['n_features']} features (full temporal)")
print(f"\nDatasets Created:")
print(f"  XGBoost: {len(results)} splits, {sum(r['n_samples'] for r in results):,} total samples")
print(f"  CNN1D:   {len(results_cnn)} splits, {sum(r['n_samples'] for r in results_cnn):,} total samples")
print("\nOutputs:")
print(f"  Directory: {OUTPUT_DIR}")
print("  XGBoost Datasets (reduced features):")
for r in results:
    print(f"    - {r['split']}.parquet ({r['n_samples']:,} samples, {r['n_features']} features)")
print("  CNN1D Datasets (full temporal features):")
for r in results_cnn:
    print(f"    - {r['split']}.parquet ({r['n_samples']:,} samples, {r['n_features']} features)")
print(f"  Logs:     {log_path}")
print(f"  Summary:  {summary_path}")
print("\nTest Split Policy:")
print("  ℹ️  Test splits use baseline proximity (no filtering)")
print("  ℹ️  Ensures unbiased evaluation on real-world distribution")
print("\nDataset Usage:")
print("  ℹ️  XGBoost datasets: Use for tree-based models (RF, XGBoost)")
print("  ℹ️  CNN1D datasets: Use for neural networks requiring temporal sequences")
print("\nNext Steps:")
if has_genus_selection:
    print("  1. Review dual datasets (XGBoost + CNN)")
    print("  2. Run exp_11_algorithm_comparison.ipynb (algorithm selection)")
    print("  3. Run 03b_berlin_optimization.ipynb (HP tuning - auto-selects correct dataset)")
else:
    print("  1. Run exp_10_genus_selection_validation.ipynb to add genus filtering")
    print("  2. Re-run this notebook (03a) to apply genus filtering")
    print("  3. Run exp_11_algorithm_comparison.ipynb (algorithm selection)")
    print("  4. Run 03b_berlin_optimization.ipynb (hyperparameter tuning)")
print("=" * 70)
