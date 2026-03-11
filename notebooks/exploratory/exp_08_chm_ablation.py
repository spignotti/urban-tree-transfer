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
# # Urban Tree Transfer - Exploratory Notebook
#
# **Title:** CHM Ablation
#
# **Phase:** 3 - Experiments
#
# **Topic:** CHM Strategy Selection
#
# **Research Question:**
# Which CHM feature strategy yields the best baseline performance for setup fixation in Phase 3?
#
# **Key Findings:**
# - Multiple CHM variants are compared with a Random Forest baseline.
# - A setup decision is exported to `setup_decisions.json`.
# - The notebook supports downstream setup fixation in `03a_setup_fixation.ipynb`.
#
# **Input:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_2_splits`
#
# **Output:** CHM decision written into `setup_decisions.json`
#
# **Author:** Silas Pignotti
#
# **Created:** 2025-01-15
#
# **Updated:** 2026-03-11

# %%
# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Not required
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================================

import subprocess
from google.colab import userdata

# Get GitHub token from Colab Secrets (for private repo access)
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        "GITHUB_TOKEN not found in Colab Secrets.\n"
        "1. Click the key icon in the left sidebar\n"
        "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
        "3. Toggle 'Notebook access' ON"
    )

# Install package from private GitHub repo
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
from urban_tree_transfer.config import RANDOM_SEED, load_experiment_config
from urban_tree_transfer.experiments import (
    ablation,
    data_loading,
    preprocessing,
    training,
    models,
)
from urban_tree_transfer.utils import ExecutionLog

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
import gc

import pandas as pd
import numpy as np

log = ExecutionLog("exp_08_chm_ablation")

warnings.filterwarnings("ignore", category=UserWarning)

print("OK: Package imports complete")

# %%
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_splits"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_3_experiments"

METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in [METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load experiment configuration
config = load_experiment_config()
variants = config["setup_ablation"]["chm"]["variants"]
rules = config["setup_ablation"]["chm"]["decision_rules"]

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Metadata:               {METADATA_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")
print(f"CV folds:               {config['global']['cv_folds']}")
print(f"CHM variants to test:   {len(variants)}")

# %%
# ============================================================================
# SECTION 1: Data Loading & Memory Optimization
# ============================================================================

log.start_step("Data Loading")

train_df, val_df, test_df = data_loading.load_berlin_splits(INPUT_DIR)

print(f"Berlin Train: {len(train_df):,} samples")
print(f"Berlin Val:   {len(val_df):,} samples")
print(f"Berlin Test:  {len(test_df):,} samples")

# Memory optimization: Convert float64 → float32 (halves memory usage)
print("\nMemory optimization: Converting float64 → float32...")
float_cols = train_df.select_dtypes(include=['float64']).columns
if len(float_cols) > 0:
    train_df[float_cols] = train_df[float_cols].astype('float32')
    val_df[float_cols] = val_df[float_cols].astype('float32')
    test_df[float_cols] = test_df[float_cols].astype('float32')
    print(f"  Converted {len(float_cols)} float columns to float32")
    print(f"  Memory savings: ~50% on numerical features")
else:
    print("  No float64 columns found")

log.end_step(status="success", records=len(train_df) + len(val_df) + len(test_df))

# %%
# ============================================================================
# SECTION 2: CHM Ablation Experiment
# ============================================================================

log.start_step("CHM Ablation Experiment")

results = []

for variant in variants:
    print(f"\nTesting variant: {variant['name']}")
    print(f"  Features: {variant['features'] or 'none (Sentinel-2 only)'}")
    
    # Apply CHM strategy
    df_train = ablation.apply_chm_strategy(train_df.copy(), variant["name"])
    df_val = ablation.apply_chm_strategy(val_df.copy(), variant["name"])
    
    # Get feature columns
    feature_cols = data_loading.get_feature_columns(
        df_train,
        include_chm=variant["name"] != "no_chm",
        chm_features=variant["features"] or None,
    )
    print(f"  Total features: {len(feature_cols)}")
    
    # Prepare features and labels
    x_train = df_train[feature_cols]
    x_val = df_val[feature_cols]
    
    y_train = df_train["genus_latin"]
    y_val = df_val["genus_latin"]
    
    y_train_enc, label_to_idx, idx_to_label = preprocessing.encode_genus_labels(y_train)
    y_val_enc = y_val.map(label_to_idx).to_numpy()
    
    # Scale features (StandardScaler)
    x_train_scaled, x_val_scaled, _, scaler = preprocessing.scale_features(
        x_train, x_val=x_val
    )
    
    # Create Spatial Block CV
    cv = training.create_spatial_block_cv(
        df_train, n_splits=config["global"]["cv_folds"]
    )
    
    # Train Random Forest with CV
    rf = models.create_model("random_forest")
    metrics = training.train_with_cv(
        rf, x_train_scaled, y_train_enc, df_train["block_id"].values, cv
    )
    
    print(f"  Val F1 (mean): {metrics['val_f1_mean']:.4f} ± {metrics['val_f1_std']:.4f}")
    print(f"  Train-Val Gap: {metrics['train_val_gap']:.4f}")
    
    results.append(
        {
            "variant": variant["name"],
            "val_f1_mean": metrics["val_f1_mean"],
            "val_f1_std": metrics["val_f1_std"],
            "train_f1_mean": metrics["train_f1_mean"],
            "train_val_gap": metrics["train_val_gap"],
        }
    )
    
    # Memory cleanup: Explicitly delete large variables and trigger garbage collection
    del df_train, df_val, x_train, x_val, y_train, y_val
    del y_train_enc, y_val_enc, x_train_scaled, x_val_scaled
    del feature_cols, scaler, cv, rf, metrics, label_to_idx, idx_to_label
    gc.collect()
    print(f"  Memory cleanup: completed")

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("CHM Ablation Results:")
print("=" * 60)
print(results_df.to_string(index=False))

log.end_step(status="success", records=len(results_df))

# %%
# ============================================================================
# SECTION 3: Feature Importance Analysis
# ============================================================================

log.start_step("Feature Importance Analysis")

# Train RF with all CHM features to get importances
all_chm_features = data_loading.get_feature_columns(
    train_df,
    include_chm=True,
    chm_features=["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"],
)

x_all = train_df[all_chm_features].to_numpy(dtype=float)
y_all, _, _ = preprocessing.encode_genus_labels(train_df["genus_latin"])

# Compute feature importance
importance_df = ablation.compute_feature_importance(x_all, y_all, all_chm_features)
chm_importance = {row["feature"]: row["importance"] for _, row in importance_df.iterrows()}

print("\nCHM Feature Importances:")
for feature in ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"]:
    imp = chm_importance.get(feature, 0.0)
    print(f"  {feature:20s}: {imp:.4f}")

log.end_step(status="success", records=len(importance_df))

# %%
# ============================================================================
# SECTION 4: Per-Feature Decision Logic
# ============================================================================

log.start_step("Per-Feature Decision Logic")

# Get baseline metrics (no_chm)
baseline_f1 = results_df.loc[results_df["variant"] == "no_chm", "val_f1_mean"].iloc[0]
baseline_gap = results_df.loc[results_df["variant"] == "no_chm", "train_val_gap"].iloc[0]

print(f"\nBaseline (no_chm):")
print(f"  Val F1:       {baseline_f1:.4f}")
print(f"  Train-Val Gap: {baseline_gap:.4f}")
print(f"\nThresholds:")
print(f"  Feature Importance:  > {rules['importance_threshold']} → EXCLUDE (dominance)")
print(f"  Gap Increase:        > {rules['max_gap_increase']} → EXCLUDE (destabilization)")
print(f"  F1 Improvement:      < {rules['min_improvement']} → EXCLUDE (marginal)")

# Per-feature evaluation against 3 thresholds
included_features = []
per_feature_results = []

feature_variant_map = [
    ("CHM_1m", "raw_chm"),
    ("CHM_1m_zscore", "zscore_only"),
    ("CHM_1m_percentile", "percentile_only"),
]

print("\n" + "=" * 80)
print("Per-Feature Evaluation:")
print("=" * 80)

for feature_name, variant_name in feature_variant_map:
    row = results_df.loc[results_df["variant"] == variant_name].iloc[0]
    importance = chm_importance.get(feature_name, 0.0)
    gap_increase = row["train_val_gap"] - baseline_gap
    f1_improvement = row["val_f1_mean"] - baseline_f1
    
    excluded = False
    reason = ""
    
    # Threshold 1: Feature dominance check
    if importance > rules["importance_threshold"]:
        excluded = True
        reason = f"Dominates model (importance={importance:.3f} > {rules['importance_threshold']})"
    # Threshold 2: Generalization destabilization check
    elif gap_increase > rules["max_gap_increase"]:
        excluded = True
        reason = f"Destabilizes generalization (gap increase={gap_increase:.3f} > {rules['max_gap_increase']})"
    # Threshold 3: Marginal improvement check
    elif f1_improvement < rules["min_improvement"]:
        excluded = True
        reason = f"Marginal improvement (f1_imp={f1_improvement:.3f} < {rules['min_improvement']})"
    else:
        included_features.append(feature_name)
        reason = f"Passes all thresholds (imp={importance:.3f}, gap_inc={gap_increase:.3f}, f1_imp={f1_improvement:.3f})"
    
    per_feature_results.append(
        {
            "feature": feature_name,
            "variant": variant_name,
            "importance": float(importance),
            "gap_increase": float(gap_increase),
            "f1_improvement": float(f1_improvement),
            "included": not excluded,
            "reason": reason,
        }
    )
    
    status = "✅ INCLUDED" if not excluded else "❌ EXCLUDED"
    print(f"\n{feature_name}:")
    print(f"  Status:        {status}")
    print(f"  Importance:    {importance:.4f}")
    print(f"  Gap Increase:  {gap_increase:+.4f}")
    print(f"  F1 Improvement: {f1_improvement:+.4f}")
    print(f"  Reason: {reason}")

# Map included features to final variant name
variant_map = {
    (): "no_chm",
    ("CHM_1m_zscore",): "zscore_only",
    ("CHM_1m_percentile",): "percentile_only",
    ("CHM_1m_zscore", "CHM_1m_percentile"): "both_engineered",
    ("CHM_1m",): "raw_chm",
}
decision = variant_map.get(tuple(sorted(included_features)), "no_chm")

print("\n" + "=" * 80)
print(f"FINAL DECISION: {decision}")
print(f"Included features: {included_features or ['none (Sentinel-2 only)']}")
print("=" * 80)

log.end_step(status="success")

# %%
# ============================================================================
# SECTION 6: Save Decision to JSON
# ============================================================================

log.start_step("Save CHM Decision")

setup_path = METADATA_DIR / "setup_decisions.json"

# Get final metrics for selected decision
final_row = results_df.loc[results_df["variant"] == decision].iloc[0]

setup = {
    "chm_strategy": {
        "decision": decision,
        "included_features": included_features,
        "reasoning": f"Per-feature evaluation: {len(included_features)} features passed all thresholds",
        "per_feature_results": per_feature_results,
        "ablation_results": results_df.to_dict(orient="records"),
        "val_f1_mean": float(final_row["val_f1_mean"]),
        "train_val_gap": float(final_row["train_val_gap"]),
        "rationale": (
            f"Selected {decision} based on threshold evaluation: "
            f"importance_threshold={rules['importance_threshold']}, "
            f"max_gap_increase={rules['max_gap_increase']}, "
            f"min_improvement={rules['min_improvement']}"
        ),
    }
}

setup_path.write_text(json.dumps(setup, indent=2))
print(f"\nWrote CHM decision to: {setup_path}")
print("✅ CHM decision saved successfully")
print(f"\n⚠️  Note: Full schema validation will run after exp_09")
print(f"   (when all strategies are complete: CHM, Proximity, Outlier, Features)")

log.end_step(status="success")


# %%
# ============================================================================
# FINDINGS SUMMARY
# ============================================================================

log.summary()
log.save(LOGS_DIR / f"{log.notebook}_execution.json")

analysis_files = []
config_files = [setup_path] if setup_path.exists() else []

print("\n" + "=" * 60)
print(f"{'EXPORT MANIFEST':^60}")
print("=" * 60)
print("\nAnalysis data:")
for file_path in sorted(analysis_files):
    print(f"  {file_path.name}")
print("\nConfig files:")
for file_path in sorted(config_files):
    print(f"  {file_path.name}")
print("\n" + "=" * 60)
print("\nOK: NOTEBOOK COMPLETE")
