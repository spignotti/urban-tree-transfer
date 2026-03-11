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
# **Title:** Outlier Removal Ablation
#
# **Phase:** 3 - Experiments
#
# **Topic:** Outlier Strategy Selection
#
# **Research Question:**
# Which outlier-removal strategy performs best once CHM and proximity decisions have already been fixed?
#
# **Key Findings:**
# - Outlier strategies are compared after earlier setup choices are fixed.
# - The winning strategy is written back into `setup_decisions.json`.
# - The result informs downstream setup fixation in `03a_setup_fixation.ipynb`.
#
# **Input:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_2_splits`
#
# **Output:** Outlier strategy written into `setup_decisions.json`
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

log = ExecutionLog("exp_08c_outlier_ablation")

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
variants = config["setup_ablation"]["outliers"]["variants"]
rules = config["setup_ablation"]["outliers"]["decision_rules"]

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Metadata:               {METADATA_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")
print(f"CV folds:               {config['global']['cv_folds']}")
print(f"Outlier variants:       {len(variants)}")

# %%
# ============================================================================
# SECTION 1: Load Prior Decisions from exp_08 and exp_08b
# ============================================================================

log.start_step("Load Prior Decisions")

setup_path = METADATA_DIR / "setup_decisions.json"

if not setup_path.exists():
    raise FileNotFoundError(
        f"setup_decisions.json not found at {setup_path}\n"
        "Run exp_08_chm_ablation.ipynb and exp_08b_proximity_ablation.ipynb first."
    )

setup = json.loads(setup_path.read_text())

required_keys = ["chm_strategy", "proximity_strategy"]
missing = [k for k in required_keys if k not in setup]
if missing:
    raise ValueError(
        f"Missing decisions: {missing}.\n"
        "Run exp_08 and exp_08b in order first."
    )

chm_strategy = setup["chm_strategy"]["decision"]
chm_features = setup["chm_strategy"].get("included_features", [])
proximity_strategy = setup["proximity_strategy"]["decision"]

print(f"\nLoaded prior decisions:")
print(f"  CHM: {chm_strategy}")
print(f"  CHM features: {chm_features or 'none (Sentinel-2 only)'}")
print(f"  Proximity: {proximity_strategy}")
print(f"\nThese decisions will be applied to ALL outlier variants.")

log.end_step(status="success")

# %%
# ============================================================================
# SECTION 2: Outlier Ablation Experiment
# ============================================================================

log.start_step("Outlier Ablation Experiment")

results = []

# Load correct proximity variant (baseline or filtered)
train_df, val_df, test_df = data_loading.load_berlin_splits(
    INPUT_DIR, variant=proximity_strategy
)

print(f"\nLoaded {proximity_strategy} dataset: {len(train_df):,} training samples")

# Memory optimization: Convert float64 → float32
float_cols = train_df.select_dtypes(include=["float64"]).columns
if len(float_cols) > 0:
    train_df[float_cols] = train_df[float_cols].astype("float32")
    val_df[float_cols] = val_df[float_cols].astype("float32")
    test_df[float_cols] = test_df[float_cols].astype("float32")
    print(f"Converted {len(float_cols)} float columns to float32")

# Apply CHM strategy to base dataset
train_df = ablation.apply_chm_strategy(train_df, chm_strategy)
val_df = ablation.apply_chm_strategy(val_df, chm_strategy)
print(f"Applied CHM strategy: {chm_strategy}")

for variant in variants:
    print(f"\nTesting variant: {variant['name']}")
    print(f"  Description: {variant['description']}")
    
    # Apply outlier removal strategy
    df_train = ablation.apply_outlier_removal(train_df.copy(), variant["name"])
    df_val = ablation.apply_outlier_removal(val_df.copy(), variant["name"])
    print(f"  Samples after removal: {len(df_train):,}")
    
    # Get feature columns with CHM decision applied
    feature_cols = data_loading.get_feature_columns(
        df_train,
        include_chm=bool(chm_features),
        chm_features=chm_features or None,
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
            "samples_retained": len(df_train),
        }
    )
    
    # Memory cleanup
    del df_train, df_val, x_train, x_val, y_train, y_val
    del y_train_enc, y_val_enc, x_train_scaled, x_val_scaled
    del feature_cols, scaler, cv, rf, metrics, label_to_idx, idx_to_label
    gc.collect()
    print(f"  Memory cleanup: completed")

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("Outlier Ablation Results:")
print("=" * 60)
print(results_df.to_string(index=False))

log.end_step(status="success", records=len(results_df))

# %%
# ============================================================================
# SECTION 3: Decision Logic
# ============================================================================

log.start_step("Decision Logic")

# Extract baseline metrics (no_removal)
no_removal_row = results_df.loc[results_df["variant"] == "no_removal"].iloc[0]
baseline_n = int(no_removal_row["samples_retained"])
baseline_f1 = no_removal_row["val_f1_mean"]
baseline_gap = no_removal_row["train_val_gap"]

print(f"\nBaseline (no_removal):")
print(f"  Samples: {baseline_n:,}")
print(f"  Val F1:  {baseline_f1:.4f}")
print(f"  Train-Val Gap: {baseline_gap:.4f}")

print(f"\nDecision Thresholds:")
print(f"  Max sample loss:  {rules['max_sample_loss']:.1%}")
print(f"  Min improvement:  {rules['min_improvement']:.3f}")

# Evaluate removal strategies
best_variant = "no_removal"
best_f1 = baseline_f1
reasoning = "Baseline (no removal) retained by default"

print(f"\n" + "=" * 70)
print("Decision Logic:")
print("=" * 70)

for variant_name in ["remove_high", "remove_high_medium"]:
    row = results_df.loc[results_df["variant"] == variant_name].iloc[0]
    variant_n = int(row["samples_retained"])
    variant_f1 = row["val_f1_mean"]
    variant_gap = row["train_val_gap"]
    sample_loss = 1 - (variant_n / baseline_n)
    f1_improvement = variant_f1 - baseline_f1
    
    print(f"\n{variant_name}:")
    print(f"  Samples: {variant_n:,} (loss={sample_loss:.1%})")
    print(f"  Val F1:  {variant_f1:.4f} (Δ={f1_improvement:+.4f})")
    print(f"  Train-Val Gap: {variant_gap:.4f}")
    
    # Check if removal strategy passes thresholds
    passes_sample_loss = sample_loss <= rules["max_sample_loss"]
    passes_improvement = f1_improvement >= rules["min_improvement"]
    
    print(f"  ✅ Sample loss ≤ {rules['max_sample_loss']:.1%}? {passes_sample_loss}")
    print(f"  ✅ F1 improvement ≥ {rules['min_improvement']:.3f}? {passes_improvement}")
    
    if passes_sample_loss and passes_improvement:
        if variant_f1 > best_f1:
            best_variant = variant_name
            best_f1 = variant_f1
            reasoning = (
                f"{variant_name}: {f1_improvement:+.3f} F1 improvement "
                f"with acceptable {sample_loss:.1%} sample loss"
            )
            print(f"  → New best variant!")
    else:
        print(f"  → Rejected (threshold violation)")

decision = best_variant
final_row = results_df.loc[results_df["variant"] == decision].iloc[0]
final_n = int(final_row["samples_retained"])
final_sample_loss = 1 - (final_n / baseline_n)

print(f"\n" + "=" * 70)
print(f"FINAL DECISION: {decision}")
print(f"Reasoning: {reasoning}")
print(f"Sample loss: {final_sample_loss:.1%} ({baseline_n:,} → {final_n:,})")
print(f"Val F1: {best_f1:.4f}")
print("=" * 70)

log.end_step(status="success")

# %%
# ============================================================================
# SECTION 5: Update setup_decisions.json
# ============================================================================

log.start_step("Update Setup Decisions")

# Get final metrics for selected decision
final_row = results_df.loc[results_df["variant"] == decision].iloc[0]

setup["outlier_strategy"] = {
    "decision": decision,
    "reasoning": reasoning,
    "sample_counts": {
        "baseline_n": baseline_n,
        "filtered_n": final_n,
        "sample_loss_pct": float(final_sample_loss),
    },
    "ablation_results": results_df.to_dict(orient="records"),
    "val_f1_mean": float(final_row["val_f1_mean"]),
    "train_val_gap": float(final_row["train_val_gap"]),
    "samples_retained": float(final_row["samples_retained"]),
    "rationale": (
        f"Selected {decision} based on threshold evaluation: "
        f"max_sample_loss={rules['max_sample_loss']:.1%}, "
        f"min_improvement={rules['min_improvement']:.3f}"
    ),
}

setup_path.write_text(json.dumps(setup, indent=2))
print(f"\nUpdated {setup_path} with outlier decision")
print("✅ Outlier decision saved successfully")
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
