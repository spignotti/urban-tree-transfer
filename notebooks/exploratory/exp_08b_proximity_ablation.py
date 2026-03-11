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
# # exp_08b: Proximity Filter Ablation
#
# **Phase 3.1 - Setup Fixation**
#
# Determines optimal proximity filter strategy (baseline vs filtered) by testing with CHM decision from exp_08.
#
# **Dependencies:** Requires `setup_decisions.json` from exp_08 (CHM strategy).

# %%
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Not required
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================

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
repo_url = f"git+https://{token}@github.com/SilasPignotti/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

print("OK: Package installed")

# %%
# Mount Google Drive for data files
from google.colab import drive

drive.mount("/content/drive")

print("Google Drive mounted")

# %%
# Package imports
from urban_tree_transfer.config import RANDOM_SEED, load_experiment_config
from urban_tree_transfer.experiments import (
    ablation,
    data_loading,
    preprocessing,
    training,
    models,
)
from urban_tree_transfer.utils import (
    ExecutionLog,
    validate_setup_decisions,
)

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings
import gc

import pandas as pd
import numpy as np

log = ExecutionLog("exp_08b_proximity_ablation")

warnings.filterwarnings("ignore", category=UserWarning)

print("OK: Package imports complete")

# %%
# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_splits"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_3_experiments"

METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in [METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load experiment configuration
config = load_experiment_config()
variants = config["setup_ablation"]["proximity"]["variants"]
rules = config["setup_ablation"]["proximity"]["decision_rules"]

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Metadata:               {METADATA_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")
print(f"CV folds:               {config['global']['cv_folds']}")
print(f"Proximity variants:     {len(variants)}")

# %%
# ============================================================
# SECTION 1: Load CHM Decision from exp_08
# ============================================================

log.start_step("Load CHM Decision")

setup_path = METADATA_DIR / "setup_decisions.json"

if not setup_path.exists():
    raise FileNotFoundError(
        f"setup_decisions.json not found at {setup_path}\n"
        "Run exp_08_chm_ablation.ipynb first to create this file."
    )

setup = json.loads(setup_path.read_text())

if "chm_strategy" not in setup:
    raise ValueError(
        "CHM strategy not found in setup_decisions.json.\n"
        "Run exp_08_chm_ablation.ipynb first."
    )

chm_strategy = setup["chm_strategy"]["decision"]
chm_features = setup["chm_strategy"].get("included_features", [])

print(f"\nLoaded CHM decision from exp_08:")
print(f"  Strategy: {chm_strategy}")
print(f"  Features: {chm_features or 'none (Sentinel-2 only)'}")
print(f"\nThis CHM strategy will be applied to BOTH proximity variants.")

log.end_step(status="success")

# %%
# ============================================================
# SECTION 2: Proximity Ablation Experiment
# ============================================================

log.start_step("Proximity Ablation Experiment")

results = []

for variant in variants:
    print(f"\nTesting variant: {variant['name']}")
    print(f"  Description: {variant['description']}")
    
    # Load baseline or filtered dataset
    train_df, val_df, test_df = data_loading.load_berlin_splits(
        INPUT_DIR, variant=variant["name"]
    )
    print(f"  Loaded samples: {len(train_df):,}")
    
    # Memory optimization: Convert float64 → float32
    float_cols = train_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        train_df[float_cols] = train_df[float_cols].astype("float32")
        val_df[float_cols] = val_df[float_cols].astype("float32")
        print(f"  Converted {len(float_cols)} float columns to float32")
    
    # Apply CHM strategy from exp_08
    train_df = ablation.apply_chm_strategy(train_df, chm_strategy)
    val_df = ablation.apply_chm_strategy(val_df, chm_strategy)
    print(f"  Applied CHM strategy: {chm_strategy}")
    
    # Get feature columns with CHM decision applied
    feature_cols = data_loading.get_feature_columns(
        train_df,
        include_chm=bool(chm_features),
        chm_features=chm_features or None,
    )
    print(f"  Total features: {len(feature_cols)}")
    
    # Prepare features and labels
    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    
    y_train = train_df["genus_latin"]
    y_val = val_df["genus_latin"]
    
    y_train_enc, label_to_idx, idx_to_label = preprocessing.encode_genus_labels(y_train)
    y_val_enc = y_val.map(label_to_idx).to_numpy()
    
    # Scale features (StandardScaler)
    x_train_scaled, x_val_scaled, _, scaler = preprocessing.scale_features(
        x_train, x_val=x_val
    )
    
    # Create Spatial Block CV
    cv = training.create_spatial_block_cv(
        train_df, n_splits=config["global"]["cv_folds"]
    )
    
    # Train Random Forest with CV
    rf = models.create_model("random_forest")
    metrics = training.train_with_cv(
        rf, x_train_scaled, y_train_enc, train_df["block_id"].values, cv
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
            "samples_retained": len(train_df),
        }
    )
    
    # Memory cleanup
    del train_df, val_df, test_df, x_train, x_val, y_train, y_val
    del y_train_enc, y_val_enc, x_train_scaled, x_val_scaled
    del feature_cols, scaler, cv, rf, metrics, label_to_idx, idx_to_label
    gc.collect()
    print(f"  Memory cleanup: completed")

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("Proximity Ablation Results:")
print("=" * 60)
print(results_df.to_string(index=False))

log.end_step(status="success", records=len(results_df))

# %%
# ============================================================
# SECTION 3: Decision Logic
# ============================================================

log.start_step("Decision Logic")

# Extract metrics
baseline_row = results_df.loc[results_df["variant"] == "baseline"].iloc[0]
filtered_row = results_df.loc[results_df["variant"] == "filtered"].iloc[0]

baseline_n = int(baseline_row["samples_retained"])
filtered_n = int(filtered_row["samples_retained"])
sample_loss = 1 - (filtered_n / baseline_n)

baseline_f1 = baseline_row["val_f1_mean"]
filtered_f1 = filtered_row["val_f1_mean"]
f1_improvement = filtered_f1 - baseline_f1

baseline_gap = baseline_row["train_val_gap"]
filtered_gap = filtered_row["train_val_gap"]

print(f"\nMetrics Summary:")
print(f"  Baseline samples: {baseline_n:,}")
print(f"  Filtered samples: {filtered_n:,}")
print(f"  Sample loss:      {sample_loss:.1%}")
print(f"\n  Baseline F1:      {baseline_f1:.4f} (gap={baseline_gap:.4f})")
print(f"  Filtered F1:      {filtered_f1:.4f} (gap={filtered_gap:.4f})")
print(f"  F1 improvement:   {f1_improvement:+.4f}")

print(f"\nDecision Thresholds:")
print(f"  Max sample loss:  {rules['max_sample_loss']:.1%}")
print(f"  Min improvement:  {rules['min_improvement']:.3f}")

# Apply decision rules
print(f"\n" + "=" * 70)
print("Decision Logic:")
print("=" * 70)

if sample_loss > rules["max_sample_loss"]:
    decision = "baseline"
    reasoning = (
        f"Sample loss {sample_loss:.1%} exceeds threshold "
        f"{rules['max_sample_loss']:.1%} - too much data lost"
    )
    print(f"✅ Rule 1: Sample Loss > {rules['max_sample_loss']:.1%}?")
    print(f"   {sample_loss:.1%} > {rules['max_sample_loss']:.1%} → YES")
    print(f"   → Decision: baseline (too much data lost)")
elif f1_improvement < rules["min_improvement"]:
    decision = "baseline"
    reasoning = (
        f"F1 improvement {f1_improvement:.3f} below threshold "
        f"{rules['min_improvement']:.3f} - marginal gain"
    )
    print(f"✅ Rule 1: Sample Loss > {rules['max_sample_loss']:.1%}?")
    print(f"   {sample_loss:.1%} ≤ {rules['max_sample_loss']:.1%} → NO")
    print(f"\n✅ Rule 2: F1 Improvement < {rules['min_improvement']:.3f}?")
    print(f"   {f1_improvement:.3f} < {rules['min_improvement']:.3f} → YES")
    print(f"   → Decision: baseline (marginal improvement)")
else:
    decision = "filtered"
    reasoning = (
        f"Filtered improves F1 by {f1_improvement:.3f} "
        f"with acceptable {sample_loss:.1%} sample loss"
    )
    print(f"✅ Rule 1: Sample Loss > {rules['max_sample_loss']:.1%}?")
    print(f"   {sample_loss:.1%} ≤ {rules['max_sample_loss']:.1%} → NO")
    print(f"\n✅ Rule 2: F1 Improvement < {rules['min_improvement']:.3f}?")
    print(f"   {f1_improvement:.3f} ≥ {rules['min_improvement']:.3f} → NO")
    print(f"   → Decision: filtered (worth the trade-off)")

print(f"\n" + "=" * 70)
print(f"FINAL DECISION: {decision}")
print(f"Reasoning: {reasoning}")
print("=" * 70)

log.end_step(status="success")

# %%
# ============================================================
# SECTION 5: Update setup_decisions.json
# ============================================================

log.start_step("Update Setup Decisions")

# Get final metrics for selected decision
final_row = results_df.loc[results_df["variant"] == decision].iloc[0]

setup["proximity_strategy"] = {
    "decision": decision,
    "reasoning": reasoning,
    "sample_counts": {
        "baseline_n": baseline_n,
        "filtered_n": filtered_n,
        "sample_loss_pct": float(sample_loss),
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
print(f"\nUpdated {setup_path} with proximity decision")
print("✅ Proximity decision saved successfully")
print(f"\n⚠️  Note: Full schema validation will run after exp_09")
print(f"   (when all strategies are complete: CHM, Proximity, Outlier, Features)")

log.end_step(status="success")

