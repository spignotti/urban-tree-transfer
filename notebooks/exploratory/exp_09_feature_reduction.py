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
# # exp_09: Feature Reduction
#
# **Phase 3.1 - Setup Fixation (Final Step)**
#
# Determines optimal feature subset by testing different k values with all prior decisions applied (CHM + Proximity + Outlier). Uses Pareto optimization to balance F1 performance vs model simplicity.
#
# **Dependencies:** Requires `setup_decisions.json` from exp_08, exp_08b, and exp_08c (CHM + Proximity + Outlier strategies).
#
# **Output:** Completes `setup_decisions.json` with `feature_set` and `selected_features` for Phase 3.2+ experiments.

# %%
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended (feature importance on full dataset)
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

log = ExecutionLog("exp_09_feature_reduction")

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
candidate_k = config["setup_ablation"]["feature_reduction"]["candidate_k"]

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Metadata:               {METADATA_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")
print(f"CV folds:               {config['global']['cv_folds']}")
print(f"Candidate k values:     {candidate_k}")

# %%
# ============================================================
# SECTION 1: Load All Prior Decisions
# ============================================================

log.start_step("Load Prior Decisions")

setup_path = METADATA_DIR / "setup_decisions.json"

if not setup_path.exists():
    raise FileNotFoundError(
        f"setup_decisions.json not found at {setup_path}\n"
        "Run exp_08, exp_08b, and exp_08c first."
    )

setup = json.loads(setup_path.read_text())

required_keys = ["chm_strategy", "proximity_strategy", "outlier_strategy"]
missing = [k for k in required_keys if k not in setup]
if missing:
    raise ValueError(
        f"Missing decisions: {missing}.\n"
        "Run exp_08, exp_08b, and exp_08c in order first."
    )

chm_strategy = setup["chm_strategy"]["decision"]
chm_features = setup["chm_strategy"].get("included_features", [])
proximity_strategy = setup["proximity_strategy"]["decision"]
outlier_strategy = setup["outlier_strategy"]["decision"]

print(f"\nLoaded prior decisions:")
print(f"  CHM: {chm_strategy}")
print(f"  CHM features: {chm_features or 'none (Sentinel-2 only)'}")
print(f"  Proximity: {proximity_strategy}")
print(f"  Outlier: {outlier_strategy}")
print(f"\nAll decisions will be applied to prepare final dataset.")

log.end_step(status="success")

# %%
# ============================================================
# SECTION 2: Prepare Final Dataset with All Decisions
# ============================================================

log.start_step("Prepare Final Dataset")

# Load correct proximity variant
train_df, val_df, test_df = data_loading.load_berlin_splits(
    INPUT_DIR, variant=proximity_strategy
)

print(f"\nLoaded {proximity_strategy} dataset: {len(train_df):,} samples")

# Memory optimization: Convert float64 → float32
float_cols = train_df.select_dtypes(include=["float64"]).columns
if len(float_cols) > 0:
    train_df[float_cols] = train_df[float_cols].astype("float32")
    val_df[float_cols] = val_df[float_cols].astype("float32")
    test_df[float_cols] = test_df[float_cols].astype("float32")
    print(f"Converted {len(float_cols)} float columns to float32")

# Apply CHM strategy
train_df = ablation.apply_chm_strategy(train_df, chm_strategy)
val_df = ablation.apply_chm_strategy(val_df, chm_strategy)
print(f"Applied CHM strategy: {chm_strategy}")

# Apply outlier removal
train_df = ablation.apply_outlier_removal(train_df, outlier_strategy)
val_df = ablation.apply_outlier_removal(val_df, outlier_strategy)
print(f"Applied outlier strategy: {outlier_strategy}")

# Get feature columns
feature_cols = data_loading.get_feature_columns(
    train_df,
    include_chm=bool(chm_features),
    chm_features=chm_features or None,
)

print(f"\nFinal dataset after all decisions:")
print(f"  Training samples: {len(train_df):,}")
print(f"  Validation samples: {len(val_df):,}")
print(f"  Features: {len(feature_cols)}")

log.end_step(status="success", records=len(train_df))

# %%
# ============================================================
# SECTION 3: Feature Importance Analysis
# ============================================================

log.start_step("Feature Importance Analysis")

# Prepare data for training
x_train = train_df[feature_cols]
x_val = val_df[feature_cols]

y_train = train_df["genus_latin"]
y_val = val_df["genus_latin"]

y_train_enc, label_to_idx, idx_to_label = preprocessing.encode_genus_labels(y_train)
y_val_enc = y_val.map(label_to_idx).to_numpy()

# Scale features
x_train_scaled, x_val_scaled, _, scaler = preprocessing.scale_features(
    x_train, x_val=x_val
)

# Train model on full feature set to compute importance
print("\nTraining Random Forest on full feature set for importance ranking...")
model = models.create_model("random_forest")
model.fit(x_train_scaled, y_train_enc)

# Compute feature importance
importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": model.feature_importances_,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print(f"\nTop 10 most important features:")
print(importance_df.head(10).to_string(index=False))

# Visualize top 30 features

# Cleanup
del model
gc.collect()

log.end_step(status="success", records=len(importance_df))

# %%
# ============================================================
# SECTION 4: Feature Subset Evaluation
# ============================================================

log.start_step("Feature Subset Evaluation")

# Create Spatial Block CV
cv = training.create_spatial_block_cv(
    train_df, n_splits=config["global"]["cv_folds"]
)

results = []

for k in candidate_k:
    print(f"\nTesting k={k} features...")
    
    # Select top k features
    selected = importance_df.head(k)["feature"].tolist()
    x_train_k = train_df[selected]
    x_val_k = val_df[selected]
    
    # Scale selected features
    x_train_k_scaled, x_val_k_scaled, _, scaler_k = preprocessing.scale_features(
        x_train_k, x_val=x_val_k
    )
    
    # Train with CV
    rf = models.create_model("random_forest")
    metrics = training.train_with_cv(
        rf, x_train_k_scaled, y_train_enc, train_df["block_id"].values, cv
    )
    
    print(f"  Val F1: {metrics['val_f1_mean']:.4f} ± {metrics['val_f1_std']:.4f}")
    print(f"  Train-Val Gap: {metrics['train_val_gap']:.4f}")
    
    results.append({
        "n_features": k,
        "val_f1_mean": metrics["val_f1_mean"],
        "val_f1_std": metrics["val_f1_std"],
        "train_f1_mean": metrics["train_f1_mean"],
        "train_val_gap": metrics["train_val_gap"],
    })
    
    # Memory cleanup
    del x_train_k, x_val_k, x_train_k_scaled, x_val_k_scaled, scaler_k, rf, metrics
    gc.collect()

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("Feature Subset Evaluation:")
print("=" * 60)
print(results_df.to_string(index=False))

log.end_step(status="success", records=len(results_df))

# %%
# ============================================================
# SECTION 5: Pareto Optimization
# ============================================================

log.start_step("Pareto Optimization")

# Find smallest k where F1 >= best_f1 - max_drop
best_f1 = results_df["val_f1_mean"].max()
max_drop = 0.01  # Allow 1% F1 drop for simplicity

pareto_threshold = best_f1 - max_drop
viable = results_df[results_df["val_f1_mean"] >= pareto_threshold]

if len(viable) == 0:
    # Fallback: select best F1
    optimal_row = results_df.sort_values("val_f1_mean", ascending=False).iloc[0]
    reasoning = f"No viable Pareto candidates; selected best F1: {optimal_row['val_f1_mean']:.4f}"
else:
    # Select smallest k from viable candidates
    optimal_row = viable.sort_values("n_features").iloc[0]
    reasoning = f"Pareto selection: smallest k with F1 >= {pareto_threshold:.4f}"

selected_k = int(optimal_row["n_features"])
selected_f1 = optimal_row["val_f1_mean"]
selected_gap = optimal_row["train_val_gap"]
selected_features = importance_df.head(selected_k)["feature"].tolist()

print(f"\n" + "=" * 70)
print("Pareto Optimization Results:")
print("=" * 70)
print(f"\nOptimal feature set:")
print(f"  k = {selected_k} features")
print(f"  Val F1 = {selected_f1:.4f} ± {optimal_row['val_f1_std']:.4f}")
print(f"  Train-Val Gap = {selected_gap:.4f}")
print(f"\nThresholds:")
print(f"  Best F1: {best_f1:.4f}")
print(f"  Pareto threshold (best - {max_drop}): {pareto_threshold:.4f}")
print(f"\nReasoning: {reasoning}")
print("=" * 70)


log.end_step(status="success")

# %%
# ============================================================
# SECTION 6: Complete setup_decisions.json
# ============================================================

log.start_step("Complete Setup Decisions")

# Add feature_set and selected_features
setup["feature_set"] = {
    "n_features": selected_k,
    "reasoning": reasoning,
    "importance_ranking": importance_df.to_dict(orient="records"),
    "ablation_results": results_df.to_dict(orient="records"),
    "val_f1_mean": float(selected_f1),
    "train_val_gap": float(selected_gap),
    "rationale": (
        f"Selected {selected_k} features using Pareto optimization: "
        f"smallest feature set with F1 >= {pareto_threshold:.4f}"
    ),
}

setup["selected_features"] = selected_features

# Add metadata
setup["metadata"] = {
    "completed_at": datetime.now(timezone.utc).isoformat(),
    "random_seed": RANDOM_SEED,
    "cv_folds": config["global"]["cv_folds"],
}

setup_path.write_text(json.dumps(setup, indent=2))
print(f"\nCompleted {setup_path} with all 4 strategies + selected features")

# Validate against schema (FINAL VALIDATION)
try:
    validate_setup_decisions(setup_path)
    print("✅ Schema validation: PASSED")
except Exception as e:
    print(f"❌ Schema validation: FAILED")
    print(f"   Error: {e}")
    log.end_step(status="error", errors=[str(e)])
    raise

print(f"\nSetup decisions summary:")
print(f"  CHM: {chm_strategy}")
print(f"  Proximity: {proximity_strategy}")
print(f"  Outlier: {outlier_strategy}")
print(f"  Features: {selected_k} selected")
print(f"\n✅ Phase 3.1 Setup Fixation COMPLETE")

log.end_step(status="success")

# %%
# ============================================================
# SECTION 7: Summary & Next Steps
# ============================================================

# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETE: exp_09 Feature Reduction")
print("=" * 70)
print(f"\nFinal Setup:")
print(f"  CHM: {chm_strategy} ({len(chm_features) if chm_features else 0} features)")
print(f"  Proximity: {proximity_strategy}")
print(f"  Outlier: {outlier_strategy}")
print(f"  Feature Set: {selected_k} features selected")
print(f"  Val F1: {selected_f1:.4f} ± {optimal_row['val_f1_std']:.4f}")
print(f"\nOutputs:")
print(f"  - JSON: {setup_path} (✅ validated)")
print(f"  - Logs: {log_path}")
print(f"\nNext Steps:")
print(f"  1. Review Pareto curve and feature importance plots")
print(f"  2. Run exp_10_algorithm_comparison.ipynb (Phase 3.2)")
print(f"  3. Or run 03a_setup_fixation.ipynb (runner notebook)")
print("=" * 70)
