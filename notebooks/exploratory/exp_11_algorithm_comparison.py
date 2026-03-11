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
# **Title:** Algorithm Comparison
#
# **Phase:** 3 - Experiments
#
# **Topic:** Baseline and Model Family Selection
#
# **Research Question:**
# Which baseline, machine-learning, and neural-network algorithms are strong enough to advance into Phase 3 tuning and transfer experiments?
#
# **Key Findings:**
# - Multiple algorithm families are compared under the fixed Phase 3 setup.
# - The notebook identifies ML and NN champions for later tuning.
# - Results are exported to `algorithm_comparison.json`.
#
# **Input:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_2_splits`
#
# **Output:** `algorithm_comparison.json` with champion selections
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
# Required: CPU (Standard) or High-RAM for grid search
# GPU: Optional (speeds up NN models)
# High-RAM: Recommended for large grid searches
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

# Install PyTorch separately (required for neural network models)
subprocess.run(["pip", "install", "torch>=2.2.0", "-q"], check=True)

print("OK: Package installed (including PyTorch for neural networks)")


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
    evaluation,
    models,
)
from urban_tree_transfer.utils import (
    ExecutionLog,
    validate_algorithm_comparison,
)

from pathlib import Path
from datetime import datetime, timezone
from itertools import product
import json
import warnings
import gc

import pandas as pd
import numpy as np

log = ExecutionLog("exp_11_algorithm_comparison")

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

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Metadata:               {METADATA_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")
print(f"CV folds:               {config['global']['cv_folds']}")

# %%
# ============================================================================
# 5. LOAD DATASETS
# ============================================================================

log.start_step("Load Datasets")

try:
    # Load complete setup decisions
    setup_path = METADATA_DIR / "setup_decisions.json"

    if not setup_path.exists():
        raise FileNotFoundError(
            f"setup_decisions.json not found at {setup_path}\n"
            "Run exp_09_feature_reduction.ipynb first to complete setup."
        )

    setup = json.loads(setup_path.read_text())

    # Extract all decisions
    chm_strategy = setup["chm_strategy"]["decision"]
    chm_features = setup["chm_strategy"].get("included_features", [])
    proximity_strategy = setup["proximity_strategy"]["decision"]
    outlier_strategy = setup["outlier_strategy"]["decision"]
    selected_features = setup["selected_features"]

    print(f"\nLoaded setup decisions:")
    print(f"  CHM: {chm_strategy}")
    print(f"  Proximity: {proximity_strategy}")
    print(f"  Outlier: {outlier_strategy}")
    print(f"  Features: {len(selected_features)} selected")

    # ✅ FIXED: No variant parameter needed - always loads baseline files
    # NOTE: proximity_strategy is stored in setup_decisions but all datasets
    #       are now baseline (no _filtered suffix exists anymore)
    train_df, val_df, test_df = data_loading.load_berlin_splits(INPUT_DIR)

    print(f"\nLoaded dataset: {len(train_df):,} samples")

    # Memory optimization
    float_cols = train_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        train_df[float_cols] = train_df[float_cols].astype("float32")
        val_df[float_cols] = val_df[float_cols].astype("float32")
        test_df[float_cols] = test_df[float_cols].astype("float32")
        print(f"Optimized: {len(float_cols)} float64→float32")

    # Apply outlier removal strategy
    if outlier_strategy == "high":
        train_df = train_df[train_df["outlier_severity"] != "high"].copy()
        val_df = val_df[val_df["outlier_severity"] != "high"].copy()
        test_df = test_df[test_df["outlier_severity"] != "high"].copy()
        print(f"Outliers removed (high): {len(train_df):,} remaining")
    elif outlier_strategy == "high_medium":
        train_df = train_df[
            ~train_df["outlier_severity"].isin(["high", "medium"])
        ].copy()
        val_df = val_df[~val_df["outlier_severity"].isin(["high", "medium"])].copy()
        test_df = test_df[~test_df["outlier_severity"].isin(["high", "medium"])].copy()
        print(f"Outliers removed (high+medium): {len(train_df):,} remaining")
    else:
        print("No outlier removal applied")

    # Filter to selected features
    feature_cols = data_loading.get_feature_columns(
        train_df, expected_features=selected_features
    )

    print(f"\nFeature filtering: {len(feature_cols)} features")
    print(f"  Train: {len(train_df):,} samples")
    print(f"  Val:   {len(val_df):,} samples")
    print(f"  Test:  {len(test_df):,} samples")

    log.end_step(status="success", records=len(train_df) + len(val_df) + len(test_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 6. APPLY GENUS SELECTION
# ============================================================================

log.start_step("Apply Genus Selection")

genus_config_path = METADATA_DIR / "genus_selection_final.json"

if not genus_config_path.exists():
    raise FileNotFoundError(
        f"genus_selection_final.json not found at {genus_config_path}\n"
        "Run exp_10_genus_selection_validation.ipynb first to determine final genus list!"
    )

genus_config = json.loads(genus_config_path.read_text())

# Extract genus mapping
FINAL_GENERA = genus_config["decision"]["final_genera_list"]
genus_to_final = genus_config["grouping_analysis"]["genus_to_final_mapping"]
genus_groups = genus_config["grouping_analysis"]["genus_groups"]

print(f"\nLoaded genus selection:")
print(f"  Final classes: {len(FINAL_GENERA)}")
print(f"  Excluded genera: {len(genus_config['validation_results']['excluded_genera'])}")
print(f"  Genus groups: {len(genus_groups)}")

if genus_groups:
    print(f"  Groups:")
    for group_name, genera_list in genus_groups.items():
        print(f"    {group_name}: {', '.join(genera_list)}")

# Apply genus filtering to all splits
print(f"\nMapping genera to final classes...")

for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    original_count = len(split_df)
    
    # Map original genus to final class (handles grouped genera)
    split_df['final_class'] = split_df['genus_latin'].map(genus_to_final)
    
    # Filter to viable genera (removes excluded genera)
    split_df = split_df[split_df['final_class'].notna()].copy()
    
    # Replace genus_latin with final_class for training
    split_df['genus_latin'] = split_df['final_class']
    split_df.drop(columns=['final_class'], inplace=True)
    
    # Update global variables
    if split_name == "train":
        train_df = split_df
    elif split_name == "val":
        val_df = split_df
    else:
        test_df = split_df
    
    print(f"  {split_name}: {original_count:,} → {len(split_df):,} samples")

print(f"\n✅ Genus filtering applied: {len(FINAL_GENERA)} final classes")
print(f"   Total samples: {len(train_df) + len(val_df) + len(test_df):,}")

# Update feature extraction (re-extract after genus filtering)
x_train = train_df[selected_features]
x_val = val_df[selected_features]

y_train = train_df["genus_latin"]
y_val = val_df["genus_latin"]

y_train_enc, label_to_idx, idx_to_label = preprocessing.encode_genus_labels(y_train)
y_val_enc = y_val.map(label_to_idx).to_numpy()

# Re-scale features
x_train_scaled, x_val_scaled, _, scaler = preprocessing.scale_features(
    x_train, x_val=x_val
)

# Re-create CV (groups may have changed)
cv = training.create_spatial_block_cv(
    train_df, n_splits=config["global"]["cv_folds"]
)
groups = train_df["block_id"].values

print(f"\nData prepared with final genus list:")
print(f"  Features: {x_train.shape[1]}")
print(f"  Classes: {len(label_to_idx)} (final)")
print(f"  CV folds: {config['global']['cv_folds']}")

log.end_step(status="success", records=len(train_df))


# %%
# ============================================================================
# 7. BASELINE MODELS
# ============================================================================

log.start_step("Baseline Models")

results = []

print("\n" + "=" * 70)
print("Testing Baseline Models")
print("=" * 70)

# 1. Majority Class Baseline
print("\n1. Majority Class Baseline")
majority = models.create_majority_classifier(y_train_enc)
majority_preds = majority(x_val_scaled)
majority_metrics = evaluation.compute_metrics(y_val_enc, majority_preds)
print(f"   Val F1: {majority_metrics['f1_score']:.4f}")

results.append({
    "algorithm": "majority",
    "type": "baseline",
    "val_f1_mean": majority_metrics["f1_score"],
    "val_f1_std": 0.0,
    "train_f1_mean": majority_metrics["f1_score"],
    "train_val_gap": 0.0,
})

# 2. Stratified Random Baseline
print("\n2. Stratified Random Baseline")
stratified = models.create_stratified_random_classifier(y_train_enc)
strat_preds = stratified(x_val_scaled)
strat_metrics = evaluation.compute_metrics(y_val_enc, strat_preds)
print(f"   Val F1: {strat_metrics['f1_score']:.4f}")

results.append({
    "algorithm": "stratified_random",
    "type": "baseline",
    "val_f1_mean": strat_metrics["f1_score"],
    "val_f1_std": 0.0,
    "train_f1_mean": strat_metrics["f1_score"],
    "train_val_gap": 0.0,
})

# Cleanup
del majority, stratified, majority_preds, strat_preds
del majority_metrics, strat_metrics
gc.collect()

log.end_step(status="success", records=len(results))

# %%
# ============================================================================
# 8. ML MODELS (GRID SEARCH)
# ============================================================================

log.start_step("ML Models Grid Search")

print("\n" + "=" * 70)
print("ML Models: Coarse Grid Search")
print("=" * 70)

for model_name in ["random_forest", "xgboost"]:
    print(f"\n{'='*70}")
    print(f"Grid Search: {model_name}")
    print("="*70)
    
    grid = config["algorithm_comparison"]["models"][model_name]["grid"]
    param_names = list(grid.keys())
    param_values = list(grid.values())
    
    # Calculate total configurations
    n_configs = 1
    for vals in param_values:
        n_configs *= len(vals)
    
    print(f"\nParameters: {param_names}")
    print(f"Total configurations: {n_configs}")
    
    best_val_f1 = -1
    best_params = {}
    best_metrics = {}
    
    # Grid search
    for i, combo in enumerate(product(*param_values), 1):
        params = dict(zip(param_names, combo))
        
        # Train with CV
        model = models.create_model(model_name, model_params=params)
        metrics = training.train_with_cv(model, x_train_scaled, y_train_enc, groups, cv)
        
        # Track best
        if metrics["val_f1_mean"] > best_val_f1:
            best_val_f1 = metrics["val_f1_mean"]
            best_params = params
            best_metrics = metrics
        
        # Progress update every 10 configs
        if i % 10 == 0:
            print(f"  Config {i}/{n_configs}: best F1 = {best_val_f1:.4f}")
        
        # Memory cleanup per iteration
        del model, metrics
        gc.collect()
    
    print(f"\n✅ Best {model_name} configuration:")
    print(f"   Val F1: {best_val_f1:.4f} ± {best_metrics['val_f1_std']:.4f}")
    print(f"   Train-Val Gap: {best_metrics['train_val_gap']:.4f}")
    print(f"   Params: {best_params}")
    
    results.append({
        "algorithm": model_name,
        "type": "ml",
        "val_f1_mean": best_metrics["val_f1_mean"],
        "val_f1_std": best_metrics["val_f1_std"],
        "train_f1_mean": best_metrics["train_f1_mean"],
        "train_val_gap": best_metrics["train_val_gap"],
        "best_params": best_params,
    })

log.end_step(status="success", records=len(results) - 2)  # Subtract baselines

# %%
# ============================================================================
# 9. NEURAL NETWORK MODELS
# ============================================================================

log.start_step("Neural Network Models")

print("\n" + "=" * 70)
print("Neural Network Models")
print("=" * 70)

# Verify PyTorch is available (should be installed via [gpu] extras)
try:
    import torch
    print(f"✅ PyTorch {torch.__version__} available")
except ImportError:
    raise ImportError(
        "PyTorch not found! Algorithm comparison requires neural network models.\n"
        "Fix: Ensure installation used [gpu] extras:\n"
        f'  pip install "{repo_url}[gpu]"'
    )

# 1. CNN-1D (requires PyTorch)
print("\n1. CNN-1D (1D Convolutional Neural Network)")
try:
    # Detect temporal features
    temporal_features = [c for c in selected_features if c.split("_")[-1].isdigit()]
    if len(temporal_features) > 0:
        months = sorted(set(int(c.split("_")[-1]) for c in temporal_features))
        bases = sorted(set("_".join(c.split("_")[:-1]) for c in temporal_features))
        n_temporal_bases = len(bases)
        n_months = len(months)
        n_static = len(selected_features) - n_temporal_bases * n_months
        
        cnn_params = {
            "n_temporal_bases": n_temporal_bases,
            "n_months": n_months,
            "n_static_features": max(n_static, 0),
            "n_classes": len(label_to_idx),
        }
        
        print(f"   Temporal bases: {n_temporal_bases}")
        print(f"   Months: {n_months}")
        print(f"   Static features: {n_static}")
        
        cnn = models.create_model("cnn_1d", cnn_params)
        metrics = training.train_with_cv(cnn, x_train_scaled, y_train_enc, groups, cv)
        
        print(f"   Val F1: {metrics['val_f1_mean']:.4f} ± {metrics['val_f1_std']:.4f}")
        print(f"   Train-Val Gap: {metrics['train_val_gap']:.4f}")
        
        results.append({
            "algorithm": "cnn_1d",
            "type": "nn",
            "val_f1_mean": metrics["val_f1_mean"],
            "val_f1_std": metrics["val_f1_std"],
            "train_f1_mean": metrics["train_f1_mean"],
            "train_val_gap": metrics["train_val_gap"],
        })
        
        del cnn, metrics
        gc.collect()
    else:
        print("   ⚠️  No temporal features detected, skipping CNN-1D")
except Exception as exc:
    print(f"   ⚠️  Skipping CNN-1D: {exc}")

# 2. TabNet (requires pytorch-tabnet)
print("\n2. TabNet (Attention-based Tabular DNN)")
try:
    tabnet = models.create_model("tabnet")
    metrics = training.train_with_cv(tabnet, x_train_scaled, y_train_enc, groups, cv)
    
    print(f"   Val F1: {metrics['val_f1_mean']:.4f} ± {metrics['val_f1_std']:.4f}")
    print(f"   Train-Val Gap: {metrics['train_val_gap']:.4f}")
    
    results.append({
        "algorithm": "tabnet",
        "type": "nn",
        "val_f1_mean": metrics["val_f1_mean"],
        "val_f1_std": metrics["val_f1_std"],
        "train_f1_mean": metrics["train_f1_mean"],
        "train_val_gap": metrics["train_val_gap"],
    })
    
    del tabnet, metrics
    gc.collect()
except Exception as exc:
    print(f"   ⚠️  Skipping TabNet: {exc}")

log.end_step(status="success")


# %%
# ============================================================================
# 10. RESULTS ANALYSIS AND CHAMPION SELECTION
# ============================================================================

log.start_step("Results Analysis")

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("Algorithm Comparison Results")
print("=" * 70)
print(results_df[["algorithm", "type", "val_f1_mean", "val_f1_std", "train_val_gap"]].to_string(index=False))

# Filter viable algorithms
min_f1 = config["algorithm_comparison"]["selection_criteria"]["min_val_f1"]
max_gap = config["algorithm_comparison"]["selection_criteria"]["max_train_val_gap"]

viable = results_df[
    (results_df["val_f1_mean"] >= min_f1) & 
    (results_df["train_val_gap"] < max_gap)
]

print(f"\n" + "=" * 70)
print("Viable Algorithms (Selection Criteria)")
print("=" * 70)
print(f"Criteria: F1 >= {min_f1:.2f}, Gap < {max_gap:.2f}")
print(f"\nViable algorithms: {len(viable)}")
if len(viable) > 0:
    print(viable[["algorithm", "type", "val_f1_mean", "train_val_gap"]].to_string(index=False))

# Select champions
ml_candidates = viable[viable["type"] == "ml"]
nn_candidates = viable[viable["type"] == "nn"]

ml_champion = {} if len(ml_candidates) == 0 else ml_candidates.sort_values("val_f1_mean", ascending=False).iloc[0].to_dict()
nn_champion = {} if len(nn_candidates) == 0 else nn_candidates.sort_values("val_f1_mean", ascending=False).iloc[0].to_dict()

print(f"\n" + "=" * 70)
print("Champions Selected for Hyperparameter Tuning")
print("=" * 70)
print(f"\nML Champion: {ml_champion.get('algorithm', 'none')}")
if ml_champion:
    print(f"  Val F1: {ml_champion['val_f1_mean']:.4f}")
    print(f"  Train-Val Gap: {ml_champion['train_val_gap']:.4f}")

print(f"\nNN Champion: {nn_champion.get('algorithm', 'none')}")
if nn_champion:
    print(f"  Val F1: {nn_champion['val_f1_mean']:.4f}")
    print(f"  Train-Val Gap: {nn_champion['train_val_gap']:.4f}")

log.end_step(status="success")

# %%
# ============================================================================
# N. DECISION
# ============================================================================

log.start_step("Save Results")

# Save algorithm comparison results
algo_path = METADATA_DIR / "algorithm_comparison.json"

algo_results = {
    "algorithms": results,
    "ml_champion": ml_champion,
    "nn_champion": nn_champion,
    "metadata": {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_algorithms_tested": len(results),
        "n_viable": len(viable),
        "selection_criteria": config["algorithm_comparison"]["selection_criteria"],
    },
}

algo_path.write_text(json.dumps(algo_results, indent=2))
print(f"\nSaved: {algo_path}")

# Validate against schema
try:
    validate_algorithm_comparison(algo_path)
    print("✅ Schema validation: PASSED")
except Exception as e:
    print(f"❌ Schema validation: FAILED")
    print(f"   Error: {e}")
    log.end_step(status="error", errors=[str(e)])
    raise

log.end_step(status="success")



# %%
# ============================================================================
# FINDINGS SUMMARY
# ============================================================================

log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

config_files = [algo_path] if algo_path.exists() else []

print("\n" + "=" * 70)
print("OK: NOTEBOOK COMPLETE")
print("=" * 70)
print(f"\nML Champion: {ml_champion.get('algorithm', 'none')}")
print(f"NN Champion: {nn_champion.get('algorithm', 'none')}")
print("\nEXPORT MANIFEST:")
for file_path in sorted(config_files):
    print(f"  - Config: {file_path}")
print(f"  - Logs: {log_path}")
