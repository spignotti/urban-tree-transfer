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

# Install TabNet before package imports so models module resolves dependency correctly
tabnet_install_error = None
try:
    subprocess.run(["pip", "install", "pytorch-tabnet", "-q"], check=True)
    print("OK: pytorch-tabnet installed")
except Exception as exc:
    tabnet_install_error = str(exc)
    print(f"WARN: pytorch-tabnet install failed (TabNet may be skipped): {exc}")

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
ALGO_PATH = METADATA_DIR / "algorithm_comparison.json"

# Re-run controls
FULL_RERUN = False
RERUN_BASELINES = False
RERUN_ML = False
RERUN_NN = False

for d in [METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load experiment configuration
config = load_experiment_config()

existing_results_by_algorithm = {}
model_status = {}
nn_skip_reasons = {}

if ALGO_PATH.exists() and not FULL_RERUN:
    existing_payload = json.loads(ALGO_PATH.read_text())
    for item in existing_payload.get("algorithms", []):
        if "algorithm" in item:
            existing_results_by_algorithm[item["algorithm"]] = item
    print(f"Resume mode: loaded {len(existing_results_by_algorithm)} existing algorithm results")
else:
    print("Fresh mode: no existing results loaded")

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Metadata:               {METADATA_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")
print(f"CV folds:               {config['global']['cv_folds']}")
print(
    "Rerun flags: "
    f"full={FULL_RERUN}, baselines={RERUN_BASELINES}, ml={RERUN_ML}, nn={RERUN_NN}"
)

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

    # Prepare Berlin datasets from setup decisions (same logic as 03a setup fixation)
    # ML: reduced selected features | NN: full temporal features
    def _prepare_split(
        split: str,
        skip_feature_selection: bool,
    ) -> tuple[pd.DataFrame, dict]:
        df, meta = ablation.prepare_ablation_dataset(
            base_path=INPUT_DIR,
            city="berlin",
            split=split,
            setup_decisions=setup,
            return_metadata=True,
            optimize_memory=True,
            skip_feature_selection=skip_feature_selection,
        )
        df = data_loading.fix_missing_genus_german(df)
        return df, meta

    train_df_ml, meta_train_ml = _prepare_split("train", skip_feature_selection=False)
    val_df_ml, meta_val_ml = _prepare_split("val", skip_feature_selection=False)
    test_df_ml, meta_test_ml = _prepare_split("test", skip_feature_selection=False)

    train_df_nn, meta_train_nn = _prepare_split("train", skip_feature_selection=True)
    val_df_nn, meta_val_nn = _prepare_split("val", skip_feature_selection=True)
    test_df_nn, meta_test_nn = _prepare_split("test", skip_feature_selection=True)

    print(f"\nLoaded ML dataset:  {len(train_df_ml):,} train samples")
    print(f"Loaded CNN dataset: {len(train_df_nn):,} train samples")
    print(
        "Setup applied (ML/NN): "
        f"proximity={meta_train_ml['proximity_strategy_used']}/{meta_train_nn['proximity_strategy_used']}, "
        f"outliers_removed={meta_train_ml['outliers_removed']}/{meta_train_nn['outliers_removed']}, "
        f"chm={chm_strategy}"
    )

    # Memory optimization
    def _optimize_float64(
        train_df_local: pd.DataFrame,
        val_df_local: pd.DataFrame,
        test_df_local: pd.DataFrame,
        label: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        float_cols = train_df_local.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            train_df_local[float_cols] = train_df_local[float_cols].astype("float32")
            val_df_local[float_cols] = val_df_local[float_cols].astype("float32")
            test_df_local[float_cols] = test_df_local[float_cols].astype("float32")
            print(f"Optimized {label}: {len(float_cols)} float64→float32")
        return train_df_local, val_df_local, test_df_local

    train_df_ml, val_df_ml, test_df_ml = _optimize_float64(
        train_df_ml, val_df_ml, test_df_ml, "ML"
    )
    train_df_nn, val_df_nn, test_df_nn = _optimize_float64(
        train_df_nn, val_df_nn, test_df_nn, "CNN"
    )

    print(f"\nOutlier strategy applied via setup: {outlier_strategy}")
    print(f"  ML train samples:  {len(train_df_ml):,}")
    print(f"  CNN train samples: {len(train_df_nn):,}")

    # ML feature filtering (reduced selected feature set)
    feature_cols_ml = data_loading.get_feature_columns(
        train_df_ml, expected_features=selected_features
    )
    # NN feature filtering (full temporal features)
    feature_cols_nn = data_loading.get_feature_columns(train_df_nn)

    print(f"\nFeatures:")
    print(f"  ML reduced: {len(feature_cols_ml)}")
    print(f"  CNN full:   {len(feature_cols_nn)}")

    log.end_step(
        status="success",
        records=(
            len(train_df_ml)
            + len(val_df_ml)
            + len(test_df_ml)
            + len(train_df_nn)
            + len(val_df_nn)
            + len(test_df_nn)
        ),
    )

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 6. APPLY GENUS SELECTION (OPTIONAL)
# ============================================================================

log.start_step("Apply Genus Selection")

# Genus selection is stored in setup_decisions.json (expanded by exp_10)
# If not present, skip genus filtering (backwards compatibility)
has_genus_selection = "genus_selection" in setup

if has_genus_selection:
    genus_config = setup["genus_selection"]
    print("Loaded genus selection from setup_decisions.json")
else:
    genus_config = None
    print("No genus selection found in setup_decisions.json (skipping genus filtering)")
    # Create empty config to avoid errors downstream
    genus_config = {
        "decision": {"final_genera_list": []},
        "grouping_analysis": {"genus_to_final_mapping": {}, "genus_groups": {}},
        "validation_results": {"excluded_genera": []},
    }

# Extract genus mapping (only if genus selection exists)
if has_genus_selection:
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

    def _apply_genus_mapping(split_df: pd.DataFrame) -> pd.DataFrame:
        split_df = split_df.copy()
        split_df["final_class"] = split_df["genus_latin"].map(genus_to_final)
        split_df = split_df[split_df["final_class"].notna()].copy()
        split_df["genus_latin"] = split_df["final_class"]
        split_df.drop(columns=["final_class"], inplace=True)
        return split_df

    for split_name, split_df in [
        ("train_ml", train_df_ml),
        ("val_ml", val_df_ml),
        ("test_ml", test_df_ml),
        ("train_nn", train_df_nn),
        ("val_nn", val_df_nn),
        ("test_nn", test_df_nn),
    ]:
        original_count = len(split_df)
        split_df = _apply_genus_mapping(split_df)

        # Update global variables
        if split_name == "train_ml":
            train_df_ml = split_df
        elif split_name == "val_ml":
            val_df_ml = split_df
        elif split_name == "test_ml":
            test_df_ml = split_df
        elif split_name == "train_nn":
            train_df_nn = split_df
        elif split_name == "val_nn":
            val_df_nn = split_df
        else:
            test_df_nn = split_df

        print(f"  {split_name}: {original_count:,} → {len(split_df):,} samples")

    print(f"\n✅ Genus filtering applied: {len(FINAL_GENERA)} final classes")
    print(
        f"   Total ML samples: {len(train_df_ml) + len(val_df_ml) + len(test_df_ml):,}"
    )
    print(
        f"   Total CNN samples: {len(train_df_nn) + len(val_df_nn) + len(test_df_nn):,}"
    )
else:
    # No genus selection - use all genera as-is
    FINAL_GENERA = sorted(train_df_ml["genus_latin"].unique())
    print(f"\nNo genus selection applied - using all {len(FINAL_GENERA)} genera")
    print(f"   Total ML samples: {len(train_df_ml) + len(val_df_ml) + len(test_df_ml):,}")
    print(f"   Total CNN samples: {len(train_df_nn) + len(val_df_nn) + len(test_df_nn):,}")

# Re-extract features after filtering (ML reduced + NN full)
feature_cols_ml = data_loading.get_feature_columns(
    train_df_ml, expected_features=selected_features
)
feature_cols_nn = data_loading.get_feature_columns(train_df_nn)

x_train_ml = train_df_ml[feature_cols_ml]
x_val_ml = val_df_ml[feature_cols_ml]

x_train_nn = train_df_nn[feature_cols_nn]
x_val_nn = val_df_nn[feature_cols_nn]

y_train = train_df_ml["genus_latin"]
y_val = val_df_ml["genus_latin"]

y_train_enc, label_to_idx, idx_to_label = preprocessing.encode_genus_labels(y_train)
y_val_enc = y_val.map(label_to_idx).to_numpy()

# Scale ML and NN separately
x_train_scaled_ml, x_val_scaled_ml, _, scaler_ml = preprocessing.scale_features(
    x_train_ml, x_val=x_val_ml
)
x_train_scaled_nn, x_val_scaled_nn, _, scaler_nn = preprocessing.scale_features(
    x_train_nn, x_val=x_val_nn
)

# Separate CV/groups for ML and NN (same strategy, split-specific groups)
cv_ml = training.create_spatial_block_cv(
    train_df_ml, n_splits=config["global"]["cv_folds"]
)
groups_ml = train_df_ml["block_id"].values

cv_nn = training.create_spatial_block_cv(
    train_df_nn, n_splits=config["global"]["cv_folds"]
)
groups_nn = train_df_nn["block_id"].values

print(f"\nData prepared with final genus list:")
print(f"  ML features: {x_train_ml.shape[1]}")
print(f"  NN features: {x_train_nn.shape[1]}")
print(f"  Classes: {len(label_to_idx)} (final)")
print(f"  CV folds: {config['global']['cv_folds']}")

log.end_step(status="success", records=len(train_df_ml) + len(train_df_nn))


# %%
# ============================================================================
# 7. BASELINE MODELS
# ============================================================================

log.start_step("Baseline Models")

results_by_algorithm = dict(existing_results_by_algorithm)


def _upsert_result(result: dict) -> None:
    results_by_algorithm[result["algorithm"]] = result

print("\n" + "=" * 70)
print("Testing Baseline Models")
print("=" * 70)

# 1. Majority Class Baseline
if (not FULL_RERUN) and (not RERUN_BASELINES) and ("majority" in results_by_algorithm):
    print("\n1. Majority Class Baseline")
    print("   ⏭️  Skipping (already computed)")
    model_status["majority"] = "skipped_existing"
else:
    print("\n1. Majority Class Baseline")
    majority = models.create_majority_classifier(y_train_enc)
    majority_preds = majority(x_val_scaled_ml)
    majority_metrics = evaluation.compute_metrics(y_val_enc, majority_preds)
    print(f"   Val F1: {majority_metrics['f1_score']:.4f}")

    _upsert_result(
        {
            "algorithm": "majority",
            "type": "baseline",
            "val_f1_mean": majority_metrics["f1_score"],
            "val_f1_std": 0.0,
            "train_f1_mean": majority_metrics["f1_score"],
            "train_val_gap": 0.0,
        }
    )
    model_status["majority"] = "completed"

    del majority, majority_preds, majority_metrics
    gc.collect()

# 2. Stratified Random Baseline
if (not FULL_RERUN) and (not RERUN_BASELINES) and ("stratified_random" in results_by_algorithm):
    print("\n2. Stratified Random Baseline")
    print("   ⏭️  Skipping (already computed)")
    model_status["stratified_random"] = "skipped_existing"
else:
    print("\n2. Stratified Random Baseline")
    stratified = models.create_stratified_random_classifier(y_train_enc)
    strat_preds = stratified(x_val_scaled_ml)
    strat_metrics = evaluation.compute_metrics(y_val_enc, strat_preds)
    print(f"   Val F1: {strat_metrics['f1_score']:.4f}")

    _upsert_result(
        {
            "algorithm": "stratified_random",
            "type": "baseline",
            "val_f1_mean": strat_metrics["f1_score"],
            "val_f1_std": 0.0,
            "train_f1_mean": strat_metrics["f1_score"],
            "train_val_gap": 0.0,
        }
    )
    model_status["stratified_random"] = "completed"

    del stratified, strat_preds, strat_metrics
    gc.collect()

baseline_count = sum(1 for item in results_by_algorithm.values() if item.get("type") == "baseline")
log.end_step(status="success", records=baseline_count)

# %%
# ============================================================================
# 8. ML MODELS (GRID SEARCH)
# ============================================================================

log.start_step("ML Models Grid Search")

print("\n" + "=" * 70)
print("ML Models: Coarse Grid Search")
print("=" * 70)

for model_name in ["random_forest", "xgboost"]:
    if (not FULL_RERUN) and (not RERUN_ML) and (model_name in results_by_algorithm):
        print(f"\n{'='*70}")
        print(f"Grid Search: {model_name}")
        print("="*70)
        print("⏭️  Skipping (already computed)")
        model_status[model_name] = "skipped_existing"
        continue

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
        metrics = training.train_with_cv(model, x_train_scaled_ml, y_train_enc, groups_ml, cv_ml)
        
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
    
    _upsert_result(
        {
            "algorithm": model_name,
            "type": "ml",
            "val_f1_mean": best_metrics["val_f1_mean"],
            "val_f1_std": best_metrics["val_f1_std"],
            "train_f1_mean": best_metrics["train_f1_mean"],
            "train_val_gap": best_metrics["train_val_gap"],
            "best_params": best_params,
        }
    )
    model_status[model_name] = "completed"

ml_count = sum(1 for item in results_by_algorithm.values() if item.get("type") == "ml")
log.end_step(status="success", records=ml_count)

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

# TabNet dependency status was resolved in Environment Setup.
if tabnet_install_error is None:
    print("✅ pytorch-tabnet available")
else:
    print(f"⚠️  pytorch-tabnet unavailable (TabNet may be skipped): {tabnet_install_error}")

# 1. CNN-1D (requires PyTorch)
print("\n1. CNN-1D (1D Convolutional Neural Network)")
if (not FULL_RERUN) and (not RERUN_NN) and ("cnn_1d" in results_by_algorithm):
    print("   ⏭️  Skipping (already computed)")
    model_status["cnn_1d"] = "skipped_existing"
else:
    try:
        temporal_features = [f for f in feature_cols_nn if f.split("_")[-1].isdigit()]
        if len(temporal_features) == 0:
            raise ValueError("No temporal features detected in CNN feature dataset.")

        temporal_bases = set("_".join(f.split("_")[:-1]) for f in temporal_features)
        months = sorted(set(int(f.split("_")[-1]) for f in temporal_features))

        n_temporal_bases = len(temporal_bases)
        n_months = len(months)
        n_static = len(feature_cols_nn) - len(temporal_features)

        cnn_params = {
            "n_temporal_bases": n_temporal_bases,
            "n_months": n_months,
            "n_static_features": n_static,
            "n_classes": len(label_to_idx),
        }

        print(f"   Total features:    {len(feature_cols_nn)}")
        print(f"   Temporal features: {len(temporal_features)}")
        print(f"   Temporal bases:    {n_temporal_bases}")
        print(f"   Months:            {n_months}")
        print(f"   Static features:   {n_static}")

        cnn = models.create_model("cnn_1d", cnn_params)
        metrics = training.train_with_cv(cnn, x_train_scaled_nn, y_train_enc, groups_nn, cv_nn)

        print(f"   Val F1: {metrics['val_f1_mean']:.4f} ± {metrics['val_f1_std']:.4f}")
        print(f"   Train-Val Gap: {metrics['train_val_gap']:.4f}")

        _upsert_result(
            {
                "algorithm": "cnn_1d",
                "type": "nn",
                "val_f1_mean": metrics["val_f1_mean"],
                "val_f1_std": metrics["val_f1_std"],
                "train_f1_mean": metrics["train_f1_mean"],
                "train_val_gap": metrics["train_val_gap"],
            }
        )
        model_status["cnn_1d"] = "completed"

        del cnn, metrics
        gc.collect()
    except Exception as exc:
        nn_skip_reasons["cnn_1d"] = str(exc)
        model_status["cnn_1d"] = "skipped_error"
        print(f"   ⚠️  Skipping CNN-1D: {exc}")

# 2. TabNet (requires pytorch-tabnet)
print("\n2. TabNet (Attention-based Tabular DNN)")
if (not FULL_RERUN) and (not RERUN_NN) and ("tabnet" in results_by_algorithm):
    print("   ⏭️  Skipping (already computed)")
    model_status["tabnet"] = "skipped_existing"
else:
    try:
        if tabnet_install_error is not None:
            raise ImportError(f"pytorch-tabnet install failed: {tabnet_install_error}")

        tabnet = models.create_model("tabnet")
        metrics = training.train_with_cv(tabnet, x_train_scaled_nn, y_train_enc, groups_nn, cv_nn)

        print(f"   Val F1: {metrics['val_f1_mean']:.4f} ± {metrics['val_f1_std']:.4f}")
        print(f"   Train-Val Gap: {metrics['train_val_gap']:.4f}")

        _upsert_result(
            {
                "algorithm": "tabnet",
                "type": "nn",
                "val_f1_mean": metrics["val_f1_mean"],
                "val_f1_std": metrics["val_f1_std"],
                "train_f1_mean": metrics["train_f1_mean"],
                "train_val_gap": metrics["train_val_gap"],
            }
        )
        model_status["tabnet"] = "completed"

        del tabnet, metrics
        gc.collect()
    except Exception as exc:
        nn_skip_reasons["tabnet"] = str(exc)
        model_status["tabnet"] = "skipped_error"
        print(f"   ⚠️  Skipping TabNet: {exc}")

nn_count = sum(1 for item in results_by_algorithm.values() if item.get("type") == "nn")
log.end_step(status="success", records=nn_count)


# %%
# ============================================================================
# 10. RESULTS ANALYSIS AND CHAMPION SELECTION
# ============================================================================

log.start_step("Results Analysis")

result_order = ["majority", "stratified_random", "random_forest", "xgboost", "cnn_1d", "tabnet"]
results = [
    results_by_algorithm[name]
    for name in result_order
    if name in results_by_algorithm
]
results.extend(
    [
        value
        for name, value in results_by_algorithm.items()
        if name not in result_order
    ]
)

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
algo_path = ALGO_PATH

algo_results = {
    "algorithms": results,
    "ml_champion": ml_champion,
    "nn_champion": nn_champion,
    "metadata": {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "n_algorithms_tested": len(results),
        "n_viable": len(viable),
        "selection_criteria": config["algorithm_comparison"]["selection_criteria"],
        "resume_mode": not FULL_RERUN,
        "full_rerun": FULL_RERUN,
        "rerun_flags": {
            "baselines": RERUN_BASELINES,
            "ml": RERUN_ML,
            "nn": RERUN_NN,
        },
        "model_status": model_status,
        "nn_skip_reasons": nn_skip_reasons,
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
