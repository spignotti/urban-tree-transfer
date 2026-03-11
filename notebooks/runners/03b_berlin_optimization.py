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
# # 03b Berlin Optimization
#
# **Phase 3.3 - Hyperparameter Tuning & Final Model Training**
#
# This notebook performs Optuna hyperparameter optimization for the ML and NN champions identified in exp_10, trains final models, and conducts comprehensive error analysis.
#
# **Dependencies:**
# - `algorithm_comparison.json` from exp_10 (identifies ML/NN champions)
# - Berlin train/val/test splits from 03a
#
# **Outputs:**
# - `hp_tuning_ml.json` - Optuna results for ML champion
# - `hp_tuning_nn.json` - Optuna results for NN champion
# - `berlin_ml_champion.pkl` - Trained ML model
# - `berlin_nn_champion.pt` - Trained NN model
# - `berlin_scaler.pkl` - Feature scaler
# - `label_encoder.pkl` - Label encoder
# - `berlin_evaluation.json` - Test metrics
# - Error analysis figures (confusion matrix, feature importance, etc.)
#
# ---
#
# **RUNTIME REQUIREMENTS:**
# - CPU: High-RAM recommended (Optuna trials memory-intensive)
# - GPU: Highly recommended for NN champion training
# - Time: ~2-4 hours for full HP tuning (50 trials × 2 models)
#
# **NOTE:** This notebook optimizes whichever models won in exp_10 - it's model-agnostic!

# %%
# ============================================================
# INSTALLATION
# ============================================================
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================

import subprocess
from google.colab import userdata

# Get GitHub token from Colab Secrets
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        "GITHUB_TOKEN not found in Colab Secrets.\n"
        "1. Click the key icon in the left sidebar\n"
        "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
        "3. Toggle 'Notebook access' ON"
    )

# Install package from GitHub
repo_url = f"git+https://{token}@github.com/SilasPignotti/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

# Install Optuna for hyperparameter tuning (always needed)
subprocess.run(["pip", "install", "optuna", "-q"], check=True)

# Install PyTorch for neural network models (if NN champion exists)
subprocess.run(["pip", "install", "torch>=2.2.0", "-q"], check=True)

print("✅ Package installed successfully")
print("✅ Optuna installed (hyperparameter tuning)")
print("✅ PyTorch installed (neural network training)")

# %%
# ============================================================
# GOOGLE DRIVE MOUNT
# ============================================================

from google.colab import drive

drive.mount("/content/drive")

print("✅ Google Drive mounted")

# %%
# ============================================================
# IMPORTS & INITIALIZATION
# ============================================================

from pathlib import Path
from datetime import datetime, timezone
import json
import pickle
import warnings
import time
import gc

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit

from urban_tree_transfer.config import load_experiment_config, RANDOM_SEED
from urban_tree_transfer.experiments import (
    data_loading,
    evaluation,
    hp_tuning,
    models,
    preprocessing,
    training,
)
from urban_tree_transfer.utils import (
    ExecutionLog,
    validate_algorithm_comparison,
    validate_evaluation_metrics,
    validate_hp_tuning_result,
    validate_setup_decisions,
)
warnings.filterwarnings("ignore", category=UserWarning)

log = ExecutionLog("03b_berlin_optimization")
start_time = time.time()

print("✅ Imports complete")

# %%
# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
DATA_DIR = DRIVE_DIR / "data"
INPUT_DIR = DATA_DIR / "phase_3_experiments"
OUTPUT_DIR = DATA_DIR / "phase_3_experiments"

METADATA_DIR = OUTPUT_DIR / "metadata"
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

# Create output directories
for d in [METADATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load experiment configuration
config = load_experiment_config()

print(f"Input (Phase 3 Datasets): {INPUT_DIR}")
print(f"Output (Phase 3):         {OUTPUT_DIR}")
print(f"Metadata:                 {METADATA_DIR}")
print(f"Models:                   {MODELS_DIR}")
print(f"Logs:                     {LOGS_DIR}")
print(f"Random seed:              {RANDOM_SEED}")
print(f"CV folds:                 {config['global']['cv_folds']}")
print(f"Optuna trials:            {config['hp_tuning']['optuna']['n_trials']}")
print(f"\n✅ Configuration complete")

# %%
# ============================================================
# SECTION 1: Load Algorithm Comparison & Data
# ============================================================

log.start_step("Load Algorithm Comparison & Data")

try:
    # Load algorithm comparison results
    algo_path = METADATA_DIR / "algorithm_comparison.json"
    
    if not algo_path.exists():
        raise FileNotFoundError(
            f"algorithm_comparison.json not found at {algo_path}\n"
            "Run exp_11_algorithm_comparison.ipynb first."
        )
    
    algorithm_comparison = validate_algorithm_comparison(algo_path)
    
    ml_name = algorithm_comparison["ml_champion"]["algorithm"]
    nn_name = algorithm_comparison.get("nn_champion", {}).get("algorithm")
    
    print("\n" + "=" * 70)
    print("Champions Identified")
    print("=" * 70)
    print(f"ML Champion: {ml_name}")
    print(f"NN Champion: {nn_name if nn_name else 'None'}")
    print("=" * 70)
    
    # Load setup decisions for feature validation
    setup_path = METADATA_DIR / "setup_decisions.json"
    if not setup_path.exists():
        raise FileNotFoundError(
            f"setup_decisions.json not found at {setup_path}\n"
            "Run exp_09 + exp_10 + 03a first."
        )
    
    setup = validate_setup_decisions(setup_path)
    expected_features_reduced = setup["selected_features"]
    
    # ============================================================
    # DATASET LOADING: Automatic selection based on champion type
    # ============================================================
    # ML Champion (tree-based) → Use reduced features (top-50)
    # NN Champion (CNN1D) → Use full temporal features (~144)
    # ============================================================
    
    print(f"\n" + "=" * 70)
    print("Loading Datasets")
    print("=" * 70)
    
    # Load ML datasets (reduced features - always needed)
    print(f"\n[ML Champion Datasets - Reduced Features]")
    train_df_ml, val_df_ml, test_df_ml = data_loading.load_berlin_splits(INPUT_DIR)
    feature_cols_ml = data_loading.get_feature_columns(
        train_df_ml,
        expected_features=expected_features_reduced,
    )
    
    print(f"  Train: {len(train_df_ml):,} samples, {len(feature_cols_ml)} features")
    print(f"  Val:   {len(val_df_ml):,} samples")
    print(f"  Test:  {len(test_df_ml):,} samples")
    
    # Memory optimization for ML datasets
    float_cols = train_df_ml.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        train_df_ml[float_cols] = train_df_ml[float_cols].astype("float32")
        val_df_ml[float_cols] = val_df_ml[float_cols].astype("float32")
        test_df_ml[float_cols] = test_df_ml[float_cols].astype("float32")
        print(f"  Optimized: {len(float_cols)} float64→float32")
    
    # Set default datasets for backward compatibility
    train_df, val_df, test_df = train_df_ml, val_df_ml, test_df_ml
    feature_cols = feature_cols_ml
    
    # Load NN datasets if NN champion exists (full features)
    if nn_name:
        print(f"\n[NN Champion Datasets - Full Temporal Features]")
        train_df_nn, val_df_nn, test_df_nn = data_loading.load_berlin_splits_cnn(INPUT_DIR)
        feature_cols_nn = data_loading.get_feature_columns(train_df_nn)
        
        print(f"  Train: {len(train_df_nn):,} samples, {len(feature_cols_nn)} features")
        print(f"  Val:   {len(val_df_nn):,} samples")
        print(f"  Test:  {len(test_df_nn):,} samples")
        
        # Memory optimization for NN datasets
        float_cols_nn = train_df_nn.select_dtypes(include=["float64"]).columns
        if len(float_cols_nn) > 0:
            train_df_nn[float_cols_nn] = train_df_nn[float_cols_nn].astype("float32")
            val_df_nn[float_cols_nn] = val_df_nn[float_cols_nn].astype("float32")
            test_df_nn[float_cols_nn] = test_df_nn[float_cols_nn].astype("float32")
            print(f"  Optimized: {len(float_cols_nn)} float64→float32")
        
        print(f"\nℹ️  Dual datasets loaded:")
        print(f"   - ML champion will use {len(feature_cols_ml)} reduced features")
        print(f"   - NN champion will use {len(feature_cols_nn)} full temporal features")
    else:
        train_df_nn, val_df_nn, test_df_nn = None, None, None
        feature_cols_nn = None
        print(f"\nℹ️  Single dataset loaded (ML champion only)")
    
    log.end_step(status="success", records=len(train_df) + len(val_df) + len(test_df))
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 2: Preprocessing
# ============================================================

log.start_step("Preprocessing")

try:
    print(f"\n" + "=" * 70)
    print("Preprocessing")
    print("=" * 70)
    
    # ========================================
    # ML Champion Preprocessing (Reduced Features)
    # ========================================
    print(f"\n[ML Champion - Reduced Features]")
    
    # Feature scaling
    print(f"Scaling features...")
    x_train_scaled_ml, x_val_scaled_ml, x_test_scaled_ml, scaler_ml = preprocessing.scale_features(
        train_df_ml[feature_cols_ml],
        x_val=val_df_ml[feature_cols_ml],
        x_test=test_df_ml[feature_cols_ml],
    )
    
    # Label encoding (same for both ML and NN)
    print(f"Encoding labels...")
    y_train, label_to_idx, idx_to_label = preprocessing.encode_genus_labels(
        train_df_ml["genus_latin"]
    )
    ml_sample_weights = preprocessing.compute_sample_weights(y_train)
    print(
        f"Computed ML sample weights for {len(ml_sample_weights):,} training samples "
        f"({np.unique(ml_sample_weights).size} class weights)"
    )
    y_val = val_df_ml["genus_latin"].map(label_to_idx).to_numpy()
    y_test = test_df_ml["genus_latin"].map(label_to_idx).to_numpy()
    
    n_classes = len(label_to_idx)
    
    # Build tuning subset for ML (fast HP tuning)
    tuning_subset_n = int(config["hp_tuning"].get("tuning_subset_n", 100000))
    tuning_subset_n = min(tuning_subset_n, len(train_df_ml))
    tuning_cv_folds = int(config["hp_tuning"].get("tuning_cv_folds", config["global"]["cv_folds"]))
    tuning_holdout = float(config["hp_tuning"].get("tuning_holdout", 0.2))
    
    if tuning_subset_n < len(train_df_ml):
        rng = np.random.default_rng(RANDOM_SEED)
        counts = train_df_ml["genus_latin"].value_counts()
        target = tuning_subset_n
        per_class = (counts / counts.sum() * target).round().astype(int)
        per_class = per_class.clip(lower=1)
        subset_indices = []
        for genus, n in per_class.items():
            genus_idx = train_df_ml.index[train_df_ml["genus_latin"] == genus]
            n = min(int(n), len(genus_idx))
            if n > 0:
                subset_indices.extend(rng.choice(genus_idx, size=n, replace=False))
        subset_indices = pd.Index(subset_indices).unique()
        if len(subset_indices) > target:
            subset_indices = pd.Index(rng.choice(subset_indices, size=target, replace=False))
        elif len(subset_indices) < target:
            remaining = train_df_ml.index.difference(subset_indices)
            add_n = target - len(subset_indices)
            if add_n > 0 and len(remaining) > 0:
                add_n = min(add_n, len(remaining))
                subset_indices = subset_indices.append(
                    pd.Index(rng.choice(remaining, size=add_n, replace=False))
                )
        subset_positions = train_df_ml.index.get_indexer(subset_indices)
        subset_positions = subset_positions[subset_positions >= 0]
        subset_positions = np.sort(subset_positions)
        tuning_df_ml = train_df_ml.iloc[subset_positions]
    else:
        tuning_df_ml = train_df_ml.copy()
        subset_positions = np.arange(len(train_df_ml))

    x_tune_scaled_ml = x_train_scaled_ml[subset_positions]
    y_tune = y_train[subset_positions]
    tune_sample_weights = preprocessing.compute_sample_weights(y_tune)
    groups_tune = tuning_df_ml["block_id"].values

    if tuning_cv_folds <= 1:
        cv_tuning = GroupShuffleSplit(
            n_splits=1,
            test_size=tuning_holdout,
            random_state=RANDOM_SEED,
        )
    else:
        cv_tuning = training.create_spatial_block_cv(
            tuning_df_ml, n_splits=tuning_cv_folds
        )

    print(f"Tuning subset: {len(tuning_df_ml):,} samples (cv_folds={tuning_cv_folds})")

    # Combine train+val for final training (ML)
    print(f"Combining train+val for final models...")
    x_combined_scaled_ml = np.vstack([x_train_scaled_ml, x_val_scaled_ml])
    y_combined = np.concatenate([y_train, y_val])
    combined_sample_weights = preprocessing.compute_sample_weights(y_combined)
    
    print(f"  Train samples:    {len(x_train_scaled_ml):,}")
    print(f"  Val samples:      {len(x_val_scaled_ml):,}")
    print(f"  Combined:         {len(x_combined_scaled_ml):,}")
    print(f"  Test samples:     {len(y_test):,}")
    print(f"  Classes:          {n_classes}")
    print(f"  Features (ML):    {x_train_scaled_ml.shape[1]}")
    
    # Set defaults for backward compatibility
    x_train_scaled, x_val_scaled, x_test_scaled = x_train_scaled_ml, x_val_scaled_ml, x_test_scaled_ml
    x_combined_scaled = x_combined_scaled_ml
    x_tune_scaled = x_tune_scaled_ml
    scaler = scaler_ml
    
    # ========================================
    # NN Champion Preprocessing (Full Features)
    # ========================================
    if nn_name:
        print(f"\n[NN Champion - Full Temporal Features]")
        print(f"Scaling features...")
        x_train_scaled_nn, x_val_scaled_nn, x_test_scaled_nn, scaler_nn = preprocessing.scale_features(
            train_df_nn[feature_cols_nn],
            x_val=val_df_nn[feature_cols_nn],
            x_test=test_df_nn[feature_cols_nn],
        )
        
        # Use same tuning subset positions for NN (same tree IDs)
        x_tune_scaled_nn = x_train_scaled_nn[subset_positions]
        
        # Combine train+val for final training (NN)
        x_combined_scaled_nn = np.vstack([x_train_scaled_nn, x_val_scaled_nn])
        
        print(f"  Features (NN):    {x_train_scaled_nn.shape[1]}")
        print(f"  Combined:         {len(x_combined_scaled_nn):,}")
    else:
        x_train_scaled_nn, x_val_scaled_nn, x_test_scaled_nn = None, None, None
        x_combined_scaled_nn, x_tune_scaled_nn, scaler_nn = None, None, None
    
    # Save label encoder (M4)
    label_encoder_data = {
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
    }
    label_encoder_path = MODELS_DIR / "label_encoder.pkl"
    with label_encoder_path.open("wb") as f:
        pickle.dump(label_encoder_data, f)
    
    print(f"\n✅ Saved: {label_encoder_path.name}")
    
    # Memory cleanup
    gc.collect()
    
    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 3: ML Champion - Hyperparameter Tuning (C2)
# ============================================================

log.start_step("ML Hyperparameter Tuning")

try:
    print(f"\n" + "=" * 70)
    print(f"ML CHAMPION: {ml_name.upper()}")
    print("=" * 70)
    
    ml_hp_path = METADATA_DIR / "hp_tuning_ml.json"
    if ml_hp_path.exists():
        ml_hp_results = json.loads(ml_hp_path.read_text())
        print(f"\n✅ Found existing {ml_hp_path.name} - skipping Optuna")
        log.end_step(status="skipped", records=ml_hp_results.get("n_trials", 0))
    else:
        # Get search space from config
        ml_search_space = config["hp_tuning"]["search_spaces"][ml_name]
    
        print(f"\nSearch space:")
        for param, values in ml_search_space.items():
            print(f"  {param}: {values}")
    
        def compute_search_space_size(search_space: dict) -> int:
            total = 1
            for value in search_space.values():
                if isinstance(value, dict):
                    total *= compute_search_space_size(value)
                elif isinstance(value, list):
                    total *= len(value)
            return total

        total_grid_size = compute_search_space_size(ml_search_space)

        # Create Optuna study with pruning for efficiency
        ml_study = hp_tuning.create_study(
            direction="maximize",
            sampler=config["hp_tuning"]["optuna"]["sampler"],
            pruner=config["hp_tuning"]["optuna"]["pruner"],
            random_seed=RANDOM_SEED,
        )
    
        ml_objective = hp_tuning.build_objective(
            model_name=ml_name,
            x=x_tune_scaled,
            y=y_tune,
            groups=groups_tune,
            cv=cv_tuning,
            search_space=ml_search_space,
            sample_weight=tune_sample_weights,
        )
    
        # Run Optuna optimization
        n_trials = config["hp_tuning"]["optuna"]["n_trials"]
        timeout = config["hp_tuning"]["optuna"]["timeout_seconds"]
    
        print(f"\nRunning Optuna optimization...")
        print(f"  Trials: {n_trials}")
        print(f"  Timeout: {timeout}s ({timeout/3600:.1f}h)")
        print(f"  Sampler: {config['hp_tuning']['optuna']['sampler']}")
        print(f"  Pruner: {config['hp_tuning']['optuna']['pruner']}")
        print(
            f"NOTE: Optuna evaluated {n_trials} of {total_grid_size} possible combinations. "
            "Champion selection is based on partial search results."
        )
    
        ml_hp_results = hp_tuning.run_optuna_search(
            ml_study,
            ml_objective,
            n_trials=n_trials,
            timeout_seconds=timeout,
            model_name=ml_name,
        )
    
        # Save HP tuning results
        ml_hp_path.write_text(json.dumps(ml_hp_results, indent=2))
        validate_hp_tuning_result(ml_hp_path)
    
        print(f"\n" + "=" * 70)
        print("Best ML Hyperparameters")
        print("=" * 70)
        for k, v in ml_hp_results["best_params"].items():
            print(f"  {k}: {v}")
        print(f"\n  CV F1: {ml_hp_results['best_value']:.4f}")
        print(f"  Trials completed: {ml_hp_results['n_trials']}")
        print("=" * 70)
    
        print(f"\n✅ Saved: {ml_hp_path.name}")
    
        # Memory cleanup
        del ml_study, ml_objective
        gc.collect()
    
        log.end_step(status="success", records=ml_hp_results["n_trials"])
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 4: ML Champion - Final Training
# ============================================================

log.start_step("ML Final Training")

try:
    print(f"\n" + "=" * 70)
    print("Training Final ML Model")
    print("=" * 70)

    ml_model_path = MODELS_DIR / "berlin_ml_champion.pkl"
    if ml_model_path.exists():
        ml_model = training.load_model(ml_model_path)
        print(f"\n✅ Found existing {ml_model_path.name} - skipping training")
        log.end_step(status="skipped")
    else:
        ml_best_params = ml_hp_results["best_params"]

        # Create model with best params
        ml_model = models.create_model(ml_name, model_params=ml_best_params)

        print(f"\nTraining on combined train+val ({len(x_combined_scaled):,} samples)...")
        ml_model = training.train_final_model(
            ml_model,
            x_combined_scaled,
            y_combined,
            sample_weight=combined_sample_weights,
        )

        # Save ML model
        training.save_model(
            ml_model,
            ml_model_path,
            metadata={
                "model_name": ml_name,
                "feature_columns": feature_cols,
                "label_to_idx": label_to_idx,
                "idx_to_label": idx_to_label,
                "best_params": ml_best_params,
                "hp_tuning_cv_f1": ml_hp_results["best_value"],
                "n_features": len(feature_cols),
                "n_classes": n_classes,
                "training_samples": len(x_combined_scaled),
            },
        )

        print(f"\n✅ Saved: {ml_model_path.name}")

        log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 5: NN Champion - Hyperparameter Tuning (C3)
# ============================================================

log.start_step("NN Hyperparameter Tuning")

try:
    if not nn_name:
        print(f"\nℹ️  No NN champion - skipping NN optimization")
        nn_hp_results = None
        log.end_step(status="skipped")
    else:
        print(f"\n" + "=" * 70)
        print(f"NN CHAMPION: {nn_name.upper()}")
        print("=" * 70)
        print(
            "KNOWN ISSUE: TabNet evaluation was not possible due to pytorch-tabnet "
            "dependency conflicts in Colab. NN comparison is limited to CNN-1D."
        )

        # Add structural params for CNN1D (H4)
        structural_params = {}
        if nn_name == "cnn_1d":
            # IMPORTANT: Detect temporal features from FULL feature set (CNN datasets)
            # NOT from reduced feature set - CNN needs complete temporal sequences
            temporal_features = [f for f in feature_cols_nn if f.split("_")[-1].isdigit()]
            temporal_bases = set("_".join(f.split("_")[:-1]) for f in temporal_features)
            months = sorted(set(int(f.split("_")[-1]) for f in temporal_features))

            n_temporal_bases = len(temporal_bases)
            n_months = len(months)
            n_static = len(feature_cols_nn) - len(temporal_features)

            structural_params = {
                "n_temporal_bases": n_temporal_bases,
                "n_months": n_months,
                "n_static_features": n_static,
                "n_classes": n_classes,
            }

            print(f"\nCNN1D structure (auto-detected from FULL features):")
            print(f"  Total features:    {len(feature_cols_nn)}")
            print(f"  Temporal features: {len(temporal_features)}")
            print(f"  Temporal bases:    {n_temporal_bases}")
            print(f"  Months detected:   {n_months} ({min(months)}-{max(months)})")
            print(f"  Static features:   {n_static}")
            print(f"  Classes:           {n_classes}")

        nn_hp_path = METADATA_DIR / "hp_tuning_nn.json"
        if nn_hp_path.exists():
            nn_hp_results = json.loads(nn_hp_path.read_text())
            print(f"\n✅ Found existing {nn_hp_path.name} - skipping Optuna")
            log.end_step(status="skipped", records=nn_hp_results.get("n_trials", 0))
        else:
            # Get search space from config
            nn_search_space = config["hp_tuning"]["search_spaces"][nn_name]

            print(f"\nSearch space:")
            for param, values in nn_search_space.items():
                print(f"  {param}: {values}")

            # Create NN study
            nn_study = hp_tuning.create_study(
                direction="maximize",
                sampler=config["hp_tuning"]["optuna"]["sampler"],
                pruner=config["hp_tuning"]["optuna"]["pruner"],
                random_seed=RANDOM_SEED,
            )

            # IMPORTANT: Use NN datasets (full features) for hyperparameter tuning
            nn_objective = hp_tuning.build_objective(
                model_name=nn_name,
                x=x_tune_scaled_nn,  # Use NN tuning subset with full features
                y=y_tune,
                groups=groups_tune,
                cv=cv_tuning,
                search_space=nn_search_space,
                base_params=structural_params,  # Pass structural params as base_params
            )

            n_trials = config["hp_tuning"]["optuna"]["n_trials"]
            timeout = config["hp_tuning"]["optuna"]["timeout_seconds"]

            print(f"\nRunning Optuna optimization...")
            print(f"  Trials: {n_trials}")
            print(f"  Timeout: {timeout}s ({timeout/3600:.1f}h)")

            nn_hp_results = hp_tuning.run_optuna_search(
                nn_study,
                nn_objective,
                n_trials=n_trials,
                timeout_seconds=timeout,
                model_name=nn_name,
            )

            # Save NN HP results
            nn_hp_path.write_text(json.dumps(nn_hp_results, indent=2))
            validate_hp_tuning_result(nn_hp_path)

            print(f"\n" + "=" * 70)
            print("Best NN Hyperparameters")
            print("=" * 70)
            for k, v in nn_hp_results["best_params"].items():
                print(f"  {k}: {v}")
            print(f"\n  CV F1: {nn_hp_results['best_value']:.4f}")
            print(f"  Trials completed: {nn_hp_results['n_trials']}")
            print("=" * 70)

            print(f"\n✅ Saved: {nn_hp_path.name}")

            # Memory cleanup
            del nn_study, nn_objective
            gc.collect()

            log.end_step(status="success", records=nn_hp_results["n_trials"])

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 6: NN Champion - Final Training
# ============================================================

log.start_step("NN Final Training")

try:
    if not nn_name or nn_hp_results is None:
        print(f"\nℹ️  No NN champion - skipping NN training")
        log.end_step(status="skipped")
    else:
        print(f"\n" + "=" * 70)
        print("Training Final NN Model")
        print("=" * 70)

        nn_model_path = MODELS_DIR / "berlin_nn_champion.pt"
        if nn_model_path.exists():
            metadata_path = nn_model_path.with_suffix(nn_model_path.suffix + ".metadata.json")
            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Metadata not found for existing NN model: {metadata_path}"
                )
            nn_meta = json.loads(metadata_path.read_text())
            nn_best_params = nn_meta.get("best_params", {})
            model_class = models.CNN1D if nn_name == "cnn_1d" else None
            nn_model = training.load_model(
                nn_model_path,
                model_class=model_class,
                model_params=nn_best_params,
            )
            print(f"\n✅ Found existing {nn_model_path.name} - skipping training")
            log.end_step(status="skipped")
        else:
            nn_best_params = nn_hp_results["best_params"].copy()

            # Add structural params for CNN1D
            if nn_name == "cnn_1d":
                nn_best_params.update(structural_params)

            # Create model with best params
            nn_model = models.create_model(nn_name, model_params=nn_best_params)

            # IMPORTANT: Train on NN datasets (full features)
            print(f"\nTraining on combined train+val ({len(x_combined_scaled_nn):,} samples)...")
            print(f"  Features: {x_combined_scaled_nn.shape[1]} (full temporal sequences)")

            # Extract training params
            fit_params = {}
            if nn_name == "cnn_1d":
                fit_params = {
                    "batch_size": nn_best_params.get("batch_size", 64),
                    "epochs": nn_best_params.get("epochs", 50),
                    "learning_rate": nn_best_params.get("learning_rate", 0.001),
                    "early_stopping_patience": nn_best_params.get("early_stopping_patience", 10),
                }

            nn_model = training.train_final_model(
                nn_model, x_combined_scaled_nn, y_combined, fit_params=fit_params
            )

            # Save NN model
            training.save_model(
                nn_model,
                nn_model_path,
                metadata={
                    "model_name": nn_name,
                    "feature_columns": feature_cols_nn,  # Save full feature list
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                    "best_params": nn_best_params,
                    "hp_tuning_cv_f1": nn_hp_results["best_value"],
                    "n_features": len(feature_cols_nn),
                    "n_classes": n_classes,
                    "training_samples": len(x_combined_scaled_nn),
                },
            )

            print(f"\n✅ Saved: {nn_model_path.name}")

            log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 7: Save Scalers
# ============================================================

log.start_step("Save Scalers")

try:
    # Save ML scaler (reduced features)
    scaler_ml_path = MODELS_DIR / "berlin_scaler_ml.pkl"
    with scaler_ml_path.open("wb") as f:
        pickle.dump(scaler_ml, f)
    print(f"✅ Saved: {scaler_ml_path.name} (ML - reduced features)")
    
    # Save NN scaler if NN champion exists (full features)
    if nn_name and scaler_nn is not None:
        scaler_nn_path = MODELS_DIR / "berlin_scaler_nn.pkl"
        with scaler_nn_path.open("wb") as f:
            pickle.dump(scaler_nn, f)
        print(f"✅ Saved: {scaler_nn_path.name} (NN - full temporal features)")
    
    # Save primary scaler for backward compatibility (ML scaler)
    scaler_path = MODELS_DIR / "berlin_scaler.pkl"
    with scaler_path.open("wb") as f:
        pickle.dump(scaler_ml, f)
    print(f"✅ Saved: {scaler_path.name} (backward compatibility)")
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 8: Evaluation on Test Set
# ============================================================

log.start_step("Evaluation")

try:
    print(f"\n" + "=" * 70)
    print("EVALUATION ON TEST SET")
    print("=" * 70)
    
    # ML predictions (using ML test data with reduced features)
    print(f"\nEvaluating ML champion ({ml_name})...")
    print(f"  Features: {x_test_scaled_ml.shape[1]} (reduced)")
    ml_preds = ml_model.predict(x_test_scaled_ml)
    ml_metrics = evaluation.compute_metrics(y_test, ml_preds)
    ml_per_class = evaluation.compute_per_class_metrics(
        y_test, ml_preds, list(idx_to_label.values())
    )
    ml_cm = evaluation.compute_confusion_matrix(
        y_test, ml_preds, labels=list(range(n_classes))
    )
    ml_confidence_intervals = {}
    for metric_name in ["f1_score", "precision", "recall", "accuracy"]:
        ml_confidence_intervals[metric_name] = evaluation.bootstrap_confidence_interval(
            y_test,
            ml_preds,
            lambda y_true_sample, y_pred_sample, key=metric_name: evaluation.compute_metrics(
                y_true_sample,
                y_pred_sample,
            )[key],
            n_bootstrap=config["metrics"]["n_bootstrap"],
            confidence_level=config["metrics"]["confidence_level"],
        )
    
    print(f"\nML Champion Test Results:")
    for metric, value in ml_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # Compute CV-Test gap for ML (generalization within Berlin)
    ml_cv_f1 = ml_hp_results["best_value"]
    ml_test_f1 = ml_metrics["f1_score"]
    ml_cv_test_gap = ml_cv_f1 - ml_test_f1

    print(f"\nML Generalization (CV → Test):")
    print(f"  CV F1:        {ml_cv_f1:.4f}")
    print(f"  Test F1:      {ml_test_f1:.4f}")
    print(f"  CV-Test Gap:  {ml_cv_test_gap:+.4f}")

    # NN predictions (using NN test data with full features - if exists)
    nn_cv_test_gap = None
    nn_confidence_intervals = None
    if nn_name and "nn_model" in locals():
        print(f"\nEvaluating NN champion ({nn_name})...")
        print(f"  Features: {x_test_scaled_nn.shape[1]} (full temporal)")
        nn_preds = nn_model.predict(x_test_scaled_nn)
        nn_metrics = evaluation.compute_metrics(y_test, nn_preds)
        nn_confidence_intervals = {}
        for metric_name in ["f1_score", "precision", "recall", "accuracy"]:
            nn_confidence_intervals[metric_name] = evaluation.bootstrap_confidence_interval(
                y_test,
                nn_preds,
                lambda y_true_sample, y_pred_sample, key=metric_name: evaluation.compute_metrics(
                    y_true_sample,
                    y_pred_sample,
                )[key],
                n_bootstrap=config["metrics"]["n_bootstrap"],
                confidence_level=config["metrics"]["confidence_level"],
            )

        print(f"\nNN Champion Test Results:")
        for metric, value in nn_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Compute CV-Test gap for NN
        if nn_hp_results:
            nn_cv_f1 = nn_hp_results["best_value"]
            nn_test_f1 = nn_metrics["f1_score"]
            nn_cv_test_gap = nn_cv_f1 - nn_test_f1

            print(f"\nNN Generalization (CV → Test):")
            print(f"  CV F1:        {nn_cv_f1:.4f}")
            print(f"  Test F1:      {nn_test_f1:.4f}")
            print(f"  CV-Test Gap:  {nn_cv_test_gap:+.4f}")
    else:
        nn_metrics = None

    # Save evaluation results (schema-compliant)
    eval_data = {
        "metrics": ml_metrics,
        "cv_test_gap": ml_cv_test_gap,
        "confidence_intervals": {
            "ml": ml_confidence_intervals,
            "nn": nn_confidence_intervals,
        },
        "per_class": ml_per_class.to_dict(orient="records"),
        "confusion_matrix": ml_cm.tolist(),
        "metadata": {
            "ml_champion": ml_name,
            "nn_champion": nn_name if nn_name else None,
            "nn_metrics": nn_metrics,
            "nn_cv_test_gap": nn_cv_test_gap,
            "test_samples": len(y_test),
            "ml_features": len(feature_cols_ml),
            "nn_features": len(feature_cols_nn) if nn_name else None,
        },
    }
    
    metrics_path = METADATA_DIR / "berlin_evaluation.json"
    metrics_path.write_text(json.dumps(eval_data, indent=2))
    validate_evaluation_metrics(metrics_path)
    
    print(f"\n✅ Saved: {metrics_path.name}")
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 9: Error Analysis (H4)
# ============================================================

log.start_step("Error Analysis")

try:
    print(f"\n" + "=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)
    
    class_labels = list(idx_to_label.values())
    
    # 3. Worst confused pairs (FIXED: top_k instead of top_n)
    print(f"3. Analyzing worst confused pairs...")
    confused_pairs = evaluation.analyze_worst_confused_pairs(
        y_test,
        ml_preds,
        class_labels,
        top_k=10,  # ✅ FIXED: was top_n
    )
    
    if len(confused_pairs) > 0:
        print(f"\n   Top 5 confused pairs:")
        for _, pair in confused_pairs.head(5).iterrows():
            print(
                f"     {pair['true_genus']} → {pair['pred_genus']}: "
                f"{pair['count']} errors ({pair['confusion_rate']:.2%})"
            )
    else:
        print(f"\n   No confusion pairs found")
    
    # 4. Conifer vs Deciduous (FIXED: removed undefined feature_cfg)
    print(f"\n4. Analyzing conifer vs deciduous...")
    cd_metrics = evaluation.analyze_conifer_deciduous(
        y_test,
        ml_preds,
        class_labels,
        config["genus_groups"],  # ✅ Correct: from experiment config
    )
    
    print(f"   Conifer F1:   {cd_metrics['conifer_f1']:.4f} (n={cd_metrics['conifer_n']})")
    print(
        f"   Deciduous F1: {cd_metrics['deciduous_f1']:.4f} "
        f"(n={cd_metrics['deciduous_n']})"
    )
    
    # 5. Feature importance (if available)
    print(f"\n5. Computing feature importance...")
    if hasattr(ml_model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": feature_cols,  # Uses feature_cols from Section 1
                "importance": ml_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        
        print(f"\n   Top 5 features:")
        for _, row in importance_df.head(5).iterrows():
            print(f"     {row['feature']}: {row['importance']:.4f}")
    else:
        print(f"   Feature importance not available for {ml_name}")
    
    print(f"\n✅ Error analysis complete")
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================
# SECTION 10: Save Execution Log
# ============================================================

log.start_step("Save Execution Log")

try:
    elapsed = time.time() - start_time
    
    # Create execution summary
    execution_summary = {
        "notebook": "03b_berlin_optimization",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60,
        "champions": {
            "ml": ml_name,
            "nn": nn_name if nn_name else None,
        },
        "ml_hp_tuning": {
            "n_trials": ml_hp_results["n_trials"],
            "best_cv_f1": ml_hp_results["best_value"],
            "best_params": ml_hp_results["best_params"],
        },
        "nn_hp_tuning": (
            {
                "n_trials": nn_hp_results["n_trials"],
                "best_cv_f1": nn_hp_results["best_value"],
                "best_params": nn_hp_results["best_params"],
            }
            if nn_hp_results
            else None
        ),
        "test_metrics": {
            "ml": ml_metrics,
            "nn": nn_metrics if nn_metrics else None,
        },
        "generalization": {
            "ml_cv_test_gap": ml_cv_test_gap,
            "nn_cv_test_gap": nn_cv_test_gap if nn_cv_test_gap is not None else None,
        },
    }
    
    # Save execution log
    log.summary()
    log_path = LOGS_DIR / f"{log.notebook}_execution.json"
    log.save(log_path)
    
    # Save summary
    summary_path = METADATA_DIR / "03b_summary.json"
    summary_path.write_text(json.dumps(execution_summary, indent=2))
    
    print(f"\n✅ Execution log saved: {log_path}")
    print(f"✅ Summary saved: {summary_path}")
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("NOTEBOOK COMPLETE: 03b Berlin Optimization")
print("=" * 70)

print(f"\nRuntime: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")

print(f"\nChampions Optimized:")
print(f"  ML: {ml_name}")
print(f"    - CV F1:        {ml_hp_results['best_value']:.4f}")
print(f"    - Test F1:      {ml_metrics['f1_score']:.4f}")
print(f"    - CV-Test Gap:  {ml_cv_test_gap:+.4f}")
print(f"    - Test Acc:     {ml_metrics['accuracy']:.4f}")
print(f"    - Trials:       {ml_hp_results['n_trials']}")

if nn_name and nn_hp_results:
    print(f"  NN: {nn_name}")
    print(f"    - CV F1:        {nn_hp_results['best_value']:.4f}")
    if nn_metrics:
        print(f"    - Test F1:      {nn_metrics['f1_score']:.4f}")
        if nn_cv_test_gap is not None:
            print(f"    - CV-Test Gap:  {nn_cv_test_gap:+.4f}")
        print(f"    - Test Acc:     {nn_metrics['accuracy']:.4f}")
    print(f"    - Trials:       {nn_hp_results['n_trials']}")

print(f"\nOutputs:")
print(f"  Models:")
print(f"    - berlin_ml_champion.pkl")
if nn_name and "nn_model" in locals():
    print(f"    - berlin_nn_champion.pt")
print(f"    - berlin_scaler.pkl")
print(f"    - label_encoder.pkl")
print(f"  Metadata:")
print(f"    - hp_tuning_ml.json")
if nn_hp_results:
    print(f"    - hp_tuning_nn.json")
print(f"    - berlin_evaluation.json")
print(f"    - 03b_summary.json")
print(f"  Logs: {log_path}")

print(f"\nNext Steps:")
print(f"  1. Review hyperparameter tuning results")
print(f"  2. Inspect error analysis figures")
print(f"  3. Run 03c_transfer_evaluation.ipynb (Leipzig zero-shot)")
print("=" * 70)

