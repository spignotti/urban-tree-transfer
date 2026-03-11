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
# # 03d Fine-tuning
#
# **Phase 3.4 - Warm-Start Fine-Tuning on Leipzig Data**
#
# This notebook performs warm-start fine-tuning of Berlin champions on Leipzig data with incremental sample fractions, analyzing:
# - ML champion fine-tuning (warm-start XGBoost)
# - NN champion fine-tuning (warm-start CNN1D)
# - From-scratch baselines for comparison
# - McNemar significance tests
# - Power-law curve fitting for 95% recovery extrapolation
# - Per-genus recovery analysis
#
# **Dependencies:** Requires Berlin champions from 03b, Leipzig data from 03a, transfer evaluation from 03c.
#
# **Output:** Creates `finetuning_curve.json` with learning curves, significance tests, and per-genus recovery analysis.
#
# ---
#
# **RUNTIME REQUIREMENTS:**
# - CPU: Standard (12GB RAM sufficient with float32)
# - GPU: Recommended for NN fine-tuning (not required)
# - High-RAM: Optional (for comfort)
#
# **IMPORTANT:** Test set is scaled with Berlin scaler for transfer learning evaluation (warm-start). From-scratch baseline uses Leipzig-specific scaler for fair comparison.

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

# Install PyTorch for neural network fine-tuning
subprocess.run(["pip", "install", "torch>=2.2.0", "-q"], check=True)

print("✅ Package installed successfully")

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
import gc

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from urban_tree_transfer.config import load_experiment_config, RANDOM_SEED
from urban_tree_transfer.experiments import (
    data_loading,
    evaluation,
    models,
    training,
    transfer,
)
from urban_tree_transfer.utils import (
    ExecutionLog,
    validate_finetuning_curve,
)
warnings.filterwarnings("ignore", category=UserWarning)

log = ExecutionLog("03d_finetuning")

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
MODEL_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"

for path in [METADATA_DIR, MODEL_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

config = load_experiment_config()  # ✅ FIXED: removed "phase3" argument

# GPU detection (for NN + XGBoost)
try:
    import torch
    nn_device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    nn_device = "cpu"

# Skip if results already exist
existing_curve_path = METADATA_DIR / "finetuning_curve.json"
if existing_curve_path.exists():
    print(f"✅ Existing results found: {existing_curve_path}")
    existing_data = json.loads(existing_curve_path.read_text())
    meta = existing_data.get('metadata', {})
    print(f"   ML Champion: {meta.get('ml_champion')}")
    print(f"   NN Champion: {meta.get('nn_champion')}")
    raise SystemExit("Skipping 03d: results already exist")


# %%
# ============================================================
# SECTION 1: Load Models & Metadata
# ============================================================

log.start_step("Load Models & Metadata")

try:
    print("\n" + "=" * 70)
    print("Loading Berlin Champions")
    print("=" * 70)
    
    # Load ML champion
    ml_model_path = MODEL_DIR / "berlin_ml_champion.pkl"
    ml_metadata_path = MODEL_DIR / "berlin_ml_champion.pkl.metadata.json"
    
    if not ml_model_path.exists():
        raise FileNotFoundError(
            f"ML champion not found at {ml_model_path}\n"
            "Run 03b_berlin_optimization.ipynb first."
        )
    
    ml_model = training.load_model(ml_model_path)
    ml_metadata = json.loads(ml_metadata_path.read_text())
    ml_name = ml_metadata["model_name"]
    ml_feature_cols = ml_metadata["feature_columns"]
    
    # Load NN champion
    nn_model_path = MODEL_DIR / "berlin_nn_champion.pt"
    nn_metadata_path = MODEL_DIR / "berlin_nn_champion.pt.metadata.json"
    
    if not nn_model_path.exists():
        raise FileNotFoundError(
            f"NN champion not found at {nn_model_path}\n"
            "Run 03b_berlin_optimization.ipynb first."
        )
    
    nn_metadata = json.loads(nn_metadata_path.read_text())
    nn_name = nn_metadata["model_name"]
    nn_feature_cols = nn_metadata["feature_columns"]
    
    # ✅ FIXED: Filter model_params for CNN1D (remove training params)
    # CNN1D __init__ only accepts: n_temporal_bases, n_months, n_static_features, n_classes,
    #                              conv_filters, kernel_size, dropout, dense_units
    all_params = nn_metadata.get("best_params", {})
    model_param_keys = {
        "n_temporal_bases", "n_months", "n_static_features", "n_classes",
        "conv_filters", "kernel_size", "dropout", "dense_units"
    }
    model_params = {k: v for k, v in all_params.items() if k in model_param_keys}
    
    # ✅ FIXED: Load NN model with model_class and filtered model_params
    nn_model = training.load_model(
        nn_model_path,
        model_class=models.CNN1D if nn_name == "cnn_1d" else None,
        model_params=model_params,
    )
    
    # Load label encoder
    with (MODEL_DIR / "label_encoder.pkl").open("rb") as f:
        label_encoder = pickle.load(f)
        label_to_idx = label_encoder["label_to_idx"]
        idx_to_label = label_encoder["idx_to_label"]
    
    # Load Berlin scalers (separate for ML/NN)
    with (MODEL_DIR / "berlin_scaler_ml.pkl").open("rb") as f:
        berlin_scaler_ml = pickle.load(f)
    
    with (MODEL_DIR / "berlin_scaler_nn.pkl").open("rb") as f:
        berlin_scaler_nn = pickle.load(f)
    
    # Set defaults for backward compatibility
    feature_cols = ml_feature_cols
    berlin_scaler = berlin_scaler_ml
    
    class_labels = list(idx_to_label.values())
    
    print(f"\nML Champion: {ml_name}")
    print(f"  Features: {len(ml_feature_cols)} (reduced)")
    print(f"NN Champion: {nn_name}")
    print(f"  Features: {len(nn_feature_cols)} (full temporal)")
    print(f"Genera:      {len(class_labels)}")
    print("=" * 70)
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 2: Load Leipzig Data
# ============================================================

log.start_step("Load Leipzig Data")

try:
    print("\n" + "=" * 70)
    print("Loading Leipzig Data")
    print("=" * 70)
    
    # Load Leipzig ML splits (reduced features)
    print(f"\n[ML Champion Datasets - Reduced Features]")
    # ✅ FIXED: No variant parameter needed - always loads baseline files
    leipzig_finetune_ml, leipzig_test_ml = data_loading.load_leipzig_splits(INPUT_DIR)
    
    print(f"  Finetune: {len(leipzig_finetune_ml):,} samples, {len(ml_feature_cols)} features")
    print(f"  Test:     {len(leipzig_test_ml):,} samples")
    
    # Memory optimization
    float_cols = leipzig_test_ml.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        leipzig_test_ml[float_cols] = leipzig_test_ml[float_cols].astype("float32")
        leipzig_finetune_ml[float_cols] = leipzig_finetune_ml[float_cols].astype("float32")
        print(f"  Optimized: {len(float_cols)} float64→float32")
    
    # Prepare ML test set (scaled with Berlin ML scaler)
    x_test_ml = leipzig_test_ml[ml_feature_cols].to_numpy(dtype=np.float32)
    y_test = leipzig_test_ml["genus_latin"].map(label_to_idx).to_numpy()
    x_test_scaled_ml = berlin_scaler_ml.transform(x_test_ml)
    
    print(f"  ✅ ML datasets prepared (scaled with berlin_scaler_ml)")
    
    # Set defaults for backward compatibility
    leipzig_finetune, leipzig_test = leipzig_finetune_ml, leipzig_test_ml
    x_test = x_test_ml
    x_test_scaled = x_test_scaled_ml
    
    # Load Leipzig NN splits (full temporal features)
    print(f"\n[NN Champion Datasets - Full Temporal Features]")
    # ✅ FIXED: No variant parameter needed - always loads _cnn files
    leipzig_finetune_nn, leipzig_test_nn = data_loading.load_leipzig_splits_cnn(INPUT_DIR)
    
    print(f"  Finetune: {len(leipzig_finetune_nn):,} samples, {len(nn_feature_cols)} features")
    print(f"  Test:     {len(leipzig_test_nn):,} samples")
    
    # Memory optimization for NN datasets
    float_cols_nn = leipzig_test_nn.select_dtypes(include=["float64"]).columns
    if len(float_cols_nn) > 0:
        leipzig_test_nn[float_cols_nn] = leipzig_test_nn[float_cols_nn].astype("float32")
        leipzig_finetune_nn[float_cols_nn] = leipzig_finetune_nn[float_cols_nn].astype("float32")
        print(f"  Optimized: {len(float_cols_nn)} float64→float32")
    
    # Prepare NN test set (scaled with Berlin NN scaler)
    x_test_nn = leipzig_test_nn[nn_feature_cols].to_numpy(dtype=np.float32)
    x_test_scaled_nn = berlin_scaler_nn.transform(x_test_nn)
    
    print(f"  ✅ NN datasets prepared (scaled with berlin_scaler_nn)")
    
    # Get fine-tuning fractions from config
    fractions = config["finetuning"]["fractions"]
    
    print(f"\nℹ️  Dual datasets loaded:")
    print(f"   - ML fine-tuning will use {len(ml_feature_cols)} reduced features")
    print(f"   - NN fine-tuning will use {len(nn_feature_cols)} full temporal features")
    print(f"\nFractions: {fractions}")
    print("=" * 70)
    
    log.end_step(status="success", records=len(leipzig_finetune_ml) + len(leipzig_test_ml))
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 3: ML Fine-Tuning (Warm-Start)
# ============================================================

log.start_step("ML Fine-Tuning")

from sklearn.model_selection import train_test_split
import xgboost

try:
    print("\n" + "=" * 70)
    print(f"ML Fine-Tuning: {ml_name} (Warm-Start)")
    print("=" * 70)
    
    ml_results = []
    n_estimators = config["finetuning"]["ml_warm_start_estimators"]

    use_xgb_gpu = False
    if ml_name == "xgboost" and nn_device == "cuda":
        try:
            xgb_major = int(xgboost.__version__.split(".")[0])
            if xgb_major >= 2:
                ml_model.set_params(tree_method="hist", device="cuda", predictor="auto")
            else:
                ml_model.set_params(tree_method="gpu_hist", predictor="gpu_predictor")
            use_xgb_gpu = True
            print("✅ XGBoost GPU enabled for fine-tuning")
        except Exception as exc:
            print(f"⚠️  XGBoost GPU unavailable: {exc}")
            use_xgb_gpu = False
    
    # ✅ FIXED: Prepare full dataset arrays ONCE before loop
    x_full_ml = leipzig_finetune_ml[ml_feature_cols].to_numpy(dtype=np.float32)
    y_full_ml = leipzig_finetune_ml["genus_latin"].map(label_to_idx).to_numpy()

    ml_val_ratio = config.get("finetuning", {}).get("ml_val_ratio", 0.10)
    x_train_ml, x_val_ml, y_train_ml, y_val_ml = train_test_split(
        x_full_ml,
        y_full_ml,
        test_size=ml_val_ratio,
        stratify=y_full_ml,
        random_state=RANDOM_SEED,
    )
    x_val_scaled_ml = berlin_scaler_ml.transform(x_val_ml)
    
    for frac in fractions:
        print(f"\n[{frac:.1%}] Fine-tuning with {int(frac * len(leipzig_finetune_ml))} samples...")
        
        # ✅ FIXED: create_stratified_subsets expects numpy arrays, not DataFrame
        subsets = training.create_stratified_subsets(
            x_train_ml,
            y_train_ml,
            fractions=[frac],
            random_seed=RANDOM_SEED,
        )
        x_finetune, y_finetune = subsets[frac]
        
        # Scale with Berlin ML scaler for warm-start
        x_finetune_scaled = berlin_scaler_ml.transform(x_finetune)
        
        # Warm-start fine-tuning
        try:
            finetuned_model = training.finetune_xgboost(
                ml_model,
                x_finetune_scaled,
                y_finetune,
                n_additional_estimators=n_estimators,
                x_val=x_val_scaled_ml,
                y_val=y_val_ml,
            )
        except Exception as exc:
            if use_xgb_gpu and "GPU" in str(exc):
                print("⚠️  GPU error during XGBoost fine-tuning, retrying on CPU")
                ml_model.set_params(tree_method="hist", predictor="auto")
                use_xgb_gpu = False
                finetuned_model = training.finetune_xgboost(
                    ml_model,
                    x_finetune_scaled,
                    y_finetune,
                    n_additional_estimators=n_estimators,
                    x_val=x_val_scaled_ml,
                    y_val=y_val_ml,
                )
            else:
                raise
        
        # Evaluate on Leipzig test set (using ML test data)
        preds = finetuned_model.predict(x_test_scaled_ml)
        metrics = evaluation.compute_metrics(y_test, preds)
        
        ml_results.append({
            "fraction": frac,
            "n_samples": len(y_finetune),
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "predictions": preds.tolist(),
        })
        
        print(f"  F1:       {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Memory cleanup
        gc.collect()
    
    log.end_step(status="success", records=len(ml_results))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 4: NN Fine-Tuning (Warm-Start)
# ============================================================

log.start_step("NN Fine-Tuning")

from sklearn.model_selection import train_test_split

try:
    print("\n" + "=" * 70)
    print(f"NN Fine-Tuning: {nn_name} (Warm-Start)")
    print("=" * 70)
    
    nn_results = []
    finetune_cfg = config.get("finetuning", {})
    epochs = finetune_cfg.get("nn_epochs", 50)
    batch_size = finetune_cfg.get("nn_batch_size", 64)
    if "nn_epochs" not in finetune_cfg or "nn_batch_size" not in finetune_cfg:
        print("⚠️  Missing nn_epochs/nn_batch_size in config; using defaults (epochs=50, batch_size=64)")
    
    # ✅ FIXED: Prepare full dataset arrays ONCE before loop
    x_full_nn = leipzig_finetune_nn[nn_feature_cols].to_numpy(dtype=np.float32)
    y_full_nn = leipzig_finetune_nn["genus_latin"].map(label_to_idx).to_numpy()

    nn_val_ratio = finetune_cfg.get("nn_val_ratio", 0.10)
    x_train_nn, x_val_nn, y_train_nn, y_val_nn = train_test_split(
        x_full_nn,
        y_full_nn,
        test_size=nn_val_ratio,
        stratify=y_full_nn,
        random_state=RANDOM_SEED,
    )
    x_val_scaled_nn = berlin_scaler_nn.transform(x_val_nn)
    
    for frac in fractions:
        print(f"\n[{frac:.1%}] Fine-tuning with {int(frac * len(leipzig_finetune_nn))} samples...")
        
        # ✅ FIXED: create_stratified_subsets expects numpy arrays, not DataFrame
        subsets = training.create_stratified_subsets(
            x_train_nn,
            y_train_nn,
            fractions=[frac],
            random_seed=RANDOM_SEED,
        )
        x_finetune, y_finetune = subsets[frac]
        
        # Scale with Berlin NN scaler for warm-start
        x_finetune_scaled = berlin_scaler_nn.transform(x_finetune)
        
        # Warm-start fine-tuning (using NN validation split)
        finetuned_model = training.finetune_neural_network(
            nn_model,
            x_finetune_scaled,
            y_finetune,
            x_val=x_val_scaled_nn,
            y_val=y_val_nn,
            epochs=epochs,
            batch_size=batch_size,
            device=nn_device,
        )
        
        # Evaluate on Leipzig test set (using NN test data)
        preds = finetuned_model.predict(x_test_scaled_nn, device=nn_device)
        metrics = evaluation.compute_metrics(y_test, preds)
        
        nn_results.append({
            "fraction": frac,
            "n_samples": len(y_finetune),
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "predictions": preds.tolist(),
        })
        
        print(f"  F1:       {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Memory cleanup
        del finetuned_model, x_finetune, x_finetune_scaled
        gc.collect()
    
    print("\n" + "=" * 70)
    print(f"NN Fine-tuning complete: {len(nn_results)} fractions")
    print("=" * 70)
    
    log.end_step(status="success", records=len(nn_results))
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 5: From-Scratch Baselines
# ============================================================

log.start_step("From-Scratch Baselines")

try:
    print("\n" + "=" * 70)
    print(f"From-Scratch Baselines: {ml_name}")
    print("=" * 70)
    print("ℹ️  Using Leipzig-specific scaler for fair comparison")
    print("=" * 70)
    
    ml_baseline_results = []
    
    # ✅ FIXED: Prepare full dataset arrays ONCE before loop (reuse from ML fine-tuning)
    # x_full_ml and y_full_ml already created in SECTION 3
    
    for frac in fractions:
        print(f"\n[{frac:.1%}] Training from scratch with {int(frac * len(leipzig_finetune_ml))} samples...")
        
        # ✅ FIXED: create_stratified_subsets expects numpy arrays, not DataFrame
        subsets = training.create_stratified_subsets(
            x_full_ml,
            y_full_ml,
            fractions=[frac],
            random_seed=RANDOM_SEED,
        )
        x_train, y_train = subsets[frac]
        
        # Leipzig-specific scaler (IMPORTANT for fair comparison!)
        leipzig_scaler = StandardScaler()
        x_train_scaled = leipzig_scaler.fit_transform(x_train)
        x_test_leipzig_scaled = leipzig_scaler.transform(x_test_ml)
        
        # Train from scratch with same hyperparameters as Berlin model
        baseline_params = ml_metadata.get("best_params", {})
        baseline_model = models.create_model(ml_name, model_params=baseline_params)
        baseline_model.fit(x_train_scaled, y_train)
        
        # Evaluate on Leipzig test set
        preds = baseline_model.predict(x_test_leipzig_scaled)
        metrics = evaluation.compute_metrics(y_test, preds)
        
        ml_baseline_results.append({
            "fraction": frac,
            "n_samples": len(y_train),
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
            "predictions": preds.tolist(),
        })
        
        print(f"  F1:       {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        
        # Memory cleanup
        del baseline_model, x_train, x_train_scaled
        gc.collect()
    
    print("\n" + "=" * 70)
    print(f"From-scratch baselines complete: {len(ml_baseline_results)} fractions")
    print("=" * 70)
    
    log.end_step(status="success", records=len(ml_baseline_results))
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 6: McNemar Significance Tests
# ============================================================

log.start_step("McNemar Tests")

try:
    print("\n" + "=" * 70)
    print("McNemar Significance Tests")
    print("=" * 70)
    
    # Get zero-shot predictions (re-evaluate with Berlin ML model using ML test data)
    zero_shot_preds = ml_model.predict(x_test_scaled_ml)
    
    # Get full fine-tuning and from-scratch predictions
    ml_full_preds = np.array(ml_results[-1]["predictions"])
    ml_baseline_full_preds = np.array(ml_baseline_results[-1]["predictions"])
    
    # McNemar test: Fine-tuning vs Zero-shot
    mcnemar_ft_vs_zs = transfer.mcnemar_test(
        y_test,
        zero_shot_preds,
        ml_full_preds,
    )
    
    # McNemar test: Fine-tuning vs From-scratch
    mcnemar_ft_vs_fs = transfer.mcnemar_test(
        y_test,
        ml_baseline_full_preds,
        ml_full_preds,
    )
    
    print("\n[Fine-tuning vs Zero-shot]")
    print(f"  Statistic:  {mcnemar_ft_vs_zs['statistic']:.2f}")
    print(f"  p-value:    {mcnemar_ft_vs_zs['p_value']:.4f}")
    print(f"  Significant: {mcnemar_ft_vs_zs['significant']} (α=0.05)")
    
    print("\n[Fine-tuning vs From-scratch]")
    print(f"  Statistic:  {mcnemar_ft_vs_fs['statistic']:.2f}")
    print(f"  p-value:    {mcnemar_ft_vs_fs['p_value']:.4f}")
    print(f"  Significant: {mcnemar_ft_vs_fs['significant']} (α=0.05)")
    
    print("=" * 70)
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================
# SECTION 7: Power-Law Curve Fitting
# ============================================================

log.start_step("Power-Law Curve Fitting")

try:
    print("\n" + "=" * 70)
    print("Power-Law Curve Fitting")
    print("=" * 70)
    
    # Prepare data for curve fitting
    ml_df = pd.DataFrame(ml_results)
    x_data = ml_df["n_samples"].values
    y_data = ml_df["f1_score"].values
    
    # Get target F1 (95% of Berlin performance)
    berlin_eval_path = METADATA_DIR / "berlin_evaluation.json"
    if not berlin_eval_path.exists():
        raise FileNotFoundError(
            f"Berlin evaluation not found at {berlin_eval_path}\n"
            "Run 03b_berlin_optimization.ipynb first."
        )
    
    berlin_eval = json.loads(berlin_eval_path.read_text())
    berlin_f1 = berlin_eval["metrics"]["f1_score"]
    target_f1 = 0.95 * berlin_f1
    
    # Fit power law: y = a * x^b
    power_law_fit = evaluation.fit_power_law(
        x_data,
        y_data,
        target_y=target_f1,
    )
    
    print(f"\nPower-law fit: y = {power_law_fit['a']:.4f} * x^{power_law_fit['b']:.4f}")
    print(f"R²: {power_law_fit['r_squared']:.4f}")
    
    print(f"\nBerlin F1:   {berlin_f1:.4f}")
    print(f"Target (95%): {target_f1:.4f}")
    
    if power_law_fit["extrapolated_x"] is not None:
        estimated_samples = power_law_fit["extrapolated_x"]
        total_available = len(leipzig_finetune)
        print(f"\nEstimated samples for 95% recovery: {estimated_samples:.0f}")
        print(f"Available Leipzig samples:          {total_available:,}")
        
        if estimated_samples > total_available:
            print(f"⚠️  WARNING: Requires more samples than available")
        else:
            print(f"✅ 95% recovery achievable with available data")
    else:
        print(f"\nℹ️  95% recovery not achievable (already exceeded target or poor fit)")
    
    print("=" * 70)
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================
# SECTION 8: Per-Genus Recovery Analysis
# ============================================================

log.start_step("Per-Genus Recovery")

try:
    print("\n" + "=" * 70)
    print("Per-Genus Recovery Analysis")
    print("=" * 70)
    
    # Compute per-genus metrics for each fraction
    per_genus_recovery = []
    
    for result in ml_results:
        preds = np.array(result["predictions"])
        per_class_df = evaluation.compute_per_class_metrics(y_test, preds, class_labels)
        
        # ✅ FIXED: compute_per_class_metrics returns DataFrame, not list
        for _, genus_row in per_class_df.iterrows():
            per_genus_recovery.append({
                "fraction": result["fraction"],
                "genus": genus_row["genus"],
                "f1_score": genus_row["f1_score"],
                "support": genus_row["support"],
            })
    
    per_genus_df = pd.DataFrame(per_genus_recovery)
    
    # Show top 5 best and worst recovering genera (at full fine-tuning)
    full_fraction_data = per_genus_df[per_genus_df["fraction"] == fractions[-1]]
    top_5 = full_fraction_data.nlargest(5, "f1_score")
    bottom_5 = full_fraction_data.nsmallest(5, "f1_score")
    
    print("\nTop 5 Best Recovering Genera (Full Fine-tuning):")
    for _, row in top_5.iterrows():
        print(f"  {row['genus']:20s} F1: {row['f1_score']:.4f} (n={int(row['support'])})")
    
    print("\nTop 5 Worst Recovering Genera (Full Fine-tuning):")
    for _, row in bottom_5.iterrows():
        print(f"  {row['genus']:20s} F1: {row['f1_score']:.4f} (n={int(row['support'])})")
    
    print("=" * 70)
    
    log.end_step(status="success", records=len(per_genus_recovery))
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================
# SECTION 10: Save Results
# ============================================================

log.start_step("Save Results")

try:
    print("\n" + "=" * 70)
    print("Saving Fine-tuning Results")
    print("=" * 70)
    
    # Prepare schema-compliant fine-tuning curve
    results = []
    for row in ml_results:
        results.append(
            {
                "fraction": row["fraction"],
                "model_type": "ml",
                "metrics": {
                    "f1_score": row["f1_score"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "accuracy": row["accuracy"],
                },
            }
        )
    for row in nn_results:
        results.append(
            {
                "fraction": row["fraction"],
                "model_type": "nn",
                "metrics": {
                    "f1_score": row["f1_score"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "accuracy": row["accuracy"],
                },
            }
        )

    finetuning_data = {
        "results": results,
        "powerlaw_fit": power_law_fit,
        "metadata": {
            "ml_champion": ml_name,
            "nn_champion": nn_name,
            "fractions": fractions,
            "ml_results_raw": ml_results,
            "nn_results_raw": nn_results,
            "ml_baseline_results": ml_baseline_results,
            "mcnemar_tests": {
                "finetuning_vs_zeroshot": mcnemar_ft_vs_zs,
                "finetuning_vs_fromscratch": mcnemar_ft_vs_fs,
            },
            "per_genus_recovery": per_genus_recovery,
        },
    }
    
    # Save to JSON
    curve_path = METADATA_DIR / "finetuning_curve.json"
    curve_path.write_text(json.dumps(finetuning_data, indent=2))
    
    # Validate schema
    validate_finetuning_curve(curve_path)
    
    print(f"\n✅ Saved: {curve_path}")
    print(f"✅ Schema validation passed")
    print("=" * 70)
    
    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================
# SECTION 11: Save Execution Log
# ============================================================

log.start_step("Save Execution Log")

try:
    # Add summary metadata
    execution_summary = {
        "notebook": "03d_finetuning",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "ml_champion": ml_name,
        "nn_champion": nn_name,
        "fractions": fractions,
        "ml_final_f1": ml_results[-1]["f1_score"],
        "nn_final_f1": nn_results[-1]["f1_score"],
        "baseline_final_f1": ml_baseline_results[-1]["f1_score"],
        "power_law": {
            "a": power_law_fit["a"],
            "b": power_law_fit["b"],
            "r_squared": power_law_fit["r_squared"],
            "extrapolated_x": power_law_fit["extrapolated_x"],
        },
        "mcnemar_significant": {
            "vs_zeroshot": mcnemar_ft_vs_zs["significant"],
            "vs_fromscratch": mcnemar_ft_vs_fs["significant"],
        },
    }
    
    # Save execution log
    log.summary()
    log_path = LOGS_DIR / f"{log.notebook}_execution.json"
    log.save(log_path)
    
    # Save summary
    summary_path = METADATA_DIR / "03d_summary.json"
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
print("NOTEBOOK COMPLETE: 03d Fine-tuning")
print("=" * 70)

print(f"\nModels:")
print(f"  ML Champion: {ml_name}")
print(f"  NN Champion: {nn_name}")

print(f"\nFine-tuning Fractions: {fractions}")

print(f"\nFinal Performance (Full Fine-tuning):")
print(f"  ML:        {ml_results[-1]['f1_score']:.4f} F1")
print(f"  NN:        {nn_results[-1]['f1_score']:.4f} F1")
print(f"  Baseline:  {ml_baseline_results[-1]['f1_score']:.4f} F1 (from scratch)")

print(f"\nPower-Law Fit:")
print(f"  Formula: y = {power_law_fit['a']:.4f} * x^{power_law_fit['b']:.4f}")
print(f"  R²: {power_law_fit['r_squared']:.4f}")
if power_law_fit["extrapolated_x"] is not None:
    print(f"  95% recovery: ~{power_law_fit['extrapolated_x']:.0f} samples")
else:
    print(f"  95% recovery: Not achievable")

print(f"\nMcNemar Tests:")
print(f"  FT vs Zero-shot:    {'Significant' if mcnemar_ft_vs_zs['significant'] else 'Not significant'} (p={mcnemar_ft_vs_zs['p_value']:.4f})")
print(f"  FT vs From-scratch: {'Significant' if mcnemar_ft_vs_fs['significant'] else 'Not significant'} (p={mcnemar_ft_vs_fs['p_value']:.4f})")

print(f"\nOutputs:")
print(f"  Results:  {curve_path}")
print(f"  Logs:     {log_path}")
print(f"  Summary:  {summary_path}")

print(f"\nNext Steps:")
print(f"  1. Review fine-tuning results and visualizations")
print(f"  2. Analyze per-genus recovery patterns")
print(f"  3. Review power-law extrapolation for sample requirements")
print(f"  4. Phase 3 experiments complete! Review all results.")

print("=" * 70)
