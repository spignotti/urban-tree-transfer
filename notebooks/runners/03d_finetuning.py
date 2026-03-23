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
# # Urban Tree Transfer - Runner Notebook
#
# **Title:** Fine-Tuning
#
# **Phase:** 3 - Experiments
#
# **Step:** 03d - Fine-Tuning
#
# **Purpose:** Fine-tune Berlin champion models on Leipzig data and export sample-efficiency results.
#
# **Input:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_3_experiments`
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
        "GITHUB_TOKEN not found in Colab Secrets.\n"
        "1. Click the key icon in the left sidebar\n"
        "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
        "3. Toggle 'Notebook access' ON"
    )

# Install package from GitHub
repo_url = f"git+https://{token}@github.com/silas-workspace/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

# Install PyTorch for neural network fine-tuning
subprocess.run(["pip", "install", "torch>=2.2.0", "-q"], check=True)

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

print("OK: Imports complete")

# %%
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
DATA_DIR = DRIVE_DIR / "data"
INPUT_DIR = DATA_DIR / "phase_3_experiments"
OUTPUT_DIR = DATA_DIR / "phase_3_experiments"

METADATA_DIR = OUTPUT_DIR / "metadata"
MODEL_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = DRIVE_DIR / "outputs" / "report"

for path in [METADATA_DIR, MODEL_DIR, LOGS_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

config = load_experiment_config()  # ✅ FIXED: removed "phase3" argument

# Re-run controls
FULL_RERUN = False
RERUN_FINETUNING = False


def _extract_cnn_model_params(all_params: dict[str, object]) -> dict[str, object]:
    """Filter CNN1D constructor parameters from stored best_params."""
    model_param_keys = {
        "n_temporal_bases",
        "n_months",
        "n_static_features",
        "n_classes",
        "conv_filters",
        "kernel_size",
        "dropout",
        "dense_units",
    }
    return {k: v for k, v in all_params.items() if k in model_param_keys}


def load_finetuning_checkpoint(curve_path: Path) -> dict[str, object] | None:
    """Load intermediate fine-tuning checkpoint if available."""
    if not curve_path.exists() or FULL_RERUN or RERUN_FINETUNING:
        return None

    data = json.loads(curve_path.read_text())
    metadata = data.get("metadata", {})
    checkpoint = metadata.get("checkpoint", {})
    if checkpoint.get("completed", False):
        print(f"✅ Existing completed results found: {curve_path}")
        print(f"   ML Champion: {metadata.get('ml_champion')}")
        print(f"   NN Champion: {metadata.get('nn_champion')}")
        raise SystemExit("Skipping 03d: results already exist")

    print(f"ℹ️  Resuming from partial checkpoint: {curve_path}")
    return data


def save_finetuning_checkpoint(
    curve_path: Path,
    *,
    ml_name: str,
    nn_name: str | None,
    fractions: list[float],
    ml_results: list[dict[str, object]],
    nn_results: list[dict[str, object]],
    ml_baseline_results: list[dict[str, object]],
    stage: str,
    completed: bool,
    extra_metadata: dict[str, object] | None = None,
) -> None:
    """Persist incremental fine-tuning progress to JSON."""
    summarized_results: list[dict[str, object]] = []

    for row in ml_results:
        summarized_results.append(
            {
                "fraction": float(row["fraction"]),
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
        summarized_results.append(
            {
                "fraction": float(row["fraction"]),
                "model_type": "nn",
                "metrics": {
                    "f1_score": row["f1_score"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "accuracy": row["accuracy"],
                },
            }
        )

    if not summarized_results and not completed:
        return

    payload = {
        "results": summarized_results,
        "metadata": {
            "ml_champion": ml_name,
            "nn_champion": nn_name,
            "fractions": fractions,
            "ml_results_raw": ml_results,
            "nn_results_raw": nn_results,
            "ml_baseline_results": ml_baseline_results,
            "checkpoint": {
                "completed": completed,
                "stage": stage,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        },
    }

    if extra_metadata:
        payload["metadata"].update(extra_metadata)

    curve_path.write_text(json.dumps(payload, indent=2))


def get_result_for_fraction(results: list[dict[str, object]], target_fraction: float) -> dict[str, object]:
    """Return result row for fraction with tolerance-aware matching."""
    for row in results:
        if abs(float(row["fraction"]) - target_fraction) < 1e-9:
            return row
    raise ValueError(f"Missing result for fraction {target_fraction}")

# GPU detection (for NN + XGBoost)
try:
    import torch
    nn_device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    nn_device = "cpu"

# Skip if results already exist
curve_path = METADATA_DIR / "finetuning_curve.json"
checkpoint_data = load_finetuning_checkpoint(curve_path)
print(f"Rerun flags: full={FULL_RERUN}, finetuning={RERUN_FINETUNING}")


# %%
# ============================================================================
# 5. LOAD MODELS AND METADATA
# ============================================================================

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
    
    # Load NN champion (optional)
    nn_model_path_pt = MODEL_DIR / "berlin_nn_champion.pt"
    nn_model_path_zip = MODEL_DIR / "berlin_nn_champion.zip"
    nn_model_path = nn_model_path_pt if nn_model_path_pt.exists() else nn_model_path_zip

    if nn_model_path.exists():
        nn_metadata_path = nn_model_path.with_suffix(nn_model_path.suffix + ".metadata.json")
        nn_metadata = json.loads(nn_metadata_path.read_text())
        nn_name = nn_metadata["model_name"]
        nn_feature_cols = nn_metadata["feature_columns"]

        all_params = nn_metadata.get("best_params", {})
        model_params = _extract_cnn_model_params(all_params) if nn_name == "cnn_1d" else all_params

        nn_model = training.load_model(
            nn_model_path,
            model_class=models.CNN1D if nn_name == "cnn_1d" else None,
            model_params=model_params,
        )
    else:
        nn_model = None
        nn_name = None
        nn_feature_cols = None
        nn_metadata = None
        print("ℹ️  No NN champion artifact found - continuing with ML-only fine-tuning")
    
    # Load label encoder
    with (MODEL_DIR / "label_encoder.pkl").open("rb") as f:
        label_encoder = pickle.load(f)
        label_to_idx = label_encoder["label_to_idx"]
        idx_to_label = label_encoder["idx_to_label"]
    
    # Load Berlin scalers (separate for ML/NN)
    with (MODEL_DIR / "berlin_scaler_ml.pkl").open("rb") as f:
        berlin_scaler_ml = pickle.load(f)
    
    if nn_model is not None:
        with (MODEL_DIR / "berlin_scaler_nn.pkl").open("rb") as f:
            berlin_scaler_nn = pickle.load(f)
    else:
        berlin_scaler_nn = None
    
    # Set defaults for backward compatibility
    feature_cols = ml_feature_cols
    berlin_scaler = berlin_scaler_ml
    
    class_labels = list(idx_to_label.values())
    
    print(f"\nML Champion: {ml_name}")
    print(f"  Features: {len(ml_feature_cols)} (reduced)")
    print(f"NN Champion: {nn_name if nn_name else 'None'}")
    if nn_feature_cols is not None:
        print(f"  Features: {len(nn_feature_cols)} (full temporal)")
    print(f"Genera:      {len(class_labels)}")
    print("=" * 70)
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 6. LOAD LEIPZIG DATA
# ============================================================================

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
    
    if nn_model is not None and nn_feature_cols is not None and berlin_scaler_nn is not None:
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
    else:
        leipzig_finetune_nn = None
        leipzig_test_nn = None
        x_test_nn = None
        x_test_scaled_nn = None
        print("\nℹ️  NN artifacts missing - skipping NN dataset loading")
    
    # Get fine-tuning fractions from config
    fractions = config["finetuning"]["fractions"]
    
    print(f"\nℹ️  Dataset summary:")
    print(f"   - ML fine-tuning will use {len(ml_feature_cols)} reduced features")
    if nn_feature_cols is not None:
        print(f"   - NN fine-tuning will use {len(nn_feature_cols)} full temporal features")
    else:
        print("   - NN fine-tuning skipped (no NN champion artifacts)")
    print(f"\nFractions: {fractions}")
    print("=" * 70)
    
    log.end_step(status="success", records=len(leipzig_finetune_ml) + len(leipzig_test_ml))
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# Initialize resumable result containers
checkpoint_metadata = (checkpoint_data or {}).get("metadata", {}) if checkpoint_data else {}
ml_results = list(checkpoint_metadata.get("ml_results_raw", []))
nn_results = list(checkpoint_metadata.get("nn_results_raw", []))
ml_baseline_results = list(checkpoint_metadata.get("ml_baseline_results", []))

if checkpoint_data:
    print("\nResuming cached progress:")
    print(f"  ML fractions cached: {len(ml_results)}")
    print(f"  NN fractions cached: {len(nn_results)}")
    print(f"  Baseline fractions cached: {len(ml_baseline_results)}")


# %%
# ============================================================================
# 7. ML FINE-TUNING (WARM-START)
# ============================================================================

log.start_step("ML Fine-Tuning")

from sklearn.model_selection import train_test_split
import xgboost

try:
    print("\n" + "=" * 70)
    print(f"ML Fine-Tuning: {ml_name} (Warm-Start)")
    print("=" * 70)
    
    completed_ml_fractions = {float(row["fraction"]) for row in ml_results}
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
        if float(frac) in completed_ml_fractions:
            print(f"\n[{frac:.1%}] Already completed in checkpoint - skipping")
            continue

        print(f"\n[{frac:.1%}] Fine-tuning with {int(frac * len(leipzig_finetune_ml))} samples...")
        
        # ✅ FIXED: create_stratified_subsets expects numpy arrays, not DataFrame
        subsets = training.create_stratified_subsets(
            x_train_ml,
            y_train_ml,
            fractions=[frac],
            random_seed=RANDOM_SEED,
        )
        x_finetune, y_finetune = subsets[frac]
        sample_weight = preprocessing.compute_sample_weights(y_finetune)
        
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
                sample_weight=sample_weight,
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
                    sample_weight=sample_weight,
                )
            else:
                raise
        
        # Evaluate on Leipzig test set (using ML test data)
        preds = finetuned_model.predict(x_test_scaled_ml)
        metrics = evaluation.compute_metrics(y_test, preds)
        ci_f1 = evaluation.bootstrap_confidence_interval(
            y_test,
            preds,
            lambda y_true_sample, y_pred_sample: evaluation.compute_metrics(
                y_true_sample,
                y_pred_sample,
            )["f1_score"],
            n_bootstrap=config["metrics"]["n_bootstrap"],
            confidence_level=config["metrics"]["confidence_level"],
        )
        
        ml_results.append({
            "fraction": frac,
            "n_samples": len(y_finetune),
            "f1_score": metrics["f1_score"],
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "ci_f1": ci_f1,
            "predictions": preds.tolist(),
        })
        
        print(f"  F1:       {metrics['f1_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")

        save_finetuning_checkpoint(
            curve_path,
            ml_name=ml_name,
            nn_name=nn_name,
            fractions=fractions,
            ml_results=ml_results,
            nn_results=nn_results,
            ml_baseline_results=ml_baseline_results,
            stage="ml_finetuning",
            completed=False,
        )
        
        # Memory cleanup
        gc.collect()
    
    log.end_step(status="success", records=len(ml_results))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 8. NN FINE-TUNING (WARM-START)
# ============================================================================

log.start_step("NN Fine-Tuning")

from sklearn.model_selection import train_test_split

try:
    if nn_model is None or nn_name is None or leipzig_finetune_nn is None or x_test_scaled_nn is None:
        print("\nℹ️  No NN champion artifacts available - skipping NN fine-tuning")
        log.end_step(status="skipped")
    else:
        print("\n" + "=" * 70)
        print(f"NN Fine-Tuning: {nn_name} (Warm-Start)")
        print("=" * 70)

        completed_nn_fractions = {float(row["fraction"]) for row in nn_results}
        finetune_cfg = config.get("finetuning", {})
        epochs = finetune_cfg.get("nn_epochs", 50)
        batch_size = finetune_cfg.get("nn_batch_size", 64)
        lr_factor = finetune_cfg.get("nn_lr_factor", 0.1)
        if "nn_epochs" not in finetune_cfg or "nn_batch_size" not in finetune_cfg:
            print(
                "⚠️  Missing nn_epochs/nn_batch_size in config; "
                "using defaults (epochs=50, batch_size=64)"
            )

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

        print("\nNN Fine-Tuning Diagnostics")
        print(f"  berlin_scaler_nn.mean_[:3]: {berlin_scaler_nn.mean_[:3]}")
        print(f"  Epochs from config:         {epochs}")
        print(f"  Learning rate factor:       {lr_factor}")

        for frac in fractions:
            if float(frac) in completed_nn_fractions:
                print(f"\n[{frac:.1%}] Already completed in checkpoint - skipping")
                continue

            print(
                f"\n[{frac:.1%}] Fine-tuning with {int(frac * len(leipzig_finetune_nn))} samples..."
            )

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
                lr_factor=lr_factor,
                batch_size=batch_size,
                device=nn_device,
            )
            ft_history = getattr(finetuned_model, "finetune_history_", {})
            if ft_history:
                train_loss = ft_history.get("train_loss", [])
                final_train_loss = train_loss[-1] if train_loss else None
                print(f"  Final train_loss: {final_train_loss}")
                print(f"  Stopped early:    {ft_history.get('stopped_early')}")
                print(f"  Best epoch:       {ft_history.get('best_epoch')}")
            else:
                print("  No fine-tuning history returned")

            # Evaluate on Leipzig test set (using NN test data)
            preds = finetuned_model.predict(x_test_scaled_nn, device=nn_device)
            metrics = evaluation.compute_metrics(y_test, preds)
            ci_f1 = evaluation.bootstrap_confidence_interval(
                y_test,
                preds,
                lambda y_true_sample, y_pred_sample: evaluation.compute_metrics(
                    y_true_sample,
                    y_pred_sample,
                )["f1_score"],
                n_bootstrap=config["metrics"]["n_bootstrap"],
                confidence_level=config["metrics"]["confidence_level"],
            )

            nn_results.append(
                {
                    "fraction": frac,
                    "n_samples": len(y_finetune),
                    "f1_score": metrics["f1_score"],
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "ci_f1": ci_f1,
                    "predictions": preds.tolist(),
                }
            )

            print(f"  F1:       {metrics['f1_score']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")

            save_finetuning_checkpoint(
                curve_path,
                ml_name=ml_name,
                nn_name=nn_name,
                fractions=fractions,
                ml_results=ml_results,
                nn_results=nn_results,
                ml_baseline_results=ml_baseline_results,
                stage="nn_finetuning",
                completed=False,
            )

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
# ============================================================================
# 9. FROM-SCRATCH BASELINES
# ============================================================================

log.start_step("From-Scratch Baselines")

try:
    print("\n" + "=" * 70)
    print(f"From-Scratch Baselines: {ml_name}")
    print("=" * 70)
    print("ℹ️  Using Leipzig-specific scaler for fair comparison")
    print("=" * 70)
    
    completed_baseline_fractions = {float(row["fraction"]) for row in ml_baseline_results}
    
    # ✅ FIXED: Prepare full dataset arrays ONCE before loop (reuse from ML fine-tuning)
    # x_full_ml and y_full_ml already created in SECTION 3
    
    for frac in fractions:
        if float(frac) in completed_baseline_fractions:
            print(f"\n[{frac:.1%}] Baseline already completed in checkpoint - skipping")
            continue

        print(f"\n[{frac:.1%}] Training from scratch with {int(frac * len(leipzig_finetune_ml))} samples...")
        
        # ✅ FIXED: create_stratified_subsets expects numpy arrays, not DataFrame
        subsets = training.create_stratified_subsets(
            x_full_ml,
            y_full_ml,
            fractions=[frac],
            random_seed=RANDOM_SEED,
        )
        x_train, y_train = subsets[frac]
        sample_weight = preprocessing.compute_sample_weights(y_train)
        
        # Leipzig-specific scaler (IMPORTANT for fair comparison!)
        leipzig_scaler = StandardScaler()
        x_train_scaled = leipzig_scaler.fit_transform(x_train)
        x_test_leipzig_scaled = leipzig_scaler.transform(x_test_ml)
        
        # Train from scratch with same hyperparameters as Berlin model
        baseline_params = ml_metadata.get("best_params", {})
        baseline_model = models.create_model(ml_name, model_params=baseline_params)
        baseline_model.fit(x_train_scaled, y_train, sample_weight=sample_weight)
        
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

        save_finetuning_checkpoint(
            curve_path,
            ml_name=ml_name,
            nn_name=nn_name,
            fractions=fractions,
            ml_results=ml_results,
            nn_results=nn_results,
            ml_baseline_results=ml_baseline_results,
            stage="from_scratch_baseline",
            completed=False,
        )
        
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
# ============================================================================
# 10. MCNEMAR SIGNIFICANCE TESTS
# ============================================================================

log.start_step("McNemar Tests")

try:
    print("\n" + "=" * 70)
    print("McNemar Significance Tests")
    print("=" * 70)
    
    # Get zero-shot predictions (re-evaluate with Berlin ML model using ML test data)
    zero_shot_preds = ml_model.predict(x_test_scaled_ml)

    full_fraction = max(float(frac) for frac in fractions)

    # Get full fine-tuning and from-scratch predictions
    ml_full_row = get_result_for_fraction(ml_results, full_fraction)
    ml_baseline_full_row = get_result_for_fraction(ml_baseline_results, full_fraction)
    ml_full_preds = np.array(ml_full_row["predictions"])
    ml_baseline_full_preds = np.array(ml_baseline_full_row["predictions"])
    
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
# ============================================================================
# 11. POWER-LAW CURVE FITTING
# ============================================================================

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
# ============================================================================
# 12. PER-GENUS RECOVERY ANALYSIS
# ============================================================================

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
    full_fraction = max(float(frac) for frac in fractions)
    full_fraction_data = per_genus_df[per_genus_df["fraction"] == full_fraction]
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
# ============================================================================
# 13. HYPOTHESIS H4 - FINE-TUNING EFFICIENCY
# ============================================================================

log.start_step("Hypothesis H4")

try:
    print("\n" + "=" * 70)
    print("Hypothesis H4: Fine-Tuning Efficiency")
    print("=" * 70)

    transfer_eval_path = METADATA_DIR / "transfer_evaluation.json"
    if not transfer_eval_path.exists():
        raise FileNotFoundError(
            f"transfer_evaluation.json not found at {transfer_eval_path}\n"
            "Run 03c_transfer_evaluation.ipynb first."
        )
    transfer_eval = json.loads(transfer_eval_path.read_text())

    berlin_per_genus = {
        row["genus"]: row["f1_score"]
        for row in berlin_eval["per_class"]
    }
    transfer_gap_by_genus = {
        genus: values["absolute_drop"]
        for genus, values in transfer_eval["metadata"]["per_genus_robustness"].items()
    }

    recovery_rows = []
    for genus, berlin_genus_f1 in berlin_per_genus.items():
        target_f1 = 0.9 * berlin_genus_f1
        genus_curve = per_genus_df[per_genus_df["genus"] == genus].sort_values("fraction")
        recovery_fraction = None
        for _, row in genus_curve.iterrows():
            if row["f1_score"] >= target_f1:
                recovery_fraction = float(row["fraction"])
                break

        recovery_rows.append(
            {
                "genus": genus,
                "berlin_f1": berlin_genus_f1,
                "transfer_gap": transfer_gap_by_genus.get(genus),
                "recovery_fraction": recovery_fraction,
            }
        )

    recovery_df = pd.DataFrame(recovery_rows)
    valid_recovery_df = recovery_df.dropna(subset=["transfer_gap", "recovery_fraction"])

    from scipy.stats import spearmanr

    if len(valid_recovery_df) < 3:
        h4_result = {
            "id": "H4",
            "description": "Genera with larger zero-shot transfer gaps require more fine-tuning data",
            "test_type": "spearman",
            "statistic": None,
            "p_value": None,
            "effect_size": None,
            "conclusion": "Insufficient data for Spearman test",
        }
    else:
        statistic, p_value = spearmanr(
            valid_recovery_df["transfer_gap"],
            valid_recovery_df["recovery_fraction"],
        )
        statistic = float(statistic)
        p_value = float(p_value)
        h4_result = {
            "id": "H4",
            "description": "Genera with larger zero-shot transfer gaps require more fine-tuning data",
            "test_type": "spearman",
            "statistic": statistic,
            "p_value": p_value,
            "effect_size": statistic,
            "conclusion": (
                f"Significant rank correlation (rho={statistic:.3f}, p={p_value:.4f})"
                if p_value < 0.05
                else f"No significant rank correlation (rho={statistic:.3f}, p={p_value:.4f})"
            ),
        }

    print(f"Valid genera with recovery threshold reached: {len(valid_recovery_df)}")
    print(f"  Statistic:   {h4_result['statistic']}")
    print(f"  p-value:     {h4_result['p_value']}")
    print(f"  Effect size: {h4_result['effect_size']}")
    print(f"  Result:      {h4_result['conclusion']}")

    hypothesis_path = METADATA_DIR / "hypothesis_tests.json"
    if not hypothesis_path.exists():
        raise FileNotFoundError(
            f"hypothesis_tests.json not found at {hypothesis_path}\n"
            "Run 03c_transfer_evaluation.ipynb first."
        )
    hypothesis_results = json.loads(hypothesis_path.read_text())
    hypothesis_results = [result for result in hypothesis_results if result.get("id") != "H4"]
    hypothesis_results.append(h4_result)
    hypothesis_path.write_text(json.dumps(hypothesis_results, indent=2))
    print(f"\n✅ Updated: {hypothesis_path.name}")

    log.end_step(status="success", records=len(valid_recovery_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 14. SAVE RESULTS
# ============================================================================

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
            "checkpoint": {
                "completed": True,
                "stage": "finalized",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
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
def build_curve_report(
    results: list[dict[str, object]],
    model_type: str,
) -> list[dict[str, float | int | str]]:
    curve_records = []
    for row in results:
        _, ci_low, ci_high = evaluation.bootstrap_confidence_interval(
            y_test,
            np.array(row["predictions"]),
            lambda y_true_sample, y_pred_sample: evaluation.compute_metrics(
                y_true_sample,
                y_pred_sample,
            )["f1_score"],
            n_bootstrap=config["metrics"]["n_bootstrap"],
            confidence_level=config["metrics"]["confidence_level"],
        )
        curve_records.append(
            {
                "fraction": row["fraction"],
                "model_type": model_type,
                "f1_score": row["f1_score"],
                "n_samples": row["n_samples"],
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )
    return curve_records

try:
    curve_report = build_curve_report(ml_results, "ml") + build_curve_report(nn_results, "nn")
    curve_report_path = REPORT_DIR / "finetuning_curves.json"
    curve_report_path.write_text(json.dumps(curve_report, indent=2), encoding="utf-8")
    print(f"Saved report JSON: {curve_report_path}")

    per_genus_report_path = REPORT_DIR / "finetuning_per_genus.json"
    per_genus_report = per_genus_df[["fraction", "genus", "f1_score", "support"]].to_dict(
        orient="records"
    )
    per_genus_report_path.write_text(json.dumps(per_genus_report, indent=2), encoding="utf-8")
    print(f"Saved report JSON: {per_genus_report_path}")

    report_hypothesis_path = REPORT_DIR / "hypothesis_tests.json"
    report_hypothesis_path.write_text(
        hypothesis_path.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    print(f"Saved report JSON: {report_hypothesis_path}")
except Exception as e:
    print(f"WARNING: Failed to export report JSONs: {e}")


# %%
# ============================================================================
# 15. SAVE EXECUTION LOG
# ============================================================================

log.start_step("Save Execution Log")

try:
    full_fraction = max(float(frac) for frac in fractions)
    ml_final_row = get_result_for_fraction(ml_results, full_fraction)
    baseline_final_row = get_result_for_fraction(ml_baseline_results, full_fraction)
    nn_final_row = get_result_for_fraction(nn_results, full_fraction) if nn_results else None

    # Add summary metadata
    execution_summary = {
        "notebook": "03d_finetuning",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "ml_champion": ml_name,
        "nn_champion": nn_name,
        "fractions": fractions,
        "ml_final_f1": ml_final_row["f1_score"],
        "nn_final_f1": nn_final_row["f1_score"] if nn_final_row else None,
        "baseline_final_f1": baseline_final_row["f1_score"],
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
# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("OK: NOTEBOOK COMPLETE")
print("=" * 70)

print(f"\nModels:")
print(f"  ML Champion: {ml_name}")
print(f"  NN Champion: {nn_name}")

print(f"\nFine-tuning Fractions: {fractions}")

full_fraction = max(float(frac) for frac in fractions)
ml_final_row = get_result_for_fraction(ml_results, full_fraction)
baseline_final_row = get_result_for_fraction(ml_baseline_results, full_fraction)
nn_final_row = get_result_for_fraction(nn_results, full_fraction) if nn_results else None

print(f"\nFinal Performance (Full Fine-tuning):")
print(f"  ML:        {ml_final_row['f1_score']:.4f} F1")
if nn_final_row is not None:
    print(f"  NN:        {nn_final_row['f1_score']:.4f} F1")
else:
    print("  NN:        skipped")
print(f"  Baseline:  {baseline_final_row['f1_score']:.4f} F1 (from scratch)")

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
