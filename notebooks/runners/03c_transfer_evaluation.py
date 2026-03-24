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
# **Title:** Transfer Evaluation
#
# **Phase:** 3 - Experiments
#
# **Step:** 03c - Transfer Evaluation
#
# **Purpose:** Evaluate zero-shot Berlin-to-Leipzig transfer performance and export transfer metrics.
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
import inspect
import json
import pickle
import warnings
import time
import gc

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from urban_tree_transfer.config import load_experiment_config, RANDOM_SEED
from urban_tree_transfer.experiments import (
    data_loading,
    evaluation,
    models,
    preprocessing,
    training,
    transfer,
)
from urban_tree_transfer.utils import (
    ExecutionLog,
    validate_evaluation_metrics,
)
warnings.filterwarnings("ignore", category=UserWarning)

log = ExecutionLog("03c_transfer_evaluation")
start_time = time.time()

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
MODELS_DIR = OUTPUT_DIR / "models"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = DRIVE_DIR / "outputs" / "report"

for path in [METADATA_DIR, MODELS_DIR, LOGS_DIR, REPORT_DIR]:
    path.mkdir(parents=True, exist_ok=True)

config = load_experiment_config()  # ✅ FIXED: removed "phase3" argument

# Re-run controls
FULL_RERUN = False
RERUN_TRANSFER_EVAL = False


def _supports_kwarg(func: object, kwarg_name: str) -> bool:
    """Return True when callable signature contains kwarg."""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return kwarg_name in signature.parameters


def load_model_compat(
    model_path: Path,
    model_class: type | None = None,
    model_params: dict[str, object] | None = None,
):
    """Call training.load_model with backward-compatible kwargs."""
    kwargs: dict[str, object] = {}
    if model_class is not None and _supports_kwarg(training.load_model, "model_class"):
        kwargs["model_class"] = model_class
    if model_params is not None and _supports_kwarg(training.load_model, "model_params"):
        kwargs["model_params"] = model_params

    if model_class is not None and "model_class" not in kwargs:
        print("WARN: training.load_model has no model_class kwarg; trying legacy call")
    if model_params is not None and "model_params" not in kwargs:
        print("WARN: training.load_model has no model_params kwarg; trying legacy call")

    return training.load_model(model_path, **kwargs)

# GPU detection (for NN inference)
try:
    import torch
    nn_device = "cuda" if torch.cuda.is_available() else "cpu"
except Exception:
    nn_device = "cpu"

# Skip if results already exist
existing_eval_path = METADATA_DIR / "transfer_evaluation.json"
print(f"Rerun flags: full={FULL_RERUN}, transfer_eval={RERUN_TRANSFER_EVAL}")
if existing_eval_path.exists() and not (FULL_RERUN or RERUN_TRANSFER_EVAL):
    print(f"✅ Existing results found: {existing_eval_path}")
    existing_data = json.loads(existing_eval_path.read_text())
    print(f"   ML Champion: {existing_data.get('metadata', {}).get('ml_champion')}")
    print(f"   NN Champion: {existing_data.get('metadata', {}).get('nn_champion')}")
    raise SystemExit("Skipping 03c: results already exist")


# %%
# ============================================================================
# 5. LOAD BERLIN MODELS AND METADATA
# ============================================================================

log.start_step("Load Berlin Models")

try:
    print("\n" + "=" * 70)
    print("Loading Berlin Champions")
    print("=" * 70)
    
    # Load ML champion
    ml_model_path = MODELS_DIR / "berlin_ml_champion.pkl"
    if not ml_model_path.exists():
        raise FileNotFoundError(
            f"ML champion not found at {ml_model_path}\n"
            "Run 03b_berlin_optimization.ipynb first."
        )
    
    # Load model and metadata separately (no return_metadata parameter)
    ml_model = load_model_compat(ml_model_path)
    ml_metadata_path = ml_model_path.with_suffix(ml_model_path.suffix + ".metadata.json")
    ml_metadata = json.loads(ml_metadata_path.read_text())
    
    ml_name = ml_metadata["model_name"]
    ml_feature_cols = ml_metadata["feature_columns"]
    
    print(f"\n✅ Loaded ML champion: {ml_name}")
    print(f"   Features: {len(ml_feature_cols)} (reduced)")
    
    # Load NN champion (optional)
    nn_model_path_pt = MODELS_DIR / "berlin_nn_champion.pt"
    nn_model_path_zip = MODELS_DIR / "berlin_nn_champion.zip"
    nn_model_path = nn_model_path_pt if nn_model_path_pt.exists() else nn_model_path_zip
    if nn_model_path.exists():
        nn_metadata_path = nn_model_path.with_suffix(nn_model_path.suffix + ".metadata.json")
        nn_metadata = json.loads(nn_metadata_path.read_text())
        nn_name = nn_metadata["model_name"]
        nn_feature_cols = nn_metadata["feature_columns"]
        all_params = nn_metadata.get("best_params", {})
        if nn_name == "cnn_1d":
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
            model_params = {k: v for k, v in all_params.items() if k in model_param_keys}
        else:
            model_params = all_params

        # Load NN model with model-specific params
        nn_model = load_model_compat(
            nn_model_path,
            model_class=models.CNN1D if nn_name == "cnn_1d" else None,
            model_params=model_params,
        )
        
        print(f"✅ Loaded NN champion: {nn_name}")
        print(f"   Features: {len(nn_feature_cols)} (full temporal)")
    else:
        nn_model = None
        nn_name = None
        nn_feature_cols = None
        nn_metadata = None
        print(f"ℹ️  No NN champion found - skipping NN evaluation")
    
    # Load label encoder
    with (MODELS_DIR / "label_encoder.pkl").open("rb") as f:
        label_encoder = pickle.load(f)
        label_to_idx = label_encoder["label_to_idx"]
        idx_to_label = label_encoder["idx_to_label"]
    
    # Load scalers (separate for ML and NN)
    with (MODELS_DIR / "berlin_scaler_ml.pkl").open("rb") as f:
        berlin_scaler_ml = pickle.load(f)
    
    if nn_model is not None:
        scaler_nn_path = MODELS_DIR / "berlin_scaler_nn.pkl"
        if scaler_nn_path.exists():
            with scaler_nn_path.open("rb") as f:
                berlin_scaler_nn = pickle.load(f)
        else:
            # Fallback to ML scaler if NN scaler doesn't exist
            berlin_scaler_nn = berlin_scaler_ml
            print("⚠️  Using ML scaler for NN (NN scaler not found)")
    else:
        berlin_scaler_nn = None
    
    # Set defaults for backward compatibility
    feature_cols = ml_feature_cols
    berlin_scaler = berlin_scaler_ml
    
    class_labels = list(idx_to_label.values())
    n_classes = len(class_labels)
    
    print(f"\n  Classes: {n_classes}")
    
    # Load Berlin evaluation results
    berlin_eval_path = METADATA_DIR / "berlin_evaluation.json"
    if not berlin_eval_path.exists():
        raise FileNotFoundError(
            f"Berlin evaluation not found at {berlin_eval_path}\n"
            "Run 03b_berlin_optimization.ipynb first."
        )
    
    berlin_eval = json.loads(berlin_eval_path.read_text())
    berlin_f1 = berlin_eval["metrics"]["f1_score"]
    
    print(f"\n✅ Loaded Berlin evaluation (F1: {berlin_f1:.4f})")
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %% [markdown]
# # ⚠️ Genus Selection Already Applied
#
# **Note:** Genus selection filtering is already applied in 03a when creating the datasets. The loaded Parquet files contain only viable genera with grouping already applied.
#
# No additional filtering needed here!

# %%
# ============================================================================
# 6. LOAD LEIPZIG TEST DATA
# ============================================================================

log.start_step("Load Leipzig Data")

try:
    print("\n" + "=" * 70)
    print("Loading Leipzig Data")
    print("=" * 70)
    
    # Load ML datasets (reduced features - always needed)
    print(f"\n[ML Champion Datasets - Reduced Features]")
    leipzig_finetune_ml, leipzig_test_ml = data_loading.load_leipzig_splits(INPUT_DIR)
    
    print(f"  Finetune: {len(leipzig_finetune_ml):,} samples, {len(ml_feature_cols)} features")
    print(f"  Test:     {len(leipzig_test_ml):,} samples")
    
    # Memory optimization for ML datasets
    float_cols = leipzig_test_ml.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        leipzig_test_ml[float_cols] = leipzig_test_ml[float_cols].astype("float32")
        leipzig_finetune_ml[float_cols] = leipzig_finetune_ml[float_cols].astype("float32")
        print(f"  Optimized: {len(float_cols)} float64→float32")
    
    # Extract features and labels for Leipzig test (ML)
    x_test_ml = leipzig_test_ml[ml_feature_cols].to_numpy()
    y_test = leipzig_test_ml["genus_latin"].map(label_to_idx).to_numpy()
    
    # Scale with Berlin ML scaler
    x_test_scaled_ml = berlin_scaler_ml.transform(x_test_ml)
    
    print(f"  ✅ ML datasets prepared (scaled with berlin_scaler_ml)")
    
    # Set defaults for backward compatibility
    leipzig_finetune, leipzig_test = leipzig_finetune_ml, leipzig_test_ml
    x_test = x_test_ml
    x_test_scaled = x_test_scaled_ml
    
    # Load NN datasets if NN champion exists (full features)
    if nn_model is not None:
        print(f"\n[NN Champion Datasets - Full Temporal Features]")
        leipzig_finetune_nn, leipzig_test_nn = data_loading.load_leipzig_splits_cnn(INPUT_DIR)
        
        print(f"  Finetune: {len(leipzig_finetune_nn):,} samples, {len(nn_feature_cols)} features")
        print(f"  Test:     {len(leipzig_test_nn):,} samples")
        
        # Memory optimization for NN datasets
        float_cols_nn = leipzig_test_nn.select_dtypes(include=["float64"]).columns
        if len(float_cols_nn) > 0:
            leipzig_test_nn[float_cols_nn] = leipzig_test_nn[float_cols_nn].astype("float32")
            leipzig_finetune_nn[float_cols_nn] = leipzig_finetune_nn[float_cols_nn].astype("float32")
            print(f"  Optimized: {len(float_cols_nn)} float64→float32")
        
        # Extract features for NN (using NN feature columns)
        x_test_nn = leipzig_test_nn[nn_feature_cols].to_numpy()
        
        # Scale with Berlin NN scaler
        x_test_scaled_nn = berlin_scaler_nn.transform(x_test_nn)
        
        print(f"  ✅ NN datasets prepared (scaled with berlin_scaler_nn)")
        
        print(f"\nℹ️  Dual datasets loaded:")
        print(f"   - ML champion will use {len(ml_feature_cols)} reduced features")
        print(f"   - NN champion will use {len(nn_feature_cols)} full temporal features")
    else:
        leipzig_finetune_nn, leipzig_test_nn = None, None
        x_test_nn, x_test_scaled_nn = None, None
    
    print("\n✅ Leipzig test data prepared")
    
    log.end_step(status="success", records=len(leipzig_test_ml))
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 7. ZERO-SHOT EVALUATION (ML CHAMPION)
# ============================================================================

log.start_step("Zero-Shot ML Evaluation")

try:
    print("\n" + "=" * 70)
    print(f"Zero-Shot Transfer: {ml_name.upper()}")
    print("=" * 70)
    
    # Predict with Berlin model on Leipzig data (using ML-specific scaled data)
    ml_preds = ml_model.predict(x_test_scaled_ml)
    
    # Compute metrics with bootstrap CIs
    ml_transfer_metrics = transfer.compute_transfer_metrics(
        y_test,
        ml_preds,
        class_labels=class_labels,
        include_ci=True,
        n_bootstrap=config["metrics"]["n_bootstrap"],
    )
    
    leipzig_f1 = ml_transfer_metrics["metrics"]["f1_score"]
    
    print(f"\nZero-Shot Transfer Results:")
    print(f"  F1:       {leipzig_f1:.4f}")
    print(f"  Accuracy: {ml_transfer_metrics['metrics']['accuracy']:.4f}")
    
    if "confidence_intervals" in ml_transfer_metrics:
        f1_ci = ml_transfer_metrics["confidence_intervals"]["f1_score"]
        print(f"  F1 95% CI: [{f1_ci[0]:.4f}, {f1_ci[1]:.4f}]")
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 8. TRANSFER GAP ANALYSIS
# ============================================================================

log.start_step("Transfer Gap Analysis")

try:
    print("\n" + "=" * 70)
    print("Transfer Gap Analysis")
    print("=" * 70)
    
    # Compute transfer gap
    gap = transfer.compute_transfer_gap(berlin_f1, leipzig_f1)
    
    # Compute transfer efficiency manually (not in TransferGap dataclass)
    transfer_efficiency = 1.0 - gap.relative_drop
    
    print(f"\n  Berlin F1:          {berlin_f1:.4f}")
    print(f"  Leipzig F1:         {leipzig_f1:.4f}")
    print(f"  Absolute drop:      {gap.absolute_drop:.4f}")
    print(f"  Relative drop:      {gap.relative_drop:.1%}")
    print(f"  Transfer efficiency: {transfer_efficiency:.1%}")
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 9. LEIPZIG FROM-SCRATCH BASELINE (M5)
# ============================================================================

log.start_step("Leipzig From-Scratch Baseline")

try:
    print("\n" + "=" * 70)
    print("Leipzig From-Scratch Baseline")
    print("=" * 70)
    
    # Extract features and labels for Leipzig finetune (using ML features)
    x_finetune = leipzig_finetune_ml[ml_feature_cols].to_numpy()
    y_finetune = leipzig_finetune_ml["genus_latin"].map(label_to_idx).to_numpy()
    sample_weight = preprocessing.compute_sample_weights(y_finetune)
    
    # Fit Leipzig-specific scaler (IMPORTANT for fair comparison!)
    print(f"\nFitting Leipzig-specific scaler...")
    leipzig_scaler = StandardScaler()
    x_finetune_scaled = leipzig_scaler.fit_transform(x_finetune)
    x_test_leipzig_scaled = leipzig_scaler.transform(x_test_ml)
    
    # Train from scratch with same hyperparameters as Berlin model
    from_scratch_params = ml_metadata.get("best_params", {})
    
    print(f"Training from scratch with Berlin hyperparameters...")
    print(f"  Training samples: {len(x_finetune_scaled):,}")
    
    from_scratch_model = models.create_model(ml_name, model_params=from_scratch_params)
    from_scratch_model.fit(x_finetune_scaled, y_finetune, sample_weight=sample_weight)
    
    # Evaluate
    from_scratch_preds = from_scratch_model.predict(x_test_leipzig_scaled)
    from_scratch_metrics = evaluation.compute_metrics(y_test, from_scratch_preds)
    from_scratch_f1 = from_scratch_metrics["f1_score"]
    
    print(f"\nFrom-Scratch Results:")
    print(f"  F1: {from_scratch_f1:.4f}")
    print(f"\nComparison:")
    print(f"  Transfer (zero-shot): {leipzig_f1:.4f}")
    print(f"  From-Scratch:         {from_scratch_f1:.4f}")
    print(f"  Difference:           {leipzig_f1 - from_scratch_f1:+.4f}")
    
    # Memory cleanup
    del x_finetune_scaled, x_test_leipzig_scaled
    gc.collect()
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 10. FEATURE STABILITY ANALYSIS
# ============================================================================

log.start_step("Feature Stability")

try:
    print("\n" + "=" * 70)
    print("Feature Stability Analysis")
    print("=" * 70)
    
    # Check if models have feature_importances_
    if hasattr(ml_model, "feature_importances_") and hasattr(
        from_scratch_model, "feature_importances_"
    ):
        # Get Berlin importance (using ML feature columns)
        berlin_importance = pd.DataFrame(
            {"feature": ml_feature_cols, "importance": ml_model.feature_importances_}
        )
        
        # Get Leipzig importance (using ML feature columns)
        leipzig_importance = pd.DataFrame(
            {"feature": ml_feature_cols, "importance": from_scratch_model.feature_importances_}
        )
        
        # Compute stability (returns dict with spearman_rho, n_features)
        stability = transfer.compute_feature_stability(
            berlin_importance, leipzig_importance
        )
        
        # Compute p-value separately if needed
        from scipy.stats import spearmanr
        merged = pd.merge(berlin_importance, leipzig_importance, on='feature', suffixes=('_berlin', '_leipzig'))
        _, p_value = spearmanr(merged['importance_berlin'], merged['importance_leipzig'])
        
        print(f"\nFeature Stability:")
        print(f"  Spearman ρ: {stability['spearman_rho']:.3f}")
        print(f"  p-value:    {p_value:.4f}")
        print(f"  n_features: {int(stability['n_features'])}")
        
        if p_value < 0.05:
            print(
                f"  ✅ Significant correlation between Berlin and Leipzig "
                f"feature rankings"
            )
        else:
            print(f"  ⚠️  No significant correlation in feature rankings")
        
        # Store p-value for later use
        stability['p_value'] = p_value
    else:
        print(f"\nℹ️  Feature importance not available for {ml_name}")
        stability = None
        berlin_importance = None
        leipzig_importance = None
        p_value = None
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 11. PER-GENUS TRANSFER ROBUSTNESS (6.3)
# ============================================================================

log.start_step("Per-Genus Robustness")

try:
    print("\n" + "=" * 70)
    print("Per-Genus Transfer Robustness")
    print("=" * 70)
    
    # Get per-genus metrics
    berlin_per_genus = pd.DataFrame(berlin_eval["per_class"])
    leipzig_per_genus = ml_transfer_metrics["per_class"]
    
    # Convert DataFrames to dicts (genus -> f1_score)
    berlin_per_genus_dict = dict(zip(berlin_per_genus['genus'], berlin_per_genus['f1_score']))
    leipzig_per_genus_dict = dict(zip(leipzig_per_genus['genus'], leipzig_per_genus['f1_score']))
    
    # Get thresholds from config
    thresholds = config["transfer_evaluation"]["robustness_thresholds"]
    
    # Classify robustness
    robustness = transfer.classify_transfer_robustness(
        berlin_per_genus_dict,
        leipzig_per_genus_dict,
        robust_threshold=thresholds.get("robust", 0.05),
        medium_threshold=thresholds.get("medium", 0.15),
    )
    
    # Compute ranking (returns DataFrame)
    ranking_df = transfer.compute_transfer_robustness_ranking(robustness)
    ranking = ranking_df['genus'].tolist()  # Convert to list for iteration
    
    # Summary (use "label" field from robustness dict)
    print(f"\nTransfer Robustness Summary:")
    for category in ["robust", "medium", "poor"]:
        genera = [g for g in ranking if robustness[g]["label"] == category]
        print(f"  {category.capitalize():8s}: {len(genera):2d} genera")
    
    print(f"\nTop 5 Most Robust:")
    for genus in ranking[:5]:
        absolute_drop = robustness[genus]["absolute_drop"]
        label = robustness[genus]["label"]
        print(f"  {genus:15s} F1 drop: {absolute_drop:+.3f} ({label})")
    
    print(f"\nTop 5 Least Robust:")
    for genus in ranking[-5:]:
        absolute_drop = robustness[genus]["absolute_drop"]
        label = robustness[genus]["label"]
        print(f"  {genus:15s} F1 drop: {absolute_drop:+.3f} ({label})")
    
    log.end_step(status="success")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 12. HYPOTHESIS TESTING (6.1)
# ============================================================================

log.start_step("Hypothesis Testing")

try:
    print("\n" + "=" * 70)
    print("A-Priori Hypotheses Testing")
    print("=" * 70)

    setup_path = METADATA_DIR / "setup_decisions.json"
    if not setup_path.exists():
        raise FileNotFoundError(
            f"setup_decisions.json not found at {setup_path}\n"
            "Run exp_10_genus_selection_validation.ipynb first."
        )
    setup = json.loads(setup_path.read_text())

    grouping_analysis = setup["genus_selection"]["grouping_analysis"]
    genus_to_final_mapping = grouping_analysis["genus_to_final_mapping"]
    mean_jm_per_genus = grouping_analysis["mean_jm_per_genus"]

    final_to_members: dict[str, list[str]] = {}
    for original_genus, final_genus in genus_to_final_mapping.items():
        final_to_members.setdefault(final_genus, []).append(original_genus)

    conifer_genera = set(config["genus_groups"]["conifer"])

    genus_summary = pd.merge(
        berlin_per_genus[["genus", "f1_score", "support"]].rename(
            columns={"f1_score": "berlin_f1", "support": "berlin_n"}
        ),
        leipzig_per_genus[["genus", "f1_score"]].rename(columns={"f1_score": "leipzig_f1"}),
        on="genus",
    )
    genus_summary["transfer_gap"] = genus_summary["berlin_f1"] - genus_summary["leipzig_f1"]
    genus_summary["is_conifer"] = genus_summary["genus"].map(
        lambda genus: any(member in conifer_genera for member in final_to_members.get(genus, [genus]))
    )
    genus_summary["mean_jm_distance"] = genus_summary["genus"].map(
        lambda genus: float(
            np.mean([mean_jm_per_genus[member] for member in final_to_members.get(genus, [genus])])
        )
    )

    sample_data = pd.DataFrame(
        {
            "genus": leipzig_test_ml["genus_latin"].to_numpy(),
            "correct": (ml_preds == y_test).astype(int),
        }
    )

    print(f"\nPrepared genus summary rows: {len(genus_summary)}")
    print(f"Prepared per-sample rows:     {len(sample_data)}")

    hypotheses = {hyp["id"]: hyp for hyp in config["transfer_evaluation"]["hypotheses"]}
    hypothesis_results = []

    for hyp_id in ["H1", "H2", "H3"]:
        hyp = hypotheses[hyp_id]
        test_data = sample_data if hyp_id == "H1" else genus_summary

        print(f"\n{hyp['id']}: {hyp['description']}")
        result = transfer.test_hypothesis(hyp, genus_data=test_data)
        hypothesis_results.append(result)

        if "statistic" in result:
            print(f"  Statistic:   {result['statistic']:.3f}")
        if "p_value" in result:
            print(f"  p-value:     {result['p_value']:.4f}")
        if "effect_size" in result:
            print(f"  Effect size: {result['effect_size']:.3f}")
        print(f"  Result:      {result['conclusion']}")

    hypothesis_path = METADATA_DIR / "hypothesis_tests.json"
    hypothesis_path.write_text(json.dumps(hypothesis_results, indent=2))
    print(f"\n✅ Saved: {hypothesis_path.name}")

    log.end_step(status="success", records=len(hypothesis_results))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 13. NN CHAMPION EVALUATION (IF EXISTS)
# ============================================================================

log.start_step("NN Champion Evaluation")

try:
    if nn_model is None:
        print("\nℹ️  No NN champion - skipping NN evaluation")
        nn_transfer_metrics = None
        best_transfer_model = "ml"
        best_transfer_f1 = leipzig_f1
        log.end_step(status="skipped")
    else:
        print("\n" + "=" * 70)
        print(f"NN Champion Evaluation: {nn_name.upper()}")
        print("=" * 70)
        
        try:
            nn_preds = nn_model.predict(x_test_scaled_nn, device=nn_device)
            nn_transfer_metrics = transfer.compute_transfer_metrics(
                y_test,
                nn_preds,
                class_labels=class_labels,
                include_ci=True,
                n_bootstrap=config["metrics"]["n_bootstrap"],
            )
            nn_leipzig_f1 = nn_transfer_metrics["metrics"]["f1_score"]

            print(f"\nNN Zero-Shot Transfer:")
            print(f"  F1:       {nn_leipzig_f1:.4f}")
            print(f"  Accuracy: {nn_transfer_metrics['metrics']['accuracy']:.4f}")

            print(f"\nML vs NN Comparison:")
            print(f"  ML F1: {leipzig_f1:.4f}")
            print(f"  NN F1: {nn_leipzig_f1:.4f}")
            print(f"  Difference: {nn_leipzig_f1 - leipzig_f1:+.4f}")

            if nn_leipzig_f1 > leipzig_f1:
                best_transfer_model = "nn"
                best_transfer_f1 = nn_leipzig_f1
                print(f"\n  ✅ Best transfer model: NN ({nn_name})")
            else:
                best_transfer_model = "ml"
                best_transfer_f1 = leipzig_f1
                print(f"\n  ✅ Best transfer model: ML ({ml_name})")
        except Exception as exc:
            error_message = f"{type(exc).__name__}: {exc}"
            print("\n⚠️  NN evaluation failed")
            print(f"  Error type:    {type(exc).__name__}")
            print(f"  Error message: {exc}")
            print(f"  Test shape:    {x_test_scaled_nn.shape}")

            nn_transfer_metrics = {
                "metrics": {
                    "f1_score": None,
                    "accuracy": None,
                    "precision": None,
                    "recall": None,
                },
                "confidence_intervals": None,
                "per_class": None,
                "confusion_matrix": None,
                "error": error_message,
            }
            best_transfer_model = "ml"
            best_transfer_f1 = leipzig_f1
        
        log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 14. SAVE RESULTS
# ============================================================================

log.start_step("Save Results")

try:
    hypothesis_path = METADATA_DIR / "hypothesis_tests.json"
    if not hypothesis_path.exists():
        raise FileNotFoundError(
            f"Hypothesis test results not found at {hypothesis_path}\n"
            "Run Section 8 hypothesis testing first."
        )

    hypothesis_results = json.loads(hypothesis_path.read_text())
    required_hypothesis_keys = {
        "id",
        "description",
        "test_type",
        "statistic",
        "p_value",
        "effect_size",
        "conclusion",
    }
    for result in hypothesis_results:
        missing = required_hypothesis_keys.difference(result)
        if missing:
            raise ValueError(
                f"Invalid hypothesis test result for {result.get('id', 'unknown')}: missing {sorted(missing)}"
            )

    # Compile transfer evaluation data (schema-compliant)
    transfer_eval_data = {
        "metrics": ml_transfer_metrics["metrics"],
        "per_class": ml_transfer_metrics["per_class"].to_dict(orient="records"),
        "confusion_matrix": ml_transfer_metrics["confusion_matrix"].tolist(),
        "metadata": {
            "ml_champion": ml_name,
            "nn_champion": nn_name if nn_name else None,
            "best_transfer_model": best_transfer_model,
            "best_transfer_f1": best_transfer_f1,
            "ml_zero_shot_metrics": ml_transfer_metrics["metrics"],
            "nn_zero_shot_metrics": (
                nn_transfer_metrics["metrics"] if nn_transfer_metrics else None
            ),
            "transfer_gap": {
                "berlin_f1": berlin_f1,
                "leipzig_f1": leipzig_f1,
                "absolute_drop": gap.absolute_drop,
                "relative_drop": gap.relative_drop,
                "transfer_efficiency": transfer_efficiency,  # ✅ From manual calculation
            },
            "from_scratch_f1": from_scratch_f1,
            "feature_stability": stability if stability else None,
            "per_genus_robustness": robustness,
            "robustness_ranking": ranking,
            "hypothesis_tests_path": hypothesis_path.name,
        },
    }
    
    # Save transfer evaluation
    eval_path = METADATA_DIR / "transfer_evaluation.json"
    eval_path.write_text(json.dumps(transfer_eval_data, indent=2))
    validate_evaluation_metrics(eval_path)
    
    print(f"\n✅ Saved: {eval_path.name}")
    
    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
try:
    report_path = REPORT_DIR / "transfer_metrics.json"
    report_records = []
    for row in genus_summary.itertuples(index=False):
        relative_drop = None
        if row.berlin_f1 > 0:
            relative_drop = row.transfer_gap / row.berlin_f1
        report_records.append(
            {
                "genus": row.genus,
                "berlin_f1": row.berlin_f1,
                "leipzig_f1": row.leipzig_f1,
                "is_conifer": bool(row.is_conifer),
                "absolute_drop": row.transfer_gap,
                "relative_drop": relative_drop,
            }
        )
    report_path.write_text(json.dumps(report_records, indent=2), encoding="utf-8")
    print(f"Saved report JSON: {report_path}")
except Exception as e:
    print(f"WARNING: Failed to export report JSONs: {e}")


# %%
# ============================================================================
# 15. SAVE EXECUTION LOG
# ============================================================================

log.start_step("Save Execution Log")

try:
    elapsed = time.time() - start_time
    
    # Create execution summary (use "label" field from robustness)
    execution_summary = {
        "notebook": "03c_transfer_evaluation",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": elapsed,
        "runtime_minutes": elapsed / 60,
        "champions": {
            "ml": ml_name,
            "nn": nn_name if nn_name else None,
        },
        "best_transfer_model": best_transfer_model,
        "transfer_gap": {
            "berlin_f1": berlin_f1,
            "leipzig_f1": leipzig_f1,
            "absolute_drop": gap.absolute_drop,
            "relative_drop": gap.relative_drop,
            "transfer_efficiency": transfer_efficiency,
        },
        "robustness_summary": {
            "robust": len([g for g in ranking if robustness[g]["label"] == "robust"]),
            "medium": len([g for g in ranking if robustness[g]["label"] == "medium"]),
            "poor": len([g for g in ranking if robustness[g]["label"] == "poor"]),
        },
        "hypothesis_tests": len(hypothesis_results),
    }
    
    # Save execution log
    log.summary()
    log_path = LOGS_DIR / f"{log.notebook}_execution.json"
    log.save(log_path)
    
    # Save summary
    summary_path = METADATA_DIR / "03c_summary.json"
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

print(f"\nRuntime: {elapsed/60:.1f} minutes")

print(f"\nTransfer Performance:")
print(f"  Berlin F1:          {berlin_f1:.4f}")
print(f"  Leipzig F1:         {leipzig_f1:.4f}")
print(f"  Transfer gap:       {gap.relative_drop:.1%}")
print(f"  Transfer efficiency: {transfer_efficiency:.1%}")  # ✅ Use variable from Cell 9

print(f"\nBaseline Comparison:")
print(f"  Transfer (zero-shot): {leipzig_f1:.4f}")
print(f"  From-Scratch:         {from_scratch_f1:.4f}")
print(f"  Difference:           {leipzig_f1 - from_scratch_f1:+.4f}")

if nn_model and nn_transfer_metrics["metrics"]["f1_score"] is not None:
    print(f"\nML vs NN:")
    print(f"  ML F1: {leipzig_f1:.4f}")
    print(f"  NN F1: {nn_transfer_metrics['metrics']['f1_score']:.4f}")
    print(f"  Best: {best_transfer_model.upper()}")
elif nn_model:
    print(f"\nML vs NN:")
    print(f"  ML F1: {leipzig_f1:.4f}")
    print(f"  NN evaluation failed: {nn_transfer_metrics.get('error')}")
    print(f"  Best: {best_transfer_model.upper()}")

print(f"\nRobustness Summary:")
for category in ["robust", "medium", "poor"]:
    count = len([g for g in ranking if robustness[g]["label"] == category])  # ✅ Use "label" not "category"
    print(f"  {category.capitalize()}: {count} genera")

print(f"\nOutputs:")
print(f"  Metadata: {eval_path.name}")
print(f"  Summary:  03c_summary.json")
print(f"  Logs:     {log_path}")

print(f"\nNext Steps:")
print(f"  1. Review transfer gap analysis")
print(f"  2. Inspect per-genus robustness classification")
print(f"  3. Analyze hypothesis test results")
print(f"  4. Run 03d_finetuning.ipynb (sample efficiency)")
print("=" * 70)
