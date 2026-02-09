# PRD: Phase 3 Experiments

**PRD ID:** 003
**Status:** Draft
**Created:** 2026-02-03
**Last Updated:** 2026-02-03

---

## 1. Overview

### 1.1 Problem Statement

Phase 2 Feature Engineering has produced ML-ready datasets with spatial train/val/test splits for Berlin and a finetune/test split for Leipzig. Phase 3 must now conduct a systematic series of experiments to answer the core research questions about cross-city transfer learning for tree genus classification.

The experiments must follow a structured methodology:

1. Fix experimental setup (CHM strategy, feature set) before algorithm comparison
2. Optimize models for Berlin (source city) first
3. Evaluate zero-shot transfer to Leipzig (target city)
4. Measure fine-tuning sample efficiency

### 1.2 Research Questions

| ID      | Question                                                                                    | Experiment Phase |
| ------- | ------------------------------------------------------------------------------------------- | ---------------- |
| **RQ1** | What is the best achievable performance on Berlin with Sentinel-2 + CHM features?           | Phase 3.1–3.3    |
| **RQ2** | How much does performance drop when applying a Berlin-trained model to Leipzig (zero-shot)? | Phase 3.4        |
| **RQ3** | How much Leipzig fine-tuning data is needed to recover performance?                         | Phase 3.5        |

### 1.3 Goals

1. **Establish Berlin Upper Bound:** Optimal single-city classification performance
2. **Quantify Transfer Gap:** Measure zero-shot performance degradation
3. **Sample Efficiency Curve:** Determine minimum fine-tuning data required
4. **Methodological Rigor:** Reproducible experiments with confidence intervals
5. **Comparable Architectures:** Test both tree-based (ML) and neural (NN) approaches

### 1.4 Non-Goals

- Domain adaptation methods (future work)
- Multi-city joint training (future work)
- Species-level classification
- Real-time inference optimization

---

## 2. Architecture

### 2.1 Directory Structure

```
urban-tree-transfer/
├── configs/
│   ├── cities/                              # Existing from Phase 1
│   ├── features/                            # Existing from Phase 2
│   └── experiments/
│       └── phase3_config.yaml               # NEW: Experiment configuration
├── src/urban_tree_transfer/
│   └── experiments/
│       ├── __init__.py                      # Exports
│       ├── data_loading.py                  # Dataset loading utilities
│       ├── preprocessing.py                 # Feature scaling, encoding
│       ├── models.py                        # Model factory functions
│       ├── training.py                      # Training loops, CV
│       ├── evaluation.py                    # Metrics, confidence intervals
│       ├── transfer.py                      # Transfer-specific metrics
│       ├── ablation.py                      # Ablation study utilities
│       ├── hp_tuning.py                     # Optuna integration
│       └── visualization.py                 # Standardized plots
├── notebooks/
│   ├── runners/
│   │   ├── 03a_setup_fixation.ipynb         # NEW: Fix CHM, features
│   │   ├── 03b_berlin_optimization.ipynb    # NEW: Algorithm comparison, HP-tuning
│   │   ├── 03c_transfer_evaluation.ipynb    # NEW: Zero-shot transfer
│   │   └── 03d_finetuning.ipynb             # NEW: Fine-tuning experiments
│   └── exploratory/
│       ├── exp_07_cross_city_baseline.ipynb # NEW: Descriptive analysis
│       ├── exp_08_chm_ablation.ipynb        # NEW: CHM strategy decision
│       ├── exp_08b_proximity_ablation.ipynb  # NEW: Baseline vs. filtered dataset
│       ├── exp_08c_outlier_ablation.ipynb    # NEW: Outlier removal strategy
│       ├── exp_09_feature_reduction.ipynb   # NEW: Top-k feature selection
│       └── exp_10_algorithm_comparison.ipynb # NEW: Coarse HP comparison
├── schemas/
│   ├── setup_decisions.schema.json          # NEW
│   ├── algorithm_comparison.schema.json     # NEW
│   ├── hp_tuning_result.schema.json         # NEW
│   ├── evaluation_metrics.schema.json       # NEW
│   └── finetuning_curve.schema.json         # NEW
├── docs/documentation/
│   └── 03_Experiments/
│       ├── 00_Experiment_Overview.md         # Experiment overview & structure
│       ├── 01_Setup_Fixierung.md            # Setup decisions (CHM, proximity, outliers, features)
│       ├── 02_Berlin_Optimierung.md         # Berlin optimization (algorithms & HP-tuning)
│       ├── 03_Transfer_Evaluation.md        # Transfer evaluation methodology
│       ├── 04_Finetuning.md                 # Fine-tuning methodology
│       └── 05_Methodische_Erweiterungen.md  # Not implemented options for future work
└── outputs/
    └── phase_3/                             # NEW
        ├── metadata/
        │   ├── setup_decisions.json         # From exp_08, exp_08b, exp_08c, exp_09
        │   ├── algorithm_comparison.json    # From exp_10
        │   ├── hp_tuning_ml.json            # From 03b
        │   ├── hp_tuning_nn.json            # From 03b
        │   ├── berlin_evaluation.json       # From 03b
        │   ├── transfer_evaluation.json     # From 03c
        │   └── finetuning_curve.json        # From 03d
        ├── models/
        │   ├── berlin_ml_champion.pkl       # Trained ML model
        │   ├── berlin_nn_champion.pt        # Trained NN model
        │   └── finetuned/                   # Fine-tuned checkpoints
        ├── figures/
        │   ├── exp_07_baseline/             # Cross-city visualizations
        │   ├── exp_08_chm_ablation/         # CHM decision plots
        │   ├── exp_08b_proximity/           # Proximity filter decision plots
        │   ├── exp_08c_outlier/             # Outlier removal decision plots
        │   ├── exp_09_feature_reduction/    # Feature reduction plots
        │   ├── exp_10_algorithm_comparison/ # Model comparison plots
        │   ├── berlin_optimization/         # HP-tuning, feature importance
        │   ├── transfer/                    # Transfer confusion matrices
        │   └── finetuning/                  # Fine-tuning curves
        └── logs/
            ├── 03a_setup_fixation.json
            ├── 03b_berlin_optimization.json
            ├── 03c_transfer_evaluation.json
            └── 03d_finetuning.json
```

### 2.2 Data Flow

```
Phase 2 Outputs (ML-Ready Datasets, data/phase_2_splits/)
├── berlin_train.parquet (70%)        # ML-optimized (no geometry)
├── berlin_val.parquet (15%)
├── berlin_test.parquet (15%)
├── leipzig_finetune.parquet (80%)
├── leipzig_test.parquet (20%)
├── geometry_lookup.parquet            # tree_id → x/y for visualization
└── *.gpkg                             # Authoritative GeoPackages (kept for traceability)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_07_cross_city_baseline.ipynb (OPTIONAL, parallel)      │
│  ├── Class distribution comparison                          │
│  ├── Phenological profile comparison                        │
│  ├── CHM distribution comparison                            │
│  ├── Feature distribution comparison                        │
│  ├── Cohen's d heatmap (city differences)                   │
│  └── Correlation structure comparison                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼ (figures only, no config output)

┌─────────────────────────────────────────────────────────────┐
│  exp_08_chm_ablation.ipynb                                  │
│  ├── Compare: No CHM vs. zscore vs. percentile vs. both     │
│  ├── Feature importance analysis per variant                │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (chm_strategy)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_08b_proximity_ablation.ipynb                           │
│  ├── Compare: Baseline vs. Proximity-filtered datasets      │
│  ├── 3-Fold Spatial Block CV with RF                        │
│  ├── Dataset size impact analysis                           │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (+ proximity_strategy)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_08c_outlier_ablation.ipynb                             │
│  ├── Compare: No removal vs. high vs. high+medium           │
│  ├── 3-Fold Spatial Block CV with RF                        │
│  ├── Sample size vs. F1 trade-off                           │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (+ outlier_strategy)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_09_feature_reduction.ipynb                             │
│  ├── RF feature importance ranking                          │
│  ├── Compare: Top-30 vs. Top-50 vs. Top-80 vs. All          │
│  ├── Pareto curve (F1 vs. feature count)                    │
│  └── Decision logging                                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            setup_decisions.json (feature_set, selected_features)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  03a_setup_fixation.ipynb (Runner)                          │
│  ├── Load and validate setup_decisions.json                 │
│  ├── Create feature-reduced datasets                        │
│  └── Save processed datasets for experiments                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Processed datasets (with fixed setup)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  exp_10_algorithm_comparison.ipynb                          │
│  ├── Random Forest (coarse HP)                              │
│  ├── XGBoost (coarse HP)                                    │
│  ├── 1D-CNN (baseline config)                               │
│  ├── TabNet (baseline config)                               │
│  ├── 3-Fold Spatial Block CV                                │
│  └── Champion selection (1 ML + 1 NN)                       │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            algorithm_comparison.json (ml_champion, nn_champion)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  03b_berlin_optimization.ipynb (Runner)                     │
│  ├── Load champions from algorithm_comparison.json          │
│  ├── Optuna HP-tuning for ML champion (50+ trials)          │
│  ├── Optuna HP-tuning for NN champion (50+ trials)          │
│  ├── Final training on Train+Val                            │
│  ├── Evaluation on Berlin Test (hold-out)                   │
│  ├── Feature importance analysis                            │
│  ├── Per-genus performance analysis                         │
│  └── Save trained models                                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            hp_tuning_ml.json, hp_tuning_nn.json
            berlin_evaluation.json
            berlin_ml_champion.pkl, berlin_nn_champion.pt
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  03c_transfer_evaluation.ipynb (Runner)                     │
│  ├── Load trained Berlin models                             │
│  ├── Zero-shot evaluation on Leipzig Test                   │
│  ├── Transfer gap calculation                               │
│  ├── Per-genus transfer analysis                            │
│  ├── Confusion matrix comparison                            │
│  └── Transfer robustness ranking                            │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            transfer_evaluation.json
            figures/transfer/
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  03d_finetuning.ipynb (Runner)                              │
│  ├── Fine-tune BOTH champions (ML + NN)                     │
│  ├── Fine-tune with 10%, 25%, 50%, 100% Leipzig data        │
│  ├── Evaluate each on Leipzig Test                          │
│  ├── Leipzig from-scratch baselines (ML + NN)               │
│  ├── Sample efficiency curves (ML vs. NN comparison)        │
│  └── Statistical significance tests                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            finetuning_curve.json
            figures/finetuning/
```

---

## 3. Configuration

### 3.1 Experiment Config (`configs/experiments/phase3_config.yaml`)

```yaml
# Phase 3 Experiment Configuration
# All hyperparameters, thresholds, and experiment settings

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
global:
  random_seed: 42
  n_jobs: -1 # Use all cores
  cv_folds: 3 # 3-fold for all CV during training/tuning (speed)
  spatial_block_size_m: 500 # From Phase 2

# Genus grouping for aggregate analyses
genus_groups:
  conifer: [PINUS, PICEA] # Nadelbäume
  deciduous: [TILIA, ACER, QUERCUS, PLATANUS, AESCULUS, BETULA, FRAXINUS, ROBINIA] # Laubbäume
  # Extend based on actual genus list in dataset

# Plant year bins for post-hoc analysis
plant_year_decades:
  - label: "vor 1960"
    max_year: 1959
  - label: "1960-79"
    min_year: 1960
    max_year: 1979
  - label: "1980-99"
    min_year: 1980
    max_year: 1999
  - label: "2000-19"
    min_year: 2000
    max_year: 2019
  - label: "ab 2020"
    min_year: 2020

# Visualization conventions
visualization:
  use_german_names: true # Always use genus_german for labels
  dpi: 300
  figure_format: png
  figsize_single: [8, 6]
  figsize_double: [14, 6]
  figsize_table: [10, 4]

# =============================================================================
# EVALUATION METRICS
# =============================================================================
metrics:
  primary: weighted_f1
  secondary:
    - macro_f1
    - accuracy
  per_class: true
  confidence_intervals: true
  ci_method: bootstrap # bootstrap or cross_validation
  n_bootstrap: 1000

# =============================================================================
# PHASE 3.1: SETUP ABLATION
# =============================================================================
setup_ablation:
  # CHM Strategy Comparison
  # Each feature is evaluated individually for inclusion
  chm:
    features:
      - name: CHM_1m
        description: "Raw CHM height (absolute meters, known overfitting risk)"
      - name: CHM_1m_zscore
        description: "Z-score normalized within genus×city"
      - name: CHM_1m_percentile
        description: "Percentile rank within genus×city (0-100)"

    # Ablation variants for comparison
    variants:
      - name: no_chm
        features: []
        description: "Sentinel-2 only (baseline, transfer-safe)"
      - name: zscore_only
        features: [CHM_1m_zscore]
        description: "Z-score normalized CHM"
      - name: percentile_only
        features: [CHM_1m_percentile]
        description: "Percentile rank CHM"
      - name: both_engineered
        features: [CHM_1m_zscore, CHM_1m_percentile]
        description: "Both engineered CHM features"
      - name: raw_chm
        features: [CHM_1m]
        description: "Raw CHM (known overfitting risk)"

    # Per-feature decision criteria (applied to each CHM feature independently)
    decision_rules:
      importance_threshold: 0.25 # Single feature >25% importance = problematic
      min_improvement: 0.03 # Must improve F1 >3% to justify inclusion
      max_gap_increase: 0.05 # Must not increase Train-Val gap by >5pp
      prefer_simpler: true # Prefer fewer CHM features if difference marginal

  # Proximity Filter Ablation
  proximity_filter:
    description: "Compare baseline vs. proximity-filtered datasets"
    variants:
      - name: baseline
        dataset_suffix: "" # berlin_train.parquet etc.
        description: "All trees, no proximity filter"
      - name: filtered
        dataset_suffix: "_filtered" # berlin_train_filtered.parquet etc.
        description: "Trees with ≥20m distance to different genus"

    # Decision criteria
    decision_rules:
      min_improvement: 0.02 # Filtered must improve F1 by >2pp
      max_sample_loss: 0.20 # Filtered must not lose >20% of samples
      prefer_larger_dataset: true # If F1 difference marginal, keep baseline

  # Outlier Removal Ablation
  outlier_removal:
    description: "Determine outlier removal strategy using severity flags"
    filter_column: outlier_severity # From Phase 2 quality flags
    count_column: outlier_method_count # Number of methods flagging as outlier (0-3)
    variants:
      - name: no_removal
        remove_levels: [] # Keep all samples
        description: "No outlier removal"
      - name: remove_high
        remove_levels: [high] # Remove only high-severity outliers
        description: "Remove trees flagged as high severity (≥2 methods)"
      - name: remove_high_medium
        remove_levels: [high, medium] # Remove high + medium
        description: "Remove trees flagged as high or medium severity"

    # Decision criteria
    decision_rules:
      min_improvement: 0.02 # Removal must improve F1 by >2pp
      max_sample_loss: 0.15 # Must not lose >15% of training samples
      prefer_no_removal: true # If tied, keep all data

  # Feature Reduction
  feature_reduction:
    method: rf_importance # Random Forest feature importance
    variants:
      - name: top_30
        n_features: 30
      - name: top_50
        n_features: 50
      - name: top_80
        n_features: 80
      - name: all_features
        n_features: null # Use all

    # Decision criteria
    decision_rules:
      max_performance_drop: 0.01 # Accept smallest k with ≤1% drop
      prefer_fewer: true

# =============================================================================
# PHASE 3.2: ALGORITHM COMPARISON
# =============================================================================
algorithms:
  # ML Algorithms
  random_forest:
    class: sklearn.ensemble.RandomForestClassifier
    type: ml
    coarse_grid:
      n_estimators: [200] # Fixed for speed
      max_depth: [10, 15, 20, null]
      min_samples_leaf: [5, 10, 20]
      min_samples_split: [10, 20]
      class_weight: [balanced]
    optuna_space:
      n_estimators:
        type: int
        low: 100
        high: 500
      max_depth:
        type: int
        low: 5
        high: 30
      min_samples_leaf:
        type: int
        low: 1
        high: 20
      min_samples_split:
        type: int
        low: 2
        high: 20

  xgboost:
    class: xgboost.XGBClassifier
    type: ml
    coarse_grid:
      n_estimators: [200, 300]
      max_depth: [4, 6, 8]
      learning_rate: [0.05, 0.1]
      min_child_weight: [3, 5]
      reg_alpha: [0, 0.1]
      reg_lambda: [1, 2]
      subsample: [0.8]
      colsample_bytree: [0.8]
    optuna_space:
      n_estimators:
        type: int
        low: 100
        high: 500
      max_depth:
        type: int
        low: 3
        high: 10
      learning_rate:
        type: float
        low: 0.01
        high: 0.3
        log: true
      min_child_weight:
        type: int
        low: 1
        high: 10
      reg_alpha:
        type: float
        low: 0.0
        high: 1.0
      reg_lambda:
        type: float
        low: 0.0
        high: 5.0
      subsample:
        type: float
        low: 0.6
        high: 1.0
      colsample_bytree:
        type: float
        low: 0.6
        high: 1.0

  # Neural Network Algorithms
  cnn_1d:
    class: urban_tree_transfer.experiments.models.TemporalCNN
    type: nn
    baseline_config:
      n_conv_blocks: 2
      filters: 64
      kernel_size: 3
      dropout: 0.3
      dense_units: 128
      learning_rate: 0.001
      batch_size: 128
      epochs: 100
      early_stopping_patience: 10
    optuna_space:
      n_conv_blocks:
        type: int
        low: 1
        high: 4
      filters:
        type: categorical
        choices: [32, 64, 128]
      kernel_size:
        type: int
        low: 2
        high: 5
      dropout:
        type: float
        low: 0.1
        high: 0.5
      dense_units:
        type: categorical
        choices: [64, 128, 256]
      learning_rate:
        type: float
        low: 0.0001
        high: 0.01
        log: true
      batch_size:
        type: categorical
        choices: [64, 128, 256]

  tabnet:
    class: pytorch_tabnet.tab_model.TabNetClassifier
    type: nn
    baseline_config:
      n_d: 32
      n_a: 32
      n_steps: 5
      gamma: 1.3
      lambda_sparse: 0.001
      learning_rate: 0.02
      batch_size: 512
      epochs: 200
      patience: 15
    optuna_space:
      n_d:
        type: categorical
        choices: [8, 16, 32, 64]
      n_a:
        type: categorical
        choices: [8, 16, 32, 64]
      n_steps:
        type: int
        low: 3
        high: 10
      gamma:
        type: float
        low: 1.0
        high: 2.0
      lambda_sparse:
        type: float
        low: 0.0001
        high: 0.01
        log: true

# Champion Selection
champion_selection:
  # Filter criteria (must pass all)
  filters:
    min_val_f1: 0.50 # Minimum validation F1
    max_train_val_gap: 0.35 # Maximum overfitting gap

  # Selection criteria (within passing candidates)
  primary: val_f1 # Maximize validation F1
  tiebreaker: train_val_gap # Minimize gap if tied

# =============================================================================
# PHASE 3.3: HP TUNING
# =============================================================================
hp_tuning:
  method: optuna
  n_trials: 50 # Minimum trials per algorithm
  timeout_hours: 2 # Maximum time per algorithm
  pruning: true # Enable Optuna pruning
  sampler: TPESampler
  cv_folds: 3 # Reduced for speed during tuning

  # Goals
  targets:
    min_val_f1: 0.60
    max_gap: 0.25

# =============================================================================
# PHASE 3.4: TRANSFER EVALUATION
# =============================================================================
transfer:
  # Transfer metrics
  metrics:
    - absolute_drop # F1_source - F1_target
    - relative_drop_pct # (F1_source - F1_target) / F1_source * 100
    - per_genus_transfer # Per-genus F1 comparison

  # Robustness classification thresholds
  robustness_thresholds:
    robust: 0.05 # Δ F1 < 0.05 = robust
    medium: 0.15 # Δ F1 0.05-0.15 = medium
    poor: 1.0 # Δ F1 > 0.15 = poor

# =============================================================================
# PHASE 3.5: FINE-TUNING
# =============================================================================
finetuning:
  # Data fractions to test
  fractions: [0.10, 0.25, 0.50, 1.00]

  # Methods per algorithm type
  methods:
    xgboost:
      strategy: continue_training
      params:
        xgb_model: pretrained # Continue from pretrained model
        n_estimators: 100 # Additional trees
    random_forest:
      strategy: warm_start
      params:
        warm_start: true
        n_estimators_additional: 100
    neural_networks:
      strategy: finetune
      params:
        freeze_layers: null # Fine-tune all layers
        learning_rate_factor: 0.1 # Reduce LR by 10x
        epochs: 50

  # Preprocessing
  preprocessing:
    transfer_scaler: berlin # Keep Berlin scaler for fine-tuning (model expects Berlin-scaled features)
    from_scratch_scaler: leipzig # Fit new scaler on Leipzig for from-scratch baseline

  # Baseline comparison
  from_scratch_baseline: true # Train Leipzig-only model for comparison

  # Statistical tests
  significance_test: mcnemar # McNemar test for paired comparison
  alpha: 0.05

# =============================================================================
# VISUALIZATION
# =============================================================================
visualization:
  style: seaborn-v0_8-whitegrid
  figsize: [12, 7]
  dpi: 300
  save_format: png

  # Color schemes
  colors:
    berlin: "#1f77b4"
    leipzig: "#ff7f0e"
    algorithms:
      random_forest: "#2ca02c"
      xgboost: "#d62728"
      cnn_1d: "#9467bd"
      tabnet: "#8c564b"
```

### 3.2 Config Loader Extension

Add to `src/urban_tree_transfer/config/loader.py`:

```python
def load_experiment_config() -> dict[str, Any]:
    """Load Phase 3 experiment configuration."""
    config_path = CONFIGS_DIR / "experiments" / "phase3_config.yaml"
    return _load_yaml(config_path)

def get_algorithm_config(algorithm_name: str) -> dict[str, Any]:
    """Get configuration for a specific algorithm."""
    config = load_experiment_config()
    return config["algorithms"][algorithm_name]

def get_coarse_grid(algorithm_name: str) -> dict[str, list]:
    """Get coarse hyperparameter grid for an algorithm."""
    algo_config = get_algorithm_config(algorithm_name)
    return algo_config["coarse_grid"]

def get_optuna_space(algorithm_name: str) -> dict[str, dict]:
    """Get Optuna search space for an algorithm."""
    algo_config = get_algorithm_config(algorithm_name)
    return algo_config["optuna_space"]
```

---

## 4. Implementation Tasks

### 4.1 Exploratory Notebook: exp_07_cross_city_baseline.ipynb

**Purpose:** Descriptive analysis of Berlin vs. Leipzig datasets (hypothesis generation)

**Status:** Optional, not in critical path

**Key Analyses:**

| Analysis                | Visualization                                   | Purpose                              |
| ----------------------- | ----------------------------------------------- | ------------------------------------ |
| Class Distribution      | Stacked bar: Genus × City                       | Identify class imbalance differences |
| Phenological Profiles   | Line plots: NDVI/EVI over months (top-5 genera) | Detect seasonal differences          |
| CHM Distributions       | Violin plots: CHM per genus × city              | Quantify structural differences      |
| Feature Distributions   | Ridge plots: Key features                       | Assess distribution overlap          |
| Statistical Differences | Heatmap: Cohen's d per genus × feature          | Quantify effect sizes                |
| Correlation Structure   | Correlation heatmaps (side-by-side)             | Assess structural similarity         |

**Outputs:**

- `outputs/phase_3/figures/exp_07_baseline/*.png`
- No JSON config (purely descriptive)

---

### 4.2 Exploratory Notebook: exp_08_chm_ablation.ipynb

**Purpose:** Determine CHM strategy through systematic ablation

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet`
- `data/phase_2_splits/berlin_val.parquet`

**Key Tasks:**

1. **Prepare Variants**
   - Create 5 feature sets (no_chm, zscore, percentile, both, raw)
   - Ensure identical S2 features across variants

2. **Cross-Validation Comparison**
   - Use Random Forest with default HP (stable baseline)
   - 3-Fold Spatial Block CV
   - Record: F1, Train-Val Gap, Feature Importance

3. **Per-Feature Importance & Gap Analysis**
   - For each variant with CHM: compute per-feature importance
   - For each variant: measure Train-Val Gap increase vs. no_chm baseline
   - Evaluate each CHM feature (raw, zscore, percentile) independently

4. **Per-Feature Decision**
   - Apply decision rules to each CHM feature independently
   - Determine which (if any) CHM features to include
   - Document reasoning per feature

**Decision Logic (per CHM feature):**

```python
# Pseudocode — each CHM feature evaluated independently
baseline_f1 = no_chm_val_f1
baseline_gap = no_chm_train_val_gap

included_features = []
for feature in ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"]:
    variant = get_variant_with(feature)

    # Criterion 1: Feature must not dominate model
    if feature_importance(feature) > 0.25:
        exclude(feature, reason="dominates model, overfitting risk")
        continue

    # Criterion 2: Must not destabilize generalization
    if variant.train_val_gap - baseline_gap > 0.05:
        exclude(feature, reason="increases Train-Val gap by >5pp")
        continue

    # Criterion 3: Must provide meaningful improvement
    if variant.val_f1 - baseline_f1 < 0.03:
        exclude(feature, reason="marginal improvement <3pp")
        continue

    included_features.append(feature)

decision = included_features if included_features else "no_chm"
```

**Note:** No Leipzig data is used in this decision. Transfer effects of CHM
are evaluated later in 03c (transfer evaluation). Using target-domain data
in setup decisions would constitute an information leak.

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (partial: chm_strategy)
- `outputs/phase_3/figures/exp_08_chm_ablation/*.png`

---

### 4.2b Exploratory Notebook: exp_08b_proximity_ablation.ipynb

**Purpose:** Determine whether proximity-filtered datasets improve classification

Phase 2c created two dataset variants:

- **Baseline:** All trees meeting genus frequency threshold
- **Filtered:** Trees with ≥20m distance to nearest tree of a different genus (reduces label noise from mixed crowns)

This ablation determines which variant to use for all subsequent experiments.

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet` (baseline)
- `data/phase_2_splits/berlin_train_filtered.parquet` (filtered)
- `data/phase_2_splits/berlin_val.parquet` / `berlin_val_filtered.parquet`
- `outputs/phase_3/metadata/setup_decisions.json` (chm_strategy from exp_08)

**Key Tasks:**

1. **Load Both Variants**
   - Apply CHM decision from exp_08 (consistent feature set)
   - Report sample counts: baseline N vs. filtered N

2. **Cross-Validation Comparison**
   - Use Random Forest with default HP (stable baseline)
   - 3-Fold Spatial Block CV on each variant
   - Record: F1, Per-Genus F1, Train-Val Gap

3. **Sample Loss Analysis**
   - Compute: `sample_loss_pct = 1 - (N_filtered / N_baseline)`
   - Per-genus breakdown: which genera lose the most samples?
   - Identify if any genus drops below minimum viable count

4. **Decision**
   - Apply decision rules from config
   - Document trade-off (F1 gain vs. sample loss)

**Decision Logic:**

```python
# Pseudocode
baseline_f1 = cv_f1(baseline_dataset)
filtered_f1 = cv_f1(filtered_dataset)
sample_loss = 1 - (len(filtered) / len(baseline))

if sample_loss > 0.20:
    decision = "baseline"
    reason = f"Filtered loses {sample_loss:.0%} of samples — too much"
elif filtered_f1 - baseline_f1 < 0.02:
    decision = "baseline"
    reason = f"Improvement only {filtered_f1 - baseline_f1:.3f} — marginal"
else:
    decision = "filtered"
    reason = f"Filtered improves F1 by {filtered_f1 - baseline_f1:.3f} with acceptable sample loss"
```

**Note:** Same information-leak principle as CHM — only Berlin data used.

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (+ proximity_strategy)
- `outputs/phase_3/figures/exp_08b_proximity/*.png`

---

### 4.2c Exploratory Notebook: exp_08c_outlier_ablation.ipynb

**Purpose:** Determine outlier removal strategy using Phase 2 severity flags

Phase 2 computed three independent outlier detection methods (Z-Score, Mahalanobis, IQR) and
combined them into summary columns:

- `outlier_severity`: none / low / medium / high (based on method agreement)
- `outlier_method_count`: 0-3 (number of methods flagging a tree as outlier)

These flags were deliberately kept as metadata (not acted upon) so that Phase 3 can make
an informed, data-driven removal decision.

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet` (or `_filtered`, based on exp_08b decision)
- `data/phase_2_splits/berlin_val.parquet` (or `_filtered`)
- `outputs/phase_3/metadata/setup_decisions.json` (chm_strategy, proximity_strategy)

**Key Tasks:**

1. **Outlier Distribution Analysis**
   - Count trees per severity level: none / low / medium / high
   - Per-genus breakdown: are some genera disproportionately flagged?
   - Visualize: outlier rate by genus (bar chart)

2. **Prepare Removal Variants**
   - `no_removal`: All trees (baseline)
   - `remove_high`: Drop trees with `outlier_severity == "high"` (flagged by ≥2 methods)
   - `remove_high_medium`: Drop trees with `outlier_severity in ["high", "medium"]`

3. **Cross-Validation Comparison**
   - Use Random Forest with default HP (stable baseline)
   - Apply CHM decision from exp_08 (consistent features)
   - 3-Fold Spatial Block CV on each variant
   - Record: F1, Per-Genus F1, Train-Val Gap

4. **Sample Loss vs. F1 Trade-off**
   - For each removal level: compute sample loss %
   - Plot: F1 vs. samples retained (trade-off curve)
   - Per-genus: check that no genus drops below minimum samples

5. **Decision**
   - Apply decision rules from config
   - Default to no removal if gains are marginal

**Decision Logic:**

```python
# Pseudocode
baseline_f1 = cv_f1(no_removal)
results = {}

for variant_name, remove_levels in [
    ("remove_high", ["high"]),
    ("remove_high_medium", ["high", "medium"]),
]:
    mask = ~train["outlier_severity"].isin(remove_levels)
    variant_f1 = cv_f1(train[mask])
    sample_loss = 1 - mask.sum() / len(train)
    results[variant_name] = {
        "f1": variant_f1,
        "sample_loss": sample_loss,
        "improvement": variant_f1 - baseline_f1
    }

# Select best variant that meets criteria
decision = "no_removal"  # Default
for name, r in sorted(results.items(), key=lambda x: x[1]["improvement"], reverse=True):
    if r["sample_loss"] > 0.15:
        continue  # Too much data loss
    if r["improvement"] < 0.02:
        continue  # Marginal gain
    decision = name
    break
```

**Note:** Same information-leak principle — only Berlin data used.

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (+ outlier_strategy)
- `outputs/phase_3/figures/exp_08c_outlier/*.png`

---

### 4.3 Exploratory Notebook: exp_09_feature_reduction.ipynb

**Purpose:** Determine optimal feature count through importance-based selection

**Inputs:**

- `data/phase_2_splits/berlin_train.parquet` (or `_filtered`, based on exp_08b)
- `data/phase_2_splits/berlin_val.parquet` (or `_filtered`)
- `outputs/phase_3/metadata/setup_decisions.json` (chm_strategy, proximity_strategy, outlier_strategy)

**Key Tasks:**

1. **Compute Feature Importance**
   - Train RF with all features (applying CHM, proximity, and outlier decisions from exp_08/08b/08c)
   - Extract gain-based importance
   - Rank features

2. **Create Feature Subsets**
   - Top-30, Top-50, Top-80, All features

3. **Evaluate Each Subset**
   - 3-Fold Spatial Block CV with RF
   - Record F1, training time

4. **Pareto Analysis**
   - Plot: F1 vs. Feature Count
   - Identify knee point

5. **Decision Logging**
   - Select smallest k with F1 ≥ F1_all - 0.01
   - Save selected feature list

**Outputs:**

- `outputs/phase_3/metadata/setup_decisions.json` (complete: chm_strategy, proximity_strategy, outlier_strategy, feature_set, selected_features)
- `outputs/phase_3/figures/exp_09_feature_reduction/*.png`

**Source Module Functions:**

```python
# src/urban_tree_transfer/experiments/ablation.py

def evaluate_dataset_variants(
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    target_col: str,
    cv: BaseCrossValidator,
) -> pd.DataFrame:
    """Evaluate multiple dataset variants with CV. Used by proximity and outlier ablations."""

def apply_outlier_removal(
    df: pd.DataFrame,
    remove_levels: list[str],
    severity_col: str = "outlier_severity",
) -> pd.DataFrame:
    """Remove rows matching given outlier severity levels."""

def compute_feature_importance(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    method: str = "rf_gain",
) -> pd.DataFrame:
    """Compute and rank feature importance."""

def create_feature_subsets(
    importance_df: pd.DataFrame,
    n_features_list: list[int],
) -> dict[str, list[str]]:
    """Create feature subsets based on importance ranking."""

def evaluate_feature_subsets(
    X: np.ndarray,
    y: np.ndarray,
    feature_subsets: dict[str, list[str]],
    cv: BaseCrossValidator,
) -> pd.DataFrame:
    """Evaluate each feature subset with CV."""

def select_optimal_features(
    results_df: pd.DataFrame,
    max_drop: float = 0.01,
) -> tuple[str, list[str]]:
    """Select optimal feature set based on decision rules."""
```

---

### 4.4 Exploratory Notebook: exp_10_algorithm_comparison.ipynb

**Purpose:** Compare all 4 algorithms with coarse HP to select champions

**Inputs:**

- Processed datasets from 03a_setup_fixation.ipynb
- `outputs/phase_3/metadata/setup_decisions.json`

**Key Tasks:**

1. **Prepare Data**
   - Load datasets with fixed feature set
   - Apply StandardScaler (fit on train)
   - Encode labels

2. **Coarse Grid Search (per algorithm)**
   - Random Forest: 24 configs
   - XGBoost: 48 configs
   - 1D-CNN: baseline only
   - TabNet: baseline only
   - Use 3-Fold Spatial Block CV for speed

3. **Collect Metrics**
   - For each algorithm: Val F1, Train F1, Gap, Fit Time
   - Per-genus F1 for error analysis

4. **Champion Selection**
   - Apply filters (min F1 ≥ 0.50, gap < 35%)
   - Select best ML (RF or XGBoost)
   - Select best NN (1D-CNN or TabNet)

5. **Visualization**
   - Algorithm comparison bar chart
   - Confusion matrices for champions

**Outputs:**

- `outputs/phase_3/metadata/algorithm_comparison.json`
- `outputs/phase_3/figures/exp_10_algorithm_comparison/*.png`

---

### 4.5 Runner Notebook: 03a_setup_fixation.ipynb

**Purpose:** Apply setup decisions and prepare datasets for experiments

**Inputs:**

- `data/phase_2_splits/berlin_*.parquet`
- `data/phase_2_splits/leipzig_*.parquet`
- `outputs/phase_3/metadata/setup_decisions.json`

**Processing Steps:**

1. **Validate Setup Decisions**
   - Load and validate against schema
   - Log CHM strategy, proximity strategy, outlier strategy, and feature count

2. **Apply Dataset Selection**
   - Choose baseline or filtered datasets based on proximity_strategy
   - Apply outlier removal based on outlier_strategy

3. **Apply Feature Selection**
   - Load selected features from setup_decisions.json
   - Apply CHM feature selection
   - Filter columns in all datasets
   - Validate schema consistency

4. **Save Processed Datasets**
   - Save to `data/phase_3_experiments/`
   - Maintain train/val/test/finetune splits

5. **Generate Summary**
   - Feature count, sample counts per city/split
   - Save execution log

**Outputs:**

- `data/phase_3_experiments/berlin_train.parquet`
- `data/phase_3_experiments/berlin_val.parquet`
- `data/phase_3_experiments/berlin_test.parquet`
- `data/phase_3_experiments/leipzig_finetune.parquet`
- `data/phase_3_experiments/leipzig_test.parquet`
- `outputs/phase_3/logs/03a_setup_fixation.json`

---

### 4.6 Runner Notebook: 03b_berlin_optimization.ipynb

**Purpose:** HP-tune champions and establish Berlin upper bound

**Inputs:**

- `data/phase_3_experiments/berlin_*.parquet`
- `outputs/phase_3/metadata/algorithm_comparison.json`

**Processing Steps:**

1. **Load Champions**
   - Get ML and NN champion from algorithm_comparison.json
   - Load corresponding search spaces from config

2. **HP-Tuning with Optuna (ML Champion)**
   - Create Optuna study
   - Run 50+ trials with 3-Fold CV
   - Use pruning for efficiency
   - Track: F1, Gap, Trial parameters

3. **HP-Tuning with Optuna (NN Champion)**
   - Same procedure for NN
   - Include early stopping in objective

4. **Final Training**
   - Train both champions on Train + Val with best HP
   - Save trained models

5. **Berlin Test Evaluation**
   - Evaluate on hold-out Berlin Test
   - Compute all metrics with confidence intervals
   - Per-genus analysis
   - Feature importance (gain + permutation for ML)

6. **Post-Training Error Analysis (Extensive)**
   This step extracts maximum insight from the trained champion models using all
   available metadata. All visualizations use **German genus names**.

   a. **Confusion Matrix & Worst Pairs**
   - Normalized confusion matrix with German labels
   - Extract top-10 most confused genus pairs (off-diagonal entries)
   - Publication-quality per-genus metrics table (Precision, Recall, F1, Support)

   b. **Conifer vs. Deciduous Analysis**
   - Aggregate F1: Nadelbäume vs. Laubbäume
   - Hypothesis: Nadelbaum-Gattungen haben distinkteres Spektralprofil → höherer F1

   c. **Straßen- vs. Anlagenbäume** (Berlin only, using `tree_type`)
   - Per-genus F1 split by location type
   - Hypothesis: Straßenbäume haben homogeneres Umfeld → einfacher zu klassifizieren?
   - Or: Anlagenbäume stehen freier → klareres Spektralsignal?

   d. **Plant Year Impact** (using `plant_year` from dataset)
   - Bin plant_year into decades (pre-1960, 1960-79, 1980-99, 2000-19, 2020+)
   - Prediction accuracy per decade
   - Hypothesis: Jüngere Bäume (kleiner Kronendeckung) → schwerer zu klassifizieren

   e. **Species Breakdown for Problematic Genera**
   - For genera with F1 < 0.50: break down by `species_latin`
   - Are specific species causing the misclassifications?
   - Example: If Acer has low F1, is it Acer platanoides vs. Acer pseudoplatanus?

   f. **CHM Value vs. Accuracy**
   - Bin CHM_1m values; compute accuracy per bin
   - Hypothesis: Extreme CHM values (very small/tall) are harder

   g. **Spatial Error Map**
   - Join predictions with `geometry_lookup.parquet`
   - Compute per-block accuracy (using block_id)
   - Map: which parts of Berlin are hardest to classify?

   h. **Misclassification Flow**
   - Sankey/alluvial diagram: true genus → predicted genus (errors only)
   - Shows systematic error patterns for the Präsentation

7. **Visualization**
   - HP tuning progress (optimization history)
   - All figures from post-training analysis (see Section 6.1)
   - Total: ~15 figures for Berlin optimization

**Outputs:**

- `outputs/phase_3/metadata/hp_tuning_ml.json`
- `outputs/phase_3/metadata/hp_tuning_nn.json`
- `outputs/phase_3/metadata/berlin_evaluation.json`
- `outputs/phase_3/models/berlin_ml_champion.pkl`
- `outputs/phase_3/models/berlin_nn_champion.pt`
- `outputs/phase_3/figures/berlin_optimization/*.png`
- `outputs/phase_3/logs/03b_berlin_optimization.json`

**Source Module Functions:**

```python
# src/urban_tree_transfer/experiments/hp_tuning.py

def create_optuna_objective(
    model_class: type,
    search_space: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv: BaseCrossValidator,
    metric: str = "f1_weighted",
) -> Callable:
    """Create Optuna objective function for HP tuning."""

def run_optuna_study(
    objective: Callable,
    n_trials: int = 50,
    timeout_hours: float = 2.0,
    study_name: str = "hp_tuning",
) -> optuna.Study:
    """Run Optuna study with pruning."""

def extract_best_params(study: optuna.Study) -> dict[str, Any]:
    """Extract best parameters from completed study."""
```

```python
# src/urban_tree_transfer/experiments/evaluation.py

def evaluate_model(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    compute_ci: bool = True,
    n_bootstrap: int = 1000,
) -> dict[str, Any]:
    """Comprehensive model evaluation with confidence intervals."""

def compute_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence intervals for a metric."""
```

---

### 4.7 Runner Notebook: 03c_transfer_evaluation.ipynb

**Purpose:** Evaluate zero-shot transfer to Leipzig

**Inputs:**

- `data/phase_3_experiments/leipzig_test.parquet`
- `outputs/phase_3/models/berlin_ml_champion.pkl`
- `outputs/phase_3/models/berlin_nn_champion.pt`
- `outputs/phase_3/metadata/berlin_evaluation.json`

**Processing Steps:**

1. **Load Models and Data**
   - Load both trained Berlin models
   - Load Leipzig test set
   - Apply same preprocessing (scaler from training)

2. **Zero-Shot Evaluation**
   - Predict on Leipzig test with both models
   - Compute all metrics

3. **Transfer Gap Analysis**
   - Calculate absolute drop: F1_Berlin - F1_Leipzig
   - Calculate relative drop: (F1_Berlin - F1_Leipzig) / F1_Berlin \* 100
   - Compare ML vs. NN transfer performance

4. **Per-Genus Transfer Analysis**
   - F1 comparison per genus (Berlin vs. Leipzig)
   - Classify robustness: robust (<5%), medium (5-15%), poor (>15%)
   - Identify best/worst transferring genera

5. **Confusion Matrix Comparison**
   - Side-by-side: Berlin vs. Leipzig confusion matrices (German labels)
   - Highlight systematic differences
   - Most confused genus pairs in Leipzig vs. Berlin

6. **Extended Transfer Analysis**
   - Nadel- vs. Laubbäume: aggregate transfer performance per group
   - Species-level detail for genera with poor transfer (F1 drop >15%)
   - Identify: Are the same genera problematic in both cities?

7. **Select Best Transfer Model**
   - Compare ML and NN transfer performance
   - Select model for fine-tuning experiments

**Outputs:**

- `outputs/phase_3/metadata/transfer_evaluation.json`
- `outputs/phase_3/figures/transfer/*.png`
- `outputs/phase_3/logs/03c_transfer_evaluation.json`

**Source Module Functions:**

```python
# src/urban_tree_transfer/experiments/transfer.py

def compute_transfer_metrics(
    source_metrics: dict[str, Any],
    target_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Compute transfer-specific metrics."""

def compute_per_genus_transfer(
    source_per_genus: dict[str, float],
    target_per_genus: dict[str, float],
    thresholds: dict[str, float],
) -> dict[str, dict[str, Any]]:
    """Compute per-genus transfer analysis with robustness classification."""

def create_transfer_summary(
    ml_transfer: dict[str, Any],
    nn_transfer: dict[str, Any],
) -> dict[str, Any]:
    """Create summary comparing ML and NN transfer performance."""
```

---

### 4.8 Runner Notebook: 03d_finetuning.ipynb

**Purpose:** Measure fine-tuning sample efficiency

**Inputs:**

- `data/phase_3_experiments/leipzig_finetune.parquet`
- `data/phase_3_experiments/leipzig_test.parquet`
- `outputs/phase_3/models/berlin_ml_champion.pkl`
- `outputs/phase_3/models/berlin_nn_champion.pt`
- `outputs/phase_3/metadata/transfer_evaluation.json` (for context)

**Processing Steps:**

1. **Load Both Champions**
   - Load ML champion and NN champion from 03b
   - Both are fine-tuned to enable ML vs. NN comparison
   - Load Berlin scaler (keep for fine-tuning preprocessing)

2. **Create Fine-Tuning Subsets**
   - 10%, 25%, 50%, 100% of Leipzig finetune data
   - Stratified sampling to maintain class proportions
   - Same subsets used for both models (fair comparison)
   - Apply Berlin scaler to all subsets (model expects Berlin-scaled features)

3. **Fine-Tune Both Models at Each Level**
   - For each fraction × each model:
     - ML: Continue training / warm start
     - NN: Full fine-tune with 0.1× LR
     - Evaluate on Leipzig test
     - Record metrics

4. **Leipzig From-Scratch Baselines**
   - Train ML from scratch on 100% Leipzig finetune
   - Train NN from scratch on 100% Leipzig finetune
   - Fit new scaler on Leipzig finetune data (independent from Berlin)
   - Compare with transfer + 100% fine-tuning

5. **Statistical Significance**
   - McNemar test: Compare predictions at different levels
   - Identify when fine-tuning significantly improves over zero-shot

6. **Sample Efficiency Curves**
   - Plot: F1 vs. Fine-tuning fraction (ML and NN on same plot)
   - Mark zero-shot baselines and from-scratch baselines
   - Calculate: fraction needed to reach 90% of from-scratch performance
   - Compare: Does ML or NN benefit more from local data?

7. **Per-Genus Recovery Analysis**
   - Heatmap: per-genus F1 at each fine-tuning fraction
   - Which genera recover fastest with local data?
   - Do previously poor-transferring genera benefit most from fine-tuning?

8. **ML vs. NN Comparison**
   - Side-by-side curves: sample efficiency for ML vs. NN
   - At which fraction does fine-tuning surpass from-scratch?
   - McNemar significance heatmap across all comparison pairs

**Outputs:**

- `outputs/phase_3/metadata/finetuning_curve.json`
- `outputs/phase_3/figures/finetuning/*.png`
- `outputs/phase_3/models/finetuned/*.pkl` or `.pt`
- `outputs/phase_3/logs/03d_finetuning.json`

**Source Module Functions:**

```python
# src/urban_tree_transfer/experiments/training.py

def finetune_xgboost(
    pretrained_model,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    n_additional_estimators: int = 100,
) -> XGBClassifier:
    """Fine-tune XGBoost model with additional trees."""

def finetune_neural_network(
    pretrained_model,
    X_finetune: np.ndarray,
    y_finetune: np.ndarray,
    epochs: int = 50,
    lr_factor: float = 0.1,
) -> nn.Module:
    """Fine-tune neural network with reduced learning rate."""

def create_stratified_subsets(
    X: np.ndarray,
    y: np.ndarray,
    fractions: list[float],
    random_seed: int = 42,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Create stratified subsets at different fractions."""
```

---

## 5. JSON Schemas

### 5.1 Setup Decisions Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Setup Decisions",
  "type": "object",
  "required": [
    "timestamp",
    "chm_strategy",
    "proximity_strategy",
    "outlier_strategy",
    "feature_set",
    "selected_features"
  ],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "chm_strategy": {
      "type": "object",
      "properties": {
        "decision": {
          "type": "string",
          "enum": [
            "no_chm",
            "zscore_only",
            "percentile_only",
            "both_engineered",
            "raw_chm"
          ]
        },
        "reasoning": { "type": "string" },
        "ablation_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "variant": { "type": "string" },
              "val_f1_mean": { "type": "number" },
              "val_f1_std": { "type": "number" },
              "chm_importance": { "type": ["number", "null"] },
              "train_val_gap": { "type": "number" }
            }
          }
        }
      }
    },
    "proximity_strategy": {
      "type": "object",
      "properties": {
        "decision": {
          "type": "string",
          "enum": ["baseline", "filtered"]
        },
        "reasoning": { "type": "string" },
        "sample_counts": {
          "type": "object",
          "properties": {
            "baseline_n": { "type": "integer" },
            "filtered_n": { "type": "integer" },
            "sample_loss_pct": { "type": "number" }
          }
        },
        "ablation_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "variant": { "type": "string" },
              "val_f1_mean": { "type": "number" },
              "val_f1_std": { "type": "number" },
              "n_samples": { "type": "integer" }
            }
          }
        }
      }
    },
    "outlier_strategy": {
      "type": "object",
      "properties": {
        "decision": {
          "type": "string",
          "enum": ["no_removal", "remove_high", "remove_high_medium"]
        },
        "reasoning": { "type": "string" },
        "sample_counts": {
          "type": "object",
          "properties": {
            "total_n": { "type": "integer" },
            "removed_n": { "type": "integer" },
            "sample_loss_pct": { "type": "number" }
          }
        },
        "ablation_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "variant": { "type": "string" },
              "val_f1_mean": { "type": "number" },
              "val_f1_std": { "type": "number" },
              "n_samples": { "type": "integer" },
              "sample_loss_pct": { "type": "number" }
            }
          }
        }
      }
    },
    "feature_set": {
      "type": "object",
      "properties": {
        "decision": { "type": "string" },
        "n_features": { "type": "integer" },
        "reasoning": { "type": "string" },
        "pareto_results": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "variant": { "type": "string" },
              "n_features": { "type": "integer" },
              "val_f1_mean": { "type": "number" },
              "val_f1_std": { "type": "number" }
            }
          }
        }
      }
    },
    "selected_features": {
      "type": "array",
      "items": { "type": "string" }
    }
  }
}
```

### 5.2 Algorithm Comparison Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Algorithm Comparison",
  "type": "object",
  "required": ["timestamp", "algorithms", "ml_champion", "nn_champion"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "setup_reference": { "type": "string" },
    "algorithms": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "type": { "type": "string", "enum": ["ml", "nn"] },
          "best_params": { "type": "object" },
          "val_f1_mean": { "type": "number" },
          "val_f1_std": { "type": "number" },
          "train_f1_mean": { "type": "number" },
          "train_val_gap": { "type": "number" },
          "fit_time_seconds": { "type": "number" },
          "passed_filters": { "type": "boolean" }
        }
      }
    },
    "ml_champion": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "reasoning": { "type": "string" }
      }
    },
    "nn_champion": {
      "type": "object",
      "properties": {
        "name": { "type": "string" },
        "reasoning": { "type": "string" }
      }
    }
  }
}
```

### 5.3 Evaluation Metrics Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Evaluation Metrics",
  "type": "object",
  "required": ["timestamp", "model", "dataset", "metrics"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "model": { "type": "string" },
    "model_path": { "type": "string" },
    "dataset": { "type": "string" },
    "n_samples": { "type": "integer" },
    "metrics": {
      "type": "object",
      "properties": {
        "weighted_f1": { "type": "number" },
        "weighted_f1_ci_lower": { "type": "number" },
        "weighted_f1_ci_upper": { "type": "number" },
        "macro_f1": { "type": "number" },
        "accuracy": { "type": "number" },
        "per_genus_f1": {
          "type": "object",
          "additionalProperties": { "type": "number" }
        }
      }
    },
    "confusion_matrix": {
      "type": "array",
      "items": {
        "type": "array",
        "items": { "type": "integer" }
      }
    },
    "feature_importance": {
      "type": "object",
      "properties": {
        "method": { "type": "string" },
        "top_features": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "feature": { "type": "string" },
              "importance": { "type": "number" }
            }
          }
        }
      }
    }
  }
}
```

### 5.4 Transfer Evaluation Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Transfer Evaluation",
  "type": "object",
  "required": ["timestamp", "source_city", "target_city", "models"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "source_city": { "type": "string" },
    "target_city": { "type": "string" },
    "models": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "properties": {
          "source_f1": { "type": "number" },
          "target_f1": { "type": "number" },
          "absolute_drop": { "type": "number" },
          "relative_drop_pct": { "type": "number" },
          "per_genus_transfer": {
            "type": "object",
            "additionalProperties": {
              "type": "object",
              "properties": {
                "source_f1": { "type": "number" },
                "target_f1": { "type": "number" },
                "drop": { "type": "number" },
                "robustness": {
                  "type": "string",
                  "enum": ["robust", "medium", "poor"]
                }
              }
            }
          }
        }
      }
    },
    "best_transfer_model": { "type": "string" },
    "selection_reasoning": { "type": "string" }
  }
}
```

### 5.5 Fine-Tuning Curve Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Fine-Tuning Curve",
  "type": "object",
  "required": ["timestamp", "model", "results"],
  "properties": {
    "timestamp": { "type": "string", "format": "date-time" },
    "model": { "type": "string" },
    "finetuning_method": { "type": "string" },
    "results": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "fraction": { "type": "number" },
          "n_samples": { "type": "integer" },
          "target_f1": { "type": "number" },
          "target_f1_ci_lower": { "type": "number" },
          "target_f1_ci_upper": { "type": "number" },
          "improvement_over_zeroshot": { "type": "number" },
          "pct_of_from_scratch": { "type": "number" }
        }
      }
    },
    "baselines": {
      "type": "object",
      "properties": {
        "zero_shot_f1": { "type": "number" },
        "from_scratch_f1": { "type": "number" }
      }
    },
    "efficiency_metrics": {
      "type": "object",
      "properties": {
        "fraction_to_match_scratch": { "type": ["number", "null"] },
        "fraction_to_90pct_scratch": { "type": "number" }
      }
    },
    "significance_tests": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "comparison": { "type": "string" },
          "test": { "type": "string" },
          "p_value": { "type": "number" },
          "significant": { "type": "boolean" }
        }
      }
    }
  }
}
```

---

## 6. Visualization Standards

### 6.0 General Conventions

**German Genus Names:** All genus labels in visualizations MUST use `genus_german` from
the dataset (e.g., "Linde" instead of "Tilia"). Latin names may appear in parentheses
where space permits: "Linde (Tilia)". This ensures the Präsentation is accessible for
a German-speaking audience.

**Conifer / Deciduous Grouping:** For aggregate analyses, genera are grouped as:

| Gruppe    | Gattungen (deutsch)                                              | Gattungen (lateinisch)                                              |
| --------- | ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| Nadelbaum | Kiefer, Fichte                                                   | Pinus, Picea                                                        |
| Laubbaum  | Linde, Ahorn, Eiche, Platane, Kastanie, Birke, Esche, Robinie, … | Tilia, Acer, Quercus, Platanus, Aesculus, Betula, Fraxinus, Robinia |

This grouping is defined in the experiment config and used by the visualization module.
Currently this requires a runtime lookup; a dedicated `is_conifer` column in the Parquet
datasets is proposed as Phase 2 Improvement 7 (see PRD 002d).

**Metadata for Post-Hoc Analysis:** The Parquet datasets contain all metadata columns
needed for post-hoc analysis (`plant_year`, `height_m`, `tree_type`, `species_latin`,
`genus_german`, etc.) directly — no join required.

### 6.1 Required Figures

**Setup Ablations (exp_08, exp_08b, exp_08c, exp_09):**

| Experiment | Figure                           | Filename                            |
| ---------- | -------------------------------- | ----------------------------------- |
| exp_08     | CHM ablation F1 comparison       | `chm_ablation_results.png`          |
| exp_08     | CHM feature importance           | `chm_feature_importance.png`        |
| exp_08     | CHM Train-Val gap comparison     | `chm_train_val_gap.png`             |
| exp_08b    | Proximity F1 comparison          | `proximity_f1_comparison.png`       |
| exp_08b    | Per-genus F1 (baseline/filtered) | `proximity_per_genus_f1.png`        |
| exp_08b    | Per-genus sample loss breakdown  | `proximity_sample_loss.png`         |
| exp_08c    | Outlier severity by genus        | `outlier_distribution_by_genus.png` |
| exp_08c    | F1 vs. samples retained curve    | `outlier_tradeoff_curve.png`        |
| exp_08c    | Per-genus F1 across variants     | `outlier_per_genus_f1.png`          |
| exp_09     | Feature importance ranking       | `feature_importance_ranking.png`    |
| exp_09     | Pareto curve (F1 vs. count)      | `pareto_curve.png`                  |
| exp_09     | Feature group contribution       | `feature_group_contribution.png`    |

**Cross-City Baseline (exp_07, optional):**

| Experiment | Figure                        | Filename                            |
| ---------- | ----------------------------- | ----------------------------------- |
| exp_07     | Class distribution comparison | `genus_distribution_comparison.png` |
| exp_07     | Phenological profiles (top-5) | `phenological_profiles_top5.png`    |
| exp_07     | CHM violin per genus          | `chm_violin_per_genus.png`          |
| exp_07     | Cohen's d heatmap             | `cohens_d_heatmap.png`              |

**Algorithm Comparison (exp_10):**

| Experiment | Figure                      | Filename                          |
| ---------- | --------------------------- | --------------------------------- |
| exp_10     | Algorithm F1 comparison     | `algorithm_comparison.png`        |
| exp_10     | Champion confusion matrices | `confusion_matrix_{champion}.png` |
| exp_10     | Algorithm Train-Val gap     | `algorithm_train_val_gap.png`     |

**Berlin Optimization (03b) — Training & Tuning:**

| Experiment | Figure                      | Filename                          |
| ---------- | --------------------------- | --------------------------------- |
| 03b        | Optuna optimization history | `optuna_optimization_history.png` |
| 03b        | Feature importance top-20   | `feature_importance_top20.png`    |

**Berlin Optimization (03b) — Post-Training Error Analysis (deutsche Namen):**

| Experiment | Figure                            | Filename                            |
| ---------- | --------------------------------- | ----------------------------------- |
| 03b        | Confusion matrix (deutsche Namen) | `berlin_confusion_matrix.png`       |
| 03b        | Per-genus F1 bar chart            | `per_genus_f1_berlin.png`           |
| 03b        | Per-genus metrics table           | `per_genus_metrics_table.png`       |
| 03b        | Top-10 confused genus pairs       | `confusion_pairs_worst.png`         |
| 03b        | Nadel- vs. Laubbäume F1           | `conifer_deciduous_comparison.png`  |
| 03b        | Straßen- vs. Anlagenbäume F1      | `tree_type_comparison.png`          |
| 03b        | Accuracy by plant year decade     | `plant_year_impact.png`             |
| 03b        | Species breakdown (problematic)   | `species_breakdown_problematic.png` |
| 03b        | Spatial error map (by block)      | `spatial_error_map.png`             |
| 03b        | CHM value vs. prediction accuracy | `chm_impact_on_accuracy.png`        |
| 03b        | Misclassification flow            | `misclassification_sankey.png`      |

**Transfer Evaluation (03c):**

| Experiment | Figure                              | Filename                                  |
| ---------- | ----------------------------------- | ----------------------------------------- |
| 03c        | Transfer F1 comparison              | `transfer_comparison.png`                 |
| 03c        | Confusion comparison (side-by-side) | `confusion_comparison_berlin_leipzig.png` |
| 03c        | Per-genus transfer robustness       | `per_genus_transfer_robustness.png`       |
| 03c        | Nadel/Laub transfer comparison      | `transfer_conifer_deciduous.png`          |
| 03c        | Most confused pairs in Leipzig      | `transfer_confusion_pairs.png`            |
| 03c        | Species detail for poor genera      | `transfer_species_analysis.png`           |

**Fine-Tuning (03d):**

| Experiment | Figure                      | Filename                             |
| ---------- | --------------------------- | ------------------------------------ |
| 03d        | Fine-tuning curve (ML + NN) | `finetuning_curve.png`               |
| 03d        | Comparison with baselines   | `finetuning_vs_baselines.png`        |
| 03d        | Per-genus F1 recovery       | `finetuning_per_genus_recovery.png`  |
| 03d        | ML vs. NN comparison        | `finetuning_ml_vs_nn_comparison.png` |
| 03d        | McNemar significance matrix | `finetuning_significance_matrix.png` |

### 6.2 Visualization Module

```python
# src/urban_tree_transfer/experiments/visualization.py

# --- Setup & Utilities ---

def setup_plot_style() -> None:
    """Set up publication-ready plot style (consistent font, DPI, colors)."""

CONIFER_GENERA = {"PINUS", "PICEA"}  # Extend if dataset contains more

def get_german_labels(genus_latin_list: list[str], lookup: pd.DataFrame) -> list[str]:
    """Map Latin genus names to German display labels using genus_german column."""

# --- Ablation Plots (exp_08, 08b, 08c, 09) ---

def plot_ablation_comparison(
    results: pd.DataFrame,  # variant, val_f1_mean, val_f1_std, ...
    title: str,
    output_path: Path,
    highlight_decision: str | None = None,
) -> None:
    """Generic bar chart for ablation study F1 comparison."""

def plot_train_val_gap(
    results: pd.DataFrame,  # variant, train_f1, val_f1, gap
    title: str,
    output_path: Path,
) -> None:
    """Bar chart showing Train-Val gap across ablation variants."""

def plot_per_genus_comparison(
    results_a: dict[str, float],
    results_b: dict[str, float],
    label_a: str,
    label_b: str,
    genus_labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Grouped bar chart comparing per-genus F1 between two variants."""

def plot_sample_loss_by_genus(
    baseline_counts: pd.Series,
    filtered_counts: pd.Series,
    genus_labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Stacked bar: samples retained vs. removed per genus."""

def plot_outlier_distribution(
    severity_by_genus: pd.DataFrame,  # genus × severity counts
    genus_labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Stacked bar: outlier severity distribution by genus."""

def plot_tradeoff_curve(
    points: list[dict],  # name, f1, samples_retained
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Scatter/line plot: F1 vs. data retained (for outlier/proximity decisions)."""

def plot_pareto_curve(
    results: pd.DataFrame,  # n_features, val_f1_mean, val_f1_std
    selected_k: int,
    output_path: Path,
) -> None:
    """F1 vs. feature count with knee point and selection marker."""

def plot_feature_group_contribution(
    importance_df: pd.DataFrame,
    group_mapping: dict[str, str],  # feature → group (S2, CHM, etc.)
    output_path: Path,
) -> None:
    """Pie/bar chart showing feature importance by group (S2 temporal vs. CHM)."""

# --- Algorithm Comparison (exp_10) ---

def plot_algorithm_comparison(
    results: pd.DataFrame,
    output_path: Path,
) -> None:
    """Algorithm F1 comparison bar chart with error bars."""

# --- Core Evaluation Plots (03b, 03c, 03d) ---

def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],  # German names
    title: str,
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Confusion matrix heatmap with German genus labels."""

def plot_confusion_comparison(
    cm_source: np.ndarray,
    cm_target: np.ndarray,
    labels: list[str],  # German names
    source_name: str,
    target_name: str,
    output_path: Path,
) -> None:
    """Side-by-side confusion matrix comparison (Berlin vs. Leipzig)."""

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    output_path: Path | None = None,
) -> None:
    """Horizontal bar chart of top-N feature importances."""

def plot_per_genus_f1(
    per_genus: dict[str, float],
    genus_labels: list[str],  # German names
    title: str,
    output_path: Path,
) -> None:
    """Horizontal bar chart: per-genus F1 sorted by performance."""

def plot_per_genus_metrics_table(
    metrics: pd.DataFrame,  # genus, precision, recall, f1, support
    genus_labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Publication-quality metrics table rendered as figure."""

# --- Post-Training Error Analysis (03b) ---

def plot_confusion_pairs(
    cm: np.ndarray,
    labels: list[str],  # German names
    top_n: int = 10,
    output_path: Path | None = None,
) -> None:
    """Bar chart: top-N most confused genus pairs from confusion matrix."""

def plot_conifer_deciduous(
    per_genus_f1: dict[str, float],
    genus_labels: list[str],  # German names
    conifer_genera: set[str],
    output_path: Path,
) -> None:
    """Grouped bar: aggregate F1 for Nadel- vs. Laubbäume."""

def plot_tree_type_comparison(
    tree_type_metrics: pd.DataFrame,  # tree_type, genus, f1
    output_path: Path,
) -> None:
    """Grouped bar: Straßenbäume vs. Anlagenbäume F1 per genus (Berlin only)."""

def plot_plant_year_impact(
    pred_df: pd.DataFrame,  # plant_year, correct (bool)
    decade_bins: list[int],
    output_path: Path,
) -> None:
    """Line/bar chart: prediction accuracy by plant year decade."""

def plot_species_breakdown(
    species_metrics: pd.DataFrame,  # genus, species, f1, support
    problematic_genera: list[str],
    output_path: Path,
) -> None:
    """For problematic genera: per-species F1 to identify if specific species cause errors."""

def plot_spatial_error_map(
    block_metrics: pd.DataFrame,  # block_id, x, y, f1
    city_boundary: Any,  # GeoDataFrame
    output_path: Path,
) -> None:
    """Map: spatial distribution of prediction accuracy by block."""

def plot_chm_impact(
    pred_df: pd.DataFrame,  # CHM_1m, correct (bool)
    output_path: Path,
) -> None:
    """Binned accuracy plot: prediction accuracy as a function of CHM value."""

def plot_misclassification_flow(
    cm: np.ndarray,
    labels: list[str],  # German names
    min_count: int = 10,
    output_path: Path | None = None,
) -> None:
    """Sankey/alluvial diagram: true → predicted for incorrect predictions only."""

# --- Transfer Plots (03c) ---

def plot_per_genus_transfer(
    transfer_data: dict[str, dict],
    genus_labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Per-genus transfer robustness visualization (robust/medium/poor)."""

def plot_transfer_conifer_deciduous(
    transfer_data: dict[str, dict],
    conifer_genera: set[str],
    output_path: Path,
) -> None:
    """Aggregate transfer performance: Nadel- vs. Laubbäume."""

# --- Fine-Tuning Plots (03d) ---

def plot_finetuning_curve(
    results: list[dict],
    baselines: dict[str, float],
    output_path: Path,
) -> None:
    """Sample efficiency curve with zero-shot + from-scratch baselines."""

def plot_finetuning_ml_vs_nn(
    ml_results: list[dict],
    nn_results: list[dict],
    output_path: Path,
) -> None:
    """Side-by-side ML vs. NN fine-tuning curves."""

def plot_finetuning_per_genus_recovery(
    per_genus_by_fraction: dict[float, dict[str, float]],
    genus_labels: list[str],  # German names
    output_path: Path,
) -> None:
    """Heatmap: per-genus F1 at each fine-tuning fraction."""

def plot_significance_matrix(
    p_values: pd.DataFrame,  # comparison pairs × p-values
    output_path: Path,
) -> None:
    """Heatmap of McNemar test p-values across comparisons."""
```

---

## 7. Testing Strategy

### 7.1 Unit Tests

```
tests/experiments/
├── test_data_loading.py
├── test_preprocessing.py
├── test_models.py
├── test_training.py
├── test_evaluation.py
├── test_transfer.py
├── test_ablation.py
├── test_hp_tuning.py
└── test_visualization.py
```

**Key Test Cases:**

| Module        | Test                      | Purpose                                 |
| ------------- | ------------------------- | --------------------------------------- |
| evaluation.py | test_compute_metrics      | Verify metrics match sklearn            |
| evaluation.py | test_confidence_intervals | Verify CI coverage                      |
| transfer.py   | test_transfer_metrics     | Verify gap calculations                 |
| ablation.py   | test_decision_logic       | Verify decision rules applied correctly |
| hp_tuning.py  | test_optuna_objective     | Verify objective returns valid scores   |

### 7.2 Integration Tests

```
tests/integration/
└── test_phase3_pipeline.py
```

**Test Cases:**

- End-to-end: Setup → Algorithm → Berlin → Transfer → Finetune (on small subset)
- Schema validation for all JSON outputs
- Figure generation without errors

---

## 8. Execution Order

### 8.1 Critical Path

```
1. exp_08_chm_ablation.ipynb
   └── Output: setup_decisions.json (chm_strategy)

2. exp_08b_proximity_ablation.ipynb
   └── Output: setup_decisions.json (+ proximity_strategy)

3. exp_08c_outlier_ablation.ipynb
   └── Output: setup_decisions.json (+ outlier_strategy)

4. exp_09_feature_reduction.ipynb
   └── Output: setup_decisions.json (complete: + feature_set, selected_features)

5. 03a_setup_fixation.ipynb (Runner)
   └── Output: Processed datasets

6. exp_10_algorithm_comparison.ipynb
   └── Output: algorithm_comparison.json

7. 03b_berlin_optimization.ipynb (Runner)
   └── Output: hp_tuning_*.json, berlin_evaluation.json, models

8. 03c_transfer_evaluation.ipynb (Runner)
   └── Output: transfer_evaluation.json

9. 03d_finetuning.ipynb (Runner)
   └── Output: finetuning_curve.json
```

### 8.2 Optional/Parallel

```
exp_07_cross_city_baseline.ipynb
└── Can run anytime (no dependencies, no outputs used by others)
```

---

## 9. Colab Workflow

### 9.1 Drive Structure

```
Google Drive/
└── dev/urban-tree-transfer/
    ├── data/
    │   ├── phase_2_splits/                    # Input from Phase 2c
    │   │   ├── berlin_train.parquet            # ML-optimized (no geometry)
    │   │   ├── berlin_val.parquet
    │   │   ├── berlin_test.parquet
    │   │   ├── leipzig_finetune.parquet
    │   │   ├── leipzig_test.parquet
    │   │   ├── *_filtered.parquet              # Filtered variants
    │   │   ├── geometry_lookup.parquet         # tree_id → x/y for visualization
    │   │   └── *.gpkg                          # Authoritative GeoPackages (kept)
    │   └── phase_3_experiments/                # Processed for experiments
    │       ├── berlin_train.parquet
    │       ├── berlin_val.parquet
    │       ├── berlin_test.parquet
    │       ├── leipzig_finetune.parquet
    │       └── leipzig_test.parquet
    ├── outputs/
    │   └── phase_3/
    │       ├── metadata/                  # JSON configs
    │       ├── models/                    # Trained models
    │       ├── figures/                   # Visualizations
    │       └── logs/                      # Execution logs
    └── repo/                              # Cloned repository
```

### 9.2 Notebook Header Template

```python
# ============================================================================
# SETUP (Run once per session)
# ============================================================================

# Install package from GitHub
!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Define paths
DRIVE_BASE = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
DATA_DIR = DRIVE_BASE / "data"
OUTPUT_DIR = DRIVE_BASE / "outputs/phase_3"

# Create output directories
(OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "logs").mkdir(parents=True, exist_ok=True)

# Import project modules
from urban_tree_transfer.config import load_experiment_config
from urban_tree_transfer.experiments import (
    data_loading,
    preprocessing,
    models,
    training,
    evaluation,
    transfer,
    visualization,
)
```

### 9.3 Sync Workflow

1. **Before Running Notebook:**
   - Pull latest code to local repo
   - Push to GitHub
   - Colab installs from GitHub

2. **After Running Notebook:**
   - Download from Drive: `outputs/phase_3/metadata/*.json`
   - Download from Drive: `outputs/phase_3/figures/*.png`
   - Commit to local repo
   - Push to GitHub

3. **Model Handling:**
   - Models stay on Drive (too large for Git)
   - Reference models by Drive path in notebooks

---

## 10. Success Criteria

### 10.1 Functional Requirements

- [ ] All 6 exploratory notebooks execute without errors
- [ ] All 4 runner notebooks execute without errors
- [ ] All JSON outputs validate against schemas
- [ ] All required figures generated
- [ ] Models saved and loadable

### 10.2 Quality Requirements

- [ ] Berlin validation F1 ≥ 0.55 (minimum viable)
- [ ] Berlin validation F1 ≥ 0.60 (target)
- [ ] Train-Val gap < 30%
- [ ] Confidence intervals computed for all key metrics
- [ ] Statistical significance tests for fine-tuning comparisons

### 10.3 Documentation Requirements

- [ ] Methodology documentation complete
- [ ] All decisions logged with reasoning
- [ ] Execution logs for all notebooks

---

## 11. Timeline Estimate

| Task                                | Estimated Hours |
| ----------------------------------- | --------------- |
| Config + Schemas                    | 2h              |
| src/experiments/ modules            | 8h              |
| exp_07 (optional)                   | 2h              |
| exp_08 + exp_08b + exp_08c + exp_09 | 6h              |
| 03a runner                          | 2h              |
| exp_10                              | 4h              |
| 03b runner                          | 6h              |
| 03c runner                          | 3h              |
| 03d runner                          | 4h              |
| Testing                             | 4h              |
| Documentation                       | 3h              |
| **Total**                           | **~44h**        |

---

## 12. Design Decisions

The following methodological decisions were made during planning:

### 12.1 NN Architecture: 1D-CNN

**Decision:** Use 1D-CNN for temporal modeling (not LSTM or Transformer)

**Rationale:**

- Sequence length is only 12 time points (monthly composites) — too short for LSTM/Transformer benefits
- 1D-CNN captures local temporal patterns efficiently (spring slope, summer peak)
- Fewer parameters → more robust with limited training data (~10k-30k samples)
- Faster training and simpler hyperparameter tuning
- LSTM/Transformer would be appropriate for sequences with 100+ time points

### 12.2 Fine-tuning Strategy: Single Approach

**Decision:** Use single fine-tuning strategy per model type:

- **ML (XGBoost):** Warm-start with additional estimators
- **NN (1D-CNN/TabNet):** Full fine-tune with 0.1× original learning rate

**Rationale:**

- Testing multiple strategies (freeze layers, discriminative LR, gradual unfreezing) would multiply experiment time
- Already testing 4 fine-tuning fractions (10%, 25%, 50%, 100%)
- Strategy comparison is a separate research question beyond scope

### 12.3 Seed Strategy: Single Seed + Bootstrap CIs

**Decision:** Use single seed (42) for training, bootstrap confidence intervals for evaluation

**Rationale:**

- Multi-seed (e.g., 5 seeds) would 5× training time
- Bootstrap CIs (1000 resamples) provide variance estimates on test predictions
- HP-tuning already has inherent randomness across trials
- Optional: Final model with 3 seeds as stretch goal

### 12.4 Class Weighting: Recompute for Leipzig

**Decision:** Recompute class weights based on Leipzig distribution during fine-tuning

**Rationale:**

- Fine-tuning optimizes for Leipzig performance, not Berlin
- Leipzig has different genus distribution than Berlin
- Using `class_weight='balanced'` with Leipzig data ensures model adapts to local class frequencies
- From-scratch Leipzig baseline also uses Leipzig weights for fair comparison

---

**Status:** Ready for Implementation
