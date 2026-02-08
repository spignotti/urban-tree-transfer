# PRD 003e: Phase 3 Audit Fixes

**PRD ID:** 003e
**Status:** Draft
**Created:** 2026-02-07
**Dependencies:** PRDs 003a–003d (existing implementations)
**Audit Date:** 2026-02-07

---

## 1. Overview

### 1.1 Problem Statement

A comprehensive audit of the Phase 3 experiment infrastructure revealed a significant gap between the PRD specifications and actual notebook implementations. While the **library code** (`src/urban_tree_transfer/experiments/`) is well-structured and functionally correct, the **notebooks** are skeleton-level stubs that implement only ~10-20% of the specified experiment logic. The functions exist in the library but are not wired up in the notebooks.

This PRD documents **all** discovered issues and specifies the exact fixes required to bring notebooks and code into full compliance with PRDs 003–003d.

### 1.2 Audit Scope

| Area                  | Files Reviewed                           | Status   |
| --------------------- | ---------------------------------------- | -------- |
| PRDs                  | 003, 003a, 003b, 003c, 003d              | Complete |
| Library modules       | 8 modules in `experiments/`              | Complete |
| Runner notebooks      | 03a, 03b, 03c, 03d                       | Complete |
| Exploratory notebooks | exp_08, exp_08b, exp_08c, exp_09, exp_10 | Complete |
| Documentation         | 6 docs in `03_Experiments/`              | Complete |
| Tests                 | 11 test files in `tests/experiments/`    | Complete |
| Config                | `phase3_config.yaml`                     | Complete |
| Schemas               | 5 JSON schemas                           | Complete |

### 1.3 Summary of Findings

| Severity     | Count | Description                                                 |
| ------------ | ----- | ----------------------------------------------------------- |
| **CRITICAL** | 6     | Broken functionality, wrong methodology, missing core logic |
| **HIGH**     | 5     | Missing PRD-specified analyses, broken sequential pipeline  |
| **MEDIUM**   | 5     | Config mismatches, schema gaps, inconsistencies             |
| **LOW**      | 4     | Test coverage, logging, minor gaps                          |

---

## 2. Critical Issues

### C1. Exploratory Notebooks Don't Write `setup_decisions.json`

**Affected:** exp_08, exp_08b, exp_08c, exp_09
**PRD Ref:** 003a §4.1–4.4, §2.2 (Data Flow)

**Problem:** The PRD specifies a sequential pipeline where each exploratory notebook decides a setup parameter and appends it to `outputs/phase_3/metadata/setup_decisions.json`. None of the 4 notebooks write to this file. They compute results and generate a figure but never persist the decision.

The downstream `03a_setup_fixation.ipynb` runner calls `validate_setup_decisions(setup_path)` which will crash because the file doesn't exist.

**Required Fix:**

Each notebook must, after computing results and applying decision rules:

1. Load existing `setup_decisions.json` (or initialize empty dict if first notebook)
2. Apply the decision rules from config (thresholds, criteria)
3. Write the decision with full traceability (metrics, reasoning, ablation results)
4. Validate the updated JSON against the schema

**exp_08 must add (after computing ablation results):**

```python
# Apply per-feature decision logic (see C1 + H4 combined)
baseline_f1 = results_df.loc[results_df['variant'] == 'no_chm', 'val_f1_mean'].iloc[0]
baseline_gap = results_df.loc[results_df['variant'] == 'no_chm', 'train_val_gap'].iloc[0]

# Per-feature evaluation
included_features = []
per_feature_results = []
for feature_name, variant_name in [
    ("CHM_1m", "raw_chm"),
    ("CHM_1m_zscore", "zscore_only"),
    ("CHM_1m_percentile", "percentile_only"),
]:
    row = results_df.loc[results_df['variant'] == variant_name].iloc[0]
    importance = feature_importances.get(feature_name, 0.0)
    gap_increase = row['train_val_gap'] - baseline_gap
    f1_improvement = row['val_f1_mean'] - baseline_f1

    excluded = False
    reason = ""
    if importance > config['setup_ablation']['chm']['decision_rules']['importance_threshold']:
        excluded = True
        reason = f"Dominates model (importance={importance:.3f} > 0.25)"
    elif gap_increase > config['setup_ablation']['chm']['decision_rules']['max_gap_increase']:
        excluded = True
        reason = f"Destabilizes generalization (gap increase={gap_increase:.3f} > 0.05)"
    elif f1_improvement < config['setup_ablation']['chm']['decision_rules']['min_improvement']:
        excluded = True
        reason = f"Marginal improvement ({f1_improvement:.3f} < 0.03)"
    else:
        included_features.append(feature_name)
        reason = f"Passes all criteria (imp={importance:.3f}, gap_inc={gap_increase:.3f}, f1_imp={f1_improvement:.3f})"

    per_feature_results.append({
        'feature': feature_name,
        'variant': variant_name,
        'importance': importance,
        'gap_increase': gap_increase,
        'f1_improvement': f1_improvement,
        'included': not excluded,
        'reason': reason,
    })

# Determine variant name from included features
variant_map = {
    (): "no_chm",
    ("CHM_1m_zscore",): "zscore_only",
    ("CHM_1m_percentile",): "percentile_only",
    ("CHM_1m_zscore", "CHM_1m_percentile"): "both_engineered",
    ("CHM_1m",): "raw_chm",
}
decision = variant_map.get(tuple(sorted(included_features)), "no_chm")

# Write to setup_decisions.json
setup = {
    "chm_strategy": {
        "decision": decision,
        "included_features": included_features,
        "reasoning": f"Per-feature evaluation: {len(included_features)} features passed thresholds",
        "per_feature_results": per_feature_results,
        "ablation_results": results_df.to_dict(orient='records'),
    }
}
setup_path = OUTPUT_DIR / 'metadata/setup_decisions.json'
setup_path.write_text(json.dumps(setup, indent=2))
```

**exp_08b must add:**

```python
# Load prior decisions
setup_path = OUTPUT_DIR / 'metadata/setup_decisions.json'
setup = json.loads(setup_path.read_text())

# Apply decision rules
baseline_f1 = results_df.loc[results_df['variant'] == 'baseline', 'val_f1_mean'].iloc[0]
filtered_f1 = results_df.loc[results_df['variant'] == 'filtered', 'val_f1_mean'].iloc[0]
sample_loss = 1 - (filtered_n / baseline_n)

rules = config['setup_ablation']['proximity_filter']['decision_rules']
if sample_loss > rules['max_sample_loss']:
    decision = "baseline"
    reasoning = f"Filtered loses {sample_loss:.0%} of samples (>{rules['max_sample_loss']:.0%})"
elif (filtered_f1 - baseline_f1) < rules['min_improvement']:
    decision = "baseline"
    reasoning = f"Improvement {filtered_f1 - baseline_f1:.3f} < {rules['min_improvement']}"
else:
    decision = "filtered"
    reasoning = f"Filtered improves F1 by {filtered_f1 - baseline_f1:.3f} with {sample_loss:.0%} sample loss"

setup["proximity_strategy"] = {
    "decision": decision,
    "reasoning": reasoning,
    "sample_counts": {"baseline_n": baseline_n, "filtered_n": filtered_n, "sample_loss_pct": sample_loss},
    "ablation_results": results_df.to_dict(orient='records'),
}
setup_path.write_text(json.dumps(setup, indent=2))
```

**exp_08c and exp_09 must follow the same pattern** (load prior, apply rules, write decision).

---

### C2. 03b Skips Optuna HP Tuning Entirely

**Affected:** `notebooks/runners/03b_berlin_optimization.ipynb`
**PRD Ref:** 003b §2.3 Steps 2–3

**Problem:** The notebook imports `hp_tuning` but never calls `create_study()`, `build_objective()`, or `run_optuna_search()`. It trains the ML champion with default parameters via `training.train_final_model()`. No `hp_tuning_ml.json` or `hp_tuning_nn.json` metadata is produced.

**Required Fix:** Add Optuna HP tuning for ML champion:

```python
# Load config search space
config = load_experiment_config()
ml_name = algorithm_comparison['ml_champion']['algorithm']
search_space = config['hp_tuning']['search_spaces'][ml_name]

# Create Optuna study
study = hp_tuning.create_study(
    direction="maximize",
    random_seed=config['global']['random_seed'],
)

# Build objective with spatial CV
objective = hp_tuning.build_objective(
    model_name=ml_name,
    x=x_train_scaled,
    y=y_train,
    groups=train_df['block_id'].values,
    cv=cv,
    search_space=search_space,
)

# Run optimization
hp_results = hp_tuning.run_optuna_search(
    study,
    objective,
    n_trials=config['hp_tuning']['optuna']['n_trials'],
    timeout_seconds=config['hp_tuning']['optuna']['timeout_seconds'],
)

# Save HP tuning results
hp_path = OUTPUT_DIR / 'metadata/hp_tuning_ml.json'
hp_path.write_text(json.dumps(hp_results, indent=2))
validate_hp_tuning_result(hp_path)

# Train final model with best params
best_params = hp_results['best_params']
ml_model = models.create_model(ml_name, model_params=best_params)
```

Same pattern for NN champion (see C3).

---

### C3. 03b Only Trains ML Champion — NN Champion Missing

**Affected:** `notebooks/runners/03b_berlin_optimization.ipynb`
**PRD Ref:** 003b §2.3 Steps 3–4

**Problem:** The notebook only trains the ML champion. The NN training path (CNN1D or TabNet) is completely absent. Downstream notebooks 03c and 03d expect both `berlin_ml_champion.pkl` and `berlin_nn_champion.pt`.

**Required Fix:** After ML champion HP tuning and final training, add the full NN path:

```python
# NN Champion
nn_name = algorithm_comparison['nn_champion']['algorithm']
nn_search_space = config['hp_tuning']['search_spaces'][nn_name]

nn_study = hp_tuning.create_study(direction="maximize")
nn_objective = hp_tuning.build_objective(
    model_name=nn_name,
    x=x_train_scaled,
    y=y_train,
    groups=groups,
    cv=cv,
    search_space=nn_search_space,
)
nn_hp_results = hp_tuning.run_optuna_search(nn_study, nn_objective, n_trials=50)

# Save NN HP results
nn_hp_path = OUTPUT_DIR / 'metadata/hp_tuning_nn.json'
nn_hp_path.write_text(json.dumps(nn_hp_results, indent=2))

# Train final NN model
nn_best_params = nn_hp_results['best_params']
# Add structural params for CNN1D
if nn_name == 'cnn_1d':
    nn_best_params.update({
        'n_temporal_bases': n_temporal_bases,
        'n_months': n_months,
        'n_static_features': n_static,
        'n_classes': len(label_to_idx),
    })
nn_model = models.create_model(nn_name, model_params=nn_best_params)
nn_model = training.train_final_model(nn_model, x_combined, y_combined, fit_params={...})

# Save NN model
training.save_model(nn_model, model_dir / 'berlin_nn_champion.pt', metadata={...})
```

Also save `label_encoder.pkl` and `scaler.pkl` as standalone artifacts (see M4).

---

### C4. 03d Does From-Scratch Training, NOT Fine-Tuning

**Affected:** `notebooks/runners/03d_finetuning.ipynb`
**PRD Ref:** 003d §2.3.3

**Problem:** The notebook creates a **new** `RandomForestClassifier` at each fraction level via `models.create_model('random_forest')`. The pretrained Berlin champion is loaded for metadata but never used as a warm-start base. The existing functions `finetune_xgboost()` and `finetune_neural_network()` are never called. **The core research question (RQ3) cannot be answered** because actual fine-tuning is not occurring.

**Required Fix:** Replace the training loop with actual fine-tuning:

```python
import copy

# Load pretrained Berlin champion
pretrained_ml = training.load_model(model_path)

# Create stratified subsets (see C5)
subsets = training.create_stratified_subsets(
    x_finetune_scaled, y_finetune, fractions, random_seed=config['global']['random_seed']
)

results = []
for frac, (x_sub, y_sub) in subsets.items():
    # ML fine-tuning with warm-start
    ml_finetuned = training.finetune_xgboost(
        pretrained_model=copy.deepcopy(pretrained_ml),
        x_finetune=x_sub,
        y_finetune=y_sub,
        n_additional_estimators=config['finetuning']['ml_warm_start_estimators'],
    )

    preds = ml_finetuned.predict(x_test_scaled)
    metrics = evaluation.compute_metrics(y_test, preds)
    results.append({
        'fraction': frac,
        'model_type': 'ml',
        'metrics': metrics,
        'f1_score': metrics['f1_score'],
        'predictions': preds,  # Keep for McNemar tests
    })

    # Save fine-tuned model
    training.save_model(ml_finetuned, model_dir / f'finetuned/ml_champion_f{frac}.pkl')
```

For NN fine-tuning, use `finetune_neural_network()` with reduced learning rate (after fixing C6).

---

### C5. 03d Uses `df.sample()` Instead of `create_stratified_subsets()`

**Affected:** `notebooks/runners/03d_finetuning.ipynb`
**PRD Ref:** 003d §2.3.2

**Problem:** The notebook uses `leipzig_finetune.sample(frac=frac, random_state=...)` which is **simple random sampling**. With 12 genera and small fractions (10%), minority classes may be lost entirely. The function `training.create_stratified_subsets()` exists and is tested but unused.

**Required Fix:**

```python
# Replace:
sample = leipzig_finetune.sample(frac=frac, random_state=config['global']['random_seed'])

# With:
subsets = training.create_stratified_subsets(
    x_finetune_scaled,
    y_finetune,
    fractions=config['finetuning']['fractions'],
    random_seed=config['global']['random_seed'],
)
# Then iterate: for frac, (x_sub, y_sub) in subsets.items(): ...
```

---

### C6. `CNN1D.learning_rate` Attribute Missing — `finetune_neural_network` Will Crash

**Affected:** `src/urban_tree_transfer/experiments/training.py` line 370, `models.py` CNN1D class
**PRD Ref:** 003d §2.3.3

**Problem:** `finetune_neural_network()` accesses `pretrained_model.learning_rate` on the CNN1D code path, but `CNN1D.__init__()` never stores `self.learning_rate`. This will raise `AttributeError` at runtime.

**Required Fix (Option A — preferred, store on model):**

In `models.py`, `CNN1D.__init__()`:

```python
def __init__(self, ...) -> None:
    super().__init__()
    # ... existing code ...
    self.learning_rate = DEFAULT_CNN_PARAMS["learning_rate"]
```

And in `train_cnn()`, after training:

```python
model.learning_rate = learning_rate  # Store actual LR used
```

**Required Fix (Option B — fallback in finetune):**

In `training.py`, `finetune_neural_network()`:

```python
# Replace:
original_lr = pretrained_model.learning_rate

# With:
original_lr = getattr(pretrained_model, 'learning_rate', DEFAULT_CNN_PARAMS['learning_rate'])
```

**Recommendation:** Apply both — Option A for correctness, Option B as defensive fallback.

Also update `CNN1D._init_params` dict to include `learning_rate` so it survives serialization/cloning.

---

## 3. High Issues

### H1. Exploratory Notebooks Don't Apply Prior Decisions Sequentially

**Affected:** exp_08b, exp_08c, exp_09
**PRD Ref:** 003a §2.2 (Data Flow), §4.2–4.4

**Problem:** The PRD mandates a strict chain:

```
exp_08 → chm_strategy → exp_08b reads it
exp_08b → proximity_strategy → exp_08c reads both
exp_08c → outlier_strategy → exp_09 reads all three
```

**Reality:** Each notebook loads the raw Phase 2 data independently. None reads `setup_decisions.json`. None applies CHM filtering, proximity selection, or outlier removal from prior steps. This means feature importance rankings in exp_09 are computed on the wrong dataset.

**Required Fix for exp_08b:**

```python
# Load CHM decision from exp_08
setup_path = OUTPUT_DIR / 'metadata/setup_decisions.json'
setup = json.loads(setup_path.read_text())
chm_strategy = setup['chm_strategy']['decision']

# Apply CHM strategy to both variants
for variant in variants:
    train_df, val_df, _ = data_loading.load_berlin_splits(
        DATA_DIR / 'phase_2_splits', variant=variant['name']
    )
    # Apply CHM decision
    train_df = ablation.apply_chm_strategy(train_df, chm_strategy)
    val_df = ablation.apply_chm_strategy(val_df, chm_strategy)

    chm_features = ablation.get_chm_features(chm_strategy)
    feature_cols = data_loading.get_feature_columns(
        train_df,
        include_chm=bool(chm_features),
        chm_features=chm_features or None,
    )
    # ... rest of ablation ...
```

**Required Fix for exp_08c:** Same as above, plus:

```python
# Load BOTH prior decisions
setup = json.loads(setup_path.read_text())
chm_strategy = setup['chm_strategy']['decision']
proximity_strategy = setup['proximity_strategy']['decision']

# Load the correct dataset variant (baseline or filtered)
train_df, val_df, _ = data_loading.load_berlin_splits(
    DATA_DIR / 'phase_2_splits', variant=proximity_strategy
)
# Apply CHM strategy
train_df = ablation.apply_chm_strategy(train_df, chm_strategy)
val_df = ablation.apply_chm_strategy(val_df, chm_strategy)
```

**Required Fix for exp_09:** Load all three prior decisions, apply CHM, proximity, and outlier removal before computing feature importance.

---

### H2. exp_08 CHM Ablation Lacks Per-Feature Decision Logic

**Affected:** `notebooks/exploratory/exp_08_chm_ablation.ipynb`
**PRD Ref:** 003a §4.1

**Problem:** The PRD specifies evaluating each CHM feature **individually** against 3 thresholds:

1. `feature_importance > 0.25` → exclude (dominates model, overfitting risk)
2. `train_val_gap increase > 5pp` vs no_chm baseline → exclude (destabilizes generalization)
3. `F1 improvement < 0.03` vs no_chm baseline → exclude (marginal)

The notebook only compares **variants** (no*chm, zscore_only, etc.) as a whole. It never extracts `model.feature_importances*`, never checks individual thresholds, and never applies the per-feature decision pseudocode from the PRD.

**Required Fix:** After the variant comparison loop, add:

```python
# Train RF on full feature set to get importances
all_chm_df = train_df.copy()  # Keep all CHM features
all_feature_cols = data_loading.get_feature_columns(all_chm_df, include_chm=True)
x_all = all_chm_df[all_feature_cols].to_numpy()
y_all = y_train_enc

importance_df = ablation.compute_feature_importance(x_all, y_all, all_feature_cols)
chm_importances = importance_df[importance_df['feature'].isin(['CHM_1m', 'CHM_1m_zscore', 'CHM_1m_percentile'])]

# Per-feature decision (see C1 for full logic)
# ...

# Generate per-feature importance figure
visualization.plot_feature_importance(
    importance_df[importance_df['feature'].str.startswith('CHM')],
    output_path=fig_dir / 'chm_feature_importance.png',
)

# Generate train-val gap comparison figure
visualization.plot_train_val_gap(results_df, output_path=fig_dir / 'chm_train_val_gap.png')
```

The notebook currently generates 1 figure; PRD specifies 3 (`chm_ablation_results.png`, `chm_feature_importance.png`, `chm_train_val_gap.png`).

---

### H3. 03c Transfer Evaluation Is Minimal (~10% Complete)

**Affected:** `notebooks/runners/03c_transfer_evaluation.ipynb`
**PRD Ref:** 003c §2.3 Steps 1–10

**Problem:** The PRD describes 10 analysis steps producing ~10 figures. The notebook does 1 step (zero-shot prediction + save metrics), producing 1 figure (confusion matrix). Missing steps:

| PRD Step | Description                                       | Status                                     |
| -------- | ------------------------------------------------- | ------------------------------------------ |
| Step 1   | Load both ML + NN champions                       | Only ML                                    |
| Step 2   | Zero-shot evaluation with bootstrap CIs           | Partially done (CIs computed via function) |
| Step 3   | Transfer gap (absolute + relative + Mann-Whitney) | Missing                                    |
| Step 4   | Leipzig from-scratch training                     | Missing                                    |
| Step 5   | Feature stability analysis (Spearman ρ)           | Missing                                    |
| Step 6   | 4 a-priori hypothesis tests (H1–H4)               | Missing                                    |
| Step 7   | Per-genus transfer robustness classification      | Missing                                    |
| Step 8   | Confusion matrix comparison (Berlin vs Leipzig)   | Missing (only Leipzig CM)                  |
| Step 9   | Extended transfer analysis (Nadel/Laub, species)  | Missing                                    |
| Step 10  | Best transfer model selection                     | Missing                                    |

**Required Fix:** Implement all 10 steps. The library functions already exist:

```python
# Step 3: Transfer gap
berlin_eval = json.loads((OUTPUT_DIR / 'metadata/berlin_evaluation.json').read_text())
berlin_f1 = berlin_eval['metrics']['f1_score']
leipzig_f1 = transfer_metrics['metrics']['f1_score']
gap = transfer.compute_transfer_gap(berlin_f1, leipzig_f1)

# Step 4: Leipzig from-scratch
from sklearn.preprocessing import StandardScaler
leipzig_scaler = StandardScaler()
x_finetune_scaled = leipzig_scaler.fit_transform(leipzig_finetune[feature_cols].to_numpy(dtype=float))
y_finetune = leipzig_finetune['genus_latin'].map(label_to_idx).to_numpy()
from_scratch_model = models.create_model(ml_name, model_params=best_params)
from_scratch_model.fit(x_finetune_scaled, y_finetune)
x_test_leipzig_scaled = leipzig_scaler.transform(leipzig_test[feature_cols].to_numpy(dtype=float))
from_scratch_preds = from_scratch_model.predict(x_test_leipzig_scaled)

# Step 5: Feature stability
berlin_importance = ablation.compute_feature_importance(x_train_berlin, y_train_berlin, feature_cols)
leipzig_importance = ablation.compute_feature_importance(x_finetune_scaled, y_finetune, feature_cols)
stability = transfer.compute_feature_stability(berlin_importance, leipzig_importance)

# Step 6: Hypothesis testing (see M1 for correct hypotheses)
# Step 7: Per-genus robustness
robustness = transfer.classify_transfer_robustness(berlin_per_genus, leipzig_per_genus)
ranking = transfer.compute_transfer_robustness_ranking(robustness)

# Step 8: Side-by-side confusion
berlin_cm = np.array(berlin_eval['confusion_matrix'])
visualization.plot_confusion_comparison(berlin_cm, leipzig_cm, class_labels, ...)
```

---

### H4. 03b Missing Post-Training Error Analysis

**Affected:** `notebooks/runners/03b_berlin_optimization.ipynb`
**PRD Ref:** 003b §2.3 Steps 6a–6h

**Problem:** The PRD specifies 8 error analysis steps after final model training. None are implemented. The library functions exist but aren't called.

**Required Fix:** After evaluating on Berlin test, add:

```python
# 6a: Worst confused pairs
pairs = evaluation.analyze_worst_confused_pairs(y_test, preds, class_labels, genus_german_map)
visualization.plot_confusion_pairs(pairs, fig_dir / 'worst_confused_pairs.png')

# 6b: Conifer vs deciduous
cd_metrics = evaluation.analyze_conifer_deciduous(y_test, preds, class_labels, config['genus_groups'])
visualization.plot_conifer_deciduous_comparison(cd_metrics['conifer_f1'], cd_metrics['deciduous_f1'], ...)

# 6c: Tree type (Straßen/Anlagen) — if tree_type column available
tree_type_metrics = evaluation.analyze_by_metadata(y_test, preds, test_df['tree_type'])
visualization.plot_tree_type_comparison(tree_type_metrics, ...)

# 6d: Plant year impact
decades = evaluation.bin_plant_years(test_df['plant_year'], config['plant_year_decades'])
visualization.plot_plant_year_impact(...)

# 6e: Species breakdown for low-F1 genera
species = evaluation.analyze_species_breakdown(y_test, preds, class_labels, test_df['species_latin'])
visualization.plot_species_breakdown(...)

# 6f: Feature importance
importance = ablation.compute_feature_importance(x_combined, y_combined, feature_cols)
visualization.plot_feature_importance(importance, fig_dir / 'feature_importance.png')

# 6g: Spatial error map (if geometry_lookup available)
# 6h: Misclassification Sankey
visualization.plot_misclassification_sankey(y_test, preds, class_labels, ...)

# Bootstrap CIs on test metrics
f1_ci = evaluation.bootstrap_confidence_interval(y_test, preds, lambda yt, yp: f1_score(yt, yp, average='weighted'))
```

---

### H5. Config `proximity` Key Naming Inconsistency

**Affected:** `phase3_config.yaml`, exp_08b, PRD 003a
**PRD Ref:** 003a §3.1

**Problem:** The YAML config uses `setup_ablation.proximity.variants`, but the PRD uses `setup_ablation.proximity_filter` and the code references `config['setup_ablation']['proximity']['variants']`. Meanwhile, the `decision_rules` key is present in the PRD but **missing from the YAML config** for proximity and outlier sections.

**Required Fix:** Add missing decision rules to the YAML config:

```yaml
setup_ablation:
  proximity:
    variants: [...]
    decision_rules: # ADD THIS
      min_improvement: 0.02
      max_sample_loss: 0.20
      prefer_larger_dataset: true

  outliers:
    variants: [...]
    decision_rules: # ADD THIS
      min_improvement: 0.02
      max_sample_loss: 0.15
      prefer_no_removal: true
```

Also add the CHM `decision_rules` section which is present in the PRD but absent from the YAML config:

```yaml
chm:
  features: [...]
  variants: [...]
  decision_rules: # ADD THIS
    importance_threshold: 0.25
    min_improvement: 0.03
    max_gap_increase: 0.05
    prefer_simpler: true
```

---

## 4. Medium Issues

### M1. Hypotheses in Config Don't Match PRD 003c

**Affected:** `phase3_config.yaml` → `transfer_evaluation.hypotheses`
**PRD Ref:** 003c §2.3 Step 6

**Problem:** The YAML config defines generic placeholder hypotheses:

| Config H1 | "ML model transfers better than NN model" |
| Config H2 | "Conifers transfer more robustly than deciduous" |
| Config H3 | "Genus robustness correlates with feature stability" |
| Config H4 | "Transfer gap is significant vs. Berlin baseline" |

But PRD 003c specifies testable, literature-backed hypotheses:

| PRD H1 | "Genera with more Berlin training samples transfer better" (Pearson r) |
| PRD H2 | "Nadelbäume have lower transfer gap than Laubbäume" (Mann-Whitney U, Fassnacht 2016) |
| PRD H3 | "Genera with early leaf-out have higher transfer gap" (Hemmerling 2021) |
| PRD H4 | "Genera with high Red-Edge feature importance transfer better" (Spearman ρ, Immitzer 2019) |

**Required Fix:** Replace the config hypotheses with the PRD-specified versions:

```yaml
transfer_evaluation:
  hypotheses:
    - id: H1
      description: "Genera with more Berlin training samples transfer better"
      test: pearson_correlation
      x_metric: berlin_sample_count
      y_metric: transfer_gap
      expected_direction: negative # More samples → smaller gap
      literature: "Sample size theory"
    - id: H2
      description: "Nadelbäume have lower transfer gap than Laubbäume"
      test: mann_whitney_u
      groups:
        conifer: [PINUS, PICEA]
        deciduous:
          [TILIA, ACER, QUERCUS, PLATANUS, AESCULUS, BETULA, FRAXINUS, ROBINIA]
      expected_direction: "conifer < deciduous"
      literature: "Fassnacht 2016"
    - id: H3
      description: "Genera with early leaf-out (BETULA, SALIX) have higher transfer gap"
      test: group_comparison
      groups:
        early: [BETULA, SALIX]
        mid_season:
          [TILIA, ACER, QUERCUS, PLATANUS, AESCULUS, FRAXINUS, ROBINIA]
      expected_direction: "early > mid_season"
      literature: "Hemmerling 2021"
    - id: H4
      description: "Genera with high Red-Edge feature importance transfer better"
      test: spearman_correlation
      x_metric: red_edge_importance
      y_metric: transfer_f1
      expected_direction: positive
      literature: "Immitzer 2019"
```

---

### M2. Schema `setup_decisions.json` — Optional Fields Should Be Required

**Affected:** `schemas/setup_decisions.schema.json`
**PRD Ref:** 003a §5.1, §6.2

**Problem:** Most traceability fields (`reasoning`, `ablation_results`, `sample_counts`) are defined as **optional** in the JSON schema. The PRD implies they should be required for reproducibility — a valid `setup_decisions.json` that contains only bare `decision` strings without explanation defeats the purpose of documented decision-making.

**Required Fix:** In `setup_decisions.schema.json`, add `"required"` arrays within each strategy object:

```json
"chm_strategy": {
  "type": "object",
  "required": ["decision", "reasoning", "ablation_results"],
  ...
}
```

Apply to all 4 strategy objects (`chm_strategy`, `proximity_strategy`, `outlier_strategy`, `feature_set`).

---

### M3. Fine-Tuning Warm-Start Estimators Mismatch

**Affected:** `phase3_config.yaml`, `training.py`
**PRD Ref:** 003d §2.3.3

**Problem:** Config says `ml_warm_start_estimators: 200`, but `finetune_xgboost()` defaults to `n_additional_estimators: 100`. The PRD says 100.

**Required Fix:** Either:

- Change config to `ml_warm_start_estimators: 100` (match code + PRD), or
- Change code default to 200 and update docstring

**Recommendation:** Align to config value. The notebook should read from config:

```python
training.finetune_xgboost(
    pretrained_ml,
    x_sub,
    y_sub,
    n_additional_estimators=config['finetuning']['ml_warm_start_estimators'],
)
```

---

### M4. 03b Doesn't Save `label_encoder.pkl` Separately

**Affected:** `notebooks/runners/03b_berlin_optimization.ipynb`
**PRD Ref:** 003c §2.2 Inputs

**Problem:** PRD 003c expects `label_encoder.pkl` as a standalone artifact. 03b bakes `label_to_idx`/`idx_to_label` into the model metadata JSON. 03c then reconstructs it manually, which works but is fragile and inconsistent with the PRD architecture.

**Required Fix:** Save `label_encoder.pkl` separately in 03b:

```python
import pickle

label_encoder_data = {'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}
with (model_dir / 'label_encoder.pkl').open('wb') as f:
    pickle.dump(label_encoder_data, f)
```

Update 03c/03d to load from this file.

---

### M5. 03d Doesn't Fit Leipzig-Specific Scaler for From-Scratch Baselines

**Affected:** `notebooks/runners/03d_finetuning.ipynb`
**PRD Ref:** 003d §2.3.4

**Problem:** PRD requires: "Fit **new scaler** on Leipzig finetune data (independent from Berlin)" for from-scratch baselines. The notebook always uses the Berlin scaler for everything, including what is effectively from-scratch training.

**Required Fix:** For from-scratch baselines:

```python
# From-scratch: fit NEW scaler on Leipzig
from sklearn.preprocessing import StandardScaler

leipzig_scaler = StandardScaler()
x_finetune_leipzig_scaled = leipzig_scaler.fit_transform(
    leipzig_finetune[feature_cols].to_numpy(dtype=float)
)
x_test_leipzig_scaled = leipzig_scaler.transform(
    leipzig_test[feature_cols].to_numpy(dtype=float)
)

# Train from scratch
scratch_model = models.create_model(ml_name, model_params=best_params)
scratch_model.fit(x_finetune_leipzig_scaled, y_finetune)
scratch_preds = scratch_model.predict(x_test_leipzig_scaled)
```

For fine-tuning: continue using Berlin scaler (model expects Berlin-scaled features).

---

## 5. Low Issues

### L1. Test Coverage Gaps for Visualization

**Affected:** `tests/experiments/test_visualization.py`

**Problem:** 24 of 34 visualization functions lack tests. The 10 tested cover core paths, but PRD-specified plots for transfer analysis, fine-tuning curves, and error analysis are untested.

**Required Fix:** Add smoke tests for untested functions. Each test should verify:

- Function runs without error on synthetic data
- Output file is created at specified path
- Plot has expected title/axes

Priority functions to test:

- `plot_per_genus_comparison` (used by exp_08b)
- `plot_sample_loss_by_genus` (used by exp_08b)
- `plot_outlier_distribution` (used by exp_08c)
- `plot_performance_ladder` (used by exp_10)
- `plot_transfer_conifer_deciduous` (used by 03c)
- `plot_finetuning_ml_vs_nn` (used by 03d)
- `plot_significance_matrix` (used by 03d)
- `plot_feature_importance` (used by 03b, exp_08)
- `plot_optuna_history` (used by 03b)

---

### L2. Missing Execution Logs in Runner Notebooks

**Affected:** 03b, 03c, 03d
**PRD Ref:** 003 §2.1 (logs directory)

**Problem:** PRD specifies execution logs (`logs/03b_berlin_optimization.json`, etc.) capturing runtime metadata. Only 03a writes a log file.

**Required Fix:** Add log writing to each runner notebook (at the end):

```python
import datetime

log = {
    'status': 'completed',
    'timestamp': datetime.datetime.now().isoformat(),
    'runtime_seconds': elapsed,
    'config_hash': hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest(),
    'key_results': { ... },
}
log_path = OUTPUT_DIR / 'logs/03b_berlin_optimization.json'
log_path.write_text(json.dumps(log, indent=2))
```

---

### L3. exp_10 Algorithm Comparison — Missing Coarse Grid Search

**Affected:** `notebooks/exploratory/exp_10_algorithm_comparison.ipynb`
**PRD Ref:** 003b §2.1

**Problem:** The PRD specifies coarse grid search for ML models (24 configs for RF, 48 for XGBoost). The notebook trains each model with default parameters only via `models.create_model(model_name)` — no grid iteration. This means the "best coarse HP" selection is comparing single-config defaults, not grid-searched variants.

**Required Fix:** Iterate over the coarse grid from config:

```python
from itertools import product

for model_name in ['random_forest', 'xgboost']:
    grid = config['algorithm_comparison']['models'][model_name]['grid']
    param_names = list(grid.keys())
    param_values = list(grid.values())

    best_val_f1 = -1
    best_params = {}

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))
        model = models.create_model(model_name, model_params=params)
        metrics = training.train_with_cv(model, x_train_scaled, y_train, groups, cv)

        if metrics['val_f1_mean'] > best_val_f1:
            best_val_f1 = metrics['val_f1_mean']
            best_params = params

    results.append({
        'algorithm': model_name,
        'type': 'ml',
        'val_f1_mean': best_val_f1,
        'best_params': best_params,
        ...
    })
```

---

### L4. No `berlin_test_filtered.parquet` Handling Verification

**Affected:** `ablation.prepare_ablation_dataset()`, 03a notebook
**PRD Ref:** 003b §2.2

**Problem:** The 03a notebook calls `prepare_ablation_dataset()` for `(berlin, test)`. If the proximity decision is `"filtered"`, the function tries to load `berlin_test_filtered.parquet`. It's unclear if Phase 2 generates filtered variants for test splits.

**Required Fix:** Either:

- Verify Phase 2 generates `{city}_{split}_filtered.parquet` for ALL splits (train, val, test, finetune), or
- Only apply proximity filtering to train/val splits in 03a (test sets should not be filtered — they represent the natural population)

**Recommendation:** Test splits should NOT be proximity-filtered. They represent the real-world distribution. Modify 03a to only apply proximity filtering to train/val/finetune splits:

```python
for city, split in [
    ('berlin', 'train'),
    ('berlin', 'val'),
    ('berlin', 'test'),
    ('leipzig', 'finetune'),
    ('leipzig', 'test'),
]:
    # Override proximity for test splits — never filter test data
    effective_decisions = dict(setup)
    if split == 'test':
        effective_decisions = {**setup}
        effective_decisions['proximity_strategy'] = {'decision': 'baseline', **setup.get('proximity_strategy', {})}

    df, meta = ablation.prepare_ablation_dataset(
        base_path=splits_dir, city=city, split=split,
        setup_decisions=effective_decisions, return_metadata=True,
    )
```

---

## 6. Missing 03d Analyses

The following PRD-specified analyses are completely absent from 03d and must be implemented:

### 6.1 McNemar Significance Tests

**PRD Ref:** 003d §2.3.5

```python
# Compare each fine-tuning level vs. zero-shot
zero_shot_preds = model.predict(x_test_scaled)  # From 03c
for frac, result in frac_results.items():
    sig = transfer.mcnemar_test(y_test, zero_shot_preds, result['predictions'])
    result['mcnemar_vs_zeroshot'] = sig

# Compare each level vs. from-scratch
for frac, result in frac_results.items():
    sig = transfer.mcnemar_test(y_test, from_scratch_preds, result['predictions'])
    result['mcnemar_vs_scratch'] = sig

# Visualize significance matrix
visualization.plot_significance_matrix(p_values_df, fig_dir / 'finetuning_significance_matrix.png')
```

### 6.2 Power-Law Curve Fitting

**PRD Ref:** 003d §2.3.6

```python
from scipy.optimize import curve_fit

def power_law(x, a, b):
    return a * np.power(x, b)

fracs = np.array([r['fraction'] for r in results])
f1s = np.array([r['f1_score'] for r in results])

popt, pcov = curve_fit(power_law, fracs, f1s, p0=[1.0, 0.35], maxfev=5000)
a, b = popt
r_squared = 1 - np.sum((f1s - power_law(fracs, a, b))**2) / np.sum((f1s - np.mean(f1s))**2)

# Extrapolate to 95% recovery
target_f1 = 0.95 * from_scratch_f1
recovery_fraction = (target_f1 / a) ** (1 / b)
```

### 6.3 Per-Genus Recovery Heatmap

**PRD Ref:** 003d §2.3.7

```python
# Compute per-genus F1 at each fraction
per_genus_by_fraction = {}
for frac, result in frac_results.items():
    per_class = evaluation.compute_per_class_metrics(y_test, result['predictions'], class_labels)
    per_genus_by_fraction[frac] = {row['genus']: row['f1_score'] for row in per_class.to_dict('records')}

visualization.plot_finetuning_per_genus_recovery(per_genus_by_fraction, class_labels, fig_dir / 'per_genus_recovery.png')
```

### 6.4 ML vs. NN Comparison

**PRD Ref:** 003d §2.3.8

Both ML and NN champions must be fine-tuned (requires C3/C4 fixes first). Then:

```python
visualization.plot_finetuning_ml_vs_nn(ml_results, nn_results, fig_dir / 'ml_vs_nn_comparison.png')
```

---

## 7. Implementation Order

The fixes have dependencies. Follow this order:

```
Phase 1: Code Fixes (no notebook changes needed)
├── C6: Fix CNN1D.learning_rate attribute
├── H5: Add decision_rules to phase3_config.yaml
├── M1: Fix config hypotheses to match PRD 003c
├── M2: Tighten schema required fields
└── M3: Align warm-start estimators config

Phase 2: Exploratory Notebook Fixes
├── H2: exp_08 — add per-feature decision logic
├── C1 (exp_08): Write chm_strategy to setup_decisions.json
├── H1 (exp_08b): Load CHM decision, apply before ablation
├── C1 (exp_08b): Write proximity_strategy to setup_decisions.json
├── H1 (exp_08c): Load CHM + proximity decisions
├── C1 (exp_08c): Write outlier_strategy to setup_decisions.json
├── H1 (exp_09): Load all 3 decisions, apply before feature ranking
└── C1 (exp_09): Write feature_set + selected_features (completes JSON)

Phase 3: Runner Notebook Fixes (after exploratory notebooks work)
├── C2: 03b — add Optuna HP tuning for ML champion
├── C3: 03b — add NN champion training path
├── H4: 03b — add post-training error analysis
├── M4: 03b — save label_encoder.pkl separately
├── L2: 03b — add execution log
├── H3: 03c — implement all 10 transfer analysis steps
├── L2: 03c — add execution log
├── C4: 03d — replace from-scratch with actual fine-tuning
├── C5: 03d — use create_stratified_subsets()
├── M5: 03d — Leipzig-specific scaler for from-scratch
├── 6.1–6.4: 03d — add McNemar, power-law, per-genus, ML-vs-NN
└── L2: 03d — add execution log

Phase 4: Testing & Polish
├── L1: Add visualization test coverage
├── L3: exp_10 — add coarse grid search
└── L4: Verify test split filtering policy
```

---

## 8. Estimated Effort

| Phase                          | Items        | Estimated Effort |
| ------------------------------ | ------------ | ---------------- |
| Phase 1: Code Fixes            | 5 items      | 2–3 hours        |
| Phase 2: Exploratory Notebooks | 8 items      | 6–8 hours        |
| Phase 3: Runner Notebooks      | 12 items     | 12–16 hours      |
| Phase 4: Testing & Polish      | 3 items      | 3–4 hours        |
| **Total**                      | **28 items** | **23–31 hours**  |

---

## 9. Success Criteria

### 9.1 Functional

- [ ] Running exp_08 → exp_08b → exp_08c → exp_09 sequentially produces a complete, valid `setup_decisions.json`
- [ ] 03a reads `setup_decisions.json` and produces `phase_3_experiments/` datasets for both cities
- [ ] 03b produces Optuna-tuned ML + NN champions with `hp_tuning_*.json` metadata
- [ ] 03c produces `transfer_evaluation.json` with gap metrics, robustness ranking, and hypothesis results
- [ ] 03d produces `finetuning_curve.json` with power-law fit and McNemar significance
- [ ] All JSON outputs validate against their respective schemas
- [ ] All execution logs produced

### 9.2 Methodology

- [ ] CHM decision uses per-feature evaluation against 3 thresholds
- [ ] Exploratory notebooks apply prior decisions sequentially (no stale data)
- [ ] Fine-tuning uses warm-start from Berlin champion, not from-scratch training
- [ ] Stratified sampling maintains genus proportions at all fractions
- [ ] Leipzig from-scratch baselines use independently fitted scalers
- [ ] 4 a-priori hypotheses match PRD 003c with literature references
- [ ] McNemar tests compare paired predictions correctly

### 9.3 Quality

- [ ] All tests pass (`uv run nox -s ci`)
- [ ] No pyright type errors
- [ ] `CNN1D.learning_rate` attribute available after training
- [ ] Config decision_rules present for all ablation sections

---

_Last Updated: 2026-02-07_
