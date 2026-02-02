# Phase 2 Feature Engineering - Implementation Plan

**Created:** 2026-01-28
**Branch:** `feat/phase-2-feature-engineering`

---

## Workflow: Claude + Codex

We use a split workflow to minimize token usage and maximize implementation quality:

1. **Claude** breaks Phase 2 into single tasks and writes a **Codex prompt** for each
2. **User** gives the prompt to **Codex** along with referenced documentation files
3. **Codex** implements the task autonomously
4. **User** runs `uv run nox -s lint typecheck` and has Codex fix any issues
5. **Claude** validates the result (reviews code, checks against PRD requirements)
6. **User** commits via `/git commit` and moves to the next task

### Codex Prompt Structure

Each prompt should contain:
- **Goal** - What to implement (1-2 sentences)
- **Required Documentation** - PRD and legacy doc paths to read first
- **Reference Code** - Existing files for pattern matching
- **Key Requirements** - Bullet list of specific behaviors
- **Technical Notes** - CRS, file patterns, batch sizes, etc.
- **Validation** - How to verify correctness (`uv run nox -s lint typecheck`)

### Important Rules
- Codex should read the referenced PRD files itself (don't paste content into prompts)
- Codex should follow existing code patterns from reference files
- All functions need type hints and Google-style docstrings
- No hardcoded values - everything from `feature_config.yaml`
- Tests use synthetic rasters in `TemporaryDirectory`

---

## Task Overview

| # | Task | PRD | Module/File | Status |
|---|------|-----|-------------|--------|
| 1 | Setup: Config + Loader + Module stubs | Overview | `feature_config.yaml`, `loader.py`, stubs | Done |
| 2 | Implement `extraction.py` | 002a | `feature_engineering/extraction.py` | Done |
| 3 | Runner notebook `02a_feature_extraction.ipynb` | 002a | `notebooks/runners/` | Done |
| 4 | Exploratory `exp_01_temporal_analysis.ipynb` | 002_exploratory | `notebooks/exploratory/` | Done |
| 5 | Exploratory `exp_02_chm_assessment.ipynb` | 002_exploratory | `notebooks/exploratory/` | Done |
| 6 | Implement `quality.py` | 002b | `feature_engineering/quality.py` | Done |
| 7 | Runner notebook `02b_data_quality.ipynb` | 002b | `notebooks/runners/` | Done |
| 8 | Exploratory `exp_03_correlation_analysis.ipynb` | 002_exploratory | `notebooks/exploratory/` | Done |
| 9 | Exploratory `exp_04_outlier_thresholds.ipynb` | 002_exploratory | `notebooks/exploratory/` | Done |
| 10 | Exploratory `exp_05_spatial_autocorrelation.ipynb` | 002_exploratory | `notebooks/exploratory/` | Done |
| 11 | Exploratory `exp_06_mixed_genus_proximity.ipynb` | 002_exploratory | `notebooks/exploratory/` | Done |
| 12 | Implement `selection.py`, `outliers.py`, `splits.py`, `proximity.py` | 002c | `feature_engineering/` | Done |
| 13 | Runner notebook `02c_final_preparation.ipynb` | 002c | `notebooks/runners/` | Done |

---

## Execution Order & Dependencies

```
Task 1: Setup (config, loader, stubs)           ← DONE
Task 2: extraction.py implementation             ← DONE
    ↓
Task 3: 02a_feature_extraction.ipynb (runner)
    ↓
    [PAUSE - Run notebook on Colab to generate feature data]
    ↓
Task 4: exp_01_temporal_analysis.ipynb      ──┐
Task 5: exp_02_chm_assessment.ipynb         ──┤ (can be parallel)
    ↓                                         │
    [PAUSE - Run on Colab, sync JSON configs] │
    ↓                                         │
Task 6: quality.py implementation           ←─┘
Task 7: 02b_data_quality.ipynb (runner)
    ↓
    [PAUSE - Run notebook on Colab]
    ↓
Task 8:  exp_03_correlation_analysis.ipynb   ──┐
Task 9:  exp_04_outlier_thresholds.ipynb     ──┤
Task 10: exp_05_spatial_autocorrelation.ipynb──┤ (can be parallel)
Task 11: exp_06_mixed_genus_proximity.ipynb  ──┘
    ↓
    [PAUSE - Run on Colab, sync JSON configs]
    ↓
Task 12: selection.py + outliers.py + splits.py + proximity.py
Task 13: 02c_final_preparation.ipynb (runner)
    ↓
    Phase 2 complete → Ready for Phase 3
```

---

## Detailed Task Specifications

### Task 3: Runner Notebook 02a

**Goal:** Create the Colab runner notebook for feature extraction.

**Docs:**
- `PRDs/002_phase2/002a_feature_extraction.md` (notebook structure section)
- `docs/templates/Notebook_Templates.md`
- `notebooks/runners/01_data_processing.ipynb` (pattern reference)

**Key Requirements:**
- Follow runner template structure (setup, mount, imports, config, processing, validation)
- Per-city loop calling `extract_all_features()`
- Save output as `trees_with_features_{city}.gpkg`
- Save metadata JSON to `outputs/phase_2/metadata/`
- Skip-if-exists logic for idempotent execution

---

### Task 4: Exploratory exp_01 (Temporal Analysis)

**Goal:** JM distance analysis to determine which months to keep.

**Docs:**
- `PRDs/002_phase2/002_exploratory.md` (exp_01 section)
- `legacy/documentation/02_Feature_Engineering/03_Temporal_Feature_Selection_JM_Methodik.md`
- `docs/documentation/02_Feature_Engineering/01_Visualization_Strategy.md`

**Key Requirements:**
- Compute JM distance per month/city/genus pair
- Publication-quality plots (line chart, heatmap)
- Output: `temporal_selection.json` with `selected_months` list

---

### Task 5: Exploratory exp_02 (CHM Assessment)

**Goal:** Assess CHM feature value and determine plant year threshold.

**Docs:**
- `PRDs/002_phase2/002_exploratory.md` (exp_02 section)
- `PRDs/002_phase2_feature_engineering_overview.md` (sections 4.2, 4.4, 4.5)
- `legacy/documentation/02_Feature_Engineering/05_CHM_Relevance_Assessment_Methodik.md`

**Key Requirements:**
- CHM_1m vs plant_year regression → detect visibility threshold
- Genus classification (deciduous/coniferous) from data
- Output: `chm_assessment.json` with `plant_year_threshold`, genus classification

---

### Task 6: Implement quality.py

**Goal:** Implement all stubs in `feature_engineering/quality.py`.

**Docs:**
- `PRDs/002_phase2/002b_data_quality.md`
- `PRDs/002_phase2_feature_engineering_overview.md` (section 4.1 - data leakage fix)

**Reference Code:**
- `src/urban_tree_transfer/feature_engineering/extraction.py` (patterns)
- `src/urban_tree_transfer/config/loader.py`

**Key Requirements:**
- Within-tree interpolation only (NO cross-tree statistics - data leakage!)
- Edge NaN tolerance: 1 month fill, ≥2 months → remove tree
- CHM z-score and percentile within genus
- NDVI plausibility: max(NDVI) ≥ 0.3 in growing season
- Plant year filtering from config

---

### Task 7: Runner Notebook 02b

**Goal:** Create Colab runner for data quality control.

**Docs:**
- `PRDs/002_phase2/002b_data_quality.md` (notebook structure)
- `docs/templates/Notebook_Templates.md`

**Key Requirements:**
- Load `temporal_selection.json` and `chm_assessment.json` from Drive
- Per-city processing calling `run_quality_pipeline()`
- Output: `trees_clean_{city}.gpkg` with 0 NaN

---

### Tasks 8-11: Exploratory exp_03, exp_04, exp_05, exp_06

**Goal:** Correlation analysis, outlier thresholds, spatial autocorrelation, mixed-genus proximity.

**Docs:**
- `PRDs/002_phase2/002_exploratory.md` (exp_03, exp_04, exp_05, exp_06 sections)
- `legacy/documentation/02_Feature_Engineering/06_Correlation_Analysis_Redundancy_Reduction_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/07_Outlier_Detection_Final_Filtering_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/08_Spatial_Splits_Stratification_Methodik.md`
- `docs/documentation/02_Feature_Engineering/02_Exploratory_05_Spatial_Autocorrelation.md`
- `docs/documentation/02_Feature_Engineering/02_Exploratory_06_Mixed_Genus_Proximity.md`

**Outputs:**
- `correlation_removal.json` (features to remove)
- `outlier_thresholds.json` (detection parameters, flagging strategy)
- `spatial_autocorrelation.json` (empirically determined block size via Moran's I)
- `proximity_filter.json` (mixed-genus distance threshold, ~20m expected)

**Notes:**
- Block size is data-driven (could be 300m-600m). Used by Task 13 via JSON, NOT from feature_config.yaml.
- Proximity threshold balances retention rate (≥85%) with spectral purity (Sentinel-2 pixel contamination)

---

### Task 12: Implement selection.py + outliers.py + splits.py + proximity.py

**Goal:** Implement all stubs in the four final modules.

**Docs:**
- `PRDs/002_phase2/002c_final_preparation.md`
- `PRDs/002_phase2_feature_engineering_overview.md`
- `docs/documentation/02_Feature_Engineering/02c_Final_Preparation_Methodik.md`

**Key Requirements:**

**selection.py:**
- `remove_redundant_features()` - Drop features from correlation_removal.json
- Correlation: |r| > 0.95 threshold, keep higher-variance feature

**outliers.py:**
- `detect_zscore_outliers()` - Per-feature Z-score detection (threshold: 3.0)
- `detect_mahalanobis_outliers()` - Per-genus multivariate distance (α=0.001)
- `detect_iqr_outliers()` - Height-based Tukey fences (k=1.5)
- `apply_consensus_outlier_filter()` - Severity-based flagging (high/medium/low/none)
- **CRITICAL:** NO trees removed! Only metadata flags added (trees_removed = 0)
- Adds 5 columns: outlier_zscore, outlier_mahalanobis, outlier_iqr, outlier_severity, outlier_method_count

**splits.py:**
- `create_spatial_blocks()` - Regular grid with empirical block size
- `create_stratified_splits_berlin()` - 70/15/15 train/val/test with StratifiedGroupKFold
- `create_stratified_splits_leipzig()` - 80/20 finetune/test
- `validate_split_stratification()` - KL-divergence + spatial overlap checks
- Block size from exp_05 JSON, not hardcoded

**proximity.py:**
- `compute_nearest_different_genus_distance()` - Per-tree distance to nearest different genus
- `apply_proximity_filter()` - Remove trees < threshold_m from different genus
- `analyze_genus_specific_impact()` - Per-genus removal rate validation
- Threshold from exp_06 JSON (~20m for 2-pixel Sentinel-2 buffer)

---

### Task 13: Runner Notebook 02c

**Goal:** Create Colab runner for final preparation with dual dataset strategy.

**Docs:**
- `PRDs/002_phase2/002c_final_preparation.md` (notebook structure)
- `docs/documentation/02_Feature_Engineering/02c_Final_Preparation_Methodik.md`
- `docs/templates/Notebook_Templates.md`

**Key Requirements:**
- Load 4 exploratory configs: `correlation_removal.json`, `outlier_thresholds.json`, `spatial_autocorrelation.json`, `proximity_filter.json`
- Extract `recommended_block_size_m` from spatial config
- Extract `recommended_threshold_m` from proximity config
- Remove redundant features per correlation analysis
- **Outlier flagging only** (NO removal, assert trees_removed == 0)
- Create spatial blocks with empirical block size
- **Dual dataset strategy:** Create BOTH baseline AND proximity-filtered variants
- Generate stratified splits for both variants (Berlin: 70/15/15, Leipzig: 80/20)
- Validate spatial disjointness (spatial_overlap == 0) and genus stratification (KL < 0.01)
- **Output: 10 GeoPackages** (5 baseline + 5 filtered) + `phase_2_final_summary.json`
  - Berlin baseline: train, val, test
  - Berlin filtered: train_filtered, val_filtered, test_filtered
  - Leipzig baseline: finetune, test
  - Leipzig filtered: finetune_filtered, test_filtered

**Rationale for Dual Strategy:**
- Baseline datasets: Include all trees (with proximity-contaminated samples)
- Filtered datasets: Exclude trees < 20m from different genus (spectral purity)
- Enables ablation studies in Phase 3 (baseline vs. filtered performance comparison)

---

## Reference: Key Documentation Paths

| Document | Path |
|----------|------|
| Phase 2 Overview PRD | `PRDs/002_phase2_feature_engineering_overview.md` |
| Feature Extraction PRD | `PRDs/002_phase2/002a_feature_extraction.md` |
| Data Quality PRD | `PRDs/002_phase2/002b_data_quality.md` |
| Final Preparation PRD | `PRDs/002_phase2/002c_final_preparation.md` |
| Exploratory PRD | `PRDs/002_phase2/002_exploratory.md` |
| Notebook Templates | `docs/templates/Notebook_Templates.md` |
| Visualization Strategy | `docs/documentation/02_Feature_Engineering/01_Visualization_Strategy.md` |
| Workflow & Config | `docs/documentation/02_Feature_Engineering/00_Workflow_and_Configuration.md` |
| Feature Config | `src/urban_tree_transfer/configs/features/feature_config.yaml` |
| Legacy Methodology | `legacy/documentation/02_Feature_Engineering/` |

---

## Phase 2 Status: ✅ COMPLETE

**Completion Date:** 2026-01-30

**Summary:**
- All 13 tasks implemented and validated
- 6 exploratory notebooks (exp_01-06) with data-driven parameter determination
- 3 runner notebooks (02a, 02b, 02c) for automated processing
- 7 feature engineering modules (extraction, quality, selection, outliers, splits, proximity + tests)
- 10 ML-ready GeoPackages (baseline + filtered variants)
- Complete methodology documentation for all components
- Ready for Phase 3: Experiments

**Final Outputs:**
- `data/phase_2_splits/*.gpkg` (10 GeoPackages for training/evaluation)
- `outputs/phase_2/metadata/*.json` (6 JSON configuration files)
- `docs/documentation/02_Feature_Engineering/` (8 methodology documents)

**Next Phase:** Phase 3 - Cross-City Transfer Learning Experiments

---

**Last Updated:** 2026-01-30
