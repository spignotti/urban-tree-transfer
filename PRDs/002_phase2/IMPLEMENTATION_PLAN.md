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
| 10 | Exploratory `exp_05_spatial_autocorrelation.ipynb` | 002_exploratory | `notebooks/exploratory/` | Pending |
| 11 | Implement `selection.py`, `outliers.py`, `splits.py` | 002c | `feature_engineering/` | Pending |
| 12 | Runner notebook `02c_final_preparation.ipynb` | 002c | `notebooks/runners/` | Pending |

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
Task 9:  exp_04_outlier_thresholds.ipynb     ──┤ (can be parallel)
Task 10: exp_05_spatial_autocorrelation.ipynb──┘
    ↓
    [PAUSE - Run on Colab, sync JSON configs]
    ↓
Task 11: selection.py + outliers.py + splits.py
Task 12: 02c_final_preparation.ipynb (runner)
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

### Tasks 8-10: Exploratory exp_03, exp_04, exp_05

**Goal:** Correlation analysis, outlier thresholds, spatial autocorrelation.

**Docs:**
- `PRDs/002_phase2/002_exploratory.md` (exp_03, exp_04, exp_05 sections)
- `legacy/documentation/02_Feature_Engineering/06_Correlation_Analysis_Redundancy_Reduction_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/07_Outlier_Detection_Final_Filtering_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/08_Spatial_Splits_Stratification_Methodik.md`

**Outputs:**
- `correlation_removal.json` (features to remove)
- `outlier_thresholds.json` (detection parameters)
- `spatial_autocorrelation.json` (empirically determined block size via Moran's I)

**Note for exp_05:** Block size is data-driven (could be 300m, 400m, 500m, 600m). Used by Task 12 via JSON, NOT from feature_config.yaml.

---

### Task 11: Implement selection.py + outliers.py + splits.py

**Goal:** Implement all stubs in the three remaining modules.

**Docs:**
- `PRDs/002_phase2/002c_final_preparation.md`
- `PRDs/002_phase2_feature_engineering_overview.md`

**Key Requirements:**
- Correlation: |r| > 0.95 threshold, keep higher-variance feature
- Outliers: Z-score + Mahalanobis + IQR, consensus-based (3/3 = remove, 2/3 and 1/3 = flag)
- Splits: Spatial blocks with empirical size (from exp_05 JSON), StratifiedGroupKFold
- Berlin: 70/15/15 train/val/test
- Leipzig: 80/20 finetune/test

**Note:** `splits.py` takes `block_size_m` as function parameter, not loaded from config internally.

---

### Task 12: Runner Notebook 02c

**Goal:** Create Colab runner for final preparation.

**Docs:**
- `PRDs/002_phase2/002c_final_preparation.md` (notebook structure)
- `docs/templates/Notebook_Templates.md`

**Key Requirements:**
- Load exploratory configs: `correlation_removal.json`, `outlier_thresholds.json`, `spatial_autocorrelation.json`
- Extract `recommended_block_size_m` from spatial config (NOT from feature_config.yaml)
- Remove redundant features per correlation analysis
- Apply consensus outlier filtering (3/3 = remove, 2/3 and 1/3 = metadata flags)
- Create spatial blocks with empirical block size
- Generate stratified splits (Berlin: 70/15/15, Leipzig: 80/20)
- Output: 5 GeoPackages (3 Berlin + 2 Leipzig) + final_dataset_summary.json

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

**Last Updated:** 2026-01-30
