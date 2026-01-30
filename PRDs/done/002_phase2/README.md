# Phase 2 Feature Engineering PRDs

This directory contains modular PRDs for Phase 2 Feature Engineering implementation.

## PRD Structure

### Main PRD

- **[002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md)** - Overall architecture, workflow, critical fixes, and success metrics

### Task PRDs (Runner Notebooks)

- **[002a_feature_extraction.md](002a_feature_extraction.md)** - Extract CHM and Sentinel-2 features
  - Runner: `notebooks/runners/02a_feature_extraction.ipynb`
  - Module: `src/urban_tree_transfer/feature_engineering/extraction.py`
  - Legacy: `legacy/notebooks/02_feature_engineering/01_feature_extraction.md`

- **[002b_data_quality.md](002b_data_quality.md)** - NaN handling, temporal reduction, plausibility filters
  - Runner: `notebooks/runners/02b_data_quality.ipynb`
  - Module: `src/urban_tree_transfer/feature_engineering/quality.py`
  - Legacy: `legacy/notebooks/02_feature_engineering/02_data_quality_control.md`, `03b_nan_handling_plausibility.md`
  - **⚠️ Critical:** Fixes data leakage in NaN imputation (within-tree interpolation only)

- **[002c_final_preparation.md](002c_final_preparation.md)** - Redundancy removal, outliers, spatial splits
  - Runner: `notebooks/runners/02c_final_preparation.ipynb`
  - Modules: `selection.py`, `outliers.py`, `splits.py`
  - Legacy: `legacy/notebooks/02_feature_engineering/03d_correlation_analysis.md`, `03e_outlier_detection.md`, `04_spatial_splits.md`

### Exploratory PRD

- **[002_exploratory.md](002_exploratory.md)** - Parameter determination and validation
  - Notebooks: `exp_01` to `exp_05` in `notebooks/exploratory/`
  - Produces configuration files consumed by runner notebooks
  - Legacy: `legacy/notebooks/02_feature_engineering/03a_temporal_feature_selection_JM.md`, `03c_chm_relevance_assessment.md`

## Implementation Order

1. **Setup Phase**
   - Create `configs/features/feature_config.yaml`
   - Extend `config/loader.py` with feature config functions
   - Implement source modules

2. **Feature Extraction (PRD 002a)**
   - Runner: `02a_feature_extraction.ipynb`
   - Output: `trees_with_features_{city}.gpkg`

3. **Exploratory Analysis (Phase 1)**
   - `exp_01_temporal_analysis.ipynb` → `temporal_selection.json`
   - `exp_02_chm_assessment.ipynb` → `chm_assessment.json`

4. **Data Quality (PRD 002b)**
   - Runner: `02b_data_quality.ipynb`
   - Output: `trees_clean_{city}.gpkg`

5. **Exploratory Analysis (Phase 2)**
   - `exp_03_correlation_analysis.ipynb` → `correlation_removal.json`
   - `exp_04_outlier_thresholds.ipynb` → `outlier_thresholds.json`
   - `exp_05_spatial_autocorrelation.ipynb` → `spatial_autocorrelation.json`

6. **Final Preparation (PRD 002c)**
   - Runner: `02c_final_preparation.ipynb`
   - Output: `{city}_train.gpkg`, `{city}_val.gpkg`

## Working with Coding Agents

When assigning tasks to coding agents, provide:

1. **Main PRD:** [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md)
2. **Specific Task PRD:** e.g., [002a_feature_extraction.md](002a_feature_extraction.md)

This ensures agents have both high-level context and specific implementation details.

## Key Methodological Fixes

All PRDs address critical issues from legacy implementation:

1. **Data Leakage (PRD 002b)** - Within-tree interpolation only, no cross-tree statistics
2. **CHM Handling (PRD 002a/002b)** - 1m point sampling (no resampling), remove contaminated features
3. **JM Distance (exp_01)** - Known implementation issues, investigate but use results
4. **Plant Year Filtering (PRD 002b)** - NEW: Filter trees planted after 2018

## Configuration Files

All processing parameters defined in:

- `configs/features/feature_config.yaml` - Feature definitions, thresholds, split parameters
- `configs/cities/{city}.yaml` - City-specific settings (existing)
- `outputs/phase_2/metadata/*.json` - Exploratory analysis results

## Legacy Code References

All PRDs reference corresponding legacy notebooks in:

- `legacy/notebooks/02_feature_engineering/` - Original implementation notebooks
- `legacy/documentation/02_Feature_Engineering/` - Methodological documentation

**⚠️ Important:** Legacy code is for inspiration only. Data structures have changed, adapt logic rather than copying directly.

---

**Last Updated:** 2026-01-28
