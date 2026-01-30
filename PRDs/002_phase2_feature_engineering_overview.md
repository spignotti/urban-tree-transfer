# PRD: Phase 2 Feature Engineering (Overview)

**PRD ID:** 002
**Status:** Draft
**Created:** 2026-01-26
**Last Updated:** 2026-01-28
**Revision:** v2 - Added Edge-NaN tolerance, consensus-based outliers, correlation priority rules, dynamic genus classification

---

## 1. Overview

### 1.1 Problem Statement

Phase 1 Data Processing has been completed, producing harmonized tree cadastres, 1m CHM rasters, and monthly Sentinel-2 composites for Berlin and Leipzig. Phase 2 must transform these raw inputs into training-ready ML datasets through feature extraction, quality control, and spatial splitting.

The legacy pipeline mixed processing and exploratory analysis in single notebooks, making it hard to distinguish data preparation from methodological investigation. Additionally, several methodological issues were identified that need addressing.

### 1.2 Goals

1. **Translate** legacy Feature Engineering pipeline to new repo structure
2. **Separate** processing (runner notebooks) from analysis (exploratory notebooks)
3. **Fix** critical methodological issues (data leakage, CHM handling)
4. **Config-driven** metadata columns and feature definitions (no hardcoding)
5. **Preserve** well-working logic from legacy pipeline

### 1.3 Non-Goals

- Model training (Phase 3)
- Extensive hyperparameter tuning for feature selection
- Supporting cities beyond Berlin and Leipzig

---

## 2. Architecture

### 2.1 Modular PRD Structure

This Phase 2 work is split into modular PRDs:

- **002_phase2_feature_engineering_overview.md** (this document) - Overall architecture, workflow, and critical fixes
- **[002a_feature_extraction.md](002_phase2/002a_feature_extraction.md)** - Tree correction, CHM/S2 extraction (Runner: `02a_feature_extraction.ipynb`)
- **[002b_data_quality.md](002_phase2/002b_data_quality.md)** - NaN handling, temporal reduction, plausibility filters (Runner: `02b_data_quality.ipynb`)
- **[002c_final_preparation.md](002_phase2/002c_final_preparation.md)** - Redundancy removal, outliers, spatial splits (Runner: `02c_final_preparation.ipynb`)
- **[002_exploratory.md](002_phase2/002_exploratory.md)** - Exploratory notebooks for parameter determination (exp_01 to exp_05)

**Working with Coding Agents:** Always provide the main overview PRD + the specific task PRD for the step being implemented.

**⚠️ IMPORTANT - Workflow & Configuration:**  
Exploratory notebooks produce JSON configs that must be **manually synced** from Google Drive to Git before running subsequent runner notebooks. See [Workflow & Configuration Guide](../docs/documentation/02_Feature_Engineering/00_Workflow_and_Configuration.md) for complete details on the Colab → Drive → Local → Git → Colab cycle.

**📊 IMPORTANT - Visualization:**  
All exploratory notebooks must produce **publication-quality plots** with consistent styling. Phase 2 generates ~23+ plots for documentation and presentations. See [Visualization Strategy](../docs/documentation/02_Feature_Engineering/01_Visualization_Strategy.md) for style guide and organization.

### 2.2 Directory Structure

```
urban-tree-transfer/
├── configs/
│   ├── cities/
│   │   ├── berlin.yaml          # Existing
│   │   └── leipzig.yaml         # Existing
│   └── features/
│       └── feature_config.yaml  # NEW: Feature definitions + metadata schema
├── src/urban_tree_transfer/
│   └── feature_engineering/
│       ├── __init__.py          # Exports
│       ├── extraction.py        # Tree correction, CHM/S2 extraction
│       ├── quality.py           # NaN handling, interpolation, plausibility
│       ├── selection.py         # Temporal selection, correlation, redundancy
│       ├── outliers.py          # Z-score, Mahalanobis, IQR detection
│       └── splits.py            # Spatial blocks, stratified splits
├── notebooks/
│   ├── runners/
│   │   ├── 01_data_processing.ipynb      # Existing
│   │   ├── 02a_feature_extraction.ipynb  # NEW (PRD 002a)
│   │   ├── 02b_data_quality.ipynb        # NEW (PRD 002b)
│   │   └── 02c_final_preparation.ipynb   # NEW (PRD 002c)
│   └── exploratory/
│       ├── exp_01_temporal_analysis.ipynb        # NEW
│       ├── exp_02_chm_assessment.ipynb           # NEW
│       ├── exp_03_correlation_analysis.ipynb     # NEW
│       ├── exp_04_outlier_thresholds.ipynb       # NEW
│       └── exp_05_spatial_autocorrelation.ipynb  # NEW
└── outputs/
    ├── phase_1/                 # Existing from Phase 1
    └── phase_2/                 # NEW for Phase 2
        ├── logs/
        └── metadata/
            ├── feature_extraction_summary.json
            ├── temporal_selection.json           # From exploratory
            ├── correlation_removal.json          # From exploratory
            ├── outlier_thresholds.json           # From exploratory
            └── spatial_autocorrelation.json      # From exploratory
```

### 2.3 Data Flow

```
Phase 1 Outputs
├── trees_filtered_viable.gpkg (Berlin + Leipzig)
├── CHM_1m_{city}.tif
└── S2_{city}_{year}_{month}_median.tif (12 months × 2 cities)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  02a_feature_extraction.ipynb (PRD 002a)                    │
│  ├── Tree correction (snap-to-peak using 1m CHM)            │
│  ├── CHM feature extraction (1m point sampling)             │
│  └── Sentinel-2 feature extraction (monthly)                │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            trees_with_features_{city}.gpkg
            (277 features: 1 CHM + 276 S2, plus 10 metadata columns)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────────┐   ┌───────────────────────────────┐
│ exp_01_temporal   │   │ exp_02_chm_assessment         │
│ JM distance       │   │ η², Cohen's d, feature eng.   │
│ → month selection │   │ → CHM feature decisions       │
└───────────────────┘   └───────────────────────────────┘
        │                       │
        ▼                       ▼
  temporal_selection.json   chm_assessment.json
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  02b_data_quality.ipynb (PRD 002b)                          │
│  ├── Load temporal selection config                         │
│  ├── Apply month filtering (per temporal_selection.json)    │
│  ├── NaN filtering (>2 months → remove)                     │
│  ├── Linear interpolation (within-tree only!)               │
│  ├── CHM feature engineering (Z-score, percentile)          │
│  └── NDVI plausibility filter (max_NDVI ≥ 0.3)              │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            trees_clean_{city}.gpkg
            (0 NaN)
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────────┐   ┌───────────────────────────────┐
│ exp_03_correlation│   │ exp_04_outlier_thresholds     │
│ Intra-class r     │   │ Z, Mahalanobis, IQR analysis  │
│ → redundancy list │   │ → threshold decisions         │
└───────────────────┘   └───────────────────────────────┘
        │                       │
        ▼                       ▼
  correlation_removal.json   outlier_thresholds.json
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  02c_final_preparation.ipynb (PRD 002c)                     │
│  ├── Load correlation config → remove redundant features    │
│  ├── Load outlier config → detect and remove outliers       │
│  ├── Sample size filtering (MIN_SAMPLES_PER_GENUS)          │
│  ├── Create spatial blocks (size from exp_05)               │
│  └── City-specific stratified splits:                       │
│      • Berlin: 70/15/15 Train/Val/Test                      │
│      • Leipzig: 80/20 Finetune/Test                         │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Final Outputs
            ├── Berlin (Source City):
            │   ├── berlin_train.gpkg (70%)
            │   ├── berlin_val.gpkg (15%)
            │   └── berlin_test.gpkg (15%)
            ├── Leipzig (Target City):
            │   ├── leipzig_finetune.gpkg (80%)
            │   └── leipzig_test.gpkg (20%)
            └── final_dataset_summary.json
```

---

## 3. Configuration Architecture

### 3.1 Feature Config (`configs/features/feature_config.yaml`)

**NEW** central configuration file defining:

- **Metadata columns** - Preserved through pipeline (identifiers, labels)
- **CHM features** - 1m resolution, direct extraction, engineered features
- **Spectral bands** - 10 Sentinel-2 bands
- **Vegetation indices** - 13 indices (broadband, red-edge, water)
- **Temporal configuration** - Reference year, extraction/selected months
- **Plant year filtering** - Max planting year for satellite visibility
- **Quality thresholds** - NaN limits, NDVI plausibility, correlation cutoff
- **Outlier detection** - Z-score, Mahalanobis, IQR parameters
- **Spatial splits** - Block size, train/val ratio, stratification settings
- **Genus classification** - Deciduous/coniferous lists

**Critical:** All thresholds must be justified via exploratory analysis or literature references.

### 3.2 Config Loader Extension

Extensions to `src/urban_tree_transfer/config/loader.py`:

```python
def load_feature_config() -> dict[str, Any]:
    """Load feature engineering configuration."""

def get_metadata_columns() -> list[str]:
    """Get list of metadata columns to preserve."""

def get_spectral_features() -> list[str]:
    """Get list of all spectral band names."""

def get_vegetation_indices() -> list[str]:
    """Get flat list of all vegetation index names."""

def get_all_feature_names(months: list[int] | None = None) -> list[str]:
    """Get complete list of feature names for given months."""
```

---

## 4. Critical Methodological Fixes

### 4.1 Data Leakage in NaN Imputation (MUST FIX)

**Problem:** Legacy computed genus/city means on full dataset including test set.

**Challenge:** Train/val split happens at the end (02c), after NaN handling (02b).

**Solution:** Use **within-tree interpolation only** with conservative Edge-NaN tolerance:

1. **Interior NaNs:** Linear temporal interpolation (month 4 missing? interpolate from months 3 and 5)
2. **Edge NaNs (1 month at start/end):** Forward/backward-fill with nearest available value
3. **Edge NaNs (≥2 months at start/end):** Remove tree (insufficient temporal data)

**Rationale for 1-Month Tolerance:**

- With 1 month missing at edge, 11 of 12 months (92%) are still available
- Nearest-neighbor fill for single edge month minimizes pattern distortion
- With ≥2 months missing, significant portion of phenological curve is lost
- **Literature:** Jönsson & Eklundh (2004) "TIMESAT" recommend max 1-2 missing observations at edges

**Configuration:**

```yaml
nan_handling:
  interior_method: "linear_interpolation"
  edge_method: "nearest" # forward/backward fill
  max_edge_nan_months: 1 # ≥2 → remove tree
```

**Why this works:** Each tree is interpolated independently using only its own temporal data. No information from other trees (genus means, city means) is used, so there's no leakage regardless of train/val split.

**Implementation:** See PRD 002b for detailed implementation.

### 4.2 CHM Handling (SIMPLIFIED)

**Problem:** Legacy used 10m resampled CHM which caused neighbor contamination (r=0.638 between cadastre height and CHM_mean).

**Solution:**

- Extract directly from 1m CHM at tree point (no resampling)
- Keep only: `CHM_1m`, `CHM_1m_zscore`, `CHM_1m_percentile`
- Remove: `CHM_mean`, `CHM_max`, `CHM_std`, `crown_ratio` (all contaminated)

**Note:** `height_m` (cadastre) and `CHM_1m` (raster) are DIFFERENT sources and not interchangeable.

### 4.3 JM Distance Implementation (NEEDS FIXING)

**Problem:** Legacy JM values were consistently low (0.5-1.2) even for clearly separable genera.

**Investigation Tasks:**

1. Validate formula against Bruzzone et al. (1995)
2. Test with synthetic data (known μ, σ differences)
3. Check numerical stability (log operations, small variances)
4. Consider multivariate JM as alternative

**Note:** This is exploratory work - the runner will use whatever months the exploratory notebook determines, even if JM implementation isn't perfect. See PRD 002_exploratory for details.

### 4.4 Plant Year Filtering

**New Addition:** Filter trees by planting year to exclude recently planted trees that may not be visible in satellite data.

**Threshold:** Determined statistically in `exp_02_chm_assessment.ipynb`

**Analysis Steps (in exp_02):**

1. Plot CHM_1m vs. plant_year regression
2. Identify detection threshold (typical: CHM > 2m for Sentinel-2 visibility)
3. Find breakpoint where median CHM drops below detection threshold
4. Output `plant_year_threshold` in `chm_assessment.json`

**Expected Result:** `plant_year ≤ 2018` (trees ≥3 years old in 2021), but validated empirically.

**Rationale:** Young trees need time to develop detectable canopy in 10m Sentinel-2 pixels. The threshold is derived from data, not assumed.

**Implementation:** See PRD 002b. The `plant_year` column is kept in metadata for potential future analyses.

### 4.5 Genus Classification (Deciduous/Coniferous)

**New Addition:** Dynamic genus classification instead of hardcoded lists.

**Analysis Steps (in exp_02_chm_assessment):**

1. List all genera with sample counts from tree data
2. Classify each genus using lookup table (biological knowledge required)
3. Apply threshold check: If `n_conifer_genera < 3` OR `n_conifer_samples < 500` → exclude conifers from analysis
4. Output genus classification and analysis scope in `chm_assessment.json`

**Lookup Table (in feature_config.yaml, extensible):**

```yaml
genus_classification:
  deciduous:
    - ACER
    - TILIA
    - QUERCUS
    - PLATANUS
    - FRAXINUS
    - BETULA
    - AESCULUS
    - ROBINIA
    - CARPINUS
    - FAGUS
    - POPULUS
    - PRUNUS
    - SORBUS
    - ULMUS
    - CORYLUS
    - SALIX
    - ALNUS
    - CRATAEGUS
    - MALUS
  coniferous:
    - PINUS
    - PICEA
    - ABIES
    - LARIX
    - TAXUS
    - THUJA
    - JUNIPERUS
    - PSEUDOTSUGA

  # Thresholds for inclusion
  min_genera_for_analysis: 3
  min_samples_for_analysis: 500
```

**Rationale:** Avoids hardcoding genus lists. Analysis dynamically adapts to actual data distribution in each city.

**Implementation:** See PRD 002_exploratory (exp_02) for analysis, PRD 002b for filtering.

---

## 5. Notebook Structure & Templates

**All Phase 2 notebooks follow standardized templates** based on the proven structure from Runner 1 (`notebooks/runners/01_data_processing.ipynb`).

**Documentation:** [Notebook_Templates.md](../docs/templates/Notebook_Templates.md)

### 5.1 Runner-Notebook Template

**Purpose:** Execute data processing (feature extraction, quality control, final preparation)  
**Output:** Processed datasets (Parquet/GeoPackage), metadata JSONs, execution logs

**Standard Structure:**

1. Runtime Settings & Package Installation (with GitHub token)
2. Mount Google Drive
3. Package Imports + Plotting/Logging setup
4. Configuration (directories, cities, config loading)
5. Processing Sections (with `ExecutionLog`, skip-if-exists logic, validation)
6. Summary & Validation Report

**Key Features:**

- Consistent Git integration (install from private repo)
- Reusable setup (plotting, logging, validation)
- Skip-if-exists logic (idempotent execution)
- Standardized validation & error handling

**See:** [Runner-Notebook Template Documentation](../docs/templates/Notebook_Templates.md#1-runner-notebook-template-phase-2)

### 5.2 Exploratory-Notebook Template

**Purpose:** Parameter determination, statistical analysis, visualization  
**Output:** JSON configurations (for runner notebooks), publication-quality plots, analysis metadata

**Standard Structure:**
1-4. Same base setup as Runner notebooks 5. Analysis Sections (compute statistics, create visualizations, save JSONs) 6. Summary with "Manual Sync Required" instructions

**Key Features:**

- Multiple publication-quality plots per section (~23+ total across Phase 2)
- Consistent visualization style (via `setup_plotting()`, `save_figure()`)
- JSON outputs with thresholds/parameters for subsequent runner notebooks
- Explicit manual sync workflow (Colab → Drive → Local → Git)

**Visualization:** All plots follow [Visualization Strategy](../docs/documentation/02_Feature_Engineering/01_Visualization_Strategy.md)  
**Template:** [Exploratory-Notebook Template Documentation](../docs/templates/Notebook_Templates.md#2-exploratory-notebook-template-phase-2)

### 5.3 Benefits of Standardization

- **Easy maintenance** → Same structure across all notebooks
- **Predictable workflow** → Coding agents know where to find configuration, imports, processing logic
- **Reproducibility** → Consistent setup, logging, validation
- **Quality assurance** → Standardized error handling and output validation

---

## 6. Legacy Code References

### 6.1 Legacy Notebooks (Inspiration, not 1:1 copy)

**Location:** `legacy/notebooks/02_feature_engineering/`

- `01_feature_extraction.md` → Inspiration for PRD 002a
- `02_data_quality_control.md` → Inspiration for PRD 002b
- `03a_temporal_feature_selection_JM.md` → Inspiration for exp_01
- `03b_nan_handling_plausibility.md` → Inspiration for PRD 002b
- `03c_chm_relevance_assessment.md` → Inspiration for exp_02
- `03d_correlation_analysis.md` → Inspiration for exp_03, PRD 002c
- `03e_outlier_detection.md` → Inspiration for exp_04, PRD 002c
- `04_spatial_splits.md` → Inspiration for PRD 002c

**Note:** Data structures have changed from Phase 1 processing. Code must be adapted, not copied directly.

### 6.2 Legacy Documentation (Methodological Context)

**Location:** `legacy/documentation/02_Feature_Engineering/`

- `01_Feature_Extraction_Methodik.md`
- `02_Data_Quality_Control_Methodik.md`
- `03_Temporal_Feature_Selection_JM_Methodik.md`
- `04_NaN_Handling_Plausibility_Methodik.md`
- `05_CHM_Relevance_Assessment_Methodik.md`
- `06_Correlation_Analysis_Redundancy_Reduction_Methodik.md`
- `07_Outlier_Detection_Final_Filtering_Methodik.md`
- `08_Spatial_Splits_Stratification_Methodik.md`

**Usage:** These documents explain the methodological rationale behind each step. Consult for context and scientific justification, but adapt implementation to new codebase structure.

---

## 7. Execution Order

### 7.1 Initial Setup

1. Create `configs/features/feature_config.yaml`
2. Extend `config/loader.py` with feature config functions
3. Implement source modules in `src/urban_tree_transfer/feature_engineering/`

### 7.2 Runner Sequence

```
02a_feature_extraction.ipynb (PRD 002a)
        │
        ▼
    [PAUSE: Run exploratory notebooks exp_01, exp_02]
    - exp_01: temporal_selection.json
    - exp_02: chm_assessment.json (plant_year threshold, genus classification)
        │
        ▼
02b_data_quality.ipynb (PRD 002b)
        │
        ▼
    [PAUSE: Run exploratory notebooks exp_03, exp_04, exp_05]
    - exp_03: correlation_removal.json
    - exp_04: outlier_thresholds.json
    - exp_05: spatial_autocorrelation.json (⚠️ REQUIRED before 02c!)
        │
        ▼
02c_final_preparation.ipynb (PRD 002c)
        │
        ▼
    Ready for Phase 3 (Experiments)
```

**⚠️ CRITICAL:** `exp_05_spatial_autocorrelation` MUST run before `02c_final_preparation` because block size is determined empirically via Moran's I analysis.

### 7.3 Exploratory Notebooks (run after 02a/02b)

**After 02a:**

- **exp_01_temporal_analysis** → produces `temporal_selection.json`
- **exp_02_chm_assessment** → produces `chm_assessment.json` (incl. plant_year threshold, genus classification)

**After 02b:**

- **exp_03_correlation_analysis** → produces `correlation_removal.json`
- **exp_04_outlier_thresholds** → produces `outlier_thresholds.json`
- **exp_05_spatial_autocorrelation** → produces `spatial_autocorrelation.json` (⚠️ required for 02c block_size)

See PRD 002_exploratory for detailed specifications.

---

## 7. Validation Criteria

### 7.1 Feature Extraction (02a)

- [ ] All trees have CHM_1m value (no NaN from extraction)
- [ ] Feature count matches config (1 CHM + 23 bands × 12 months = 277)
- [ ] CRS is EPSG:25833
- [ ] Metadata columns preserved

### 7.2 Data Quality (02b)

- [ ] 0 NaN values in output
- [ ] Temporal reduction applied correctly (per temporal_selection.json)
- [ ] No data leakage: interpolation within-tree only
- [ ] NDVI plausibility: no trees with max_NDVI < 0.3
- [ ] Plant year filtering: all trees planted ≤ 2018

### 7.3 Final Preparation (02c)

- [ ] Redundant features removed per config
- [ ] Outlier removal rate reasonable (~0.5-1.5% for 3/3 consensus)
- [ ] All genera meet minimum sample threshold
- [ ] Spatial blocks are disjoint
- [ ] **Berlin:** Train/Val/Test (70/15/15) splits created
- [ ] **Leipzig:** Finetune/Test (80/20) splits created
- [ ] All splits stratified (KL-divergence < 0.01)
- [ ] No spatial overlap between any splits
- [ ] 5 output files: 3 Berlin + 2 Leipzig

### 7.4 Overall Pipeline

- [ ] Consistent tree_id tracking through pipeline
- [ ] Metadata JSON files generated at each step
- [ ] Processing can be resumed (skip logic for existing outputs)
- [ ] Both cities processed with identical logic

---

## 8. Success Metrics

- **Feature completeness:** >95% trees with all features after extraction
- **Data retention:** >85% trees after all filtering steps
- **Clean data:** 0 NaN values in final datasets
- **Split quality:** KL-divergence < 0.01 for genus stratification
- **Reproducibility:** Identical results with same config and seed

---

## 9. Risk Mitigation

| Risk                     | Mitigation                                                        |
| ------------------------ | ----------------------------------------------------------------- |
| JM implementation issues | Runner uses output regardless; fix in exploratory as time permits |
| Large dataset memory     | Batch processing (50k trees), lazy loading rasters                |
| Sentinel-2 cloud gaps    | Already handled by Phase 1 median composites                      |
| Cross-city inconsistency | Same config applied to both; validation checks                    |

---

## 10. Next Steps

1. **Read task-specific PRDs** before implementation:
   - [PRD 002a: Feature Extraction](002_phase2/002a_feature_extraction.md)
   - [PRD 002b: Data Quality](002_phase2/002b_data_quality.md)
   - [PRD 002c: Final Preparation](002_phase2/002c_final_preparation.md)
   - [PRD 002_exploratory: Exploratory Notebooks](002_phase2/002_exploratory.md)

2. **Implement in order:**
   - Setup (config + loader + modules)
   - 002a (Feature Extraction)
   - Exploratory 01-02
   - 002b (Data Quality)
   - Exploratory 03-05
   - 002c (Final Preparation)

---

**PRD Status:** Ready for implementation
**Last Updated:** 2026-01-28
