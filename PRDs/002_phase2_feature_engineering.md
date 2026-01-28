# PRD: Phase 2 Feature Engineering

**PRD ID:** 002
**Status:** Draft
**Created:** 2026-01-26
**Author:** Claude (with user guidance)

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

### 2.1 Directory Structure

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
│   │   ├── 02a_feature_extraction.ipynb  # NEW
│   │   ├── 02b_data_quality.ipynb        # NEW
│   │   └── 02c_final_preparation.ipynb   # NEW
│   └── exploratory/
│       ├── exp_01_temporal_analysis.ipynb    # NEW
│       ├── exp_02_chm_assessment.ipynb       # NEW
│       ├── exp_03_correlation_analysis.ipynb # NEW
│       └── exp_04_outlier_thresholds.ipynb   # NEW
├── docs/documentation/
│   └── 02_Feature_Engineering/
│       └── 02_Feature_Engineering_Methodik.md  # NEW
└── outputs/
    └── metadata/
        ├── feature_extraction_summary.json
        ├── temporal_selection.json       # From exploratory
        ├── correlation_removal.json      # From exploratory
        └── outlier_thresholds.json       # From exploratory
```

### 2.2 Data Flow

```
Phase 1 Outputs
├── trees_filtered_viable.gpkg (Berlin + Leipzig)
├── CHM_1m_{city}.tif
└── S2_{city}_{year}_{month}_median.tif (12 months × 2 cities)
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│  02a_feature_extraction.ipynb                               │
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
│  02b_data_quality.ipynb                                     │
│  ├── Load temporal selection config                         │
│  ├── Apply month filtering (per temporal_selection.json)    │
│  ├── NaN filtering (>2 months → remove)                     │
│  ├── Linear interpolation (training-set means only!)        │
│  ├── CHM feature engineering (Z-score, percentile)          │
│  └── NDVI plausibility filter (max_NDVI ≥ 0.3)              │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            trees_clean_{city}.gpkg
            (190 features, 0 NaN)
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
│  02c_final_preparation.ipynb                                │
│  ├── Load correlation config → remove redundant features    │
│  ├── Load outlier config → detect and remove outliers       │
│  ├── Sample size filtering (MIN_SAMPLES_PER_GENUS)          │
│  ├── Create 500×500m spatial blocks                         │
│  └── StratifiedGroupKFold splits (80/20)                    │
└─────────────────────────────────────────────────────────────┘
                    │
                    ▼
            Final Outputs (per city)
            ├── {city}_train.gpkg
            ├── {city}_val.gpkg
            └── split_statistics.json
```

---

## 3. Configuration

### 3.1 Feature Config (`configs/features/feature_config.yaml`)

```yaml
# Feature Engineering Configuration
# All metadata columns and feature definitions in one place

# Metadata columns preserved through pipeline (from Phase 1 trees schema)
# These columns are NOT features - they are identifiers and labels
metadata_columns:
  - tree_id         # Unique identifier
  - city            # City name (Berlin/Leipzig)
  - genus_latin     # Target label (UPPERCASE)
  - species_latin   # Species (lowercase, nullable)
  - genus_german    # German genus name (nullable)
  - species_german  # German species name (nullable)
  - plant_year      # Planting year (used for filtering, then dropped)
  - height_m        # Cadastre tree height (kept as reference)
  - tree_type       # Source layer identifier (nullable)
  - geometry        # Point location

# CHM Features (1m resolution, direct extraction, no resampling)
# Note: height_m (cadastre) and CHM_1m (raster) are DIFFERENT sources
chm:
  source_resolution: 1m
  features:
    - name: CHM_1m
      description: "Canopy height at tree location from 1m CHM raster"
  engineered:
    # Engineered from CHM_1m, NOT from cadastre height_m
    - name: CHM_1m_zscore
      description: "Z-score normalized CHM height per city"
    - name: CHM_1m_percentile
      description: "Percentile rank of CHM height within city"

# Sentinel-2 Spectral Bands
spectral_bands:
  - B2    # Blue (10m)
  - B3    # Green (10m)
  - B4    # Red (10m)
  - B5    # Red Edge 1 (20m)
  - B6    # Red Edge 2 (20m)
  - B7    # Red Edge 3 (20m)
  - B8    # NIR (10m)
  - B8A   # Red Edge 4 (20m)
  - B11   # SWIR 1 (20m)
  - B12   # SWIR 2 (20m)

# Vegetation Indices
vegetation_indices:
  broadband:
    - NDVI
    - EVI
    - GNDVI
    - kNDVI
    - VARI
  red_edge:
    - NDre1
    - NDVIre
    - CIre
    - IRECI
    - RTVIcore
    - MCARI
  water:
    - NDWI
    - MSI
    - NDII

# Temporal Configuration
temporal:
  extraction_months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Extract all 12 months
  # selected_months: DETERMINED BY exp_01_temporal_analysis.ipynb
  # DO NOT hardcode - let exploratory analysis decide

# Plant Year Filtering
# Trees planted recently may not be visible in satellite data
plant_year:
  min_year: null  # To be determined - filter young trees
  # After filtering, plant_year column can be dropped from features

# Quality Control Thresholds
# NOTE: All thresholds below are PRELIMINARY and need validation in exploratory notebooks
# Each should be either: (1) statistically justified, or (2) sensitivity-analyzed
quality:
  nan_month_threshold: 2        # NEEDS JUSTIFICATION: Why 2? Sensitivity analysis in exp_01
  ndvi_min_threshold: 0.3       # NEEDS JUSTIFICATION: Literature reference or sensitivity analysis
  correlation_threshold: 0.95   # NEEDS JUSTIFICATION: Literature reference (r²=0.90 common)

# Outlier Detection
# NOTE: All thresholds need statistical justification in exp_04_outlier_thresholds.ipynb
outliers:
  zscore_threshold: 3.0         # Standard, but document rationale
  zscore_feature_count: 10      # NEEDS JUSTIFICATION: Why 10 features?
  mahalanobis_alpha: 0.001      # NEEDS JUSTIFICATION: Why not 0.01 or 0.0001?
  iqr_multiplier: 1.5           # Tukey's standard, document reference

# Spatial Splits
# NOTE: block_size needs spatial autocorrelation analysis in exploratory
splits:
  block_size_m: 500             # NEEDS JUSTIFICATION: Based on spatial autocorr decay
  train_ratio: 0.8              # Standard 80/20 split
  random_seed: 42
  min_samples_per_genus: 500    # Matches Phase 1 viability filter

# Genus Classification (deciduous only for spectral homogeneity)
genus_filter:
  include_type: deciduous
  deciduous_genera:
    - ACER
    - TILIA
    - QUERCUS
    - PRUNUS
    - BETULA
    - ROBINIA
    - ULMUS
    - POPULUS
    - SORBUS
    - FRAXINUS
    - PLATANUS
    - FAGUS
    - AESCULUS
    - SALIX
    - CARPINUS
    - CORYLUS
    - ALNUS
    - CRATAEGUS
    - MALUS
  coniferous_genera:
    - PINUS
```

### 3.2 Config Loader Extension

Add to `src/urban_tree_transfer/config/loader.py`:

```python
def load_feature_config() -> dict[str, Any]:
    """Load feature engineering configuration."""
    config_path = CONFIGS_DIR / "features" / "feature_config.yaml"
    return _load_yaml(config_path)

def get_metadata_columns() -> list[str]:
    """Get list of metadata columns to preserve."""
    config = load_feature_config()
    return config["metadata_columns"]

def get_spectral_features() -> list[str]:
    """Get list of all spectral band names."""
    config = load_feature_config()
    return config["spectral_bands"]

def get_vegetation_indices() -> list[str]:
    """Get flat list of all vegetation index names."""
    config = load_feature_config()
    indices = config["vegetation_indices"]
    return indices["broadband"] + indices["red_edge"] + indices["water"]

def get_all_feature_names(months: list[int] | None = None) -> list[str]:
    """Get complete list of feature names for given months."""
    config = load_feature_config()
    if months is None:
        months = config["temporal"]["extraction_months"]

    bands = config["spectral_bands"]
    indices = get_vegetation_indices()
    chm = [f["name"] for f in config["chm"]["features"]]

    features = chm.copy()
    for month in months:
        for band in bands:
            features.append(f"{band}_{month:02d}")
        for idx in indices:
            features.append(f"{idx}_{month:02d}")

    return features
```

---

## 4. Implementation Tasks

### 4.1 Runner Notebook: 02a_feature_extraction.ipynb

**Purpose:** Extract raw features from Phase 1 outputs

**Inputs:**
- `data/phase_1_processing/trees/trees_filtered_viable.gpkg`
- `data/phase_1_processing/chm/CHM_1m_{city}.tif`
- `data/phase_1_processing/sentinel2/S2_{city}_{year}_{MM}_median.tif`

**Outputs:**
- `data/phase_2_features/trees_with_features_{city}.gpkg`
- `outputs/metadata/feature_extraction_summary.json`

**Processing Steps:**

1. **Tree Correction (per city)**
   - Load trees and 1m CHM
   - For each tree: find local CHM maximum within search radius
   - Snap tree position to peak (height-weighted)
   - Extract CHM value at corrected position → `CHM_1m`
   - Update `height_m` from CHM if cadastre value missing
   - Flag trees that couldn't be corrected → exclude

2. **Sentinel-2 Feature Extraction (per city, per month)**
   - Load monthly composite (23 bands: 10 spectral + 13 indices)
   - Point-sample at tree locations (no interpolation)
   - Name columns: `{band}_{MM}` (e.g., `NDVI_04` for April NDVI)
   - Preserve NoData as NaN

3. **Consolidation**
   - Merge CHM features with S2 features
   - Validate schema matches config
   - Save per-city GeoPackage
   - Generate extraction summary (counts, NaN rates, feature completeness)

**Source Module: `extraction.py`**

```python
def correct_tree_positions(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
    search_radius_m: float = 5.0,
    height_weight: float = 0.7,
) -> gpd.GeoDataFrame:
    """Snap tree positions to local CHM maximum."""

def extract_chm_features(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
) -> gpd.GeoDataFrame:
    """Extract CHM value at each tree location (1m resolution)."""

def extract_sentinel_features(
    trees_gdf: gpd.GeoDataFrame,
    sentinel_dir: Path,
    city: str,
    year: int,
    months: list[int],
) -> gpd.GeoDataFrame:
    """Extract all Sentinel-2 bands/indices for specified months."""

def extract_all_features(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
    sentinel_dir: Path,
    city: str,
    year: int,
    feature_config: dict,
) -> gpd.GeoDataFrame:
    """Complete feature extraction pipeline for one city."""
```

---

### 4.2 Runner Notebook: 02b_data_quality.ipynb

**Purpose:** Clean features and handle missing values

**Inputs:**
- `data/phase_2_features/trees_with_features_{city}.gpkg`
- `outputs/metadata/temporal_selection.json` (from exploratory)
- `outputs/metadata/chm_assessment.json` (from exploratory)

**Outputs:**
- `data/phase_2_features/trees_clean_{city}.gpkg`
- `outputs/metadata/data_quality_summary.json`

**Processing Steps:**

1. **Consolidate Cities**
   - Load both city datasets
   - Validate schemas match

2. **Genus Filtering**
   - Filter to deciduous genera only (from config)
   - Log genus distribution

3. **Plant Year Filtering**
   - Filter out recently planted trees (may not be visible in satellite data)
   - Threshold to be determined (e.g., planted before 2018 for 2021 imagery)
   - After filtering, `plant_year` column is no longer needed for ML
   - Keep in metadata for reference but exclude from feature set

5. **Temporal Reduction**
   - Load `temporal_selection.json` for selected months
   - Drop columns for non-selected months
   - Log feature count reduction

6. **NaN Handling** (CRITICAL: Avoid data leakage!)
   - Analyze NaN distribution per tree
   - Remove trees with >2 months missing (threshold from config)
   - **Within-tree interpolation only** (no cross-tree information):
     - Linear temporal interpolation for interior NaNs (between valid months)
     - Edge NaNs (first/last selected month missing): forward/backward fill from adjacent month
     - If edge cannot be filled: use tree's own mean across available months for that feature
   - Trees that still have NaN after within-tree interpolation → remove
   - **NO genus/city means** - this would leak information between trees

7. **CHM Feature Engineering**
   - Compute `CHM_1m_zscore` per city (using all trees - no leakage since CHM is input data)
   - Compute `CHM_1m_percentile` per city (percentile rank)

8. **NDVI Plausibility**
   - Compute `max_NDVI` across all months
   - Remove trees with `max_NDVI < 0.3`

9. **Validation**
   - Assert 0 NaN values
   - Validate all expected columns present
   - Save per-city clean datasets

**Source Module: `quality.py`**

```python
def filter_deciduous_genera(
    gdf: gpd.GeoDataFrame,
    deciduous_genera: list[str],
) -> gpd.GeoDataFrame:
    """Filter to deciduous genera only."""

def filter_by_plant_year(
    gdf: gpd.GeoDataFrame,
    min_year: int | None = None,
    max_year: int | None = None,
) -> gpd.GeoDataFrame:
    """Filter trees by planting year to exclude recently planted trees."""

def apply_temporal_selection(
    gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict,
) -> gpd.GeoDataFrame:
    """Remove columns for non-selected months."""

def analyze_nan_distribution(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute NaN statistics per tree and per feature."""

def filter_nan_trees(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    max_nan_months: int = 2,
) -> gpd.GeoDataFrame:
    """Remove trees with too many missing months."""

def interpolate_features_within_tree(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    selected_months: list[int],
) -> gpd.GeoDataFrame:
    """Interpolate NaN values using within-tree temporal information only.

    No cross-tree information is used to avoid data leakage:
    - Interior NaNs: linear interpolation between adjacent months
    - Edge NaNs: forward/backward fill, then tree's own feature mean
    """

def compute_chm_engineered_features(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add CHM_1m_zscore and CHM_1m_percentile per city.

    Note: CHM is input data, not derived from other trees,
    so computing city-level stats doesn't cause leakage.
    """

def filter_ndvi_plausibility(
    gdf: gpd.GeoDataFrame,
    ndvi_columns: list[str],
    min_threshold: float = 0.3,
) -> gpd.GeoDataFrame:
    """Remove trees with max_NDVI below threshold."""
```

---

### 4.3 Runner Notebook: 02c_final_preparation.ipynb

**Purpose:** Finalize features and create train/val splits

**Inputs:**
- `data/phase_2_features/trees_clean_{city}.gpkg`
- `outputs/metadata/correlation_removal.json` (from exploratory)
- `outputs/metadata/outlier_thresholds.json` (from exploratory)

**Outputs:**
- `data/phase_2_features/final/{city}_train.gpkg`
- `data/phase_2_features/final/{city}_val.gpkg`
- `outputs/metadata/split_statistics.json`
- `outputs/metadata/final_dataset_summary.json`

**Processing Steps:**

1. **Feature Redundancy Removal**
   - Load `correlation_removal.json` for features to drop
   - Remove redundant features
   - Log feature count reduction

2. **Outlier Detection**
   - Load thresholds from `outlier_thresholds.json`
   - Z-score: Flag trees with ≥N features exceeding threshold
   - Mahalanobis: Flag multivariate outliers (per genus)
   - IQR: Flag CHM outliers (per genus × city)
   - Hierarchical decision: CRITICAL (Mahal + one other) → remove
   - Log outlier statistics

3. **Sample Size Filtering**
   - Check genus counts against MIN_SAMPLES_PER_GENUS
   - Remove non-viable genera
   - Log final genus distribution

4. **Spatial Block Creation**
   - Create 500×500m regular grid per city
   - Assign each tree to nearest block
   - Validate all trees assigned

5. **Stratified Splits**
   - Use StratifiedGroupKFold with blocks as groups
   - 80/20 train/val split
   - Validate genus proportions in each split
   - Save separate GeoPackages

**Source Module: `outliers.py`**

```python
def detect_zscore_outliers(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    z_threshold: float = 3.0,
    min_feature_count: int = 10,
) -> pd.Series:
    """Flag trees with many extreme Z-scores."""

def detect_mahalanobis_outliers(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    alpha: float = 0.001,
) -> pd.Series:
    """Flag multivariate outliers per genus."""

def detect_iqr_outliers(
    gdf: gpd.GeoDataFrame,
    height_column: str = "height_m",
    multiplier: float = 1.5,
) -> pd.Series:
    """Flag CHM outliers using Tukey's fences."""

def apply_hierarchical_outlier_filter(
    gdf: gpd.GeoDataFrame,
    zscore_flags: pd.Series,
    mahal_flags: pd.Series,
    iqr_flags: pd.Series,
) -> gpd.GeoDataFrame:
    """Remove CRITICAL outliers, flag others."""
```

**Source Module: `splits.py`**

```python
def create_spatial_blocks(
    gdf: gpd.GeoDataFrame,
    block_size_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Create regular grid and assign trees to blocks."""

def create_stratified_splits(
    gdf: gpd.GeoDataFrame,
    train_ratio: float = 0.8,
    random_seed: int = 42,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create spatially disjoint train/val splits."""

def validate_split_stratification(
    train_gdf: gpd.GeoDataFrame,
    val_gdf: gpd.GeoDataFrame,
) -> dict[str, Any]:
    """Validate genus proportions and spatial disjointness."""
```

---

### 4.4 Exploratory Notebooks

These notebooks analyze data and produce configuration files for the runners.

#### exp_01_temporal_analysis.ipynb

**Purpose:** Determine optimal months using JM distance analysis

**Key Tasks:**
- Calculate univariate JM distance per feature × month × genus pair
- Aggregate JM scores across genus pairs
- Visualize monthly discriminability patterns
- Compare across cities for consistency
- **Fix JM implementation** (values too low in legacy)
- Output: `temporal_selection.json` with selected months

**Known Issue:** Legacy JM values were too low. Validate against:
- Bruzzone et al. (1995) reference implementation
- Synthetic test case with known separability

#### exp_02_chm_assessment.ipynb

**Purpose:** Evaluate CHM feature quality and transfer risk

**Key Tasks:**
- ANOVA η² for discriminative power
- Cohen's d for cross-city consistency
- Validate 1m CHM extraction quality
- Design engineered features (Z-score, percentile)
- Output: `chm_assessment.json` with feature decisions

#### exp_03_correlation_analysis.ipynb

**Purpose:** Identify redundant features within spectral groups

**Key Tasks:**
- Classify features into groups (bands, broadband VI, red-edge VI, water VI)
- Calculate intra-class correlation matrices
- Apply threshold (|r| > 0.95) with priority rules
- Validate temporal consistency (phenological patterns)
- Output: `correlation_removal.json` with features to drop

#### exp_04_outlier_thresholds.ipynb

**Purpose:** Determine and validate outlier detection thresholds

**Key Tasks:**
- Sensitivity analysis for Z-score threshold
- Validate Mahalanobis α level
- IQR multiplier evaluation
- Document statistical justification for thresholds
- Output: `outlier_thresholds.json` with validated parameters

---

## 5. Critical Fixes from Legacy

### 5.1 Data Leakage in NaN Imputation (MUST FIX)

**Problem:** Legacy computed genus/city means on full dataset including test set.

**Challenge:** We can't use training-set means because the train/val split happens at the end (02c), after NaN handling (02b).

**Solution:** Use **within-tree interpolation only** - no cross-tree statistics:
1. Linear temporal interpolation for interior NaNs (month 4 missing? interpolate from months 3 and 5)
2. Edge NaNs (first/last selected month): forward/backward fill from adjacent month
3. If edge still NaN: use tree's own mean for that feature across other months
4. Trees with remaining NaNs → remove (these are unfixable without cross-tree leakage)

**Implementation:**
```python
def interpolate_within_tree(row: pd.Series, feature_base: str, months: list[int]) -> pd.Series:
    """Interpolate one tree's temporal series without cross-tree info."""
    values = [row[f"{feature_base}_{m:02d}"] for m in months]
    series = pd.Series(values, index=months)

    # Step 1: Linear interpolation for interior NaNs
    series = series.interpolate(method='linear', limit_direction='both')

    # Step 2: Forward/backward fill for edges
    series = series.ffill().bfill()

    # Step 3: If still NaN, use tree's own mean (rare edge case)
    if series.isna().any():
        series = series.fillna(series.mean())

    return series
```

**Why this works:** Each tree is interpolated independently using only its own temporal data. No information from other trees (genus means, city means) is used, so there's no leakage regardless of train/val split.

### 5.2 CHM Handling (SIMPLIFIED)

**Problem:** Legacy used 10m resampled CHM which caused neighbor contamination (r=0.638).

**Solution:**
- Extract directly from 1m CHM at tree point (no resampling)
- Keep only: `CHM_1m`, `height_m_zscore`, `height_m_percentile`
- Remove: `CHM_mean`, `CHM_max`, `CHM_std`, `crown_ratio` (all contaminated)

### 5.3 JM Distance Implementation (NEEDS FIXING)

**Problem:** Legacy JM values were consistently low (0.5-1.2) even for clearly separable genera.

**Investigation Tasks:**
1. Validate formula against Bruzzone et al. (1995)
2. Test with synthetic data (known μ, σ differences)
3. Check numerical stability (log operations, small variances)
4. Consider multivariate JM as alternative

**Note:** This is exploratory work - the runner will use whatever months the exploratory notebook determines, even if JM implementation isn't perfect.

### 5.4 Remove crown_ratio Feature

**Problem:** Derived from contaminated CHM_mean.

**Solution:** Simply don't compute or include it. Not in feature config.

---

## 6. Validation Criteria

### 6.1 Feature Extraction (02a)
- [ ] All trees have CHM_1m value (no NaN from extraction)
- [ ] Feature count matches config (1 CHM + 23 bands × 12 months = 277)
- [ ] CRS is EPSG:25833
- [ ] Metadata columns preserved

### 6.2 Data Quality (02b)
- [ ] 0 NaN values in output
- [ ] Temporal reduction applied correctly (per temporal_selection.json)
- [ ] No data leakage: imputation stats from training only
- [ ] NDVI plausibility: no trees with max_NDVI < 0.3

### 6.3 Final Preparation (02c)
- [ ] Redundant features removed per config
- [ ] Outlier removal rate reasonable (~1-3%)
- [ ] All genera meet minimum sample threshold
- [ ] Spatial blocks are disjoint
- [ ] Train/val splits stratified (KL-divergence < 0.01)
- [ ] No spatial overlap between train and val

### 6.4 Overall Pipeline
- [ ] Consistent tree_id tracking through pipeline
- [ ] Metadata JSON files generated at each step
- [ ] Processing can be resumed (skip logic for existing outputs)
- [ ] Both cities processed with identical logic

---

## 7. Dependencies

### 7.1 Phase 1 Outputs Required
- `trees_filtered_viable.gpkg` - Harmonized tree cadastres
- `CHM_1m_{city}.tif` - 1m Canopy Height Models
- `S2_{city}_{year}_{MM}_median.tif` - Monthly Sentinel-2 composites

### 7.2 Python Dependencies (existing)
- geopandas, rasterio, pandas, numpy
- scikit-learn (StratifiedGroupKFold)
- scipy (Mahalanobis, chi-squared)

### 7.3 New Config File
- `configs/features/feature_config.yaml` - Must be created first

---

## 8. Execution Order

### Initial Setup
1. Create `configs/features/feature_config.yaml`
2. Extend `config/loader.py` with feature config functions
3. Implement source modules

### Runner Sequence
```
02a_feature_extraction.ipynb
        │
        ▼
    [PAUSE: Run exploratory notebooks exp_01, exp_02]
        │
        ▼
02b_data_quality.ipynb
        │
        ▼
    [PAUSE: Run exploratory notebooks exp_03, exp_04]
        │
        ▼
02c_final_preparation.ipynb
        │
        ▼
    Ready for Phase 3 (Experiments)
```

### Exploratory Notebooks (can run in parallel after 02a)
- exp_01_temporal_analysis → produces `temporal_selection.json`
- exp_02_chm_assessment → produces `chm_assessment.json`
- exp_03_correlation_analysis → produces `correlation_removal.json`
- exp_04_outlier_thresholds → produces `outlier_thresholds.json`

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| JM implementation issues | Runner uses output regardless; fix in exploratory as time permits |
| Large dataset memory | Batch processing (50k trees), lazy loading rasters |
| Sentinel-2 cloud gaps | Already handled by Phase 1 median composites |
| Cross-city inconsistency | Same config applied to both; validation checks |

---

## 10. Success Metrics

- **Feature completeness:** >95% trees with all features after extraction
- **Data retention:** >85% trees after all filtering steps
- **Clean data:** 0 NaN values in final datasets
- **Split quality:** KL-divergence < 0.01 for genus stratification
- **Reproducibility:** Identical results with same config and seed

---

## 11. References

### Codebase
- Phase 1 runner: `notebooks/runners/01_data_processing.ipynb`
- Existing utilities: `src/urban_tree_transfer/utils/`
- Config patterns: `src/urban_tree_transfer/config/`

### Legacy Documentation
- `legacy/documentation/02_Feature_Engineering/*.md`
- `legacy/notebooks/02_feature_engineering/*.md`
- `legacy/documentation/02_Feature_Engineering/99_Methodische_Verbesserungen.md`

### Literature
- Bruzzone et al. (1995) - JM distance reference
- Tukey (1977) - IQR outlier detection

---

**PRD Status:** Ready for review
