# PRD 002b: Data Quality

**Parent PRD:** [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md)
**Status:** Draft
**Created:** 2026-01-28
**Last Updated:** 2026-01-28
**Branch:** `feature/phase2-data-quality`

---

## Goal

Clean extracted features by applying temporal reduction, handling missing values without data leakage, engineering CHM features, and filtering implausible observations.

**Success Criteria:**

- [ ] 0 NaN values in output dataset
- [ ] Temporal reduction applied per `temporal_selection.json`
- [ ] NaN imputation uses within-tree interpolation only (no leakage)
- [ ] Edge-NaN rule: 1 month = fill, ≥2 months = remove tree
- [ ] CHM engineered features computed (Z-score, percentile)
- [ ] NDVI plausibility filter applied (max_NDVI ≥ 0.3)
- [ ] Plant year filter applied per `chm_assessment.json` (statistically determined)
- [ ] Genus filtering applied per `chm_assessment.json` (deciduous only if conifer threshold not met)
- [ ] > 85% data retention from input

---

## Context

**Read before implementation:**

- [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md) - Critical fixes for data leakage
- [002a_feature_extraction.md](002a_feature_extraction.md) - Input data structure
- `CLAUDE.md` / `AGENT.md` - Project conventions
- `configs/features/feature_config.yaml` - Quality thresholds

**Legacy Code (Inspiration Only):**

- `legacy/notebooks/02_feature_engineering/02_data_quality_control.md` - Original quality checks
- `legacy/notebooks/02_feature_engineering/03b_nan_handling_plausibility.md` - NaN handling (⚠️ contains data leakage)
- `legacy/documentation/02_Feature_Engineering/02_Data_Quality_Control_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/04_NaN_Handling_Plausibility_Methodik.md`

**⚠️ CRITICAL:** Legacy used genus/city means for NaN imputation, causing data leakage. This PRD implements within-tree interpolation only.

---

## Scope

### In Scope

- Genus filtering (deciduous only)
- Plant year filtering (≤2018 for satellite visibility)
- Temporal reduction (apply exploratory month selection)
- NaN analysis and filtering
- Within-tree temporal interpolation (no cross-tree stats)
- CHM feature engineering (Z-score, percentile per city)
- NDVI plausibility filtering
- Per-city clean dataset output

### Out of Scope

- Feature redundancy removal (PRD 002c)
- Outlier detection (PRD 002c)
- Spatial splits (PRD 002c)
- Temporal feature selection analysis (exploratory notebook exp_01)

---

## Implementation

### 1. Notebook Structure

**Template:** [Runner-Notebook Template](../../docs/templates/Notebook_Templates.md#1-runner-notebook-template-phase-2)

**File:** `notebooks/runners/02b_data_quality.ipynb`

**Anpassungen gegenüber Base Template:**

- **Zelle 3 (Imports):**

  ```python
  from urban_tree_transfer.feature_engineering.quality import (
      filter_deciduous_genera,
      interpolate_features_within_tree,
      apply_plausibility_filters,
      compute_chm_engineered_features,
  )
  log = ExecutionLog("02b_data_quality")
  ```

- **Zelle 4 (Config):**

  ```python
  INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"  # From 002a
  OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_quality"  # Cleaned data

  # Load exploratory config (from exp_01)
  temporal_config_path = INPUT_DIR.parent.parent / "outputs" / "phase_2" / "metadata" / "temporal_selection.json"
  with open(temporal_config_path) as f:
      temporal_config = json.load(f)
  ```

- **Processing Sections:**
  1. Load temporal selection config (JSON from exp_01)
  2. Filter deciduous genera
  3. Within-tree temporal interpolation (NO cross-tree statistics)
  4. Plausibility filters
  5. CHM engineered features

- **Validation:**
  - `features_cleaned.parquet` (schema + NaN counts + plausibility checks)

**Complete template documentation:** [Notebook_Templates.md](../../docs/templates/Notebook_Templates.md)

---

### 2. Quality Control Module

**File:** `src/urban_tree_transfer/feature_engineering/quality.py`

**Functions to implement:**

#### 2.1 Genus Filtering

```python
def filter_deciduous_genera(
    gdf: gpd.GeoDataFrame,
    deciduous_genera: list[str],
) -> gpd.GeoDataFrame:
    """Filter to deciduous genera only for spectral homogeneity.

    Args:
        gdf: Input trees with genus_latin column
        deciduous_genera: List of deciduous genus names (from config)

    Returns:
        Filtered GeoDataFrame

    Rationale: Coniferous trees have different spectral signatures,
    mixing them reduces model performance. Phase focuses on deciduous only.
    """
```

#### 1.2 Plant Year Filtering

```python
def filter_by_plant_year(
    gdf: gpd.GeoDataFrame,
    max_year: int,
    reference_year: int = 2021,
) -> gpd.GeoDataFrame:
    """Filter trees by planting year to exclude recently planted trees.

    Args:
        max_year: Maximum planting year (from chm_assessment.json, statistically determined)
        reference_year: Sentinel-2 imagery year for context

    Returns:
        Filtered GeoDataFrame with only established trees

    Rationale: Trees planted after threshold may not have sufficient canopy
    development to be detectable in 10m resolution Sentinel-2 imagery.

    The threshold is NOT hardcoded but determined empirically in exp_02:
    1. CHM vs plant_year regression analysis
    2. Detection threshold ~2m (typical Sentinel-2 visibility)
    3. Breakpoint where median CHM falls below detection threshold

    Note: plant_year column is kept in metadata for potential future analyses.
    """
```

#### 1.3 Temporal Reduction

```python
def apply_temporal_selection(
    gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict[str, Any],
) -> gpd.GeoDataFrame:
    """Remove columns for non-selected months.

    Args:
        gdf: Input trees with all 12 months
        selected_months: Months to keep (from temporal_selection.json)
        feature_config: Loaded feature config

    Returns:
        GeoDataFrame with only selected months

    Implementation:
        - Get all feature names (bands + indices)
        - For each month NOT in selected_months: drop columns
        - Preserve metadata columns
        - Log feature count reduction
    """
```

#### 1.4 NaN Analysis

```python
def analyze_nan_distribution(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute NaN statistics per tree and per feature.

    Args:
        gdf: Input trees
        feature_columns: List of feature column names

    Returns:
        DataFrame with per-feature NaN counts and percentages

    Analysis includes:
        - Total NaN count per feature
        - Percentage of trees affected
        - Distribution of NaN counts per tree (histogram bins)
    """
```

#### 1.5 NaN Filtering

```python
def filter_nan_trees(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    max_nan_months: int = 2,
) -> gpd.GeoDataFrame:
    """Remove trees with too many missing months.

    Args:
        gdf: Input trees
        feature_columns: Feature columns to check
        max_nan_months: Maximum allowed NaN months per feature

    Returns:
        Filtered GeoDataFrame

    Logic:
        - Count NaN months per feature per tree
        - Remove tree if ANY feature has >max_nan_months NaN values
        - This ensures sufficient data for temporal interpolation
    """
```

#### 1.6 Within-Tree Interpolation (CRITICAL - No Leakage)

```python
def interpolate_features_within_tree(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    selected_months: list[int],
    max_edge_nan_months: int = 1,
) -> gpd.GeoDataFrame:
    """Fill NaN values using within-tree temporal information only.

    ⚠️ CRITICAL: No cross-tree information is used to avoid data leakage.

    Args:
        gdf: Input trees (after max_nan_months filtering)
        feature_columns: Feature columns to interpolate
        selected_months: Selected months (for temporal index)
        max_edge_nan_months: Maximum NaN months tolerated at edges (default: 1)

    Returns:
        GeoDataFrame with interpolated values

    Algorithm (per tree, per feature):
        1. Extract temporal series for feature across selected months
        2. Interior NaNs: Linear interpolation between adjacent valid values
        3. Edge NaNs (1 month at start/end): Forward/backward-fill with nearest value
        4. Edge NaNs (≥2 months at start/end): Mark tree for removal
        5. Remaining NaN after interpolation: tree will be removed in validation

    Edge-NaN Tolerance Rationale:
        - 1 month edge NaN: 11/12 months available (92%) - minimal info loss
        - Nearest-neighbor fill minimizes pattern distortion
        - ≥2 months edge NaN: significant phenological data loss
        - Literature: Jönsson & Eklundh (2004) recommend max 1-2 edge gaps

    Why no cross-tree stats:
        - Genus means would leak information between train/val sets
        - City means would leak spatial patterns
        - Each tree must be independent for valid train/test split

    Implementation note:
        Use pandas.Series.interpolate(method='linear') for interior NaNs,
        then pandas.Series.ffill()/bfill() for single edge NaNs.
        Check edge gap size before applying fill.
    """
```

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/03b_nan_handling_plausibility.md` contains NaN handling, but **uses genus means (data leakage)**. Do not copy this approach.

#### 1.7 CHM Feature Engineering

```python
def compute_chm_engineered_features(
    gdf: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Add CHM_1m_zscore and CHM_1m_percentile per city.

    Args:
        gdf: Input trees with CHM_1m column

    Returns:
        GeoDataFrame with new columns:
        - CHM_1m_zscore: Z-score normalized height per city
        - CHM_1m_percentile: Percentile rank (0-100) per city

    Note: CHM is input data (not derived from other trees), so computing
    city-level statistics doesn't cause leakage. These are feature transformations
    of an independent variable (canopy height from LiDAR/stereo).

    Implementation:
        - Group by city
        - Compute mean, std for Z-score
        - Compute percentile rank using scipy.stats.percentileofscore
    """
```

#### 1.8 NDVI Plausibility Filtering

```python
def filter_ndvi_plausibility(
    gdf: gpd.GeoDataFrame,
    ndvi_columns: list[str],
    min_threshold: float = 0.3,
) -> gpd.GeoDataFrame:
    """Remove trees with implausibly low maximum NDVI.

    Args:
        gdf: Input trees
        ndvi_columns: List of NDVI column names (e.g., ['NDVI_04', 'NDVI_05', ...])
        min_threshold: Minimum max_NDVI value (default 0.3)

    Returns:
        Filtered GeoDataFrame

    Rationale: Trees should show healthy vegetation signal in growing season.
    max_NDVI < 0.3 suggests:
        - Misclassified non-vegetation
        - Dead/dying tree
        - Position error (tree not where cadastre indicates)

    Implementation:
        - Compute max_NDVI = max(NDVI_MM) across all months
        - Remove trees with max_NDVI < min_threshold
        - Log removal statistics
    """
```

---

### 2. Runner Notebook

**File:** `notebooks/runners/02b_data_quality.ipynb`

**Structure:**

#### Cell 1: Setup & Imports

```python
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd
from urban_tree_transfer.config.loader import (
    load_feature_config,
    get_vegetation_indices,
)
from urban_tree_transfer.feature_engineering.quality import (
    filter_deciduous_genera,
    filter_by_plant_year,
    apply_temporal_selection,
    analyze_nan_distribution,
    filter_nan_trees,
    interpolate_features_within_tree,
    compute_chm_engineered_features,
    filter_ndvi_plausibility,
)
```

#### Cell 2: Configuration & Load Data

```python
# Paths
DATA_DIR = Path("data/phase_2_features")
OUTPUT_DIR = Path("outputs/phase_2/metadata")

# Load feature config
feature_config = load_feature_config()

# Load temporal selection (from exp_01)
with open(OUTPUT_DIR / "temporal_selection.json") as f:
    temporal_config = json.load(f)
selected_months = temporal_config["selected_months"]

# Load CHM assessment (from exp_02) - contains plant_year threshold and genus classification
with open(OUTPUT_DIR / "chm_assessment.json") as f:
    chm_config = json.load(f)
plant_year_threshold = chm_config["plant_year_analysis"]["recommended_max_plant_year"]
analysis_scope = chm_config["genus_inventory"]["analysis_scope"]  # "deciduous_only" or "all"

# Load trees from PRD 002a outputs
cities = ["berlin", "leipzig"]
trees_dict = {}
for city in cities:
    trees_dict[city] = gpd.read_file(DATA_DIR / f"trees_with_features_{city}.gpkg")

# Concatenate for joint processing
trees_gdf = pd.concat(trees_dict.values(), ignore_index=True)
print(f"Loaded {len(trees_gdf)} trees from {len(cities)} cities")
```

#### Cell 3: Genus Filtering

```python
print("\n" + "="*60)
print("STEP 1: Genus Filtering (Deciduous Only)")
print("="*60)

deciduous_genera = feature_config["genus_filter"]["deciduous_genera"]
trees_gdf = filter_deciduous_genera(trees_gdf, deciduous_genera)

print(f"Retained {len(trees_gdf)} deciduous trees")
print(f"Genera: {sorted(trees_gdf['genus_latin'].unique())}")
```

#### Cell 4: Plant Year Filtering

```python
print("\n" + "="*60)
print("STEP 2: Plant Year Filtering")
print("="*60)

# Use threshold from exp_02 CHM assessment (statistically determined)
print(f"Plant year threshold: {plant_year_threshold} (from chm_assessment.json)")
trees_before = len(trees_gdf)
trees_gdf = filter_by_plant_year(trees_gdf, max_year=plant_year_threshold)

print(f"Removed {trees_before - len(trees_gdf)} recently planted trees")
print(f"Retained {len(trees_gdf)} trees (planted ≤ {plant_year_threshold})")
```

#### Cell 5: Temporal Reduction

```python
print("\n" + "="*60)
print("STEP 3: Temporal Reduction")
print("="*60)

print(f"Selected months: {selected_months}")
trees_gdf = apply_temporal_selection(trees_gdf, selected_months, feature_config)

# Count remaining features
feature_cols = [col for col in trees_gdf.columns
                if any(f"_{m:02d}" in col for m in selected_months)]
print(f"Remaining features: {len(feature_cols)}")
```

#### Cell 6: NaN Analysis

```python
print("\n" + "="*60)
print("STEP 4: NaN Analysis")
print("="*60)

feature_cols = [col for col in trees_gdf.columns
                if any(f"_{m:02d}" in col for m in selected_months)]
nan_stats = analyze_nan_distribution(trees_gdf, feature_cols)

print(nan_stats.head(10))
print(f"\nTotal features with NaN: {(nan_stats['nan_count'] > 0).sum()}")
```

#### Cell 7: NaN Filtering

```python
print("\n" + "="*60)
print("STEP 5: NaN Filtering")
print("="*60)

max_nan_months = feature_config["quality"]["nan_month_threshold"]
trees_before = len(trees_gdf)
trees_gdf = filter_nan_trees(trees_gdf, feature_cols, max_nan_months=max_nan_months)

print(f"Removed {trees_before - len(trees_gdf)} trees (>{max_nan_months} NaN months)")
print(f"Retained {len(trees_gdf)} trees")
```

#### Cell 8: Within-Tree Interpolation (CRITICAL)

```python
print("\n" + "="*60)
print("STEP 6: Within-Tree Temporal Interpolation")
print("="*60)

print("⚠️  Using within-tree interpolation only (no cross-tree leakage)")
trees_gdf = interpolate_features_within_tree(trees_gdf, feature_cols, selected_months)

# Validate no NaN remaining
remaining_nan = trees_gdf[feature_cols].isna().sum().sum()
if remaining_nan > 0:
    print(f"⚠️  WARNING: {remaining_nan} NaN values still present")
    # Remove trees with remaining NaN
    trees_gdf = trees_gdf.dropna(subset=feature_cols)
    print(f"Removed trees with unfixable NaN. Retained: {len(trees_gdf)}")
else:
    print("✓ All NaN values interpolated successfully")
```

#### Cell 9: CHM Feature Engineering

```python
print("\n" + "="*60)
print("STEP 7: CHM Feature Engineering")
print("="*60)

trees_gdf = compute_chm_engineered_features(trees_gdf)
print("Added features:")
print("  - CHM_1m_zscore (per city)")
print("  - CHM_1m_percentile (per city)")
```

#### Cell 10: NDVI Plausibility Filtering

```python
print("\n" + "="*60)
print("STEP 8: NDVI Plausibility Filtering")
print("="*60)

ndvi_columns = [col for col in trees_gdf.columns if col.startswith("NDVI_")]
min_threshold = feature_config["quality"]["ndvi_min_threshold"]

trees_before = len(trees_gdf)
trees_gdf = filter_ndvi_plausibility(trees_gdf, ndvi_columns, min_threshold=min_threshold)

print(f"Removed {trees_before - len(trees_gdf)} trees (max_NDVI < {min_threshold})")
print(f"Retained {len(trees_gdf)} trees")
```

#### Cell 11: Save Clean Datasets

```python
print("\n" + "="*60)
print("STEP 9: Save Per-City Clean Datasets")
print("="*60)

for city in cities:
    city_trees = trees_gdf[trees_gdf["city"] == city].copy()
    output_path = DATA_DIR / f"trees_clean_{city}.gpkg"
    city_trees.to_file(output_path, driver="GPKG")
    print(f"{city}: {len(city_trees)} trees → {output_path}")
```

#### Cell 12: Validation & Summary

```python
print("\n" + "="*60)
print("VALIDATION")
print("="*60)

# Assert 0 NaN
feature_cols = [col for col in trees_gdf.columns
                if any(f"_{m:02d}" in col for m in selected_months)]
assert trees_gdf[feature_cols].isna().sum().sum() == 0, "NaN values found!"
print("✓ 0 NaN values in features")

# Check CRS
assert trees_gdf.crs.to_epsg() == 25833, "Wrong CRS"
print("✓ CRS: EPSG:25833")

# Check engineered features
assert "CHM_1m_zscore" in trees_gdf.columns
assert "CHM_1m_percentile" in trees_gdf.columns
print("✓ CHM engineered features present")

# Summary
summary = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "selected_months": selected_months,
    "total_trees": len(trees_gdf),
    "trees_per_city": trees_gdf["city"].value_counts().to_dict(),
    "genera": sorted(trees_gdf["genus_latin"].unique()),
    "feature_count": len(feature_cols),
    "nan_values": 0,
}

with open(OUTPUT_DIR / "data_quality_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved: {OUTPUT_DIR / 'data_quality_summary.json'}")
```

---

## Testing

**File:** `tests/feature_engineering/test_quality.py`

**Test cases:**

- Genus filtering with mixed deciduous/coniferous
- Plant year filtering with various thresholds
- Temporal reduction with selected months
- NaN analysis with synthetic data
- Within-tree interpolation with known gaps (no leakage)
- CHM Z-score normalization
- NDVI plausibility with edge cases

---

## Validation Checklist

- [ ] 0 NaN values in output
- [ ] Genus filtering applied per `chm_assessment.json` analysis scope
- [ ] Plant year threshold from `chm_assessment.json` (statistically determined)
- [ ] Temporal reduction applied correctly per `temporal_selection.json`
- [ ] Edge-NaN handling: 1 month = nearest fill, ≥2 months = tree removed
- [ ] NaN imputation uses within-tree interpolation only (no leakage)
- [ ] CHM engineered features computed per city
- [ ] NDVI plausibility filter applied (max_NDVI ≥ 0.3)
- [ ] > 85% data retention from PRD 002a input
- [ ] CRS is EPSG:25833

---

## Outputs

**Data:**

- `data/phase_2_features/trees_clean_berlin.gpkg`
- `data/phase_2_features/trees_clean_leipzig.gpkg`

**Metadata:**

- `outputs/phase_2/metadata/data_quality_summary.json`

---

## Next Steps

After completing this PRD:

1. Run exploratory notebooks: `exp_03_correlation_analysis.ipynb`, `exp_04_outlier_thresholds.ipynb`, `exp_05_spatial_autocorrelation.ipynb`
2. Proceed to [PRD 002c: Final Preparation](002c_final_preparation.md)

---

**Status:** Ready for implementation
