# PRD 002c: Final Preparation

**Parent PRD:** [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md)
**Status:** Draft
**Created:** 2026-01-28
**Last Updated:** 2026-01-28
**Branch:** `feature/phase2-final-preparation`

---

## Goal

Finalize features by removing redundant columns, detecting and filtering outliers, ensuring minimum genus sample sizes, creating spatial blocks, and generating stratified splits with city-specific ratios.

**Split Strategy:**

- **Berlin (Source City):** 70/15/15 (Train/Val/Test) - Full ML pipeline with hold-out test set
- **Leipzig (Target City):** 80/20 (Finetune Pool/Test) - For transfer learning experiments

**Success Criteria:**

- [ ] Redundant features removed per `correlation_removal.json`
- [ ] Outliers detected using Z-score, Mahalanobis, IQR methods
- [ ] Consensus-based filtering: 3/3 methods = remove, 2/3 and 1/3 = metadata flags
- [ ] Outlier removal rate ~1-3% (only 3/3 consensus removed)
- [ ] All genera meet MIN_SAMPLES_PER_GENUS threshold
- [ ] Spatial blocks created (size validated by exp_05)
- [ ] Berlin: Train/Val/Test splits (70/15/15) spatially disjoint
- [ ] Leipzig: Finetune/Test splits (80/20) spatially disjoint
- [ ] Genus stratification: KL-divergence < 0.01 across all splits
- [ ] Separate GeoPackages per city per split

---

## Context

**Read before implementation:**

- [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md) - Overall architecture
- [002b_data_quality.md](002b_data_quality.md) - Input data structure (clean, no NaN)
- `CLAUDE.md` / `AGENT.md` - Project conventions
- `configs/features/feature_config.yaml` - Outlier thresholds, split parameters

**Legacy Code (Inspiration Only):**

- `legacy/notebooks/02_feature_engineering/03d_correlation_analysis.md` - Redundancy removal
- `legacy/notebooks/02_feature_engineering/03e_outlier_detection.md` - Outlier methods
- `legacy/notebooks/02_feature_engineering/04_spatial_splits.md` - Split strategy
- `legacy/documentation/02_Feature_Engineering/06_Correlation_Analysis_Redundancy_Reduction_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/07_Outlier_Detection_Final_Filtering_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/08_Spatial_Splits_Stratification_Methodik.md`

---

## Scope

### In Scope

- Feature redundancy removal (intra-class AND cross-class correlation)
- Multi-method outlier detection (Z-score, Mahalanobis, IQR)
- Consensus-based outlier filtering (3-of-3 = remove, 2-of-3 and 1-of-3 = metadata flags)
- Genus sample size validation
- Spatial block creation (size from exp_05 autocorrelation analysis)
- **Berlin:** Stratified spatial splits (70/15/15 Train/Val/Test)
- **Leipzig:** Stratified spatial splits (80/20 Finetune/Test)
- Per-city, per-split GeoPackage outputs
- Split quality validation

### Out of Scope

- Model training (Phase 3)
- Hyperparameter tuning for outlier thresholds (done in exp_04)
- Cross-city splits (Phase 3 experiments)
- Feature importance analysis (Phase 3)

---

## Implementation

### 1. Notebook Structure

**Template:** [Runner-Notebook Template](../../docs/templates/Notebook_Templates.md#1-runner-notebook-template-phase-2)

**File:** `notebooks/runners/02c_final_preparation.ipynb`

**Anpassungen gegenüber Base Template:**

- **Zelle 3 (Imports):**

  ```python
  from urban_tree_transfer.feature_engineering import (
      remove_redundant_features,  # selection.py
      detect_outliers_zscore,     # outliers.py
      detect_outliers_mahalanobis,
      detect_outliers_iqr,
      create_spatial_blocks,      # splits.py
      create_stratified_splits,
  )
  log = ExecutionLog("02c_final_preparation")
  ```

- **Zelle 4 (Config):**

  ```python
  INPUT_DIR = DRIVE_DIR / "data" / "phase_2_quality"   # From 002b
  OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_splits"   # Final datasets

  # Load all exploratory configs
  config_dir = INPUT_DIR.parent.parent / "outputs" / "phase_2" / "metadata"
  correlation_config = json.load(open(config_dir / "correlation_analysis.json"))
  outlier_config = json.load(open(config_dir / "outlier_thresholds.json"))
  spatial_config = json.load(open(config_dir / "spatial_autocorrelation.json"))
  ```

- **Processing Sections:**
  1. Redundancy removal (correlation-based)
  2. Hierarchical outlier detection
  3. Spatial block creation
  4. Stratified train/val splits

- **Validation:**
  - `features_final.parquet` (final feature set)
  - `train_berlin.parquet` / `val_berlin.parquet`
  - `train_leipzig.parquet` / `val_leipzig.parquet`

**Complete template documentation:** [Notebook_Templates.md](../../docs/templates/Notebook_Templates.md)

---

### 2. Feature Selection Module

**File:** `src/urban_tree_transfer/feature_engineering/selection.py`

#### 2.0 Correlation Priority Rules

**CRITICAL:** All correlation-based feature removal decisions follow explicit priority rules. These rules are applied in exp_03_correlation_analysis and produce the `features_to_remove` list.

**Feature Groups:**

1. **Spectral Bands** (10): B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
2. **Broadband VIs** (5): NDVI, EVI, GNDVI, kNDVI, VARI
3. **Red-Edge VIs** (5): NDre1, NDVIre, CIre, IRECI, RTVIcore
4. **Water/Moisture VIs** (3): NDWI, MSI, NDII

**Priority Rules (applied in order):**

**Rule 1: Spatial Resolution (Bands only)**

```
Prefer 10m bands over 20m bands
- 10m: B2, B3, B4, B8
- 20m: B5, B6, B7, B8A, B11, B12
Example: If B8 and B8A correlate at r>0.95 → keep B8 (10m)
```

**Rule 2: Vegetation Index Priority (within each VI group)**

```yaml
broadband_vi_priority: # Most interpretable → least
  1: NDVI # Universal standard, most literature support
  2: EVI # Atmospherically corrected, saturates less
  3: GNDVI # Green-band variant
  4: VARI # Visible-only, atmospheric resistant
  5: kNDVI # Non-linear, harder to interpret

red_edge_vi_priority:
  1: NDre1 # Core red-edge normalized difference
  2: CIre # Chlorophyll index, direct biophysical meaning
  3: IRECI # Inverted red-edge index
  4: NDVIre # Red-edge NDVI variant
  5: RTVIcore # Combines multiple bands, complex

water_vi_priority:
  1: NDWI # Standard water/moisture index (Gao 1996)
  2: NDII # Infrared index (Hunt & Rock 1989)
  3: MSI # Moisture stress index (Rock et al. 1986)
```

**Rule 3: Cross-Class Correlation (bands vs. VIs)**

```
When a band correlates with a VI at |r| > 0.95:
- Always keep the VI (derived feature with biological meaning)
- Remove the band (raw reflectance, less interpretable)
Rationale: VIs encode domain knowledge about vegetation properties
```

**Rule 4: Tiebreaker**

```
If same priority after Rules 1-3:
- Keep feature with higher variance (more discriminative power)
- If variance within 5%: keep alphabetically first
```

**Configuration (in feature_config.yaml):**

```yaml
correlation:
  threshold: 0.95
  priority_rules:
    spatial_resolution:
      bands_10m: [B2, B3, B4, B8]
      bands_20m: [B5, B6, B7, B8A, B11, B12]
    vegetation_index_priority:
      broadband: [NDVI, EVI, GNDVI, VARI, kNDVI]
      red_edge: [NDre1, CIre, IRECI, NDVIre, RTVIcore]
      water: [NDWI, NDII, MSI]
    cross_class_rule: "prefer_vi_over_band"
    tiebreaker: "higher_variance_then_alphabetical"
```

**Functions to implement:**

#### 2.1 Redundancy Removal

```python
def remove_redundant_features(
    gdf: gpd.GeoDataFrame,
    features_to_remove: list[str],
) -> gpd.GeoDataFrame:
    """Remove redundant features based on exploratory correlation analysis.

    Args:
        gdf: Input trees with clean features
        features_to_remove: List of feature names to drop (from correlation_removal.json)

    Note: The feature selection logic (priority rules, threshold application)
    is implemented in exp_03. This function simply applies the removal list.

    Returns:
        GeoDataFrame with redundant columns removed

    Note: The correlation analysis and feature selection logic is in exp_03.
    This function simply applies the removal list.
    """
```

---

### 2. Outlier Detection Module

**File:** `src/urban_tree_transfer/feature_engineering/outliers.py`

**Functions to implement:**

#### 2.1 Z-Score Outliers

```python
def detect_zscore_outliers(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    z_threshold: float = 3.0,
    min_feature_count: int = 10,
) -> pd.Series:
    """Flag trees with many extreme Z-scores.

    Args:
        gdf: Input trees
        feature_columns: Features to check
        z_threshold: Z-score absolute value threshold
        min_feature_count: Minimum features exceeding threshold to flag tree

    Returns:
        Boolean Series: True for outlier trees

    Algorithm:
        1. Compute Z-score per feature (per city)
        2. Count features with |Z| > z_threshold per tree
        3. Flag tree if count ≥ min_feature_count

    Rationale: A tree may have one extreme value by chance, but many
    extreme values suggests measurement error or misclassification.
    """
```

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/03e_outlier_detection.md` contains Z-score implementation.

#### 2.2 Mahalanobis Distance Outliers

```python
def detect_mahalanobis_outliers(
    gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
    alpha: float = 0.001,
) -> pd.Series:
    """Flag multivariate outliers per genus using Mahalanobis distance.

    Args:
        gdf: Input trees
        feature_columns: Features for multivariate analysis
        alpha: Significance level for chi-squared test

    Returns:
        Boolean Series: True for outlier trees

    Algorithm:
        1. Group by genus_latin
        2. Compute covariance matrix and mean vector per genus
        3. Calculate Mahalanobis distance for each tree
        4. Chi-squared test: critical value at alpha significance
        5. Flag if distance > critical value

    Rationale: Captures trees that are multivariate outliers even if
    individual features are not extreme. Detects unusual feature combinations.

    Note: Requires genus groups with sufficient samples for stable covariance.
    """
```

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/03e_outlier_detection.md` contains Mahalanobis implementation.

#### 2.3 IQR Outliers (CHM)

```python
def detect_iqr_outliers(
    gdf: gpd.GeoDataFrame,
    height_column: str = "CHM_1m",
    multiplier: float = 1.5,
) -> pd.Series:
    """Flag CHM outliers using Tukey's fences (per genus × city).

    Args:
        gdf: Input trees
        height_column: CHM column to check
        multiplier: IQR multiplier (1.5 = Tukey's standard)

    Returns:
        Boolean Series: True for outlier trees

    Algorithm (per genus × city):
        1. Compute Q1, Q3, IQR
        2. Lower fence = Q1 - multiplier * IQR
        3. Upper fence = Q3 + multiplier * IQR
        4. Flag if height < lower OR height > upper

    Rationale: CHM errors (e.g., building/terrain contamination) cause
    extreme height values. IQR robust to distribution assumptions.

    Reference: Tukey (1977), Exploratory Data Analysis
    """
```

#### 2.4 Consensus-Based Outlier Filtering

```python
def apply_consensus_outlier_filter(
    gdf: gpd.GeoDataFrame,
    zscore_flags: pd.Series,
    mahal_flags: pd.Series,
    iqr_flags: pd.Series,
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Remove outliers based on method consensus, add metadata flags.

    Args:
        gdf: Input trees
        zscore_flags: Z-score outlier flags
        mahal_flags: Mahalanobis outlier flags
        iqr_flags: IQR outlier flags

    Returns:
        tuple: (filtered_gdf, removal_stats)

    Consensus-Based Decision Rules:
        | Methods Flagging | Label              | Action                    |
        |------------------|--------------------|--------------------------|
        | 3/3              | outlier_critical   | REMOVE                   |
        | 2/3              | outlier_high       | Keep + metadata flag     |
        | 1/3              | outlier_low        | Keep + metadata flag     |
        | 0/3              | (none)             | Keep                     |

    Rationale:
        - Consensus reduces false positives: true outliers should be detected
          by multiple independent methods
        - Mahalanobis alone is sensitive to covariance estimation instability
        - Metadata flags enable sensitivity analysis in Phase 3:
          "What happens if we also remove outlier_high?"
        - Conservative approach appropriate for scientific study

    New Metadata Columns Added:
        - outlier_critical: bool (3/3 methods)
        - outlier_high: bool (2/3 methods)
        - outlier_low: bool (1/3 methods)
        - outlier_method_count: int (0-3)

    Output stats:
        - critical_count: Trees removed (3/3)
        - high_count: Trees flagged as high-risk (2/3)
        - low_count: Trees flagged as low-risk (1/3)
        - per_method_counts: Individual method counts
        - overlap_matrix: Method overlap statistics
    """
```

---

### 3. Spatial Splits Module

**File:** `src/urban_tree_transfer/feature_engineering/splits.py`

**Functions to implement:**

#### 3.1 Spatial Block Creation

```python
def create_spatial_blocks(
    gdf: gpd.GeoDataFrame,
    block_size_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Create regular spatial grid and assign trees to blocks.

    Args:
        gdf: Input trees (must have geometry in projected CRS)
        block_size_m: Block size in meters (from exp_05 analysis)

    Returns:
        GeoDataFrame with new column 'block_id' (str)

    Algorithm:
        1. Compute bounding box per city
        2. Create regular grid with block_size_m spacing
        3. Assign each tree to nearest block center
        4. block_id format: "{city}_{grid_x}_{grid_y}"

    Note: Block size determined by spatial autocorrelation analysis in exp_05.
    Blocks must be large enough to avoid correlation between train/val sets.
    """
```

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/04_spatial_splits.md` contains spatial block logic.

#### 3.2 Stratified Spatial Splits

```python
def create_stratified_splits_berlin(
    gdf: gpd.GeoDataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create spatially disjoint Train/Val/Test splits for Berlin (source city).

    Args:
        gdf: Input trees with block_id and genus_latin columns
        train_ratio: Proportion for training set (default 0.70)
        val_ratio: Proportion for validation set (default 0.15)
        test_ratio: Proportion for test set (default 0.15)
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (train_gdf, val_gdf, test_gdf)

    Algorithm:
        1. Use StratifiedGroupKFold from sklearn (2 iterations)
        2. First split: 70% train vs 30% held-out
        3. Second split: 50/50 of held-out → 15% val, 15% test
        4. Stratify by genus_latin (maintain genus proportions)
        5. Group by block_id (entire blocks in one split, not divided)

    Purpose:
        - Train: Model training
        - Val: Hyperparameter tuning, early stopping, model selection
        - Test: Final unbiased single-city performance evaluation (NEVER for tuning!)

    Critical: Spatial grouping prevents data leakage via autocorrelation.
    """


def create_stratified_splits_leipzig(
    gdf: gpd.GeoDataFrame,
    finetune_ratio: float = 0.80,
    test_ratio: float = 0.20,
    random_seed: int = 42,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create spatially disjoint Finetune/Test splits for Leipzig (target city).

    Args:
        gdf: Input trees with block_id and genus_latin columns
        finetune_ratio: Proportion for fine-tuning pool (default 0.80)
        test_ratio: Proportion for test set (default 0.20)
        random_seed: Random seed for reproducibility

    Returns:
        tuple: (finetune_gdf, test_gdf)

    Algorithm:
        1. Use StratifiedGroupKFold from sklearn
        2. Stratify by genus_latin (maintain genus proportions)
        3. Group by block_id (entire blocks in one split)
        4. Single fold with 80/20 split

    Purpose:
        - Finetune Pool: For fine-tuning experiments with varying data amounts
          (can be sub-sampled: 10%, 25%, 50%, 100% for fine-tuning curves)
          If validation needed during fine-tuning: nested split from pool
        - Test: Cross-city zero-shot evaluation AND post-fine-tuning evaluation
          (MUST remain untouched for fair comparisons!)

    Critical: Spatial grouping prevents data leakage via autocorrelation.
    """
```

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/04_spatial_splits.md` and `legacy/documentation/02_Feature_Engineering/08_Spatial_Splits_Stratification_Methodik.md` contain split methodology.

#### 3.3 Split Validation

```python
def validate_split_stratification(
    *split_gdfs: gpd.GeoDataFrame,
    split_names: list[str] = None,
) -> dict[str, Any]:
    """Validate split quality and spatial disjointness for multiple splits.

    Args:
        *split_gdfs: Variable number of split GeoDataFrames
        split_names: Names for each split (e.g., ['train', 'val', 'test'])

    Returns:
        Dictionary with validation metrics:
        - genus_distributions: Genus counts per split
        - kl_divergences: Pairwise KL-divergence between splits
        - spatial_overlap: Check for shared blocks (should be 0)
        - block_counts: Blocks per split
        - sample_sizes: Total trees per split

    Acceptance criteria:
        - All pairwise kl_divergence < 0.01 (distributions match)
        - spatial_overlap == 0 (no shared blocks between ANY splits)
        - All genera represented in all splits
    """
```

---

### 4. Runner Notebook

**File:** `notebooks/runners/02c_final_preparation.ipynb`

**Structure:**

#### Cell 1: Setup & Imports

```python
from pathlib import Path
import json
import pandas as pd
import geopandas as gpd
from sklearn.model_selection import StratifiedGroupKFold

from urban_tree_transfer.config.loader import load_feature_config
from urban_tree_transfer.feature_engineering.selection import remove_redundant_features
from urban_tree_transfer.feature_engineering.outliers import (
    detect_zscore_outliers,
    detect_mahalanobis_outliers,
    detect_iqr_outliers,
    apply_consensus_outlier_filter,
)
from urban_tree_transfer.feature_engineering.splits import (
    create_spatial_blocks,
    create_stratified_splits_berlin,
    create_stratified_splits_leipzig,
    validate_split_stratification,
)
from urban_tree_transfer.config.constants import RANDOM_SEED
```

#### Cell 2: Configuration & Load Data

```python
# Paths
DATA_DIR = Path("data/phase_2_features")
OUTPUT_DIR = Path("outputs/phase_2/metadata")
FINAL_DIR = DATA_DIR / "final"
FINAL_DIR.mkdir(exist_ok=True)

# Load configs
feature_config = load_feature_config()

# Load correlation removal config (from exp_03)
with open(OUTPUT_DIR / "correlation_removal.json") as f:
    correlation_config = json.load(f)
features_to_remove = correlation_config["features_to_remove"]

# Load outlier thresholds (from exp_04)
with open(OUTPUT_DIR / "outlier_thresholds.json") as f:
    outlier_config = json.load(f)

# Load clean trees
cities = ["berlin", "leipzig"]
trees_dict = {}
for city in cities:
    trees_dict[city] = gpd.read_file(DATA_DIR / f"trees_clean_{city}.gpkg")

trees_gdf = pd.concat(trees_dict.values(), ignore_index=True)
print(f"Loaded {len(trees_gdf)} clean trees")
```

#### Cell 3: Feature Redundancy Removal

```python
print("\n" + "="*60)
print("STEP 1: Remove Redundant Features")
print("="*60)

print(f"Features to remove: {len(features_to_remove)}")
print(features_to_remove)

trees_gdf = remove_redundant_features(trees_gdf, features_to_remove)

feature_cols = [col for col in trees_gdf.columns
                if col not in feature_config["metadata_columns"] + ["geometry"]]
print(f"Remaining features: {len(feature_cols)}")
```

#### Cell 4: Outlier Detection

```python
print("\n" + "="*60)
print("STEP 2: Outlier Detection")
print("="*60)

# Z-score outliers
zscore_flags = detect_zscore_outliers(
    trees_gdf,
    feature_cols,
    z_threshold=outlier_config["zscore"]["threshold"],
    min_feature_count=outlier_config["zscore"]["min_feature_count"],
)
print(f"Z-score outliers: {zscore_flags.sum()} ({zscore_flags.sum()/len(trees_gdf)*100:.2f}%)")

# Mahalanobis outliers
mahal_flags = detect_mahalanobis_outliers(
    trees_gdf,
    feature_cols,
    alpha=outlier_config["mahalanobis"]["alpha"],
)
print(f"Mahalanobis outliers: {mahal_flags.sum()} ({mahal_flags.sum()/len(trees_gdf)*100:.2f}%)")

# IQR outliers (CHM)
iqr_flags = detect_iqr_outliers(
    trees_gdf,
    height_column="CHM_1m",
    multiplier=outlier_config["iqr"]["multiplier"],
)
print(f"IQR outliers: {iqr_flags.sum()} ({iqr_flags.sum()/len(trees_gdf)*100:.2f}%)")
```

#### Cell 5: Consensus-Based Outlier Filtering

```python
print("\n" + "="*60)
print("STEP 3: Consensus-Based Outlier Filtering")
print("="*60)

trees_before = len(trees_gdf)
trees_gdf, outlier_stats = apply_consensus_outlier_filter(
    trees_gdf, zscore_flags, mahal_flags, iqr_flags
)

print(f"\nConsensus Results:")
print(f"  CRITICAL (3/3 methods) - REMOVED: {outlier_stats['critical_count']}")
print(f"  HIGH (2/3 methods) - flagged: {outlier_stats['high_count']}")
print(f"  LOW (1/3 methods) - flagged: {outlier_stats['low_count']}")
print(f"\nRemoval rate: {outlier_stats['critical_count']/trees_before*100:.2f}%")
print(f"Retained: {len(trees_gdf)} trees (with metadata flags for sensitivity analysis)")
```

#### Cell 6: Genus Sample Size Validation

```python
print("\n" + "="*60)
print("STEP 4: Genus Sample Size Validation")
print("="*60)

min_samples = feature_config["splits"]["min_samples_per_genus"]
genus_counts = trees_gdf.groupby(["city", "genus_latin"]).size()

print(f"Minimum required samples: {min_samples}")
print("\nGenus counts per city:")
print(genus_counts)

# Remove non-viable genera
viable_genera = genus_counts[genus_counts >= min_samples].index
trees_gdf = trees_gdf[
    trees_gdf.set_index(["city", "genus_latin"]).index.isin(viable_genera)
].reset_index(drop=True)

print(f"\nRetained {len(trees_gdf)} trees from viable genera")
```

#### Cell 7: Spatial Block Creation

```python
print("\n" + "="*60)
print("STEP 5: Create Spatial Blocks")
print("="*60)

block_size = feature_config["splits"]["block_size_m"]
print(f"Block size: {block_size}m (validated in exp_05)")

trees_gdf = create_spatial_blocks(trees_gdf, block_size_m=block_size)

print(f"Blocks created: {trees_gdf['block_id'].nunique()}")
print(f"Trees per block (mean): {len(trees_gdf) / trees_gdf['block_id'].nunique():.1f}")
```

#### Cell 8: Create Stratified Splits (city-specific)

```python
print("\n" + "="*60)
print("STEP 6: Create Stratified Splits")
print("="*60)

# ===== BERLIN: 70/15/15 Train/Val/Test =====
print("\nBERLIN (Source City): 70/15/15 Train/Val/Test")
berlin_trees = trees_gdf[trees_gdf["city"] == "berlin"].copy()

train_gdf, val_gdf, test_gdf = create_stratified_splits_berlin(
    berlin_trees,
    train_ratio=0.70,
    val_ratio=0.15,
    test_ratio=0.15,
    random_seed=RANDOM_SEED,
)

print(f"  Train: {len(train_gdf)} trees ({len(train_gdf)/len(berlin_trees)*100:.1f}%)")
print(f"  Val:   {len(val_gdf)} trees ({len(val_gdf)/len(berlin_trees)*100:.1f}%)")
print(f"  Test:  {len(test_gdf)} trees ({len(test_gdf)/len(berlin_trees)*100:.1f}%)")

# Validate Berlin splits
validation = validate_split_stratification(
    train_gdf, val_gdf, test_gdf,
    split_names=["train", "val", "test"]
)
print(f"  Max KL-divergence: {max(validation['kl_divergences'].values()):.6f}")
print(f"  Spatial overlap: {validation['spatial_overlap']} blocks")

assert max(validation['kl_divergences'].values()) < 0.01, "Stratification failed"
assert validation["spatial_overlap"] == 0, "Spatial overlap detected"

# Save Berlin splits
for split_name, split_gdf in [("train", train_gdf), ("val", val_gdf), ("test", test_gdf)]:
    path = FINAL_DIR / f"berlin_{split_name}.gpkg"
    split_gdf.to_file(path, driver="GPKG")
    print(f"  Saved: {path}")

# ===== LEIPZIG: 80/20 Finetune/Test =====
print("\nLEIPZIG (Target City): 80/20 Finetune/Test")
leipzig_trees = trees_gdf[trees_gdf["city"] == "leipzig"].copy()

finetune_gdf, test_gdf = create_stratified_splits_leipzig(
    leipzig_trees,
    finetune_ratio=0.80,
    test_ratio=0.20,
    random_seed=RANDOM_SEED,
)

print(f"  Finetune Pool: {len(finetune_gdf)} trees ({len(finetune_gdf)/len(leipzig_trees)*100:.1f}%)")
print(f"  Test:          {len(test_gdf)} trees ({len(test_gdf)/len(leipzig_trees)*100:.1f}%)")

# Validate Leipzig splits
validation = validate_split_stratification(
    finetune_gdf, test_gdf,
    split_names=["finetune", "test"]
)
print(f"  KL-divergence: {validation['kl_divergences']['finetune_vs_test']:.6f}")
print(f"  Spatial overlap: {validation['spatial_overlap']} blocks")

assert validation['kl_divergences']['finetune_vs_test'] < 0.01, "Stratification failed"
assert validation["spatial_overlap"] == 0, "Spatial overlap detected"

# Save Leipzig splits
for split_name, split_gdf in [("finetune", finetune_gdf), ("test", test_gdf)]:
    path = FINAL_DIR / f"leipzig_{split_name}.gpkg"
    split_gdf.to_file(path, driver="GPKG")
    print(f"  Saved: {path}")
```

#### Cell 9: Final Summary

```python
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

summary = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "feature_count": len(feature_cols),
    "removed_redundant": len(features_to_remove),
    "outlier_removal": outlier_stats,
    "splits": {
        "berlin": {
            "strategy": "70/15/15 Train/Val/Test",
            "purpose": "Source city for single-city optimization",
        },
        "leipzig": {
            "strategy": "80/20 Finetune/Test",
            "purpose": "Target city for transfer learning",
        },
    },
}

# Berlin stats
for split_name in ["train", "val", "test"]:
    split_gdf = gpd.read_file(FINAL_DIR / f"berlin_{split_name}.gpkg")
    summary["splits"]["berlin"][split_name] = {
        "samples": len(split_gdf),
        "blocks": split_gdf["block_id"].nunique(),
        "genera": sorted(split_gdf["genus_latin"].unique()),
    }

# Leipzig stats
for split_name in ["finetune", "test"]:
    split_gdf = gpd.read_file(FINAL_DIR / f"leipzig_{split_name}.gpkg")
    summary["splits"]["leipzig"][split_name] = {
        "samples": len(split_gdf),
        "blocks": split_gdf["block_id"].nunique(),
        "genera": sorted(split_gdf["genus_latin"].unique()),
    }

# Save summary
with open(OUTPUT_DIR / "final_dataset_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nSummary saved: {OUTPUT_DIR / 'final_dataset_summary.json'}")
print("\n" + "="*60)
print("✓ Phase 2 Feature Engineering COMPLETE")
print("✓ Ready for Phase 3 Experiments")
print("="*60)
print("\nOutput files:")
print("  Berlin:  berlin_train.gpkg, berlin_val.gpkg, berlin_test.gpkg")
print("  Leipzig: leipzig_finetune.gpkg, leipzig_test.gpkg")
```

---

## Testing

**File:** `tests/feature_engineering/test_splits.py`

**Test cases:**

- Spatial block creation with various grid sizes
- Stratified splits with known genus distributions
- Split validation metrics (KL-divergence, overlap)
- Edge cases (small genera, uneven spatial distribution)

**File:** `tests/feature_engineering/test_outliers.py`

**Test cases:**

- Z-score detection with synthetic extreme values
- Mahalanobis distance with known covariance
- IQR outliers with controlled distributions
- Consensus-based filtering decision logic (3-of-3 remove, 2-of-3 and 1-of-3 flag)

---

## Validation Checklist

- [ ] Redundant features removed per `correlation_removal.json` (intra- and cross-class)
- [ ] Correlation priority rules correctly applied (spatial resolution, VI priority, cross-class)
- [ ] Consensus-based outlier filtering: 3/3 = removed, 2/3 = flagged, 1/3 = flagged
- [ ] Outlier metadata columns added: `outlier_critical`, `outlier_high`, `outlier_low`
- [ ] Outlier removal rate ~0.5-1.5% (only 3/3 consensus)
- [ ] All genera meet MIN_SAMPLES_PER_GENUS
- [ ] Spatial blocks created per `spatial_autocorrelation.json` block size
- [ ] **Berlin:** Train/Val/Test (70/15/15) spatially disjoint, no shared blocks
- [ ] **Leipzig:** Finetune/Test (80/20) spatially disjoint, no shared blocks
- [ ] KL-divergence < 0.01 for genus stratification across all splits
- [ ] CRS is EPSG:25833 in all outputs
- [ ] All 5 GeoPackages generated (3 Berlin + 2 Leipzig)

---

## Outputs

**Data (Berlin - Source City):**

- `data/phase_2_features/final/berlin_train.gpkg` (70%) - Model training
- `data/phase_2_features/final/berlin_val.gpkg` (15%) - HP tuning, early stopping
- `data/phase_2_features/final/berlin_test.gpkg` (15%) - Final single-city evaluation

**Data (Leipzig - Target City):**

- `data/phase_2_features/final/leipzig_finetune.gpkg` (80%) - Fine-tuning pool
- `data/phase_2_features/final/leipzig_test.gpkg` (20%) - Cross-city & post-finetune evaluation

**Metadata:**

- `outputs/phase_2/metadata/final_dataset_summary.json`

---

## Next Steps

Phase 2 Feature Engineering is complete. Proceed to:

1. **Phase 3 Experiments** - Model training, transfer learning evaluation
2. Review PRDs in `PRDs/003_phase3/` (to be created)

---

**Status:** Ready for implementation
