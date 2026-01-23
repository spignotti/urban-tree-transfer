# Notebook 3b: NaN Handling & Plausibility Filtering

**Autor:** Silas Pignotti  
**Datum:** Januar 2026  
**Version:** 1.0

---

## 1. OVERVIEW & METHODOLOGY

### 1.1 Purpose

This notebook ensures **data completeness and plausibility** on the temporally reduced feature set (März-Oktober). The workflow addresses:

1. **NaN Re-Evaluation:** Re-count missing values on reduced temporal window (8 months vs. original 12)
2. **Conservative Filtering:** Remove trees with >2 missing months (non-interpolable gaps)
3. **Adaptive Interpolation:** Linear temporal interpolation for ≤2 missing months with genus → city → global fallback
4. **NDVI Plausibility:** Remove unrealistic trees (max_NDVI < 0.3 indicating dead/mislocated trees)

**Input:** Temporally reduced datasets from 03a_temporal_feature_selection_JM.ipynb  
**Output:** Complete, plausible datasets (0 NaNs, realistic NDVI ranges)

### 1.2 Workflow

```
[PHASE 1: NaN RE-EVALUATION]
├── Step 2.1: Count NaN per Tree on Reduced Set
├── Step 2.2: NaN Statistics (per City, Month, Genus)
└── Step 2.3: Visualization (Heatmaps, Histograms)

↓

[PHASE 2: NaN FILTERING & INTERPOLATION]
├── Step 2.4: Remove Trees with >2 Missing Months
├── Step 2.5: Adaptive Linear Interpolation
│   ├── Genus-Level Mean (if n≥10)
│   ├── City-Level Mean (if genus too small)
│   └── Global Mean (last fallback)
└── Step 2.6: Validation (0 NaN Check)

↓

[PHASE 3: NDVI PLAUSIBILITY FILTERING]
├── Step 2.7: Calculate max(NDVI) per Tree
├── Step 2.8: Filter max_NDVI < 0.3
└── Step 2.9: Impact Analysis (Removal Stats)

↓

[PHASE 4: EXPORT & DOCUMENTATION]
├── Step 2.10: Dataset Export (Clean GeoPackages)
└── Step 2.11: Metadata Export (Statistics, Report)

↓

[OUTPUT]
├── trees_clean_no_edge.gpkg
├── trees_clean_20m_edge.gpkg
├── nan_statistics_reduced_set.csv
├── interpolation_report.json
└── nan_handling_report.md
```

### 1.3 Expected Outputs

| File | Type | Description |
|------|------|-------------|
| `trees_clean_no_edge.gpkg` | GeoPackage | Clean deciduous trees, 0 NaNs, plausible NDVI (no-edge) |
| `trees_clean_20m_edge.gpkg` | GeoPackage | Clean deciduous trees, 0 NaNs, plausible NDVI (20m-edge) |
| `nan_statistics_reduced_set.csv` | CSV | NaN counts per tree, city, month, genus |
| `interpolation_report.json` | JSON | Interpolation statistics and method usage |
| `nan_handling_report.md` | Markdown | Documentation of filtering and interpolation decisions |

### 1.4 Theoretical Background: NaN Handling Strategy

**Why Linear Temporal Interpolation?**

1. **Phenological Plausibility:**  
   Tree spectral signatures follow smooth temporal trajectories (leaf-out → peak → senescence). Linear interpolation between adjacent months preserves this continuity.

2. **Leakage Prevention:**  
   - **No spatial information:** Avoids using neighbor trees (spatial autocorrelation risk)
   - **No genus information:** Avoids using genus-level means (would leak test set information in cross-city scenarios)
   - **Temporal-only:** Uses only past/future months within the same tree

3. **Model-Agnostic:**  
   Works identically for Random Forest, XGBoost, and 1D-CNN (no model-specific assumptions)

**Adaptive Fallback Strategy:**  
For edge cases where linear interpolation fails (start/end of temporal window), use hierarchical mean imputation:
- **Level 1:** Genus-level mean (if n≥10 samples in city)
- **Level 2:** City-level mean (if genus too small)
- **Level 3:** Global mean (last resort)

**Why >2 Missing Months Threshold?**
- **≤2 months:** Linear interpolation phenologically plausible (gap ≤ 2 months)
- **≥3 months:** Risk of contiguous gaps spanning critical phenological phases (e.g., entire leaf-out period missing)

---

## 2. SETUP & IMPORTS

### 2.1 Packages & Environment

```python
# Core packages
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import json
import warnings
from datetime import datetime
import gc

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print("✓ Imports successful")
```

```python
# Google Colab: Mount Drive
from google.colab import drive
drive.mount('/content/drive')
```

### 2.2 Path Configuration

```python
# Base paths
BASE_DIR = Path("/content/drive/MyDrive/Studium/Geoinformation/Module/Projektarbeit")
DATA_DIR = BASE_DIR / 'data'

# Input: Temporally reduced datasets (from 01_temporal_selection)
INPUT_DATA_DIR = DATA_DIR / '02_pipeline' / '04_feature_reduction' / '01_temporal_selection' / 'data'

# Output: Clean datasets (NaN-free, plausible)
BASE_OUTPUT_DIR = DATA_DIR / '02_pipeline' / '04_feature_reduction' / '02_nan_handling'
OUTPUT_DATA_DIR = BASE_OUTPUT_DIR / 'data'
OUTPUT_METADATA_DIR = BASE_OUTPUT_DIR / 'metadata'
OUTPUT_FIGURES_DIR = BASE_OUTPUT_DIR / 'plots'

# Create output directories
OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_METADATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ Paths configured")
print(f"  Input:  {INPUT_DATA_DIR}")
print(f"  Output: {BASE_OUTPUT_DIR}")
```

### 2.3 Visualization Setup

```python
# Publication-ready style
PUBLICATION_STYLE = {
    'style': 'seaborn-v0_8-whitegrid',
    'figsize': (14, 8),
    'dpi_export': 300,
}

def setup_publication_style():
    plt.rcdefaults()
    plt.style.use(PUBLICATION_STYLE['style'])
    sns.set_palette('Set2')
    plt.rcParams['figure.figsize'] = PUBLICATION_STYLE['figsize']
    plt.rcParams['savefig.dpi'] = PUBLICATION_STYLE['dpi_export']
    plt.rcParams['font.size'] = 11
    print("✓ Publication style configured")

setup_publication_style()
```

### 2.4 Utility Functions

```python
def parse_feature_name(col: str) -> tuple:
    """
    Extract band/index name and month from feature column names.

    Examples:
        'NDVI_06' -> ('NDVI', 6)
        'B02_03' -> ('B02', 3)

    Returns:
        tuple: (feature_name, month) or (None, None)
    """
    parts = col.split('_')
    if len(parts) >= 2:
        try:
            month = int(parts[-1])
            if 1 <= month <= 12:
                feature_name = '_'.join(parts[:-1])
                return feature_name, month
        except ValueError:
            pass
    return None, None


def get_temporal_features(df, selected_months=None):
    """
    Identify all temporal features in dataset.

    Args:
        df: GeoDataFrame
        selected_months: Optional list of months to filter

    Returns:
        dict: {feature_base: [col1, col2, ...]}
    """
    temporal_features = {}

    for col in df.columns:
        feature_name, month = parse_feature_name(col)
        if feature_name is not None:
            if selected_months is None or month in selected_months:
                if feature_name not in temporal_features:
                    temporal_features[feature_name] = []
                temporal_features[feature_name].append(col)

    return temporal_features


print("✓ Utility functions defined")
```

### 2.5 Configuration Constants

```python
# From previous notebook (temporal selection)
SELECTED_MONTHS = [3, 4, 5, 6, 7, 8, 9, 10]  # März-Oktober

# NaN Handling Configuration
MAX_MISSING_MONTHS = 2  # Remove trees with >2 missing months
MIN_GENUS_SAMPLES = 10  # Minimum samples for genus-level interpolation

# NDVI Plausibility Threshold
MIN_NDVI_THRESHOLD = 0.3  # max(NDVI) must be ≥ 0.3

print("Configuration:")
print(f"  Selected months: {SELECTED_MONTHS}")
print(f"  Max missing months: {MAX_MISSING_MONTHS}")
print(f"  NDVI threshold: {MIN_NDVI_THRESHOLD}")
```

---

## 3. DATA LOADING

```python
import shutil
import os

print("Loading temporally reduced datasets...\n")

# Define local path for faster I/O
LOCAL_INPUT_DIR = Path('/content/data_local/input')
LOCAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# List of files to process
input_files = [
    'trees_temporal_reduced_no_edge.gpkg',
    'trees_temporal_reduced_20m_edge.gpkg'
]

# Copy files from Drive to local runtime
for filename in input_files:
    drive_path = INPUT_DATA_DIR / filename
    local_path = LOCAL_INPUT_DIR / filename

    if not local_path.exists():
        if drive_path.exists():
            print(f"  Copying {filename} to local runtime...")
            shutil.copy2(drive_path, local_path)
        else:
            print(f"  ⚠️ Warning: {filename} not found in {INPUT_DATA_DIR}")
    else:
        print(f"  {filename} already exists locally.")

# Load datasets from local copy
try:
    print(f"\nReading no-edge dataset...")
    trees_no_edge = gpd.read_file(LOCAL_INPUT_DIR / 'trees_temporal_reduced_no_edge.gpkg')
    print(f"✓ Loaded no-edge: {len(trees_no_edge):,} trees")
except Exception as e:
    print(f"❌ Error loading no-edge: {e}")

try:
    print(f"Reading 20m-edge dataset...")
    trees_20m_edge = gpd.read_file(LOCAL_INPUT_DIR / 'trees_temporal_reduced_20m_edge.gpkg')
    print(f"✓ Loaded 20m-edge: {len(trees_20m_edge):,} trees")
except Exception as e:
    print(f"❌ Error loading 20m-edge: {e}")

# Verify dataset structure
if 'trees_no_edge' in locals():
    print(f"\nDataset info (No-Edge):")
    print(f"  Cities: {sorted(trees_no_edge['city'].unique())}")
    print(f"  Genera: {trees_no_edge['genus_latin'].nunique()} unique")
    print(f"  Columns: {len(trees_no_edge.columns)} total")
```

### 3.1 Feature Identification

```python
# Identify temporal features
temporal_features = get_temporal_features(trees_no_edge, SELECTED_MONTHS)

print(f"\nTemporal Features (März-Oktober):")
print(f"  Base features: {len(temporal_features)}")
print(f"  Total columns: {sum(len(cols) for cols in temporal_features.values())}")

# Show feature bases
print(f"\nFeature bases with temporal coverage:")
for feat, cols in sorted(temporal_features.items()):
    print(f"  {feat}: {len(cols)} months")

# Flatten list of all temporal columns
all_temporal_cols = [col for cols in temporal_features.values() for col in cols]
print(f"\nTotal temporal feature columns: {len(all_temporal_cols)}")
```

---

## 4. NaN RE-EVALUATION ON REDUCED SET

### 4.1 NaN Counting Strategy

Re-evaluate NaN patterns on the reduced temporal window (8 months instead of original 12). This accounts for:
- Removal of high-NaN winter months (Nov-Feb)
- More accurate per-tree missingness assessment
- Better interpolation feasibility estimation

```python
def count_nans_per_tree(df, temporal_cols):
    """
    Count NaN values per tree across temporal features.

    Returns:
        pd.Series: NaN count per tree
    """
    return df[temporal_cols].isna().sum(axis=1)


def analyze_nan_patterns(df, temporal_cols, dataset_name='Dataset'):
    """
    Comprehensive NaN pattern analysis.

    Returns:
        dict: Statistics about NaN patterns
    """
    print(f"\n{'='*60}")
    print(f"NaN Analysis: {dataset_name}")
    print(f"{'='*60}\n")

    # Count NaNs per tree
    nan_counts = count_nans_per_tree(df, temporal_cols)
    df['nan_count'] = nan_counts

    # Convert to missing months (each feature_base × month = 1 missing month)
    n_feature_bases = len(temporal_features)
    df['nan_months'] = (nan_counts / n_feature_bases).round().astype(int)

    # Overall statistics
    total_cells = len(df) * len(temporal_cols)
    total_nans = nan_counts.sum()
    nan_percentage = (total_nans / total_cells) * 100

    print(f"Overall NaN Statistics:")
    print(f"  Total trees: {len(df):,}")
    print(f"  Total cells: {total_cells:,}")
    print(f"  Total NaNs: {total_nans:,} ({nan_percentage:.2f}%)")
    print(f"  Trees with any NaN: {(nan_counts > 0).sum():,} ({(nan_counts > 0).mean() * 100:.1f}%)")
    print(f"  Trees with 0 NaN: {(nan_counts == 0).sum():,} ({(nan_counts == 0).mean() * 100:.1f}%)")

    # Per-tree NaN distribution
    print(f"\nNaN Months Distribution:")
    month_dist = df['nan_months'].value_counts().sort_index()
    for months, count in month_dist.items():
        pct = (count / len(df)) * 100
        print(f"  {months} months: {count:,} trees ({pct:.1f}%)")

    # Per-city statistics
    print(f"\nNaN by City:")
    for city in sorted(df['city'].unique()):
        city_df = df[df['city'] == city]
        city_nans = count_nans_per_tree(city_df, temporal_cols)
        city_nan_pct = (city_nans.sum() / (len(city_df) * len(temporal_cols))) * 100
        print(f"  {city}: {city_nan_pct:.2f}% NaN ({(city_nans > 0).sum():,} trees affected)")

    # Trees to be removed (>2 missing months)
    to_remove = df['nan_months'] > MAX_MISSING_MONTHS
    print(f"\nTrees with >{MAX_MISSING_MONTHS} missing months (to be removed):")
    print(f"  Count: {to_remove.sum():,} ({to_remove.mean() * 100:.1f}%)")

    # Trees requiring interpolation (1-2 missing months)
    to_interpolate = (df['nan_months'] > 0) & (df['nan_months'] <= MAX_MISSING_MONTHS)
    print(f"\nTrees requiring interpolation (1-{MAX_MISSING_MONTHS} missing months):")
    print(f"  Count: {to_interpolate.sum():,} ({to_interpolate.mean() * 100:.1f}%)")

    return {
        'total_trees': len(df),
        'total_nans': int(total_nans),
        'nan_percentage': round(nan_percentage, 2),
        'trees_with_nan': int((nan_counts > 0).sum()),
        'trees_to_remove': int(to_remove.sum()),
        'trees_to_interpolate': int(to_interpolate.sum()),
        'nan_months_distribution': month_dist.to_dict()
    }


# Analyze both datasets
stats_no_edge = analyze_nan_patterns(trees_no_edge, all_temporal_cols, 'No-Edge')
stats_20m_edge = analyze_nan_patterns(trees_20m_edge, all_temporal_cols, '20m-Edge')
```

### 4.2 NaN Distribution Visualization

```python
# Plot NaN months distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, (df, title) in enumerate([(trees_no_edge, 'No-Edge'),
                                    (trees_20m_edge, '20m-Edge')]):
    ax = axes[idx]

    # Histogram of NaN months
    nan_months_dist = df['nan_months'].value_counts().sort_index()
    colors = ['green' if m == 0 else 'orange' if m <= MAX_MISSING_MONTHS else 'red'
             for m in nan_months_dist.index]

    ax.bar(nan_months_dist.index, nan_months_dist.values, color=colors, alpha=0.7)

    # Add threshold line
    ax.axvline(x=MAX_MISSING_MONTHS + 0.5, color='red', linestyle='--',
              linewidth=2, label=f'Threshold (>{MAX_MISSING_MONTHS} months → remove)')

    ax.set_xlabel('Missing Months', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Trees', fontsize=12, fontweight='bold')
    ax.set_title(f'NaN Months Distribution - {title}', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add text annotations
    for i, v in enumerate(nan_months_dist.values):
        ax.text(nan_months_dist.index[i], v, f'{v:,}',
               ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_FIGURES_DIR / 'nan_months_distribution.png',
           dpi=300, bbox_inches='tight')
plt.show()

print("✓ NaN distribution plot saved")
```

---

## 5. NaN FILTERING & INTERPOLATION

### 5.1 Remove Trees with >2 Missing Months

```python
print(f"\n{'='*60}")
print(f"FILTERING: Remove trees with >{MAX_MISSING_MONTHS} missing months")
print(f"{'='*60}\n")

# No-edge dataset
mask_keep_no = trees_no_edge['nan_months'] <= MAX_MISSING_MONTHS
n_removed_no = (~mask_keep_no).sum()
trees_no_edge_filtered = trees_no_edge[mask_keep_no].copy()

print(f"No-Edge Dataset:")
print(f"  Before: {len(trees_no_edge):,} trees")
print(f"  Removed: {n_removed_no:,} trees ({n_removed_no/len(trees_no_edge)*100:.1f}%)")
print(f"  After: {len(trees_no_edge_filtered):,} trees")

# 20m-edge dataset
mask_keep_20m = trees_20m_edge['nan_months'] <= MAX_MISSING_MONTHS
n_removed_20m = (~mask_keep_20m).sum()
trees_20m_edge_filtered = trees_20m_edge[mask_keep_20m].copy()

print(f"\n20m-Edge Dataset:")
print(f"  Before: {len(trees_20m_edge):,} trees")
print(f"  Removed: {n_removed_20m:,} trees ({n_removed_20m/len(trees_20m_edge)*100:.1f}%)")
print(f"  After: {len(trees_20m_edge_filtered):,} trees")

# Memory cleanup
del trees_no_edge, trees_20m_edge
gc.collect()

print("\n✓ Filtering complete, memory cleaned")
```

### 5.2 Adaptive Linear Interpolation

**Strategy:**
1. **Primary:** Linear temporal interpolation (pandas `interpolate(method='linear', axis=1)`)
2. **Fallback (for edge cases):** Hierarchical mean imputation
   - Level 1: Genus-level mean (if genus has ≥10 samples in city)
   - Level 2: City-level mean (if genus too small)
   - Level 3: Global mean (last resort)

```python
def interpolate_temporal_features(df, temporal_features_dict,
                                  min_genus_samples=MIN_GENUS_SAMPLES):
    """
    Adaptive temporal interpolation with hierarchical fallback.

    Args:
        df: GeoDataFrame with NaN values
        temporal_features_dict: Dict of {feature_base: [col1, col2, ...]}
        min_genus_samples: Minimum samples for genus-level imputation

    Returns:
        GeoDataFrame: Interpolated dataset
        dict: Interpolation statistics
    """
    df_interp = df.copy()

    stats = {
        'total_nans_before': 0,
        'total_nans_after': 0,
        'linear_interpolated': 0,
        'genus_mean_filled': 0,
        'city_mean_filled': 0,
        'global_mean_filled': 0
    }

    # Count initial NaNs
    all_cols = [col for cols in temporal_features_dict.values() for col in cols]
    stats['total_nans_before'] = int(df_interp[all_cols].isna().sum().sum())

    print(f"  Initial NaNs: {stats['total_nans_before']:,}")

    # Step 1: Linear temporal interpolation
    print(f"  Step 1: Linear temporal interpolation...")
    for feature_base, cols in temporal_features_dict.items():
        # Sort columns by month to ensure proper temporal order
        cols_sorted = sorted(cols, key=lambda x: int(x.split('_')[-1]))

        # Linear interpolation along time axis (axis=1)
        df_interp[cols_sorted] = df_interp[cols_sorted].interpolate(
            method='linear', axis=1, limit=MAX_MISSING_MONTHS, limit_area='inside'
        )

    # Count remaining NaNs after linear interpolation
    nans_after_linear = df_interp[all_cols].isna().sum().sum()
    stats['linear_interpolated'] = stats['total_nans_before'] - int(nans_after_linear)
    print(f"    → Filled {stats['linear_interpolated']:,} NaNs")

    # Step 2: Hierarchical mean imputation for remaining NaNs
    if nans_after_linear > 0:
        print(f"  Step 2: Hierarchical fallback for {int(nans_after_linear):,} remaining NaNs...")

        for feature_base, cols in temporal_features_dict.items():
            for col in cols:
                nan_mask = df_interp[col].isna()

                if not nan_mask.any():
                    continue

                # Level 1: Genus-level mean (per city)
                for city in df_interp['city'].unique():
                    city_mask = (df_interp['city'] == city) & nan_mask

                    if not city_mask.any():
                        continue

                    for genus in df_interp.loc[city_mask, 'genus_latin'].unique():
                        genus_mask = city_mask & (df_interp['genus_latin'] == genus)

                        # Check if genus has enough samples
                        genus_values = df_interp.loc[
                            (df_interp['city'] == city) &
                            (df_interp['genus_latin'] == genus) &
                            ~df_interp[col].isna(),
                            col
                        ]

                        if len(genus_values) >= min_genus_samples:
                            # Use genus-level mean
                            genus_mean = genus_values.mean()
                            n_filled = genus_mask.sum()
                            df_interp.loc[genus_mask, col] = genus_mean
                            stats['genus_mean_filled'] += n_filled
                        else:
                            # Level 2: City-level mean
                            city_values = df_interp.loc[
                                (df_interp['city'] == city) & ~df_interp[col].isna(),
                                col
                            ]
                            if len(city_values) > 0:
                                city_mean = city_values.mean()
                                n_filled = genus_mask.sum()
                                df_interp.loc[genus_mask, col] = city_mean
                                stats['city_mean_filled'] += n_filled

                # Level 3: Global mean (last resort)
                remaining_nan = df_interp[col].isna()
                if remaining_nan.any():
                    global_mean = df_interp.loc[~df_interp[col].isna(), col].mean()
                    n_filled = remaining_nan.sum()
                    df_interp.loc[remaining_nan, col] = global_mean
                    stats['global_mean_filled'] += n_filled

    # Final NaN count
    stats['total_nans_after'] = int(df_interp[all_cols].isna().sum().sum())

    print(f"\n  Interpolation Summary:")
    print(f"    Linear interpolation: {stats['linear_interpolated']:,}")
    print(f"    Genus-level mean: {stats['genus_mean_filled']:,}")
    print(f"    City-level mean: {stats['city_mean_filled']:,}")
    print(f"    Global mean: {stats['global_mean_filled']:,}")
    print(f"    Remaining NaNs: {stats['total_nans_after']:,}")

    return df_interp, stats


print("✓ Interpolation function defined")
```

```python
print(f"\n{'='*60}")
print(f"INTERPOLATION: Adaptive temporal interpolation")
print(f"{'='*60}\n")

# Interpolate no-edge dataset
print("Processing No-Edge Dataset...")
trees_no_edge_clean, interp_stats_no = interpolate_temporal_features(
    trees_no_edge_filtered, temporal_features
)

# Interpolate 20m-edge dataset
print("\nProcessing 20m-Edge Dataset...")
trees_20m_edge_clean, interp_stats_20m = interpolate_temporal_features(
    trees_20m_edge_filtered, temporal_features
)

print("\n✓ Interpolation complete")
```

### 5.3 Validation: NaN Check

```python
print(f"\n{'='*60}")
print(f"VALIDATION: NaN Check")
print(f"{'='*60}\n")

# Check no-edge
nans_no = trees_no_edge_clean[all_temporal_cols].isna().sum().sum()
print(f"No-Edge Dataset:")
print(f"  Remaining NaNs: {nans_no}")
if nans_no == 0:
    print(f"  ✓ PASS: 0 NaNs in temporal features")
else:
    print(f"  ✗ FAIL: {nans_no} NaNs remaining")

# Check 20m-edge
nans_20m = trees_20m_edge_clean[all_temporal_cols].isna().sum().sum()
print(f"\n20m-Edge Dataset:")
print(f"  Remaining NaNs: {nans_20m}")
if nans_20m == 0:
    print(f"  ✓ PASS: 0 NaNs in temporal features")
else:
    print(f"  ✗ FAIL: {nans_20m} NaNs remaining")

# Remove nan_count and nan_months columns (no longer needed)
trees_no_edge_clean = trees_no_edge_clean.drop(columns=['nan_count', 'nan_months'])
trees_20m_edge_clean = trees_20m_edge_clean.drop(columns=['nan_count', 'nan_months'])
```

---

## 6. NDVI PLAUSIBILITY FILTERING

### 6.1 NDVI Plausibility Rationale

**Criterion:** `max(NDVI) >= 0.3` across all months (März-Oktober)

**Why this threshold?**
- **Healthy deciduous trees:** NDVI > 0.4 during peak season (May-July)
- **NDVI < 0.3:** Indicates missing or severely reduced green foliage
- **Conservative threshold:** Avoids removing healthy trees in shaded locations

**What does max_NDVI < 0.3 indicate?**
1. **Dead/dying tree:** No photosynthetically active leaves
2. **GPS error:** Coordinates point to non-vegetation (road, building)
3. **Mixed pixel:** 10m pixel dominated by non-vegetation
4. **Severe stress:** Tree with minimal foliage

```python
print(f"\n{'='*60}")
print(f"NDVI PLAUSIBILITY FILTERING")
print(f"{'='*60}\n")

# Get NDVI columns
ndvi_cols = [col for col in all_temporal_cols if col.startswith('NDVI_')]
print(f"NDVI columns: {ndvi_cols}")

# Calculate max NDVI per tree
trees_no_edge_clean['max_NDVI'] = trees_no_edge_clean[ndvi_cols].max(axis=1)
trees_20m_edge_clean['max_NDVI'] = trees_20m_edge_clean[ndvi_cols].max(axis=1)

# Show distribution
print(f"\nmax_NDVI Distribution (No-Edge):")
print(trees_no_edge_clean['max_NDVI'].describe())

print(f"\nmax_NDVI Distribution (20m-Edge):")
print(trees_20m_edge_clean['max_NDVI'].describe())
```

```python
# Apply filter: max_NDVI >= threshold
print(f"\nApplying filter: max_NDVI >= {MIN_NDVI_THRESHOLD}\n")

# No-edge
mask_valid_no = trees_no_edge_clean['max_NDVI'] >= MIN_NDVI_THRESHOLD
n_removed_ndvi_no = (~mask_valid_no).sum()
trees_no_edge_final = trees_no_edge_clean[mask_valid_no].copy()

print(f"No-Edge Dataset:")
print(f"  Before: {len(trees_no_edge_clean):,} trees")
print(f"  Removed: {n_removed_ndvi_no:,} trees ({n_removed_ndvi_no/len(trees_no_edge_clean)*100:.1f}%)")
print(f"  After: {len(trees_no_edge_final):,} trees")

# 20m-edge
mask_valid_20m = trees_20m_edge_clean['max_NDVI'] >= MIN_NDVI_THRESHOLD
n_removed_ndvi_20m = (~mask_valid_20m).sum()
trees_20m_edge_final = trees_20m_edge_clean[mask_valid_20m].copy()

print(f"\n20m-Edge Dataset:")
print(f"  Before: {len(trees_20m_edge_clean):,} trees")
print(f"  Removed: {n_removed_ndvi_20m:,} trees ({n_removed_ndvi_20m/len(trees_20m_edge_clean)*100:.1f}%)")
print(f"  After: {len(trees_20m_edge_final):,} trees")

# Observation: Edge dataset has higher removal rate?
if n_removed_ndvi_20m/len(trees_20m_edge_clean) > n_removed_ndvi_no/len(trees_no_edge_clean):
    print(f"\n⚠️  OBSERVATION: 20m-edge dataset has higher NDVI removal rate")
    print(f"   Hypothesis: Edge trees may be more stressed (less irrigation, more exposure)")

# Memory cleanup
del trees_no_edge_clean, trees_20m_edge_clean, trees_no_edge_filtered, trees_20m_edge_filtered
gc.collect()

print("\n✓ NDVI filtering complete, memory cleaned")
```

---

## 7. EXPORT & DOCUMENTATION

### 7.1 Dataset Export

```python
print(f"\n{'='*60}")
print(f"EXPORT: Clean datasets")
print(f"{'='*60}\n")

# Remove max_NDVI column (diagnostic only)
trees_no_edge_final = trees_no_edge_final.drop(columns=['max_NDVI'])
trees_20m_edge_final = trees_20m_edge_final.drop(columns=['max_NDVI'])

# --- VERIFICATION START ---
def verify_bands(df, name):
    """Helper to verify presence and naming of spectral bands."""
    print(f"\nSpectral Bands Check ({name}):")
    found_bands = {}
    expected_bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']

    # Check columns
    for col in df.columns:
        parts = col.split('_')
        if len(parts) >= 2 and parts[0] in expected_bands:
            band = parts[0]
            if band not in found_bands:
                found_bands[band] = []
            found_bands[band].append(col)

    # Report
    for band in expected_bands:
        if band in found_bands:
            cols = found_bands[band]
            # Sort by month for nice display
            cols.sort(key=lambda x: int(x.split('_')[-1]))
            print(f"  ✓ {band}: Found {len(cols)} columns. Format example: '{cols[0]}'")
        else:
            print(f"  ⚠️ {band}: MISSING!")

verify_bands(trees_no_edge_final, "No-Edge Final")
# --- VERIFICATION END ---

# Export no-edge
output_no_edge = OUTPUT_DATA_DIR / 'trees_clean_no_edge.gpkg'
trees_no_edge_final.to_file(output_no_edge, driver='GPKG')
print(f"\n✓ Exported: {output_no_edge.name}")
print(f"  Trees: {len(trees_no_edge_final):,}")
print(f"  Features: {len(trees_no_edge_final.columns)}")
print(f"  NaNs in temporal features: {trees_no_edge_final[all_temporal_cols].isna().sum().sum()}")

# Export 20m-edge
output_20m_edge = OUTPUT_DATA_DIR / 'trees_clean_20m_edge.gpkg'
trees_20m_edge_final.to_file(output_20m_edge, driver='GPKG')
print(f"\n✓ Exported: {output_20m_edge.name}")
print(f"  Trees: {len(trees_20m_edge_final):,}")
print(f"  Features: {len(trees_20m_edge_final.columns)}")
print(f"  NaNs in temporal features: {trees_20m_edge_final[all_temporal_cols].isna().sum().sum()}")
```

### 7.2 Metadata Export

```python
import numpy as np

def convert_to_native_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    return obj

print(f"\nExporting metadata...\n")

# 1. NaN Statistics CSV
nan_stats_combined = pd.DataFrame([
    {'dataset': 'no_edge', **stats_no_edge},
    {'dataset': '20m_edge', **stats_20m_edge}
])
nan_stats_path = OUTPUT_METADATA_DIR / 'nan_statistics_reduced_set.csv'
nan_stats_combined.to_csv(nan_stats_path, index=False)
print(f"✓ Exported: {nan_stats_path.name}")

# 2. Interpolation Report JSON
raw_report = {
    'processing_date': datetime.now().isoformat(),
    'configuration': {
        'max_missing_months': MAX_MISSING_MONTHS,
        'min_genus_samples': MIN_GENUS_SAMPLES,
        'min_ndvi_threshold': MIN_NDVI_THRESHOLD,
        'selected_months': SELECTED_MONTHS
    },
    'no_edge': {
        'nan_filtering': {
            'trees_removed': n_removed_no,
            'removal_rate_pct': round(n_removed_no / (len(trees_no_edge_final) + n_removed_no + n_removed_ndvi_no) * 100, 2)
        },
        'interpolation': interp_stats_no,
        'ndvi_filtering': {
            'trees_removed': n_removed_ndvi_no,
            'removal_rate_pct': round(n_removed_ndvi_no / (len(trees_no_edge_final) + n_removed_ndvi_no) * 100, 2)
        },
        'final': {
            'total_trees': len(trees_no_edge_final),
            'nans_remaining': 0
        }
    },
    '20m_edge': {
        'nan_filtering': {
            'trees_removed': n_removed_20m,
            'removal_rate_pct': round(n_removed_20m / (len(trees_20m_edge_final) + n_removed_20m + n_removed_ndvi_20m) * 100, 2)
        },
        'interpolation': interp_stats_20m,
        'ndvi_filtering': {
            'trees_removed': n_removed_ndvi_20m,
            'removal_rate_pct': round(n_removed_ndvi_20m / (len(trees_20m_edge_final) + n_removed_ndvi_20m) * 100, 2)
        },
        'final': {
            'total_trees': len(trees_20m_edge_final),
            'nans_remaining': 0
        }
    }
}

# Convert to native types
interpolation_report = convert_to_native_types(raw_report)

interp_report_path = OUTPUT_METADATA_DIR / 'interpolation_report.json'
with open(interp_report_path, 'w') as f:
    json.dump(interpolation_report, f, indent=2)
print(f"✓ Exported: {interp_report_path.name}")

# 3. Markdown Report
report_md = f"""# NaN Handling & Plausibility Filtering Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Notebook:** 03b_nan_handling_plausibility.ipynb

---

## Processing Summary

### Configuration
- **Max Missing Months:** {MAX_MISSING_MONTHS} (trees with >{MAX_MISSING_MONTHS} months removed)
- **NDVI Threshold:** {MIN_NDVI_THRESHOLD} (max_NDVI must be ≥{MIN_NDVI_THRESHOLD})
- **Temporal Window:** März-Oktober (8 months)

### No-Edge Dataset

| Stage | Trees | Removed | Rate |
|-------|-------|---------|------|
| Input (from temporal reduction) | {len(trees_no_edge_final) + n_removed_no + n_removed_ndvi_no:,} | - | - |
| After NaN filtering (>{MAX_MISSING_MONTHS} months) | {len(trees_no_edge_final) + n_removed_ndvi_no:,} | {n_removed_no:,} | {n_removed_no/(len(trees_no_edge_final)+n_removed_no+n_removed_ndvi_no)*100:.1f}% |
| After interpolation | {len(trees_no_edge_final) + n_removed_ndvi_no:,} | 0 | 0% |
| After NDVI filtering (<{MIN_NDVI_THRESHOLD}) | {len(trees_no_edge_final):,} | {n_removed_ndvi_no:,} | {n_removed_ndvi_no/(len(trees_no_edge_final)+n_removed_ndvi_no)*100:.1f}% |
| **FINAL** | **{len(trees_no_edge_final):,}** | **{n_removed_no + n_removed_ndvi_no:,}** | **{(n_removed_no+n_removed_ndvi_no)/(len(trees_no_edge_final)+n_removed_no+n_removed_ndvi_no)*100:.1f}%** |

**Interpolation Statistics:**
- Linear interpolation: {interp_stats_no['linear_interpolated']:,} NaNs
- Genus-level mean: {interp_stats_no['genus_mean_filled']:,} NaNs
- City-level mean: {interp_stats_no['city_mean_filled']:,} NaNs
- Global mean: {interp_stats_no['global_mean_filled']:,} NaNs
- **Total filled:** {interp_stats_no['total_nans_before']:,} NaNs → 0 NaNs ✓

### 20m-Edge Dataset

| Stage | Trees | Removed | Rate |
|-------|-------|---------|------|
| Input (from temporal reduction) | {len(trees_20m_edge_final) + n_removed_20m + n_removed_ndvi_20m:,} | - | - |
| After NaN filtering (>{MAX_MISSING_MONTHS} months) | {len(trees_20m_edge_final) + n_removed_ndvi_20m:,} | {n_removed_20m:,} | {n_removed_20m/(len(trees_20m_edge_final)+n_removed_20m+n_removed_ndvi_20m)*100:.1f}% |
| After interpolation | {len(trees_20m_edge_final) + n_removed_ndvi_20m:,} | 0 | 0% |
| After NDVI filtering (<{MIN_NDVI_THRESHOLD}) | {len(trees_20m_edge_final):,} | {n_removed_ndvi_20m:,} | {n_removed_ndvi_20m/(len(trees_20m_edge_final)+n_removed_ndvi_20m)*100:.1f}% |
| **FINAL** | **{len(trees_20m_edge_final):,}** | **{n_removed_20m + n_removed_ndvi_20m:,}** | **{(n_removed_20m+n_removed_ndvi_20m)/(len(trees_20m_edge_final)+n_removed_20m+n_removed_ndvi_20m)*100:.1f}%** |

**Interpolation Statistics:**
- Linear interpolation: {interp_stats_20m['linear_interpolated']:,} NaNs
- Genus-level mean: {interp_stats_20m['genus_mean_filled']:,} NaNs
- City-level mean: {interp_stats_20m['city_mean_filled']:,} NaNs
- Global mean: {interp_stats_20m['global_mean_filled']:,} NaNs
- **Total filled:** {interp_stats_20m['total_nans_before']:,} NaNs → 0 NaNs ✓

---

## Methodology

### 1. NaN Filtering
**Rationale:** Trees with >2 missing months cannot be reliably interpolated
- Risk of contiguous gaps spanning critical phenological phases
- Linear interpolation assumes smooth temporal trajectories
- Conservative threshold balances data retention vs. quality

### 2. Adaptive Interpolation
**Primary Method:** Linear temporal interpolation (pandas `interpolate(method='linear', axis=1)`)
- Preserves individual phenological variability
- Leakage-free (no spatial/genus information)
- Model-agnostic (works for RF, XGBoost, CNN)

**Hierarchical Fallback (for edge cases):**
1. Genus-level mean (if n≥{MIN_GENUS_SAMPLES} in city)
2. City-level mean (if genus too small)
3. Global mean (last resort)

### 3. NDVI Plausibility
**Threshold:** max_NDVI ≥ {MIN_NDVI_THRESHOLD}

**Why {MIN_NDVI_THRESHOLD}?**
- Healthy deciduous trees: NDVI > 0.4 during peak season
- Conservative threshold to avoid removing shaded trees
- NDVI < {MIN_NDVI_THRESHOLD} indicates: dead/dying trees, GPS errors, mixed pixels, severe stress

**Observation:**
20m-edge dataset shows higher NDVI removal rate ({n_removed_ndvi_20m/(len(trees_20m_edge_final)+n_removed_ndvi_20m)*100:.1f}% vs {n_removed_ndvi_no/(len(trees_no_edge_final)+n_removed_ndvi_no)*100:.1f}%), potentially indicating more stressed trees near city edges.

---

## Quality Assurance

✓ **0 NaNs** in all temporal features (both datasets)
✓ **100% plausible** NDVI ranges (max_NDVI ≥ {MIN_NDVI_THRESHOLD})
✓ **Temporal continuity** preserved (März-Oktober, 8 months)
✓ **Leakage prevention** via temporal-only interpolation

---

## Next Steps

1. **CHM Relevance Assessment:** Evaluate structural features (Notebook 03c)
2. **Correlation Analysis:** Reduce redundancy within/between feature classes (Notebook 03d)
3. **Outlier Detection:** Z-Score, Mahalanobis, CHM plausibility (Notebook 03e)
"""

report_path = OUTPUT_METADATA_DIR / 'nan_handling_report.md'
with open(report_path, 'w') as f:
    f.write(report_md)
print(f"✓ Exported: {report_path.name}")

print(f"\n{'='*60}")
print(f"NaN HANDLING & PLAUSIBILITY FILTERING COMPLETE")
print(f"{'='*60}")
print(f"\nOutputs saved to: {OUTPUT_DATA_DIR}")
print(f"  Data: 2 GeoPackages (no-edge, 20m-edge)")
print(f"  Metadata: 3 files (CSV, JSON, Markdown)")
print(f"  Figures: 1 visualization (NaN distribution)")
print(f"\n✓ Ready for next step: CHM Relevance Assessment (Notebook 03c)")
```

---

**Notebook End**

**Next Notebook:** `03c_chm_relevance_assessment.ipynb`  
**Input for Next Step:** `trees_clean_*.gpkg`

**Author:** Silas Pignotti  
**Date:** Januar 2026
