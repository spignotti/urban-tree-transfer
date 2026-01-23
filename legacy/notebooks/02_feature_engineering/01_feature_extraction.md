---

## 1. OVERVIEW & METHODOLOGY

### 1.1 Purpose

This notebook extracts multi-temporal remote sensing features for urban tree classification across three German cities (Berlin, Hamburg, Rostock). For each tree in the provided cadastres (standard and edge-filtered variants), we extract:

- **CHM-derived features (4):** `height_m` (from cadastre), `CHM_mean`, `CHM_max`, `CHM_std` (10m resolution)
- **Sentinel-2 time series (276):** 10 spectral bands + 13 vegetation indices × 12 months (2021)
- **Total:** 280 features per tree + metadata (tree_id, city, genus_latin, species_latin, geometry)

**Key methodological steps:**
1. **Data Loading:** Load city-specific tree cadastres (standard and 20m edge-filtered).
2. **CHM Extraction:** Point-based extraction from 10m CHM rasters (mean, max, std).
3. **Sentinel-2 Extraction:** Monthly median composites (Jan-Dec 2021) extracting 23 spectral features (bands + indices).
4. **Result Compilation:** Aggregation of all features into city-specific GeoPackages.

**Methodological constraints:**
- **Spatial consistency:** Point-based extraction ensures exact tree location correspondence.
- **Temporal consistency:** All cities use identical 12-month window (Jan-Dec 2021).
- **Data Preservation:** Full preservation of raw extracted values (including NoData/NaN) for downstream analysis.

### 1.2 Workflow

```
[PHASE 1: DATA LOADING]
├── Step 1.1: Load tree cadastres (Standard & Edge-filtered) └── Step 1.2: Validate geometry and attributes

↓

[PHASE 2: CHM FEATURE EXTRACTION]
├── Step 2.1: Extract height_m (cadastre)
└── Step 2.2: Extract CHM_mean, CHM_max, CHM_std (10m rasters)

↓

[PHASE 3: SENTINEL-2 FEATURE EXTRACTION]
├── Step 3.1: Extract monthly values (23 bands × 12 months)
└── Step 3.2: Track NoData statistics

↓

[OUTPUT: Feature datasets per city]
```

### 1.3 Expected Outputs

| File                                     | Type       | Description                                                                 |
| ---------------------------------------- | ---------- | --------------------------------------------------------------------------- |
| `trees_with_features_Berlin.gpkg`       | GeoPackage | Berlin trees with 184 features (4 CHM + 180 S2)                            |
| `trees_with_features_Hamburg.gpkg`      | GeoPackage | Hamburg trees with 184 features (4 CHM + 180 S2)                           |
| `trees_with_features_Rostock.gpkg`      | GeoPackage | Rostock trees with 184 features (4 CHM + 180 S2)                           |
| `feature_extraction_summary.json`        | JSON       | Processing statistics: N trees per city, NoData stats, feature completeness |

**Feature naming convention:**
- CHM: `height_m`, `CHM_mean`, `CHM_max`, `CHM_std`
- Sentinel-2: `{band}_{month:02d}` (e.g., `B02_01`, `NDVI_06`, `RTVIcore_12`)

---

## 2. SETUP & IMPORTS

### 2.1 Packages & Environment

```python
!pip install geopandas rasterio --quiet
```

```python
import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import rowcol
from pathlib import Path
from tqdm.notebook import tqdm
import warnings

warnings.filterwarnings('ignore')

print("✅ Imports successful")
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2.2 Visualization & Utility Functions

```python
import matplotlib.pyplot as plt
import seaborn as sns

PUBLICATION_STYLE = {
    'style': 'seaborn-v0_8-whitegrid',
    'figsize': (12, 7),
    'dpi_export': 300,
}

def setup_publication_style():
    plt.rcdefaults()
    plt.style.use(PUBLICATION_STYLE['style'])
    sns.set_palette('Set2')
    plt.rcParams['figure.figsize'] = PUBLICATION_STYLE['figsize']
    plt.rcParams['savefig.dpi'] = PUBLICATION_STYLE['dpi_export']
    print("✅ Publication style configured")

setup_publication_style()
```

```python
def extract_raster_values_at_points(gdf, raster_path, band=1):
    """
    Extract raster values at point geometries.

    Args:
        gdf: GeoDataFrame with point geometries
        raster_path: Path to raster file
        band: Band number to extract (default=1)

    Returns:
        np.ma.MaskedArray with extracted values (NoData masked)
    """
    with rasterio.open(raster_path) as src:
        coords = [(geom.x, geom.y) for geom in gdf.geometry]
        values = np.array([x[0] for x in src.sample(coords, indexes=band)])

        # Create masked array with NoData as mask
        nodata = src.nodata if src.nodata is not None else -9999
        masked_values = np.ma.masked_equal(values, nodata)

        return masked_values

print("✅ Utility functions defined")
```

---

## 3. CONFIGURATION & PARAMETERS

### 3.1 Paths

```python
BASE_DIR = Path("/content/drive/MyDrive/Studium/Geoinformation/Module/Projektarbeit")
DATA_DIR = BASE_DIR / "data"

# Input directories
CADASTRE_DIR = DATA_DIR / "02_pipeline" / "01_corrected" / "data"
S2_DIR = DATA_DIR / "01_raw" / "sentinel2_2021" / "images"
CHM_DIR = DATA_DIR / "01_raw" / "CHM" / "processed" / "CHM_10m"

# Output directory
OUTPUT_DIR = DATA_DIR / "02_pipeline" / "02_all_features"

# Subdirectories
OUTPUT_DATA_DIR = OUTPUT_DIR / "data"
OUTPUT_METADATA_DIR = OUTPUT_DIR / "metadata"

# Create directories
for d in [OUTPUT_DATA_DIR, OUTPUT_METADATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"✅ Base directory: {BASE_DIR}")
print(f"✅ Output structure created in: {OUTPUT_DIR}")
print(f"   ├─ data/")
print(f"   └─ metadata/")
```

### 3.2 Processing Parameters

```python
PROCESSING_PARAMS = {
    # Cities
    'cities': ["Berlin", "Hamburg", "Rostock"],

    # CHM configuration
    'chm_variants': ["mean", "max", "std"],
    'chm_reference_year': 2021,

    # Sentinel-2 configuration
    's2_bands': [
        # Spectral Bands (10)
        "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
        # Vegetation Indices (13)
        "NDVI", "GNDVI", "EVI", "VARI", "NDre1", "NDVIre", "CIre", "IRECI",
        "RTVIcore", "NDWI", "MSI", "NDII", "kNDVI"
    ],
    's2_months': list(range(1, 13)),  # Jan-Dec
    's2_year': 2021,
}

# Display parameters
print("Processing Parameters:")
print("-" * 50)
for key, value in PROCESSING_PARAMS.items():
    if isinstance(value, list) and len(value) > 5:
        print(f"  {key:<30} {len(value)} items")
    else:
        print(f"  {key:<30} {str(value):<20}")
```

---

## 4. DATA LOADING

### 4.1 Load Input Datasets

```python
print("="*80)
print("PHASE 1: DATA LOADING")
print("="*80)

DATASET_VARIANTS = {
    'standard': "trees_corrected_{city}.gpkg",
    'edge_20m': "trees_corrected_edge_filtered_20m_{city}.gpkg"
}

all_trees = []

for city in PROCESSING_PARAMS['cities']:
    print(f"\nLoading {city}...")

    for variant_name, filename_pattern in DATASET_VARIANTS.items():
        filename = filename_pattern.format(city=city)
        cadastre_path = CADASTRE_DIR / filename

        if not cadastre_path.exists():
            print(f"  ⚠ Warning: File not found: {filename}")
            continue

        print(f"  Loading {variant_name}: {filename}")
        city_trees = gpd.read_file(cadastre_path)
        city_trees['city'] = city
        city_trees['dataset_variant'] = variant_name
        all_trees.append(city_trees)

if not all_trees:
    raise FileNotFoundError("No tree cadastre files were loaded. Please check paths.")

trees_gdf = pd.concat(all_trees, ignore_index=True)

print(f"\n✅ Loaded {len(trees_gdf):,} trees in total")
print(f"  Cities: {trees_gdf['city'].value_counts().to_dict()}")
print(f"  Variants: {trees_gdf['dataset_variant'].value_counts().to_dict()}")
```

### 4.2 Data Validation

```python
# Validate required columns
# User specified FINAL_COLUMNS: tree_id, city, tree_type, genus_latin, species_latin, height_m, geometry
required_cols = ["tree_id", "city", "tree_type", "genus_latin", "species_latin", "height_m", "geometry"]
missing_cols = [col for col in required_cols if col not in trees_gdf.columns]

if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

print("✅ All required columns present")

# Check geometry validity
invalid_geom = (~trees_gdf.geometry.is_valid).sum()
if invalid_geom > 0:
    print(f"⚠ Warning: {invalid_geom} trees with invalid geometries")
    trees_gdf = trees_gdf[trees_gdf.geometry.is_valid].copy()
    print(f"  Removed invalid geometries, {len(trees_gdf):,} trees remaining")

print("✅ Data validation complete")
```

---

## 5. MAIN PROCESSING

### 5.1 Pre-Filtering

```python
# Track initial count
trees_original = len(trees_gdf)
print(f
```

### 5.2 CHM Feature Extraction

```python
print("\n" + "="*80)
print("PHASE 2: CHM FEATURE EXTRACTION")
print("="*80)

# Initialize CHM columns
for variant in PROCESSING_PARAMS['chm_variants']:
    trees_gdf[f"CHM_{variant}"] = np.nan

for city in PROCESSING_PARAMS['cities']:
    city_mask = trees_gdf['city'] == city
    city_trees = trees_gdf[city_mask]

    if len(city_trees) == 0:
        continue

    print(f"\n  Processing {city} ({len(city_trees):,} trees)")

    for variant in PROCESSING_PARAMS['chm_variants']:
        chm_path = CHM_DIR / f"CHM_10m_{variant}_{city}.tif"

        if not chm_path.exists():
            print(f"    ⚠ Warning: CHM file not found: {chm_path.name}")
            continue

        print(f"    Extracting CHM_{variant}...")
        values = extract_raster_values_at_points(city_trees, chm_path, band=1)

        trees_gdf.loc[city_mask, f"CHM_{variant}"] = values.filled(np.nan)

        nodata_count = values.mask.sum()
        print(f"      NoData: {nodata_count:,}/{len(city_trees):,} ({nodata_count/len(city_trees)*100:.1f}%)")

print(f"\n✅ CHM extraction complete")
```

### 5.3 Sentinel-2 Feature Extraction

```python
print("\n" + "="*80)
print("PHASE 3: SENTINEL-2 FEATURE EXTRACTION (BATCHED)")
print("="*80)

BATCH_SIZE = 50000  # Process trees in chunks to save RAM

# Initialize S2 columns
s2_bands = PROCESSING_PARAMS['s2_bands']
s2_months = PROCESSING_PARAMS['s2_months']

for band in s2_bands:
    for month in s2_months:
        trees_gdf[f"{band}_{month:02d}"] = np.nan

trees_gdf['nodata_months'] = 0

for city in PROCESSING_PARAMS['cities']:
    city_mask = trees_gdf['city'] == city
    city_trees = trees_gdf[city_mask]

    if len(city_trees) == 0:
        continue

    print(f"\n  Processing {city} ({len(city_trees):,} trees)")

    nodata_counter = np.zeros(len(city_trees), dtype=int)

    for month in tqdm(s2_months, desc="  Months"):
        s2_path = S2_DIR / f"S2_{city}_{PROCESSING_PARAMS['s2_year']}_{month:02d}_median.tif"

        if not s2_path.exists():
            print(f"    ⚠ Warning: S2 file not found: {s2_path.name}")
            nodata_counter += 1
            continue

        with rasterio.open(s2_path) as src:
            # Map band names to indices
            band_indices = {src.descriptions[i-1]: i for i in range(1, src.count + 1)}

            valid_bands = []
            valid_indices = []
            for band in s2_bands:
                if band in band_indices:
                    valid_bands.append(band)
                    valid_indices.append(band_indices[band])

            if not valid_indices:
                print(f"      ⚠ No valid bands found in {s2_path.name}")
                continue

            # Process batches
            for start_idx in range(0, len(city_trees), BATCH_SIZE):
                end_idx = min(start_idx + BATCH_SIZE, len(city_trees))
                batch_trees = city_trees.iloc[start_idx:end_idx]
                batch_coords = [(g.x, g.y) for g in batch_trees.geometry]
                batch_indices = batch_trees.index

                try:
                    sampled_data = list(src.sample(batch_coords, indexes=valid_indices))
                    sampled_array = np.array(sampled_data)
                    nodata = src.nodata if src.nodata is not None else -9999

                    for i, band in enumerate(valid_bands):
                        band_values = sampled_array[:, i]

                        if np.issubdtype(band_values.dtype, np.floating):
                            band_values[band_values == nodata] = np.nan
                        else:
                            band_values = band_values.astype(float)
                            band_values[band_values == nodata] = np.nan

                        trees_gdf.loc[batch_indices, f"{band}_{month:02d}"] = band_values

                except Exception as e:
                    print(f"      ⚠ Error extracting batch {start_idx}-{end_idx}: {e}")

        # Track NoData (using the first valid band as proxy)
        if valid_bands:
            first_col = f"{valid_bands[0]}_{month:02d}"
            month_nodata = trees_gdf.loc[city_mask, first_col].isna()
            nodata_counter += month_nodata.values.astype(int)

    trees_gdf.loc[city_mask, 'nodata_months'] = nodata_counter

    print(f"    NoData distribution:")
    nodata_dist = pd.Series(nodata_counter).value_counts().sort_index()
    for n_months, count in nodata_dist.items():
        print(f"      {n_months} months: {count:,} trees ({count/len(city_trees)*100:.1f}%)")

print("\n✅ S2 extraction complete")
```

### 5.4 Temporal Interpolation & NoData Filtering

```python
# Prepare final dataset
trees_final = trees_gdf.copy()
print(f
```

---

## 6. RESULTS & OUTPUTS

### 6.1 Summary Statistics

```python
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

summary = {}

for city in PROCESSING_PARAMS['cities']:
    city_trees = trees_final[trees_final['city'] == city]

    summary[city] = {
        'n_trees': len(city_trees),
        'n_genera': city_trees['genus_latin'].nunique(),
        'chm_height_mean': city_trees['height_m'].mean(),
        'chm_height_std': city_trees['height_m'].std(),
        'feature_completeness': (1 - city_trees.isna().sum(axis=1).mean() / len(city_trees.columns)) * 100
    }

# Display summary
summary_df = pd.DataFrame(summary).T
print("\n", summary_df.to_string())

# Overall statistics
print(f"\n{'='*80}")
print("OVERALL STATISTICS")
print("="*80)
print(f"  Total trees: {len(trees_final):,}")
print(f"  Total features per tree: {len([c for c in trees_final.columns if c.startswith(('B', 'ND', 'kN', 'VA', 'RT', 'CHM'))])}")
print(f"  CHM features: 4")
print(f"  S2 features: {len(s2_bands) * len(s2_months)}")
```

### 6.2 Export Results

```python
print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Export per city and variant
for city in PROCESSING_PARAMS['cities']:
    for variant in trees_gdf['dataset_variant'].unique():
        # Filter for specific city AND variant
        subset_mask = (trees_final['city'] == city) & (trees_final['dataset_variant'] == variant)
        subset_trees = trees_final[subset_mask].copy()

        if len(subset_trees) == 0:
            continue

        # Construct filename based on variant
        if variant == 'standard':
            filename = f"trees_with_features_{city}.gpkg"
        elif variant == 'edge_20m':
            filename = f"trees_with_features_edge_filtered_20m_{city}.gpkg"
        else:
            filename = f"trees_with_features_{variant}_{city}.gpkg"

        # Remove internal column before export (optional, but cleaner)
        if 'dataset_variant' in subset_trees.columns:
            subset_trees = subset_trees.drop(columns=['dataset_variant'])

        output_path = OUTPUT_DATA_DIR / filename
        subset_trees.to_file(output_path, driver="GPKG")
        print(f"  ✅ Exported {city} ({variant}): {len(subset_trees):,} trees → data/{output_path.name}")

# Export summary JSON
summary_path = OUTPUT_METADATA_DIR / "feature_extraction_summary.json"
with open(summary_path, 'w') as f:
    json.dump({
        'processing_date': pd.Timestamp.now().isoformat(),
        'cities': summary,
        'parameters': PROCESSING_PARAMS,
        'total_trees': len(trees_final),
        'variants': list(trees_gdf['dataset_variant'].unique())
    }, f, indent=2, default=str)

print(f"  ✅ Exported summary → metadata/{summary_path.name}")
print("\n✅ All exports complete")
```

### 6.3 Visualizations

```python
print("Visualizations skipped (processing only).")
```

---

## 7. SUMMARY & INSIGHTS

### 7.1 Key Findings

```python
print("\n" + "="*80)
print("NOTEBOOK COMPLETE - KEY FINDINGS")
print("="*80)

print(f"\n✅ Feature extraction completed successfully")
print(f"\n  Total trees processed: {len(trees_final):,}")
print(f"  Trees removed (all filters): {trees_original - len(trees_final):,} ({(trees_original - len(trees_final))/trees_original*100:.1f}%)")
print(f"\n  Feature structure:")
print(f"    - CHM features: 4 (height_m, CHM_mean, CHM_max, CHM_std)")
print(f"    - S2 features: {len(s2_bands) * len(s2_months)} ({len(s2_bands)} bands × {len(s2_months)} months)")
print(f"    - Total: {4 + len(s2_bands) * len(s2_months)} features per tree")
print(f"\n  City breakdown:")
for city in PROCESSING_PARAMS['cities']:
    city_count = (trees_final['city'] == city).sum()
    print(f"    - {city}: {city_count:,} trees")
print(f"\n  Output files:")
print(f"    - trees_with_features_<city>.gpkg (2 variants per city)")
print(f"    - feature_extraction_summary.json")

print("\n" + "="*80)
print("Next step: Proceed to experiment design")
print("="*80)
```

---

**Notebook End**

Exported: 2025-01-11

Author: Silas Pignotti
