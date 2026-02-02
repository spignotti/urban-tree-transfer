# PRD 002a: Feature Extraction

**Parent PRD:** [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md)
**Status:** Draft
**Created:** 2026-01-28
**Branch:** `feature/phase2-feature-extraction`

---

## Goal

Extract raw features from Phase 1 outputs: correct tree positions using 1m CHM, extract CHM height values, and sample monthly Sentinel-2 composites at tree locations.

**Success Criteria:**

- [ ] All viable trees have CHM_1m value (no extraction failures)
- [ ] 277 features extracted: 1 CHM + 23 S2 bands × 12 months
- [ ] Metadata columns preserved (tree_id, genus_latin, city, etc.)
- [ ] Output schema matches `feature_config.yaml` specification
- [ ] CRS is EPSG:25833 throughout
- [ ] Extraction summary generated with NaN statistics

---

## Context

**Read before implementation:**

- [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md) - Overall Phase 2 architecture
- `CLAUDE.md` / `AGENT.md` - Project conventions
- `configs/features/feature_config.yaml` - Feature definitions (must be created first)

**Legacy Code (Inspiration Only):**

- `legacy/notebooks/02_feature_engineering/01_feature_extraction.md` - Original implementation
- `legacy/documentation/02_Feature_Engineering/01_Feature_Extraction_Methodik.md` - Methodological documentation

**⚠️ Important:** Data structures have changed from Phase 1 processing. Do NOT copy legacy code directly - adapt logic to new schema.

---

## Scope

### In Scope

- Tree position correction using 1m CHM local maxima
- CHM height extraction at corrected positions (1m resolution)
- Sentinel-2 feature extraction (10 bands + 13 indices) for all 12 months
- Handling missing/invalid values (preserve as NaN)
- Per-city GeoPackage output
- Extraction summary statistics

### Out of Scope

- NaN filling/imputation (handled in PRD 002b)
- Temporal feature selection (exploratory notebook exp_01)
- CHM feature engineering (Z-score, percentile - handled in PRD 002b)
- Feature redundancy removal (PRD 002c)

---

## Implementation

### 1. Notebook Structure

**Template:** [Runner-Notebook Template](../../docs/templates/Notebook_Templates.md#1-runner-notebook-template-phase-2)

**File:** `notebooks/runners/02a_feature_extraction.ipynb`

**Anpassungen gegenüber Base Template:**

- **Zelle 3 (Imports):**

  ```python
  from urban_tree_transfer.feature_engineering.extraction import (
      correct_tree_positions,
      extract_chm_features,
      extract_sentinel_features,
  )
  log = ExecutionLog("02a_feature_extraction")
  ```

- **Zelle 4 (Config):**

  ```python
  INPUT_DIR = DRIVE_DIR / "data" / "phase_1_processing"
  OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
  ```

- **Processing Sections:**
  1. Tree Position Correction (CHM local maxima)
  2. CHM Feature Extraction (1m point sampling)
  3. Sentinel-2 Feature Extraction (temporal bands + indices)

- **Validation:**
  - `trees_corrected.gpkg` (geometry validation)
  - `chm_features.parquet` (schema + NaN counts)
  - `sentinel_features.parquet` (schema + NaN counts)

**Complete template documentation:** [Notebook_Templates.md](../../docs/templates/Notebook_Templates.md)

---

### 2. Configuration Extension

**File:** `src/urban_tree_transfer/config/loader.py`

**Add functions:**

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

### 3. Feature Extraction Module

**File:** `src/urban_tree_transfer/feature_engineering/extraction.py`

**Functions to implement:**

#### 3.1 Tree Position Correction

```python
def correct_tree_positions(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
    search_radius_m: float = 5.0,
    height_weight: float = 0.7,
) -> gpd.GeoDataFrame:
    """Snap tree positions to local CHM maximum within search radius.

    Args:
        trees_gdf: Input tree point geometries
        chm_path: Path to 1m CHM GeoTIFF
        search_radius_m: Maximum search distance for local peak
        height_weight: Weight for height vs distance in peak selection

    Returns:
        GeoDataFrame with corrected geometries and metadata column 'corrected'

    Algorithm:
        1. For each tree, extract CHM values within search_radius_m
        2. Find local maximum using height-weighted distance score:
           score = height * height_weight - distance * (1 - height_weight)
        3. Snap tree to pixel with highest score
        4. Flag trees that couldn't be corrected (no valid CHM data)

    Note: Use rasterio windowed reading for memory efficiency.
    """
```

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/01_feature_extraction.md` contains tree correction logic, but data schema differs.

#### 3.2 CHM Feature Extraction

```python
def extract_chm_features(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
) -> gpd.GeoDataFrame:
    """Extract CHM_1m value at each tree location.

    Args:
        trees_gdf: Tree points (after position correction)
        chm_path: Path to 1m CHM GeoTIFF

    Returns:
        GeoDataFrame with new column 'CHM_1m' (float, may contain NaN)

    Implementation:
        - Point sampling at exact tree coordinates (no interpolation)
        - Preserve NoData as NaN
        - Update 'height_m' from CHM if cadastre value is missing
        - 1m resolution - no resampling!

    Critical: This is direct point sampling, NOT neighborhood aggregation.
    We do NOT compute CHM_mean, CHM_max, CHM_std (these were contaminated in legacy).
    """
```

#### 3.3 Sentinel-2 Feature Extraction

```python
def extract_sentinel_features(
    trees_gdf: gpd.GeoDataFrame,
    sentinel_dir: Path,
    city: str,
    year: int,
    months: list[int],
) -> gpd.GeoDataFrame:
    """Extract all Sentinel-2 bands/indices for specified months.

    Args:
        trees_gdf: Tree points (with CHM features)
        sentinel_dir: Directory containing monthly composites
        city: City name (for filename pattern)
        year: Reference year (2021)
        months: List of months to extract (default: 1-12)

    Returns:
        GeoDataFrame with new columns: {band}_{MM}, {index}_{MM}
        Example: 'NDVI_04' for April NDVI

    Implementation:
        - Load monthly composite: S2_{city}_{year}_{MM}_median.tif
        - Composite has 23 bands: 10 spectral + 13 vegetation indices
        - Point-sample at tree locations (no interpolation)
        - Column naming: f"{feature_name}_{month:02d}"
        - Preserve NoData as NaN (will be handled in PRD 002b)

    Band order (from Phase 1 processing):
        Bands 1-10: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
        Bands 11-23: NDVI, EVI, GNDVI, NDre1, NDVIre, CIre, IRECI,
                     RTVIcore, NDWI, MSI, NDII, kNDVI, VARI

    Note: Use rasterio for efficient multi-band reading.
    """
```

#### 2.4 Pipeline Integration

```python
def extract_all_features(
    trees_gdf: gpd.GeoDataFrame,
    chm_path: Path,
    sentinel_dir: Path,
    city: str,
    feature_config: dict[str, Any],
) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Complete feature extraction pipeline for one city.

    Args:
        trees_gdf: Input trees (from Phase 1)
        chm_path: Path to CHM raster
        sentinel_dir: Directory with Sentinel-2 composites
        city: City name
        feature_config: Loaded from configs/features/feature_config.yaml

    Returns:
        tuple: (trees_with_features, extraction_summary)

    Processing steps:
        1. Correct tree positions
        2. Extract CHM features
        3. Extract Sentinel-2 features (all months)
        4. Validate schema
        5. Generate extraction summary

    Summary includes:
        - Total trees processed
        - Trees successfully corrected
        - Feature completeness per month
        - NaN statistics per feature
        - Processing timestamp
    """
```

---

### 3. Runner Notebook

**File:** `notebooks/runners/02a_feature_extraction.ipynb`

**Structure:**

#### Cell 1: Setup & Imports

```python
from pathlib import Path
import geopandas as gpd
from urban_tree_transfer.config.loader import (
    load_city_config,
    load_feature_config,
    get_all_feature_names,
)
from urban_tree_transfer.feature_engineering.extraction import extract_all_features
from urban_tree_transfer.config.constants import PROJECT_CRS, RANDOM_SEED
```

#### Cell 2: Configuration

```python
# Paths
DATA_DIR = Path("data")
PHASE1_DIR = DATA_DIR / "phase_1_processing"
PHASE2_DIR = DATA_DIR / "phase_2_features"
PHASE2_DIR.mkdir(exist_ok=True)

OUTPUT_DIR = Path("outputs/phase_2")
METADATA_DIR = OUTPUT_DIR / "metadata"
METADATA_DIR.mkdir(parents=True, exist_ok=True)

# Load configs
feature_config = load_feature_config()
cities = ["berlin", "leipzig"]
```

#### Cell 3: Load Trees

```python
# Load Phase 1 filtered trees
trees_path = PHASE1_DIR / "trees" / "trees_filtered_viable.gpkg"
trees_gdf = gpd.read_file(trees_path)

print(f"Loaded {len(trees_gdf)} viable trees")
print(f"Cities: {trees_gdf['city'].value_counts().to_dict()}")
print(f"Genera: {trees_gdf['genus_latin'].nunique()}")
```

#### Cell 4: Feature Extraction (per city)

```python
results = {}

for city in cities:
    print(f"\n{'='*60}")
    print(f"Processing {city.upper()}")
    print(f"{'='*60}")

    # Filter city trees
    city_trees = trees_gdf[trees_gdf["city"] == city].copy()
    print(f"Trees: {len(city_trees)}")

    # Paths
    chm_path = PHASE1_DIR / "chm" / f"CHM_1m_{city}.tif"
    sentinel_dir = PHASE1_DIR / "sentinel2"

    # Extract features
    trees_with_features, summary = extract_all_features(
        trees_gdf=city_trees,
        chm_path=chm_path,
        sentinel_dir=sentinel_dir,
        city=city,
        feature_config=feature_config,
    )

    # Save
    output_path = PHASE2_DIR / f"trees_with_features_{city}.gpkg"
    trees_with_features.to_file(output_path, driver="GPKG")
    print(f"\nSaved: {output_path}")
    print(f"Features: {len(trees_with_features.columns)}")

    results[city] = {
        "output_path": str(output_path),
        "summary": summary,
    }
```

#### Cell 5: Validation

```python
# Validate schema
expected_features = get_all_feature_names()
expected_metadata = feature_config["metadata_columns"]

for city, result in results.items():
    gdf = gpd.read_file(result["output_path"])

    # Check columns
    actual_cols = set(gdf.columns)
    expected_cols = set(expected_metadata + expected_features + ["geometry"])
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    print(f"\n{city.upper()} Validation:")
    print(f"  Total columns: {len(actual_cols)}")
    print(f"  Missing: {missing if missing else 'None'}")
    print(f"  Extra: {extra if extra else 'None'}")

    # Check CRS
    assert gdf.crs.to_epsg() == 25833, f"Wrong CRS: {gdf.crs}"
    print(f"  CRS: ✓ EPSG:25833")

    # NaN statistics
    feature_cols = [col for col in gdf.columns if col in expected_features]
    nan_counts = gdf[feature_cols].isna().sum()
    print(f"  Features with NaN: {(nan_counts > 0).sum()} / {len(feature_cols)}")
```

#### Cell 6: Summary Report

```python
import json

# Consolidate summaries
consolidated_summary = {
    "timestamp": pd.Timestamp.now().isoformat(),
    "cities": results,
    "feature_count": len(expected_features),
    "metadata_columns": expected_metadata,
}

# Save
summary_path = METADATA_DIR / "feature_extraction_summary.json"
with open(summary_path, "w") as f:
    json.dump(consolidated_summary, f, indent=2)

print(f"\nSummary saved: {summary_path}")
```

---

## Testing

**File:** `tests/feature_engineering/test_extraction.py`

**Test cases:**

- Tree position correction with synthetic CHM
- CHM extraction with known raster values
- Sentinel-2 extraction with mock multi-band rasters
- Pipeline integration with minimal real data

---

## Validation Checklist

- [ ] All trees have CHM_1m value (check NaN count)
- [ ] Feature count = 277 (1 CHM + 23 × 12 months)
- [ ] Column names match pattern: `{feature}_{MM:02d}`
- [ ] Metadata columns preserved from Phase 1
- [ ] CRS is EPSG:25833
- [ ] Extraction summary JSON generated
- [ ] Both cities processed identically
- [ ] Output GeoPackages can be loaded without errors

---

## Outputs

**Data:**

- `data/phase_2_features/trees_with_features_berlin.gpkg`
- `data/phase_2_features/trees_with_features_leipzig.gpkg`

**Metadata:**

- `outputs/phase_2/metadata/feature_extraction_summary.json`

---

## Next Steps

After completing this PRD:

1. Run exploratory notebooks: `exp_01_temporal_analysis.ipynb`, `exp_02_chm_assessment.ipynb`
2. Proceed to [PRD 002b: Data Quality](002b_data_quality.md)

---

**Status:** Ready for implementation
