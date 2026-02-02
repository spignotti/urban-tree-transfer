# Phase 2 Feature Engineering - Code Review & Fixes

**Review Date:** 2026-02-01  
**Status:** Pre-Notebook Execution Review  
**Reviewer:** GitHub Copilot (Claude Sonnet 4.5)

## Executive Summary

Conducted comprehensive review of Phase 2 Feature Engineering implementation covering:

- ✅ Data flow and schema progression (Phase 2a → 2b → 2c)
- ✅ Missing value handling across all modules
- ✅ Module transition points and dependencies
- ✅ Data leakage prevention
- ❌ Found 1 critical bug and 3 high-priority improvements needed

**Verdict:** Generally clean implementation with logical architecture. One critical bug must be fixed before running notebooks. Additional robustness improvements recommended.

---

## Critical Issues (Must Fix Before Running)

### 🔴 Issue #1: Edge NaN Logic Bug in `filter_nan_trees()`

**Location:** `src/urban_tree_transfer/feature_engineering/quality.py:177`

**Severity:** Critical - Logic error affecting data retention

**Problem:**  
The edge NaN counting logic computes a single scalar count across ALL trees, then compares it per-tree. This causes incorrect removal behavior:

```python
# CURRENT (BUGGY) CODE
edge_nan_count = edge_missing.sum()  # Scalar: total across all trees

to_remove = (nan_counts > max_nan_months) | (
    edge_missing & (edge_nan_count > max_edge_nan_months)  # Bug!
)
```

**Impact:**

- If ≥1 tree has edge NaN, `edge_nan_count > max_edge_nan_months` is always True
- ALL trees with any edge NaN get removed, regardless of threshold
- Expected behavior: Remove trees with >1 edge NaN, but code removes trees with ≥1 edge NaN when any tree has edge NaN

**Fix:**

```python
def filter_nan_trees(
    trees_gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict[str, Any],
    max_nan_months: int = 2,
    max_edge_nan_months: int = 1,
) -> gpd.GeoDataFrame:
    """Filter trees with excessive NaN values.

    Removal criteria:
    1. Trees with >max_nan_months total NaN values in any feature base
    2. Trees with >max_edge_nan_months NaN values at temporal edges

    Args:
        trees_gdf: GeoDataFrame with tree features.
        selected_months: Months to analyze (1-12).
        feature_config: Feature configuration.
        max_nan_months: Maximum allowed NaN months per feature base (default: 2).
        max_edge_nan_months: Maximum allowed NaN months at edges (default: 1).

    Returns:
        Filtered GeoDataFrame.
    """
    from urban_tree_transfer.config.loader import load_feature_config

    if feature_config is None:
        feature_config = load_feature_config()

    feature_bases = feature_config["sentinel_features"]["bases"]
    initial_count = len(trees_gdf)
    removal_stats = {}

    logger.info(f"Filtering trees with excessive NaN values (max_nan={max_nan_months}, max_edge={max_edge_nan_months})...")

    for base in feature_bases:
        cols = [f"{base}_{m:02d}" for m in selected_months]
        nan_counts = trees_gdf[cols].isna().sum(axis=1)

        # FIX: Compute per-tree edge NaN count
        first_col, last_col = cols[0], cols[-1]
        edge_nan_counts_per_tree = (
            trees_gdf[first_col].isna().astype(int) +
            trees_gdf[last_col].isna().astype(int)
        )

        # Removal logic: (>max_nan_months total) OR (>max_edge_nan_months at edges)
        to_remove = (
            (nan_counts > max_nan_months) |
            (edge_nan_counts_per_tree > max_edge_nan_months)
        )

        removal_stats[base] = {
            "total_removed": to_remove.sum(),
            "mean_nan_count": nan_counts.mean(),
            "trees_with_edge_nan": (edge_nan_counts_per_tree > 0).sum(),
        }

        trees_gdf = trees_gdf[~to_remove]
        logger.info(f"  {base}: Removed {to_remove.sum()} trees (remaining: {len(trees_gdf)})")

    final_count = len(trees_gdf)
    logger.info(
        f"Filtering complete: {initial_count - final_count} trees removed "
        f"({initial_count} → {final_count}, {100 * final_count / initial_count:.1f}% retained)"
    )

    return trees_gdf
```

**Testing:**

1. Create test case with known edge NaN patterns
2. Verify removal matches expected threshold logic
3. Compare results before/after fix

---

## High Priority Issues (Should Fix)

### 🟡 Issue #2: Missing JSON Existence Checks in Runner Notebooks

**Location:**

- `notebooks/runners/02b_data_quality.ipynb` (temporal_selection.json, chm_assessment.json)
- `notebooks/runners/02c_final_preparation.ipynb` (correlation_removal.json, outlier_thresholds.json, spatial_autocorrelation.json, proximity_threshold.json)

**Severity:** High - UX issue, causes cryptic errors

**Problem:**  
Runner notebooks load JSON configs without checking if they exist first. If exploratory notebooks weren't run, users get cryptic FileNotFoundError with no guidance.

**Impact:**

- Poor user experience
- No data corruption, but wastes time debugging

**Fix for 02b_data_quality.ipynb:**

```python
# Cell: Configuration & Load Data

# Paths
DATA_DIR = Path("/content/drive/MyDrive/TreeClassifikation/data")
OUTPUT_DIR = Path("/content/drive/MyDrive/TreeClassifikation/outputs/phase_2")
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
METADATA_DIR = OUTPUT_DIR / "metadata"

# Validate exploratory JSONs exist
required_jsons = [
    ("temporal_selection.json", "exp_01_temporal_analysis.ipynb"),
    ("chm_assessment.json", "exp_02_chm_assessment.ipynb"),
]

for json_file, notebook in required_jsons:
    json_path = METADATA_DIR / json_file
    if not json_path.exists():
        raise FileNotFoundError(
            f"❌ Missing {json_file}\n"
            f"   Run exploratory/{notebook} first to generate it.\n"
            f"   Expected location: {json_path}"
        )

# Load temporal selection config
with open(METADATA_DIR / "temporal_selection.json") as f:
    temporal_config = json.load(f)
selected_months = temporal_config["selected_months"]
months_to_drop = temporal_config["months_to_drop"]

print(f"✓ Loaded temporal config: {len(selected_months)} months selected")
print(f"  Selected: {selected_months}")
print(f"  Dropped: {months_to_drop}")

# Load CHM assessment config
with open(METADATA_DIR / "chm_assessment.json") as f:
    chm_config = json.load(f)

print(f"✓ Loaded CHM config: {chm_config['clip_range']}")
```

**Fix for 02c_final_preparation.ipynb:**

```python
# Cell: Configuration & Load Data

# Paths
DATA_DIR = Path("/content/drive/MyDrive/TreeClassifikation/data")
OUTPUT_DIR = Path("/content/drive/MyDrive/TreeClassifikation/outputs/phase_2")
FIGURES_DIR = OUTPUT_DIR / "figures"
LOGS_DIR = OUTPUT_DIR / "logs"
METADATA_DIR = OUTPUT_DIR / "metadata"

# Validate ALL exploratory JSONs exist
required_jsons = [
    ("temporal_selection.json", "exp_01_temporal_analysis.ipynb"),
    ("chm_assessment.json", "exp_02_chm_assessment.ipynb"),
    ("correlation_removal.json", "exp_03_correlation_analysis.ipynb"),
    ("outlier_thresholds.json", "exp_04_outlier_thresholds.ipynb"),
    ("spatial_autocorrelation.json", "exp_05_spatial_autocorrelation.ipynb"),
    ("proximity_threshold.json", "exp_06_mixed_genus_proximity.ipynb"),
]

missing_jsons = []
for json_file, notebook in required_jsons:
    json_path = METADATA_DIR / json_file
    if not json_path.exists():
        missing_jsons.append((json_file, notebook))

if missing_jsons:
    error_msg = "❌ Missing configuration files:\n"
    for json_file, notebook in missing_jsons:
        error_msg += f"   - {json_file} (run exploratory/{notebook})\n"
    raise FileNotFoundError(error_msg)

# Load all configs
with open(METADATA_DIR / "temporal_selection.json") as f:
    temporal_config = json.load(f)
selected_months = temporal_config["selected_months"]

with open(METADATA_DIR / "correlation_removal.json") as f:
    correlation_config = json.load(f)
redundant_features = correlation_config["redundant_features"]

with open(METADATA_DIR / "outlier_thresholds.json") as f:
    outlier_config = json.load(f)

with open(METADATA_DIR / "spatial_autocorrelation.json") as f:
    spatial_config = json.load(f)
block_size = spatial_config["selected_block_size"]

with open(METADATA_DIR / "proximity_threshold.json") as f:
    proximity_config = json.load(f)
proximity_threshold = proximity_config["selected_threshold"]

print("✓ All configuration files loaded successfully")
print(f"  - Temporal: {len(selected_months)} months")
print(f"  - Correlation: {len(redundant_features)} redundant features")
print(f"  - Spatial: {block_size}m block size")
print(f"  - Proximity: {proximity_threshold}m threshold")
```

---

### 🟡 Issue #3: No Validation for Interpolation Feasibility

**Location:** `src/urban_tree_transfer/feature_engineering/quality.py:252` (`interpolate_features_within_tree()`)

**Severity:** Medium - Edge case, but possible

**Problem:**  
Trees with only 1 valid month pass through to interpolation. Linear interpolation requires ≥2 points, so:

- `interpolate()` does nothing
- `ffill().bfill()` copies the single value to all months
- Result: All months filled with identical value → low confidence

**Impact:**

- Plausible but uninformative features
- May degrade model performance
- Difficult to detect post-hoc

**Fix (Add validation to `filter_nan_trees()`):**

```python
def filter_nan_trees(
    trees_gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict[str, Any],
    max_nan_months: int = 2,
    max_edge_nan_months: int = 1,
    min_valid_months: int = 2,  # NEW PARAMETER
) -> gpd.GeoDataFrame:
    """Filter trees with excessive NaN values.

    Removal criteria:
    1. Trees with >max_nan_months total NaN values in any feature base
    2. Trees with >max_edge_nan_months NaN values at temporal edges
    3. Trees with <min_valid_months valid values (NEW)

    Args:
        trees_gdf: GeoDataFrame with tree features.
        selected_months: Months to analyze (1-12).
        feature_config: Feature configuration.
        max_nan_months: Maximum allowed NaN months per feature base (default: 2).
        max_edge_nan_months: Maximum allowed NaN months at edges (default: 1).
        min_valid_months: Minimum required valid months for interpolation (default: 2).

    Returns:
        Filtered GeoDataFrame.
    """
    from urban_tree_transfer.config.loader import load_feature_config

    if feature_config is None:
        feature_config = load_feature_config()

    feature_bases = feature_config["sentinel_features"]["bases"]
    initial_count = len(trees_gdf)
    removal_stats = {}

    logger.info(
        f"Filtering trees with excessive NaN values "
        f"(max_nan={max_nan_months}, max_edge={max_edge_nan_months}, "
        f"min_valid={min_valid_months})..."
    )

    for base in feature_bases:
        cols = [f"{base}_{m:02d}" for m in selected_months]
        nan_counts = trees_gdf[cols].isna().sum(axis=1)
        valid_counts = trees_gdf[cols].notna().sum(axis=1)

        # Per-tree edge NaN count
        first_col, last_col = cols[0], cols[-1]
        edge_nan_counts_per_tree = (
            trees_gdf[first_col].isna().astype(int) +
            trees_gdf[last_col].isna().astype(int)
        )

        # Removal logic
        to_remove = (
            (nan_counts > max_nan_months) |
            (edge_nan_counts_per_tree > max_edge_nan_months) |
            (valid_counts < min_valid_months)  # NEW CONDITION
        )

        removal_stats[base] = {
            "total_removed": to_remove.sum(),
            "mean_nan_count": nan_counts.mean(),
            "trees_with_edge_nan": (edge_nan_counts_per_tree > 0).sum(),
            "trees_below_min_valid": (valid_counts < min_valid_months).sum(),  # NEW
        }

        trees_gdf = trees_gdf[~to_remove]
        logger.info(f"  {base}: Removed {to_remove.sum()} trees (remaining: {len(trees_gdf)})")

    final_count = len(trees_gdf)
    logger.info(
        f"Filtering complete: {initial_count - final_count} trees removed "
        f"({initial_count} → {final_count}, {100 * final_count / initial_count:.1f}% retained)"
    )

    return trees_gdf
```

---

### 🟡 Issue #4: No Phase 2c Schema Validation Function

**Location:** `src/urban_tree_transfer/utils/schema_validation.py` (missing function)

**Severity:** Medium - No validation of final output

**Problem:**  
`validate_phase2a_output()` and `validate_phase2b_output()` exist, but no corresponding function for Phase 2c. Final datasets (baseline/filtered splits) have no schema validation.

**Impact:**

- Schema errors in final datasets go undetected
- Harder to debug issues in downstream experiments
- No guarantee of consistency across train/val/test splits

**Fix (Add to `schema_validation.py`):**

```python
def validate_phase2c_output(
    trees_gdf: gpd.GeoDataFrame,
    selected_months: list[int],
    feature_config: dict[str, Any] | None = None,
    redundant_features: list[str] | None = None,
) -> None:
    """Validate Phase 2c final output schema.

    Validates:
    - Metadata columns (11)
    - CHM features (3: raw + zscore + percentile)
    - Sentinel-2 features (selected months, minus redundant)
    - Outlier metadata (5 columns)
    - Spatial assignment (1 column: block_id)

    Args:
        trees_gdf: Final GeoDataFrame (after redundancy removal + splits).
        selected_months: Months selected in temporal analysis.
        feature_config: Feature configuration (loaded if None).
        redundant_features: Features removed in correlation analysis.

    Raises:
        ValueError: If schema validation fails.
    """
    from urban_tree_transfer.config.loader import load_feature_config
    from urban_tree_transfer.utils.schema import (
        get_metadata_columns,
        get_chm_feature_names,
        get_temporal_feature_names,
    )

    if feature_config is None:
        feature_config = load_feature_config()

    _require_project_crs(trees_gdf)

    # Expected columns
    metadata_columns = get_metadata_columns(feature_config)
    chm_features = get_chm_feature_names(include_engineered=True)

    # S2 features MINUS redundant
    temporal_features = get_temporal_feature_names(months=selected_months, config=feature_config)
    if redundant_features:
        temporal_features = [f for f in temporal_features if f not in redundant_features]

    # Outlier metadata
    outlier_cols = [
        "outlier_zscore",
        "outlier_mahalanobis",
        "outlier_iqr",
        "outlier_severity",
        "outlier_method_count",
    ]

    # Spatial assignment
    spatial_cols = ["block_id"]

    required_columns = (
        metadata_columns + chm_features + temporal_features + outlier_cols + spatial_cols
    )

    missing = [col for col in required_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(
            f"Phase 2c output missing {len(missing)} columns:\n"
            f"  {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    # Check for unexpected columns (excluding geometry)
    unexpected = [
        col for col in trees_gdf.columns
        if col not in required_columns and col != "geometry"
    ]
    if unexpected:
        logger.warning(f"Phase 2c output has {len(unexpected)} unexpected columns: {unexpected[:5]}...")

    logger.info(
        f"✓ Phase 2c schema validated: {len(required_columns)} required columns present\n"
        f"  - Metadata: {len(metadata_columns)}\n"
        f"  - CHM: {len(chm_features)}\n"
        f"  - S2 features: {len(temporal_features)}\n"
        f"  - Outlier metadata: {len(outlier_cols)}\n"
        f"  - Spatial: {len(spatial_cols)}"
    )
```

**Update `splits.py` to use validation:**

```python
# In src/urban_tree_transfer/feature_engineering/splits.py

def create_spatial_splits(...) -> dict[str, gpd.GeoDataFrame]:
    """Create train/val/test splits with spatial blocking."""
    # ... existing code ...

    # VALIDATE EACH SPLIT
    from urban_tree_transfer.utils.schema_validation import validate_phase2c_output

    for split_name, split_gdf in splits.items():
        logger.info(f"Validating {split_name} split schema...")
        validate_phase2c_output(
            split_gdf,
            selected_months=selected_months,
            feature_config=feature_config,
            redundant_features=redundant_features
        )

    return splits
```

---

## Implementation Checklist

### Pre-Notebook Execution

- [ ] **Apply Fix #1** (Edge NaN logic bug) - CRITICAL
- [ ] **Apply Fix #2** (JSON existence checks in 02b, 02c)
- [ ] **Apply Fix #3** (Interpolation validation with min_valid_months)
- [ ] **Apply Fix #4** (Phase 2c schema validation function)
- [ ] Run `uv run nox -s ci` to verify all tests pass
- [ ] Update tests to cover new validation logic

### Notebook Execution

- [ ] Run 01_data_processing.ipynb (Phase 1 if needed)
- [ ] Run 02a_feature_extraction.ipynb
- [ ] Run exp_01_temporal_analysis.ipynb → Verify temporal_selection.json created
- [ ] Run exp_02_chm_assessment.ipynb → Verify chm_assessment.json created
- [ ] Run 02b_data_quality.ipynb → Verify JSON checks work, 0 NaN guarantee
- [ ] Run exp_03_correlation_analysis.ipynb → Verify correlation_removal.json created
- [ ] Run exp_04_outlier_thresholds.ipynb → Verify outlier_thresholds.json created
- [ ] Run exp_05_spatial_autocorrelation.ipynb → Verify spatial_autocorrelation.json created
- [ ] Run exp_06_mixed_genus_proximity.ipynb → Verify proximity_threshold.json created
- [ ] Run 02c_final_preparation.ipynb → Verify schema validation passes
- [ ] Verify final schema matches expected formula (metadata + CHM + selected S2 + outlier + spatial)

### Post-Execution Verification

- [ ] Check output file sizes (should be reasonable, not bloated)
- [ ] Verify 0 NaN in `trees_clean_{city}.gpkg`
- [ ] Verify split sizes match expected proportions (60/20/20)
- [ ] Verify block_id assignment is spatially coherent
- [ ] Run spot checks on interpolated values (visual inspection)

---

## Data Flow Verification Summary

### ✅ Confirmed Clean Patterns

1. **Schema Progression:**
   - Phase 2a: 11 metadata + 1 CHM + 276 S2 (12 months × 23 bases) = ~288 cols
   - Phase 2b: 11 metadata + 3 CHM + (23 bases × 12 months S2)
   - Phase 2b: 11 metadata + 3 CHM + (23 bases × selected months)
   - Phase 2c: 11 metadata + 3 CHM + (retained bases × selected months) + 5 outlier + 1 spatial
2. **Missing Value Handling:**
   - Phase 2a: Preserves NaN from satellite composites (correct)
   - Phase 2b: Within-tree interpolation only (no leakage)
   - Phase 2b: 0 NaN guarantee enforced
   - Phase 2c: No new NaN introduction

3. **Data Leakage Prevention:**
   - Interpolation: Per-tree only ✓
   - CHM engineering: Uses independent source data ✓
   - Outlier detection: Flagging only, no removal ✓
   - Spatial splits: Block-based, prevents spatial autocorrelation ✓

4. **Module Transitions:**
   - Phase boundaries clearly defined
   - JSON configs bridge exploratory → runner notebooks
   - File naming conventions consistent

---

## Testing Recommendations

### Unit Tests to Add

```python
# tests/feature_engineering/test_quality.py

def test_filter_nan_trees_edge_logic():
    """Test per-tree edge NaN counting logic."""
    # Create test GeoDataFrame with known edge NaN pattern
    # Tree 1: 0 edge NaN
    # Tree 2: 1 edge NaN (first month)
    # Tree 3: 2 edge NaN (both edges)
    # Tree 4: 1 edge NaN (last month)

    # With max_edge_nan_months=1:
    # Expected: Tree 3 removed, Trees 1,2,4 retained
    pass

def test_filter_nan_trees_min_valid_requirement():
    """Test minimum valid months requirement."""
    # Create test GeoDataFrame with:
    # Tree 1: 0 valid months
    # Tree 2: 1 valid month
    # Tree 3: 2 valid months
    # Tree 4: 8 valid months

    # With min_valid_months=2:
    # Expected: Trees 1,2 removed, Trees 3,4 retained
    pass
```

### Integration Test

```python
# tests/integration/test_phase2_pipeline.py

def test_full_pipeline_zero_nan_guarantee():
    """Test that Phase 2b output has 0 NaN."""
    # Run full pipeline on small sample
    # Assert final output has 0 NaN in all feature columns
    pass

def test_schema_progression():
    """Test schema evolution across phases."""
    # Run Phase 2a → verify ~288 columns
    # Run Phase 2b → verify 11 metadata + 1 CHM + (23 × 12) S2
    # Run Phase 2b → verify 11 metadata + 3 CHM + (23 × selected_months) S2
    # Run Phase 2c → verify adds outlier metadata + spatial assignment
```

---

## Additional Notes

### Why These Fixes Matter

1. **Fix #1 (Edge NaN):** Prevents incorrect data removal that could bias dataset
2. **Fix #2 (JSON checks):** Improves UX, prevents cryptic errors
3. **Fix #3 (Min valid):** Ensures interpolation quality, prevents low-confidence features
4. **Fix #4 (Phase 2c validation):** Catches schema errors before experiments, ensures consistency

### Implementation Order

1. Fix #1 first (critical bug)
2. Fix #3 second (depends on Fix #1 signature)
3. Fix #4 third (schema validation)
4. Fix #2 last (notebook UX improvements)

### Estimated Time

- Implementing fixes: 1-2 hours
- Testing: 1 hour
- Full notebook run: 2-4 hours (depending on data size)
- **Total: ~5-7 hours**

---

**Review Status:** Ready for implementation  
**Next Action:** Apply Fix #1 immediately, then proceed with other fixes before notebook execution
