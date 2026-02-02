# Technical Review: Phase 2 Feature Engineering

**Review Date:** 2026-01-31
**Reviewer:** Claude Sonnet 4.5
**Scope:** Complete Phase 2 implementation (Modules, Tests, Notebooks, Data Flow)
**Status:** ⚠️ **ANALYSIS COMPLETE - ISSUES FOUND**

---

## Executive Summary

**Overall Assessment:** Phase 2 implementation is **technically solid** with **good architectural foundations**, but has **critical gaps** in test coverage, JSON schema validation, and end-to-end data flow documentation.

### Severity Distribution

| Severity | Count | Category |
|----------|-------|----------|
| 🔴 **Critical** | 3 | Data integrity, Pipeline validation |
| 🟠 **High** | 8 | Test coverage, Edge cases |
| 🟡 **Medium** | 12 | Documentation, Type safety |
| 🟢 **Low** | 7 | Code quality, Optimization |
| **Total** | **30** | **Issues identified** |

### Critical Issues Requiring Immediate Attention

1. **CRITICAL:** No integration tests for complete 02a → 02b → 02c pipeline
2. **CRITICAL:** JSON schema validation missing between notebook layers
3. **CRITICAL:** No end-to-end test validating final 10 GeoPackages meet specifications

---

## Layer 1: Module Implementation Review

### 1.1 `extraction.py` - ✅ STRONG

**Strengths:**
- Excellent docstrings with algorithm explanations
- Robust error handling for edge cases (empty geometries, NoData)
- Adaptive radius approach is well-implemented
- Batch processing for memory efficiency
- CRS validation throughout

**Issues Found:**

#### 🟡 MEDIUM-1: `extract_sentinel_features()` band ordering assumption
**Location:** Lines 320-325
**Issue:**
```python
# Comment says: "Band order in composites (23 bands total)"
# But NO runtime validation that src.count == 23
```
**Risk:** If Phase 1 output has different band order, silent errors occur.

**Recommendation:**
```python
if src.count != len(s2_features):
    raise ValueError(
        f"Expected {len(s2_features)} bands, got {src.count}. "
        f"Check Phase 1 Sentinel-2 composite generation."
    )
# Also validate band ORDER (not just count)
# Read first band metadata, check band names match expected order
```

#### 🟢 LOW-1: Magic number `_DEFAULT_BATCH_SIZE = 50000`
**Location:** Line 30
**Issue:** Hardcoded batch size, no config option.
**Recommendation:** Add to `feature_config.yaml`:
```yaml
processing:
  batch_size: 50000  # Configurable for different memory scenarios
```

#### 🟡 MEDIUM-2: `correct_tree_positions()` no validation of correction_distance
**Location:** Lines 235
**Issue:** `correction_distance` can theoretically exceed `adaptive_max_radius` due to scoring logic, but this isn't validated.

**Recommendation:**
```python
# After line 235:
if correction_distances[-1] > adaptive_max_radius + 1e-6:  # Allow float epsilon
    raise RuntimeError(
        f"Correction distance {correction_distances[-1]:.2f}m exceeds "
        f"adaptive_max_radius {adaptive_max_radius:.2f}m - algorithm error"
    )
```

---

### 1.2 `quality.py` - ✅ GOOD (with minor gaps)

**Strengths:**
- Within-tree interpolation correctly implemented (no leakage)
- Clear docstrings explaining data leakage prevention
- Comprehensive NaN handling with multiple strategies

**Issues Found:**

#### 🔴 CRITICAL-1: `interpolate_features_within_tree()` - Missing validation
**Location:** Lines 197-297
**Issue:** Function returns interpolated data but **doesn't validate** that all NaNs are actually filled (except edge cases).

**Expected:** After interpolation, interior NaNs should be **zero**.
**Actual:** No assertion to catch bugs in interpolation logic.

**Recommendation:**
```python
# After line 297 (before return):
interior_nan_count = 0
for base in feature_bases:
    base_cols = [f"{base}_{month:02d}" for month in selected_months]
    values = interpolated[base_cols].to_numpy()
    # Check interior (not first/last month)
    if len(base_cols) > 2:
        interior_values = values[:, 1:-1]  # Exclude edges
        interior_nan_count += np.isnan(interior_values).sum()

if interior_nan_count > 0:
    raise RuntimeError(
        f"Interior NaN interpolation failed: {interior_nan_count} remaining "
        f"(expected 0). Check interpolation logic."
    )
```

#### 🟠 HIGH-1: `compute_chm_engineered_features()` - Percentile edge case
**Location:** Lines 336-345
**Issue:** `percentileofscore()` with `kind="rank"` can return values outside [0, 100] for edge cases.

**Recommendation:**
```python
# After line 344:
def _percentile(series: pd.Series) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return pd.Series(np.full(len(series), np.nan), ...)
    result = series.map(
        lambda x: np.nan if pd.isna(x) else percentileofscore(non_null, x, kind="rank")
    )
    # Clamp to [0, 100] to handle edge cases
    result = result.clip(0.0, 100.0)
    return cast(pd.Series, result)
```

#### 🟡 MEDIUM-3: `filter_nan_trees()` - Feature base extraction brittle
**Location:** Lines 180-185
**Issue:**
```python
if "_" in col:
    base = col.rsplit("_", 1)[0]  # Assumes EXACTLY one underscore separator
```
**Risk:** Fails for features like `CHM_1m_zscore` (2 underscores).

**Recommendation:**
```python
# Use explicit feature base list from config instead of parsing
feature_bases = {
    feature
    for feature in get_all_s2_features(feature_config)
}
# Or document that feature names MUST have exactly one underscore
```

---

### 1.3 `selection.py` - ⚠️ NEEDS IMPROVEMENT

**Strengths:**
- Clean separation of correlation computation, identification, removal
- CRS validation

**Issues Found:**

#### 🟠 HIGH-2: `identify_redundant_features()` - Naive selection logic
**Location:** Lines 53-92
**Issue:** Always marks `feature_b` for removal, not the **less important** one.

**Current:**
```python
redundant.append({
    "feature_to_remove": feature_b,  # Always B, not based on importance
    "correlated_with": feature_a,
    ...
})
```

**Expected (per PRD):** Choose based on variance or domain knowledge.

**Recommendation:**
```python
# Add parameter: feature_importance (optional dict)
def identify_redundant_features(
    correlation_matrices: dict[str, pd.DataFrame],
    threshold: float = 0.95,
    feature_importance: dict[str, float] | None = None,  # NEW
) -> list[dict[str, Any]]:
    ...
    if abs(corr_value) > threshold:
        # Choose which to remove based on importance
        if feature_importance:
            if feature_importance.get(feature_a, 0) > feature_importance.get(feature_b, 0):
                to_remove = feature_b
            else:
                to_remove = feature_a
        else:
            to_remove = feature_b  # Fallback
```

#### 🟡 MEDIUM-4: Missing VIF validation function
**Location:** N/A (not implemented)
**Issue:** PRD 002d mentions VIF < 10 validation, but no function exists.

**Recommendation:**
```python
def compute_vif(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute Variance Inflation Factor for features."""
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    data = trees_gdf[feature_columns].dropna()
    vif_data = pd.DataFrame()
    vif_data["feature"] = feature_columns
    vif_data["VIF"] = [
        variance_inflation_factor(data.values, i)
        for i in range(len(feature_columns))
    ]
    return vif_data
```

---

### 1.4 `outliers.py` - ✅ EXCELLENT

**Strengths:**
- All three methods correctly implemented
- Genus-specific Mahalanobis (prevents false positives)
- Clean separation of concerns
- Comprehensive docstrings
- No removal, only flagging (correct per PRD)

**Issues Found:**

#### 🟢 LOW-2: `detect_mahalanobis_outliers()` - Hardcoded minimum samples
**Location:** Line 110
**Issue:**
```python
if valid_mask.sum() < len(feature_columns) + 2:  # Hardcoded +2
    continue
```
**Recommendation:** Add parameter `min_samples_per_genus` (default: `len(features) + 2`)

#### 🟡 MEDIUM-5: Missing covariance matrix singularity check
**Location:** Lines 114-115
**Issue:**
```python
cov = np.cov(valid_data, rowvar=False)
inv_cov = np.linalg.pinv(cov)  # Pseudo-inverse, but no warning if singular
```

**Recommendation:**
```python
cov = np.cov(valid_data, rowvar=False)
if np.linalg.cond(cov) > 1e10:  # Condition number check
    logger.warning(
        f"Genus {_genus}: Covariance matrix near-singular (cond={np.linalg.cond(cov):.2e}). "
        "Mahalanobis distances may be unreliable."
    )
inv_cov = np.linalg.pinv(cov)
```

---

### 1.5 `splits.py` - ✅ VERY GOOD

**Strengths:**
- Spatial disjointness correctly enforced via `StratifiedGroupKFold`
- KL-divergence validation for stratification quality
- Comprehensive validation function
- Nearest-join fallback for boundary trees

**Issues Found:**

#### 🟠 HIGH-3: `validate_split_stratification()` - KL threshold not enforced
**Location:** Lines 249-324
**Issue:** Computes KL-divergence but **doesn't raise error** if KL > threshold.

**Current:**
```python
kl_divergences[f"{name_i}_vs_{name_j}"] = kl
# Returns dict, but NO assertion
```

**Recommendation:**
```python
# Add parameter: kl_threshold (default: 0.01)
def validate_split_stratification(
    *split_gdfs: gpd.GeoDataFrame,
    split_names: list[str] | None = None,
    kl_threshold: float = 0.01,  # NEW
) -> dict[str, Any]:
    ...
    # After computing all KL divergences:
    max_kl = max(kl_divergences.values())
    if max_kl > kl_threshold:
        raise ValueError(
            f"Stratification quality check failed: max KL={max_kl:.4f} > {kl_threshold:.4f}. "
            f"Increase n_splits or check genus distributions."
        )
```

#### 🟡 MEDIUM-6: `create_spatial_blocks()` - Block ID collision risk
**Location:** Lines 56-66
**Issue:**
```python
"block_id": f"{city}_{ix}_{iy}"  # If cities have similar names: "berlin" vs "berliner"
```
**Low probability, but could happen.**

**Recommendation:**
```python
"block_id": f"{city[:8]}_{ix:04d}_{iy:04d}"  # Fixed width, truncate city name
```

---

### 1.6 `proximity.py` - ✅ GOOD

**Strengths:**
- Algorithm correctly identifies mixed-genus contamination risk
- Clean API
- Genus-specific impact analysis

**Issues Found:**

#### 🟠 HIGH-4: `compute_nearest_different_genus_distance()` - O(n²) complexity
**Location:** Lines 60-63
**Issue:**
```python
distances.loc[same_genus.index] = same_genus.geometry.apply(
    lambda geom, diff_geom=diff_geom: diff_geom.distance(geom).min()
)
# For 100k trees × 100k other trees = 10 billion distance calculations
```

**Impact:** Colab timeout for Berlin (~800k trees).

**Recommendation:**
```python
# Use spatial index for nearest neighbor search
from shapely.strtree import STRtree

diff_tree = STRtree(diff_geom)
distances.loc[same_genus.index] = same_genus.geometry.apply(
    lambda geom: diff_tree.nearest(geom).distance(geom)
)
# O(n log m) instead of O(n × m)
```

#### 🟢 LOW-3: `analyze_genus_specific_impact()` - No uniformity check
**Location:** Lines 130-175
**Issue:** Computes removal rates but doesn't validate uniformity (per PRD: max deviation < 10%).

**Recommendation:**
```python
# After line 172:
mean_rate = genus_stats["removal_rate"].mean()
max_deviation = (genus_stats["removal_rate"] - mean_rate).abs().max()
if max_deviation > 0.10:
    logger.warning(
        f"Genus removal rates not uniform: max deviation {max_deviation:.2%} > 10%. "
        f"Some genera disproportionately affected."
    )
```

---

## Layer 2: Test Coverage Analysis

### 2.1 Overall Coverage Assessment

**Test Files Reviewed:**
- `tests/feature_engineering/test_extraction.py` ❌ **MISSING**
- `tests/feature_engineering/test_quality.py` ✅ **EXISTS** (11 tests)
- `tests/feature_engineering/test_selection.py` ❌ **MISSING**
- `tests/feature_engineering/test_outliers.py` ✅ **EXISTS** (5 tests)
- `tests/feature_engineering/test_splits.py` ✅ **EXISTS** (8 tests)
- `tests/feature_engineering/test_proximity.py` ✅ **EXISTS** (5 tests)

### 2.2 Critical Test Gaps

#### 🔴 CRITICAL-2: `extraction.py` has ZERO unit tests
**Missing Tests:**
1. `correct_tree_positions()` - adaptive radius calculation
2. `correct_tree_positions()` - scoring logic
3. `extract_chm_features()` - batch processing
4. `extract_sentinel_features()` - multi-month extraction
5. `extract_all_features()` - complete pipeline

**Impact:** No validation that core feature extraction works correctly.

**Recommendation:** Create `test_extraction.py` with minimum 15 tests.

#### 🟠 HIGH-5: `selection.py` has ZERO unit tests
**Missing Tests:**
1. `compute_feature_correlations()` - matrix computation
2. `identify_redundant_features()` - threshold logic
3. `remove_redundant_features()` - column removal

**Recommendation:** Create `test_selection.py` with minimum 8 tests.

#### 🟠 HIGH-6: Integration tests completely missing
**What's missing:**
- Test that 02a output schema matches 02b input expectations
- Test that exploratory JSON outputs can be loaded by runners
- Test that final GeoPackages have correct schema (277 features → ~187 after reduction)

**Recommendation:** Create `test_integration_phase2.py`:
```python
def test_phase2a_to_2b_schema():
    """Verify 02a output is valid 02b input."""
    # Mock 02a output
    # Run 02b quality functions
    # Assert schema consistency

def test_exploratory_json_consumption():
    """Verify runner notebooks can load exploratory JSONs."""
    # Create mock temporal_selection.json
    # Load with quality.apply_temporal_selection()
    # Assert correct months selected

def test_final_geopackages_schema():
    """Verify 02c outputs match Phase 3 expectations."""
    # Mock 02c pipeline
    # Assert 10 GeoPackages created
    # Assert column schemas correct
```

### 2.3 Edge Case Test Gaps

#### 🟡 MEDIUM-7: quality.py edge cases undertested
**Missing:**
- Empty GeoDataFrame handling
- Single-tree GeoDataFrame (can't interpolate)
- All-NaN feature column
- Extreme outliers in CHM (negative values, >100m)

#### 🟡 MEDIUM-8: splits.py edge cases undertested
**Missing:**
- Single genus (stratification should fail gracefully)
- Fewer blocks than splits requested
- All trees in one block (spatial split impossible)

---

## Layer 3: Notebook-Module Integration

### 3.1 Runner Notebooks Review

**Status:** Cannot fully review without `.ipynb` file access, but can analyze expected structure.

#### 🟠 HIGH-7: No notebook execution tests
**Issue:** Notebooks are code, but aren't tested.

**Recommendation:**
```python
# tests/notebooks/test_runner_notebooks.py
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def test_02a_feature_extraction_executes():
    """Verify 02a notebook can execute without errors."""
    with open("notebooks/runners/02a_feature_extraction.ipynb") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    # Mock data paths
    ep.preprocess(nb, {"metadata": {"path": "notebooks/runners/"}})
    # Assert no cell errors

# Similar for 02b, 02c, exp_01-06
```

#### 🟡 MEDIUM-9: Notebook skip-if-exists logic not validated
**Issue:** PRD mentions "skip-if-exists" for idempotency, but:
- No test that it actually skips
- No test that it re-runs if output is corrupted

---

## Layer 4: JSON Schema Validation

### 4.1 JSON Schema Specifications

**Expected JSONs (from PRD):**
1. `temporal_selection.json`
2. `chm_assessment.json`
3. `correlation_removal.json`
4. `outlier_thresholds.json`
5. `spatial_autocorrelation.json`
6. `proximity_filter.json`

#### 🔴 CRITICAL-3: NO JSON schema definitions exist
**Location:** N/A
**Issue:** PRDs describe JSON structures, but:
- No `.json` schema files
- No validation functions
- Notebooks could produce malformed JSON silently

**Recommendation:**
Create `src/urban_tree_transfer/schemas/` with JSON schemas:
```json
// temporal_selection.schema.json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "required": ["selected_months", "selection_method"],
  "properties": {
    "selected_months": {
      "type": "array",
      "items": {"type": "integer", "minimum": 1, "maximum": 12},
      "minItems": 1
    },
    "selection_method": {"type": "string"},
    "cross_city_validation": {
      "type": "object",
      "properties": {
        "spearman_rho": {"type": "number"},
        "p_value": {"type": "number"}
      }
    }
  }
}
```

Then create validator:
```python
# src/urban_tree_transfer/utils/json_validation.py
import json
import jsonschema

def validate_temporal_selection(json_path: Path) -> dict:
    """Load and validate temporal_selection.json."""
    with open(json_path) as f:
        data = json.load(f)

    schema_path = Path(__file__).parent.parent / "schemas" / "temporal_selection.schema.json"
    with open(schema_path) as f:
        schema = json.load(f)

    jsonschema.validate(data, schema)
    return data
```

### 4.2 JSON Loading Patterns

#### 🟡 MEDIUM-10: Inconsistent JSON loading
**Issue:** Each notebook loads JSON differently (no shared utility).

**Recommendation:**
```python
# src/urban_tree_transfer/utils/config_loader.py
from pathlib import Path
import json

def load_exploratory_json(filename: str) -> dict:
    """Load and validate exploratory JSON config."""
    json_path = Path("outputs/phase_2/metadata") / filename
    if not json_path.exists():
        raise FileNotFoundError(
            f"Required config not found: {filename}. "
            f"Run corresponding exploratory notebook first."
        )

    with open(json_path) as f:
        data = json.load(f)

    # Schema validation here
    return data
```

---

## Layer 5: Data Flow & Schema Consistency

### 5.1 Pipeline Schema Progression

**Expected Schema Evolution:**

```
Phase 1 Output
  ├─ tree_id, city, genus_latin, ... (9 metadata)
  └─ geometry

Phase 2a Output (02a_feature_extraction)
  ├─ [Phase 1 metadata] (9 columns)
  ├─ position_corrected, correction_distance (2 new)
  ├─ CHM_1m (1 feature)
  └─ {band/index}_{01-12} (276 features)
  = 288 columns total

Phase 2b Output (02b_data_quality)
  ├─ [Phase 1 metadata] (9 columns)
  ├─ position_corrected, correction_distance (2)
  ├─ CHM_1m, CHM_1m_zscore, CHM_1m_percentile (3 features)
  └─ {band/index}_{MM} for selected months (23 × 8 = 184 features)
  = 198 columns total

Phase 2c Output (02c_final_preparation)
  ├─ [Phase 1 metadata] (9 columns)
  ├─ position_corrected, correction_distance (2)
  ├─ CHM features (3)
  ├─ Sentinel-2 features (after redundancy removal: ~170)
  ├─ Outlier flags (5: zscore, mahal, iqr, severity, method_count)
  └─ block_id (1)
  = ~190 columns total
```

#### 🟠 HIGH-8: No schema validation between pipeline stages
**Issue:** If 02a produces wrong schema, 02b silently fails or corrupts data.

**Recommendation:**
```python
# src/urban_tree_transfer/utils/schema_validation.py
def validate_phase2a_output(gdf: gpd.GeoDataFrame) -> None:
    """Validate that GeoDataFrame matches Phase 2a expected schema."""
    required_metadata = [
        "tree_id", "city", "genus_latin", "species_latin",
        "genus_german", "species_german", "plant_year",
        "height_m", "tree_type"
    ]
    required_features = [
        "position_corrected", "correction_distance", "CHM_1m"
    ]

    missing = [col for col in required_metadata + required_features if col not in gdf.columns]
    if missing:
        raise ValueError(f"Phase 2a output missing columns: {missing}")

    # Check S2 features (should have 12 months)
    s2_features = get_all_s2_features()
    for feature in s2_features:
        for month in range(1, 13):
            col = f"{feature}_{month:02d}"
            if col not in gdf.columns:
                raise ValueError(f"Phase 2a missing S2 feature: {col}")

    # Check CRS
    if str(gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Phase 2a wrong CRS: {gdf.crs}")
```

### 5.2 Metadata Preservation Tracking

#### 🟡 MEDIUM-11: No explicit metadata preservation tests
**Issue:** Metadata should be untouched through pipeline, but not tested.

**Recommendation:**
```python
def test_metadata_preservation_through_pipeline():
    """Verify metadata columns unchanged from Phase 1 → Phase 2c."""
    # Create mock Phase 1 output
    phase1_gdf = create_mock_phase1_output()

    # Run 02a
    phase2a_gdf, _ = extract_all_features(phase1_gdf, ...)

    # Run 02b
    phase2b_gdf = run_quality_pipeline(phase2a_gdf, ...)

    # Run 02c
    phase2c_gdf = run_final_preparation(phase2b_gdf, ...)

    # Assert metadata untouched
    metadata_cols = ["tree_id", "city", "genus_latin", ...]
    for col in metadata_cols:
        pd.testing.assert_series_equal(
            phase1_gdf[col],
            phase2c_gdf[col],
            check_names=False
        )
```

---

## Layer 6: Critical Path Analysis (Data Leakage)

### 6.1 Leakage Risk Assessment

#### ✅ PASS: `interpolate_features_within_tree()`
**Analysis:** Correctly implements within-tree interpolation.
- No genus means
- No city means
- No cross-tree operations
- **Leakage Risk: NONE**

#### ✅ PASS: `compute_chm_engineered_features()`
**Analysis:** City-level normalization is acceptable.
- CHM is independent variable (LiDAR)
- Not derived from spectral features
- **Leakage Risk: MINIMAL** (see PRD 002d Improvement 3 for enhancement)

#### ✅ PASS: `create_stratified_splits_*()`
**Analysis:** StratifiedGroupKFold ensures block-level disjointness.
- Groups = block_id (no block in multiple splits)
- Stratify = genus_latin (balanced distributions)
- **Leakage Risk: NONE** (if block_size correct)

### 6.2 Subtle Leakage Risks

#### 🟡 MEDIUM-12: `detect_mahalanobis_outliers()` - Genus-level covariance
**Location:** Lines 114
**Issue:** Covariance computed per genus **across all splits**.

**Question:** Is this leakage?

**Answer:** **Borderline.** Outlier detection is **meta-information** about data quality, not predictive features. BUT:
- If used to **remove** outliers → could leak (removes bad samples from train based on val/test distribution)
- If used to **flag** outliers → acceptable (flags are metadata, ablation in Phase 3)

**Current implementation:** Flags only → **Acceptable**, but document in Phase 3 that outlier flags should NOT be used as features.

---

## Layer 7: End-to-End Validation

### 7.1 Final Output Specifications

**Expected:** 10 GeoPackages
- `berlin_train.gpkg` (~31,500 trees)
- `berlin_val.gpkg` (~6,750 trees)
- `berlin_test.gpkg` (~6,750 trees)
- `leipzig_finetune.gpkg` (~28,000 trees)
- `leipzig_test.gpkg` (~7,000 trees)
- **+ 5 Filtered variants**

#### 🟡 MEDIUM-13: No specification of exact final schema
**Issue:** PRD 002c describes outputs, but doesn't give **exact column list**.

**Recommendation:**
Document in `docs/documentation/02_Feature_Engineering/Final_Output_Schema.md`:
```markdown
## berlin_train.gpkg Schema (Example)

### Identifiers (3)
- tree_id (str)
- city (str: "berlin")
- genus_latin (str)

### Spatial (2)
- geometry (Point, EPSG:25833)
- block_id (str)

### Metadata (7)
- species_latin, genus_german, species_german, plant_year, height_m, tree_type
- position_corrected, correction_distance

### Features (~187 after redundancy removal)
- CHM_1m, CHM_1m_zscore, CHM_1m_percentile (3)
- {band/index}_{MM} for selected months (23 bases × 8 months = 184)
- MINUS redundant features (~15) = ~170 S2 features

### Outlier Metadata (5)
- outlier_zscore, outlier_mahalanobis, outlier_iqr
- outlier_severity, outlier_method_count

**Total:** ~200 columns
```

### 7.2 Zero-NaN Guarantee

#### 🟠 HIGH-9: No final NaN check in 02c
**Issue:** 02c should assert **zero NaN** in all feature columns before export.

**Recommendation:**
```python
# In 02c final preparation notebook, before export:
def validate_zero_nan(gdf: gpd.GeoDataFrame, dataset_name: str) -> None:
    """Assert no NaN in feature columns."""
    feature_cols = [col for col in gdf.columns if col not in METADATA_COLUMNS]
    nan_counts = gdf[feature_cols].isna().sum()
    nan_features = nan_counts[nan_counts > 0]

    if not nan_features.empty:
        raise ValueError(
            f"{dataset_name}: Found NaN in feature columns:\n{nan_features}\n"
            "All NaN should be resolved in Phase 2b."
        )
```

---

## Summary Dashboard

### Issues by Severity

#### 🔴 Critical (3)
1. No integration tests for 02a → 02b → 02c pipeline
2. `extraction.py` has ZERO unit tests
3. No JSON schema validation

#### 🟠 High (8)
1. `interpolate_features_within_tree()` missing validation
2. `compute_chm_engineered_features()` percentile edge case
3. `identify_redundant_features()` naive selection
4. `validate_split_stratification()` KL threshold not enforced
5. `compute_nearest_different_genus_distance()` O(n²) complexity
6. `selection.py` has ZERO unit tests
7. No notebook execution tests
8. No final NaN check in 02c

#### 🟡 Medium (12)
1. `extract_sentinel_features()` band ordering assumption
2. `correct_tree_positions()` no validation
3. `filter_nan_trees()` feature base extraction brittle
4. Missing VIF validation function
5. Missing covariance singularity check
6. Block ID collision risk
7. quality.py edge cases undertested
8. splits.py edge cases undertested
9. Notebook skip logic not validated
10. Inconsistent JSON loading
11. No metadata preservation tests
12. Mahalanobis genus-level covariance borderline leakage

#### 🟢 Low (7)
1. Magic batch size
2. Mahalanobis hardcoded min samples
3. `analyze_genus_specific_impact()` no uniformity check
4. Various code quality improvements

---

## Priority Matrix

### Week 1: Must Fix (Critical + High Priority)

- [x] **Create integration test suite** (`tests/integration/test_phase2_pipeline.py`)
  - [x] Test 02a → 02b schema compatibility
  - [x] Test exploratory JSON → runner consumption
  - [x] Test final 10 GeoPackages schema

- [x] **Create `test_extraction.py`** with minimum 15 tests
  - [x] Adaptive radius calculation
  - [x] Position correction scoring
  - [x] CHM extraction
  - [x] S2 multi-month extraction

- [x] **Implement JSON schema validation**
  - [x] Create schema files for all 6 JSONs
  - [x] Add validation utilities
  - [x] Wire into pipeline modules (notebooks can import the validators)

- [x] **Add final NaN assertion** in 02c pipeline validation

- [x] **Fix O(n²) proximity computation** with STRtree

### Week 2: Should Fix (Medium Priority)

- [x] Create `test_selection.py` with minimum 8 tests
- [x] Add feature importance parameter to `identify_redundant_features()`
- [x] Implement VIF validation function
- [x] Add schema validation between pipeline stages
- [x] Add metadata preservation tests
- [x] Document exact final output schema

### Week 3: Nice to Have (Low Priority)

- [x] Make batch_size configurable
- [x] Add covariance singularity warnings
- [x] Improve edge case test coverage
- [x] Add notebook execution tests

---

## Recommendations

### Architecture
✅ **GOOD:** Clean module separation, clear responsibilities
✅ **GOOD:** Consistent error handling patterns
✅ **GOOD:** CRS validation throughout
⚠️ **IMPROVE:** Add schema validation layer between modules

### Testing
❌ **CRITICAL:** Integration tests completely missing
❌ **CRITICAL:** extraction.py untested
⚠️ **NEEDS WORK:** Edge case coverage
✅ **GOOD:** Existing tests are well-structured

### Documentation
✅ **EXCELLENT:** Module docstrings
✅ **GOOD:** PRD-to-code traceability
⚠️ **IMPROVE:** Final output schema not precisely documented

### Data Quality
✅ **EXCELLENT:** Data leakage prevention
✅ **GOOD:** NaN handling methodology
⚠️ **IMPROVE:** Missing final validation assertions

---

## Conclusion

**Overall Grade: B+**

Phase 2 implementation is **architecturally sound** and **methodologically correct**, but has **gaps in testing and validation** that could lead to silent failures in production.

**Primary Risk:** Without integration tests, a notebook could produce malformed output that breaks the next stage **without detection until Phase 3**.

**Recommendation:** Prioritize **Week 1 fixes** (integration tests, extraction tests, JSON validation) before proceeding to Phase 3.

---

**Review Completed:** 2026-01-31
**Next Action:** Address Critical issues, then proceed with methodological improvements (PRD 002d)
