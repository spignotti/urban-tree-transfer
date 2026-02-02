# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed - Phase 2: Code Review Fixes

- Fix tile-based CHM correction window bounds handling for negative offsets
- Fix edge NaN logic in `filter_nan_trees()` to use per-tree edge NaN counting (quality.py:166-234)
  - Add `max_edge_nan_months` parameter to filter trees with excessive edge NaN values
  - Add `min_valid_months` parameter to ensure sufficient data for interpolation
  - Previously had no edge NaN validation, now validates both edge NaN and minimum valid months
- Fix syntax errors in runner notebooks (02a, 02b, 02c) caused by broken f-string literals
  - Repair unterminated string literals split across lines
  - Fix broken print statements in summary sections
- Improve JSON existence checks in 02c_final_preparation.ipynb
  - Add comprehensive validation for all required exploratory JSON configs
  - Provide clear error messages indicating which notebook to run if JSON is missing
- Update test_filter_nan_trees_ignores_non_temporal_columns to account for new filtering parameters
  - Test now explicitly sets min_valid_months=1 to match test expectations

### Added - Phase 2: Feature Engineering

- Add geometric definition, pixel footprint visualization, and threshold sensitivity curve to exp_06 proximity analysis
- Add notebook test dependencies and enable notebook execution tests in nox
- Add notebook test mode guard to runner notebooks for non-Colab execution
- Add macOS quarantine cleanup in nox to allow scikit-learn imports in CI
- Add post-split spatial independence validation (Moran's I) to exp_05 with JSON output and visualization
- Add redundancy removal, outlier flagging, and spatial split implementations for final preparation
- Add JSON schema files and validation utilities for Phase 2 exploratory outputs
- Add Phase 2 schema validation helpers for 02a/02b/02c pipeline stages
- Add zero-NaN validation utility for final outputs
- Add extraction, selection, and Phase 2 integration test suites
- Add runner notebook execution smoke tests (opt-in via env var)
- Add Phase 2 final output schema documentation
- Add unit tests for selection, outlier detection, and spatial splits modules
- Add exploratory notebook `notebooks/exploratory/exp_01_temporal_analysis.ipynb` for temporal feature selection via JM distance
- Add runner notebook `notebooks/runners/02a_feature_extraction.ipynb` for CHM and Sentinel-2 feature extraction
- Add runner notebook `notebooks/runners/02b_data_quality.ipynb` for Phase 2b quality control pipeline:
  - 12-step sequential quality pipeline (genus filter, plant year filter, temporal selection, NaN handling, interpolation, CHM engineering, NDVI plausibility)
  - Within-tree temporal interpolation (no data leakage)
  - Per-city skip logic with retention rate validation (>85%)
  - Dual execution logs (JSON + plain text format)
  - Metadata preservation throughout pipeline (11 columns)
  - Output: `trees_clean_{city}.gpkg` (0 NaN, quality-assured datasets)
- Add exploratory notebook `notebooks/exploratory/exp_01_temporal_analysis.ipynb` for JM-based temporal feature selection
- Add exploratory notebook `notebooks/exploratory/exp_02_chm_assessment.ipynb` for CHM assessment (eta², Cohen's d, plant-year threshold, genus classification)
- Add CHM assessment methodology documentation: `docs/documentation/02_Feature_Engineering/02_Exploratory_02_CHM_Assessment.md`
- Add data quality module implementation and tests for temporal filtering, NaN handling, and CHM/NDVI checks
- Add data quality methodology documentation: `docs/documentation/02_Feature_Engineering/02b_Data_Quality_Methodik.md`
- Add `configs/features/feature_config.yaml` with complete feature engineering configuration:
  - Metadata columns (9 fields preserved through pipeline)
  - CHM features (1m resolution, engineered z-score/percentile)
  - Sentinel-2 spectral bands (10) and vegetation indices (13)
  - Temporal configuration (12 months extraction)
  - Quality thresholds (NaN handling, NDVI plausibility, correlation)
  - Outlier detection parameters (Z-score, Mahalanobis, IQR, consensus)
  - Spatial split configuration (500m blocks, city-specific ratios)
  - Genus classification (deciduous/coniferous lists)
  - Tree position correction parameters
- Add feature config loader functions to `config/loader.py`:
  - `load_feature_config()` - Load feature engineering YAML
  - `get_metadata_columns()` - Get preserved metadata column names
  - `get_spectral_bands()` - Get S2 band names
  - `get_vegetation_indices()` - Get vegetation index names
  - `get_all_s2_features()` - Get combined S2 features (23 total)
  - `get_temporal_feature_names()` - Get temporal feature names with month suffix
  - `get_chm_feature_names()` - Get CHM feature names
  - `get_all_feature_names()` - Get complete feature list (277 for 12 months)
- Add exploratory notebook `notebooks/exploratory/exp_03_correlation_analysis.ipynb` for correlation-based redundancy detection:
  - Stacked temporal-vector correlations (all months concatenated per feature base)
  - Pearson correlation with threshold 0.95 (Kuhn & Johnson 2013)
  - Cross-city consistency requirement (features removed only if redundant in both cities)
  - Temporal consistency (all month instances removed if base is redundant)
  - Interpretability-based selection preferences (NDVI > EVI, B8 > B8A)
  - Family constraint for vegetation indices (≥1 retained per family)
  - VIF validation (target < 10) for retained features
  - Clustered heatmaps with bold borders for |r| > 0.95
  - Output: `correlation_removal.json` with complete temporal removal list
- Add exploratory notebook `notebooks/exploratory/exp_04_outlier_thresholds.ipynb` for outlier detection threshold validation:
  - Tripartite detection system (Z-score, Mahalanobis D², IQR/Tukey)
  - Sensitivity analysis for threshold parameters (Z=3.0, α=0.001, k=1.5)
  - Genus-specific Mahalanobis detection with chi² critical values
  - Genus×city-specific IQR detection for structural outliers
  - Severity-based flagging system (high/medium/low/none) for ablation studies
  - Method overlap analysis via Venn diagram
  - Genus-wise impact validation (uniformity check)
  - No automatic removal - all trees retained with flags for Phase 3 experiments
  - 7 publication-quality plots (sensitivity, distributions, Venn, rates, severity, genus)
  - Output: `outlier_thresholds.json` with validated parameters and flagging statistics
- Add biological context analysis for outlier thresholds (age, tree type, spatial clustering, genus patterns) with JSON output
- Add exploratory notebook `notebooks/exploratory/exp_05_spatial_autocorrelation.ipynb` for Moran's I spatial autocorrelation analysis:
  - Distance lag analysis (100-1200m) to identify autocorrelation decay distance
  - Representative feature analysis (NDVI, CHM, spectral bands)
  - Per-city Moran's I curves with permutation-based significance testing
  - Data-driven block size recommendation (decay distance + safety buffer)
  - 3 publication-quality plots (distance decay, feature comparison, spatial patterns)
  - Output: `spatial_autocorrelation.json` with recommended_block_size_m
- Add exploratory notebook `notebooks/exploratory/exp_06_mixed_genus_proximity.ipynb` for mixed-genus proximity analysis:
  - Nearest different-genus distance computation for all trees
  - Threshold sensitivity analysis [5m, 10m, 15m, 20m, 30m]
  - Retention rate vs. spectral purity trade-off validation
  - Genus-specific impact uniformity check (max deviation < 10%)
  - Sentinel-2 pixel contamination zone visualization (10m × 10m pixels)
  - 5 publication-quality plots (distribution, retention, genus impact, spatial, contamination schematic)
  - Output: `proximity_filter.json` with recommended_threshold_m (~20m expected)
- Add proximity filter module `feature_engineering/proximity.py` with three functions:
  - `compute_nearest_different_genus_distance()` - Per-tree distance to nearest different genus
  - `apply_proximity_filter()` - Filter trees by threshold with retention statistics
  - `analyze_genus_specific_impact()` - Per-genus removal rate validation
  - Includes 5 unit tests for distance computation, filtering logic, and impact analysis
- Add runner notebook `notebooks/runners/02c_final_preparation.ipynb` for final ML-ready dataset creation:
  - 18-cell pipeline: load configs → remove redundancy → flag outliers → assign blocks → create splits (baseline + filtered) → validate → export
  - Dual dataset strategy: 10 GeoPackages (5 baseline + 5 filtered variants for ablation studies)
  - Berlin: train/val/test (70/15/15), Leipzig: finetune/test (80/20)
  - Spatial disjointness validation (spatial_overlap == 0 assertion)
  - Genus stratification validation (KL-divergence < 0.01)
  - Outlier flagging only (trees_removed == 0, metadata added for Phase 3 experiments)
  - Proximity filtering with data-driven threshold from exp_06
  - 2 summary visualizations (split size comparison, genus distribution)
  - Output: `phase_2_final_summary.json` with complete validation metrics
- Add spatial autocorrelation methodology documentation: `docs/documentation/02_Feature_Engineering/02_Exploratory_05_Spatial_Autocorrelation.md`
- Add mixed-genus proximity methodology documentation: `docs/documentation/02_Feature_Engineering/02_Exploratory_06_Mixed_Genus_Proximity.md`
- Add final preparation methodology documentation: `docs/documentation/02_Feature_Engineering/02c_Final_Preparation_Methodik.md` covering:
  - Complete module architecture (selection.py, outliers.py, splits.py, proximity.py)
  - Mathematical foundations (Moran's I, KL-divergence, Mahalanobis D², Tukey fences)
  - Dual dataset strategy rationale and Phase 3 experiment matrix
  - Quality assurance protocols and validation assertions
  - Phase 3 integration examples for baseline/filtered/transfer-learning experiments
- Add outlier detection methodology documentation: `docs/documentation/02_Feature_Engineering/02_Exploratory_04_Outlier_Detection.md`
- Add correlation analysis methodology documentation: `docs/documentation/02_Feature_Engineering/02_Exploratory_03_Correlation_Analysis.md`
- Add `feature_engineering/extraction.py` module stub with function signatures:
  - `correct_tree_positions()` - Snap trees to CHM local maxima
  - `extract_chm_features()` - 1m point sampling from CHM
  - `extract_sentinel_features()` - Monthly S2 band/index extraction
  - `extract_all_features()` - Complete extraction pipeline
- Add `feature_engineering/quality.py` module stub with function signatures:
  - `apply_temporal_selection()` - Filter to selected months
  - `interpolate_nan_values()` - Within-tree temporal interpolation (no data leakage)
  - `engineer_chm_features()` - Add z-score and percentile features
  - `filter_by_ndvi_plausibility()` - Remove low-NDVI trees
  - `filter_by_plant_year()` - Remove recently planted trees
  - `run_quality_pipeline()` - Complete quality control pipeline
- Add `feature_engineering/selection.py` module stub with function signatures:
  - `compute_feature_correlations()` - Correlation matrices by feature group
  - `identify_redundant_features()` - Find highly correlated pairs
  - `remove_redundant_features()` - Drop redundant columns
- Add `feature_engineering/outliers.py` module stub with function signatures:
  - `detect_zscore_outliers()` - Z-score based detection
  - `detect_mahalanobis_outliers()` - Multivariate distance detection
  - `detect_iqr_outliers()` - IQR-based detection (for CHM)
  - `classify_outliers_by_consensus()` - Multi-method consensus classification
  - `remove_outliers()` - Remove based on classification
- Add `feature_engineering/splits.py` module stub with function signatures:
  - `create_spatial_blocks()` - Regular grid creation
  - `assign_trees_to_blocks()` - Spatial join trees to blocks
  - `create_stratified_spatial_splits()` - Stratified splitting with spatial disjointness
  - `validate_splits()` - Check disjointness and stratification quality
- Implement feature extraction module functions for tree correction, CHM sampling, Sentinel-2 extraction, and extraction summaries
- Add comprehensive unit test suite for `feature_engineering/extraction.py` (15 tests, 319 lines):
  - Tree position correction tests (empty geometry, scoring, adaptive radius, metadata validation)
  - CHM feature extraction tests (batch processing, NoData handling, missing files)
  - Sentinel-2 extraction tests (multi-month, band validation, missing file warnings)
  - Full pipeline integration tests
- Add comprehensive unit test suite for `feature_engineering/selection.py` (9 tests, 127 lines):
  - Correlation matrix computation tests
  - Redundancy identification tests (with/without feature importance)
  - VIF validation tests
  - Geometry column preservation tests
- Add integration test suite `tests/integration/test_phase2_pipeline.py`:
  - 02a → 02b → 02c pipeline schema compatibility tests
  - JSON schema consumption validation (exploratory → runner notebooks)
  - Metadata preservation tests across pipeline stages
  - Final GeoPackage schema validation (10 output files)
- Add schema validation utilities in `utils/`:
  - `schema_validation.py` - Phase 2a/2b/2c output validators with CRS and column checks
  - `json_validation.py` - JSON schema validation for 6 exploratory outputs
  - `final_validation.py` - Zero-NaN validation for ML-ready datasets
- Add 6 JSON schema files in `src/urban_tree_transfer/schemas/`:
  - `temporal_selection.schema.json` - exp_01 output validation
  - `chm_assessment.schema.json` - exp_02 output validation
  - `correlation_removal.schema.json` - exp_03 output validation
  - `outlier_thresholds.schema.json` - exp_04 output validation
  - `spatial_autocorrelation.schema.json` - exp_05 output validation
  - `proximity_filter.schema.json` - exp_06 output validation
- Add notebook idempotency tests in `tests/notebooks/test_runner_idempotency.py`:
  - Validate skip-if-exists logic for 02a, 02b, 02c runner notebooks
  - Ensure notebooks skip execution when outputs already exist
- Add data leakage documentation to Mahalanobis outlier detection (outliers.py):
  - Clarify why genus-level covariance across splits is acceptable
  - Document that outlier flags are metadata, not predictive features
  - Note for Phase 3: outlier flags should NOT be used as model inputs
- Add uniformity check warning to proximity filter analysis (proximity.py):
  - Validate genus-specific removal rates have max deviation < 10%
  - Warn when proximity filter disproportionately affects certain genera
  - Ensures filter maintains balanced genus representation
- Add methodological improvements PRD: `docs/documentation/02_Feature_Engineering/PRD_002d_Methodological_Improvements.md`:
  - Cross-city JM distance consistency validation
  - Spatial independence validation for block splits
  - Genus-specific CHM normalization (planned for future)
  - Biological context analysis for outlier interpretation
  - Geometric clarity in proximity filter
- Add technical review documentation: `docs/documentation/02_Feature_Engineering/reviews/Technical_Review_Phase2.md`:
  - Comprehensive review of all Phase 2 modules, tests, notebooks, and data flow
  - 30 identified issues (3 Critical, 8 High, 12 Medium, 7 Low)
  - Complete resolution tracking with implementation status
  - Grade: A (excellent implementation after fixes)

### Changed - Phase 2: Feature Engineering

- Change tree position correction to legacy snap-to-peak logic with local maxima scoring
  - Use P90 adaptive radius without safety factor and 5m sampling radius
  - Add minimum peak height filter and local maxima footprint
  - Add tile-based CHM processing for large-scale performance
- Implement genus-specific CHM normalization in quality pipeline (PRD 002d Improvement 3):
  - Change from city-level to genus×city-level normalization for `CHM_1m_zscore` and `CHM_1m_percentile`
  - Prevents genus-leakage by removing genus-mean effect from features
  - Rare genus fallback (<10 samples) to city-level normalization with logged warnings
  - Add comprehensive unit tests for genus-specific normalization and edge cases
  - Update CHM Assessment documentation with genus-specific normalization rationale
  - Add validation plot specification: `chm_percentile_genus_validation.png`
- Update PRD 002c (Final Preparation) to load spatial block size from `spatial_autocorrelation.json` instead of `feature_config.yaml`
- Update Implementation Plan to clarify data-driven block size determination workflow via Moran's I analysis
- Spatial block size now empirically determined in exp_05, not hardcoded
- Add cross-city JM consistency validation to exp_01 temporal selection (PRD 002d Improvement 1):
  - Spearman rank correlation between Berlin/Leipzig JM rankings
  - Consistency-aware month selection (ρ > 0.7 threshold)
  - New visualization: `jm_rank_consistency.png`
  - Extend `temporal_selection.json` with `cross_city_validation` section
- Add post-split spatial independence validation to exp_05 spatial autocorrelation (PRD 002d Improvement 2):
  - Within-split Moran's I validation for train/val/test splits (|I| < 0.1 threshold)
  - Between-split boundary Moran's I validation (|I| < 0.05 threshold)
  - Iterative block size refinement logic (20% increments, optional)
  - New visualization: `split_spatial_independence.png` (3-panel split comparison)
  - Extend `spatial_autocorrelation.json` with `validation.post_split` section

### Added - Phase 1: Data Processing

- Add city YAML configs for Berlin and Leipzig
- Add config constants and loader utilities
- Add plotting and execution logging utilities
- Add Phase 1 data processing modules (boundaries, trees, elevation, CHM, Sentinel-2)
- Add Phase 1 runner notebook and runner template
- Add methodology template and Phase 1 data processing methodology
- Add Leipzig WFS tree mapping with species extraction and metadata export
- Add tree_type mapping support for Berlin WFS layers
- Add zip-list elevation download with mosaic + boundary clipping for Leipzig DGM
- Add Leipzig DOM zip-list configuration
- Add Berlin DOM/DGM Atom feed URLs from legacy config
- Add `utils/geo.py` with shared geographic utilities (ensure_project_crs, buffer_boundaries, validate_geometries, clip_to_boundary)
- Add `utils/validation.py` with dataset validation utilities (validate_crs, validate_schema, validate_within_boundary, generate_validation_report)
- Add `validate_polygon_geometries()` to boundaries module using shapely make_valid
- Add `filter_trees_to_boundary()` for spatial filtering with configurable buffer
- Add `remove_duplicate_trees()` for deduplication by tree_id or proximity
- Add Atom feed parser for Berlin elevation data with spatial tile filtering
- Add `clip_chm_to_boundary()` for CHM boundary clipping
- Add `monitor_tasks()` and `batch_validate_sentinel()` for GEE task management
- Add 500m buffer to all boundary-based clipping operations
- Add `outputs/` directory for Colab-generated metadata and logs
- Add runtime settings section to runner notebook (CPU, High-RAM recommended)
- Add integration and unit tests for data processing endpoints and helpers
- Add unit tests for XYZ to GeoTIFF conversion with various data scenarios
- Add integration tests that download single Berlin and Leipzig DOM/DGM tiles to verify full pipeline
- Add `scripts/gee_sentinel_preview.js` for manual GEE Sentinel-2 preview (Berlin/Leipzig)

### Changed
- Change documentation structure to include Phase 1 methodology
- Change nox sessions to run `uv` with active environment to avoid warnings
- Change notebook to clone repo for metadata commits (large data stays on Drive)
- Change harmonize_trees to use consistent dtypes (Int64, Float64) across cities
- Change Sentinel-2 export to use 500m buffered boundaries
- Change Sentinel-2 spectral band constants to match GEE band names (B2, B3, ...)
- Update Phase 1 methodology with complete processing steps, parameters, and quality criteria
- Change data processing runner notebook to skip sections when outputs already exist
- Change elevation downloads to log per-tile progress
- Refactor tree position correction to adaptive radius + scoring, including correction distance metadata

### Fixed
- Fix Berlin Atom feed downloads by resolving section links to ZIP tiles and validating ZIP responses
- Fix Berlin DOM/DGM extraction by adding XYZ ASCII to GeoTIFF conversion (Berlin provides .txt/.xyz point clouds, not GeoTIFF)
- Fix Phase 2 interpolation to validate interior NaN resolution (quality.py:292-324)
  - Add post-interpolation validation that interior NaNs = 0
  - Raise RuntimeError if interpolation fails for non-edge positions
  - Prevents silent bugs in temporal interpolation logic
- Fix CHM engineered percentile edge cases with clamping (quality.py:382)
  - Add .clip(0.0, 100.0) to percentile computation to handle edge cases
  - Prevents percentileofscore returning values outside [0, 100] range
- Fix redundant feature identification with importance-based selection (selection.py:91-96)
  - Add feature_importance parameter to identify_redundant_features()
  - Remove less important feature when correlations exceed threshold
  - Previously always removed feature_b regardless of importance
- Fix feature base extraction brittleness in filter_nan_trees (quality.py:183-189)
  - Validate suffix is exactly 2 digits before treating as month suffix
  - Handles multi-underscore features like CHM_1m_zscore_05 correctly
  - Previously failed with rsplit("_", 1) on features with multiple underscores
- Fix split stratification validation to enforce KL threshold
- Fix mixed-genus proximity computation performance with STRtree (O(n²) → O(n log n))
- Fix VIF validation availability for selection outputs
- Fix tree position correction by validating correction distances
- Fix Sentinel-2 extraction by validating band count and order
  - Add runtime validation that band count matches expected features
  - Add band order validation to catch Phase 1 output mismatches
- Fix Leipzig DOM/DGM downloads by using browser-like headers for Nextcloud/WebDAV server
- Fix harmonize_elevation block size error by setting explicit 256x256 tile blocks (TIFF requires multiples of 16)
- Fix large raster file handling by adding BIGTIFF=IF_SAFER to mosaic, clip, and reproject operations

### Changed
- Add parallel downloads for elevation tiles (4 workers by default, configurable)
- Add retry logic with exponential backoff for failed downloads (3 retries)
- Add resume support for elevation downloads (skips already downloaded tiles)
- Elevation downloads now continue with partial data if some tiles fail
## [0.1.0] - YYYY-MM-DD

### Added
- Initial project setup
- Basic project structure
- Core functionality implementation
- Development environment configuration (UV, Nox, Ruff, Pyright)
- Test framework setup (pytest, coverage)
- Documentation (README.md, CLAUDE.md)

---

## Guidelines for Maintaining This Changelog

### Version Format
- Use [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH
  - **MAJOR**: Incompatible API changes
  - **MINOR**: Backwards-compatible new features
  - **PATCH**: Backwards-compatible bug fixes

### Categories
Use these standard categories in order:
1. **Added** - New features
2. **Changed** - Changes in existing functionality
3. **Deprecated** - Soon-to-be removed features
4. **Removed** - Removed features
5. **Fixed** - Bug fixes
6. **Security** - Security-related changes

### Writing Entries
- Write entries in **present tense** (e.g., "Add feature" not "Added feature")
- Start each entry with a **verb** (Add, Change, Fix, Remove, etc.)
- Be **specific** and **concise**
- Include **issue/PR references** when applicable: `(#123)`
- Group related changes together
- Keep the audience in mind (users, not developers)

### Examples

#### Good Entries ✅
```markdown
### Added
- Add user authentication with JWT tokens (#45)
- Add support for PostgreSQL database backend
- Add comprehensive API documentation with examples

### Fixed
- Fix memory leak in data processing pipeline (#67)
- Fix incorrect calculation in statistics module (#72)
```

#### Bad Entries ❌
```markdown
### Added
- Added stuff
- Various improvements
- Updated code

### Fixed
- Fixed bug (too vague)
- Refactored everything (not user-facing)
```

### When to Update
- Update `[Unreleased]` section as you develop
- Create a new version section when releasing
- Move items from `[Unreleased]` to the new version
- Update the version links at the bottom

### Version Links (Optional)
If using Git tags, add comparison links at the bottom:
```markdown
[unreleased]: https://github.com/username/repo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/username/repo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/username/repo/releases/tag/v0.1.0
```

---

## Template for New Releases

When releasing a new version, copy this template:
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- 

### Changed
- 

### Deprecated
- 

### Removed
- 

### Fixed
- 

### Security
- 
```

Remove empty sections before publishing.
