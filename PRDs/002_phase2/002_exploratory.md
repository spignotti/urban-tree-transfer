# PRD 002_exploratory: Exploratory Analysis Notebooks

**Parent PRD:** [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md)
**Status:** Draft
**Created:** 2026-01-28
**Last Updated:** 2026-01-28
**Branch:** `feature/phase2-exploratory`

---

## Goal

Conduct exploratory analyses to determine optimal parameters and validate methodological decisions for the Feature Engineering pipeline. Generate configuration files consumed by runner notebooks.

**Success Criteria:**

- [ ] `temporal_selection.json` produced with JM distance analysis
- [ ] `chm_assessment.json` produced with η² and Cohen's d analysis
- [ ] `correlation_removal.json` produced with intra-class correlation analysis
- [ ] `outlier_thresholds.json` produced with sensitivity analysis
- [ ] `spatial_autocorrelation.json` produced with Moran's I analysis
- [ ] All thresholds statistically justified or literature-referenced
- [ ] **Publication-quality visualizations** saved for all analyses
- [ ] Plots follow consistent style (see [Visualization Strategy](../../docs/documentation/02_Feature_Engineering/01_Visualization_Strategy.md))

---

## Context

**Read before implementation:**

- [002_phase2_feature_engineering_overview.md](../002_phase2_feature_engineering_overview.md) - Critical fixes and methodology
- [Visualization Strategy](../../docs/documentation/02_Feature_Engineering/01_Visualization_Strategy.md) - **REQUIRED:** Plot style and organization
- `CLAUDE.md` / `AGENT.md` - Project conventions

**Legacy Code (Inspiration Only):**

- `legacy/notebooks/02_feature_engineering/03a_temporal_feature_selection_JM.md` - JM distance (⚠️ has implementation issues)
- `legacy/notebooks/02_feature_engineering/03c_chm_relevance_assessment.md` - CHM analysis
- `legacy/notebooks/02_feature_engineering/03d_correlation_analysis.md` - Correlation analysis
- `legacy/notebooks/02_feature_engineering/03e_outlier_detection.md` - Outlier methods
- `legacy/documentation/02_Feature_Engineering/03_Temporal_Feature_Selection_JM_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/05_CHM_Relevance_Assessment_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/06_Correlation_Analysis_Redundancy_Reduction_Methodik.md`
- `legacy/documentation/02_Feature_Engineering/07_Outlier_Detection_Final_Filtering_Methodik.md`

**⚠️ Important:** Legacy JM implementation has known issues (values too low). Investigate and fix as time permits.

---

## Scope

### In Scope

- Parameter determination via exploratory analysis
- Statistical validation of thresholds
- Visualization for documentation
- Configuration file generation (JSON outputs)
- Cross-city consistency validation

### Out of Scope

- Runner notebook implementation (separate PRDs)
- Model training (Phase 3)
- Hyperparameter optimization (beyond threshold validation)

---

## Exploratory Notebooks

**Template:** [Exploratory-Notebook Template](../../docs/documentation/02_Feature_Engineering/02_Notebook_Templates.md#2-exploratory-notebook-template-phase-2)

All exploratory notebooks follow a **consistent structure** based on the Runner-Notebook template with additional visualization and JSON-output capabilities.

**Key differences from Runner notebooks:**

- Output directory: `outputs/phase_2` (not `data/phase_2_*`)
- Figures directory: `outputs/phase_2/figures/exp_XX_*` (notebook-specific)
- Multiple publication-quality plots per section
- JSON configurations as primary output
- Manual sync required (JSONs needed by subsequent Runner notebooks)

**Visualization requirements:** See [Visualization Strategy](../../docs/documentation/02_Feature_Engineering/01_Visualization_Strategy.md)  
**Template documentation:** See [Notebook Templates](../../docs/documentation/02_Feature_Engineering/02_Notebook_Templates.md)

---

### exp_01_temporal_analysis.ipynb

**Purpose:** Determine optimal months using Jeffries-Matusita (JM) distance

**Inputs:**

- `data/phase_2_features/trees_with_features_{city}.gpkg` (from PRD 002a)

**Analysis Tasks:**

1. **JM Distance Calculation (per feature × month × genus pair)**
   - Formula: `JM = 2 * (1 - e^(-B))` where `B = Bhattacharyya distance`
   - Bhattacharyya: `B = 0.25 * ln(0.25 * (σ1²/σ2² + σ2²/σ1² + 2)) + 0.25 * ((μ1-μ2)² / (σ1² + σ2²))`
   - Compute for all genus pairs per feature per month
   - **Known Issue:** Legacy JM values were consistently low (0.5-1.2). Validate implementation:
     - Test with synthetic data (known separability)
     - Compare with Bruzzone et al. (1995) reference
     - Check numerical stability (log operations, small variances)

2. **Aggregation**
   - Mean JM per feature × month across all genus pairs
   - Identify months with high discriminability
   - Visualize seasonal patterns

3. **Cross-City Validation**
   - Compare Berlin vs Leipzig JM patterns
   - Ensure selected months are consistent across cities

4. **Month Selection**
   - Threshold-based selection (e.g., JM > 1.4)
   - Or: Top N months by mean JM
   - Document rationale

**Visualizations (REQUIRED):**

1. **JM Distance Heatmap (per city)**
   - X: Months (1-12), Y: Feature groups
   - Color: Mean JM distance
   - Annotations: Selected months marked
   - Files: `jm_heatmap_berlin.png`, `jm_heatmap_leipzig.png`

2. **Monthly JM Comparison (Berlin vs Leipzig)**
   - Line plot with error bars
   - Shows cross-city consistency
   - File: `jm_monthly_comparison.png`

3. **Feature Group Discriminability**
   - Box plots per feature group
   - Highlights best-performing groups
   - File: `jm_feature_groups.png`

4. **Temporal Selection Summary**
   - Bar chart with threshold line
   - Selected months highlighted
   - File: `temporal_selection_summary.png`

**Example Visualization Code:**

```python
from urban_tree_transfer.utils.plotting import setup_plotting, save_figure, PUBLICATION_STYLE
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup plotting
setup_plotting()
FIGURES_DIR = Path("/content/drive/MyDrive/urban-tree-transfer/outputs/phase_2/figures/exp_01_temporal")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Plot 1: JM Heatmap
fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])
sns.heatmap(jm_matrix, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
ax.set_title("JM Distance: Temporal Discriminability (Berlin)", fontsize=14, fontweight="bold")
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Feature Group", fontsize=12)
save_figure(fig, FIGURES_DIR / "jm_heatmap_berlin.png")

# ... more plots ...
```

**Outputs:**

```json
{
  "selected_months": [3, 4, 5, 6, 7, 8, 9, 10],
  "selection_method": "threshold",
  "threshold": 1.4,
  "jm_statistics": {
    "per_month": {...},
    "per_feature_group": {...}
  },
  "validation": {
    "berlin_leipzig_correlation": 0.87
  }
}
```

**Visualizations:**

- Heatmap: JM distance per feature group × month
- Line plot: Mean JM per month (Berlin vs Leipzig)
- Box plot: JM distribution per month

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/03a_temporal_feature_selection_JM.md` and `legacy/documentation/02_Feature_Engineering/03_Temporal_Feature_Selection_JM_Methodik.md`

**Note:** If JM implementation issues persist, document findings but still select months based on available results. Runner notebooks will use output regardless.

---

### exp_02_chm_assessment.ipynb

**Purpose:** Evaluate CHM feature quality, cross-city transferability, **plant year threshold**, and **genus classification**

**Inputs:**

- `data/phase_2_features/trees_with_features_{city}.gpkg` (from PRD 002a)

**Analysis Tasks:**

1. **Discriminative Power (ANOVA η²)**
   - One-way ANOVA: CHM_1m ~ genus_latin (per city)
   - Compute η² (effect size): variance explained by genus
   - Interpretation: η² > 0.06 = medium effect, > 0.14 = large effect

2. **Cross-City Consistency (Cohen's d)**
   - Compare genus mean CHM between Berlin and Leipzig
   - Cohen's d = (μ_Berlin - μ_Leipzig) / pooled_std
   - Interpretation: |d| < 0.2 = small difference (good for transfer)

3. **Feature Engineering Validation**
   - Validate CHM_1m extraction quality (check against cadastre height_m)
   - Correlation: should be moderate (r ~ 0.4-0.6), not too high (contamination)
   - Test Z-score and percentile transformations

4. **Decision: Include CHM Features?**
   - If η² high AND Cohen's d low → good for transfer
   - If η² high BUT Cohen's d high → transfer risk (city-specific)
   - If η² low → low discriminative power

5. **Plant Year Threshold Determination (NEW)**
   - Plot median CHM_1m by plant_year cohort
   - Identify detection threshold (~2m for Sentinel-2 10m visibility)
   - Find breakpoint year where median CHM drops below detection threshold
   - Validate with sample counts per year
   - **Output:** `recommended_max_plant_year` (e.g., 2018)

6. **Genus Classification & Scope (NEW)**
   - Inventory all genera with sample counts per city
   - Classify each genus as deciduous/coniferous using lookup table
   - Apply threshold check:
     - If `n_conifer_genera < 3` OR `n_conifer_samples < 500` → exclude conifers
   - **Output:** `analysis_scope` ("deciduous_only" or "all")

**Visualizations (REQUIRED):**

1. **CHM Distribution by Genus**
   - Violin/box plots per genus, faceted by city
   - Shows discriminative power visually
   - File: `chm_boxplot_per_genus.png`

2. **CHM vs Cadastre Correlation**
   - Scatter plot with regression line
   - Separate colors per city
   - Annotated with r value
   - File: `chm_cadastre_correlation.png`

3. **Discriminative Power (η²) Comparison**
   - Bar chart comparing Berlin vs Leipzig
   - Effect size interpretation guidelines
   - File: `eta2_comparison.png`

4. **Cohen's d Forest Plot**
   - Forest plot per genus
   - Confidence intervals
   - Reference lines (small/medium/large)
   - File: `cohens_d_forest_plot.png`

5. **CHM Distribution Comparison**
   - Overlaid histograms (Berlin vs Leipzig)
   - Shows cross-city consistency
   - File: `chm_distribution_cities.png`

6. **CHM vs Plant Year (NEW)**
   - Scatter/box plot: CHM_1m by plant_year
   - Detection threshold line (2m)
   - Recommended cutoff year highlighted
   - File: `chm_vs_plant_year.png`

7. **Genus Inventory (NEW)**
   - Bar chart: sample counts per genus, colored by type (deciduous/coniferous)
   - Threshold lines for min_samples
   - File: `genus_inventory.png`

**Outputs:**

```json
{
  "chm_features": ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"],
  "include_chm": true,
  "discriminative_power": {
    "berlin_eta2": 0.18,
    "leipzig_eta2": 0.16
  },
  "transfer_risk": {
    "cohens_d_mean": 0.12,
    "interpretation": "low transfer risk"
  },
  "validation": {
    "chm_cadastre_correlation": 0.53
  },
  "plant_year_analysis": {
    "detection_threshold_m": 2.0,
    "median_chm_by_year": {"2015": 8.2, "2016": 6.5, "2017": 4.1, "2018": 2.8, "2019": 1.5},
    "recommended_max_plant_year": 2018,
    "justification": "Trees planted after 2018 have median CHM < 2m in 2021 imagery"
  },
  "genus_inventory": {
    "berlin": {"TILIA": 45000, "ACER": 32000, "PINUS": 1200, ...},
    "leipzig": {"TILIA": 12000, "ACER": 8500, "PINUS": 450, ...},
    "classification": {
      "deciduous": ["TILIA", "ACER", "QUERCUS", ...],
      "coniferous": ["PINUS", "PICEA"]
    },
    "conifer_analysis": {
      "n_genera": 2,
      "n_samples": 1650,
      "include_in_analysis": false,
      "reason": "n_genera < 3"
    },
    "analysis_scope": "deciduous_only"
  }
}
```

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/03c_chm_relevance_assessment.md` and `legacy/documentation/02_Feature_Engineering/05_CHM_Relevance_Assessment_Methodik.md`

---

### exp_03_correlation_analysis.ipynb

**Purpose:** Identify redundant features via intra-class AND cross-class correlation analysis

**Inputs:**

- `data/phase_2_features/trees_clean_{city}.gpkg` (from PRD 002b)

**Analysis Tasks:**

1. **Feature Grouping**
   - Group 1: Spectral bands (B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12)
   - Group 2: Broadband VIs (NDVI, EVI, GNDVI, kNDVI, VARI)
   - Group 3: Red-edge VIs (NDre1, NDVIre, CIre, IRECI, RTVIcore)
   - Group 4: Water VIs (NDWI, MSI, NDII)

2. **Intra-Class Correlation**
   - Compute Pearson correlation within each group
   - Threshold: |r| > 0.95 for redundancy
   - Rationale: Features within same group measure similar spectral properties

3. **Cross-Class Correlation (NEW)**
   - Compute Pearson correlation BETWEEN groups (bands vs. each VI group)
   - Identify band-VI pairs with |r| > 0.95
   - Apply priority rule: **Always prefer VIs over raw bands** (VIs encode domain knowledge)
   - Document cross-class redundancy

4. **Priority Rules for Removal (Complete Hierarchy)**

   **Rule 1: Spatial Resolution (Bands only)**
   - Prefer 10m bands (B2, B3, B4, B8) over 20m bands (B5, B6, B7, B8A, B11, B12)

   **Rule 2: Vegetation Index Priority (within VI groups)**
   - Broadband VIs: NDVI > EVI > GNDVI > VARI > kNDVI
   - Red-edge VIs: NDre1 > CIre > IRECI > NDVIre > RTVIcore
   - Water VIs: NDWI > NDII > MSI
   - Rationale: Biological interpretability and literature support

   **Rule 3: Cross-Class (Bands vs. VIs)**
   - Always keep VI, remove band (VIs have biological meaning)

   **Rule 4: Tiebreaker**
   - Higher variance → keep
   - If variance within 5% → alphabetically first

5. **Temporal Consistency**
   - Validate correlation patterns across selected months
   - Ensure removal doesn't affect phenological patterns

6. **Cross-City Validation**
   - Compare correlation matrices between Berlin and Leipzig
   - Ensure redundancy patterns are consistent

**Outputs:**

```json
{
  "features_to_remove": [
    "GNDVI_04", "GNDVI_05", ...,
    "NDVIre_04", "NDVIre_05", ...,
    "MSI_04", "MSI_05", ...,
    "B8A_04", "B8A_05", ...
  ],
  "intra_class_removal": {
    "GNDVI": {"correlated_with": "NDVI", "r": 0.97, "rule": "lower VI priority"},
    "NDVIre": {"correlated_with": "NDre1", "r": 0.96, "rule": "lower VI priority"},
    "MSI": {"correlated_with": "NDWI", "r": 0.94, "rule": "lower VI priority"},
    "B8A": {"correlated_with": "B8", "r": 0.98, "rule": "20m vs 10m"}
  },
  "cross_class_removal": {
    "B6": {"correlated_with": "NDre1", "r": 0.96, "rule": "band vs VI → remove band"},
    "B11": {"correlated_with": "MSI", "r": 0.95, "rule": "band vs VI → remove band"}
  },
  "priority_rules_applied": {
    "spatial_resolution": ["B8A → B8"],
    "vi_priority": ["GNDVI → NDVI", "NDVIre → NDre1", "MSI → NDWI"],
    "cross_class": ["B6 → NDre1", "B11 → MSI"]
  },
  "correlation_threshold": 0.95,
  "features_removed_count": 48,
  "features_retained_count": 173
}
```

**Visualizations (REQUIRED):**

1. **Intra-Class Correlation Heatmaps** (4 plots, one per group)
   - Spectral bands, Broadband VIs, Red-edge VIs, Water VIs
   - Annotated with r values
   - Mask for |r| > 0.95 (redundancy)
   - Files: `correlation_heatmap_spectral.png`, `correlation_heatmap_broadband_vi.png`, `correlation_heatmap_rededge_vi.png`, `correlation_heatmap_water_vi.png`

2. **Cross-Class Correlation Matrix (NEW)**
   - Heatmap: Bands (rows) vs. all VIs (columns)
   - Highlights |r| > 0.95 cross-class pairs
   - File: `correlation_cross_class.png`

3. **Redundant Feature Pairs Scatter**
   - Selected high-correlation pairs (intra- and cross-class)
   - Shows visual redundancy
   - File: `redundant_features_scatter.png`

4. **Variance Comparison**
   - Bar chart for redundant pairs
   - Justifies which to keep based on variance rule
   - File: `variance_comparison.png`

5. **Temporal Correlation Stability**
   - Line plot: correlation across months
   - Validates removal consistency
   - File: `temporal_correlation_stability.png`

6. **Priority Rule Application Summary (NEW)**
   - Sankey/flow diagram showing rule application
   - How many features removed by each rule
   - File: `priority_rules_summary.png`

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/03d_correlation_analysis.md` and `legacy/documentation/02_Feature_Engineering/06_Correlation_Analysis_Redundancy_Reduction_Methodik.md`

---

### exp_04_outlier_thresholds.ipynb

**Purpose:** Validate outlier detection thresholds and decision rules

**Inputs:**

- `data/phase_2_features/trees_clean_{city}.gpkg` (from PRD 002b)

**Analysis Tasks:**

1. **Z-Score Sensitivity Analysis**
   - Test thresholds: z ∈ [2.5, 3.0, 3.5]
   - Test min_feature_counts: n ∈ [5, 10, 15]
   - Evaluate removal rates and genus impact
   - **Decision:** Balance between removal rate (~1-3%) and genus preservation

2. **Mahalanobis α Level Justification**
   - Test α ∈ [0.0001, 0.001, 0.01]
   - Chi-squared critical values for multivariate outliers
   - Literature reference: common practice α = 0.001
   - Validate per-genus sample sizes (sufficient for stable covariance)

3. **IQR Multiplier Evaluation**
   - Test multipliers: k ∈ [1.5, 2.0, 3.0]
   - Tukey's standard: k = 1.5
   - Evaluate CHM outlier rates per genus × city
   - Literature reference: Tukey (1977)

4. **Consensus-Based Decision Rule Validation (UPDATED)**
   - Test consensus rules:
     - **3-of-3 methods** → REMOVE (outlier_critical)
     - **2-of-3 methods** → FLAG as outlier_high
     - **1-of-3 methods** → FLAG as outlier_low
   - Compare removal rates vs. alternative rules (e.g., Mahal + any other)
   - Validate that 3-of-3 consensus is conservative (low false positives)
   - Analyze method overlap to justify consensus approach

5. **Method Overlap Analysis (NEW)**
   - Compute overlap matrix: how many trees flagged by each combination
   - Visualize with Venn diagram
   - Verify that true outliers are captured by multiple methods
   - Rationale for consensus: single-method flags may be genus-specific

6. **Genus-Specific Analysis**
   - Check if certain genera have higher outlier rates
   - Validate that outliers are not genus-specific characteristics
   - Ensure no genus is disproportionately affected by consensus rule

**Outputs:**

```json
{
  "zscore": {
    "threshold": 3.0,
    "min_feature_count": 10,
    "justification": "Standard 3-sigma rule, 10 features = ~5% of total"
  },
  "mahalanobis": {
    "alpha": 0.001,
    "justification": "Common practice for multivariate outliers (Rousseeuw & Van Zomeren, 1990)"
  },
  "iqr": {
    "multiplier": 1.5,
    "justification": "Tukey (1977) standard for box plots"
  },
  "consensus_rule": "3_of_3_remove",
  "flag_columns": ["outlier_critical", "outlier_high", "outlier_low"],
  "method_overlap": {
    "zscore_only": 245,
    "mahal_only": 89,
    "iqr_only": 312,
    "zscore_and_mahal": 156,
    "zscore_and_iqr": 98,
    "mahal_and_iqr": 72,
    "all_three": 67
  },
  "expected_removal_rate": 0.008,
  "expected_flag_rates": {
    "outlier_high": 0.024,
    "outlier_low": 0.052
  },
  "validation": {
    "genus_impact": "uniform across genera",
    "false_positive_estimate": "< 0.3%",
    "sensitivity_analysis": "removing outlier_high adds ~2.4% removal"
  }
}
```

**Visualizations (REQUIRED):**

1. **Outlier Detection Rates**
   - Bar chart: Rates per method (Z-score, Mahal, IQR)
   - Per city and overall
   - File: `outlier_rates_per_method.png`

2. **Venn Diagram (Method Overlap)**
   - 3-circle Venn: Z-score, Mahalanobis, IQR
   - Shows consensus levels (1/3, 2/3, 3/3)
   - Annotated with counts
   - File: `outlier_venn_diagram.png`

3. **Z-Score Sensitivity**
   - Line plot: Removal rate vs threshold
   - Multiple lines for min_feature_count
   - File: `zscore_sensitivity.png`

4. **Mahalanobis Distribution**
   - Histogram with chi² critical value line
   - Faceted per genus
   - File: `mahalanobis_distribution.png`

5. **IQR Box Plots**
   - CHM per genus × city
   - Tukey fences visualized
   - File: `iqr_boxplots_per_genus.png`

6. **Consensus Decision Flow (UPDATED)**
   - Sankey/flowchart diagram
   - Shows: All trees → method flags → 3/3, 2/3, 1/3, 0/3 categories
   - Clearly shows REMOVE vs FLAG decisions
   - File: `consensus_decision_flow.png`

7. **Genus-wise Outlier Distribution (NEW)**
   - Stacked bar chart: outlier counts per genus by consensus level
   - Validates uniform impact across genera
   - File: `outlier_by_genus.png`

**Legacy Reference:** `legacy/notebooks/02_feature_engineering/03e_outlier_detection.md` and `legacy/documentation/02_Feature_Engineering/07_Outlier_Detection_Final_Filtering_Methodik.md`

---

### exp_05_spatial_autocorrelation.ipynb

**Purpose:** Determine optimal spatial block size to avoid train/val correlation

**Inputs:**

- `data/phase_2_features/trees_clean_{city}.gpkg` (from PRD 002b)

**Analysis Tasks:**

1. **Moran's I Calculation**
   - Compute Moran's I for representative features at multiple distance lags
   - Distance lags: [100m, 200m, 300m, 400m, 500m, 600m, 800m, 1000m]
   - Features to test: NDVI_mean, CHM_1m, spectral bands

2. **Autocorrelation Decay Analysis**
   - Identify distance threshold where Moran's I becomes negligible (< 0.05)
   - Visualize decay curves per feature
   - Compare across cities for consistency

3. **Block Size Determination**
   - Block size should exceed autocorrelation range
   - Test 500m (current default) against decay threshold
   - Validate that 500m blocks minimize train/val correlation

4. **Cross-City Validation**
   - Ensure block size works for both Berlin and Leipzig
   - Urban structure differences (dense vs sparse)

**Outputs:**

```json
{
  "block_size_m": 500,
  "autocorrelation_decay_distance": 350,
  "justification": "500m exceeds autocorrelation range (~350m) across features and cities",
  "morans_i_analysis": {
    "ndvi_decay_distance": 320,
    "chm_decay_distance": 380,
    "spectral_bands_decay_distance": 340
  },
  "validation": {
    "berlin_leipzig_consistency": "high",
    "sufficient_blocks": true
  }
}
```

**Visualizations (REQUIRED):**

1. **Moran's I Decay Curves**
   - Line plot: Moran's I vs distance
   - Multiple lines for features
   - Threshold line (I = 0.05)
   - File: `morans_i_decay_curves.png`

2. **Spatial Autocorrelation Heatmap**
   - X: Features, Y: Distance lags
   - Color: Moran's I value
   - File: `spatial_autocorrelation_heatmap.png`

3. **Block Overlay Maps** (per city)
   - Tree locations with 500×500m grid
   - Color-coded blocks
   - Files: `block_overlay_map_berlin.png`, `block_overlay_map_leipzig.png`

4. **Autocorrelation Comparison**
   - Faceted plot: Berlin vs Leipzig decay
   - Validates consistent block size
   - File: `autocorrelation_comparison.png`

**Legacy Reference:** `legacy/documentation/02_Feature_Engineering/08_Spatial_Splits_Stratification_Methodik.md` (spatial splits methodology)

**Note:** This is a NEW analysis not present in legacy (legacy used 500m without validation).

---

## Implementation Order

1. **exp_01_temporal_analysis** → Run after PRD 002a (needs raw features)
2. **exp_02_chm_assessment** → Run after PRD 002a (needs raw CHM)
3. **exp_03_correlation_analysis** → Run after PRD 002b (needs clean data)
4. **exp_04_outlier_thresholds** → Run after PRD 002b (needs clean data)
5. **exp_05_spatial_autocorrelation** → Run after PRD 002b (needs clean data)

---

## Testing

**Note:** Exploratory notebooks typically don't have formal unit tests, but should include:

- Data validation checks (schema, CRS, completeness)
- Statistical assumption checks (normality, homoscedasticity)
- Sanity checks on computed metrics

---

## Validation Checklist

- [ ] All JSON configs generated with required fields
- [ ] Thresholds statistically justified or literature-referenced
- [ ] Cross-city consistency validated
- [ ] **All required visualizations created with consistent style**
- [ ] Visualizations saved to `outputs/phase_2/figures/{notebook_name}/`
- [ ] Configuration files saved to `outputs/phase_2/metadata/`
- [ ] Plots use `setup_plotting()` for style consistency
- [ ] Plot file names follow naming convention
- [ ] Documentation updated with methodological decisions

---

## Visualization Best Practices

**CRITICAL:** All exploratory notebooks MUST follow the visualization strategy.

### Standard Setup (Every Notebook)

```python
from urban_tree_transfer.utils.plotting import setup_plotting, save_figure, PUBLICATION_STYLE
import matplotlib.pyplot as plt
from pathlib import Path

# Setup consistent plotting style
setup_plotting()

# Create figures directory
FIGURES_DIR = Path("/content/drive/MyDrive/urban-tree-transfer/outputs/phase_2/figures/exp_01_temporal")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

### Saving Plots

```python
# Create plot
fig, ax = plt.subplots(figsize=PUBLICATION_STYLE["figsize"])
# ... plotting code ...

# Save with consistent settings
save_figure(fig, FIGURES_DIR / "plot_name.png")
```

### Manual Sync Reminder

**Add to end of each notebook:**

```python
print("\n" + "="*70)
print("⚠️  MANUAL SYNC REQUIRED:")
print("="*70)
print(f"1. Plots saved to: {FIGURES_DIR}")
print(f"2. Copy to local repo: outputs/phase_2/figures/{notebook_name}/")
print(f"3. Review plots before committing")
print(f"4. Commit: git add outputs/phase_2/figures/{notebook_name}/")
print(f"   git commit -m 'feat: Add {notebook_name} visualizations'")
print("="*70)
```

**Full Details:** See [Visualization Strategy](../../docs/documentation/02_Feature_Engineering/01_Visualization_Strategy.md)

---

## Outputs

**Configuration Files:**

- `outputs/phase_2/metadata/temporal_selection.json`
- `outputs/phase_2/metadata/chm_assessment.json`
- `outputs/phase_2/metadata/correlation_removal.json`
- `outputs/phase_2/metadata/outlier_thresholds.json`
- `outputs/phase_2/metadata/spatial_autocorrelation.json`

**Visualizations:**

- `outputs/phase_2/figures/exp_01_temporal/*.png` (4 plots minimum)
- `outputs/phase_2/figures/exp_02_chm/*.png` (5 plots minimum)
- `outputs/phase_2/figures/exp_03_correlation/*.png` (4 plots minimum)
- `outputs/phase_2/figures/exp_04_outliers/*.png` (6 plots minimum)
- `outputs/phase_2/figures/exp_05_spatial/*.png` (4 plots minimum)

**Total:** ~23+ publication-quality plots for documentation and presentations

- `outputs/phase_2/figures/exp_04_*.png`
- `outputs/phase_2/figures/exp_05_*.png`

---

## Next Steps

After completing exploratory analyses:

1. Configuration files are consumed by runner notebooks (PRDs 002b, 002c)
2. Visualizations inform methodology documentation
3. Findings may be published in thesis/paper

---

**Status:** Ready for implementation
