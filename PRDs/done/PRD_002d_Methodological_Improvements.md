# PRD 002d: Methodological Improvements - Phase 2 Feature Engineering

**Status:** 📋 Planning
**Priority:** High
**Created:** 2026-01-30
**Owner:** Silas Pignotti

---

## Executive Summary

Dieses Dokument beschreibt 7 methodische Verbesserungen für Phase 2 Feature Engineering, die aus der kritischen Methodenbewertung hervorgegangen sind. Die Verbesserungen adressieren:

1. **Cross-City Consistency** in temporal feature selection
2. **Spatial Independence Validation** für Block-Splits
3. **Genus-Specific CHM Normalization** statt Stadt-Level
4. **Biological Context Analysis** für Outlier-Interpretation
5. **Geometric Clarity** im Proximity Filter
6. **Deutsche Gattungsnamen in Visualisierungen** statt lateinische
7. **Nadel-/Laubbaum-Spalte** als Metadatum im Datensatz

**Scope:** Alle Änderungen betreffen explorative Notebooks (exp_01, exp_02, exp_05, exp_04, exp_06), Module (quality.py, splits.py, outliers.py, proximity.py) und Visualisierungen.

**Out of Scope:** Feature Importance Analysis (wird in Phase 3 durchgeführt).

---

## Improvement 1: Cross-City JM Consistency Validation

### Priority

⭐ **Kritisch** - Verhindert city-specific temporal selection

### Problem Statement

**Aktuell:** Temporal month selection basiert auf JM-Distance, aber es wird nicht geprüft, ob die selektierten Monate in **beiden Städten** diskriminativ sind.

**Risiko:**

```
Beispiel:
- Mai zeigt hohe JM-Distance in Berlin (Blattaustrieb-Timing)
- Leipzig hat leicht andere Phänologie (1-2 Wochen früher/später)
→ Mai-Features könnten city-specific sein, nicht genus-specific
→ Schadet Transfer-Performance (Hauptziel des Projekts)
```

### Methodological Justification

**Forschungsfrage:** Wie gut transferieren Modelle von Berlin → Leipzig?

**Implikation:** Features müssen **transfer-robust** sein, nicht nur **separabel**.

**Literatur:**

- Pan & Yang (2010, _IEEE TKDE_): "Domain Adaptation requires domain-invariant features"
- Tuia et al. (2016, _ISPRS_): "Temporal consistency critical for cross-site transfer in remote sensing"

### Solution Approach

**Erweiterte Validierung in exp_01:**

1. **Separate JM-Distance Berechnung:**
   - JM_Berlin(month) für alle Genus-Paare
   - JM_Leipzig(month) für alle Genus-Paare

2. **Cross-City Ranking Consistency:**

   ```python
   # Rank months by JM-Distance per city
   rank_berlin = rank(JM_Berlin)  # 1 (best) to 12 (worst)
   rank_leipzig = rank(JM_Leipzig)

   # Spearman Rank Correlation
   ρ = spearman_correlation(rank_berlin, rank_leipzig)
   ```

3. **Consistency-Based Selection:**

   ```python
   IF ρ > 0.7:  # High consistency
       → Select top-N months by AVERAGE JM (current approach OK)
   ELSE:
       → Select only months in TOP-8 in BOTH cities (intersection)
       → Document city-specific months separately
   ```

4. **Genus-Level Consistency Check:**

   ```python
   # For each genus pair (e.g., ACER vs. TILIA):
   # Check if high-JM months are consistent across cities

   consistency_rate = |months_high_JM_both| / |months_high_JM_either|

   # Expected: > 0.75 (most genera show similar phenology)
   ```

### Implementation

**Modified Notebook:** `exp_01_temporal_analysis.ipynb`

**New Sections:**

#### Section 3.5: Cross-City JM Consistency Analysis

```python
# Calculate JM per city separately (already done for plots)
jm_berlin_by_month = {1: 0.65, 2: 0.58, ..., 12: 0.62}
jm_leipzig_by_month = {1: 0.68, 2: 0.55, ..., 12: 0.60}

# Rank months
rank_berlin = scipy.stats.rankdata(list(jm_berlin_by_month.values()))
rank_leipzig = scipy.stats.rankdata(list(jm_leipzig_by_month.values()))

# Spearman correlation
rho, p_value = scipy.stats.spearmanr(rank_berlin, rank_leipzig)

# Visualization: Scatter plot (Rank Berlin vs. Rank Leipzig)
# Diagonal line = perfect consistency
```

#### Section 3.6: Consistency-Aware Month Selection

```python
# Identify top-8 months per city
top8_berlin = months_sorted_by_JM[:8]
top8_leipzig = months_sorted_by_JM[:8]

# Intersection (months good in BOTH cities)
consistent_months = set(top8_berlin) & set(top8_leipzig)

# City-specific months (good in only one city)
berlin_specific = set(top8_berlin) - consistent_months
leipzig_specific = set(top8_leipzig) - consistent_months

# Final selection
if rho > 0.7:
    selected_months = top8_by_average  # Current approach
else:
    selected_months = consistent_months  # Conservative
    # May result in 6-7 months instead of 8
```

### Expected Output

**Updated `temporal_selection.json`:**

```json
{
  "cross_city_validation": {
    "spearman_rho": 0.82,
    "p_value": 0.001,
    "interpretation": "high consistency - average JM selection valid",
    "rank_berlin": [1, 2, 3, ...],
    "rank_leipzig": [1, 3, 2, ...],
    "top8_berlin": [5, 6, 7, 8, 9, 4, 10, 3],
    "top8_leipzig": [6, 7, 8, 5, 9, 4, 10, 11],
    "consistent_months": [5, 6, 7, 8, 9, 4, 10],
    "city_specific": {
      "berlin_only": [3],
      "leipzig_only": [11]
    }
  },
  "selection_method": "average_jm_with_consistency_check",
  "selected_months": [4, 5, 6, 7, 8, 9, 10],  # May differ from current
  "consistency_validated": true
}
```

**New Visualization:** `jm_rank_consistency.png`

- Scatter plot: Rank Berlin (x) vs. Rank Leipzig (y)
- Diagonal line (perfect consistency)
- Annotated months (numbers at points)
- Spearman ρ in title

### Acceptance Criteria

- [ ] Spearman ρ calculated and documented
- [ ] ρ > 0.7 OR conservative intersection selection applied
- [ ] City-specific months identified and documented
- [ ] Consistency plot generated
- [ ] JSON includes all validation metrics
- [ ] Rationale for final selection documented

---

## Improvement 2: Spatial Independence Validation

### Priority

⭐ **Kritisch** - Beweist, dass Spatial Splits funktionieren

### Problem Statement

**Aktuell:** Block size wird via Moran's I decay distance bestimmt, aber es wird nicht validiert, dass die finalen Splits **tatsächlich spatial independence** erreichen.

**Gap:** Ohne Post-Split-Validierung kann **spatial leakage** nicht ausgeschlossen werden.

### Methodological Justification

**Roberts et al. (2017, _Ecology_):** "Spatial cross-validation requires verification that folds are spatially independent."

**Validierung notwendig:**

- Moran's I gibt Empfehlung für Block-Size
- Aber: Stratified sampling könnte benachbarte Blocks in verschiedene Splits legen
- Post-hoc Check: Sind Train/Val/Test räumlich unkorreliert?

### Solution Approach

**In exp_05 (nach Block-Size-Bestimmung):**

#### Step 1: Create Preliminary Splits

```python
# Use recommended block_size from Moran's I analysis
block_size = 500  # From autocorrelation decay

# Assign blocks (without tree assignment yet)
blocks_gdf = create_spatial_blocks(trees_gdf, block_size)

# Stratified split at BLOCK level
train_blocks, val_blocks, test_blocks = stratified_block_split(
    blocks_gdf, ratios=[0.7, 0.15, 0.15]
)
```

#### Step 2: Compute Within-Split Moran's I

```python
# For each split: Calculate Moran's I for representative features
features_to_test = ["NDVI_06", "CHM_1m", "B8_07"]

for split_name, split_blocks in [("train", train_blocks), ...]:
    trees_in_split = trees_gdf[trees_gdf.block_id.isin(split_blocks)]

    for feature in features_to_test:
        I_within, p_value = calculate_morans_i(
            trees_in_split, feature, distance_threshold=block_size
        )

        # Target: |I| < 0.1 (negligible autocorrelation)
```

#### Step 3: Compute Between-Split Moran's I

```python
# Identify spatially adjacent blocks across splits
adjacent_pairs = find_adjacent_blocks(train_blocks, val_blocks)

# For each adjacent pair: Calculate cross-split autocorrelation
for feature in features_to_test:
    I_between = calculate_cross_split_morans_i(
        train_trees_near_boundary,
        val_trees_near_boundary,
        feature
    )

    # Target: I_between ≈ 0 (no correlation across split boundary)
```

#### Step 4: Iterative Adjustment

```python
if any(|I_within| > 0.1) or any(I_between > 0.05):
    # Block size too small
    block_size *= 1.2
    repeat_steps_1_to_3()
else:
    # Spatial independence confirmed
    save_validated_block_size()
```

### Implementation

**Modified Notebook:** `exp_05_spatial_autocorrelation.ipynb`

**New Sections:**

#### Section 4: Post-Split Spatial Independence Validation

```python
from urban_tree_transfer.feature_engineering.splits import (
    create_spatial_blocks,
    create_stratified_splits_berlin
)

# Load trees
trees_gdf = gpd.read_file("data/phase_2_features/trees_clean_berlin.gpkg")

# Use recommended block size from Section 3
recommended_block_size = 500  # meters

# Create preliminary splits
trees_with_blocks = create_spatial_blocks(trees_gdf, recommended_block_size)
train, val, test = create_stratified_splits_berlin(
    trees_with_blocks,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)

# Validate spatial independence
validation_results = validate_spatial_independence(
    train, val, test,
    features=["NDVI_06", "CHM_1m", "B8_07"],
    distance_threshold=recommended_block_size
)
```

#### Section 5: Iterative Block Size Refinement (if needed)

```python
iteration = 1
while not validation_results["passed"]:
    print(f"Iteration {iteration}: Spatial independence not achieved")
    recommended_block_size *= 1.2
    # Re-run Section 4
    iteration += 1

    if iteration > 5:
        raise ValueError("Cannot achieve spatial independence - check data")
```

### Expected Output

**Updated `spatial_autocorrelation.json`:**

```json
{
  "recommended_block_size_m": 500,
  "validation": {
    "method": "post_split_morans_i",
    "features_tested": ["NDVI_06", "CHM_1m", "B8_07"],
    "within_split_autocorrelation": {
      "train": {
        "NDVI_06": {"I": 0.03, "p_value": 0.12},
        "CHM_1m": {"I": -0.01, "p_value": 0.85},
        "B8_07": {"I": 0.05, "p_value": 0.08}
      },
      "val": {...},
      "test": {...}
    },
    "between_split_autocorrelation": {
      "train_val_boundary": {
        "NDVI_06": {"I": 0.02, "p_value": 0.45},
        ...
      },
      "train_test_boundary": {...},
      "val_test_boundary": {...}
    },
    "spatial_independence_achieved": true,
    "max_within_I": 0.05,
    "max_between_I": 0.02,
    "iterations_needed": 1
  }
}
```

**New Visualization:** `split_spatial_independence.png`

- Map showing Train/Val/Test blocks color-coded
- Overlay: Moran's I values at block centroids (text annotations)
- Border highlighting between splits
- Inset: Histogram of I-values (should be centered near 0)

### Acceptance Criteria

- [ ] Within-split Moran's I: max(|I|) < 0.1 for all features and splits
- [ ] Between-split Moran's I: |I| < 0.05 for all boundaries
- [ ] P-values non-significant (p > 0.05)
- [ ] Validation results in JSON
- [ ] Spatial independence map generated
- [ ] If iterations needed: documented in JSON

---

## Improvement 3: Genus-Specific CHM Normalization

### Priority

⭐ **Kritisch** - Korrekte Feature Engineering ohne Genus-Leakage

### Problem Statement

**Aktuell:** CHM-Features sind stadt-normalisiert:

```python
CHM_1m_zscore = (CHM_1m - μ_city) / σ_city
CHM_1m_percentile = rank(CHM_1m, city) / n_city
```

**Problem:** Vermischt Genera mit unterschiedlichen natürlichen Höhenbereichen.

**Beispiel:**

```
Baum A: QUERCUS, 20m → Percentile 60 (median für QUERCUS)
Baum B: MALUS, 8m → Percentile 60 (hoch für MALUS)
→ Beide haben gleiche engineered features trotz völlig unterschiedlicher ökologischer Position
```

**Besser:** Genus-spezifische Normalisierung entfernt genus-mean-Effekt, behält relative Position **innerhalb Genus**.

### Methodological Justification

**Biologischer Kontext:**

- QUERCUS (Eichen): 10-40m typische Höhe
- MALUS (Äpfel): 3-12m typische Höhe
- TILIA (Linden): 15-35m typische Höhe

**Stadt-Normalisierung:**

- Mischt diese Bereiche
- Könnte Genus-Information implizit leaken (hoher Percentile → wahrscheinlich großer Baum → wahrscheinlich QUERCUS)

**Genus-Normalisierung:**

- Entfernt genus-spezifischen Höhen-Bias
- Features repräsentieren **relative Größe innerhalb Gattung**
- Relevant für: Alter, Vitalität, Standortqualität

### Solution Approach

**Überarbeitung statt Addition:** Die bestehenden 2 Features werden genus-spezifisch berechnet.

#### Neue Formel

```python
# Statt Stadt-Level:
CHM_1m_zscore = (CHM_1m - μ_genus,city) / σ_genus,city
CHM_1m_percentile = rank(CHM_1m, genus×city) / n_genus,city
```

**Wichtig:** Normalisierung bleibt **pro Stadt**, aber **innerhalb Genus**.

- QUERCUS in Berlin: Normalisiert gegen alle QUERCUS in Berlin
- QUERCUS in Leipzig: Normalisiert gegen alle QUERCUS in Leipzig
- Verhindert city-leakage, aber nutzt genus-spezifische Verteilungen

### Implementation

**Modified Module:** `feature_engineering/quality.py`

**Function:** `engineer_chm_features()`

#### Before (Stadt-Level)

```python
def engineer_chm_features(gdf: gpd.GeoDataFrame, city_column: str = "city") -> gpd.GeoDataFrame:
    result = gdf.copy()

    # Stadt-spezifische Normalisierung
    for city in result[city_column].unique():
        city_mask = result[city_column] == city
        city_chm = result.loc[city_mask, "CHM_1m"]

        # Z-score
        result.loc[city_mask, "CHM_1m_zscore"] = (
            (city_chm - city_chm.mean()) / city_chm.std()
        )

        # Percentile
        result.loc[city_mask, "CHM_1m_percentile"] = (
            city_chm.rank(pct=True) * 100
        )

    return result
```

#### After (Genus×Stadt-Level)

```python
def engineer_chm_features(
    gdf: gpd.GeoDataFrame,
    genus_column: str = "genus_latin",
    city_column: str = "city"
) -> gpd.GeoDataFrame:
    """
    Engineer CHM features with genus-specific normalization per city.

    Rationale:
    - Different genera have different natural height ranges (QUERCUS: 10-40m, MALUS: 3-12m)
    - Genus-specific normalization removes genus-mean effect
    - Features represent relative size WITHIN genus
    - City-level grouping prevents cross-city leakage

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Input with CHM_1m column
    genus_column : str
        Column containing genus labels
    city_column : str
        Column containing city labels

    Returns
    -------
    gpd.GeoDataFrame
        With added CHM_1m_zscore and CHM_1m_percentile columns
    """
    result = gdf.copy()

    # Genus×Stadt-spezifische Normalisierung
    for city in result[city_column].unique():
        for genus in result[genus_column].unique():
            mask = (result[city_column] == city) & (result[genus_column] == genus)

            if mask.sum() < 10:
                # Too few samples for stable statistics - use city-level fallback
                logger.warning(
                    f"Genus {genus} in {city} has <10 samples - using city-level normalization"
                )
                city_mask = result[city_column] == city
                genus_chm = result.loc[city_mask, "CHM_1m"]
            else:
                genus_chm = result.loc[mask, "CHM_1m"]

            # Z-score (genus-specific)
            result.loc[mask, "CHM_1m_zscore"] = (
                (result.loc[mask, "CHM_1m"] - genus_chm.mean()) / genus_chm.std()
            )

            # Percentile (genus-specific)
            result.loc[mask, "CHM_1m_percentile"] = (
                result.loc[mask, "CHM_1m"].rank(pct=True) * 100
            )

    return result
```

**Edge Case Handling:**

```python
# If genus has <10 samples in a city: fallback to city-level
# Prevents unstable statistics for rare genera
# Logged as warning for documentation
```

### Expected Output

**Feature Distribution Changes:**

**Before (Stadt-Level):**

```
QUERCUS in Berlin: CHM_percentile mean ≈ 75 (tall trees dominate high percentiles)
MALUS in Berlin: CHM_percentile mean ≈ 25 (short trees dominate low percentiles)
→ Percentile implicitly encodes genus information
```

**After (Genus-Level):**

```
QUERCUS in Berlin: CHM_percentile mean ≈ 50 (centered within genus)
MALUS in Berlin: CHM_percentile mean ≈ 50 (centered within genus)
→ Percentile encodes RELATIVE position, not absolute genus-height
```

**Validation Plot (New):**

```python
# Box plot: CHM_1m_percentile distribution per genus
# Before: Genera clearly separated
# After: All genera centered around 50 (by design)
```

### Acceptance Criteria

- [ ] `engineer_chm_features()` updated to genus×city grouping
- [ ] Edge case handling for rare genera (<10 samples)
- [ ] Unit tests updated
- [ ] Validation plot generated (percentile distribution per genus)
- [ ] Phase 2b methodology documentation updated
- [ ] No breaking changes in downstream notebooks

### Impact on Phase 3

**Ablation Study Opportunity:**

- Keep both approaches in feature_config.yaml
- Generate datasets with both normalization strategies
- Compare transfer performance (Berlin → Leipzig)
- **Hypothesis:** Genus-specific normalization transfers better (removes genus-bias)

---

## Improvement 4: Biological Context Analysis for Outliers

### Priority

🔵 **Nice-to-Have** - Improves interpretability, nicht kritisch

### Problem Statement

**Aktuell:** Outlier detection ist rein statistisch (Z-Score, Mahalanobis, IQR).

**Gap:** Keine Unterscheidung zwischen:

- **Data quality issues:** Sensor errors, processing artifacts
- **Biological extremes:** Very old trees, exceptional specimens

**Beispiel:**

```
Baum: 200-jährige QUERCUS in Park
CHM: 35m (sehr hoch)
Spektral: Alte Blattstruktur → extreme Werte
→ Alle 3 Methoden flaggen als Outlier (Severity: HIGH)
→ Aber: Biologisch valide, nur außergewöhnlich alt
```

### Solution Approach

**In exp_04: Exploriere Metadata-Kontext von Outliers**

#### Analysis 1: Plant Year Distribution

```python
# Compare plant_year distribution for outliers vs. non-outliers
outlier_high = trees[trees.outlier_severity == "high"]
outlier_none = trees[trees.outlier_severity == "none"]

print(f"High-severity: median plant_year = {outlier_high.plant_year.median()}")
print(f"Non-outlier: median plant_year = {outlier_none.plant_year.median()}")

# Hypothesis: High-severity correlates with old trees (low plant_year)
```

#### Analysis 2: Tree Type Distribution

```python
# Compare tree_type (anlagenbaeume vs. strassenbaeume)
contingency_table = pd.crosstab(
    trees.outlier_severity,
    trees.tree_type,
    normalize="index"  # Row percentages
)

# Hypothesis: Park trees (anlagenbaeume) more likely to be biological extremes
```

#### Analysis 3: Spatial Clustering

```python
# Are high-severity outliers spatially clustered? (parks, monuments)
outlier_high_gdf = trees[trees.outlier_severity == "high"]

# Ripley's K for clustering test
K_observed = ripleys_k(outlier_high_gdf, distances=[50, 100, 200])

# If K > expected under CSR: Clustered (suggests biological contexts like parks)
```

#### Analysis 4: Genus-Specific Patterns

```python
# Are certain genera more likely to be high-severity outliers?
genus_outlier_rates = trees.groupby("genus_latin").apply(
    lambda g: (g.outlier_severity == "high").mean()
)

# Expected: QUERCUS, PLATANUS higher rates (large trees, high variability)
```

### Implementation

**Modified Notebook:** `exp_04_outlier_thresholds.ipynb`

**New Section 5: Biological Context Analysis**

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Analysis 1: Age distribution
fig, ax = plt.subplots(1, 3, figsize=(15, 5))

for i, severity in enumerate(["high", "medium", "low"]):
    outlier_subset = trees[trees.outlier_severity == severity]
    none_subset = trees[trees.outlier_severity == "none"]

    ax[i].hist(outlier_subset.plant_year.dropna(), bins=20, alpha=0.5, label=severity)
    ax[i].hist(none_subset.plant_year.dropna(), bins=20, alpha=0.5, label="none")
    ax[i].set_title(f"Severity: {severity}")
    ax[i].legend()

plt.suptitle("Plant Year Distribution by Outlier Severity")
plt.savefig("outputs/phase_2/figures/exp_04/outlier_age_distribution.png", dpi=300)

# Statistical test: Are high-severity outliers significantly older?
from scipy.stats import mannwhitneyu
u_stat, p_value = mannwhitneyu(
    outlier_high.plant_year.dropna(),
    outlier_none.plant_year.dropna(),
    alternative="less"  # High-severity has LOWER plant_year (older)
)
print(f"Mann-Whitney U test: U={u_stat}, p={p_value}")
```

### Expected Output

**Updated `outlier_thresholds.json`:**

```json
{
  "biological_context_analysis": {
    "age_correlation": {
      "high_severity_median_plant_year": 1975,
      "non_outlier_median_plant_year": 2005,
      "mann_whitney_u": 12345,
      "p_value": 0.001,
      "interpretation": "High-severity outliers significantly older (likely biological extremes)"
    },
    "tree_type_association": {
      "high_severity_anlagenbaeume_pct": 0.68,
      "high_severity_strassenbaeume_pct": 0.32,
      "non_outlier_anlagenbaeume_pct": 0.45,
      "interpretation": "High-severity enriched in park trees (anlagenbaeume)"
    },
    "spatial_clustering": {
      "ripleys_k_50m": 1.8,
      "ripleys_k_100m": 2.1,
      "expected_under_csr": 1.0,
      "interpretation": "High-severity outliers spatially clustered (parks, monuments)"
    },
    "recommendation": "High-severity outliers likely include biological extremes (old park trees). Consider separate flag: outlier_likely_biological for Phase 3 ablation studies."
  }
}
```

**New Visualization:** `outlier_biological_context.png` (4-panel figure)

- Panel A: Age distribution (histogram)
- Panel B: Tree type (stacked bar)
- Panel C: Spatial clustering (map with high-severity highlighted)
- Panel D: Genus-specific rates (bar chart)

### Acceptance Criteria

- [ ] 4 biological context analyses completed
- [ ] Statistical tests for age correlation (Mann-Whitney U)
- [ ] Results documented in JSON
- [ ] 4-panel visualization generated
- [ ] Interpretation included in JSON
- [ ] Recommendation for Phase 3 documented

---

## Improvement 5: Geometric Clarity in Proximity Filter

### Priority

🔵 **Nice-to-Have** - Verbessert Dokumentation und Reproduzierbarkeit

### Problem Statement

**Aktuell:** 20m threshold beschrieben als "2-pixel buffer", aber geometrische Details unklar.

**Ambiguität:**

```
Sentinel-2 Pixel: 10m × 10m
Pixel-Diagonal: √(10² + 10²) ≈ 14.1m

Frage: Was bedeutet "20m Distanz"?
- Edge-to-Edge (Pixel-Zentren)?
- Bounding-Box Overlap?
- Diagonal consideration?
```

### Solution Approach

**In exp_06: Explizite geometrische Definition und Visualisierung**

#### Step 1: Clarify Distance Metric

```python
# Aktuelle Implementierung nutzt:
distance = euclidean_distance(tree_A.geometry, tree_B.geometry)
# → Point-to-Point (Baum-Zentren)

# Klarstellung in Dokumentation:
# "20m threshold = Point-to-Point distance between tree centroids"
# "Corresponds to ~2 Sentinel-2 pixels (10m × 2 = 20m)"
```

#### Step 2: Pixel Footprint Visualization

```python
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(10, 10))

# Example: Two trees 15m apart
tree_A = Point(0, 0)  # QUERCUS
tree_B = Point(15, 0)  # TILIA

# Plot tree points
ax.plot(0, 0, 'ro', markersize=10, label='QUERCUS')
ax.plot(15, 0, 'go', markersize=10, label='TILIA')

# Plot Sentinel-2 pixel footprints (10m × 10m centered on trees)
pixel_A = patches.Rectangle((-5, -5), 10, 10, linewidth=2, edgecolor='r', facecolor='red', alpha=0.2)
pixel_B = patches.Rectangle((10, -5), 10, 10, linewidth=2, edgecolor='g', facecolor='green', alpha=0.2)
ax.add_patch(pixel_A)
ax.add_patch(pixel_B)

# Annotate distance
ax.plot([0, 15], [0, 0], 'k--', linewidth=1)
ax.text(7.5, -1, '15m', ha='center', fontsize=12)

# Highlight overlap zone
overlap_start = 10 - 5  # Right edge of pixel_A
overlap_end = 15 - 5    # Left edge of pixel_B
if overlap_end > overlap_start:
    overlap = patches.Rectangle(
        (overlap_start, -5), overlap_end - overlap_start, 10,
        linewidth=0, facecolor='yellow', alpha=0.5
    )
    ax.add_patch(overlap)
    ax.text(7.5, 6, 'Pixel Overlap\n(Spectral Contamination)', ha='center', fontsize=10, color='orange')

ax.set_xlim(-10, 25)
ax.set_ylim(-10, 10)
ax.set_aspect('equal')
ax.legend()
ax.set_title('Sentinel-2 Pixel Contamination: 15m Distance')
ax.set_xlabel('Distance (m)')
ax.set_ylabel('Distance (m)')
ax.grid(True, alpha=0.3)

plt.savefig("outputs/phase_2/figures/exp_06/pixel_contamination_schematic.png", dpi=300)
```

#### Step 3: Threshold Sensitivity Analysis

```python
# Test multiple thresholds
thresholds = [10, 15, 20, 25, 30]  # meters

results = []
for threshold in thresholds:
    filtered_gdf, stats = apply_proximity_filter(trees_gdf, threshold)
    results.append({
        "threshold": threshold,
        "retention_rate": stats["retention_rate"],
        "removed_count": stats["removed_count"]
    })

# Plot: Retention rate vs. threshold
fig, ax = plt.subplots()
ax.plot([r["threshold"] for r in results], [r["retention_rate"] for r in results], 'o-')
ax.axvline(20, color='r', linestyle='--', label='Selected: 20m')
ax.set_xlabel("Proximity Threshold (m)")
ax.set_ylabel("Retention Rate")
ax.set_title("Threshold Sensitivity Analysis")
ax.legend()
```

### Implementation

**Modified Notebook:** `exp_06_mixed_genus_proximity.ipynb`

**New Section 2: Geometric Definition and Pixel Footprint Analysis**

```python
# Section 2.1: Distance Metric Clarification
print("""
Distance Metric: Euclidean Point-to-Point
- Measured between tree geometry centroids
- Sentinel-2 pixel footprint: 10m × 10m (bands B2, B3, B4, B8)
- 20m threshold ≈ 2 pixels (edge-to-edge at 10m, overlap starts at <20m)
""")

# Section 2.2: Pixel Footprint Scenarios
scenarios = [
    {"distance": 10, "overlap": "Full (1 pixel)", "contamination": "High"},
    {"distance": 15, "overlap": "Partial (~50%)", "contamination": "Medium"},
    {"distance": 20, "overlap": "Edge contact", "contamination": "Low"},
    {"distance": 25, "overlap": "None", "contamination": "None"}
]

for scenario in scenarios:
    plot_pixel_footprint_scenario(scenario["distance"])
    # Generates individual plots for each distance
```

### Expected Output

**Updated `proximity_filter.json`:**

```json
{
  "geometric_definition": {
    "distance_metric": "euclidean_point_to_point",
    "sentinel_pixel_size": "10m × 10m",
    "threshold_interpretation": "20m = approx 2-pixel separation (edge-to-edge contact)",
    "contamination_zones": {
      "high": "< 10m (full pixel overlap)",
      "medium": "10-15m (partial overlap)",
      "low": "15-20m (edge contact)",
      "none": "> 20m (no overlap)"
    }
  },
  "threshold_sensitivity": {
    "10m": { "retention": 0.62, "removed": 17100 },
    "15m": { "retention": 0.75, "removed": 11250 },
    "20m": { "retention": 0.85, "removed": 6750 },
    "25m": { "retention": 0.91, "removed": 4050 },
    "30m": { "retention": 0.95, "removed": 2250 }
  },
  "recommended_threshold_m": 20,
  "rationale": "Balances spectral purity (contamination zone eliminated) with sample retention (85%). 15m too aggressive (75% retention), 25m too liberal (edge contact remains)."
}
```

**New Visualizations:**

1. `pixel_contamination_scenarios.png` - 4-panel figure (10m, 15m, 20m, 25m distances)
2. `threshold_sensitivity_curve.png` - Retention rate vs. threshold
3. `spatial_contamination_map.png` - Real trees with proximity zones color-coded

### Acceptance Criteria

- [ ] Distance metric explicitly defined
- [ ] Pixel footprint scenarios visualized (4 distances)
- [ ] Threshold sensitivity curve generated
- [ ] Contamination zones documented in JSON
- [ ] Rationale for 20m threshold clearly stated
- [ ] Real-world contamination map created

---

## Improvement 6: Deutsche Gattungsnamen in Visualisierungen

### Priority

⭐ **Niedrig** - Kosmetisch, aber wichtig für Präsentationskonsistenz

### Problem Statement

**Aktuell:** Alle explorativen Notebooks (exp_01–exp_06) verwenden `genus_latin` als Achsenlabel
in Plots (z. B. „TILIA", „ACER"). Phase 3 schreibt dagegen vor, dass alle Visualisierungen
**deutsche Gattungsnamen** verwenden sollen (`genus_german`: „Linde", „Ahorn").

**Risiko:** Inkonsistenz zwischen Phase 2-Visualisierungen und Phase 3-Ergebnissen in der fertigen
Arbeit. Wenn Phase 2-Plots in die Präsentation übernommen werden, müssen die Labels übereinstimmen.

### Methodological Justification

- Die Daten enthalten bereits die Spalte `genus_german` — es ist kein Mapping nötig
- Konvention in Phase 3 PRD (Section 6.0): „All genus labels in visualizations MUST use `genus_german`"
- Für eine deutschsprachige Abschlussarbeit/Präsentation sind deutsche Namen lesbarer

### Solution Approach

In allen explorativen Notebooks, die Gattungen visualisieren:

1. **Label-Mapping erstellen** (pro Notebook, einmalig):

   ```python
   # genus_german ist bereits im Datensatz vorhanden
   label_map = df.drop_duplicates("genus_latin").set_index("genus_latin")["genus_german"].to_dict()
   ```

2. **Achsenlabels ersetzen:** Überall wo `genus_latin` als Axis-Label oder Facet-Variable
   verwendet wird, stattdessen `genus_german` nutzen oder Tick-Labels über `label_map` umschreiben.

3. **Betroffene Notebooks:**
   - `exp_01_temporal_analysis.ipynb` — JM-Distance Plots pro Gattung
   - `exp_04_outlier_thresholds.ipynb` — Outlier-Verteilung pro Gattung
   - `exp_06_mixed_genus_proximity.ipynb` — Proximity-Analyse pro Gattung

### Acceptance Criteria

- [ ] Alle Genus-Labels in Phase 2-Plots zeigen deutsche Namen
- [ ] Lateinische Namen optional in Klammern wo Platz vorhanden: „Linde (Tilia)"
- [ ] Konsistent mit Phase 3-Konvention (Section 6.0 im PRD)

---

## Improvement 7: Nadel-/Laubbaum-Spalte im Datensatz

### Priority

⭐ **Mittel** - Ermöglicht sauberere Analysen in Phase 3, einfach umzusetzen

### Problem Statement

**Aktuell:** Die Parquet-Datensätze enthalten keine Spalte, die angibt, ob ein Baum ein
Nadel- oder Laubbaum ist. In Phase 3 wird diese Gruppierung für mehrere Analysen benötigt
(Conifer vs. Deciduous F1-Vergleich, Transfer-Performance pro Baumgruppe). Das Mapping
ist derzeit nur als Config-Lookup im Experiment-Config definiert:

```yaml
genus_groups:
  conifer: [PINUS, PICEA]
  deciduous: [TILIA, ACER, QUERCUS, ...]
```

**Risiko:** Zur Laufzeit muss jedes Mal ein dict-Lookup auf `genus_latin` gemacht werden.
Bei einer Erweiterung der Gattungsliste oder bei unerwarteten Gattungen im Datensatz
könnte das Mapping unvollständig sein.

### Methodological Justification

- Eine Spalte `is_conifer` (bool) oder `tree_group` (categorical: `"Nadelbaum"` / `"Laubbaum"`)
  im Datensatz ist robuster und transparenter
- Kann direkt beim Proximity-Filter-Schritt oder beim Parquet-Export berechnet werden,
  da dort bereits über alle Gattungen iteriert wird
- Besser für Reproduzierbarkeit: das Mapping ist dann im Datensatz fixiert, nicht nur im Config

### Solution Approach

**Stelle:** Im Parquet-Export (Phase 2c, `selection.py` oder `02c_final_preparation.ipynb`),
nach dem Join mit den harmonisierten Baumdaten:

```python
CONIFER_GENERA = {"PINUS", "PICEA"}
df["is_conifer"] = df["genus_latin"].isin(CONIFER_GENERA)
```

Alternativ als kategoriale Spalte:

```python
df["tree_group"] = df["genus_latin"].apply(
    lambda g: "Nadelbaum" if g in CONIFER_GENERA else "Laubbaum"
)
```

### Acceptance Criteria

- [ ] Parquet-Datensätze enthalten `is_conifer`-Spalte (oder `tree_group`)
- [ ] Werte korrekt für alle 10 Gattungen (2 Nadel, 8 Laub)
- [ ] Phase 3 PRD Config kann vereinfacht werden (direkter Spalten-Zugriff statt Lookup)
- [ ] Unit-Test prüft korrekte Zuordnung

---

## Implementation Plan

### Phase 1: Critical Improvements (Week 1)

**Target:** Methodik-kritische Gaps schließen

- [ ] **Day 1-2:** Improvement 1 (Cross-City JM Consistency)
  - Modify `exp_01_temporal_analysis.ipynb`
  - Run analysis on current data
  - Update `temporal_selection.json`
  - Generate consistency plot

- [ ] **Day 3-4:** Improvement 2 (Spatial Independence Validation)
  - Modify `exp_05_spatial_autocorrelation.ipynb`
  - Implement post-split Moran's I checks
  - Update `spatial_autocorrelation.json`
  - Generate validation map

- [ ] **Day 5:** Improvement 3 (Genus-Specific CHM)
  - Modify `quality.py` - `engineer_chm_features()`
  - Update unit tests
  - Re-run Phase 2b with new features
  - Update documentation

### Phase 2: Documentation Improvements (Week 2)

**Target:** Interpretierbarkeit und Reproduzierbarkeit

- [ ] **Day 1-2:** Improvement 4 (Biological Context)
  - Add Section 5 to `exp_04_outlier_thresholds.ipynb`
  - Run biological context analyses
  - Update `outlier_thresholds.json`
  - Generate 4-panel context plot

- [ ] **Day 2-3:** Improvement 5 (Geometric Clarity)
  - Add Section 2 to `exp_06_mixed_genus_proximity.ipynb`
  - Create pixel footprint visualizations
  - Update `proximity_filter.json`
  - Generate contamination map

### Phase 3: Integration & Validation (Week 2, End)

**Target:** Alles funktioniert, dokumentiert, reproduzierbar

- [ ] **Day 4:** Re-run Complete Pipeline
  - Phase 2a (unchanged)
  - Phase 2b (new CHM features)
  - All exploratory notebooks (updated)
  - Phase 2c (using updated JSONs)

- [ ] **Day 5:** Documentation Update
  - Update all methodology docs
  - Update CHANGELOG.md
  - Commit all changes
  - Tag as `v0.2-phase2-methodological-improvements`

---

## Success Metrics

### Quantitative

- [ ] Cross-city JM rank correlation ρ > 0.7
- [ ] Spatial independence: max(|I_within|) < 0.1
- [ ] Spatial independence: max(I_between) < 0.05
- [ ] All unit tests pass
- [ ] No breaking changes in downstream code

### Qualitative

- [ ] Methodology documentation complete
- [ ] All decisions statistically justified
- [ ] Visualizations publication-ready
- [ ] Code review passed
- [ ] Reproducible from scratch

---

## Dependencies

### External

- None (all within existing Phase 2 scope)

### Internal

- Phase 2a must be complete (unchanged)
- Phase 2b partially re-run (new CHM features)
- All exploratory notebooks re-executed
- Phase 2c uses updated JSONs

---

## Risks & Mitigation

| Risk                                       | Impact                              | Likelihood | Mitigation                                        |
| ------------------------------------------ | ----------------------------------- | ---------- | ------------------------------------------------- |
| JM consistency low (ρ < 0.7)               | Month selection changes             | Medium     | Use conservative intersection approach            |
| Spatial independence not achievable        | Block size too large → fewer blocks | Low        | Iterative refinement with 20% increments          |
| Genus-CHM requires re-processing all data  | Time-intensive                      | High       | Accept - correctness > speed                      |
| Biological context shows no clear patterns | Less interpretable                  | Medium     | Document null result - still scientifically valid |

---

## Timeline

**Week 1:** Critical improvements (1-3)
**Week 2:** Documentation improvements (4-5) + Integration
**Total:** 10 working days

**Milestone:** Phase 2 methodologically robust and publication-ready

---

## Acceptance

**Definition of Done:**

- [ ] All 5 improvements implemented
- [ ] All acceptance criteria met
- [ ] Pipeline runs end-to-end
- [ ] Documentation updated
- [ ] Code committed and tagged
- [ ] Ready for Phase 3

---

**Erstellt:** 2026-01-30
**Version:** 1.0
**Status:** Bereit für Implementierung
