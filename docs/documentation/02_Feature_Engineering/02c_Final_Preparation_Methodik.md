# Runner 02c: Final Preparation & Dataset Creation

**Notebook:** `notebooks/runners/02c_final_preparation.ipynb`
**Module:** `selection.py`, `outliers.py`, `splits.py`, `proximity.py`
**Zweck:** Erstellung ML-ready Train/Val/Test Splits mit Dual-Dataset-Strategie
**Status:** ✅ Implementiert

---

## Ziel

Transformation der QC-validierten Feature-Datensätze in finale ML-ready Splits für Phase 3 Experimente. Implementiert werden:

1. **Feature Redundancy Removal** (Korrelationsbasiert)
2. **Outlier Detection & Flagging** (Consensus-Methode, keine Removal)
3. **Spatial Block Assignment** (Data-driven Block-Größe)
4. **Stratified Spatial Splits** (Train/Val/Test mit Genus-Stratifikation)
5. **Proximity Filtering** (Mixed-Genus Pixel-Contamination Reduktion)
6. **Dual Dataset Strategy** (Baseline + Filtered Varianten für Ablation Studies)

**Kernoutput:** 10 GeoPackages (5 Baseline + 5 Filtered) für Phase 3

---

## Input

**Daten:** `data/phase_2_features/trees_clean_{city}.gpkg` (Phase 2b Output)

**Schema-Erwartungen:**

- Metadaten: tree_id, city, genus_latin, species_latin, genus_german, species_german, plant_year, height_m, tree_type
- Features: Sentinel-2 Bänder/Indizes (nach temporal selection) + CHM Features
- **Garantie:** 0 NaN-Werte, CRS: EPSG:25833 (projected)

**JSON Konfigurationen:** (aus Exploratory Notebooks)

- `correlation_removal.json` (exp_03): Liste redundanter Features
- `outlier_thresholds.json` (exp_04): Z-score, Mahalanobis, IQR Thresholds
- `spatial_autocorrelation.json` (exp_05): Empirische Block-Größe
- `proximity_filter.json` (exp_06): Mixed-Genus Distance Threshold

---

## Modul-Architektur

### 1. Feature Selection (`selection.py`)

**Funktion:** `remove_redundant_features(gdf, features_to_remove)`

**Zweck:** Entfernung hochkorrelierter Features zur Reduktion von Multikollinearität

**Methodik:**

- Input: Liste redundanter Feature-Namen aus `correlation_removal.json`
- Strategie: Aus jedem hochkorrelierten Paar (|r| > 0.95) wird Feature mit geringerer Varianz entfernt
- Output: GeoDataFrame ohne redundante Spalten

**Validierung:**

- Warnung bei fehlenden Spalten (falls bereits entfernt)
- Schutz der Geometry-Spalte
- CRS-Validierung

---

### 2. Outlier Detection (`outliers.py`)

**WICHTIG:** Outliers werden **NUR GETAGGT**, nicht entfernt! (Ablation Study in Phase 3)

#### Funktion 1: `detect_zscore_outliers(gdf, feature_columns, z_threshold, min_feature_count)`

**Methodik:** Univariate Z-Score Methode

```
Z = (x - μ) / σ

Outlier wenn: |Z| > threshold (default: 3.0)
```

**Per-Feature Detection:**

- Für jedes Feature separat Z-Score berechnen
- Baum wird getaggt wenn ≥ min_feature_count Features outlier sind

**Return:** Boolean Series (True = Outlier)

#### Funktion 2: `detect_mahalanobis_outliers(gdf, feature_columns, alpha)`

**Methodik:** Multivariate Distanz-Methode (per Genus)

```
D² = (x - μ)ᵀ Σ⁻¹ (x - μ)

D² ~ χ²(p) unter H₀ (p = Anzahl Features)
```

**Signifikanz-Test:**

- Chi-Square Test mit α = 0.001 (default)
- Outlier wenn p-value < α

**Per-Genus Analyse:**

- Separate Kovarianzmatrix pro Genus
- Berücksichtigt Genus-spezifische Feature-Korrelationen

**Return:** Boolean Series (True = Outlier)

#### Funktion 3: `detect_iqr_outliers(gdf, height_column, multiplier)`

**Methodik:** Tukey's Fences (Höhen-basiert)

```
Q1 = 25th Percentile
Q3 = 75th Percentile
IQR = Q3 - Q1

Lower Fence = Q1 - k × IQR
Upper Fence = Q3 + k × IQR

Outlier wenn: x < Lower oder x > Upper
```

**Gruppierung:** Per (Genus, City)

- Berücksichtigt natürliche Höhenunterschiede zwischen Genera
- k = 1.5 (default, moderates Outlier-Kriterium)

**Return:** Boolean Series (True = Outlier)

#### Funktion 4: `apply_consensus_outlier_filter(gdf, zscore_flags, mahal_flags, iqr_flags)`

**Consensus-Strategie:** Severity-basierte Kategorisierung

**Severity Levels:**

- **high:** 3/3 Methoden taggen als Outlier
- **medium:** 2/3 Methoden taggen als Outlier
- **low:** 1/3 Methode taggt als Outlier
- **none:** 0/3 Methoden taggen als Outlier

**Output Metadata Columns:**

```python
result["outlier_zscore"] = bool          # Z-score Methode Flag
result["outlier_mahalanobis"] = bool     # Mahalanobis Methode Flag
result["outlier_iqr"] = bool             # IQR Methode Flag
result["outlier_severity"] = str         # "high", "medium", "low", "none"
result["outlier_method_count"] = int     # 0, 1, 2, oder 3
```

**Return:** (GeoDataFrame mit Metadata, Statistics Dict)

**Statistics:**

```python
{
    "trees_removed": 0,           # IMMER 0! Keine Removal
    "high_count": int,            # Anzahl High-Severity
    "medium_count": int,          # Anzahl Medium-Severity
    "low_count": int,             # Anzahl Low-Severity
    "none_count": int             # Anzahl No-Outlier
}
```

**Ablation Study Strategie (Phase 3):**

- Baseline: Alle Bäume (inkl. Outliers)
- Strategy 1: Nur "high" entfernen
- Strategy 2: "high" + "medium" entfernen
- Strategy 3: Alle Outliers entfernen

---

### 3. Spatial Splits (`splits.py`)

#### Funktion 1: `create_spatial_blocks(trees_gdf, block_size_m)`

**Zweck:** Regular Grid Creation für Spatial Disjointness

**Methodik:**

- Per-City Grid: Vermeidet Stadt-übergreifende Blocks
- Block-ID Format: `{city}_{grid_x}_{grid_y}` (z.B. "berlin_42_73")
- Spatial Join: `within` predicate (trees → blocks)
- Fallback: `sjoin_nearest` für Bäume auf Block-Grenzen

**Block-Größe:** Data-driven aus `spatial_autocorrelation.json`

- Empirisch bestimmt via Moran's I Analyse (exp_05)
- Größer als Autokorrelations-Reichweite
- Typisch: 400-600m

**Output:** GeoDataFrame mit zusätzlicher `block_id` Spalte

#### Funktion 2: `create_stratified_splits_berlin(gdf, train_ratio, val_ratio, test_ratio)`

**Split-Strategie:** StratifiedGroupKFold

**Ratio:** 70/15/15 (Train/Val/Test)

**Constraints:**

1. **Spatial Disjointness:** Blocks als Groups (kein Block in mehreren Splits)
2. **Genus Stratification:** Erhalt Genus-Proportionen über Splits
3. **Reproducibility:** Fixed Random Seed (42)

**Algorithmus:**

```python
# Step 1: 10-Fold Cross-Validation (oder weniger wenn < 10 Blocks)
StratifiedGroupKFold(n_splits=min(10, n_blocks))
    stratify: genus_latin
    groups: block_id

# Step 2: Erste ~3 Folds → Holdout (Val + Test, ~30%)
holdout_folds = round((1 - train_ratio) * n_splits)

# Step 3: 2-Fold Split von Holdout → Val (50%) + Test (50%)
StratifiedGroupKFold(n_splits=2)
    stratify: genus_latin
    groups: block_id (nur Holdout-Blocks)
```

**Return:** (train_gdf, val_gdf, test_gdf)

#### Funktion 3: `create_stratified_splits_leipzig(gdf, finetune_ratio, test_ratio)`

**Split-Strategie:** Transfer-Learning Setup

**Ratio:** 80/20 (Finetune-Pool/Test)

**Unterschied zu Berlin:**

- Nur 2 Splits (kein Val-Split)
- Finetune-Pool: Für Few-Shot Sampling in Phase 3
- Test-Split: Independent Leipzig Evaluation

**Return:** (finetune_gdf, test_gdf)

#### Funktion 4: `validate_split_stratification(*split_gdfs, split_names)`

**Validierung:** Spatial Disjointness + Stratification Quality

**Spatial Overlap Check:**

```python
block_sets = {name: set(gdf["block_id"].unique()) for name, gdf in splits}
overlap = ∩ all block_sets

# MUST BE: overlap == ∅ (empty set)
```

**KL-Divergence Check:**

```
KL(P || Q) = Σᵢ P(i) log(P(i) / Q(i))

wobei:
P, Q = Genus-Verteilungen in zwei Splits
```

**Quality Threshold:** KL < 0.07 (exzellente Stratifizierung trotz Spatial Blocking + Proximity Filtering Constraints)

**Return:** Validation Dictionary

```python
{
    "genus_distributions": {split_name: {genus: count}},
    "kl_divergences": {f"{split_a}_vs_{split_b}": kl_value},
    "spatial_overlap": int,        # MUST BE 0
    "block_counts": {split_name: n_blocks},
    "sample_sizes": {split_name: n_trees}
}
```

---

### 4. Proximity Filtering (`proximity.py`)

**Problem:** Sentinel-2 Mixed Pixel Contamination

**Sentinel-2 Spatial Resolution:** 10m × 10m pixels (B2, B3, B4, B8)

**Kontamination:** Wenn Baum A (Genus X) < 20m von Baum B (Genus Y):

- Pixel von Baum A enthält Spektralsignatur von Genus Y
- Feature-Label Mismatch → Noisige Labels
- Modell lernt inkorrekte Genus-Spektral-Mappings

#### Funktion 1: `compute_nearest_different_genus_distance(trees_gdf, genus_column)`

**Zweck:** Identifikation kontaminationsgefährdeter Bäume

**Algorithmus:**

```python
for each genus G:
    same_genus_trees = trees where genus == G
    diff_genus_trees = trees where genus != G

    for each tree T in same_genus_trees:
        distance[T] = min(euclidean_distance(T, D)
                         for D in diff_genus_trees)
```

**Output:** Series mit Distanzen (meters)

- `np.inf` wenn Genus isoliert (keine anderen Genera existieren)

**Complexity:** O(n × m) pro Genus (n = Genus-Bäume, m = Andere Bäume)

#### Funktion 2: `apply_proximity_filter(trees_gdf, threshold_m, genus_column)`

**Threshold:** Data-driven aus `proximity_filter.json` (exp_06)

- Empirisch bestimmt: Balanciert Retention Rate vs. Spectral Purity
- Typisch: 20m (2-Pixel Buffer)

**Filter-Logik:**

```python
distances = compute_nearest_different_genus_distance(trees_gdf)
filtered_gdf = trees_gdf[distances >= threshold_m]
```

**Statistics:**

```python
{
    "original_count": int,
    "removed_count": int,
    "retained_count": int,
    "retention_rate": float,      # ~0.85-0.90 erwartet
    "threshold_used": float
}
```

**Return:** (filtered_gdf, stats_dict)

#### Funktion 3: `analyze_genus_specific_impact(trees_gdf, threshold_m)`

**Zweck:** Validation der Filter-Uniformität

**Analyse:**

```python
for each genus G:
    total_trees = count(trees where genus == G)
    removed_trees = count(trees where genus == G and distance < threshold)
    removal_rate = removed_trees / total_trees
```

**Uniformity Criterion:** max(removal_rate) - min(removal_rate) < 0.10

- Verhindert Bias gegen bestimmte Genera
- Erhält Genus-Balance in Datensatz

**Return:** DataFrame mit Genus-spezifischen Removal Rates

---

## Runner Notebook Pipeline

### Workflow

```
Input: trees_clean_{city}.gpkg
  ↓
[1] Load JSON Configurations
  ↓
[2] Remove Redundant Features (selection.py)
  ↓
[3] Detect & Flag Outliers (outliers.py)
  ↓
[4] Assign Spatial Blocks (splits.py)
  ↓
[5] Create Baseline Splits
     ├─ Berlin: Train/Val/Test (70/15/15)
     └─ Leipzig: Finetune/Test (80/20)
  ↓
[6] Validate Baseline Splits (KL-divergence, Spatial Overlap)
  ↓
[7] Apply Proximity Filter (proximity.py)
  ↓
[8] Create Filtered Splits
     ├─ Berlin: Train/Val/Test (70/15/15)
     └─ Leipzig: Finetune/Test (80/20)
  ↓
[9] Validate Filtered Splits
  ↓
[10] Export 10 GeoPackages + Summary JSON
```

### Dual Dataset Strategy

**Rationale:** Ablation Study Flexibility in Phase 3

**Baseline Datasets (5 GeoPackages):**

1. `berlin_train.gpkg` - Alle Bäume (inkl. Proximity-nahe)
2. `berlin_val.gpkg`
3. `berlin_test.gpkg`
4. `leipzig_finetune.gpkg`
5. `leipzig_test.gpkg`

**Filtered Datasets (5 GeoPackages):**

1. `berlin_train_filtered.gpkg` - Proximity-gefiltert (≥ 20m zu anderen Genera)
2. `berlin_val_filtered.gpkg`
3. `berlin_test_filtered.gpkg`
4. `leipzig_finetune_filtered.gpkg`
5. `leipzig_test_filtered.gpkg`

**Phase 3 Experiment Matrix:**

```
Training Dataset     Test Dataset      Experiment Type
─────────────────────────────────────────────────────────
Baseline             Baseline          Standard (inkl. Noise)
Baseline             Filtered          Generalization Test
Filtered             Baseline          Robustness Test
Filtered             Filtered          Clean → Clean (Best Case)
```

---

## Output

### GeoPackages (10 Files)

**Location:** `data/phase_2_splits/`

**Berlin Baseline:**

- `berlin_train.gpkg` (70% der Bäume)
- `berlin_val.gpkg` (15% der Bäume)
- `berlin_test.gpkg` (15% der Bäume)

**Berlin Filtered:**

- `berlin_train_filtered.gpkg` (Proximity-gefiltert)
- `berlin_val_filtered.gpkg`
- `berlin_test_filtered.gpkg`

**Leipzig Baseline:**

- `leipzig_finetune.gpkg` (80% der Bäume)
- `leipzig_test.gpkg` (20% der Bäume)

**Leipzig Filtered:**

- `leipzig_finetune_filtered.gpkg` (Proximity-gefiltert)
- `leipzig_test_filtered.gpkg`

**Column Schema (alle GeoPackages):**

```
Identifier:
- tree_id (str)
- city (str)
- genus_latin (str)

Metadata (für Analysen):
- species_latin (str)
- genus_german (str)
- species_german (str)
- tree_type (str: "Laubbaum", "Nadelbaum")
- plant_year (int) — für Fehlermetrik-Analyse
- position_corrected (bool) — Position-QC-Flag
- correction_distance (float) — Korrektur-Distanz in Metern

Spatial:
- geometry (Point, EPSG:25833)
- block_id (str)

Features:
- B2_04, B3_04, B4_04, B8_04 (Sentinel-2 Bänder)
- NDVI_04, EVI_04, ... (Indizes)
- CHM_1m, CHM_5m, ... (Height-Statistiken)
- ~187 Features total (nach Redundancy-Removal)

Outlier Metadata:
- outlier_zscore (bool)
- outlier_mahalanobis (bool)
- outlier_iqr (bool)
- outlier_severity (str: "high", "medium", "low", "none")
- outlier_method_count (int: 0-3)

NICHT enthalten (vor Export entfernt):
- height_m (Katasterhöhe, redundant mit CHM-Features)
```

### Parquet Files (10 + 1 Geometry Lookup)

**Location:** `data/phase_2_splits/`

**Rationale:** ML-optimierte Alternative zu GeoPackages für Phase 3

- **5-10× schnelleres Laden** vs. GeoPackage (GeoPackage nutzt SQLite intern)
- **Keine Geometrie** (nicht für Training benötigt, siehe geometry_lookup.parquet)
- **Selektives Spalten-Laden** (Parquet unterstützt Column Pruning)
- **Snappy-Kompression** (optimiert für Lese-Geschwindigkeit)
- **Identisches Schema** wie GeoPackages (minus geometry)

**Berlin Baseline:**

- `berlin_train.parquet`
- `berlin_val.parquet`
- `berlin_test.parquet`

**Berlin Filtered:**

- `berlin_train_filtered.parquet`
- `berlin_val_filtered.parquet`
- `berlin_test_filtered.parquet`

**Leipzig Baseline:**

- `leipzig_finetune.parquet`
- `leipzig_test.parquet`

**Leipzig Filtered:**

- `leipzig_finetune_filtered.parquet`
- `leipzig_test_filtered.parquet`

**Geometry Lookup:**

- `geometry_lookup.parquet` (x/y Koordinaten für Visualisierung)

**Column Schema (Parquet Files):**

```
Identifier:
- tree_id (str)
- city (str)
- genus_latin (str)

Metadata (für Analysen):
- species_latin (str)
- genus_german (str)
- species_german (str)
- tree_type (str: "Laubbaum", "Nadelbaum")
- plant_year (int) — für Fehlermetrik-Analyse
- position_corrected (bool) — Position-QC-Flag
- correction_distance (float) — Korrektur-Distanz in Metern

Features:
- B2_04, B3_04, B4_04, B8_04 (Sentinel-2 Bänder)
- NDVI_04, EVI_04, ... (Indizes)
- CHM_1m, CHM_5m, ... (Height-Statistiken)
- ~187 Features total (nach Redundancy-Removal)

Outlier Metadata:
- outlier_zscore (bool)
- outlier_mahalanobis (bool)
- outlier_iqr (bool)
- outlier_severity (str)
- outlier_method_count (int)

Spatial:
- block_id (str)

NICHT in Parquet (vs. GeoPackages):
- geometry (→ geometry_lookup.parquet für Visualisierung)

Hinweis: height_m wurde bereits vor GeoPackage-Export entfernt (redundant mit CHM-Features)
```

**Geometry Lookup Schema:**

```
geometry_lookup.parquet:
- tree_id (str, non-null) — Baum-Identifier (nicht unique, Bäume erscheinen einmal pro Split)
- city (str, non-null)
- split (str, non-null) — z.B. "berlin_train", "leipzig_test"
- filtered (bool, non-null) — ob Baum in gefilterten Datasets
- x (float64, non-null) — UTM Easting (EPSG:25833)
- y (float64, non-null) — UTM Northing (EPSG:25833)

Hinweis: Bäume können in mehreren Splits erscheinen (baseline + filtered).
Composite Key: (tree_id, split)
```

**Verwendung in Phase 3:**

```python
# Laden für Training (kein geopandas benötigt)
import pandas as pd
train = pd.read_parquet("data/phase_2_splits/berlin_train.parquet")

# Visualisierung (Geometrie hinzufügen)
geo_lookup = pd.read_parquet("data/phase_2_splits/geometry_lookup.parquet")
train_with_geo = train.merge(geo_lookup, on="tree_id")
```

### Summary Metadata

**File:** `outputs/phase_2/metadata/phase_2_final_summary.json`

**JSON Schema:**

```json
{
  "version": string,
  "created": ISO8601_timestamp,
  "random_seed": int,
  "configurations": {
    "block_size_m": float,
    "proximity_threshold_m": float,
    "z_threshold": float,
    "mahalanobis_alpha": float,
    "iqr_multiplier": float,
    "features_removed": int
  },
  "outlier_stats": {
    "<city>": {
      "trees_removed": int,  // IMMER 0 (nur Flagging)
      "high_count": int,
      "medium_count": int,
      "low_count": int
    }
  },
  "proximity_filter_stats": {
    "<city>": {
      "original_count": int,
      "removed_count": int,
      "retention_rate": float
    }
  },
  "baseline_splits": {
    "<city>": {
      "<split_name>": {"count": int, "blocks": int}
    }
  },
  "filtered_splits": {/* wie baseline_splits */},
  "validation": {
    "<city>_<dataset_type>": {
      "spatial_overlap": int,  // MUSS 0 sein
      "kl_divergences": {
        "<split_a>_vs_<split_b>": float  // < 0.07 erwartet
      }
    }
  }
}
```

### Visualizations

**Location:** `outputs/phase_2/figures/02c_final_prep/`

**Plots:**

1. `split_size_comparison.png` - Baseline vs. Filtered Tree Counts (Grouped Bar Chart)
2. `genus_distribution_comparison.png` - Genus Proportionen Berlin Train (Baseline vs. Filtered)

---

## Quality Assurance

### Automated Assertions

**Spatial Disjointness:**

```python
assert validation["spatial_overlap"] == 0
# Kein Block darf in mehreren Splits vorkommen
```

**Outlier Non-Removal:**

```python
assert len(result) == len(input)
assert stats["trees_removed"] == 0
# Outliers nur getaggt, nicht entfernt
```

**KL-Divergence:**

```python
for kl_value in validation["kl_divergences"].values():
    assert kl_value < 0.07
# Genus-Verteilungen exzellent stratifiziert trotz Spatial Blocking Constraints
```

**Proximity Retention:**

```python
assert 0.80 <= filter_stats["retention_rate"] <= 0.95
# Filter entfernt 5-20% (erwarteter Range)
```

### Manual Validation Checks

**Pre-Execution:**

- [ ] Alle 4 JSON-Konfigurationen vorhanden
- [ ] Input GeoPackages existieren und sind nicht korrupt
- [ ] CRS ist EPSG:25833

**Post-Execution:**

- [ ] Genau 10 GeoPackages erstellt
- [ ] Alle Assertions passed (keine Fehler im Log)
- [ ] Summary JSON enthält alle erwarteten Felder
- [ ] Visualizations korrekt generiert (2 PNGs)

**Phase 3 Readiness:**

- [ ] GeoPackages nach Git committed
- [ ] Summary JSON nach Git committed
- [ ] Implementation Plan PRD 002c aktualisiert (Status: Done)

---

## Phase 3 Integration

### Data Format Strategy

Phase 2c produces **dual outputs** for each split (see [PRD 002c-ext: Parquet Export](../../PRDs/002c_parquet_export.md)):

- **GeoPackage** (`.gpkg`): Authoritative format with geometry — for traceability and spatial debugging
- **Parquet** (`.parquet`): ML-optimized format without geometry — for efficient training in Phase 3
- **Geometry Lookup** (`geometry_lookup.parquet`): tree_id → x/y coordinates for visualizing predictions

Phase 3 notebooks should **always load Parquet** (5-10× faster, no geopandas dependency needed):

### Experiment Setup

**Baseline Experiments:**

```python
import pandas as pd

train_df = pd.read_parquet("data/phase_2_splits/berlin_train.parquet")
test_df = pd.read_parquet("data/phase_2_splits/berlin_test.parquet")

# Outlier Ablation:
train_high_removed = train_df[train_df["outlier_severity"] != "high"]
train_high_med_removed = train_df[~train_df["outlier_severity"].isin(["high", "medium"])]
```

**Filtered Experiments:**

```python
train_df = pd.read_parquet("data/phase_2_splits/berlin_train_filtered.parquet")
test_df = pd.read_parquet("data/phase_2_splits/berlin_test_filtered.parquet")
```

**Transfer Learning:**

```python
# Pre-train on Berlin
train_df = pd.read_parquet("data/phase_2_splits/berlin_train.parquet")

# Fine-tune pool for Leipzig
finetune_pool = pd.read_parquet("data/phase_2_splits/leipzig_finetune.parquet")

# Few-shot sampling strategies:
finetune_10_per_genus = finetune_pool.groupby("genus_latin").sample(n=10, random_state=42)
finetune_50_per_genus = finetune_pool.groupby("genus_latin").sample(n=50, random_state=42)

# Test on Leipzig
test_df = pd.read_parquet("data/phase_2_splits/leipzig_test.parquet")
```

**Visualization (joining predictions with geometry):**

```python
import geopandas as gpd

geo_lookup = pd.read_parquet("data/phase_2_splits/geometry_lookup.parquet")
predictions = pd.merge(predictions_df, geo_lookup, on="tree_id")
gdf = gpd.GeoDataFrame(predictions, geometry=gpd.points_from_xy(predictions.x, predictions.y, crs="EPSG:25833"))
```

---

## Technische Details

### Performance

**Computational Complexity:**

- Data Loading: Linear in Dateigröße
- Feature Removal: O(n) - Spalten-Drop
- Outlier Detection: O(n × p) für Z-score/IQR, O(n × p²) für Mahalanobis (Kovarianzmatrix)
- Spatial Blocks: O(n) mit Spatial Index
- Splits Creation: O(n log n) - StratifiedGroupKFold
- Proximity Filter: O(n × m) per Genus (n = Genus-Bäume, m = Andere Bäume)
- Export GeoPackages: Linear in Datensatzgröße

**Memory:** Linear in Anzahl Bäume × Features (Standard Colab ausreichend)

### Error Handling

**Common Issues:**

1. **Missing JSON Config:**

   ```
   FileNotFoundError: Required config not found: proximity_filter.json
   → Solution: Run exp_06 notebook first
   ```

2. **CRS Mismatch:**

   ```
   ValueError: Expected CRS EPSG:25833, got EPSG:4326
   → Solution: Input GeoPackages wurden nicht korrekt projiziert (Phase 2b Fehler)
   ```

3. **Insufficient Blocks:**

   ```
   ValueError: Not enough blocks to create splits
   → Solution: block_size_m zu groß, zu wenige Blocks für Stratification
   ```

4. **Assertion Failure (Spatial Overlap):**
   ```
   AssertionError: Berlin: Spatial overlap detected!
   → Solution: Bug in StratifiedGroupKFold - prüfe block_id Zuordnung
   ```

---

## Literatur & Methodische Referenzen

**Spatial Cross-Validation:**

- Roberts et al. (2017): "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure"
- Valavi et al. (2019): "blockCV: An R package for generating spatially or environmentally separated folds for k-fold cross-validation of species distribution models"

**Outlier Detection:**

- Tukey, J.W. (1977): "Exploratory Data Analysis" (IQR Method)
- Mahalanobis, P.C. (1936): "On the generalized distance in statistics"

**Stratified Sampling:**

- Scikit-learn StratifiedGroupKFold Documentation
- Kohavi, R. (1995): "A study of cross-validation and bootstrap for accuracy estimation and model selection"

**Sentinel-2 Mixed Pixels:**

- Wulder et al. (2018): "Current status of Landsat program, science, and applications"
- Immitzer et al. (2016): "Tree Species Classification with Random Forest Using Very High Spatial Resolution 8-Band WorldView-2 Satellite Data"

---

**Autor:** Claude Sonnet 4.5
**Erstellung:** 2026-01-30
**Version:** 1.0
**Status:** Ready for Phase 3
