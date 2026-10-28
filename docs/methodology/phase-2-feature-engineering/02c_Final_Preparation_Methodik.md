# 02c_Final_Preparation_Methodik

## Ziel

Erstellung ML-ready Train/Val/Test Splits mit Dual-Dataset-Strategie (Baseline + Proximity-gefiltert) für Phase 3 Ablationsstudien.

**Output:** 10 Datensätze (5 Baseline + 5 Filtered), 147 Features nach Redundancy-Removal, Outlier-Flags

---

## Pipeline-Schritte

### 1. Feature Redundancy Removal

Entfernung hochkorrelierter Features aus `correlation_removal.json` (exp\_03). Details siehe exp\_03.

### 2. Outlier Detection & Flagging

Tripartite Consensus-Methode aus `outlier_thresholds.json` (exp\_04). **Nur Flagging, keine Removal.** Details siehe exp\_04.

### 3. Spatial Block Assignment

Regular Grid mit empirischer Block-Größe aus `spatial_autocorrelation.json` (exp\_05). Block-ID pro Baum. Details siehe exp\_05.

### 4. Stratified Spatial Splits

**Berlin: 70/15/15 (Train/Val/Test)**

- StratifiedGroupKFold mit `genus_latin` als Stratifikation und `block_id` als Group
- Spatial Disjointness: Kein Block in mehreren Splits

**Leipzig: 80/20 (Finetune-Pool/Test)**

- Finetune-Pool (89.335 Bäume): Quelle für stratifizierte Fraktions-Subsets in Phase 3 (10/25/50/100 % → 8.040/20.100/40.200/80.401 Trainingssamples)
- Independent Test für Leipzig-Evaluation

### 5. Split-Validierung

**Spatial Disjointness:** Block-Overlap = 0 (obligatorisch)

**KL-Divergenz:** \< 0.07 (Genus-Proportionen über Splits)

- Literatur-Standard 0.05 gilt für rein zufällige Stratifizierung
- 0.07 berücksichtigt zusätzliche Constraints (Spatial Blocking + Proximity Filtering)
- Empirisch validiert: Genus-Proportionen weichen \< 1 Prozentpunkt ab

### 6. Proximity Filtering + Dual Datasets

Anwendung des 5m-Filters aus `proximity_filter.json` (exp\_06, `recommended_threshold_m` = 5). Details siehe exp\_06.

**Dual Dataset Strategy:**

| Datensatz | Berlin | Leipzig |
| --- | --- | --- |
| **Baseline** | Train / Val / Test | Finetune / Test |
| **Filtered** | Train / Val / Test (filtered) | Finetune / Test (filtered) |

→ Phase 3 Experiment-Matrix: Baseline↔Filtered Kombinationen für Ablation

---

## Output

**10 GeoPackages + 10 Parquet** (ML-optimiert, 5–10× schneller)

Jeder Datensatz enthält:

- Features (147 nach Redundancy-Removal)
- Outlier-Flags (outlier\_severity: high/medium/low/none)
- Spatial Block-ID
- Metadaten (tree\_id, city, genus\_latin, plant\_year, etc.)

**Parquet-Rationale:** Kein geopandas für Training nötig, Column Pruning, Snappy-Kompression. Geometry separat in `geometry_lookup.parquet` für Visualisierung.

---

## Schlüssel-Parameter

| Parameter | Wert | Quelle |
| --- | --- | --- |
| Berlin Split-Ratio | 70/15/15 | Standard ML-Praxis |
| Leipzig Split-Ratio | 80/20 | Transfer Learning Setup |
| KL-Divergenz-Limit | \< 0.07 | Empirisch validiert |
| Random Seed | 42 | Reproduzierbarkeit |

**Literatur:**

- Roberts et al. (2017): Spatial Cross-Validation Strategies
- Valavi et al. (2019): blockCV für räumlich getrennte Folds
- Tukey (1977): IQR-Methode
