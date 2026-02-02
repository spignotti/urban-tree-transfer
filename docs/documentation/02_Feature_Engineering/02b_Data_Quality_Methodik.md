# Phase 2b: Data Quality Control Methodik

**Phase:** Feature Engineering - Part 2 (Quality Control)
**Letzte Aktualisierung:** 2026-01-30
**Status:** ✅ Abgeschlossen

---

## Überblick

Diese Phase bereinigt die in Phase 2a extrahierten Features durch Temporal Reduction, NaN-Handling, CHM Feature Engineering und Plausibilitätsfilterung. Der zentrale methodische Fokus liegt auf der **Vermeidung von Data Leakage** durch ausschließlich within-tree Operationen.

**Input:** `data/phase_2_features/trees_with_features_{city}.gpkg` (Phase 2a)

**Output:** `data/phase_2_features/trees_clean_{city}.gpkg` (0 NaN, quality-assured)

**Kritischer Unterschied zur Legacy-Pipeline:**

- ❌ **Legacy:** NaN-Imputation mit Genus-/Stadt-Mittelwerten → Data Leakage
- ✅ **Aktuell:** Ausschließlich within-tree Temporal Interpolation → Kein Leakage

---

## Methodische Schritte

### 1. Genus-Filterung (Deciduous Only)

**Zweck:** Fokussierung auf Laubbäume zur Gewährleistung spektraler Homogenität.

**Methodischer Ansatz:**

Der Analysis Scope wird durch `chm_assessment.json` (exp_02) bestimmt. Die Entscheidung basiert auf zwei Kriterien:

```
WENN (n_conifer_genera < 3) ODER (n_conifer_samples < 500):
    → Filterung auf deciduous_only
SONST:
    → Alle Genera behalten
```

**Rationale:**

- **Spektrale Unterschiede:** Nadelbäume haben fundamental andere NDVI-Jahresverläufe (immergrün vs. laubabwerfend)
- **Phenologische Differenzierung:** Modell-Performance verbessert sich bei spektral homogenen Klassen
- **Sample-Anforderung:** Mindestens 3 Genera und 500 Samples für statistisch stabile Klassifikation

**Genera-Klassifikation:**

- **Deciduous (Laubbäume):** TILIA, ACER, QUERCUS, FRAXINUS, PLATANUS, BETULA, PRUNUS, CARPINUS, ALNUS, SORBUS, ULMUS, POPULUS, ROBINIA, SALIX, FAGUS, AESCULUS
- **Coniferous (Nadelbäume):** PINUS, PICEA, THUJA, TAXUS, ABIES, LARIX

**Praktische Konsequenz:** In urbanen Beständen dominieren Laubbäume (>95% der Samples), sodass typischerweise `analysis_scope = "deciduous_only"` resultiert.

---

### 2. Plant-Year-Filterung (Tree Visibility)

**Zweck:** Entfernung von Bäumen, die zum Sentinel-2 Aufnahmezeitpunkt zu jung für zuverlässige spektrale Detektion sind.

**Methodischer Ansatz:**

Der Plant-Year-Threshold wird **nicht hardcoded**, sondern stammt aus statistischer Analyse in exp_02:

1. **CHM vs. Plant Year Regression:** Median-CHM pro Pflanzjahr-Kohorte
2. **Detection Threshold:** 2.0m (Sentinel-2 10m-Pixel-Sichtbarkeit)
3. **Breakpoint-Identifikation:** Erstes Jahr mit median CHM < 2.0m
4. **Threshold:** Letztes Jahr mit median CHM ≥ 2.0m

**Beispiel (typische Werte):**

- Plant Year 2015: Median CHM = 8.2m → OK
- Plant Year 2016: Median CHM = 6.5m → OK
- Plant Year 2017: Median CHM = 4.1m → OK
- Plant Year 2018: Median CHM = 2.8m → OK (Grenzwertig)
- Plant Year 2019: Median CHM = 1.5m → **Verwerfen**
- **Threshold:** 2018 (recommended_max_plant_year)

**Filter-Regel:**

```
Behalte Baum WENN:
    (plant_year ≤ recommended_max_plant_year) ODER (plant_year = NaN)
```

**Rationale:**

- Jungbäume mit CHM <2m sind in 10m-Sentinel-2-Pixeln vom Untergrund kontaminiert
- Spektrale Signatur nicht repräsentativ für etablierte Baumkronen
- NaN-plant_year wird beibehalten (konservativ: kein Alter bekannt = potentiell alt)

**Metadaten-Erhaltung:** Die Spalte `plant_year` bleibt im Datensatz erhalten (Metadaten), nur Filterung basiert darauf.

---

### 3. Temporal Reduction (Month Selection)

**Zweck:** Reduktion auf diskriminative Monate basierend auf JM-Distance-Analyse (exp_01).

**Methodischer Ansatz:**

Laden der selektierten Monate aus `temporal_selection.json`:

- **Typische Selektion:** 6-10 Monate (z.B. März-Oktober)
- **Verworfene Monate:** Winter (November-Februar), niedrige JM-Distance

**Feature-Reduktion:**

- **Vor Temporal Reduction:** 23 Features × 12 Monate = 276 Features
- **Nach Temporal Reduction:** 23 Features × 8 Monate = 184 Features (~33% Reduktion)

**Erhaltene Features:**

- Alle Metadaten-Spalten (tree_id, city, genus_latin, etc.)
- CHM_1m (nicht temporal)
- Sentinel-2 Features nur für selektierte Monate

**Rationale:**

- Winter-Monate zeigen geringe genus-spezifische Variabilität (Laubbäume laublos)
- Fokus auf Growing Season maximiert phenologische Differenzierung
- Dimensionsreduktion verbessert Model Generalization

---

### 4. NaN-Analyse und Filterung

**Zweck:** Identifikation von Bäumen mit zu vielen fehlenden Monaten für zuverlässige Imputation.

**Methodischer Ansatz:**

**Phase 1 - NaN-Verteilung:**

- Analyse pro Feature: Anzahl und Prozentsatz betroffener Bäume
- Identifikation systematischer Muster (z.B. bestimmte Monate häufiger betroffen)

**Phase 2 - Tree-Removal-Schwellenwert:**

```
max_nan_months = 2  (aus feature_config.yaml)

Für jede Feature-Basis (z.B. "NDVI"):
    Wenn tree hat >2 NaN-Monate in NDVI-Features:
        → Baum entfernen
```

**Begründung Schwellenwert:**

- **≤2 NaN-Monate:** Lineare Interpolation zuverlässig (z.B. 6/8 Monate vorhanden = 75%)
- **>2 NaN-Monate:** Interpolation unsicher, phenologisches Muster zu lückenhaft
- **Konservativ:** Lieber weniger Bäume mit hoher Qualität als mehr Bäume mit Artefakten

**NaN-Quellen:**

- Cloud-masked Sentinel-2 Pixel (~0.5-1% pro Feature)
- CHM NoData (Bäume außerhalb Raster-Bounds, <2%)
- Datenprozessierungs-Fehler (selten)

**Erwartete Removal-Rate:** ~3-5% der Bäume

---

### 5. Within-Tree Temporal Interpolation (KRITISCH)

**Zweck:** Füllen verbleibender NaN-Werte ohne Data Leakage zwischen Train/Val/Test-Sets.

**Methodischer Ansatz:**

**Unabhängigkeit:** Jeder Baum wird isoliert prozessiert. Die Interpolation eines Baums nutzt **ausschließlich dessen eigene Zeitreihe**.

**Algorithmus (pro Baum, pro Feature):**

**Schritt 1 - Interior NaN (inmitten der Zeitreihe):**

- **Methode:** Lineare Interpolation zwischen benachbarten validen Werten
- **Formel:** `value(t) = value(t-1) + (value(t+1) - value(t-1)) × (t - (t-1)) / ((t+1) - (t-1))`
- **Beispiel:** NDVI [0.5, NaN, 0.7] → [0.5, **0.6**, 0.7]

**Schritt 2 - Edge NaN (Anfang oder Ende der Zeitreihe):**

- **1 Monat Edge-Gap:** Nearest-Neighbor Fill (Forward-fill / Backward-fill)
  - Start-Edge: `value(0) = value(1)` (Forward-fill)
  - End-Edge: `value(n) = value(n-1)` (Backward-fill)
- **≥2 Monate Edge-Gap:** **Keine Imputation** → Baum wird im nächsten Schritt entfernt

**Edge-NaN-Tolerance-Begründung:**

- **1 Monat:** 87.5% Daten vorhanden (7/8 Monate), minimale Information-Loss
- **≥2 Monate:** Signifikanter Verlust phenologischer Information (z.B. Frühling oder Herbst fehlt)
- **Literatur-Referenz:** Jönsson & Eklundh (2004) empfehlen max 1-2 Edge-Gaps für Vegetations-Zeitreihen

**Data-Leakage-Prävention:**

❌ **Verboten (Legacy-Ansatz):**

- Genus-Mittelwerte für NaN-Imputation
- Stadt-Mittelwerte für NaN-Imputation
- Irgendwelche cross-tree Statistiken

✅ **Erlaubt (Aktuelle Methode):**

- Within-tree linear interpolation
- Within-tree nearest-neighbor fill (max 1 Edge-month)

**Validierung der Unabhängigkeit:**

```
Test: Entferne einen Baum aus dem Datensatz
→ Interpolierte Werte aller anderen Bäume MÜSSEN identisch bleiben
```

**Technische Umsetzung:** `pandas.Series.interpolate(method='linear')` mit `limit_area='inside'` für Interior, dann `ffill(limit=1)` / `bfill(limit=1)` für Edges.

---

### 6. CHM Feature Engineering

**Purpose:** Normalize CHM values to improve cross-city transferability while removing genus-height bias.

**Updated (2026-02-01): Genus×City Normalization**

**Feature 1: CHM_1m_zscore (Z-Score Normalization)**

```
CHM_zscore = (CHM_1m - μ_genus,city) / σ_genus,city
```

**Feature 2: CHM_1m_percentile (Percentile Rank)**

```
CHM_percentile = rank(CHM_1m, genus×city) / n_genus,city × 100
```

**Rationale:**
- Different genera have different natural height ranges (e.g., QUERCUS vs. MALUS).
- City-level normalization mixes these ranges and implicitly encodes genus.
- Genus-specific normalization removes the genus mean effect, so features represent
  relative size within genus (age, vitality, site quality).

**Edge Cases (rare genera):**
- If a genus has <10 samples in a city, fall back to city-level normalization.
- This prevents unstable statistics for rare genera; fallbacks are logged as warnings.

---

### 7. NDVI Plausibility Filtering

**Zweck:** Entfernung von Bäumen mit implausibel niedrigen Vegetations-Signalen.

**Methodischer Ansatz:**

**Plausibilitäts-Metrik:**

```
max_NDVI = max(NDVI_03, NDVI_04, ..., NDVI_10)

Schwellenwert: max_NDVI ≥ 0.3
```

**Filter-Regel:**

```
Behalte Baum WENN: max_NDVI ≥ 0.3
Entferne Baum WENN: max_NDVI < 0.3
```

**Begründung Schwellenwert 0.3:**

NDVI < 0.3 in **allen** Growing-Season-Monaten deutet auf:

1. **Fehlklassifikation:** Kein Baum, sondern versiegelte Fläche / Gebäude
2. **Toter/sterbender Baum:** Keine gesunde Vegetation
3. **Positions-Fehler:** Kataster-Koordinate zeigt nicht auf Baum
4. **Spektrale Kontamination:** Baum zu klein für 10m-Pixel

**Literatur-Kontext:**

- Gesunde Vegetation: NDVI = 0.6-0.9 (Growing Season)
- Sparse Vegetation: NDVI = 0.2-0.4
- Bare Soil/Urban: NDVI = 0.0-0.2
- **0.3 ist konservativer Schwellenwert** (Minimum für "Vegetation erkennbar")

**Erwartete Removal-Rate:** ~1-3% der Bäume

**Alternativer Ansatz (nicht implementiert):**

- Min-NDVI über alle Monate (zu streng, entfernt Winter-Werte)
- Mean-NDVI (weniger sensitiv für Outliers)
- **Max-NDVI gewählt:** Wenn selbst im besten Monat NDVI <0.3 → sicher kein gesunder Baum

---

## Data Leakage Prevention - Zusammenfassung

**Zentrale Regel:** Jeder Baum muss **unabhängig prozessierbar** sein für valide Train/Val/Test-Splits.

### ✅ **Erlaubte Operationen (Kein Leakage):**

1. **Within-Tree Temporal Interpolation**
   - Jeder Baum nutzt nur eigene Zeitreihe
   - Unabhängigkeit: Baum-Removal ändert keine anderen Bäume

2. **Stadt-Level CHM-Normalisierung**
   - CHM ist externe Datenquelle (LiDAR), nicht von anderen Bäumen abgeleitet
   - Normalisiert stadtspezifische Unterschiede (z.B. Baumalter-Verteilung)

3. **Schwellenwert-basierte Filter**
   - Plant Year, NDVI Plausibility: Absolute Kriterien, keine relativen Vergleiche

### ❌ **Verbotene Operationen (Leakage-Risiko):**

1. **Genus-/Stadt-Mittelwerte für NaN-Imputation**
   - Information von Train-Set würde in Val/Test-Set fließen
   - Legacy-Pipeline-Fehler

2. **Genus-Level NDVI-Normalisierung**
   - NDVI ist abhängige Variable, Genus-Mittelwert würde Klassen-Information leaken

3. **Cross-Tree Outlier-Detection innerhalb Quality-Control**
   - Outlier-Detection kommt in Phase 2c (nach Splits)

---

## Output-Datensatz: `trees_clean_{city}.gpkg`

### Dateiformat

**Datei:** `data/phase_2_features/trees_clean_{city}.gpkg` (GeoPackage)
**CRS:** EPSG:25833 (UTM Zone 33N)
**Geometrie-Typ:** Point

### Schema

#### Metadaten-Spalten (11)

| Spalte              | Typ     | Quelle   | Beschreibung                            |
| ------------------- | ------- | -------- | --------------------------------------- |
| tree_id             | str     | Phase 1  | Eindeutige Baum-ID                      |
| city                | str     | Phase 1  | Stadtname (berlin/leipzig)              |
| genus_latin         | str     | Phase 1  | Gattung lateinisch (UPPERCASE)          |
| species_latin       | str     | Phase 1  | Art lateinisch (lowercase, nullable)    |
| genus_german        | str     | Phase 1  | Gattung deutsch (nullable)              |
| species_german      | str     | Phase 1  | Art deutsch (nullable)                  |
| plant_year          | Int64   | Phase 1  | Pflanzjahr (nullable)                   |
| height_m            | Float64 | Phase 1  | Kataster-Höhe in Metern (nullable)      |
| tree_type           | str     | Phase 1  | anlagenbaeume/strassenbaeume (nullable) |
| position_corrected  | bool    | Phase 2a | Wurde Position korrigiert?              |
| correction_distance | Float64 | Phase 2a | Korrektur-Distanz in Metern             |

**Hinweis:** Alle Metadaten bleiben unverändert aus Phase 1/2a.

#### CHM-Features (3)

| Feature           | Typ     | Quelle   | Beschreibung                      |
| ----------------- | ------- | -------- | --------------------------------- |
| CHM_1m            | Float64 | Phase 2a | Kronenhöhe aus 1m CHM (Rohdaten)  |
| CHM_1m_zscore     | Float64 | Phase 2b | Z-Score normalisiert (pro Stadt)  |
| CHM_1m_percentile | Float64 | Phase 2b | Percentile Rang 0-100 (pro Stadt) |

**0 NaN-Werte** in allen CHM-Features.

#### Sentinel-2 Features (Temporal)

**Anzahl:** 23 Features × n_selected_months

**Typische Konfiguration (8 Monate):** 23 × 8 = **184 Features**

**Feature-Basis (23):**

- 10 Spektrale Bänder: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
- 13 Vegetations-Indices: NDVI, EVI, GNDVI, VARI, NDre1, NDVIre, CIre, IRECI, RTVIcore, NDWI, MSI, NDII, kNDVI

**Temporal Suffix:** `_{MM}` (z.B. `_03`, `_04`, ..., `_10`)

**Beispiel-Features:**

- `NDVI_03`, `NDVI_04`, `NDVI_05`, `NDVI_06`, `NDVI_07`, `NDVI_08`, `NDVI_09`, `NDVI_10`
- `B8_03`, `B8_04`, ..., `B8_10`

**0 NaN-Werte** in allen Sentinel-2-Features (interpoliert oder Baum entfernt).

**Wertebereich:**

- Spektrale Bänder: 0-10000 (Surface Reflectance)
- Vegetations-Indices: -1 bis +1 (normalisiert)

### Qualitätskriterien (Validierung)

**Obligatorisch:**

- ✅ 0 NaN-Werte in allen Feature-Spalten
- ✅ CRS ist EPSG:25833
- ✅ CHM_1m_zscore und CHM_1m_percentile vorhanden
- ✅ Temporal Features nur für selected_months
- ✅ Alle genus_latin in deciduous_genera (wenn scope = "deciduous_only")
- ✅ Alle plant_year ≤ recommended_max_plant_year (oder NaN)
- ✅ Alle max_NDVI ≥ 0.3
- ✅ Retention-Rate >85%

**Optional (Plausibilität):**

- CHM_1m Range: 1-50m (urbane Bäume)
- CHM_zscore Range: -3 bis +3 (99.7% in ±3σ)
- NDVI Range (Growing Season): 0.3-0.9
- Keine duplizierten tree_ids

---

## Verwendung für Phase 2c und Phase 3

### Phase 2c: Final Preparation

**Input:** `trees_clean_{city}.gpkg` (dieser Output)

**Weitere Schritte:**

1. Correlation Analysis → Redundante Features entfernen
2. Outlier Detection → Consensus-based Removal
3. Spatial Splits → Train/Val/Test mit spatial disjointness

**Garantie:** Keine NaN-Werte, keine Data-Leakage-Risiken aus Quality Control.

### Phase 3: Experiments

**Input:** Finale Split-Datasets (aus Phase 2c)

**ML-Ready:**

- Alle Features numerisch (Float64)
- 0 NaN-Werte
- Standardisierte Temporal Resolution
- CHM-Features normalisiert für Cross-City-Transfer

---

## Vergleich Legacy vs. Aktuelle Pipeline

| Aspekt                 | Legacy Pipeline                    | Aktuelle Pipeline                   |
| ---------------------- | ---------------------------------- | ----------------------------------- |
| **NaN-Imputation**     | Genus-Mittelwerte                  | Within-tree temporal interpolation  |
| **Data Leakage**       | ✗ Ja (Train/Val/Test kontaminiert) | ✓ Nein (tree-independent)           |
| **Edge NaN**           | Global-fill oder entfernen         | 1 month = fill, ≥2 months = remove  |
| **CHM Features**       | CHM_mean, CHM_max, CHM_std (10m)   | CHM_1m, zscore, percentile (1m)     |
| **CHM Resolution**     | 10m (resampled, cross-tree)        | 1m (direct, no contamination)       |
| **Plant Year Filter**  | Hardcoded (2018)                   | Statistisch bestimmt (exp_02)       |
| **Temporal Reduction** | Hardcoded (März-Oktober)           | JM-basiert (exp_01)                 |
| **Genus Scope**        | Hardcoded (deciduous only)         | Datengetrieben (exp_02)             |
| **NDVI Plausibility**  | Implizit in Outlier-Detection      | Expliziter Filter (max_NDVI ≥ 0.3)  |
| **Retention-Rate**     | ~88%                               | >90% (weniger aggressive Filterung) |

**Zentrale Verbesserung:** Elimination von Data Leakage durch within-tree Operationen.

---

## Nächste Schritte

**Nach Abschluss Phase 2b:**

1. **Exploratory Notebooks ausführen:**
   - `exp_03_correlation_analysis.ipynb` → Identifikation redundanter Features
   - `exp_04_outlier_thresholds.ipynb` → Validierung Outlier-Detection-Parameter
   - `exp_05_spatial_autocorrelation.ipynb` → Bestimmung optimaler Block-Size

2. **Phase 2c implementieren:**
   - Correlation-based Feature Removal
   - Consensus Outlier Detection
   - Spatial Block Splits
   - Output: 5 GeoPackages (berlin_train, berlin_val, berlin_test, leipzig_finetune, leipzig_test)

3. **Phase 3 vorbereiten:**
   - ML-Ready Datasets für Model Training
   - Baseline Models (Random Forest, XGBoost)
   - Transfer Learning Experimente

---

**Dokumentations-Status:** ✅ Phase 2b (Data Quality Control) vollständig dokumentiert
