# 03_Feature_Engineering_Ergebnisse

> Migriert aus Notion am 2026-10-28 (Seite `31f356451f668010a576eb8e9c46df49`, zuletzt bearbeitet 2026-03-24). Text unverändert übernommen.

**Phase:** Feature Engineering \| **Status:** ✅ Abgeschlossen \| **Pipeline-Rerun:** 24.03.2026

---

## 1 Pipeline-Übersicht

| Schritt | Berlin | Leipzig | Features |
| --- | --- | --- | --- |
| **Input (Phase 1)** | 905.132 | 167.867 | 11 (Metadaten) |
| **02a Feature Extraction** | 905.132 | 167.867 | 277 |
| **02b Data Quality** | 830.153 | 153.629 | 187 |
| **02c Correlation Removal** | 830.153 | 153.629 | 147 |
| **02c Proximity Filter** | 659.266 | 122.756 | 147 |

---

## 2 Exploratory Analyses

### 2.1 Exp 01 -- Temporal Selection (JM-Distanz)

**Methode:** Jeffries-Matusita-Distanz zwischen Gattungspaaren pro Monat

**Datenbasis:** 333.344 Bäume (Sample Berlin + Leipzig), 496 Gattungspaare

**Cross-City Consistency:** Spearman ρ = 0.951 (p \< 0.001)

**Ausgewählte Monate:** 4--11 (April bis November)

| Rank | Monat | JM |  |
| --- | --- | --- | --- |
| 1 | **8** | **0.905** | ✅ Peak |
| 2 | 6 | 0.904 | ✅ |
| 3 | 9 | 0.894 | ✅ |
| 4 | 7 | 0.881 | ✅ |
| 5 | 10 | 0.871 | ✅ |
| 6 | 5 | 0.870 | ✅ |
| 7 | 4 | 0.824 | ✅ |
| 8 | 11 | 0.808 | ✅ Cut-off |
| 9 | 3 | 0.801 | ❌ |
| 10 | 2 | 0.782 | ❌ |
| 11 | 12 | 0.750 | ❌ |
| 12 | 1 | 0.708 | ❌ |

---

### 2.2 Exp 02 -- CHM Assessment

**Methode:** Diskriminierungskraft (η²), Transfer-Risiko (Cohen's d), Kataster-Validierung

| Metrik | Wert |
| --- | --- |
| η² Berlin | 0.169 |
| η² Leipzig | 0.140 |
| Mean Cohen's d | 0.184 (Low Transfer Risk) |
| Korrelation CHM ↔ Kataster-Höhe | r = 0.316 |

**Top-5 Cohen's d (größter Stadt-Unterschied):**

| Genus | d |  | Genus | d |
| --- | --- | --- | --- | --- |
| ALNUS | 0.739 |  | CRATAEGUS | −0.470 |
| BETULA | 0.426 |  | SOPHORA | −0.428 |
| PLATANUS | 0.342 |  | CARPINUS | −0.230 |
| TILIA | 0.294 |  | PYRUS | −0.193 |
| ROBINIA | 0.245 |  | FRAXINUS | −0.187 |

⚠️ **NOT FOUND:** Welch-F pro Genus, p-Wert pro Genus

---

### 2.3 Exp 03 -- Correlation Analysis

**Methode:** Pearson-Korrelation, Schwellenwert r > 0.95, Sample 10.000 Bäume/Stadt

**Entfernte Feature-Familien (5):**

| Entfernt | Redundant mit | r |
| --- | --- | --- |
| B6 | B7 | 0.966 |
| B7 | B8A | 0.984 |
| GNDVI | NDVI | 0.966 |
| NDII | NDWI | 0.959 |
| kNDVI | NDVI | 0.993 |

**Feature-Reduktion:** 276 → 184 (Temporal) → 147 (Correlation). Reduktion: 40 Features (21.4%)

**Behaltene Bänder (8):** B2, B3, B4, B5, B8, B8A, B11, B12

**Behaltene Indizes (10):** NDVI, EVI, VARI, NDre1, NDVIre, CIre, IRECI, RTVIcore, NDWI, MSI

---

### 2.4 Exp 04 -- Outlier Detection

**Methode:** Z-Score (3σ), Mahalanobis (α=0.001), IQR (1.5×). Flagging ohne Removal.

**Gesamt (983.782 Bäume):**

| Severity | Anzahl | Anteil |
| --- | --- | --- |
| None | 943.750 | 95.9% |
| Low (1 Methode) | 36.165 | 3.7% |
| Medium (2 Methoden) | 3.822 | 0.4% |
| High (3 Methoden) | 45 | 0.005% |

**Per-City:**

| Metrik | Berlin | Leipzig |
| --- | --- | --- |
| None | 797.031 (96.0%) | 146.719 (95.5%) |
| Low | 29.610 (3.6%) | 6.555 (4.3%) |
| Medium | 3.467 (0.4%) | 355 (0.2%) |
| High | 45 (0.005%) | 0 (0%) |
| Z-Score | 19.881 | 4.133 |
| Mahalanobis | 6.252 | 577 |
| IQR | 10.546 | 2.555 |

---

### 2.5 Exp 05 -- Spatial Autocorrelation (Moran's I)

**Methode:** Moran's I pro Feature, 5 repräsentative Features (B4\_07, B8\_07, CHM\_1m, NDVI\_07, NDre1\_07)

**Empfohlene Block-Größe:** 1200m

**Residual Autocorrelation bei 1200m:** 0.170

| Lag (m) | Berlin | Leipzig |
| --- | --- | --- |
| 100 | 0.565 | 0.557 |
| 150 | 0.504 | 0.492 |
| 200 | 0.460 | 0.444 |
| 250 | 0.424 | 0.406 |
| 300 | 0.396 | 0.377 |
| 350 | 0.372 | 0.352 |
| 400 | 0.351 | 0.330 |
| 450 | 0.333 | 0.311 |
| 500 | 0.316 | 0.293 |
| 600 | 0.289 | 0.263 |
| 700 | 0.265 | 0.237 |
| 800 | 0.244 | 0.216 |
| 1000 | 0.210 | 0.182 |
| **1200** | **0.185** | **0.154** |

**p-Werte:** Alle Kombinationen (Stadt × Feature × Lag) signifikant bei p = 0.001 (Permutationstest, 999 Permutationen). Significant Rate = 1.000 über alle 5 Features und 14 Lags.

---

### 2.6 Exp 06 -- Proximity Filter

**Methode:** Punkt-zu-Punkt Distanz, Mixed-Genus Filtering

| Threshold | Retention | Behalten |
| --- | --- | --- |
| **5m** | **79.4%** | **659.266** |
| 10m | 51.1% | 424.466 |
| 15m | 34.7% | 287.897 |
| 20m | 26.2% | 217.654 |

**Empfehlung:** 5m (Balance Spektral-Reinheit / Sample Size)

---

## 3 Runner

### 3.1 Runner 02a -- Feature Extraction

**Laufzeit:** 27 min \| **Input:** 1.072.999 → **Output:** 1.072.999 Bäume, 277 Features

**Feature-Kategorien:**

- **Metadaten (11):** tree\_id, city, genus\_latin, species\_latin, genus\_german, species\_german, plant\_year, height\_m, tree\_type, position\_corrected, correction\_distance
- **CHM (1):** CHM\_1m
- **Sentinel-2 (276):** 10 Spektralbänder + 13 Vegetationsindizes × 12 Monate

**Position Correction:** Berlin 0 korrigiert, Leipzig 158.931 korrigiert (94.7%, Median \~5m)

---

### 3.2 Runner 02b -- Data Quality

**Laufzeit:** 17 min \| **Input:** 1.072.999 → **Output:** 983.782 Bäume, 187 Features

**Retention Rates:**

| Filter | Berlin | Leipzig |
| --- | --- | --- |
| Plant-year | 0.983 | 0.959 |
| NaN | 0.954 | 0.986 |
| NDVI | 0.978 | 0.967 |
| **Gesamt QC** | **0.917** | **0.915** |
| Proximity | 0.794 | 0.799 |

**Baumanzahl:**

| Schritt | Berlin | Leipzig |
| --- | --- | --- |
| Input | 905.132 | 167.867 |
| Nach QC | 830.153 | 153.629 |
| Nach Proximity | 659.266 | 122.756 |

**Feature-Schritte:**

- Temporal Selection: 276 → 184 (92 Winter-Features entfernt)
- CHM Engineering: +2 (CHM\_1m\_zscore, CHM\_1m\_percentile)
- Correlation Removal: 187 → 147 (40 entfernt, 21.4%)

---

### 3.3 Runner 02c -- Final Preparation

**Laufzeit:** 41 min \| **Input:** 983.782 Bäume, 187 Features → **Output:** 2 Split-Varianten, 147 Features, 21 Export-Dateien

**Finale Features (147):**

- Spektralbänder: 8 × 8 Monate = 64
- Vegetationsindizes: 10 × 8 Monate = 80
- CHM: 3 (CHM\_1m, CHM\_1m\_zscore, CHM\_1m\_percentile)

---

## 4 Splits

### 4.1 Baseline (983.782 Bäume)

**Berlin (830.153):**

| Split | Bäume | Ratio |
| --- | --- | --- |
| Train | 600.299 | 72.3% |
| Val | 120.874 | 14.6% |
| Test | 108.980 | 13.1% |

**Leipzig (153.629):**

| Split | Bäume | Ratio |
| --- | --- | --- |
| Finetune | 122.538 | 79.8% |
| Test | 31.091 | 20.2% |

**KL-Divergenzen:** Train/Val 0.015 \| Train/Test 0.029 \| Val/Test 0.048 \| Leipzig FT/Test 0.053

---

### 4.2 Filtered (782.022 Bäume, 79.5% Retention)

**Berlin (659.266):**

| Split | Bäume | Ratio |
| --- | --- | --- |
| Train | 471.815 | 71.6% |
| Val | 99.603 | 15.1% |
| Test | 87.848 | 13.3% |

**Leipzig (122.756):**

| Split | Bäume | Ratio |
| --- | --- | --- |
| Finetune | 91.080 | 74.2% |
| Test | 31.676 | 25.8% |

**KL-Divergenzen:** Train/Val 0.024 \| Train/Test 0.017 \| Val/Test 0.036 \| Leipzig FT/Test 0.061

---

## 5 Genus-Verteilung

**30 Genera nach QC (beide Städte):**

ACER, AESCULUS, AILANTHUS, ALNUS, BETULA, CARPINUS, CORNUS, CORYLUS, CRATAEGUS, FAGUS, FRAXINUS, GLEDITSIA, JUGLANS, LIQUIDAMBAR, MALUS, PICEA, PINUS, PLATANUS, POPULUS, PRUNUS, PYRUS, QUERCUS, ROBINIA, SALIX, SOPHORA, SORBUS, TAXUS, THUJA, TILIA, ULMUS

**is\_conifer:** PICEA, PINUS, TAXUS, THUJA (4). Alle übrigen 26 = Laubbaum.

⚠️ Absolute Counts pro Stadt nicht aus Pipeline-JSONs extrahiert.

---

## 6 Qualitätssicherung

| Kriterium | Ergebnis |
| --- | --- |
| Spatial Independence | ✅ 0 Block-Overlaps in allen Splits |
| Genus Stratification | ✅ Alle KL \< 0.07 |
| Outlier Flagging | ✅ 96.0% clean (Berlin), 95.5% (Leipzig) |
| Proximity Retention | ✅ 79.4% (Berlin), 79.9% (Leipzig) |
| Cross-City Consistency | ✅ Spearman ρ = 0.951 |
| CHM Transfer Risk | ✅ Mean Cohen's d = 0.184 |

---

## 7 Fehlende Daten

- Welch-F und p-Werte pro Genus (chm\_assessment.json)
- Absolute Genus-Counts pro Stadt

---

*Letzte Aktualisierung: 24.03.2026*
