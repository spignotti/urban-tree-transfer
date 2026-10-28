# 02b_Data_Quality_Methodik

## Überblick

Bereinigung der Phase-2a-Features: Temporal Reduction, NaN-Handling, CHM-Engineering, Plausibilitätsfilterung. Zentraler Fokus: **Vermeidung von Data Leakage** durch ausschließlich within-tree Operationen.

**Output:** 0 NaN, quality-assured, \~187 Features

---

## Methodische Schritte

### 1. Genus-Filterung

Scope aus `chm_assessment.json` (exp\_02): \< 3 Nadelbaum-Genera oder \< 500 Samples → `deciduous_only`. Urbane Bestände >95% Laubbäume → typischerweise deciduous\_only.

### 2. Plant-Year-Filterung

Threshold aus exp\_02 (statistisch, nicht hardcoded): Letztes Jahr mit median(CHM) ≥ 2.0m. Bäume mit `plant_year > threshold` entfernen. NaN-plant\_year wird beibehalten (konservativ).

### 3. Temporal Reduction

Selektierte Monate aus `temporal_selection.json` (exp\_01, JM-basiert). Typisch 8 Monate → 23 × 8 = **184 Features** (\~33% Reduktion). Winter-Monate (geringe Variabilität) verworfen.

### 4. NaN-Analyse und Filterung

Pro Feature-Basis (z.B. "NDVI"): Baum entfernen wenn:

- \> 2 NaN-Monate total
- \> 1 konsekutive Edge-NaN (Start/Ende der Zeitreihe)
- \< 2 valide Monate

**Begründung:** ≤2 Interior-NaN → lineare Interpolation zuverlässig. ≤1 Edge-NaN → Nearest-Neighbor akzeptabel. ≥2 Edge-NaN → signifikanter phenologischer Informationsverlust.

### 5. Within-Tree Temporal Interpolation (KRITISCH)

Jeder Baum wird **isoliert** prozessiert — keine cross-tree Statistiken:

- **Interior NaN:** Lineare Interpolation zwischen benachbarten validen Werten
- **Edge NaN (max 1):** Nearest-Neighbor Fill (Forward/Backward)

**Data-Leakage-Prävention:**

- ❌ **Verboten:** Genus-/Stadt-Mittelwerte für NaN-Imputation (Legacy-Fehler)
- ✅ **Erlaubt:** Within-tree Interpolation, Schwellenwert-basierte Filter

### 6. CHM Feature Engineering

Genus×City-Normalisierung (wie in exp\_02 beschrieben):

- **CHM\_1m\_zscore:** `(CHM - μ_genus,city) / σ_genus,city`
- **CHM\_1m\_percentile:** Rang innerhalb Genus×Stadt (0–100)
- Fallback bei \< 10 Samples: City-level Normalisierung

### 7. NDVI Plausibilitätsfilter

`max_NDVI = max(NDVI über selektierte Monate)` — Baum entfernen wenn \< 0.3.

Wenn selbst im besten Monat NDVI \< 0.3 → kein gesunder Baum (Fehlklassifikation, toter Baum, Positionsfehler). Erwartete Removal: \~1–3%.

---

## Data Leakage Prevention — Zusammenfassung

| Operation | Status | Begründung |
| --- | --- | --- |
| Within-tree Temporal Interpolation | ✅ Erlaubt | Nur eigene Zeitreihe |
| Genus×Stadt CHM-Normalisierung | ✅ Erlaubt | Externe Datenquelle (LiDAR) |
| Schwellenwert-Filter (Plant Year, NDVI) | ✅ Erlaubt | Absolute Kriterien |
| Genus-/Stadt-Mittelwerte für NaN | ❌ Verboten | Train→Val/Test Leakage |
| Cross-tree Outlier-Detection | ❌ Hier nicht | Erst Phase 2c (nach Splits) |
