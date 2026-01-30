# Exploratory 03: Feature Correlation Analysis

**Notebook:** `notebooks/exploratory/exp_03_correlation_analysis.ipynb`
**Zweck:** Identifikation redundanter Features zur Dimensionsreduktion
**Status:** ⏳ In Implementierung

---

## Ziel

Identifikation hochkorrelierter Features zur Vermeidung von Multikollinearität in ML-Modellen. Redundante Features werden dokumentiert und für Phase 2c (Feature Removal) markiert, wobei Cross-City-Konsistenz gewährleistet wird.

---

## Input

**Daten:** `data/phase_2_features/trees_clean_{city}.gpkg` (Phase 2b Output)

**Charakteristika:**

- Berlin: ~750.000 Bäume
- Leipzig: ~180.000 Bäume
- Features: ~187 total
  - 3 CHM-Features (CHM_1m, CHM_1m_zscore, CHM_1m_percentile)
  - ~184 Sentinel-2 Features (23 bases × 8 Monate nach temporal selection)
- **Garantie:** 0 NaN-Werte, alle Quality Checks bestanden

---

## Methodik

### 1. Sampling-Strategie

**Sample-Größe:** 10.000 Bäume pro Stadt

**Begründung:**

- Korrelationsschätzung ist stabil ab ~5.000 Samples
- 10.000 bietet Puffer für genus-stratifizierte Ziehung
- Reduziert Rechenzeit bei ausreichender statistischer Power

**Stratifikation:** Proportional zur Genus-Verteilung

Erhält die natürliche Klassen-Balance im Sample, vermeidet Bias durch dominante Genera.

**Reproduzierbarkeit:** Random Seed = 42

---

### 2. Feature-Gruppierung

Korrelationen werden **ausschließlich innerhalb von Feature-Gruppen** analysiert. Keine Cross-Group-Vergleiche.

#### Gruppe 1: CHM-Features (3)

- CHM_1m (Absolute Höhe)
- CHM_1m_zscore (Standardisiert pro Stadt)
- CHM_1m_percentile (Rang-basiert pro Stadt)

**Erwartung:** Wahrscheinlich keine Removal – unterschiedliche Normalisierungen dienen verschiedenen Zwecken.

#### Gruppe 2: Spektrale Bänder (10 Basen)

- Visible: B2 (Blue), B3 (Green), B4 (Red)
- Red-Edge: B5, B6, B7
- NIR: B8, B8A
- SWIR: B11, B12

**Typische Redundanz:** B8 und B8A (beide NIR, unterschiedliche Bandbreiten)

#### Gruppe 3: Vegetations-Indices (13 Basen)

Untergruppen nach ökologischer Funktion:

- **Broadband:** NDVI, EVI, GNDVI, VARI
- **Red-Edge:** NDre1, NDVIre, CIre, IRECI, RTVIcore
- **Wasser:** NDWI, MSI, NDII, kNDVI

**Typische Redundanz:**

- NDVI vs. GNDVI (beide Greenness-Indices)
- NDWI vs. MSI (beide Wasser-Sensitivität)

---

### 3. Korrelations-Berechnung

**Metrik:** Pearson Correlation Coefficient (r)

**Temporal-Aggregation:**

Für Feature-Paare mit temporalen Instanzen (z.B. B8 vs. B8A):

1. **Stacking:** Alle Monate pro Feature werden zu einem langen Vektor konkateniert
   - Beispiel: B8 → [B8_03, B8_04, ..., B8_10] für alle Bäume
   - Resultierende Vektorlänge: 10.000 Bäume × 8 Monate = 80.000 Werte

2. **Single Correlation:** Ein r-Wert über alle temporalen Instanzen
   - Repräsentiert Gesamtkorrelation über Raum (Bäume) und Zeit (Monate)

**Standardisierung:** Features werden z-transformiert (Mittelwert 0, Standardabweichung 1) vor Korrelation.

**Wichtig:** Keine Korrelation zwischen verschiedenen Monaten desselben Features (z.B. NDVI_06 vs. NDVI_07). Temporale Variabilität ist phenologisch relevant und wird beibehalten.

---

### 4. Redundanz-Schwellenwert

**Threshold:** |r| > 0.95

**Literatur-Begründung:**

- Dormann et al. (2013, _Ecography_): Empfehlen |r| > 0.7–0.9 für Multikollinearitätsprobleme
- Kuhn & Johnson (2013, _Applied Predictive Modeling_): Schlagen 0.90–0.95 für hochdimensionale Daten vor
- **0.95 ist konservativ** – entfernt nur sehr hohe Redundanz

**Validierung:** Variance Inflation Factor (VIF) für verbleibende Features

- Ziel: VIF < 10 (Standard-Literaturwert)
- Prüft, ob Multikollinearität ausreichend reduziert wurde

---

### 5. Feature-Auswahl-Logik

Wenn zwei Features |r| > 0.95 zeigen, wird eines entfernt basierend auf:

#### Kriterium 1: Interpretierbarkeit

Bevorzugung von:

- Standardisierten Indices (NDVI > GNDVI)
- Breiteren Bändern (B8 > B8A, stabiler)
- Ökologisch etablierten Metriken

#### Kriterium 2: Diskriminative Power

Berechnung der mittleren Korrelation mit Target (genus_latin):

- Für jedes Feature: Mittelwert der absoluten Korrelationen mit allen Genus-Klassen
- Feature mit höherer Target-Korrelation wird bevorzugt

#### Preservation-Constraints

**Spektrale Bänder:**

- Minimum: 7 von 10 Bändern müssen erhalten bleiben
- Begründung: Spektrale Diversität für unterschiedliche Vegetations-Signaturen

**Vegetations-Indices:**

- Minimum: 1 Index pro ökologischer Familie (Broadband, Red-Edge, Wasser)
- Begründung: Erhalt verschiedener physiologischer Informationen (Chlorophyll, Wassergehalt)

**CHM-Features:**

- Erwartung: Alle 3 bleiben erhalten
- Unterschiedliche Normalisierungen haben verschiedene ML-Eigenschaften

---

### 6. Temporal-Konsistenz

**Kritische Regel:** Wenn ein Feature-Basis als redundant identifiziert wird, werden **alle temporalen Instanzen** entfernt.

**Beispiel:**

- WENN B8A mit B8 korreliert (|r| > 0.95)
- DANN entferne: B8A_03, B8A_04, B8A_05, ..., B8A_10
- NICHT: Nur einzelne Monate wie B8A_06

**Begründung:** Temporal-Inkonsistenz würde phenologische Muster fragmentieren und ML-Modelle destabilisieren.

---

### 7. Cross-City-Konsistenz

**Strikte Regel:** Ein Feature wird nur entfernt, wenn es in **BEIDEN** Städten redundant ist.

**Algorithmus:**

```
redundant_berlin = {Features mit |r| > 0.95 in Berlin}
redundant_leipzig = {Features mit |r| > 0.95 in Leipzig}

features_to_remove = redundant_berlin ∩ redundant_leipzig
```

**Begründung:**

- **Transfer-Robustheit:** Was in Berlin redundant ist, könnte in Leipzig informativ sein
- **Konservativer Ansatz:** Schützt vor Überanpassung an stadtspezifische Muster
- **Generalisierung:** Verbessert Cross-City-Transfer-Performance (Hauptziel des Projekts)

**Metrik:** Cross-City Agreement Rate

```
agreement_rate = |removed| / |redundant_berlin ∪ redundant_leipzig|
```

**Ziel:** > 0.80 (hohe Konsistenz)

**Niedrige Agreement Rate (<0.60):** Indikator für stadtspezifische Feature-Beziehungen – Features werden beibehalten, aber für Phase 3 Monitoring markiert.

---

## Visualisierungen

### Plot 1–2: Spectral Bands Correlation Heatmaps

**Typ:** Annotierte Korrelationsmatrix (Berlin, Leipzig separat)

**Elemente:**

- Hierarchical Clustering der Zeilen/Spalten (ähnliche Features gruppiert)
- Diverging Colormap (Rot-Weiß-Blau, zentriert bei r=0)
- Annotations: r-Werte in jeder Zelle
- **Hervorhebung:** Zellen mit |r| > 0.95 mit dicker Umrandung

**Interpretation:** Clustered Regions zeigen spektral ähnliche Bänder (z.B. NIR-Gruppe: B8, B8A).

---

### Plot 3–4: Vegetation Indices Correlation Heatmaps

**Typ:** Annotierte Korrelationsmatrix (Berlin, Leipzig separat)

**Struktur:** Visuelle Separation nach Index-Familie (Broadband / Red-Edge / Wasser)

**Elemente:** Identisch zu Spectral Bands Heatmaps

**Interpretation:** Innerhalb-Familie-Korrelationen zeigen funktionale Redundanz (z.B. NDVI–GNDVI in Broadband-Familie).

---

**Specifications (alle Plots):**

- DPI: 300 (Publication-ready)
- Figure Size: (12, 10) für Lesbarkeit großer Matrizen
- Font Size: 10pt für Annotations

---

## Output

### JSON-Konfiguration: `correlation_removal.json`

**Pfad:** `outputs/phase_2/metadata/correlation_removal.json`

**Zweck:** Feature-Removal-Anweisungen für Phase 2c Runner-Notebook

**Schema-Struktur:**

#### Section 1: Metadata

- Analysis date, sample size, threshold
- Literature reference für Threshold-Begründung

#### Section 2: Feature Groups

Pro Gruppe (CHM, Bands, Indices):

- Analyzed features (alle geprüften Features)
- Removed features (Basen, die entfernt werden)
- Retained features (verbleibende Basen)
- Rationale per feature (Begründung für Removal mit r-Werten)

#### Section 3: Temporal Removal

- **Kritisch:** Vollständige Liste aller temporalen Feature-Spalten
- Beispiel: `["B8A_03", "B8A_04", ..., "B8A_10"]`
- Total count (Validierung: count = removed_bases × n_months)

#### Section 4: Cross-City Consistency

- Redundant in Berlin (Liste)
- Redundant in Leipzig (Liste)
- Final removed (Intersection)
- City-specific kept (Union - Intersection)
- Agreement rate (numerisch)
- Interpretation (Text: "high/medium/low consistency")

#### Section 5: Feature Reduction Summary

- Before: Count pro Gruppe + Total
- After: Count pro Gruppe + Total
- Reduction percentage

#### Section 6: Validation

- Max VIF der verbleibenden Features
- VIF threshold (10.0)
- Check passed (boolean)

**Hinweis:** Alle numerischen Werte im JSON sind Platzhalter zur Illustration der Schema-Struktur.

**Verwendung:** Phase 2c lädt diese JSON und entfernt alle in `temporal_removal.removed_temporal_features` gelisteten Spalten.

---

## Erwartete Ergebnisse

**Typische Feature-Reduktion:** 10–15% der Sentinel-2-Features

**Häufig redundante Feature-Basen (Literatur-basiert):**

- **B8A:** Hochkorreliert mit B8 (beide NIR, B8A ist schmaleres Band)
- **GNDVI:** Hochkorreliert mit NDVI (beide Greenness, NDVI ist Standard)
- **MSI:** Hochkorreliert mit NDWI (beide Wasser-Sensitivität)
- **CIre:** Möglicherweise redundant mit anderen Red-Edge-Indices

**CHM-Features:** Wahrscheinlich keine Removal – unterschiedliche Normalisierungen (absolut / z-score / percentile) haben verschiedene ML-Eigenschaften.

**Cross-City Agreement:** Typisch 0.75–0.90 (hohe Konsistenz zwischen Berlin und Leipzig für spektrale Redundanzen)

---

## Validierung

**In-Notebook Checks:**

- ✅ Korrelationsmatrizen symmetrisch (r_AB = r_BA)
- ✅ Keine NaN-Werte in Korrelationen
- ✅ Keine temporalen Cross-Korrelationen (NDVI_06 vs. NDVI_07 nicht analysiert)
- ✅ Preservation-Constraints erfüllt:
  - Mindestens 7 von 10 Spektral-Bändern erhalten
  - Mindestens 1 Index pro Familie erhalten
- ✅ Cross-City Agreement Rate > 0.75
- ✅ VIF < 10 für alle verbleibenden Features
- ✅ Temporal Removal List vollständig:
  - Length = removed_bases × n_selected_months
  - Alle Suffixe (Monate) für jeden removed base vorhanden

**JSON-Schema Validation:**

- Alle erforderlichen Keys vorhanden
- Temporal removal list matches detected months
- Rationale für jedes entfernte Feature dokumentiert

---

## Nächste Schritte

**Nach Abschluss exp_03:**

1. **Manual Sync:**
   - Download `correlation_removal.json` von Google Drive
   - Commit zu Git: `outputs/phase_2/metadata/correlation_removal.json`
   - Download Plots (optional für Dokumentation)
   - Push zu GitHub

2. **Verbleibende Exploratory Notebooks:**
   - `exp_04_outlier_thresholds.ipynb` – Outlier-Detection-Parameter validieren
   - `exp_05_spatial_autocorrelation.ipynb` – Block-Size für Spatial Splits bestimmen

3. **Phase 2c Vorbereitung:**
   - Nach Abschluss aller Exploratory Notebooks (03–05)
   - Implementierung des Final Preparation Runners
   - Verwendet alle drei JSON-Configs (correlation, outlier, spatial)

---

**Dokumentations-Status:** ✅ Exploratory 03 (Correlation Analysis) vollständig dokumentiert
