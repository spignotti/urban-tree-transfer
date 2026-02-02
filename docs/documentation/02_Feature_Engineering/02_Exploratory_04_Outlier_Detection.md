# Phase 2: Exploratory Analysis - Outlier Detection Threshold Validation

**Phase:** Feature Engineering - Exploratory Analysis
**Last Updated:** 2026-02-01
**Status:** 🔄 In Implementation

---

## Überblick

Diese explorative Analyse validiert die Schwellenwerte und Entscheidungsregeln für die multivariate Outlier-Detektion. Der methodische Fokus liegt auf **statistischer Begründung der Threshold-Wahl** und der Etablierung eines **konservativen Flagging-Systems** für spätere Ablationsstudien.

**Input:** `data/phase_2_features/trees_clean_{city}.gpkg` (Phase 2b, quality-assured)

**Output:** `outlier_thresholds.json` (Parameter-Konfiguration + Flagging-Statistiken)

**Zentrale Methodische Entscheidung:**
- ❌ **Keine automatische Removal** von Outliers in dieser Phase
- ✅ **Severity-basiertes Flagging** (high/medium/low/none) für Ablationsstudien in Phase 3

---

## Methodischer Ansatz: Tripartite Outlier Detection

Die Outlier-Detektion verwendet **drei komplementäre Methoden**, die unterschiedliche Arten von Anomalien erfassen:

| Methode            | Typ             | Erfasst                                    | Blind für                              |
| ------------------ | --------------- | ------------------------------------------ | -------------------------------------- |
| **Z-Score**        | Univariat       | Spektrale Extreme in einzelnen Features    | Ungewöhnliche Feature-Kombinationen    |
| **Mahalanobis**    | Multivariat     | Anomale Feature-Kombinationen (Korrelation)| Strukturelle Fehler, Schiefe Verteilungen |
| **IQR (Tukey)**    | Univariat       | Strukturelle Höhen-Anomalien (CHM)         | Spektrale Sensorfehler                 |

**Rationale für Tripartite Ansatz:**
- Einzelne Methode kann Bias haben (z.B. Z-Score bei schiefen Verteilungen)
- Consensus mehrerer Methoden → höhere Confidence für echte Outliers
- Verschiedene Datentypen (spektral vs. strukturell) erfordern verschiedene Ansätze

---

## Methode 1: Z-Score (Univariate Spektrale Outliers)

### Methodische Grundlage

**Zweck:** Identifikation von Bäumen mit vielen spektral extremen Features.

**Formel:**
```
z = (x - μ) / σ

wobei:
x = Feature-Wert eines Baums
μ = Mittelwert des Features über alle Bäume
σ = Standardabweichung des Features
```

**Threshold-Logik:**
```
Für jeden Baum:
    count_extreme = Σ [|z_i| > threshold]  über alle S2-Features i

    WENN count_extreme ≥ min_feature_count:
        → outlier_zscore = True
```

### Threshold-Validierung

**Parameter zu testen:**
- **Threshold:** [2.5, 3.0, 3.5] (σ-Vielfache)
- **Min Feature Count:** [5, 10, 15] (Anzahl extreme Features)

**Standard-Wahl (basierend auf statistischen Konventionen):**
- **Threshold = 3.0:** Standard 3-Sigma-Regel (99.7% Abdeckung bei Normalverteilung)
- **Min Feature Count = 10:** Bei ~180 S2-Features bedeutet 10 Extreme ≈5% aller Features

**Begründung:**
- 1-2 extreme Features pro Baum sind normal (statistische Fluktuationen)
- 10+ extreme Features → systematische Anomalie (Sensor-Fehler, Cloud-Artefakt)
- Bei Normalverteilung: P(|z| > 3) ≈ 0.3% → 10 unabhängige Features gleichzeitig extrem: P < 10^-20

**Erwartete Flagging-Rate:** ~3-7% der Bäume (sensitivste Methode)

---

## Methode 2: Mahalanobis Distance (Multivariate Spektrale Outliers)

### Methodische Grundlage

**Zweck:** Identifikation von Bäumen mit anomalen Feature-Kombinationen unter Berücksichtigung von Korrelationen.

**Formel:**
```
D² = (x - μ)ᵀ Σ⁻¹ (x - μ)

wobei:
x = Feature-Vektor eines Baums
μ = Mittelwert-Vektor (genus-spezifisch)
Σ = Kovarianz-Matrix (genus-spezifisch)
```

**Threshold-Logik:**
```
Für jeden Baum mit Genus g:
    D² = Mahalanobis-Distanz zur Genus-g-Verteilung

    χ²_critical = χ²(p, α)  mit p = Anzahl Features, α = Signifikanzniveau

    WENN D² > χ²_critical:
        → outlier_mahalanobis = True
```

### Threshold-Validierung

**Parameter zu testen:**
- **α (Signifikanzniveau):** [0.0001, 0.001, 0.01]
- Entsprechende χ²-Kritische-Werte bei p=180 Features:
  - α=0.0001: χ² ≈ 248
  - α=0.001:  χ² ≈ 230
  - α=0.01:   χ² ≈ 210

**Standard-Wahl (basierend auf statistischen Konventionen):**
- **α = 0.001:** Übliches Signifikanzniveau für multivariate Outlier-Detection
- **Genus-spezifisch:** Kovarianz-Matrix pro Genus (nicht global)

**Begründung α-Wahl:**
- α=0.0001: Sehr konservativ (nur ultra-extreme Outliers)
- α=0.001: Balance zwischen False Positives und Sensitivity
- α=0.01: Zu liberal (würde normale Variabilität als Outliers flaggen)

**Genus-Spezifität:**
- QUERCUS vs. TILIA haben unterschiedliche spektrale Signaturen
- Globale Kovarianz-Matrix würde genus-spezifische Variabilität als Outliers fehlinterpretieren
- Pro-Genus-Analyse: Outliers sind Anomalien **innerhalb** ihrer Genus-Gruppe

**Erwartete Flagging-Rate:** ~0.5-2% der Bäume (konservativste Methode)

---

## Methode 3: IQR (Strukturelle CHM-Outliers)

### Methodische Grundlage

**Zweck:** Identifikation von Bäumen mit implausiblen Höhen-Werten (CHM-Fehler).

**Formel (Tukey's Fences):**
```
IQR = Q3 - Q1

Lower Fence = Q1 - k × IQR
Upper Fence = Q3 + k × IQR

wobei:
Q1 = 25. Perzentil
Q3 = 75. Perzentil
k = Multiplikator (Standard: 1.5)
```

**Threshold-Logik:**
```
Für jeden Baum mit Genus g in Stadt s:
    CHM_value = CHM_1m

    WENN (CHM_value < Lower_Fence_g,s) ODER (CHM_value > Upper_Fence_g,s):
        → outlier_iqr = True
```

### Threshold-Validierung

**Parameter zu testen:**
- **k (Multiplier):** [1.5, 2.0, 3.0]
- Standard-Werte aus Literatur:
  - k=1.5: Standard (Tukey 1977, Box-Plot-Definition)
  - k=2.0: Moderat konservativ
  - k=3.0: Sehr konservativ

**Standard-Wahl:**
- **k = 1.5:** Weitverbreiteter Standard für Outlier-Definition

**Begründung Genus×Stadt-Spezifität:**
- Höhenbereiche variieren dramatisch zwischen Genera:
  - QUERCUS: 5-30m (große Bäume)
  - MALUS: 3-10m (kleine Bäume)
- Stadtspezifische Unterschiede (z.B. Baumalter-Verteilung):
  - Berlin: Durchschnittlich ältere/höhere Bäume
  - Leipzig: Jüngere Bestände

**Erwartete Flagging-Rate:** ~5-8% der Bäume (genus/stadt-abhängig)

---

## Consensus-Based Flagging Strategy

### Severity-Level Definition

Statt sofortiger Removal werden Bäume nach **Consensus-Level** kategorisiert:

```
Severity = Anzahl Methoden, die Baum als Outlier flaggen

WENN Severity == 3:  → outlier_severity = "high"    (alle 3 Methoden)
WENN Severity == 2:  → outlier_severity = "medium"  (2 von 3 Methoden)
WENN Severity == 1:  → outlier_severity = "low"     (1 von 3 Methoden)
WENN Severity == 0:  → outlier_severity = "none"    (keine Methode)
```

**Zusätzlich:** Methoden-spezifische Boolean-Flags für detaillierte Analysen:
- `outlier_zscore` (boolean)
- `outlier_mahalanobis` (boolean)
- `outlier_iqr` (boolean)

### Rationale für Flagging (statt Removal)

**Warum keine automatische Entfernung?**

1. **Ablationsstudien ermöglichen:**
   - Testen ob "high"-Removal Performance verbessert
   - Testen ob auch "medium" entfernt werden sollte
   - Vergleich verschiedener Removal-Strategien

2. **Konservative Validierung:**
   - Nicht alle "high"-Outliers sind notwendigerweise Fehler
   - Könnten biologisch extreme, aber valide Bäume sein (z.B. sehr alte QUERCUS)

3. **Methodische Transparenz:**
   - Dokumentiert welche Methode welchen Baum flaggt
   - Ermöglicht Post-hoc-Analyse der Method-Overlap

**Erwartete Consensus-Verteilung:**
- **None:** ~92% (Mehrheit der Bäume)
- **Low:** ~5% (von einzelner Methode geflaggt)
- **Medium:** ~2% (von 2 Methoden geflaggt)
- **High:** ~1% (von allen 3 Methoden geflaggt → höchste Confidence)

### Method Overlap Analysis

**Venn-Diagramm-Interpretation:**

```
         Z-Score
            ◯
           /|\
          / | \
         /  |  \
    Mahal ◯-+-◯ IQR
           \|/
            ◯
         (Overlap)
```

**Erwartete Overlap-Muster:**
- **Große Solo-Bereiche:** Jede Methode fängt spezifische Anomalie-Typen
- **Kleiner 3-fach-Overlap:** Nur ultra-extreme Outliers werden von allen geflaggt
- **Moderater 2-fach-Overlap:** Z-Score + Mahalanobis (spektrale Anomalien), IQR Solo (strukturelle Anomalien)

**Validierung:**
- WENN 3-fach-Overlap sehr groß (>5%): Threshold zu liberal
- WENN Solo-Bereiche zu groß (>10% pro Methode): Methoden zu unkorreliert (eventuell False Positives)

---

## Genus-Specific Impact Validation

### Zweck

Sicherstellen, dass Outlier-Flagging **genus-uniform** ist und nicht bestimmte Genera überproportional trifft.

### Validierungs-Metriken

**Pro Genus:**
```
Flagging Rate = (Anzahl geflaggter Bäume / Gesamt-Anzahl Bäume) × 100%
```

**Uniformitäts-Check:**
```
Max Deviation = max(|Flagging_Rate_genus - Mean_Flagging_Rate|)

WENN Max Deviation > 2 × Mean_Flagging_Rate:
    → WARNUNG: Genus X überproportional betroffen
```

**Erwartetes Ergebnis:**
- Alle Genera sollten Flagging-Rates innerhalb ±50% des Mittelwerts haben
- Beispiel: Mean = 3% → Genus-Raten sollten 1.5%-4.5% sein

**Falls Genus disproportional betroffen:**
- Hinweis auf genus-spezifische Charakteristik (nicht Fehler)
- Eventuell genus-spezifische Threshold-Anpassung nötig
- Oder: Bestätigung dass Genus valide extremer ist (z.B. QUERCUS größere Höhen-Variabilität)

---

## Sensitivity Analysis

### Z-Score Threshold

**Test-Matrix:**
| Threshold | Min Feature Count | Erwartete Rate | Trade-off                     |
| --------- | ----------------- | -------------- | ----------------------------- |
| 2.5       | 5                 | ~8-12%         | Sensitiv, aber mehr FP        |
| 3.0       | 10                | ~3-5%          | **Balanced** (Empfohlen)      |
| 3.5       | 15                | ~1-2%          | Sehr konservativ, weniger FN  |

**Interpretation:**
- **Niedrigerer Threshold:** Mehr Outliers gefunden, aber höheres False-Positive-Risiko
- **Höherer Min Count:** Reduziert False Positives (Zufall), aber könnte echte Outliers verpassen

### Mahalanobis α-Level

**Test-Matrix:**
| α       | χ²(180) approx | Erwartete Rate | Trade-off                  |
| ------- | -------------- | -------------- | -------------------------- |
| 0.0001  | 248            | ~0.2%          | Ultra-konservativ          |
| 0.001   | 230            | ~0.5-1%        | **Standard** (Empfohlen)   |
| 0.01    | 210            | ~2-3%          | Liberal, viele Flaggings   |

**Interpretation:**
- α bestimmt False-Positive-Rate direkt
- α=0.001 balanciert Sensitivity und Specificity optimal

### IQR Multiplier

**Test-Matrix:**
| k   | Expected at Normal | Erwartete Rate | Trade-off                |
| --- | ------------------ | -------------- | ------------------------ |
| 1.5 | ~0.7% per tail     | ~5-8%          | **Standard** (Empfohlen) |
| 2.0 | ~0.3% per tail     | ~2-4%          | Moderat konservativ      |
| 3.0 | ~0.1% per tail     | ~1-2%          | Sehr konservativ         |

**Interpretation:**
- k=1.5 ist etablierter Standard aus Literatur
- CHM-Verteilungen oft nicht perfekt normal (rechtsschiefe), daher k=1.5 angemessen

---

## Output: `outlier_thresholds.json`

### JSON-Struktur

```json
{
  "zscore": {
    "threshold": 3.0,
    "min_feature_count": 10,
    "justification": "Standard 3-sigma rule for normal distributions"
  },
  "mahalanobis": {
    "alpha": 0.001,
    "justification": "Common practice for multivariate outliers"
  },
  "iqr": {
    "multiplier": 1.5,
    "justification": "Standard Tukey box plot threshold"
  },
  "flagging_strategy": {
    "columns_added": [
      "outlier_zscore",
      "outlier_mahalanobis",
      "outlier_iqr",
      "outlier_severity"
    ],
    "severity_levels": {
      "high": "flagged by all 3 methods",
      "medium": "flagged by 2 methods",
      "low": "flagged by 1 method",
      "none": "flagged by 0 methods"
    },
    "removal_strategy": "none - all trees retained for ablation studies"
  },
  "method_overlap": {
    "zscore_only": 245,
    "mahalanobis_only": 89,
    "iqr_only": 312,
    "zscore_and_mahalanobis": 156,
    "zscore_and_iqr": 98,
    "mahalanobis_and_iqr": 72,
    "all_three": 67
  },
  "flagging_rates": {
    "high": 0.008,
    "medium": 0.024,
    "low": 0.052,
    "none": 0.916
  },
  "per_city_statistics": {
    "berlin": { /* Stadt-spezifische Counts */ },
    "leipzig": { /* Stadt-spezifische Counts */ }
  },
  "validation": {
    "genus_impact": "uniform across genera - no genus disproportionately affected",
    "false_positive_estimate": "< 0.3% for high severity",
    "sensitivity_analysis": "threshold variations tested - 3.0/0.001/1.5 optimal",
    "ablation_study_ready": true
  },
  "biological_context_analysis": {
    "age_correlation": {
      "high_severity_median_plant_year": 1985,
      "non_outlier_median_plant_year": 2005,
      "mann_whitney_u": 12345.0,
      "p_value": 0.001,
      "interpretation": "High-severity outliers significantly older (likely biological extremes)"
    },
    "tree_type_association": {
      "high_severity_anlagenbaeume_pct": 68.0,
      "high_severity_strassenbaeume_pct": 32.0,
      "non_outlier_anlagenbaeume_pct": 45.0,
      "interpretation": "High-severity enriched in park trees (anlagenbaeume)"
    },
    "spatial_clustering": {
      "ripleys_k_50m": 1.8,
      "ripleys_k_100m": 2.1,
      "ripleys_k_200m": 1.9,
      "expected_under_csr": 1.0,
      "interpretation": "High-severity outliers spatially clustered (parks, monuments)"
    },
    "genus_patterns": {
      "high_severity_rates_per_genus": {
        "QUERCUS": 0.12,
        "PLATANUS": 0.10,
        "TILIA": 0.08
      },
      "interpretation": "Large genera show higher biological extreme rates"
    },
    "recommendation": "High-severity outliers likely include biological extremes (old park trees). Consider separate flag for Phase 3 ablation studies."
  }
}
```

### Verwendung in Phase 2c

**Nicht verwendet für automatische Removal**, sondern als **Metadata-Flags**:
- Flags werden zu Datensatz hinzugefügt
- Phase 2c: Weitere Feature-Reduktion, Spatial Splits
- Phase 3: Ablationsstudien testen Removal-Strategien

**Mögliche Ablations-Szenarien:**
1. **Baseline:** Alle Bäume (inkl. Outliers)
2. **Conservative:** Nur "high"-Severity entfernen
3. **Moderate:** "high" + "medium" entfernen
4. **Aggressive:** Alle geflaggten entfernen ("high" + "medium" + "low")

Performance-Vergleich zeigt optimale Removal-Strategie.

---

## Visualisierungen (11 Plots: 7 Core + 4 Biological Context)

### 1. Outlier Rates per Method
- **Typ:** Bar chart
- **X-Achse:** Methode (Z-Score, Mahalanobis, IQR)
- **Y-Achse:** Flagging Rate (%)
- **Facetten:** Pro Stadt (berlin, leipzig)
- **Interpretation:** Vergleich der Method Sensitivity

### 2. Venn Diagram (Method Overlap)
- **Typ:** 3-Circle Venn
- **Kreise:** Z-Score, Mahalanobis, IQR
- **Annotations:** Tree counts in jedem Segment
- **Interpretation:** Visualisiert Consensus-Levels

### 3. Z-Score Sensitivity
- **Typ:** Line plot
- **X-Achse:** Threshold (2.5, 3.0, 3.5)
- **Y-Achse:** Flagging Rate (%)
- **Linien:** Verschiedene min_feature_counts (5, 10, 15)
- **Interpretation:** Trade-off zwischen Threshold und Count

### 4. Mahalanobis Distribution
- **Typ:** Histogram + Chi² Line
- **X-Achse:** Mahalanobis D²
- **Y-Achse:** Frequency
- **Facetten:** Pro Genus
- **Overlay:** Chi²-Kritischer-Wert-Linien für α=[0.0001, 0.001, 0.01]
- **Interpretation:** Verteilungs-Fit und Threshold-Platzierung

### 5. IQR Box Plots per Genus
- **Typ:** Box plot
- **X-Achse:** Genus
- **Y-Achse:** CHM_1m (m)
- **Facetten:** Pro Stadt
- **Overlay:** Tukey Fences (k=1.5)
- **Interpretation:** Genus- und Stadt-spezifische Höhenbereiche

### 6. Severity Distribution
- **Typ:** Bar chart
- **X-Achse:** Severity Level (none, low, medium, high)
- **Y-Achse:** Count
- **Facetten:** Gesamt + pro Stadt
- **Interpretation:** Consensus-Level-Verteilung

### 7. Outlier by Genus
- **Typ:** Stacked bar chart
- **X-Achse:** Genus
- **Y-Achse:** Count
- **Stacks:** Severity levels (none/low/medium/high)
- **Interpretation:** Genus-Uniformität validieren

### 8. Biological Context: Age Distribution

**Added: 2026-02-01 (PRD 002d Improvement 4)**

- **Typ:** Histogram comparison (overlaid or faceted)
- **X-Achse:** Plant Year
- **Y-Achse:** Frequency
- **Groups:** High-severity outliers vs. non-outliers
- **Statistics:** Mann-Whitney U test results (median plant year comparison)
- **Interpretation:** If high-severity outliers are significantly older, suggests biological extremes (old park trees)

### 9. Biological Context: Tree Type Association

**Added: 2026-02-01 (PRD 002d Improvement 4)**

- **Typ:** Stacked or grouped bar chart
- **X-Achse:** Outlier severity (high, medium, low, none)
- **Y-Achse:** Percentage or count
- **Groups:** Tree type (anlagenbaeume vs. strassenbaeume)
- **Interpretation:** Enrichment of high-severity in park trees (anlagenbaeume) indicates biological contexts

### 10. Biological Context: Spatial Clustering

**Added: 2026-02-01 (PRD 002d Improvement 4)**

- **Typ:** Spatial map overlay
- **Background:** All trees (light gray, small points)
- **Foreground:** High-severity outliers (red stars, prominent)
- **Annotation:** Ripley's K statistics at 50/100/200m
- **Interpretation:** Spatial clustering (K ratio > 1.2) indicates parks/monuments as biological contexts

### 11. Biological Context: Genus-Specific Rates

**Added: 2026-02-01 (PRD 002d Improvement 4)**

- **Typ:** Bar chart
- **X-Achse:** Genus (filtered for minimum sample size)
- **Y-Achse:** High-severity outlier rate (%)
- **Interpretation:** Genera with high natural variability (QUERCUS, PLATANUS) may have higher biological extreme rates

---

## Output Files

**Metadata:**
```
outputs/phase_2/metadata/outlier_thresholds.json
```

**Visualizations (11 plots):**
```
outputs/phase_2/figures/exp_04/
├── outlier_rates_per_method.png          # Plot 1: Core analysis
├── method_overlap_venn.png                # Plot 2: Core analysis
├── zscore_sensitivity.png                 # Plot 3: Core analysis
├── mahalanobis_distribution.png           # Plot 4: Core analysis
├── iqr_boxplots_per_genus.png            # Plot 5: Core analysis
├── severity_distribution.png              # Plot 6: Core analysis
├── outlier_by_genus.png                   # Plot 7: Core analysis
├── outlier_age_distribution.png           # Plot 8: Biological context (PRD 002d)
├── outlier_tree_type.png                  # Plot 9: Biological context (PRD 002d)
├── outlier_spatial_clustering.png         # Plot 10: Biological context (PRD 002d)
└── outlier_genus_rates.png                # Plot 11: Biological context (PRD 002d)
```

**DPI:** 300 (Publication-ready)

---

## Methodische Verbesserungen gegenüber Legacy

| Aspekt                  | Legacy Pipeline                 | Aktuelle Pipeline                              |
| ----------------------- | ------------------------------- | ---------------------------------------------- |
| **Outlier-Entscheidung**| Hierarchisch (CRITICAL = remove)| Flagging-only (Ablation in Phase 3)           |
| **Threshold-Wahl**      | Trial-and-error                 | Sensitivity analysis + Literatur-Standards     |
| **Method-Overlap**      | Nicht analysiert                | Venn-Diagramm + Overlap-Statistiken            |
| **Genus-Validierung**   | Nicht durchgeführt              | Genus-wise uniformity check                    |
| **Dokumentation**       | Ergebnisse only                 | Methodischer Prozess + Justification           |
| **Ablationsstudien**    | Nicht möglich                   | Severity-Flags ermöglichen systematische Tests |

**Zentrale Verbesserung:** Methodisch fundierte, transparente Threshold-Wahl statt ad-hoc-Entscheidungen.

---

## Biological Context Analysis

**Added:** 2026-02-01 (PRD 002d Improvement 4)

**Purpose:** Distinguish data quality issues from biological extremes (analysis-only; flags unchanged).

**Analyses:**
1. **Age Correlation**: Mann-Whitney U test comparing plant_year of high-severity vs. non-outliers
2. **Tree Type Association**: Row-normalized contingency table for anlagenbaeume vs. strassenbaeume
3. **Spatial Clustering**: Ripley's K at 50/100/200 m for high-severity outliers
4. **Genus Patterns**: High-severity rates per genus (minimum sample size filter)

**Interpretation Guidance:**
- If high-severity outliers are older and enriched in park trees, they may represent biological extremes.
- If spatial clustering is strong (Ripley's K ratio > 1.2 at 100 m), parks/monuments are likely drivers.

**Phase 3 Recommendation (conditional):**
- Consider a derived flag like `is_biological_extreme` for ablation studies if age/tree type analyses support it.

---

## Nächste Schritte

**Nach Abschluss dieser Analyse:**

1. **outlier_thresholds.json committen:**
   - Download von Google Drive
   - Commit zu `outputs/phase_2/metadata/`

2. **Flags zu Datensatz hinzufügen (Optional in Phase 2c):**
   - Kann bereits in Phase 2c geschehen
   - Oder erst in Phase 3 bei Bedarf

3. **Weiter zu exp_05_spatial_autocorrelation.ipynb:**
   - Bestimmung optimaler Block-Size für Spatial Splits

4. **Phase 3 - Ablationsstudien:**
   - Test verschiedener Removal-Strategien
   - Performance-Metriken mit/ohne Outliers

---

**Dokumentations-Status:** ✅ Methodischer Prozess vollständig dokumentiert
