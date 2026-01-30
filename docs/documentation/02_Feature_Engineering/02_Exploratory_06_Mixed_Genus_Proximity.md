# Exploratory 06: Mixed-Genus Proximity Analysis

**Notebook:** `notebooks/exploratory/exp_06_mixed_genus_proximity.ipynb`
**Zweck:** Bestimmung optimaler Proximity-Filter-Schwelle für spektrale Reinheit
**Status:** ⏳ In Implementierung

---

## Ziel

Identifikation des optimalen Mindestabstands zwischen Bäumen verschiedener Gattungen, um spektrale Reinheit in Sentinel-2 10m-Pixeln zu gewährleisten. Bäume, die zu nahe an anderen Gattungen stehen, könnten durch Pixel-Mixing kontaminierte Spektralsignaturen aufweisen.

**Kernfrage:** Welcher Proximity-Threshold balanciert optimalen spektrale Reinheit mit akzeptablem Datenverlust?

---

## Input

**Daten:** `data/phase_2_features/trees_clean_{city}.gpkg` (Phase 2b Output)

**Charakteristika:**

- Berlin: ~45.000 Bäume (nach Quality-Filters)
- Leipzig: ~35.000 Bäume
- CRS: EPSG:25833 (projected, Meter)
- **Wichtig:** Outliers noch NICHT geflaggt (aber irrelevant für Proximity-Analyse)

**Warum Outliers irrelevant?**

- Proximity basiert auf **geometry** + **genus_latin**
- Outlier-Flags betreffen **Feature-Werte**, nicht Position/Taxonomie
- Filter ist **spatial/taxonomisch**, nicht spektral

---

## Methodischer Ansatz

### Problem: Sentinel-2 Pixel Mixing

**Sentinel-2 Charakteristika:**

- **Spatial Resolution:** 10m × 10m (Bands 2, 3, 4, 8)
- **Ground Coverage:** Ein Pixel kann mehrere Bäume enthalten
- **Spectral Mixing:** Signal = gewichtete Summe aller Objekte im Pixel

**Contamination Scenario:**

```
┌─────────────────┐
│  10m × 10m      │  Pixel enthält:
│                 │  - 70% QUERCUS
│  🌳 QUERCUS     │  - 30% TILIA (5m entfernt)
│      + 🌳 TILIA │
│                 │  → Spektralsignatur: gemischt
└─────────────────┘  → ML lernt "verschmiertes" QUERCUS-Signal
```

**Implikationen für Transfer Learning:**

- **Training auf Mixed Pixels:** Modell lernt genus-übergreifende Durchschnitts-Signaturen
- **Risiko:** Schlechtere genus-spezifische Diskriminierung
- **Trade-off:** Zu starkes Filtern → Overfitting auf "ideale" Reinheits-Szenarien

---

## Nearest Different-Genus Distance

### Methodik

**Für jeden Baum:**

```
1. Identifiziere alle Bäume ANDERER Gattung (derselben Stadt)
2. Berechne Distanz zu jedem dieser Bäume
3. Speichere Minimum: nearest_diff_genus_m
```

**Spatial Operations:**

- Verwende GeoPandas `distance()` für projected CRS (Meter)
- Per-city Analyse (keine Cross-City-Distanzen)
- Complexity: O(n²) im worst case → eventuell Spatial Indexing für Performance

**Edge Cases:**

- Bäume mit nur einer Gattung in Umgebung (nearest_diff_genus_m = sehr groß)
- Isolierte Bäume (Parks, Suburbs)
- City Boundaries (Nearest könnte außerhalb sein → ignoriert)

---

## Threshold Sensitivity Analysis

### Tested Thresholds

**Range:** [5m, 10m, 15m, 20m, 30m]

**Rationale für Range:**

- **5m:** < 1 Sentinel-2 Pixel → minimaler Effekt
- **10m:** = 1 Pixel Puffer → moderate Kontaminations-Reduktion
- **15m:** = 1.5 Pixel → starke Reduktion
- **20m:** = 2 Pixel Puffer → sehr starke Reduktion
- **30m:** = 3 Pixel → sehr konservativ

### Metriken pro Threshold

**Pro Stadt und Threshold:**

**1. Absolute Counts:**

```
trees_affected = Anzahl Bäume mit nearest_diff_genus_m < threshold
trees_retained = total_trees - trees_affected
```

**2. Retention Rate:**

```
retention_rate = trees_retained / total_trees
```

**Akzeptanz-Kriterium:** retention_rate > 0.85 (Target aus PRD)

**3. Genus-Specific Impact:**

```
removal_rate_per_genus = trees_affected_genus / total_trees_genus
```

**Uniformitäts-Check:**

```
max_deviation = max(|removal_rate_genus - mean_removal_rate|)

WENN max_deviation < 0.10:
    → genus_uniform = True
SONST:
    → genus_uniform = False
    → WARNUNG: Genus-spezifischer Bias
```

---

## Genus-Specific Impact Validation

### Zweck

Sicherstellen, dass Proximity-Filter **genus-neutral** ist und nicht bestimmte Gattungen überproportional trifft.

### Hypothesen

**H1: Uniform Impact:**

- Alle Genera haben ähnliche Removal-Rates (±5%)
- Indikator: Keine biologischen/ökologischen Biases

**H2: Genus-Specific Impact:**

- Bestimmte Genera stärker betroffen (>10% Abweichung)
- Mögliche Ursachen:
  - **Ökologisch:** TILIA oft in Mischbeständen (Parks)
  - **Urban Planning:** ACER oft als Straßenbaum (isoliert)
  - **Bias:** Filter bevorzugt solitäre Genera

### Validation

**Metrik:**

```
deviation_per_genus = |removal_rate_genus - mean_removal_rate|

IF all(deviation < 0.10):
    → PASS: Genus-uniform
ELSE:
    affected_genera = [g where deviation > 0.10]
    → WARNING: Genus {affected_genera} disproportionally affected
```

**Interpretation:**

- **Pass:** Filter ist genus-neutral → safe für Training
- **Warning:** Filter könnte Klassen-Imbalance erzeugen → **Option B: Defer to Phase 3**

---

## Spatial Distribution Analysis

### Zweck

Visualisieren, **wo** betroffene Bäume liegen: Dense urban areas vs. Parks?

### Erwartete Patterns

**Dense Urban (Innenstadt):**

- Hohe Baumdichte
- Viele verschiedene Genera auf kleinem Raum
- **Erwartung:** Höhere Removal-Rate

**Parks/Grünanlagen:**

- Homogene Bestände (oft mono-genus oder wenige Genera)
- Größere Abstände
- **Erwartung:** Niedrigere Removal-Rate

**Suburbs:**

- Mittlere Dichte
- Mischbestände (Alleen mit wechselnden Genera)
- **Erwartung:** Moderate Removal-Rate

### Implikationen

**Falls strong spatial bias:**

- Training-Datensatz könnte Parks über-repräsentieren
- Model könnte auf "ideale" Parks overfitting (nicht urban scenarios)
- **Mitigation:** Spatial Blocks + Stratifikation sollten Bias minimieren

---

## Theoretical Pixel Contamination

### Sentinel-2 Pixel Geometry

**10m Pixel:**

- Kreisförmiger Footprint mit ~10m Durchmesser (simplified)
- Bäume < 5m Abstand: **Definitiv im selben Pixel**
- Bäume 5-10m Abstand: **Wahrscheinlich im selben Pixel** (abhängig von Positionen)
- Bäume > 10m Abstand: **Wahrscheinlich in verschiedenen Pixeln**

### Contamination Probability

**Vereinfachtes Modell:**

```
P(contamination) ≈ 1 - (distance / pixel_size)²  für distance < pixel_size

wobei pixel_size = 10m
```

**Beispiele:**

- **5m Abstand:** P ≈ 0.75 (75% Kontamination wahrscheinlich)
- **10m Abstand:** P ≈ 0.00 (Pixel-Grenze)
- **20m Abstand:** P ≈ 0.00 (klar getrennt)

**Threshold-Interpretationen:**

- **5m:** Reduziert extreme Mixing (innerhalb desselben Pixels)
- **10m:** Eliminiert Same-Pixel-Mixing
- **20m:** Garantiert 1-Pixel-Puffer (benachbarte Pixel auch rein)

### Literature Context

**Spectral Mixing Models:**

- Linear Mixing: Pixel-Signal = Σ (fraction_i × endmember_i)
- Urban Vegetation: Typisch 30-70% within-pixel purity
- Transfer Learning: **Robustheit wichtiger als Reinheit?**

**Relevante Studien:**

- Powell et al. (2007, _RSE_): Mixed pixels in urban areas common
- Kempeneers et al. (2013, _IEEE JSTARS_): Sub-pixel classification methods
- **Implication:** Real-world scenarios HABEN Mixed Pixels → Model muss robust sein

---

## Recommendation Logic

### Decision Criteria

**Kriterium 1: Retention Rate**

```
MUST: retention_rate > 0.85
```

Zu viel Datenverlust gefährdet Sample Size für Splits.

**Kriterium 2: Genus Uniformity**

```
MUST: max_genus_deviation < 0.10
```

Filter darf keine Klassen-Imbalance erzeugen.

**Kriterium 3: Pixel Coverage**

```
PREFER: threshold >= 2 × pixel_size = 20m
```

2-Pixel-Puffer bietet starke Kontaminations-Reduktion.

**Kriterium 4: Scientific Rigor**

```
PREFER: Threshold mit Literatur-Begründung
```

Standard-Werte aus Remote Sensing Literatur bevorzugen.

---

## Visualisierungen

### Plot 1: Distance Distribution Histogram

**Typ:** Histogram (faceted per city)

**X-Achse:** Nearest Different-Genus Distance (m)

**Y-Achse:** Frequency (count)

**Annotations:**

- Vertical Lines für Tested Thresholds [5, 10, 15, 20, 30]
- Percentile Labels (10%, 25%, 50%, 75%, 90%)

**Interpretation:**

- Left-skewed Distribution: Viele Bäume nah an anderen Genera (urban density)
- Long Tail: Einige isolierte Bäume (Parks, Suburbs)

---

### Plot 2: Retention Rate Sensitivity

**Typ:** Line Plot

**X-Achse:** Threshold (m)

**Y-Achse:** Retention Rate (%)

**Linien:** Pro Stadt (berlin, leipzig)

**Annotations:**

- Horizontal Line: 85% Target
- Recommended Threshold (vertikal, fett)

**Interpretation:** Trade-off zwischen Reinheit (niedriger Threshold) und Sample Size (hoher Threshold).

---

### Plot 3: Genus-Specific Removal Rate

**Typ:** Grouped Bar Chart (faceted per city)

**X-Achse:** Genus

**Y-Achse:** Removal Rate (%) bei 20m Threshold

**Bars:** Pro Genus

**Annotations:**

- Horizontal Line: Mean Removal Rate
- Error bars (wenn applicable)

**Interpretation:** Visualisiert Genus-Uniformität – alle Bars sollten nahe Mean sein.

---

### Plot 4–5: Spatial Distribution Maps

**Typ:** Spatial Point Map (per city)

**Elemente:**

- All Trees: Gray, small
- Affected Trees (< 20m): Red, larger
- City Boundary: Black outline

**Purpose:** Zeigt räumliche Patterns – sind betroffene Bäume geclustert (Innenstadt) oder uniformiert?

---

### Plot 6: Pixel Contamination Schematic

**Typ:** Conceptual Diagram

**Elemente:**

- 10m × 10m Pixel Grid (schematisch)
- Tree Positions (circles)
- Distance Circles: 5m, 10m, 20m Radii

**Purpose:** Illustriert Pixel-Mixing-Konzept visuell für Dokumentation.

---

## Output: `proximity_filter.json`

### JSON-Struktur

```json
{
  "version": "1.0",
  "created": "2026-01-30T...",
  "recommended_threshold_m": 20,
  "justification": "20m threshold balances spectral purity (2-pixel buffer) with data retention (87-89%) and genus-uniform impact (max 4.2% deviation)",

  "thresholds_tested": [5, 10, 15, 20, 30],

  "impact_per_threshold": {
    "5": {
      "berlin": { "trees_removed": 458, "retention_rate": 0.99 },
      "leipzig": { "trees_removed": 312, "retention_rate": 0.991 }
    },
    "10": {
      "berlin": { "trees_removed": 2134, "retention_rate": 0.953 },
      "leipzig": { "trees_removed": 1678, "retention_rate": 0.952 }
    },
    "20": {
      "berlin": { "trees_removed": 5847, "retention_rate": 0.87 },
      "leipzig": { "trees_removed": 4123, "retention_rate": 0.882 }
    },
    "30": {
      "berlin": { "trees_removed": 8234, "retention_rate": 0.817 },
      "leipzig": { "trees_removed": 6012, "retention_rate": 0.828 }
    }
  },

  "genus_specific_uniform": true,
  "max_genus_deviation_percent": 4.2,

  "distance_percentiles": {
    "berlin": {
      "10th": 3.2,
      "25th": 5.8,
      "50th": 12.4,
      "75th": 25.6,
      "90th": 48.2,
      "95th": 78.5
    },
    "leipzig": {
      "10th": 3.5,
      "25th": 6.1,
      "50th": 13.1,
      "75th": 26.8,
      "90th": 51.3,
      "95th": 82.1
    }
  },

  "spatial_analysis": {
    "urban_core_removal_rate": 0.15,
    "parks_removal_rate": 0.08,
    "suburbs_removal_rate": 0.12,
    "interpretation": "Higher removal in dense urban core (expected)"
  },

  "validation": {
    "retention_exceeds_85_percent": true,
    "genus_impact_uniform": true,
    "covers_two_pixels": true,
    "literature_supported": true
  }
}
```

### Verwendung in Phase 2c

**Dual Dataset Strategy:**
Phase 2c erstellt **ZWEI** Datensatz-Varianten:

**1. Baseline (No Filter):**

- Alle Bäume (keine Proximity-Filterung)
- Real-world scenario (Mixed Pixels vorhanden)

**2. Filtered (Proximity-Filter angewendet):**

- Bäume mit nearest_diff_genus_m < 20m entfernt
- Spektral "reiner" (weniger Pixel-Mixing)

**Beide Varianten durchlaufen identische Pipeline:**

- Outlier-Flagging
- Redundancy Removal
- Spatial Blocks
- Stratified Splits

**Outputs:** 10 GeoPackages (5 baseline + 5 filtered)

**Rationale:**

- Phase 3 kann **direkt vergleichen:** Hilft der Filter?
- Ablationsstudie: Baseline vs. Filtered Performance
- Wissenschaftlich sauberer als Ad-hoc-Entscheidung

---

## Methodische Verbesserungen gegenüber Legacy

| Aspekt                      | Legacy Pipeline    | Aktuelle Pipeline                               |
| --------------------------- | ------------------ | ----------------------------------------------- |
| **Proximity-Filter**        | Nicht durchgeführt | Empirische Threshold-Validierung                |
| **Threshold-Wahl**          | N/A                | Sensitivity Analysis + Retention Target         |
| **Genus-Bias-Check**        | Nicht durchgeführt | Genus-spezifische Impact-Analyse                |
| **Spatial Patterns**        | Nicht analysiert   | Maps zeigen urban/park/suburb Unterschiede      |
| **Ablationsstudien**        | Nicht möglich      | Dual Datasets (baseline + filtered) für Phase 3 |
| **Pixel-Mixing-Begründung** | Nicht dokumentiert | Theoretische + Empirische Validierung           |

**Zentrale Verbesserung:** Wissenschaftlich fundierte, daten-getriebene Entscheidung für spektrale Reinheit vs. Sample Size Trade-off.

---

## Limitationen

**Proximity ≠ Contamination:**

- 20m Abstand garantiert NICHT 100% Reinheit
- Sentinel-2 Point Spread Function (PSF) kann über Pixel-Grenzen streuen
- Atmospheric Scattering kann benachbarte Pixel beeinflussen

**Binary Decision:**

- Filter ist hart (< 20m = remove, ≥ 20m = keep)
- Real Contamination ist kontinuierlich (0-100%)
- **Alternative:** Weighted Training (Bäume < 20m downweight) – nicht implementiert

**Genus-Only:**

- Ignoriert Baumgröße (kleiner Baum vs. großer Baum)
- Ignoriert Phänologie (Laubbaum im Winter vs. Sommer)
- **Rationale:** Genus-Level ist Target-Variable → genus-mixing ist relevanter als size-mixing

**Static Analysis:**

- Proximity wird einmal berechnet (nicht temporal)
- Annahme: Baumverteilung ändert sich nicht (valide für urbane Bestände)

---

## Nächste Schritte

**Nach Abschluss dieser Analyse:**

1. **proximity_filter.json committen:**
   - Download von Google Drive
   - Commit zu `outputs/phase_2/metadata/`

2. **Phase 2c Implementierung (Updated):**
   - **Dual Pipeline:** Baseline + Filtered Varianten
   - Beide durchlaufen identische Schritte
   - **Output:** 10 GeoPackages (5 × 2)

3. **Naming Convention:**
   - Baseline: `berlin_train.gpkg`, `berlin_val.gpkg`, ...
   - Filtered: `berlin_train_filtered.gpkg`, `berlin_val_filtered.gpkg`, ...

4. **Phase 3 - Ablationsstudie:**
   - Training auf Baseline vs. Filtered
   - Performance-Vergleich: Hilft der Filter?
   - Threshold-Sensitivität (wenn mehrere Varianten erstellt)

---

**Dokumentations-Status:** ✅ Methodischer Prozess vollständig dokumentiert
