# Exploratory 02: CHM Feature Assessment

**Notebook:** `notebooks/exploratory/exp_02_chm_assessment.ipynb`
**Zweck:** Bewertung der CHM-Feature-Qualität, Transferierbarkeit und Bestimmung methodischer Parameter
**Status:** ⏳ In Implementierung

---

## Ziel

Systematische Evaluation der CHM-Features (Kronenhöhe) zur Entscheidung über deren Verwendung in der Genus-Klassifikation. Bestimmung von zwei kritischen Parametern: (1) Plant-Year-Threshold für Tree-Visibility-Filtering und (2) Genus-Classification-Scope (deciduous-only vs. all).

---

## Methodik

### Discriminative Power (ANOVA η²)

**Formel (Effektstärke):**

```
η² = SS_between / SS_total

wobei:
SS_between = Σ(n_i × (μ_i - μ_grand)²)  [Varianz zwischen Genera]
SS_total = Σ(x_ij - μ_grand)²            [Gesamtvarianz]
```

**Interpretation:**

- η² < 0.06: Kleiner Effekt (CHM diskriminiert schwach)
- η² ∈ [0.06, 0.14): Mittlerer Effekt
- η² ≥ 0.14: Großer Effekt (CHM ist genus-spezifisch)

**Berechnung:**

1. One-way ANOVA: `CHM_1m ~ genus_latin` pro Stadt (Berlin, Leipzig)
2. F-Statistik und p-Wert aus `scipy.stats.f_oneway`
3. Manuelle η²-Berechnung aus Sum-of-Squares

**Rationale:** Quantifiziert, wie viel der CHM-Varianz durch Genus-Unterschiede erklärt wird. Hohe η² → CHM ist informatives Feature für Klassifikation.

---

### Cross-City Transferability (Cohen's d)

**Formel (Effektstärke für Mittelwertdifferenzen):**

```
d = (μ_Berlin - μ_Leipzig) / s_pooled

wobei:
s_pooled = √[((n_Berlin - 1) × s²_Berlin + (n_Leipzig - 1) × s²_Leipzig) / (n_Berlin + n_Leipzig - 2)]
```

**Interpretation:**

- |d| < 0.2: Kleine Differenz (gut für Transfer)
- |d| ∈ [0.2, 0.5): Mittlere Differenz (Transfer-Risiko)
- |d| ≥ 0.5: Große Differenz (city-specific, schlecht für Transfer)

**Berechnung:**

1. Für jedes Genus: Mittelwert und Standardabweichung `CHM_1m` in Berlin und Leipzig
2. Cohen's d pro Genus mit gepoolter Standardabweichung
3. Bootstrap-Konfidenzintervalle (1000 Iterationen, 95% CI)
4. Aggregation: Mittleres |d| über alle Genera

**Rationale:** Kleine Cohen's d-Werte bedeuten, dass Genera in beiden Städten ähnliche Kronenhöhen haben → CHM-basierte Features übertragen sich gut von Berlin nach Leipzig.

---

### Feature Engineering Validation

**CHM vs. Kataster-Korrelation:**

- Korrelation: `CHM_1m` vs. `height_m` (Kataster-Baumhöhe)
- **Erwartete Range:** r ∈ [0.4, 0.6]
- **Interpretation:**
  - r < 0.3: CHM-Extraktion fehlerhaft oder Kataster ungenau
  - r ∈ [0.4, 0.6]: Moderate Korrelation (erwünscht)
  - r > 0.7: Zu hohe Korrelation → Hinweis auf Data Leakage (Legacy-Problem)

**Begründung moderate Korrelation:**

- CHM misst Kronenhöhe (top of canopy)
- Kataster misst Gesamthöhe (ground to top)
- Unterschied: Stammhöhe, Messzeitpunkt, Ungenauigkeiten
- Moderate Korrelation validiert CHM-Plausibilität ohne Redundanz

**Transformationen:**

1. **Z-Score (within-genus):**

   ```
   CHM_zscore = (CHM_1m - μ_genus) / σ_genus
   ```

   Normalisiert relative Höhe innerhalb Genus (entfernt genus-spezifische Baseline).

2. **Percentile (within-genus):**
   ```
   CHM_percentile = Rang(CHM_1m, Genus) / n_genus × 100
   ```
   Rang-basiert (0-100), robust gegen Outliers.

**Rationale:** Beide Transformationen reduzieren absolute Höhen-Unterschiede zwischen Städten (Transfer-Verbesserung) und fokussieren auf relative Positionen innerhalb Genus.

---

### Plant Year Threshold Determination

**Zweck:** Identifikation eines Schwellenwerts für `plant_year`, unterhalb dessen Bäume zu jung für zuverlässige Sentinel-2-Detektion sind.

**Methodik:**

1. **Gruppierung nach Pflanzjahr:**
   - Filtere Bäume mit validen `plant_year` und `CHM_1m` (NaN ausschließen)
   - Gruppiere in 1-Jahres- oder 2-Jahres-Kohorten (je nach Datenverteilung)

2. **Median-CHM-Berechnung:**
   - Für jede Kohorte: `median(CHM_1m)`
   - Visualisierung: Median CHM vs. Plant Year

3. **Detection Threshold:**
   - **Schwellenwert:** 2.0m (Sentinel-2 10m-Pixel-Sichtbarkeit)
   - **Begründung:** Bäume <2m Kronenhöhe sind in 10m-Pixeln schwer als dominantes Signal detektierbar (Untergrund-Kontamination)

4. **Cutoff-Bestimmung:**
   - Identifiziere Jahr, ab dem `median(CHM_1m) < 2.0m`
   - **Empfohlener Threshold:** Letztes Jahr mit `median(CHM) ≥ 2.0m`

**Beispiel:**

```
Plant Year  | Median CHM | Interpretation
------------|------------|---------------
2015        | 8.2m       | Ausreichend sichtbar
2016        | 6.5m       | Ausreichend sichtbar
2017        | 4.1m       | Ausreichend sichtbar
2018        | 2.8m       | Grenzwertig, aber OK
2019        | 1.5m       | Zu niedrig → verwerfen
2020        | 0.8m       | Zu niedrig → verwerfen
```

→ **recommended_max_plant_year = 2018**

**Verwendung:** Phase 2b (Quality Control) entfernt alle Bäume mit `plant_year > 2018` (Stadt-spezifischer Schwellenwert möglich).

---

### Genus Classification & Analysis Scope

**Zweck:** Bestimmung, ob Nadelbäume (conifers) in die Klassifikation einbezogen werden sollen.

**Klassifikation (Lookup-Table):**

- **Deciduous (Laubbäume):** TILIA, ACER, QUERCUS, FRAXINUS, PLATANUS, BETULA, PRUNUS, CARPINUS, ALNUS, SORBUS, ULMUS, POPULUS, ROBINIA, SALIX, FAGUS, AESCULUS
- **Coniferous (Nadelbäume):** PINUS, PICEA, THUJA, TAXUS, ABIES, LARIX

**Schwellenwert-Regel:**

```
WENN (n_conifer_genera < 3) ODER (n_conifer_samples < 500):
    analysis_scope = "deciduous_only"
SONST:
    analysis_scope = "all"
```

**Begründung:**

- **Genera-Threshold (3):** Minimum für sinnvolle Klassifikation (mindestens 3 Klassen)
- **Sample-Threshold (500):** Minimum für stabile Statistiken und Train/Val/Test-Split
- **Praxis:** Urbane Baumbestände sind dominiert von Laubbäumen; Nadelbäume oft <2% der Samples

**Konsequenz:**

- `analysis_scope = "deciduous_only"` → Nadelbäume werden in Phase 2b vollständig gefiltert
- `analysis_scope = "all"` → Nadelbäume bleiben im Datensatz

---

## Visualisierungen

### Plot 1: CHM Distribution by Genus

**Typ:** Violin/Box Plots, faceted nach Stadt

**Interpretation:**

- Zeigt genus-spezifische CHM-Verteilungen
- Visuelle Bestätigung der ANOVA η² (getrennte Verteilungen = hohe Diskriminierung)
- Cross-city Vergleich: Ähnliche Verteilungen = gute Transferierbarkeit

---

### Plot 2: CHM vs Cadastre Correlation

**Typ:** Scatter Plot mit Regressionslinie

**Elemente:**

- X-Achse: `height_m` (Kataster)
- Y-Achse: `CHM_1m`
- Farben: Berlin (blau), Leipzig (orange)
- Annotation: Pearson r-Wert

**Interpretation:**

- r ∈ [0.4, 0.6]: CHM-Extraktion valide, keine Redundanz
- Streuung zeigt Messunsicherheiten (normal)

---

### Plot 3: Discriminative Power (η²) Comparison

**Typ:** Bar Chart (Berlin vs Leipzig)

**Elemente:**

- Balken: η²-Werte pro Stadt
- Referenzlinien: medium (0.06), large (0.14)
- Annotation: Interpretation ("medium effect", "large effect")

**Interpretation:**

- Ähnliche η² in beiden Städten → konsistente Feature-Qualität
- η² > 0.14 in beiden Städten → CHM ist genus-diskriminativ

---

### Plot 4: Cohen's d Forest Plot

**Typ:** Forest Plot mit Konfidenzintervallen

**Elemente:**

- Y-Achse: Genera (sortiert nach |d|)
- X-Achse: Cohen's d
- Error Bars: 95% Bootstrap-CI
- Referenzlinien: small (0.2), medium (0.5), large (0.8)

**Interpretation:**

- CIs überlappen mit 0 → keine signifikante Differenz
- |d| < 0.2 für alle Genera → geringe Transfer-Risiken
- Ausreißer-Genera mit |d| > 0.5 → potenzielle Transfer-Probleme

---

### Plot 5: CHM Distribution Comparison

**Typ:** Overlaid Histograms (Berlin vs Leipzig)

**Interpretation:**

- Ähnliche Verteilungsformen → städteübergreifende Konsistenz
- Shift in Mittelwerten → systematische Unterschiede (z.B. Stadtstruktur)

---

### Plot 6: CHM vs Plant Year

**Typ:** Box Plot / Scatter mit Median-Linie

**Elemente:**

- X-Achse: Plant Year (oder Kohorten)
- Y-Achse: CHM_1m
- Horizontale Linie: Detection Threshold (2.0m)
- Highlight: Empfohlenes Cutoff-Jahr

**Interpretation:**

- Klarer Abfall der Median-CHM mit jüngerem Pflanzjahr
- Schwellenwert-Jahr identifiziert Grenze für Visibility

---

### Plot 7: Genus Inventory

**Typ:** Bar Chart (Sample Counts pro Genus)

**Elemente:**

- X-Achse: Genus (sortiert nach Count)
- Y-Achse: Sample Count (log-scale optional)
- Farben: Deciduous (grün), Coniferous (braun)
- Horizontale Linie: Min-Sample-Threshold (500)

**Interpretation:**

- Dominanz von Laubbäumen visuell offensichtlich
- Nadelbäume unterhalb Threshold → Begründung für "deciduous_only"

---

## Output

### JSON: `chm_assessment.json`

**Schema:**

```json
{
  "chm_features": ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"],
  "include_chm": true,
  "discriminative_power": {
    "berlin_eta2": <float>,
    "leipzig_eta2": <float>
  },
  "transfer_risk": {
    "cohens_d_mean": <float>,
    "interpretation": "low/medium/high transfer risk"
  },
  "validation": {
    "chm_cadastre_correlation": <float>
  },
  "plant_year_analysis": {
    "detection_threshold_m": 2.0,
    "median_chm_by_year": {<year>: <median>, ...},
    "recommended_max_plant_year": <int>,
    "justification": "<text>"
  },
  "genus_inventory": {
    "berlin": {<GENUS>: <count>, ...},
    "leipzig": {<GENUS>: <count>, ...},
    "classification": {
      "deciduous": [<list>],
      "coniferous": [<list>]
    },
    "conifer_analysis": {
      "n_genera": <int>,
      "n_samples": <int>,
      "include_in_analysis": <bool>,
      "reason": "<text>"
    },
    "analysis_scope": "deciduous_only" | "all"
  }
}
```

**Verwendung:** Phase 2b (Quality Control) lädt `recommended_max_plant_year` und `analysis_scope` für Tree-Filtering.

---

## Plots (7 Dateien)

```
outputs/phase_2/figures/exp_02_chm/
├── chm_boxplot_per_genus.png           # CHM-Verteilungen pro Genus
├── chm_cadastre_correlation.png        # CHM vs. Kataster-Korrelation
├── eta2_comparison.png                 # ANOVA η² (Berlin vs Leipzig)
├── cohens_d_forest_plot.png            # Cohen's d mit CIs
├── chm_distribution_cities.png         # Overlaid Histogramme
├── chm_vs_plant_year.png               # CHM vs. Pflanzjahr
└── genus_inventory.png                 # Sample Counts pro Genus
```

**DPI:** 300 (Publication-ready)

---

## Validierung

**In-Notebook Checks:**

- ✅ η² in Range [0, 1]
- ✅ Cohen's d Konfidenzintervalle überlappen nicht pathologisch
- ✅ CHM-Kataster-Korrelation r ∈ [0.3, 0.7]
- ✅ Plant-Year-Cutoff plausibel (2016-2019)
- ✅ Genus-Klassifikation vollständig (alle viable genera klassifiziert)
- ✅ Cross-city Schema-Konsistenz

**Post-Execution:**

- JSON-Schema vollständig
- 7 PNG-Plots gespeichert
- Execution Log generiert

---

## Nächste Schritte

**Manual Sync (nach Colab-Ausführung):**

1. Download `chm_assessment.json` von Drive
2. Commit zu Git: `outputs/phase_2/metadata/chm_assessment.json`
3. Download Plots von Drive: `outputs/phase_2/figures/exp_02_chm/*.png`
4. Commit Plots zu Git
5. Push zu GitHub

**Verwendung in Phase 2b:**

Die JSON-Datei wird geladen und die folgenden Parameter extrahiert:

- `recommended_max_plant_year` → Filter: Entferne Bäume mit `plant_year > threshold`
- `analysis_scope` → Filter: Wenn "deciduous_only", entferne alle Nadelbäume

CHM-Features (`CHM_1m`, `CHM_1m_zscore`, `CHM_1m_percentile`) werden in allen nachfolgenden Schritten verwendet, falls `include_chm = true`.

---

**Dokumentations-Status:** ✅ Exploratory 02 (CHM Assessment) dokumentiert
