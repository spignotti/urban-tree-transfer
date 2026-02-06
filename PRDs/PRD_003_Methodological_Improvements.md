# PRD: Methodological Improvements Phase 3

**PRD ID:** 003_Methodological_Improvements  
**Status:** Draft  
**Created:** 2026-02-06  
**Phase:** 3 - Experiments  
**Purpose:** Systematische methodische Verbesserungen basierend auf Literatur-Review

---

## Executive Summary

Dieses PRD beschreibt **sechs methodische Verbesserungen** für Phase 3 (Experiments), die aus einem systematischen Abgleich mit der wissenschaftlichen Literatur abgeleitet wurden. Alle Verbesserungen erhöhen die **wissenschaftliche Rigorosität**, **Interpretierbarkeit** und **Vergleichbarkeit** mit publizierten Studien.

**Scope:** Änderungen betreffen Experiment-Design, Evaluierung und Dokumentation (exp_08-10, 03b-03d).  
**Out of Scope:** Änderungen an Phase 1/2 Pipelines.

---

## Motivation

Das Urban Tree Transfer Projekt implementiert bereits viele Best Practices (Spatial CV, CHM-Integration, Multi-temporale Daten). Ein Literatur-Review (siehe `docs/literature/LITERATURE_ANALYSIS.md` und `METHODOLOGY_REVIEW.md`) identifizierte jedoch spezifische Lücken:

1. **Fehlende Performance-Untergrenzen:** Keine Baselines (z.B. Majority Class) als Vergleichspunkt
2. **Unsicherheitsquantifizierung fehlt:** Keine Konfidenzintervalle für Test-Metriken
3. **Transfer-Mechanismen unklar:** Welche Features sind stabil vs. city-specific?
4. **Post-hoc Interpretation:** Keine a-priori Hypothesen für Per-Genus-Transfer-Analyse
5. **Literatur-Kontext fehlt:** Wie viele Features sind üblich in Tree-Classification-Studien?
6. **Sample-Effizienz unpräzise:** Fine-Tuning-Kurve nicht mathematisch modelliert

Diese Lücken schränken die **wissenschaftliche Verteidigbarkeit** und **praktische Interpretierbarkeit** der Ergebnisse ein.

---

## Improvement 1: Feature-Reduction - Pareto-Kurve & Literatur-Comparison

### Problem

**Identifiziert in:** Setup-Fixierung (exp_09_feature_reduction)

Die aktuelle Feature-Selektion testet verschiedene Feature-Counts (Top-30, Top-50, Top-80, All), dokumentiert aber nicht:

1. **Literatur-Kontext fehlt:** Wie viele Features verwenden vergleichbare Studien?
   - Hemmerling et al. (2021): ~276 Features
   - Immitzer et al. (2019): 49 Features  
   - Grabska et al. (2019): 40 Features
   - Unser Projekt: 30-80 → Ist das viel oder wenig?

2. **Kein visueller Trade-off:** Feature Count vs. Performance ist nicht als Pareto-Kurve dargestellt
   - Wo liegt der "Knee-Point" (optimal Trade-off)?
   - Ab wann gibt es diminishing returns?

3. **Hughes-Effekt nicht explizit:** Fassnacht et al. (2016) warnen vor "Curse of Dimensionality"
   - Sinkt Performance bei ALL Features vs. Top-50?
   - Falls ja → expliziter Kommentar fehlt

### Solution

#### 1.1 Literatur-Benchmark-Tabelle

Erstelle Vergleichstabelle in `exp_09` Dokumentation:

| Studie | Features | Input Typ | Accuracy | Anmerkung |
|--------|----------|-----------|----------|-----------|
| Hemmerling et al. 2021 | ~276 | 12 months × 23 features | 82-94% | Dense time series |
| Immitzer et al. 2019 | 49 | Feature-selected | 76% | Top S2 features |
| Dieses Projekt | 30-80 | 8 months × selected | TBD | RF importance-based |

**Interpretation:** Positioniert Projekt im wissenschaftlichen Kontext.

#### 1.2 Pareto-Kurve-Visualisierung

- **X-Achse:** Feature Count (10, 20, 30, 50, 80, 100, ALL)
- **Y-Achse:** Weighted F1-Score
- **Markierung:** "Knee-Point" (optimaler Trade-off)
- **Caption:** "Diminishing returns beyond X features"

#### 1.3 Hughes-Effekt-Analyse

Falls F1 bei ALL Features < F1 bei Top-50:
- **Dokumentation:** Expliziter Hinweis auf Hughes-Effekt (Fassnacht et al. 2016)
- **Interpretation:** "Feature-Selektion ist essentiell, nicht optional"

### Acceptance Criteria

- [ ] Literatur-Vergleichstabelle in exp_09 Dokumentation
- [ ] Pareto-Kurve als `feature_pareto_curve.png`
- [ ] Knee-Point identifiziert und dokumentiert
- [ ] Hughes-Effekt-Kommentar falls zutreffend

### Effort & Priority

**Effort:** 2h | **Priority:** 🟡 Mittel | **Impact:** 🎯 Hoch  
**Phase:** 3.1 (Setup-Fixierung, exp_09)

---

## Improvement 2: Berlin-Optimierung - Naive Baselines

### Problem

**Identifiziert in:** exp_10_algorithm_comparison + 03b_berlin_optimization

Die Algorithmen-Evaluierung testet RF, XGBoost, 1D-CNN, TabNet, hat aber **keine Performance-Untergrenze**:

1. **Absolute Performance unklar:** Wenn ML-Champion F1 = 0.74 erreicht → ist das gut oder schlecht?
2. **Kein dummer Baseline:** Was wäre die Performance ohne Features (nur Koordinaten/Mehrheitsklasse)?
3. **Literatur-Standard:** Belgiu & Drăguţ (2016) empfehlen Dummy Classifiers als Baseline

### Solution

#### 2.1 Drei Naive Baselines hinzufügen

**Baseline 1: Majority Class Classifier**
- Strategie: Immer die häufigste Gattung vorhersagen
- Zweck: Absolute Untergrenze

**Baseline 2: Stratified Random Classifier**
- Strategie: Zufällige Vorhersagen proportional zur Klassenverteilung
- Zweck: Zeigt, ob Modell besser ist als gewichtetes Raten

**Baseline 3: Spatial-Only Random Forest**
- Features: Nur x/y-Koordinaten (keine Sentinel-2, kein CHM)
- Zweck: Zeigt Wert der Remote-Sensing-Features

#### 2.2 Performance Ladder Visualisierung

Horizontale Bar-Chart (sortiert nach F1):
- Majority Class: F1 = 0.18 (rot)
- Stratified Random: F1 = 0.22 (rot)
- Spatial-Only RF: F1 = 0.35 (orange)
- RF Default: F1 = 0.68 (gelb)
- ML Champion (tuned): F1 = 0.74 (grün)

**Caption:** "Champion ist 42pp besser als Spatial-Only, 52pp besser als Majority Class"

### Acceptance Criteria

- [ ] Drei Baselines in exp_10 implementiert
- [ ] Performance Ladder Plot in 03b
- [ ] Relative Improvements dokumentiert (pp = percentage points)

### Effort & Priority

**Effort:** 1h | **Priority:** 🔴 Hoch | **Impact:** 🎯 Hoch  
**Phase:** 3.2 (exp_10) + 3.3 (03b)

---

## Improvement 3: Transfer-Evaluation - Feature-Stability-Analyse

### Problem

**Identifiziert in:** 03c_transfer_evaluation

Die Transfer-Evaluation berechnet Zero-Shot-Performance und Transfer-Gap, analysiert aber nicht **welche Features transferieren**:

1. **Feature-Mechanismen unklar:** Welche Features bleiben wichtig (Berlin → Leipzig)?
2. **City-Specific Overfitting:** Sind CHM-Features city-specific (hohe Importance in Berlin, niedrig in Leipzig)?
3. **Robuste Features fehlen:** Welche Features sind transfer-robust (z.B. Red-Edge)?

**Literatur-Basis:**
- Tuia et al. (2016): "Feature-Importance-Shifts zeigen Domain Shift"
- Tong et al. (2019): "Spektrale Verschiebungen reduzieren Effektivität"

### Solution

#### 3.1 Leipzig-From-Scratch Training

- Trainiere Leipzig-Modell mit identischen Hyperparametern wie Berlin-Champion
- Zweck: Fair Comparison der Feature Importances

#### 3.2 Feature Importance Comparison

- **Metric:** Spearman Rank Correlation (ρ) zwischen Berlin und Leipzig Importances
- **Interpretation:**
  - ρ > 0.7: Hohe Stabilität → Features transferieren gut
  - ρ < 0.5: Niedrige Stabilität → city-specific Features dominieren

#### 3.3 Visualisierung: Scatter Plot

- **X-Achse:** Berlin Feature Importance (Rank)
- **Y-Achse:** Leipzig Feature Importance (Rank)
- **Punkte:** Features (coloriert nach Typ: Spectral, Red-Edge, CHM, Moisture)
- **Diagonale:** Perfekte Übereinstimmung
- **Interpretation:** Punkte nahe Diagonale = stabile Features

#### 3.4 Transfer-Robustness-Ranking

JSON-Output in `transfer_evaluation.json`:

```json
{
  "feature_stability": {
    "spearman_rho": 0.68,
    "interpretation": "Moderate stability",
    "most_stable_features": ["NDVIre_Jun", "NDVI_Jul", "B8A_Aug"],
    "most_unstable_features": ["CHM_1m_zscore", "MSI_Nov"],
    "literature_validation": "Red-Edge stable (Immitzer 2019), CHM city-specific (expected)"
  }
}
```

### Acceptance Criteria

- [ ] Leipzig-from-scratch Modell trainiert
- [ ] Spearman ρ berechnet und dokumentiert
- [ ] Scatter Plot mit Feature-Typ-Colorierung
- [ ] Top-5 stable/unstable Features identifiziert
- [ ] Literatur-Referenzen für stabile Features

### Effort & Priority

**Effort:** 4h | **Priority:** 🔴 Hoch | **Impact:** 🎯 Hoch  
**Phase:** 3.4 (03c_transfer_evaluation)

---

## Improvement 4: Confidence Intervals - Bootstrap-Estimates

### Problem

**Identifiziert in:** 03b_berlin_optimization + 03c_transfer_evaluation

Alle finalen Test-Metriken sind Punktschätzungen ohne Unsicherheitsangaben:

1. **Keine Konfidenzintervalle:** "F1 = 0.74" → aber wie sicher ist dieser Wert?
2. **Fassnacht et al. (2016):** "Genauigkeit kann je nach Split um 5–10% schwanken"
3. **Roberts et al. (2017):** "Konfidenzintervalle essentiell für robuste Aussagen"
4. **Transfer-Gap-Signifikanz unklar:** Ist Berlin F1 = 0.74 vs. Leipzig F1 = 0.62 statistisch signifikant?

### Solution

#### 4.1 Bootstrap Confidence Intervals

**Methode:** 1000 Bootstrap-Resamples des Test-Sets
- Für jedes Resample: Berechne Metrik (F1, Accuracy, etc.)
- CI: 2.5% und 97.5% Perzentile der Bootstrap-Verteilung

**Output-Format:** "F1 = 0.742 (95% CI: [0.731, 0.753])"

**Anwendung:**
- Berlin Test Metriken (03b)
- Leipzig Zero-Shot Metriken (03c)
- Transfer-Gap CI (Differenz der Bootstrap-Verteilungen)

#### 4.2 Statistical Significance Tests

**Transfer-Gap-Test:**
- H0: Berlin F1 = Leipzig F1 (kein Transfer-Gap)
- H1: Berlin F1 > Leipzig F1 (Transfer-Gap existiert)
- Test: Mann-Whitney U (non-parametric) auf Bootstrap-Verteilungen
- Output: "Transfer-Gap statistisch signifikant (p < 0.001)"

#### 4.3 JSON-Output-Format

```json
{
  "berlin_test_metrics": {
    "weighted_f1": {
      "point_estimate": 0.742,
      "ci_95_lower": 0.731,
      "ci_95_upper": 0.753,
      "method": "bootstrap_1000"
    }
  },
  "transfer_evaluation": {
    "transfer_gap": {
      "weighted_f1_loss": 0.119,
      "ci_95_lower": 0.093,
      "ci_95_upper": 0.145,
      "statistical_significance": "p < 0.001"
    }
  }
}
```

### Acceptance Criteria

- [ ] Bootstrap CI für alle finale Test-Metriken
- [ ] Mann-Whitney U Test für Transfer-Gap
- [ ] CI in allen Visualisierungen (Error Bars)
- [ ] JSON-Output mit CI-Werten

### Effort & Priority

**Effort:** 3h | **Priority:** 🔴 Hoch | **Impact:** 🎯 Hoch  
**Phase:** 3.3 (03b) + 3.4 (03c)

---

## Improvement 5: Per-Genus Transfer - Hypothesen-Testing

### Problem

**Identifiziert in:** 03c_transfer_evaluation

Die Per-Genus-Analyse berechnet Transfer-Gaps pro Gattung, hat aber **keine a-priori Hypothesen**:

1. **Post-hoc Storytelling:** Interpretation erfolgt nach Sichtung der Ergebnisse
2. **Keine Struktur:** Welche Muster erwarten wir? (z.B. Nadelbäume robuster?)
3. **Literatur-Hypothesen ungenutzt:**
   - Fassnacht et al. (2016): "Nadelbäume spektral distinkt"
   - Velasquez-Camacho et al. (2021): "Häufige Genera robuster"
   - Hemmerling et al. (2021): "Regionale Phänologie-Unterschiede"

### Solution

#### 5.1 A-Priori Hypothesen formulieren

**Vor 03c-Ausführung** in Dokumentation (`03_Transfer_Evaluation.md`) festhalten:

**H1 (Sample Size):**
- Hypothese: Genera mit mehr Berlin-Trainingssamples transferieren besser
- Test: Pearson r zwischen `berlin_samples` und `transfer_gap`
- Erwartung: r < 0 (mehr Samples → geringerer Gap)

**H2 (Conifer vs. Deciduous):**
- Hypothese: Nadelbäume (PINUS, PICEA) haben geringeren Transfer-Gap als Laubbäume
- Test: Mann-Whitney U zwischen Nadel-Gap und Laub-Gap
- Begründung: Fassnacht - Nadelbäume immergrün → distinkteres Spektralprofil

**H3 (Phenological Distinctness):**
- Hypothese: Genera mit frühem Leaf-Out (BETULA, SALIX) haben höheren Transfer-Gap
- Begründung: Hemmerling - regional unterschiedliche Phänologie

**H4 (Red-Edge Robustness):**
- Hypothese: Genera mit hoher Red-Edge-Feature-Importance transferieren besser
- Begründung: Immitzer - Red-Edge-Indizes optimal für Baumarten

#### 5.2 Quantitative Tests

Für jede Hypothese:
- Statistischer Test durchführen
- p-Wert dokumentieren
- Hypothese bestätigt/verworfen

#### 5.3 Transfer-Robustness-Ranking

Visualisierung: Horizontal Bar Chart
- Y-Achse: Genera (sortiert nach Transfer-Gap)
- X-Achse: Transfer-Gap (niedrig → robust, hoch → fragil)
- Color: Genus-Typ (Nadel vs. Laub)
- Annotation: Sample Size

### Acceptance Criteria

- [ ] 4 Hypothesen vor 03c in Doku formuliert
- [ ] Quantitative Tests für alle Hypothesen
- [ ] p-Werte und Interpretationen dokumentiert
- [ ] Transfer-Robustness-Ranking Plot

### Effort & Priority

**Effort:** 3h | **Priority:** 🟡 Mittel | **Impact:** 🎯 Hoch  
**Phase:** 3.4 (03c_transfer_evaluation)

---

## Improvement 6: Fine-Tuning - Learning-Curve Power-Law-Fit

### Problem

**Identifiziert in:** 03d_finetuning

Die Fine-Tuning-Experimente testen 10%, 25%, 50%, 100% Leipzig-Daten, aber:

1. **Keine mathematische Modellierung:** Learning Curves folgen typischerweise Power-Law: `Performance = a × N^b`
2. **Extrapolation fehlt:** Wie viel Prozent für 95% der from-scratch Performance?
3. **Unpräzise Empfehlungen:** Statt "~10-25% reichen" → "18.3% für 95% Recovery"

**Literatur-Basis:**
- Tong et al. (2019): "Transfer Learning reduziert Label-Bedarf um 50-70%"

### Solution

#### 6.1 Power-Law-Fit

- Funktion: `y = a × x^b`
- Input: Sample Sizes [0.10, 0.25, 0.50, 1.0] + corresponding F1 Scores
- Output: Parameter a, b + Fitted Curve

#### 6.2 Extrapolation: 95% Recovery Point

- Berechne: `target_f1 = 0.95 × from_scratch_f1`
- Finde: x (sample %) wo fitted curve `target_f1` erreicht
- Output: "Für 95% der from-scratch Performance: X% Fine-Tuning-Daten nötig"

#### 6.3 Visualisierung: Learning Curve mit Fit

- **X-Achse:** % Leipzig Fine-Tuning Data (log scale)
- **Y-Achse:** F1-Score
- **Punkte:** Tatsächliche Messungen (10%, 25%, 50%, 100%)
- **Linie:** Power-Law-Fit (extrapoliert bis 5%)
- **Horizontale Linie:** from-scratch F1 (Leipzig-only)
- **Marker:** "95% Recovery Point"

#### 6.4 Literatur-Comparison

```markdown
**Sample Efficiency:**
- Zero-shot: F1 = 0.60 (→ 18pp gap to from-scratch)
- 25% fine-tuning: F1 = 0.70 (→ 8pp gap, 55% recovery)
- 50% fine-tuning: F1 = 0.73 (→ 5pp gap, 72% recovery)

**Interpretation:**
- Mit 25% der Leipzig-Daten erreichen wir 55% der Performance-Recovery
- Konsistent mit Tong et al. (2019): Transfer Learning spart ~50% Labels
- Power-Law-Extrapolation: 18.3% für 95% Recovery
```

### Acceptance Criteria

- [ ] Power-Law-Fit implementiert (scipy.optimize.curve_fit)
- [ ] 95% Recovery Point berechnet
- [ ] Learning Curve Plot mit Fit-Linie
- [ ] Literatur-Comparison (Tong et al.)

### Effort & Priority

**Effort:** 2h | **Priority:** 🟢 Niedrig | **Impact:** 🎯 Mittel  
**Phase:** 3.5 (03d_finetuning)

---

## Summary & Prioritization

### Overview Table

| ID | Improvement | Priority | Effort | Impact | Phase | Critical? |
|----|-------------|----------|--------|--------|-------|-----------|
| **Imp 2** | Naive Baselines | 🔴 Hoch | 1h | 🎯 Hoch | 3.2, 3.3 | ✅ Ja |
| **Imp 4** | Bootstrap CI | 🔴 Hoch | 3h | 🎯 Hoch | 3.3, 3.4 | ✅ Ja |
| **Imp 3** | Feature-Stability | 🔴 Hoch | 4h | 🎯 Hoch | 3.4 | ✅ Ja |
| **Imp 1** | Pareto-Kurve | 🟡 Mittel | 2h | 🎯 Hoch | 3.1 | ⏸️ Optional |
| **Imp 5** | Hypothesen-Testing | 🟡 Mittel | 3h | 🎯 Hoch | 3.4 | ⏸️ Optional |
| **Imp 6** | Power-Law-Fit | 🟢 Niedrig | 2h | 🎯 Mittel | 3.5 | ⏸️ Optional |

**Total Critical:** 8h (Imp 2-4)  
**Total Optional:** +7h (Imp 1, 5-6)  
**Grand Total:** 15h

### Recommended Implementation Order

**Sofort (vor exp_10):**
1. Imp 1: Pareto-Kurve in exp_09 (2h) ← während Setup-Fixierung

**Während Phase 3.2-3.3 (Berlin-Optimierung):**
2. Imp 2: Naive Baselines in exp_10 + 03b (1h)
3. Imp 4: Bootstrap CI in 03b (1.5h)

**Während Phase 3.4 (Transfer-Evaluation):**
4. Imp 3: Feature-Stability in 03c (4h) ← **höchste Priorität**
5. Imp 4: Bootstrap CI in 03c (1.5h)
6. Imp 5: Hypothesen-Testing in 03c (3h)

**Während Phase 3.5 (Fine-Tuning):**
7. Imp 6: Power-Law-Fit in 03d (2h)

---

## Implementation Plan

### Modified Notebooks

| Notebook | Changes | Affected Sections |
|----------|---------|-------------------|
| exp_09_feature_reduction.ipynb | Pareto-Kurve, Literatur-Tabelle | Feature Selection Analysis |
| exp_10_algorithm_comparison.ipynb | Naive Baselines | Baseline Comparison |
| 03b_berlin_optimization.ipynb | Naive Baselines, Bootstrap CI | Final Evaluation |
| 03c_transfer_evaluation.ipynb | Feature-Stability, Bootstrap CI, Hypothesen | Transfer Analysis |
| 03d_finetuning.ipynb | Power-Law-Fit | Learning Curve Analysis |

### Modified Documentation

| Document | Changes |
|----------|---------|
| 03_Experiments/01_Setup_Fixierung.md | Pareto-Kurve-Referenz |
| 03_Experiments/02_Berlin_Optimierung.md | Naive Baselines, Bootstrap CI |
| 03_Experiments/03_Transfer_Evaluation.md | Feature-Stability, Hypothesen (a-priori) |
| 03_Experiments/04_Finetuning.md | Power-Law-Fit, Literatur-Comparison |

### New Outputs

```
outputs/phase_3/
├── figures/
│   ├── exp_09_feature_reduction/
│   │   └── feature_pareto_curve.png          # NEW (Imp 1)
│   ├── berlin_optimization/
│   │   └── performance_ladder.png            # NEW (Imp 2)
│   └── transfer/
│       ├── feature_stability_scatter.png     # NEW (Imp 3)
│       └── transfer_robustness_ranking.png   # NEW (Imp 5)
├── metadata/
│   ├── setup_decisions.json                  # EXTENDED (Imp 1)
│   ├── berlin_evaluation.json                # EXTENDED (Imp 2, 4)
│   ├── transfer_evaluation.json              # EXTENDED (Imp 3, 4, 5)
│   └── finetuning_curve.json                 # EXTENDED (Imp 6)
```

---

## Dependencies & Risks

### External Dependencies

- scipy (curve_fit für Power-Law)
- Keine neuen Python-Pakete nötig

### Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Bootstrap zu langsam** (1000 iterations) | Reduce zu 500 iterations (akzeptable CI-Präzision) |
| **Leipzig-from-scratch Training zeitaufwendig** | Nur für ML-Champion durchführen (nicht NN) |
| **Power-Law-Fit konvergiert nicht** | Fallback: Linear Interpolation |
| **Hypothesen werden alle verworfen** | Wertvoll für Diskussion ("warum nicht wie erwartet?") |

---

## Success Criteria

### Definition of Done

- [ ] Alle 6 Improvements implementiert (mindestens Imp 2-4)
- [ ] Alle neuen Visualisierungen generiert
- [ ] JSON-Outputs erweitert mit neuen Metriken
- [ ] Dokumentation aktualisiert mit Methoden-Beschreibungen
- [ ] Literatur-Referenzen in allen relevanten Sektionen

### Quality Gates

- [ ] Bootstrap CI: 95% Intervalle plausibel (nicht zu schmal/breit)
- [ ] Feature-Stability: Spearman ρ berechnet und statistisch getestet
- [ ] Hypothesen: Alle 4 Tests durchgeführt, p-Werte dokumentiert
- [ ] Power-Law-Fit: R² > 0.90 (gute Anpassung)

---

## References

### Literature

- **Fassnacht et al. (2016):** Tree species classification review - Hughes-Effekt, Nadel vs. Laub
- **Roberts et al. (2017):** Spatial Cross-Validation - Konfidenzintervalle
- **Tuia et al. (2016):** Domain Adaptation - Feature-Importance-Shifts
- **Tong et al. (2019):** Transfer Learning - Sample-Effizienz
- **Belgiu & Drăguţ (2016):** Random Forest - Baseline-Comparison
- **Hemmerling et al. (2021):** Dense Time Series - Feature Count (276)
- **Immitzer et al. (2019):** Optimal S2 Features - Feature Count (49)

### Internal Documents

- `docs/literature/LITERATURE_ANALYSIS.md` - Systematische Literatur-Extraktion
- `docs/literature/METHODOLOGY_REVIEW.md` - Abgleich Projekt vs. Literatur
- `PRDs/003_phase3_experiments.md` - Hauptexperiment-PRD

---

_Erstellt: 2026-02-06_  
_Status: Draft - Ready for Review_
