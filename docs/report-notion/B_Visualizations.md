# B. Visualizations

> Migriert aus Notion am 2026-10-28 (Seite `8005e1d025f64eaf9cf5e9c8f1cb37e1`, zuletzt bearbeitet 2026-03-30). Text unverändert übernommen.

Definition aller Figures. Figure IDs matchen `[FIGURE: fig-slug]` Marker im Draft.

---

### fig-study-area

- **Purpose:** Räumliche Einordnung Berlin + Leipzig mit Baumstandorten
- **Type:** hybrid
- **Target Section:** 3.1 Study Areas
- **Data Hint:** Stadtgrenzen (Open Data Polygon) + Baumstandorte (GeoPackage)
- **Style Hint:** Karte mit zwei Insets oder Side-by-Side
- **Source:** own
- **Status:** ⏳

### fig-genus-distribution

- **Purpose:** Klassenimbalance + Domain Shift zwischen Berlin und Leipzig visualisieren
- **Type:** plot
- **Target Section:** 3.3 Genus Distribution
- **Data Hint:** `pipeline_summary.json` -> genus\_distribution
- **Style Hint:** Grouped bar chart, sortiert nach Häufigkeit
- **Source:** own
- **Status:** ⏳

### fig-pipeline

- **Purpose:** Gesamtworkflow auf einen Blick (Data -> FE -> Experiments)
- **Type:** diagram
- **Target Section:** 4 Methods
- **Data Hint:** Manuell (Mermaid / [Draw.io](http://Draw.io))
- **Style Hint:** Flowchart, clean, minimal
- **Source:** own
- **Status:** ⏳

### fig-jm-monthly

- **Purpose:** Begründung Monatsauswahl Apr-Nov via JM-Distance Ranking
- **Type:** plot
- **Target Section:** 4.2 Feature Engineering
- **Data Hint:** `jm_distances.json` (exp\_01)
- **Style Hint:** Heatmap oder Bar chart
- **Source:** own
- **Status:** ⏳

### fig-morans-i

- **Purpose:** Begründung Block-Size 1200 m via räumliche Autokorrelation
- **Type:** plot
- **Target Section:** 4.2 Feature Engineering
- **Data Hint:** `morans_i_decay.json` (exp\_05)
- **Style Hint:** Line plot, decay curves für beide Städte
- **Source:** own
- **Status:** ⏳

### fig-chm-cohens-d

- **Purpose:** CHM-Trennfähigkeit pro Genus quantifizieren
- **Type:** plot
- **Target Section:** 4.2 Feature Engineering
- **Data Hint:** `chm_assessment.json` (exp\_02)
- **Style Hint:** Forest plot, horizontal
- **Source:** own
- **Status:** ⏳

### fig-correlation

- **Purpose:** Feature-Redundanz sichtbar machen
- **Type:** plot
- **Target Section:** 4.2 Feature Engineering
- **Data Hint:** `correlation_matrix.json` (exp\_03)
- **Style Hint:** Heatmap, diverging color scale
- **Source:** own
- **Status:** ⏳

### fig-algorithm-comparison

- **Purpose:** Champion-Selektion (RF, XGB, CNN, TabNet) visualisieren
- **Type:** plot
- **Target Section:** 5.1 Berlin Optimisation
- **Data Hint:** `algorithm_comparison.json` (exp\_11 / 03b)
- **Style Hint:** Bar chart + Confidence Intervals
- **Source:** own
- **Status:** ⏳

### fig-feature-importance

- **Purpose:** Prädiktive Features identifizieren (Top-20)
- **Type:** plot
- **Target Section:** 5.1 Berlin Optimisation
- **Data Hint:** `feature_importances.json` (03b)
- **Style Hint:** Horizontal bar chart
- **Source:** own
- **Status:** ⏳

### fig-confusion-berlin

- **Purpose:** Verwechslungsmuster Berlin (XGBoost) aufzeigen
- **Type:** plot
- **Target Section:** 5.1 Berlin Optimisation
- **Data Hint:** `confusion_matrix_berlin.json` (03b)
- **Style Hint:** Heatmap, annotated
- **Source:** own
- **Status:** ⏳

### fig-per-genus-transfer

- **Purpose:** Transfer-Gap pro Genus visualisieren (Berlin vs. Leipzig F1)
- **Type:** plot
- **Target Section:** 5.2 Zero-Shot Transfer
- **Data Hint:** `transfer_metrics.json` (03c)
- **Style Hint:** Paired dot/bar chart
- **Source:** own
- **Status:** ⏳

### fig-finetuning-curve

- **Purpose:** Sample Efficiency beider Modelle vergleichen (Learning Curve)
- **Type:** plot
- **Target Section:** 5.4 Fine-Tuning Analysis
- **Data Hint:** `finetuning_curves.json` (03d)
- **Style Hint:** Line plot + Confidence Intervals, beide Modelle
- **Source:** own
- **Status:** ⏳

### fig-per-genus-recovery

- **Purpose:** Genus-spezifischen Fine-Tuning-Bedarf aufzeigen
- **Type:** plot
- **Target Section:** 5.4 Fine-Tuning Analysis
- **Data Hint:** `finetuning_per_genus.json` (03d)
- **Style Hint:** Heatmap, Genera x Fractions
- **Source:** own
- **Status:** ⏳

### fig-transfer-robustness

- **Purpose:** Zusammenfassung Transfer-Qualität über alle Experimente
- **Type:** hybrid
- **Target Section:** 6 Discussion
- **Data Hint:** `transfer_metrics.json` + `finetuning_curves.json` (03c+03d)
- **Style Hint:** Summary chart, composite
- **Source:** own
- **Status:** ⏳

---

## Appendix Figures (A1–A4)

### fig-confusion-berlin-xgb

- **Purpose:** Normalisierte Confusion Matrix Berlin — XGBoost (Appendix A1)
- **Type:** plot
- **Target Section:** A.3 Supplementary Confusion Matrices
- **Data Hint:** `confusion_matrix_berlin_xgb.json` (03b) — Shape: `{"matrix": [[...]], "labels": [...]}`
- **Style Hint:** Heatmap, normalisiert, annotiert, 17×17
- **Source:** own
- **Status:** ⏳

### fig-confusion-berlin-cnn

- **Purpose:** Normalisierte Confusion Matrix Berlin — 1D-CNN (Appendix A2)
- **Type:** plot
- **Target Section:** A.3 Supplementary Confusion Matrices
- **Data Hint:** `confusion_matrix_berlin_cnn.json` (03b) — Shape: `{"matrix": [[...]], "labels": [...]}`
- **Style Hint:** Heatmap, normalisiert, annotiert, 17×17
- **Source:** own
- **Status:** ⏳

### fig-confusion-leipzig-zeroshot

- **Purpose:** Normalisierte Confusion Matrix Leipzig Zero-Shot — XGBoost + 1D-CNN als 2-Panel-Figure (Appendix A3)
- **Type:** plot
- **Target Section:** A.3 Supplementary Confusion Matrices
- **Data Hint:** `confusion_matrix_leipzig_zeroshot.json` (03c) — Shape: `{"xgb": [[...]], "cnn": [[...]], "labels": [...]}`
- **Style Hint:** 2-Panel Heatmap (links XGBoost, rechts 1D-CNN), normalisiert, annotiert
- **Source:** own
- **Status:** ⏳

### fig-confusion-leipzig-finetuned100

- **Purpose:** Normalisierte Confusion Matrix Leipzig Fine-Tuned 100% — XGBoost (Appendix A4)
- **Type:** plot
- **Target Section:** A.3 Supplementary Confusion Matrices
- **Data Hint:** `confusion_matrix_leipzig_finetuned100.json` (03d) — Shape: `{"matrix": [[...]], "labels": [...]}`
- **Style Hint:** Heatmap, normalisiert, annotiert, 17×17
- **Source:** own
- **Status:** ⏳
