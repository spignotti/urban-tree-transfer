# PRD: Phase 3 Experiments (Overview)

**PRD ID:** 003
**Status:** Draft
**Created:** 2026-01-28
**Last Updated:** 2026-01-28

---

## 1. Overview

### 1.1 Context

Phase 2 Feature Engineering produces ML-ready datasets:

- **Berlin:** `train.gpkg` (70%), `val.gpkg` (15%), `test.gpkg` (15%)
- **Leipzig:** `finetune.gpkg` (80%), `test.gpkg` (20%)

Phase 3 uses these datasets to answer the core research questions about cross-city transfer learning for tree genus classification.

### 1.2 Research Questions

1. **RQ1 (Single-City):** What is the best achievable performance on Berlin with Sentinel-2 + CHM features?
2. **RQ2 (Transfer):** How much does performance drop when applying a Berlin-trained model to Leipzig (zero-shot)?
3. **RQ3 (Fine-Tuning):** How much Leipzig fine-tuning data is needed to recover performance?

### 1.3 Experiment Structure

```
Phase 3: Experiments
│
├── 3.0: Cross-City Baseline Analysis
│       Deskriptive Analyse der Datasets
│       → Hypothesen für Transfer-Schwierigkeiten
│
├── 3.1: Berlin Single-City Optimization
│       HP-Tuning, Modellvergleich, Ablation Studies
│       → Best Berlin Model
│
├── 3.2: Cross-City Zero-Shot Transfer
│       Berlin-Modell → Leipzig-Test (ohne Fine-Tuning)
│       → Transfer Gap quantifizieren
│
└── 3.3: Fine-Tuning Experiments
        10%, 25%, 50%, 100% Leipzig-Finetune
        → Fine-Tuning Curve, Sample Efficiency
```

---

## 2. Experiment 3.0: Cross-City Baseline Analysis

### 2.1 Purpose

Deskriptive Analyse der finalen Datasets **vor** den Modellexperimenten. Schafft die Grundlage für:

- Hypothesenbildung über erwartete Transfer-Schwierigkeiten
- Erklärung von Performance-Unterschieden in späteren Experimenten
- Dokumentation der Ausgangslage für die methodische Ausarbeitung

### 2.2 Key Questions

- Wie unterscheiden sich die **Klassenverteilungen** zwischen Berlin und Leipzig?
- Gibt es **phänologische Unterschiede** (saisonale Muster) zwischen den Städten?
- Sind die **CHM-Verteilungen** pro Genus konsistent?
- Welche **Features** zeigen die größten Stadt-Unterschiede?
- Ist die **Korrelationsstruktur** zwischen Features ähnlich?

### 2.3 Analyses & Visualizations

| Sektion                          | Visualisierung                            | Zweck                                             |
| -------------------------------- | ----------------------------------------- | ------------------------------------------------- |
| **1. Sample Overview**           | Stacked Bar: Genus × Stadt                | Klassenverteilung, Class Imbalance erkennen       |
| **2. Phänologische Profile**     | Line Plots: VI über Monate (Top-5 Genera) | Saisonale Muster Berlin vs. Leipzig vergleichen   |
| **3. CHM Verteilungen**          | Violin/Box Plots: CHM pro Genus × Stadt   | Höhenunterschiede zwischen Städten quantifizieren |
| **4. Feature Distributions**     | Ridge Plots: Key Features (NDVI, EVI, B8) | Verteilungsüberlappung visualisieren              |
| **5. Statistische Unterschiede** | Heatmap: Cohen's d pro Genus × Feature    | Effektstärken der Stadt-Unterschiede              |
| **6. Korrelationsstrukturen**    | Correlation Heatmaps (Berlin vs. Leipzig) | Strukturelle Ähnlichkeit der Feature-Räume        |

### 2.4 Expected Insights

Dieses Notebook ermöglicht Aussagen wie:

- _"Berlin hat 3× mehr TILIA-Samples als Leipzig, was das Transfer-Learning erschwert"_
- _"Die NDVI-Profile zeigen in Leipzig einen phänologischen Versatz von ~2 Wochen"_
- _"Cohen's d > 0.5 für CHM bei QUERCUS deutet auf standortbedingte Höhenunterschiede hin"_
- _"Die Korrelationsstrukturen sind hochgradig ähnlich (r > 0.95), was auf gutes Transfer-Potenzial hindeutet"_

### 2.5 Outputs

**Nur Visualisierungen** (keine JSON-Config, da deskriptiv):

```
outputs/phase_3/figures/exp_00_baseline/
├── genus_distribution_comparison.png
├── phenological_profiles_top5.png
├── chm_violin_per_genus.png
├── feature_ridge_plots.png
├── cohens_d_heatmap.png
└── correlation_structure_comparison.png
```

### 2.6 Notebook Location

```
notebooks/
└── experiments/
    └── exp_00_cross_city_baseline.ipynb
```

---

## 3. Experiment 3.1: Berlin Single-City Optimization

_To be defined_

**Goal:** Achieve best possible performance on Berlin using Train/Val/Test splits.

**Key Tasks:**

- Model comparison (RandomForest, XGBoost, LightGBM, ...)
- Hyperparameter optimization (Optuna)
- Ablation studies (CHM features, VI subsets, temporal subsets)
- Feature importance analysis

---

## 4. Experiment 3.2: Cross-City Zero-Shot Transfer

_To be defined_

**Goal:** Quantify performance drop when applying Berlin-trained model to Leipzig without fine-tuning.

**Key Tasks:**

- Apply best Berlin model to Leipzig test set
- Per-genus performance comparison
- Confusion matrix analysis
- Identify most/least transferable genera

---

## 5. Experiment 3.3: Fine-Tuning Experiments

_To be defined_

**Goal:** Determine how much Leipzig fine-tuning data is needed to recover performance.

**Key Tasks:**

- Fine-tuning with 10%, 25%, 50%, 100% of Leipzig finetune pool
- Learning curves: Performance vs. fine-tuning samples
- Compare fine-tuning strategies (full retrain vs. partial)
- Statistical significance of improvements

---

## 6. Methodology Notes

### 6.1 CHM Features

Based on legacy experiments, CHM features will likely be **excluded after ablation studies** in Experiment 3.1. Reasons:

- High correlation with cadastre height (contamination risk)
- City-specific CHM characteristics reduce transferability
- Sentinel-2 spectral features sufficient for genus classification

This decision will be **empirically validated**, not assumed.

### 6.2 Evaluation Metrics

- **Primary:** Weighted F1-Score (handles class imbalance)
- **Secondary:** Per-genus F1, Macro F1, Accuracy
- **Transfer-specific:** Relative performance drop (%), Per-genus transfer efficiency

---

## 7. Next Steps

1. Complete Phase 2 Feature Engineering (all PRD 002 notebooks)
2. Implement Experiment 3.0 (Cross-City Baseline Analysis)
3. Define detailed PRDs for Experiments 3.1-3.3

---

**Status:** Awaiting Phase 2 completion
