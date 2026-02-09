# Experimentelle Struktur: Übersicht

## Ziel der Experimente

Die Experimente in Phase 3 verfolgen das übergeordnete Forschungsziel:

> **Wie gut lassen sich Klassifikationsmodelle für Baumgattungen, die auf Berliner Daten trainiert wurden, auf Leipzig übertragen — und wie viele lokale Trainingsdaten werden benötigt, um eine akzeptable Performance zu erreichen?**

Diese Fragestellung ist für die praktische Anwendung hochrelevant: Städte mit existierenden Baumkatastern könnten als Trainingsquelle dienen, um Modelle für Städte mit geringerer Datenverfügbarkeit zu entwickeln.

---

## Experimentelle Phasen

Die Experimente gliedern sich in drei aufeinander aufbauende Phasen:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    PHASE 3: EXPERIMENT PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3.1-3.3: BERLIN-OPTIMIERUNG                              │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  • Setup-Entscheidungen (CHM, Proximity, Outlier, Feature-Selektion) │   │
│  │  • Algorithmenvergleich (RF, XGBoost, 1D-CNN, TabNet)           │   │
│  │  • Hyperparameter-Tuning der Champions                          │   │
│  │  • Berlin Upper Bound etablieren                                │   │
│  │                                                                  │   │
│  │  Output: Optimierte Modelle mit Berlin Test F1                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3.4: TRANSFER-EVALUATION                                 │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  • Zero-Shot Transfer nach Leipzig                              │   │
│  │  • Transfer-Gap quantifizieren                                  │   │
│  │  • Per-Genus Robustheitsanalyse                                 │   │
│  │  • ML vs. NN Transfer vergleichen                               │   │
│  │                                                                  │   │
│  │  Output: Transfer-Metriken, Best Transfer Model                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              │                                          │
│                              ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  PHASE 3.5: FINE-TUNING                                         │   │
│  │  ─────────────────────────────────────────────────────────────  │   │
│  │  • Fine-Tuning mit 10%, 25%, 50%, 100% Leipzig-Daten            │   │
│  │  • Sample Efficiency Curve                                       │   │
│  │  • Vergleich mit From-Scratch Baseline                          │   │
│  │  • Statistische Signifikanztests                                │   │
│  │                                                                  │   │
│  │  Output: Fine-Tuning Curve, Effizienz-Metriken                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Algorithmen

### Machine Learning Algorithmen

| Algorithmus       | Begründung                                                                               |
| ----------------- | ---------------------------------------------------------------------------------------- |
| **Random Forest** | Robuster Baseline, geringes Overfitting-Risiko, interpretierbar durch Feature Importance |
| **XGBoost**       | State-of-the-art für tabellarische Daten, oft beste Performance bei Kaggle-Wettbewerben  |

### Neuronale Netze

| Algorithmus | Begründung                                                                                      |
| ----------- | ----------------------------------------------------------------------------------------------- |
| **1D-CNN**  | Effizient für kurze temporale Sequenzen (12 Monate), erfasst lokale Muster wie Frühjahrsanstieg |
| **TabNet**  | Speziell für tabellarische Daten entwickelt, Attention-basiert, gute Interpretierbarkeit        |

### Champion-Selektion

Nach dem initialen Vergleich in **exp_11** (Algorithm Comparison) werden zwei "Champions" ausgewählt:

- **1 ML Champion** (RF oder XGBoost): Bestes ML-Modell nach Validation F1
- **1 NN Champion** (1D-CNN oder TabNet): Bestes NN nach Validation F1

**Aktuelle Auswahl:**

⚠️ **Status:** Noch nicht durchgeführt. Platzhalter für Ergebnisse nach exp_11-Ausführung.

- **ML Champion:** TBD
- **NN Champion:** TBD

Beide Champions werden in Phase 3.2 (03b) mit Hyperparameter-Tuning optimiert.

---

## Evaluationsmetriken

### Primärmetrik

**Weighted F1-Score** — Gewichtet nach Klassenhäufigkeit, robust bei Klassenimbalance

### Sekundärmetriken

| Metrik        | Zweck                                                              |
| ------------- | ------------------------------------------------------------------ |
| Macro F1      | Ungewichteter Durchschnitt, zeigt Performance auf seltenen Klassen |
| Accuracy      | Intuitive Gesamtperformance                                        |
| Per-Genus F1  | Detailanalyse pro Baumgattung                                      |
| Train-Val Gap | Overfitting-Indikator                                              |

### Transfer-spezifische Metriken

| Metrik               | Berechnung                                                    | Interpretation              |
| -------------------- | ------------------------------------------------------------- | --------------------------- |
| Absolute Drop        | F1_Berlin - F1_Leipzig                                        | Direkter Performanceverlust |
| Relative Drop        | (Drop / F1_Berlin) × 100                                      | Prozentualer Verlust        |
| Per-Genus Robustheit | Klassifikation nach Drop: <5% robust, 5-15% mittel, >15% poor |

---

## Cross-Validation Strategie

### Spatial Block CV

Da Bäume räumlich autokorreliert sind (benachbarte Bäume ähneln sich), verwenden wir **Spatial Block Cross-Validation**:

1. Stadtgebiet wird in 500m × 500m Blöcke aufgeteilt
2. Blöcke (nicht einzelne Bäume) werden den Folds zugewiesen
3. Verhindert Data Leakage durch räumliche Nähe

```
┌────────────────────────────────────────┐
│ Standard CV (FALSCH)                   │
│ ○ ● ○ ● ○ ● ○ ● ○ ●                   │
│ Bäume zufällig verteilt →              │
│ Nachbarbäume in Train UND Val          │
│ → Optimistische Schätzung              │
└────────────────────────────────────────┘

┌────────────────────────────────────────┐
│ Spatial Block CV (RICHTIG)             │
│ ┌───┐ ┌───┐ ┌───┐ ┌───┐               │
│ │ ● │ │ ○ │ │ ● │ │ ○ │               │
│ │ ● │ │ ○ │ │ ● │ │ ○ │               │
│ └───┘ └───┘ └───┘ └───┘               │
│ Ganze Blöcke in Train ODER Val         │
│ → Realistischere Schätzung             │
└────────────────────────────────────────┘
```

### Fold-Konfiguration

- **CV-Durchläufe (exp_08, exp_09, exp_10, exp_11):** 3-Fold Spatial Block CV
- **03b HP-Tuning:** 1-Fold Group Holdout (20%) auf einem 100k-Subset (Speed-Run)
- **Finale Evaluation (03b Test, 03c, 03d):** Hold-out Test Set (kein CV)

**Begründung:** 3-Fold bleibt Standard für methodisch robuste Entscheidungen. Für 03b
verwenden wir ein beschleunigtes Tuning (Subset + 1-Fold Holdout), um innerhalb
des Zeitfensters zu bleiben. Finale Modelle werden weiterhin auf Train+Val trainiert
und auf dem unabhängigen Test-Set bewertet.

---

## Hyperparameter-Tuning

### Methode: Optuna

Wir verwenden **Optuna** mit folgenden Einstellungen:

| Parameter | Wert                                   | Begründung                                              |
| --------- | -------------------------------------- | ------------------------------------------------------- |
| Sampler   | TPE (Tree-structured Parzen Estimator) | Effizienter als Random/Grid bei kontinuierlichen Räumen |
| Pruner    | MedianPruner                           | Bricht unpromising Trials früh ab                       |
| Trials    | 50+                                    | Ausreichend für Konvergenz                              |
| Timeout   | 2h pro Modell                          | Colab-Runtime-Limit beachten                            |

### Tuning-Ablauf

1. **Coarse Grid Search** (exp_11): Grobe Hyperparameter-Bereiche, wenige Kombinationen
2. **Fine Optuna Search** (03b): Präzise Suche im vielversprechendsten Bereich

---

## Datenaufteilung

### Berlin (Source City)

| Split      | Anteil | Verwendung                     |
| ---------- | ------ | ------------------------------ |
| Train      | 70%    | Modelltraining                 |
| Validation | 15%    | HP-Tuning, Early Stopping      |
| Test       | 15%    | Finale Evaluation (nur einmal) |

### Leipzig (Target City)

| Split    | Anteil | Verwendung                           |
| -------- | ------ | ------------------------------------ |
| Finetune | 80%    | Fine-Tuning Experimente              |
| Test     | 20%    | Transfer- und Fine-Tuning Evaluation |

---

## Konfidenzintervalle

Alle finalen Metriken werden mit **Bootstrap Confidence Intervals** berichtet:

```python
# Pseudocode
for i in range(1000):
    sample = resample(test_data, replace=True)
    metrics[i] = compute_f1(sample)

ci_lower = percentile(metrics, 2.5)
ci_upper = percentile(metrics, 97.5)
```

Dies ermöglicht Aussagen wie: "Berlin Test F1 = 0.62 [0.59, 0.65]"

---

## Notebook-Struktur

### Exploratory Notebooks

| Notebook                    | Zweck                                      | Abhängigkeiten   |
| --------------------------- | ------------------------------------------ | ---------------- |
| exp_07_cross_city_baseline  | Deskriptive Analyse Berlin vs. Leipzig     | Keine (optional) |
| exp_08_chm_ablation         | CHM-Strategie bestimmen                    | Keine            |
| exp_08b_proximity_ablation  | Baseline vs. Filtered Datensatz            | exp_08           |
| exp_08c_outlier_ablation    | Outlier-Removal-Strategie bestimmen        | exp_08b          |
| exp_09_feature_reduction    | Feature-Anzahl optimieren                  | exp_08c          |
| exp_10_genus_selection      | Genus-Auswahl validieren & gruppieren (JM) | exp_09           |
| exp_11_algorithm_comparison | 4 Algorithmen vergleichen (Coarse Grid)    | exp_10           |

### Runner Notebooks

| Notebook                | Zweck                       | Abhängigkeiten |
| ----------------------- | --------------------------- | -------------- |
| 03a_setup_fixation      | Datensätze vorbereiten      | exp_09, exp_10 |
| 03b_berlin_optimization | Champions HP-tunen          | 03a, exp_11    |
| 03c_transfer_evaluation | Zero-Shot Transfer messen   | 03b            |
| 03d_finetuning          | Fine-Tuning Curve erstellen | 03c            |

---

## Ausführungsreihenfolge

```
Critical Path:
─────────────

exp_08 ──→ exp_08b ──→ exp_08c ──→ exp_09 ──→ exp_10 ──→ exp_11 ──→ 03a
  │           │           │          │          │          │          │
  ▼           ▼           ▼          ▼          ▼          ▼          ▼
CHM      Proximity     Outlier   Features     Genus    Algorithm  Datasets
Decision  Decision    Decision   Selected   Selection  Comparison  Applied
                                              (JM)     (Champions)   │
                                               │          │         │
                                               └──────────┴─────────┘
                                                          │
                                                          ▼
                                                    setup_decisions.json
                                                    (alle Decisions)
                                                          │
                                                          ▼
                                                  03b ──→ 03c ──→ 03d
                                                  │       │       │
                                                  ▼       ▼       ▼
                                                Models  Transfer Finetune
                                                Tuned   Tested   Tested


Optional (parallel):
────────────────────

exp_07 (Cross-City Baseline Analysis)
```

---

## Erfolgskriterien

### Minimum Viable

- Berlin Validation F1 ≥ 0.55
- Train-Val Gap < 35%
- Transfer funktioniert (Leipzig F1 > 0.30)

### Target

- Berlin Validation F1 ≥ 0.60
- Berlin Test F1 ≥ 0.58
- Transfer Drop < 25%
- 25% Leipzig-Daten erreichen 90% der From-Scratch Performance

---

## Technische Infrastruktur

### Ausführungsumgebung

- **Google Colab** für GPU-Training (NN)
- **Google Drive** für persistente Datenspeicherung
- **GitHub** für Versionskontrolle des Codes

### Workflow

1. Code lokal entwickeln → GitHub pushen
2. Colab Notebook lädt Code von GitHub
3. Ergebnisse auf Google Drive speichern
4. Metadaten/Figures lokal synchronisieren → GitHub committen

---

_Letzte Aktualisierung: 2026-02-03_
