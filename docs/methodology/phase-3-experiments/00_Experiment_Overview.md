# 00_Experiment_Overview

> Migriert aus Notion am 2026-10-28 (Seite `31f356451f66801194aef3cc0b8f16ad`, zuletzt bearbeitet 2026-03-12). Text unverändert übernommen.

# Experimentelle Struktur: Übersicht

## Ziel der Experimente

> **Wie gut lassen sich Klassifikationsmodelle für Baumgattungen, die auf Berliner Daten trainiert wurden, auf Leipzig übertragen — und wie viele lokale Trainingsdaten werden benötigt, um eine akzeptable Performance zu erreichen?**

Städte mit existierenden Baumkatastern könnten als Trainingsquelle dienen, um Modelle für Städte mit geringerer Datenverfügbarkeit zu entwickeln.

---

## Experimentelle Phasen

1. **Phase 3.1–3.3: Berlin-Optimierung** — Setup-Entscheidungen (CHM, Proximity, Outlier, Feature-Selektion), Algorithmenvergleich (RF, XGBoost, 1D-CNN, TabNet), Hyperparameter-Tuning der Champions → Berlin Upper Bound
2. **Phase 3.4: Transfer-Evaluation** — Zero-Shot Transfer nach Leipzig, Transfer-Gap quantifizieren, Per-Genus Robustheitsanalyse, ML vs. NN Transfer-Vergleich
3. **Phase 3.5: Fine-Tuning** — Fine-Tuning mit 10%, 25%, 50%, 100% Leipzig-Daten, Sample Efficiency Curve, From-Scratch Baseline, statistische Signifikanztests

---

## Algorithmen

| Kategorie | Algorithmus | Begründung |
| --- | --- | --- |
| ML | **Random Forest** | Robuster Baseline, geringes Overfitting, interpretierbar durch Feature Importance |
| ML | **XGBoost** | State-of-the-art für tabellarische Daten |
| NN | **1D-CNN** | Effizient für kurze temporale Sequenzen (12 Monate), erfasst lokale Muster |
| NN | **TabNet** | Speziell für tabellarische Daten, Attention-basiert |

**Champion-Selektion:** Nach initialem Vergleich werden **1 ML-Champion** und **1 NN-Champion** ausgewählt und per Hyperparameter-Tuning optimiert.

---

## Evaluationsmetriken

**Primärmetrik:** Weighted F1-Score — gewichtet nach Klassenhäufigkeit, robust bei Klassenimbalance.

**Sekundärmetriken:** Macro F1, Accuracy, Per-Genus F1, Train-Val Gap (Overfitting-Indikator)

| Transfer-Metrik | Berechnung | Interpretation |
| --- | --- | --- |
| Absolute Drop | F1\_Berlin − F1\_Leipzig | Direkter Performanceverlust |
| Relative Drop | (Drop / F1\_Berlin) × 100 | Prozentualer Verlust |
| Per-Genus Robustheit | \<5% robust, 5–15% mittel, >15% poor | Genus-spezifische Transferierbarkeit |

---

## Cross-Validation Strategie

**Spatial Block CV:** Stadtgebiet in Blöcke aufgeteilt, Blöcke (nicht einzelne Bäume) den Folds zugewiesen → verhindert Data Leakage durch räumliche Nähe.

| Phase | CV-Strategie | Begründung |
| --- | --- | --- |
| Setup-Ablationen (exp\_08–11) | 3-Fold Spatial Block CV | Methodisch robuste Entscheidungen |
| HP-Tuning (03b) | 1-Fold Group Holdout (20%), 100k-Subset | Beschleunigtes Tuning |
| Finale Evaluation | Hold-out Test Set | Unabhängige Bewertung |

3-Fold für methodisch robuste Entscheidungen. Beschleunigtes Tuning mit Subset + 1-Fold Holdout innerhalb des Zeitfensters. Finale Modelle auf Train+Val trainiert, auf unabhängigem Test-Set bewertet.

---

## Hyperparameter-Tuning (Optuna)

| Parameter | Wert | Begründung |
| --- | --- | --- |
| Sampler | TPE | Effizienter als Random/Grid bei kontinuierlichen Räumen |
| Pruner | MedianPruner | Bricht unpromising Trials früh ab |
| Trials | 50+ | Ausreichend für Konvergenz |

**Ablauf:** Coarse Grid Search (exp\_11) → Fine Optuna Search (03b)

---

## Datenaufteilung

| Stadt | Split | Anteil | Verwendung |
| --- | --- | --- | --- |
| Berlin (Source) | Train | 70% | Modelltraining |
| Berlin | Validation | 15% | HP-Tuning, Early Stopping |
| Berlin | Test | 15% | Finale Evaluation (nur einmal) |
| Leipzig (Target) | Finetune | 80% | Fine-Tuning (in 03d weiter in Train/Val gesplittet) |
| Leipzig | Test | 20% | Transfer- und Fine-Tuning Evaluation |

Alle finalen Metriken mit **Bootstrap Confidence Intervals** (1000 Resamples, 95% CI).

---

## Erfolgskriterien

| Level | Kriterium |
| --- | --- |
| **Minimum** | Berlin Val F1 ≥ 0.55, Train-Val Gap \< 35%, Leipzig F1 > 0.30 |
| **Target** | Berlin Val F1 ≥ 0.60, Test F1 ≥ 0.58, Transfer Drop \< 25%, 25% Leipzig-Daten → 90% From-Scratch |
