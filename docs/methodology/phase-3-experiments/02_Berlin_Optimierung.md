# 02_Berlin_Optimierung

# Phase 3.2–3.3: Berlin-Optimierung

## Einleitung

Mit fixiertem Setup (CHM, Datensatz, Outlier, Features) vergleichen wir vier Algorithmen und optimieren die beiden Champions durch Hyperparameter-Tuning. Ziel: **Upper Bound** — bestmögliche Performance auf Berliner Daten als Referenz für Transfer-Experimente.

---

## Forschungsfragen

1. Welcher ML- und welcher NN-Algorithmus performt am besten?
2. Wie viel Verbesserung bringt Hyperparameter-Tuning gegenüber Default-Parametern?
3. Wie hoch ist die Berlin Upper Bound?

---

## Phase 3.2: Algorithmenvergleich

| Kategorie | Algorithmen | Coarse Grid | Features |
| --- | --- | --- | --- |
| ML | Random Forest, XGBoost | 24–48 Configs | 50 reduzierte Features |
| NN | 1D-CNN, TabNet | Baseline + Varianten | \~144 volle temporale Features |

ML- und NN-Algorithmen verwenden automatisch die passende Datensatz-Variante (reduziert bzw. voll temporal).

**TabNet-Ergebnis:** TabNet wurde im explorativen Algorithmenvergleich (exp\_11) evaluiert. Ergebnis: val F1 = 0.546, Train-Val Gap = 20.5%. CNN-1D war überlegen (val F1 = 0.567, Gap = 4.7%). TabNet daher nicht als NN-Champion selektiert. Hauptgrund: deutlich höheres Overfitting bei gleichzeitig niedrigerem F1.

**Class Balancing (XGBoost):** `sample_weight` proportional zur inversen Klassenhäufigkeit. Kompensiert die starke Imbalance (dominante Klassen wie Group 14 vs. seltene wie AILANTHUS).

### Naive Baselines

Drei Baselines **ohne** Sentinel-2 Features zur absoluten Performance-Einordnung:

1. **Majority Class Classifier:** Immer häufigste Gattung → untere Grenze
2. **Stratified Random Classifier:** Zufällig gewichtet nach Klassenverteilung → Chance-Level
3. **Spatial-Only RF:** Nur x/y-Koordinaten (UTM) → testet ob räumliche Autokorrelation allein ausreicht

Falls Spatial-Only RF > 0.20 → Hinweis auf starke räumliche Autokorrelation.

### Coarse Grid Search

24–48 Configs pro Algorithmus statt 100+ → identifiziert beste Region effizient bei 3-Fold Spatial Block CV. Fine-Tuning in Phase 3.3 via Optuna.

### Champion-Selektion

1. Minimum F1 ≥ 0.50
2. Train-Val Gap \< 35%
3. Bestes ML und bestes NN **separat** → ermöglicht ML vs. NN Transfer-Vergleich in Phase 3.4

---

## Phase 3.3: Hyperparameter-Optimierung

| Parameter | Wert | Begründung |
| --- | --- | --- |
| Sampler | TPE | Nutzt bisherige Trials für effiziente Suche |
| Pruner | MedianPruner | Bricht Trials unter Median ab |
| Tuning-Subset | 100k | Repräsentative Teilmenge |
| n\_trials | 40 (XGBoost) / 20 (CNN-1D) | Optuna TPE-Trials |
| CV | 3-Fold Spatial Block CV (1200m) | Räumliche Unabhängigkeit der Folds |

**Final Training:** Beste HP aus Optuna → Modell auf **Train + Validation** trainiert (Validation war nur für HP-Selektion nötig; mehr Daten → besseres Modell) → einmalige Evaluation auf Hold-Out Test.

---

## Post-Training Fehleranalyse

Umfassende Fehleranalyse nach finalem Training. Alle Gattungsnamen als **deutsche Namen**. Alle Metriken mit **Bootstrap CI** (1000 Resamples, 95% CI). Naive Baselines ebenfalls mit CI evaluiert für statistischen Vergleich.

### Analysekategorien

- **Konfusionsmatrix & Fehlermuster:** Normalisierte Matrix, Top-10 Verwechslungspaare, Precision/Recall/F1 pro Gattung
- **Nadel- vs. Laubbäume:** Hypothese: Nadelbäume mit distinktiverem Spektralprofil (immergrün) → höherer F1
- **Straßen- vs. Anlagenbäume:** Berlin unterscheidet Straßenbäume und Anlagenbäume (Parks) → unterschiedliches Spektralsignal?
- **Pflanzjahr-Analyse:** Gruppiert in Dekaden. Hypothese: Jüngere Bäume (kleinere Krone) → weniger dominantes Signal → niedrigere Genauigkeit
- **Artanalyse für problematische Gattungen:** Für Gattungen mit F1 \< 0.50: Art-aufgelöste Analyse (z.B. *A. platanoides* vs. *A. pseudoplatanus*)
- **CHM-Einfluss:** Klassifikationsgenauigkeit nach Baumhöhen-Bins
- **Räumliche Fehlerverteilung:** Vorhersagegenauigkeit pro Block als Kartenvisualisierung

Diese Analysen werden als wiederverwendbare Funktionen implementiert für 03c (Transfer) und 03d (Fine-Tuning).

---

## Erwartete Ergebnisse

### Naive Baselines

| Baseline | Erwarteter F1 | Begründung |  |
| --- | --- | --- | --- |
| Majority Class | 0.01–0.03 | \~10 Klassen |  |
| Stratified Random | 0.02–0.05 | Minimal besser |  |
| Spatial-Only RF | 0.10–0.20 | Falls räumliche Cluster vorhanden |  |

### Literatur-Kontext

| Studie | Klassen | Daten | F1/Accuracy |
| --- | --- | --- | --- |
| Schiefer et al. (2020) | 7 Arten | Hyperspektral | \~0.75 |
| Hartling et al. (2019) | 5 Gattungen | S2 + LiDAR | \~0.65 |
| Immitzer et al. (2016) | 10 Arten | S2 | \~0.55 |

**Target** von 0.60 realistisch für \~10 Gattungen mit S2 + CHM.
