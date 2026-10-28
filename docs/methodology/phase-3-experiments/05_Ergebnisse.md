# 05_Ergebnisse

> Migriert aus Notion am 2026-10-28 (Seite `31f356451f668093ad57c4304d08e185`, zuletzt bearbeitet 2026-03-25). Text unverändert übernommen.

# Phase 3: Experiments, Ergebnisse

**Phase:** Experiments

**Ausführungszeitraum:** 07.02.2026 bis 10.02.2026

**Status:** ✅ **Abgeschlossen**

**Pipeline-Rerun:** 24.03.2026

---

## Finale Konfiguration

| Parameter | Konfiguration | Begründung |
| --- | --- | --- |
| **CHM** | Keine CHM-Features | Destabilisiert Generalisierung (+9 bis 13pp Gap) |
| **Datensatz** | Filtered (≥5m Proximity) | +2.7pp F1 trotz 21% Datenverlust |
| **Outliers** | Keine Removal | Kein messbarer Nutzen (\<0.1pp) |
| **Features** | Top-50 | Pareto-Optimum F1/Gap |
| **Klassen** | 17 (13 Einzelgenera + 4 Gruppen) | JM-Separabilität > 1.37, 98.2% Retention |

**Finaler Datensatz:**

- **Berlin Train:** 463.654 Bäume, 50 Features, 17 Klassen
- **Berlin Val:** 97.972 Bäume
- **Berlin Test:** 106.758 Bäume (ungefiltert, Real-World)
- **Leipzig Finetune:** 89.335 Bäume
- **Leipzig Test:** 30.480 Bäume (ungefiltert, Real-World)

---

## Setup Decisions (Ablationsstudien)

Alle Ablationen: 3-Fold Spatial Block CV (1200m), Random Forest als Basisalgorithmus.

### CHM-Strategie

| Variante | Val F1 | Train-Val Gap | Entscheidung |
| --- | --- | --- | --- |
| **no\_chm** | **0.377** | 0.489 | ✅ **Gewählt** |
| raw\_chm | 0.385 | 0.582 | ❌ Gap +9.3pp |
| zscore\_only | 0.377 | 0.623 | ❌ Gap +13.4pp |
| percentile\_only | 0.375 | 0.625 | ❌ Gap +13.6pp |
| both\_engineered | 0.378 | 0.622 | ❌ Gap +13.3pp |

### Proximity-Filter

| Variante | Val F1 | Train-Val Gap | Samples | Entscheidung |
| --- | --- | --- | --- | --- |
| baseline | 0.377 | 0.489 | 600.299 | ❌ |
| **filtered** | **0.404** | 0.555 | 471.815 | ✅ **Gewählt** |

### Outlier-Removal

| Variante | Val F1 | Train-Val Gap | Entscheidung |
| --- | --- | --- | --- |
| **no\_removal** | **0.404** | 0.555 | ✅ **Gewählt** |
| remove\_high | 0.403 | 0.556 | ❌ |
| remove\_high\_medium | 0.401 | 0.557 | ❌ |

### Feature-Anzahl

| Features | Val F1 | Train-Val Gap | Entscheidung |
| --- | --- | --- | --- |
| 30 | 0.380 | 0.578 | ❌ Unterfitted |
| **50** | **0.407** | 0.552 | ✅ **Gewählt** |
| 80 | 0.409 | 0.550 | ❌ Marginaler Gewinn |
| all (144) | 0.403 | 0.556 | ❌ Overfitting-Risiko |

---

## Genus Selection

**Finale 17 Klassen:** AESCULUS, AILANTHUS, ALNUS, CARPINUS, CORYLUS, FAGUS, Group 1, Group 9, Group 11, Group 14, MALUS, PLATANUS, PYRUS, QUERCUS, SALIX, TILIA, ULMUS

**Gruppen:**

| Gruppe | Mitglieder | Biologische Basis |
| --- | --- | --- |
| **Group 14** | ACER, BETULA, FRAXINUS, PRUNUS | Feinblättrige Laubbäume, ähnliche spektrale Textur |
| **Group 11** | POPULUS, ROBINIA | Pionierarten, ähnliche Vitalitätssignaturen |
| **Group 9** | CRATAEGUS, SORBUS | Rosaceae, ähnliche Phänologie |
| **Group 1** | PICEA, PINUS | Nadelbäume (Pinaceae) |

**JM-Statistiken:** Min 0.729, P10 1.373, Median 1.812, Max 1.998

**Sample-Counts pro Klasse:**

| Klasse | Berlin | Leipzig |
| --- | --- | --- |
| Group 14 | 156.377 | 32.526 |
| TILIA | 104.215 | 21.904 |
| QUERCUS | 58.300 | 8.349 |
| Group 11 | 29.007 | 4.930 |
| PLATANUS | 19.939 | 4.396 |
| AESCULUS | 18.691 | 3.285 |
| CARPINUS | 15.604 | 3.290 |
| Group 1 | 14.089 | 1.116 |
| FAGUS | 8.680 | 637 |
| Group 9 | 8.090 | 1.860 |
| ULMUS | 7.961 | 661 |
| CORYLUS | 5.847 | 1.012 |
| SALIX | 5.277 | 1.504 |
| ALNUS | 4.539 | 1.047 |
| MALUS | 3.647 | 1.015 |
| PYRUS | 1.953 | 1.391 |
| AILANTHUS | 1.438 | 412 |

---

## Algorithm Comparison (exp\_11)

| Modell | Weighted F1 | Std | Train-Val Gap |
| --- | --- | --- | --- |
| Majority | 0.182 | 0.000 | 0.000 |
| Stratified Random | 0.188 | 0.000 | 0.000 |
| Random Forest | 0.475 | 0.006 | 0.398 |
| **XGBoost** | **0.510** | 0.005 | 0.138 |
| **CNN-1D** | **0.567** | 0.009 | 0.047 |

**Auswahlentscheidung:** XGBoost (ML Champion) und CNN-1D (NN Champion) für Hyperparameter-Tuning.

### Beste Hyperparameter (nach Tuning)

**XGBoost Champion:**

```py
{
  "n_estimators": 358,
  "max_depth": 12,
  "learning_rate": 0.054,
  "subsample": 0.851,
  "colsample_bytree": 0.905,
  "reg_lambda": 1.358,
  "reg_alpha": 0.077,
  "min_child_weight": 2
}
```

**CNN-1D Champion:**

```py
{
  "conv_filters": [64, 128, 128],
  "kernel_size": 5,
  "dropout": 0.419,
  "dense_units": [128, 64],
  "learning_rate": 0.001,
  "batch_size": 64,
  "epochs": 60,
  "early_stopping_patience": 8
}
```

⚠️ **NOT FOUND:** Bootstrap 95% CIs pro Modell

---

## Berlin Optimization (03b)

### Gesamt-Metriken

| Metrik | XGBoost | CNN-1D |
| --- | --- | --- |
| **CV F1** | 0.540 | 0.554 |
| **Test F1** | **0.751** | **0.607** |
| Test Precision | 0.762 | 0.631 |
| Test Recall | 0.752 | 0.633 |
| Test Accuracy | 0.752 | 0.633 |

**Methodische Änderung (Pipeline-Rerun 24.03.2026):** XGBoost nutzt ab diesem Rerun `sample_weight` zur Kompensation der Klassenimbalance (Vor-Rerun ohne Class Balancing: F1 = 0.571, z.B. ULMUS Recall 3.6%, CARPINUS Recall 7.9%). CNN-1D unverändert (+0.5pp).

### Per-Genus Metriken (XGBoost, Berlin Test)

| Genus | Precision | Recall | F1 |
| --- | --- | --- | --- |
| PLATANUS | 0.792 | 0.922 | **0.852** |
| TILIA | 0.847 | 0.831 | **0.839** |
| AESCULUS | 0.704 | 0.851 | 0.770 |
| Group 14 | 0.824 | 0.664 | 0.735 |
| Group 1 | 0.687 | 0.860 | 0.764 |
| QUERCUS | 0.694 | 0.775 | 0.732 |
| CORYLUS | 0.638 | 0.855 | 0.731 |
| ALNUS | 0.729 | 0.717 | 0.723 |
| PYRUS | 0.712 | 0.729 | 0.720 |
| ULMUS | 0.804 | 0.637 | 0.711 |
| CARPINUS | 0.683 | 0.716 | 0.699 |
| Group 11 | 0.644 | 0.742 | 0.690 |
| SALIX | 0.652 | 0.730 | 0.689 |
| FAGUS | 0.669 | 0.678 | 0.673 |
| MALUS | 0.601 | 0.744 | 0.665 |
| AILANTHUS | 0.670 | 0.597 | 0.632 |
| Group 9 | 0.586 | 0.651 | 0.617 |

⚠️ **NOT FOUND:** CNN-1D Per-Genus Berlin-Metriken

### Top-20 Feature Importances (XGBoost)

| Rank | Feature | Importance |
| --- | --- | --- |
| 1 | MSI\_07 | 0.036 |
| 2 | NDVI\_06 | 0.034 |
| 3 | B3\_06 | 0.034 |
| 4 | VARI\_04 | 0.029 |
| 5 | VARI\_06 | 0.027 |
| 6 | EVI\_09 | 0.027 |
| 7 | B4\_06 | 0.027 |
| 8 | EVI\_06 | 0.026 |
| 9 | NDVI\_10 | 0.024 |
| 10 | NDVI\_05 | 0.024 |
| 11 | NDWI\_09 | 0.024 |
| 12 | B8A\_04 | 0.024 |
| 13 | NDWI\_07 | 0.023 |
| 14 | VARI\_10 | 0.022 |
| 15 | B3\_04 | 0.022 |
| 16 | NDVI\_09 | 0.022 |
| 17 | B5\_04 | 0.020 |
| 18 | B11\_06 | 0.020 |
| 19 | NDVI\_04 | 0.020 |
| 20 | VARI\_05 | 0.020 |

⚠️ **NOT FOUND:** RF Feature Importances

### Top-5 häufigste Verwechslungen (XGBoost, Berlin Test)

| True → Pred | Absolut | Normalisiert |
| --- | --- | --- |
| Group 14 → QUERCUS | 2.937 | 0.083 |
| Group 14 → Group 11 | 1.806 | 0.051 |
| Group 14 → TILIA | 1.702 | 0.048 |
| Group 14 → Group 1 | 1.701 | 0.048 |
| TILIA → Group 14 | 1.099 | 0.059 |

---

## Transfer Evaluation (03c)

### Gesamt-Metriken (Zero-Shot)

| Champion | Berlin Test F1 | Leipzig Test F1 | Abs. Drop | Rel. Drop |
| --- | --- | --- | --- | --- |
| **XGBoost** | 0.751 | 0.377 | 0.374 | 49.8% |
| **CNN-1D** | 0.607 | 0.380 | 0.227 | 37.4% |

### Per-Genus Transfer (XGBoost)

| Genus | Berlin F1 | Leipzig F1 | Abs. Drop | Rel. Drop |
| --- | --- | --- | --- | --- |
| TILIA | 0.839 | 0.541 | 0.298 | 0.355 |
| Group 14 | 0.735 | 0.502 | 0.233 | 0.317 |
| PLATANUS | 0.852 | 0.403 | 0.449 | 0.527 |
| AESCULUS | 0.770 | 0.291 | 0.480 | 0.623 |
| QUERCUS | 0.732 | 0.239 | 0.493 | 0.674 |
| Group 1 | 0.764 | 0.236 | 0.529 | 0.692 |
| Group 11 | 0.690 | 0.168 | 0.521 | 0.756 |
| FAGUS | 0.673 | 0.155 | 0.519 | 0.770 |
| CARPINUS | 0.699 | 0.135 | 0.564 | 0.806 |
| SALIX | 0.689 | 0.115 | 0.573 | 0.833 |
| MALUS | 0.665 | 0.079 | 0.586 | 0.882 |
| AILANTHUS | 0.632 | 0.051 | 0.581 | 0.919 |
| Group 9 | 0.617 | 0.032 | 0.584 | 0.947 |
| ALNUS | 0.723 | 0.016 | 0.707 | 0.978 |
| CORYLUS | 0.731 | 0.009 | 0.722 | 0.988 |
| ULMUS | 0.711 | 0.009 | 0.702 | 0.988 |
| PYRUS | 0.720 | 0.007 | 0.713 | 0.991 |

⚠️ **NOT FOUND:** CNN-1D Per-Genus Transfer-Metriken

**Conifer vs. Deciduous (Mean ΔF1):** Conifer 0.529, Deciduous 0.545

**Robustheitskategorie:** Alle 17 Klassen → **Poor**

⚠️ **NOT FOUND:** Bootstrap 95% CIs, Mann-Whitney U (JSON: "Insufficient data for Mann-Whitney U test")

---

## Hypothesis Tests

| Hypothese | Test | Statistik | p-Wert | Effekt | Ergebnis |
| --- | --- | --- | --- | --- | --- |
| **H1** | Kruskal-Wallis | 4953.409 | 0.000 | η² = 0.163 | ✅ **Bestätigt** |
| **H2** | Mann-Whitney U | NOT FOUND | NOT FOUND | NOT FOUND | ⚠️ **Inconclusive** |
| **H3** | Spearman ρ | −0.132 | 0.613 | ρ = −0.132 | ❌ **Verworfen** |
| **H4** | Spearman ρ | 0.500 | 0.117 | ρ = 0.500 | ❌ **Verworfen** |

---

## Fine-Tuning (03d)

### XGBoost Finetuning Curve

| Fraktion | F1 Score | 95% CI | Δ vs. Zero-Shot |
| --- | --- | --- | --- |
| 0% (Zero-Shot) | 0.377 |  | Baseline |
| 10% | 0.499 | \[0.493, 0.505\] | +12.2pp |
| 25% | 0.574 | \[0.568, 0.580\] | +19.7pp |
| 50% | 0.662 | \[0.656, 0.667\] | +28.5pp |
| 100% | **0.771** | \[0.766, 0.776\] | +39.4pp |

### CNN-1D Finetuning Curve

| Fraktion | F1 Score | 95% CI |
| --- | --- | --- |
| 0% (Zero-Shot) | 0.380 |  |
| 10% | 0.450 | \[0.444, 0.457\] |
| 25% | 0.455 | \[0.449, 0.461\] |
| 50% | 0.458 | \[0.452, 0.465\] |
| 100% | 0.458 | \[0.451, 0.464\] |

CNN-1D zeigt keine signifikante Verbesserung über die Fraktionen hinweg: Plateauierung ab 10%.

### From-Scratch Baseline

| Modell | From-Scratch F1 (100% Leipzig) | Finetuned F1 (100%) | Differenz |
| --- | --- | --- | --- |
| XGBoost | 0.786 | 0.771 | −1.5pp |
| CNN-1D | NOT FOUND | 0.458 |  |

**Fraction to Match Scratch (XGBoost):** NOT REACHED

**Fraction to 90% of Scratch (XGBoost):** 100% (Zielwert 0.707)

### McNemar Tests

- finetuning\_vs\_zeroshot: **p = 0.000** ✅ Signifikant
- finetuning\_vs\_fromscratch: **p = 0.000** ✅ Signifikant
- Weitere Paare: NOT FOUND

### Power-Law Fit

- **XGBoost Exponent b:** 0.194
- **CNN-1D:** a = 0.423, b = 0.007, R² = 0.877

### Per-Genus F1 pro Finetuning-Fraktion (XGBoost)

| Genus | 10% | 25% | 50% | 100% |
| --- | --- | --- | --- | --- |
| AESCULUS | 0.458 | 0.558 | 0.679 | 0.801 |
| AILANTHUS | 0.128 | 0.325 | 0.473 | 0.674 |
| ALNUS | 0.112 | 0.378 | 0.500 | 0.637 |
| CARPINUS | 0.306 | 0.402 | 0.533 | 0.716 |
| CORYLUS | 0.102 | 0.247 | 0.433 | 0.575 |
| FAGUS | 0.316 | 0.441 | 0.603 | 0.807 |
| Group 1 | 0.365 | 0.441 | 0.564 | 0.686 |
| Group 9 | 0.136 | 0.258 | 0.448 | 0.623 |
| Group 11 | 0.329 | 0.415 | 0.550 | 0.686 |
| Group 14 | 0.616 | 0.670 | 0.725 | 0.804 |
| MALUS | 0.240 | 0.326 | 0.419 | 0.577 |
| PLATANUS | 0.554 | 0.643 | 0.729 | 0.856 |
| PYRUS | 0.135 | 0.224 | 0.412 | 0.575 |
| QUERCUS | 0.385 | 0.483 | 0.589 | 0.728 |
| SALIX | 0.238 | 0.345 | 0.490 | 0.697 |
| TILIA | 0.624 | 0.683 | 0.749 | 0.835 |
| ULMUS | 0.170 | 0.282 | 0.409 | 0.574 |

⚠️ **NOT FOUND:** CNN-1D Per-Genus Fine-Tuning

---

## Fehlende / nicht extrahierbare Daten

**Phase 2 (aus vorheriger Extraktion):**

- Welch-F und p-Werte pro Genus in chm\_assessment.json
- p-Werte pro Lag in morans\_i\_decay.json

**Phase 3:**

- Bootstrap 95% CIs in algorithm\_comparison.json
- RF Feature Importances in feature\_importances.json
- CNN-1D Per-Genus Berlin-Metriken
- CNN-1D Per-Genus Transfer-Metriken
- Bootstrap 95% CIs für Transfer-Metriken
- CNN-1D From-Scratch Baseline
- CNN-1D Per-Genus Fine-Tuning

---

**Letzte Aktualisierung:** 24.03.2026
