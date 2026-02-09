# Phase 3.2-3.3: Berlin-Optimierung

## Einleitung

Die Berlin-Optimierung baut auf den Setup-Entscheidungen aus Phase 3.1 auf. Mit fixiertem CHM, Datensatz, Outlier-Strategie und Feature-Set vergleichen wir vier Algorithmen und optimieren die beiden besten (Champions) durch Hyperparameter-Tuning. Das Ziel ist die **Upper Bound** — die bestmögliche Performance auf Berliner Daten als Referenz für Transfer-Experimente.

---

## Forschungsfragen

1. **Algorithmen:** Welcher ML- und welcher NN-Algorithmus performt am besten?
2. **Optimierung:** Wie viel Verbesserung bringt Hyperparameter-Tuning gegenüber Default-Parametern?
3. **Berlin Upper Bound:** Wie hoch ist die maximal erreichbare Performance auf Berlin?

---

## Phase 3.2: Algorithmenvergleich

### Experimentelles Design

Mit fixiertem Setup (CHM-Strategie + Datensatzwahl + Outlier-Strategie + selektierte Features) vergleichen wir:

| Kategorie | Algorithmen            | Coarse Grid Configs         | Features                      |
| --------- | ---------------------- | --------------------------- | ----------------------------- |
| ML        | Random Forest, XGBoost | 24-48 pro Algorithmus       | 50 reduzierte Features        |
| NN        | 1D-CNN, TabNet         | Baseline + wenige Varianten | ~144 volle temporale Features |

**🔄 Dataset-Selektion:** Algorithmen verwenden automatisch die passende Datensatz-Variante:

- ML-Algorithmen: `load_berlin_splits()` → `berlin_train.parquet` (reduzierte Features)
- NN-Algorithmen: `load_berlin_splits_cnn()` → `berlin_train_cnn.parquet` (volle Features)

### Naive Baselines (Improvement 2)

**Zweck:** Absolute Performance-Einordnung durch Etablierung einer unteren Schranke.

Neben ML- und NN-Algorithmen evaluieren wir drei **Naive Baselines**, die KEINE Sentinel-2 Features nutzen:

#### 1. Majority Class Classifier

- **Strategie:** Immer die häufigste Gattung vorhersagen
- **Erwartete Performance:** F1 ≈ 1 / (Anzahl Klassen) bei balanciertem Datensatz
- **Zweck:** Absolute untere Grenze

#### 2. Stratified Random Classifier

- **Strategie:** Zufällige Vorhersagen gewichtet nach Klassenverteilung
- **Erwartete Performance:** Minimal besser als Majority Class
- **Zweck:** Chance-Level Performance

#### 3. Spatial-Only Random Forest

- **Features:** Nur x/y-Koordinaten (UTM), KEIN Sentinel-2, KEIN CHM
- **Erwartete Performance:** F1 ≈ 0.10-0.20 (falls räumliche Cluster existieren)
- **Zweck:** Test ob räumliche Autokorrelation allein ausreicht

**Performance Ladder:**

```
Majority Class < Stratified Random < Spatial-Only RF << ML/NN mit S2+CHM
```

Diese Baselines werden auf demselben Berlin Test Set evaluiert wie die Champions.

### Coarse Grid Search Strategie

**Warum Coarse statt Fine Search?**

```
Fine Search (verworfen):           Coarse Search (gewählt):
─────────────────────────          ──────────────────────────
• 100+ Configs pro Algo            • 24-48 Configs pro Algo
• Nur 3-Fold CV möglich            • 3-Fold CV ausreichend
• Zu zeitaufwendig                 • Identifiziert beste Region
• Overfitting auf Val-Set          • Fine-Tuning in Phase 3.3
```

**Beispiel Random Forest Coarse Grid:**

```python
{
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2]
}
# 3 × 3 × 2 × 2 = 36 Kombinationen → reduziert auf 24 wichtigste
```

### Champion-Selektion

#### Selektionskriterien

1. **Minimum F1 ≥ 0.50:** Filterung nicht-funktionierender Konfigurationen
2. **Train-Val Gap < 35%:** Ausschluss stark overfittender Modelle
3. **Innerhalb Kategorie:** Bestes ML und bestes NN separat wählen

#### Warum zwei Champions (ML + NN)?

```
Szenario: Nur bestes Modell insgesamt
────────────────────────────────────
• Wenn XGBoost gewinnt → Kein NN-Transfer-Vergleich
• Verpassen wir: "Transferieren NNs besser als ML?"

Szenario: Bestes pro Kategorie (gewählt)
────────────────────────────────────────
• ML Champion (RF oder XGBoost)
• NN Champion (1D-CNN oder TabNet)
• Ermöglicht: ML vs. NN Transfer-Vergleich
```

---

## Phase 3.3: Hyperparameter-Optimierung

### Optuna-Konfiguration

| Parameter         | Wert                       | Begründung                                                                 |
| ----------------- | -------------------------- | -------------------------------------------------------------------------- |
| **Sampler**       | TPE                        | Effizienter als Random bei kontinuierlichen Räumen, nutzt bisherige Trials |
| **Pruner**        | MedianPruner               | Bricht Trials ab, die unter Median der bisherigen liegen                   |
| **n_trials**      | 10–15                      | Schneller Suchlauf, Ziel: < 1–2h Runtime                                   |
| **timeout**       | 1h                         | Zeitfenster für pragmatischen Speed-Run                                    |
| **Tuning-Subset** | 100k                       | Repräsentative Teilmenge für schnellere Trials                             |
| **CV (Tuning)**   | 1-Fold Group Holdout (20%) | Schnelle Schätzung, kein Voll-CV                                           |

**Hinweis:** Finales Training erfolgt weiterhin auf Train+Val (voller Datensatz).

### Optuna Search Space (Beispiel XGBoost)

```python
{
    "n_estimators": {"type": "int", "low": 200, "high": 400},
    "max_depth": {"type": "int", "low": 4, "high": 8},
    "learning_rate": {"type": "float", "low": 0.05, "high": 0.2},
    "subsample": {"type": "float", "low": 0.8, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.8, "high": 1.0},
    "reg_lambda": {"type": "float", "low": 1.0, "high": 2.0},
    "reg_alpha": {"type": "float", "low": 0.0, "high": 0.5},
    "min_child_weight": {"type": "int", "low": 1, "high": 5}
}
```

### Final Training

Nach HP-Tuning:

```
1. Beste Hyperparameter aus Optuna extrahieren
2. Modell auf Train + Validation trainieren (mehr Daten)
3. Auf Hold-Out Test evaluieren (EINMALIG)
4. Modell speichern für Transfer-Phase
```

**Warum Train + Val für Final Model?**

- Validation war nur für HP-Selektion nötig
- Mehr Trainingsdaten → besseres Modell
- Test bleibt unberührt bis finale Evaluation

---

## Post-Training Fehleranalyse (03b)

Nach dem finalen Training des Berlin-Champions wird eine **umfassende Fehleranalyse** durchgeführt, die alle verfügbaren Metadaten auswertet. Ziel: maximale Informationsgewinnung für die Präsentation und Interpretation der Ergebnisse.

### Visualisierungskonvention

Alle Gattungsnamen werden als **deutsche Namen** dargestellt (`genus_german` aus dem Datensatz), z.B. "Linde" statt "Tilia". Lateinische Namen können in Klammern ergänzt werden.

### Bootstrap Konfidenzintervalle (Improvement 4)

**Zweck:** Statistische Robustheit der Evaluationsmetriken quantifizieren.

Alle Performance-Metriken (F1, Precision, Recall) werden mit **Bootstrap Confidence Intervals** berechnet:

```python
def bootstrap_ci(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=0.95):
    """Bootstrap confidence intervals for any metric."""
    scores = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(y_true), size=len(y_true), replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)
    return np.mean(scores), lower, upper
```

**Anwendung:**

- **n_bootstrap = 1000:** Anzahl Bootstrap-Resamples
- **ci = 0.95:** 95% Konfidenzintervall
- **Output Format:** `F1 = 0.623 (95% CI: [0.598, 0.647])`

**Vergleich mit Naive Baselines:**

- Alle drei Naive Baselines werden ebenfalls mit Bootstrap CI evaluiert
- Performance Ladder Visualisierung zeigt CI als Fehlerbalken
- Ermöglicht statistischen Vergleich: Ist ML-Champion signifikant besser als Spatial-Only RF?

### Analysen im Detail

#### a) Konfusionsmatrix & Fehlermuster

- **Normalisierte Konfusionsmatrix** mit deutschen Genus-Labels
- **Top-10 Verwechslungspaare**: Welche Gattungen werden am häufigsten miteinander verwechselt?
- **Metriken-Tabelle**: Precision, Recall, F1 pro Gattung für präzise Berichterstattung
- **Sankey-Diagramm**: Visualisiert den Fluss von wahrer → vorhergesagter Gattung (nur Fehler)

#### b) Nadel- vs. Laubbäume

| Gruppe    | Gattungen                                               |
| --------- | ------------------------------------------------------- |
| Nadelbaum | Kiefer (Pinus), Fichte (Picea)                          |
| Laubbaum  | Linde, Ahorn, Eiche, Platane, Kastanie, Birke, Esche, … |

**Hypothese:** Nadelbäume haben ein distinkteres Spektralprofil (immergrün) → höherer F1.

#### c) Straßen- vs. Anlagenbäume (nur Berlin)

Berlin unterscheidet in der Baumkartierung zwischen:

- **Straßenbäume** (`tree_type = "strassenbaeume"`): Bäume entlang von Straßen
- **Anlagenbäume** (`tree_type = "anlagenbaeume"`): Bäume in Parks und Grünanlagen

**Hypothese:** Anlagenbäume stehen häufiger freistehend → klareres Spektralsignal. Oder: Straßenbäume sind regelmäßiger angeordnet → weniger Mischpixel.

#### d) Pflanzjahr-Analyse

Pflanzjahr (direkt im Datensatz als `plant_year` verfügbar) wird in Dekaden gruppiert:

- vor 1960, 1960-79, 1980-99, 2000-19, ab 2020

**Hypothese:** Jüngere Bäume (kleinere Krone) → weniger dominantes Spektralsignal → niedrigere Klassifikationsgenauigkeit. Ältere Bäume könnten durch größere Kronenfläche besser erkannt werden.

#### e) Artanalyse für problematische Gattungen

Für Gattungen mit F1 < 0.50 wird eine **Art-aufgelöste Analyse** durchgeführt:

- Welche Arten innerhalb der Gattung werden verwechselt?
- Beispiel: Falls Acer (Ahorn) schlecht klassifiziert wird — liegt es daran, dass A. platanoides und A. pseudoplatanus schwer zu unterscheiden sind?
- Nutzt `species_latin` / `species_german` aus dem Datensatz

#### f) CHM-Einfluss auf Genauigkeit

- CHM_1m-Werte in Bins gruppieren (z.B. 5-10m, 10-15m, 15-20m, 20-25m, >25m)
- Klassifikationsgenauigkeit pro Bin
- **Frage:** Sind sehr kleine oder sehr große Bäume schwerer zu klassifizieren?

#### g) Räumliche Fehlerverteilung

- Vorhersagegenauigkeit pro Block (500m × 500m)
- Visualisierung als Karte: Welche Berliner Stadtteile sind schwierig?
- Nutzt `geometry_lookup.parquet` für Koordinaten und `block_id` für Aggregation

### Wiederverwendbarkeit

Diese Analysen werden als **Funktionen im Visualisierungsmodul** implementiert und können
in 03c (Transfer) und 03d (Fine-Tuning) für Leipzig wiederverwendet werden.

---

## Outputs der Berlin-Optimierung

### Metadaten-Dateien

| Datei                     | Inhalt                                                                             |
| ------------------------- | ---------------------------------------------------------------------------------- |
| algorithm_comparison.json | Ergebnisse aller Algorithmen, Champion-Auswahl, **Naive Baselines (Imp 2)**        |
| hp_tuning_ml.json         | Optuna-Trials, beste Parameter für ML                                              |
| hp_tuning_nn.json         | Optuna-Trials, beste Parameter für NN                                              |
| berlin_evaluation.json    | Test-Metriken mit **Bootstrap CI (Imp 4)**, Feature Importance, Baseline-Vergleich |

### Modelle

| Datei                  | Format  | Inhalt                                     |
| ---------------------- | ------- | ------------------------------------------ |
| berlin_ml_champion.pkl | Pickle  | Trainiertes XGBoost/RF mit optimalen HP    |
| berlin_nn_champion.pt  | PyTorch | Trainiertes 1D-CNN/TabNet mit optimalen HP |
| scaler.pkl             | Pickle  | StandardScaler (für Test und Transfer)     |
| label_encoder.pkl      | Pickle  | LabelEncoder (für Genus-Mapping)           |

### Visualisierungen

**Algorithmenvergleich & HP-Tuning:**

| Abbildung                          | Zweck                                                |
| ---------------------------------- | ---------------------------------------------------- |
| algorithm_comparison.png           | Alle 4 Algorithmen F1-Vergleich                      |
| **performance_ladder.png (Imp 2)** | Baselines → ML/NN Champions mit Bootstrap CI (Imp 4) |
| algorithm_train_val_gap.png        | Train-Val Gap pro Algorithmus                        |
| optuna_optimization_history.png    | HP-Tuning Konvergenz                                 |
| feature_importance_top20.png       | Top-20 Features des Champions                        |

**Post-Training Fehleranalyse (deutsche Namen):**

| Abbildung                         | Zweck                                     |
| --------------------------------- | ----------------------------------------- |
| berlin_confusion_matrix.png       | Konfusionsmatrix (deutsche Gattungsnamen) |
| per_genus_f1_berlin.png           | Pro-Gattung F1 (sortiert, deutsch)        |
| per_genus_metrics_table.png       | Vollständige Metriken-Tabelle pro Gattung |
| confusion_pairs_worst.png         | Top-10 verwechselte Gattungspaare         |
| conifer_deciduous_comparison.png  | Nadel- vs. Laubbäume F1                   |
| tree_type_comparison.png          | Straßen- vs. Anlagenbäume                 |
| plant_year_impact.png             | Genauigkeit nach Pflanzjahr-Dekade        |
| species_breakdown_problematic.png | Art-Analyse für problematische Gattungen  |
| spatial_error_map.png             | Räumliche Fehlerverteilung (Berlin)       |
| chm_impact_on_accuracy.png        | Baumhöhe vs. Klassifikationsgenauigkeit   |
| misclassification_sankey.png      | Fehlklassifikations-Flussmuster           |

---

## Erwartete Ergebnisse

### Naive Baselines (Improvement 2)

| Baseline            | Erwarteter F1 | Begründung                               |
| ------------------- | ------------- | ---------------------------------------- |
| Majority Class      | 0.01-0.03     | ~10 Klassen → 1/10 bei balanciert        |
| Stratified Random   | 0.02-0.05     | Minimal besser als Majority              |
| Spatial-Only RF     | 0.10-0.20     | Falls räumliche Cluster vorhanden        |
| **Performance Gap** | **+0.35+**    | ML/NN Champions sollten >>0.50 erreichen |

**Interpretation:** Falls Spatial-Only RF > 0.20, existiert starke räumliche Autokorrelation.

### Berlin Upper Bound

| Metrik  | Minimum | Target | Begründung                                    |
| ------- | ------- | ------ | --------------------------------------------- |
| Val F1  | 0.55    | 0.60   | Basierend auf Literatur zu Baumklassifikation |
| Test F1 | 0.53    | 0.58   | Leicht niedriger als Val (keine HP-Leak)      |
| Gap     | <35%    | <25%   | Akzeptables Overfitting-Level                 |

### Typische F1-Werte in der Literatur

| Studie                 | Klassen     | Daten         | F1/Accuracy |
| ---------------------- | ----------- | ------------- | ----------- |
| Schiefer et al. (2020) | 7 Arten     | Hyperspektral | ~0.75       |
| Hartling et al. (2019) | 5 Gattungen | S2 + LiDAR    | ~0.65       |
| Immitzer et al. (2016) | 10 Arten    | S2            | ~0.55       |

Unsere Target von 0.60 ist realistisch für 10 Gattungen mit S2 + CHM.

---

_Letzte Aktualisierung: 2026-02-06_
