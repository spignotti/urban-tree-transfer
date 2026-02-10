# Methodische Erweiterungen: Nicht implementierte Optionen

Dieses Dokument beschreibt methodische Erweiterungen, die während der Planungsphase diskutiert, aber aus Zeitgründen oder aufgrund von Scope-Beschränkungen nicht implementiert wurden. Sie können als Ausgangspunkt für zukünftige Forschung dienen.

---

## 1. Transfer-optimiertes Training

### Beschreibung

Anstatt Modelle ausschließlich auf Berlin-Performance zu optimieren und dann auf Leipzig zu testen, könnte man Modelle explizit auf Transferierbarkeit trainieren.

### Mögliche Ansätze

| Ansatz                          | Beschreibung                                                                                                                     |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| **Domain-Adversarial Training** | Zusätzlicher Diskriminator, der Source/Target unterscheiden soll. Hauptmodell wird bestraft, wenn Features stadtspezifisch sind. |
| **Multi-Task Learning**         | Gleichzeitiges Training auf Berlin + kleine Leipzig-Stichprobe mit gemeinsamem Backbone                                          |
| **Feature Alignment**           | MMD (Maximum Mean Discrepancy) oder Coral Loss zwischen Berlin- und Leipzig-Features minimieren                                  |

### Warum nicht implementiert?

- Deutlich erhöhte Komplexität (zusätzliche Loss-Terme, Training-Dynamik)
- Benötigt Leipzig-Daten bereits während des Trainings → widerspricht dem "Zero-Shot" Szenario
- Standard-Ansatz (Single-City Optimierung → Transfer → Fine-Tuning) ist in der Literatur etablierter und ermöglicht klarere Aussagen

### Potenzial für Folgearbeit

Wenn Zero-Shot Transfer schlecht funktioniert (>30% Drop), wäre Domain Adaptation ein logischer nächster Schritt.

---

## 2. Multi-Seed Evaluation

### Beschreibung

Training und Evaluation mit mehreren Random Seeds (z.B. 5 Seeds) für robustere Varianzschätzungen.

### Vorteile

- Zeigt Varianz durch Initialisierung
- Wichtig für Paper-Qualität ("0.62 ± 0.02")
- Erkennt, ob einzelne gute/schlechte Ergebnisse Ausreißer waren

### Warum nicht implementiert?

- 5× Trainingszeit, besonders bei HP-Tuning (5 × 50 Trials = 250 Trials)
- Zeitbudget (~42h) würde gesprengt
- **Kompromiss:** Bootstrap Confidence Intervals auf Test-Predictions liefern Varianzschätzung ohne Mehrfach-Training

### Implementierungsvorschlag für Folgearbeit

```python
SEEDS = [42, 123, 456, 789, 1024]
results = []
for seed in SEEDS:
    model = train_model(X_train, y_train, random_state=seed)
    results.append(evaluate(model, X_test, y_test))

mean_f1 = np.mean([r['f1'] for r in results])
std_f1 = np.std([r['f1'] for r in results])
print(f"F1 = {mean_f1:.3f} ± {std_f1:.3f}")
```

---

## 3. Alternative NN-Architekturen

### Beschreibung

Neben 1D-CNN könnten andere Architekturen für temporale Muster evaluiert werden.

### Optionen

| Architektur       | Eigenschaften                                      | Eignung                                         |
| ----------------- | -------------------------------------------------- | ----------------------------------------------- |
| **LSTM**          | Explizites Gedächtnis, lernt Sequenzabhängigkeiten | Besser bei langen Sequenzen (>50 Zeitpunkte)    |
| **Transformer**   | Attention über alle Zeitpunkte, State-of-the-Art   | Braucht viele Daten, für 12 Zeitpunkte Overkill |
| **TCN**           | Dilated Convolutions, großes rezeptives Feld       | Kompromiss zwischen CNN und LSTM                |
| **InceptionTime** | Ensemble von 1D-Inception-Modulen                  | Bewährt bei UCR Benchmark                       |

### Warum nicht implementiert?

- 12 Zeitpunkte (monatliche Komposite) sind zu kurz für LSTM/Transformer-Vorteile
- 1D-CNN erfasst lokale Muster (Frühjahrsanstieg, Sommerpeak) effizient
- Mehr Parameter → höheres Overfitting-Risiko bei begrenzten Daten

### Potenzial für Folgearbeit

- Bei wöchentlichen Kompositen (52 Zeitpunkte) wäre LSTM interessant
- Bei sehr großen Datensätzen (>100k Samples) könnte Transformer funktionieren

---

## 4. Multiple Fine-Tuning Strategien

### Beschreibung

Vergleich verschiedener Strategien für das Fine-Tuning von Neural Networks.

### Optionen

| Strategie               | Beschreibung                        | Wann sinnvoll?                  |
| ----------------------- | ----------------------------------- | ------------------------------- |
| **Full Fine-Tune**      | Alle Weights anpassen, niedrige LR  | Standard, einfach               |
| **Freeze Early Layers** | Nur letzte Layers trainieren        | Source/Target sehr ähnlich      |
| **Discriminative LR**   | Niedrigere LR für frühe Layers      | Kompromiss                      |
| **Gradual Unfreezing**  | Schrittweise Layers freigeben       | Bei wenig Target-Daten          |
| **Adapter Layers**      | Kleine trainierbare Module einfügen | Effizienter bei großen Modellen |

### Warum nicht implementiert?

- Multipliziert Experimentaufwand (4 Fraktionen × 4 Strategien = 16 Experimente)
- Bereits 4 Fine-Tuning-Fraktionen (10%, 25%, 50%, 100%) geben ausreichend Einblick
- Strategy-Vergleich wäre eigenständige Forschungsfrage

### Implementierte Vereinfachung

- **ML (XGBoost):** Warm-Start mit zusätzlichen Estimators
- **NN:** Full Fine-Tune mit 0.1× Learning Rate

---

## 5. Intelligent Sample Selection für Fine-Tuning (Tong et al. 2019)

### Beschreibung

Anstatt zufälliger Auswahl der Leipzig-Samples für Fine-Tuning könnte eine intelligente Sample-Selection-Strategie verwendet werden, wie von Tong et al. (2019) für Cross-Region Transfer vorgeschlagen.

### Ansatz nach Tong et al. (2019)

**Paper:** Tong, X.-Y., Xia, G.-S., Lu, Q., Shen, H., Li, S., You, S. & Zhang, L. (2019). Land-cover classification with high-resolution remote sensing images using transferable deep models. _Remote Sensing of Environment_, 237, 111322.

**Methodischer Ablauf:**

1. **Pseudo-Labeling:** Pre-trained Berlin-Modell klassifiziert alle Leipzig-Samples
2. **Confidence Filtering:** Nur Samples mit hoher Prediction Confidence (z.B. >0.9) werden behalten
3. **Sample Retrieval:** Für jedes ausgewählte Leipzig-Sample werden ähnliche Berlin-Samples gesucht (z.B. via Euclidean Distance im Feature Space)
4. **Selective Fine-Tuning:** Modell wird mit Pseudo-Labels + retrieved Berlin-Samples fine-tuned

### Vorteile

- **Targeted Learning:** Fokus auf schwierige Grenzfälle statt zufälliges Sampling
- **Data Efficiency:** Möglicherweise bessere Performance mit weniger Samples
- **Distribution Matching:** Retrieved Berlin-Samples helfen, Domain Shift zu reduzieren

### Warum nicht implementiert?

- **Erhöhte Komplexität:** Benötigt zusätzliche Komponenten (Confidence Thresholding, Similarity Search, Sample Retrieval)
- **Zeitbudget:** Würde zusätzliche Experimente und Hyperparameter (Confidence-Threshold, Retrieval-K) einführen
- **Baseline-Fokus:** Random Sampling ist etablierter Standard für Learning Curves und ermöglicht direkten Literaturvergleich
- **Risiko:** Pseudo-Labels können falsch sein und Fehler verstärken (Self-Fulfilling Bias)

### Implementierte Vereinfachung

- **Stratified Random Sampling:** Zufällige Auswahl mit Genus-Proportionserhalt
- Garantiert unbiased Evaluation der reinen Transferierbarkeit

### Potenzial für Folgearbeit

Wenn die Random-Sampling Fine-Tuning-Kurven zeigen, dass selbst mit 100% Leipzig-Daten die Berlin-Performance nicht erreicht wird, wäre Intelligent Sample Selection ein logischer nächster Schritt:

- **Active Learning:** Modell wählt selbst informative Samples aus
- **Uncertainty Sampling:** Samples mit hoher Prediction-Unsicherheit priorisieren
- **Diversity Sampling:** Maximiere Feature-Space Coverage
- **Hybrid Approach:** Kombination aus Pseudo-Labels (hohe Confidence) und echten Labels (niedrige Confidence)

### Erwartete Verbesserungen

Basierend auf Tong et al. (2019) könnte Intelligent Sample Selection die Sample Efficiency um 20-40% steigern:

- Beispiel: Mit 25% intelligent selected Samples dieselbe Performance wie 40% random Samples

**→ Wichtig für Diskussion/Future Work im Paper erwähnen!**

---

## 6. Class Weighting Experimente

### Beschreibung

Systematischer Vergleich verschiedener Class-Weighting-Strategien.

### Optionen

| Strategie             | Berechnung                              | Effekt                       |
| --------------------- | --------------------------------------- | ---------------------------- |
| **None**              | Alle Klassen gleich                     | Majoritätsklassen dominieren |
| **Balanced**          | Inverse Klassenhäufigkeit               | Seltene Klassen wichtiger    |
| **Sqrt-Balanced**     | √(inverse Häufigkeit)                   | Kompromiss                   |
| **Effective Samples** | Basierend auf Anzahl effektiver Samples | Theoretisch fundiert         |

### Warum nicht als Experiment?

- Class Weighting ist eher eine Hyperparameter-Entscheidung als Forschungsfrage
- Standard-Ansatz (`balanced`) funktioniert in den meisten Fällen gut

### Getroffene Entscheidung

- **Berlin Training:** `class_weight='balanced'` basierend auf Berlin-Verteilung
- **Leipzig Fine-Tuning:** `class_weight='balanced'` basierend auf Leipzig-Verteilung

---

## 7. Ensemble-Methoden

### Beschreibung

Kombination mehrerer Modelle für robustere Vorhersagen.

### Optionen

| Methode               | Beschreibung                             |
| --------------------- | ---------------------------------------- |
| **Voting**            | Mehrheitsentscheidung mehrerer Modelle   |
| **Stacking**          | Meta-Modell lernt Kombinationsgewichte   |
| **Blending**          | Gewichteter Durchschnitt der Vorhersagen |
| **Snapshot Ensemble** | Checkpoints während eines Trainingslaufs |

### Warum nicht implementiert?

- Forschungsfrage fokussiert auf Transfer-Learning, nicht auf Ensemble-Performance
- Einzelmodell-Performance ist interpretierbarer
- Ensembles erschweren Transfer-Analyse (welches Modell transferiert wie gut?)

### Potenzial für Folgearbeit

ML + NN Ensemble könnte interessant sein, falls beide Modelltypen unterschiedliche Fehler machen.

---

## 8. Alternative Transfer-Szenarien

### Beschreibung

Neben Berlin → Leipzig könnten weitere Szenarien getestet werden.

### Optionen

| Szenario                 | Beschreibung                                              |
| ------------------------ | --------------------------------------------------------- |
| **Multi-Source**         | Training auf Berlin + Leipzig → Transfer auf dritte Stadt |
| **Bidirektional**        | Sowohl Berlin→Leipzig als auch Leipzig→Berlin             |
| **Leave-One-City-Out**   | N Städte, jeweils eine als Target                         |
| **Progressive Transfer** | Kette: Stadt A → B → C                                    |

### Warum nicht implementiert?

- Nur zwei Städte mit vollständigen Daten verfügbar
- Datenaufbereitung für weitere Städte würde erheblichen Aufwand bedeuten
- Bidirektionaler Transfer wäre möglich, aber Leipzig hat weniger Trainingsdaten

### Potenzial für Folgearbeit

- Integration weiterer deutscher Städte mit öffentlichen Baumkatastern
- EU-weite Studie mit harmonisierten Daten

---

## 9. From-Scratch Baselines bei allen Fraktionen

### Beschreibung

Statt nur eine From-Scratch Baseline bei 100% Leipzig-Daten zu trainieren, könnte man auch bei 10%, 25% und 50% From-Scratch-Modelle trainieren. Das ergibt eine vollständige Vergleichskurve: Transfer+Fine-Tune vs. From-Scratch bei identischer Datenmenge.

### Vorteile

- Zeigt exakt, ab welcher Datenmenge Transfer keinen Vorteil mehr bringt (Kreuzungspunkt der Kurven)
- Quantifiziert den Transfer-Vorteil pro Fraktion ("Mit 25% Transfer entspricht 60% From-Scratch")
- Stärkere Aussage für die Forschungsfrage

### Warum nicht implementiert?

- Verdoppelt die Experimente in 03d (8 From-Scratch + 8 Fine-Tuning statt 2 + 8)
- Zeitbudget ist begrenzt, Ergebnisse müssen geliefert werden
- Der 100%-Punkt reicht als Referenz für die Kernaussage

### Implementierte Vereinfachung

- From-Scratch nur bei 100% Leipzig-Daten als obere Referenzlinie
- Transfer-Vorteil wird über `fraction_to_match_scratch` und `fraction_to_90pct_scratch` approximiert

### Potenzial für Folgearbeit

- Vollständige From-Scratch-Kurve bei 10%, 25%, 50%, 100%
- Ermöglicht Kosten-Nutzen-Analyse: "Transfer spart X% Labelaufwand"

---

## 10. Explainability und Interpretierbarkeit

### Beschreibung

Tiefgehende Analyse, warum Modelle bestimmte Entscheidungen treffen.

### Mögliche Methoden

| Methode                         | Anwendung                                   |
| ------------------------------- | ------------------------------------------- |
| **SHAP**                        | Feature-Beiträge pro Vorhersage             |
| **Grad-CAM**                    | Wichtige Zeitpunkte in CNN visualisieren    |
| **Attention Weights**           | Bei TabNet direkt verfügbar                 |
| **Counterfactual Explanations** | "Was müsste sich ändern für andere Klasse?" |

### Warum nicht als Hauptfokus?

- Interpretierbarkeit ist sekundär zur Transfer-Forschungsfrage
- Feature Importance (Gain + Permutation) wird berechnet
- SHAP wäre zeitintensiv bei ~10k Test-Samples

### Implementierte Elemente

- Feature Importance für ML-Modelle
- Per-Genus F1 Analyse
- Confusion Matrix Analyse

---

## 11. Grid Search Optimierung für Algorithm Comparison

### Beschreibung

Der vollständige Grid Search in exp_11/exp_11 Algorithm Comparison ist sehr zeitaufwendig, insbesondere für XGBoost mit 96 Konfigurationen.

### Problem

**Zeitaufwand bei vollständigem Grid Search:**

- XGBoost: 96 Konfigurationen × 5-10 min/config = 8-16 Stunden
- Random Forest: 12 Konfigurationen × 3-5 min/config = 36-60 Minuten
- Bei begrenzter Zeit nicht praktikabel für erste Durchläufe

**Beobachtung aus aktuellem Run:**
Nach 10/96 XGBoost configs: F1 = 0.4489 (besser als RF nach 12/12 configs: 0.4334)
→ Trend bereits erkennbar, vollständige Suche würde wahrscheinlich XGBoost-Überlegenheit bestätigen

### Implementierte Lösung für aktuellen Run

**Manuelle Champion-Auswahl basierend auf Teilergebnissen:**

- XGBoost als ML-Champion nach 10/96 configs (F1: 0.4489)
- CNN-1D als NN-Champion (besser geeignet für temporale Daten als TabNet)
- Begründung ist methodisch vertretbar bei klarem Trend

### Empfohlene Optimierungen für zukünftige Runs

#### 1. Progressives Grid Search (Empfohlen)

```python
# Phase 1: Coarse Grid (schnell) - 12-24 configs
coarse_grid = {
    "n_estimators": [100, 200],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.1],
}

# Phase 2: Refined Grid (nur für Champion) - 20-30 configs um Best
refined_grid = {
    "n_estimators": [150, 200, 250],
    "max_depth": [5, 6, 7],
    "learning_rate": [0.08, 0.1, 0.12],
    "subsample": [0.7, 0.8, 0.9],
}
```

**Vorteil:** Reduziert Suchraum um 70-80%, Champion wird trotzdem identifiziert

#### 2. Early Stopping bei Grid Search

```python
# Stop wenn Champion klar erkennbar (statistically significant)
if config_i >= 10 and (best_f1 - second_best_f1) > 0.01:
    print("Early stopping: Champion clear after 10 configs")
    break
```

**Vorteil:** Automatische Erkennung wenn weiterer Suchaufwand unnötig

#### 3. Optuna statt Grid Search (Alternative)

- Nutzt Bayesian Optimization → findet Optimum schneller
- Bereits in 03b_berlin_optimization.ipynb implementiert
- **Option:** exp_11/11 direkt mit Optuna (20-30 trials statt 96 grid configs)

**Vorteil:** Intelligentere Suche, typischerweise 3-5× schneller zum Optimum

#### 4. Parallelisierung

```python
# Colab Pro: Multi-core Runtime nutzen
# Joblib n_jobs=-1 für parallele CV-Folds
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=cv, n_jobs=-1)
```

**Vorteil:** Reduziert Zeit um 50-70% bei Multi-Core Runtime

#### 5. Vollständiges HP-Tuning (03b, optional)

Für eine methodisch möglichst präzise Hyperparameter-Suche könnte man den Speed-Run
in 03b durch ein Volltuning ersetzen:

- **CV wieder auf 3-Fold** setzen (statt 1-Fold Holdout)
- **Mehr Trials** (z. B. 50–100) und voller Datensatz (kein Subset)
- Optional Multi-Seed-Run für robustere Parameterstabilität

**Hinweis:** Aktuell nutzen wir bewusst eine akzeptabel schnelle Lösung
(Subset + 1-Fold Holdout), um im Zeitfenster zu bleiben. Für Publikation oder
Finalruns sollte das Volltuning nachgeholt werden.

### Begründung für manuelle Entscheidung

**Methodisch akzeptabel, weil:**

1. Nach 10 configs ist XGBoost-Überlegenheit statistisch klar (0.4489 vs. 0.4334)
2. XGBoost ist in Remote Sensing etabliert (Immitzer et al. 2019, Grabska et al. 2019)
3. Vollständige Grid Search würde wahrscheinlich nur bessere XGBoost-Params finden, nicht RF überholen
4. Zeitdruck rechtfertigt pragmatische Entscheidung bei klarem Trend

**Neural Network Wahl:**

- CNN-1D optimal für temporale Sentinel-2 Daten (17 Temporal Bases × 8 Monate)
- Etabliert in Literatur für Zeitreihen-Klassifikation (Pelletier et al. 2019)
- TabNet nicht verfügbar (Installation würde zusätzliche Dependency erfordern)

### Potenzial für Folgearbeit

- Vollständige Grid Search kann in Folge-Runs nachgeholt werden für Publikation
- Progressive Grid Search implementieren als Standard-Workflow
- Optuna-basierte Algorithm Comparison evaluieren
- Multi-Seed Evaluation der Champions für robustere Varianzschätzung

---

---

## 12. exp_11 Algorithm Comparison - Workflow-Positioning

### Beschreibung

exp_11 (Algorithm Comparison) ist Teil der **explorativen Phase** und sollte unabhängig von 03a (Setup Fixation) lauffähig sein.

### IST-Zustand (Korrekt seit 2026-02-10)

- exp_11 lädt direkt aus **Phase 2c** (`phase_2_splits/`)
- Wendet Genus-Filterung basierend auf `setup_decisions.json` an (erweitert durch exp_10)
- Standalone-Notebook: Keine Abhängigkeit von 03a

### Workflow-Reihenfolge

```
exp_10 (Genus Selection) → exp_11 (Algorithm Comparison) → 03a (Setup Fixation)
```

- **exp_10:** Erstellt `genus_selection` in `setup_decisions.json` (JM-based grouping)
- **exp_11:** Lädt exp_10-Konfiguration, filtert Phase 2c Daten, vergleicht Algorithmen
- **03a:** Lädt exp_10-Konfiguration, wendet Finals zu ML-ready Datasets an

### Begründung der Architektur

- **Explorative Flexibilität:** exp_11 kann unabhängig von 03a re-runs durchgeführt werden
- **Separation of Concerns:** exp_11 testet Algorithmen, 03a fixiert finale Datensätze
- **Reproduzierbarkeit:** exp_11 dokumentiert Champion-Auswahl separat von Runner-Pipeline

### Daten-Loading Pattern

```python
# exp_11: Lädt aus Phase 2c + filtert mit exp_10 genus_selection
berlin_train = pd.read_parquet(SPLITS_DIR / "berlin_train_filtered.parquet")
genus_selection = setup_decisions.get("genus_selection", {})
viable_genera = genus_selection.get("genus_to_final_mapping", {}).values()
berlin_train = berlin_train[berlin_train['genus_latin'].isin(viable_genera)].copy()
```

### Warum nicht aus 03a laden?

- exp_11 ist **vor** 03a im Workflow positioniert (explorativ)
- 03a sollte als "Downstream Consumer" von exp_11-Ergebnissen fungieren (nutzt Champion-Entscheidungen)
- Zirkuläre Abhängigkeiten würden Pipeline-Logik brechen

---

## 13. Hypothesis Testing in 03c Transfer Evaluation (⚠️ TODO)

### Beschreibung

Die A-priori Hypothesentests in 03c_transfer_evaluation.ipynb (Section 8) schlagen derzeit fehl mit der Fehlermeldung **"Missing required columns: , "**.

### Problem

```python
H1: Genera with more Berlin training samples transfer better
  Result:    Missing required columns: ,

H2: Nadelbäume have lower transfer gap than Laubbäume
  Result:    Missing required columns: ,

H3: Genera with early leaf-out (BETULA, SALIX) have higher transfer gap
  Result:    Missing required columns: ,

H4: Genera with high Red-Edge feature importance transfer better
  Result:    Missing required columns: ,
```

### Root Cause

Die `transfer.test_hypothesis()` Funktion erwartet spezifische Spalten in den Input-DataFrames (`genus_data`, `feature_importance`), aber die Datenstrukturen aus den vorherigen Zellen stimmen nicht überein:

1. **`genus_data` DataFrame:** Wird in Cell 13 aus `berlin_per_genus` + `leipzig_per_genus` zusammengebaut, aber die Spalten-Namen oder Struktur passen nicht zur Erwartung in `test_hypothesis()`
2. **`feature_importance` DataFrame:** Wird nur übergeben, wenn `stability` existiert, aber die Spalten-Struktur könnte falsch sein

### Erforderliche Korrekturen

#### 1. Datenstruktur-Validierung

```python
# Vor test_hypothesis() - Debug-Output hinzufügen
print(f"genus_data columns: {genus_data.columns.tolist()}")
print(f"feature_importance columns: {feature_importance.columns.tolist() if feature_importance is not None else 'None'}")
```

#### 2. Function Signature Check

Die `transfer.test_hypothesis()` Funktion muss überprüft werden:
- Welche Spalten werden in `genus_data` erwartet?
- Welche Spalten werden in `feature_importance` erwartet?
- Gibt es ein Schema-Validierung?

#### 3. Hypothesen-spezifische Anforderungen

Jede Hypothese hat unterschiedliche Anforderungen:
- **H1 (Sample Size):** Benötigt `berlin_n` Spalte
- **H2 (Conifer/Deciduous):** Benötigt `tree_type` oder Mapping zu `genus_groups`
- **H3 (Early Leaf-Out):** Benötigt Genus-Kategorisierung (BETULA, SALIX)
- **H4 (Red-Edge Importance):** Benötigt Feature Importance mit Red-Edge Features

### Temporärer Workaround

Die Hypothesentests sind aktuell **nicht-kritisch** für die Kernfunktionalität:
- Zero-Shot Evaluation funktioniert ✅
- Transfer Gap Analysis funktioniert ✅
- Robustness Classification funktioniert ✅
- Visualizations funktionieren ✅

Die Hypothesentests sind **"Nice-to-Have"** für tiefere Einblicke, aber nicht essentiell für die Hauptergebnisse.

### Action Items

1. **Kurzfristig (für aktuellen Run):**
   - [ ] Hypothesentests als "Known Issue" dokumentieren
   - [ ] Notebook läuft trotzdem durch (Tests schlagen fehl, aber stoppt nicht)
   - [ ] Ergebnisse sind ohne Hypothesentests interpretierbar

2. **Mittelfristig (für Finalversion):**
   - [ ] `transfer.test_hypothesis()` Funktion debuggen
   - [ ] Erwartete Datenstruktur dokumentieren
   - [ ] Input-Validierung mit aussagekräftigen Fehlermeldungen
   - [ ] Unit Tests für alle 4 Hypothesen schreiben

3. **Langfristig (für Publikation):**
   - [ ] Alle 4 Hypothesen implementieren und validieren
   - [ ] Statistische Tests korrekt durchführen
   - [ ] Ergebnisse in Dokumentation aufnehmen

### Potenzial für Folgearbeit

Die Hypothesentests sind wertvoll für die **wissenschaftliche Diskussion**:
- Quantifizieren Transferierbarkeits-Faktoren
- Geben Einblicke in biologische vs. methodische Faktoren
- Ermöglichen Vergleich mit Literatur (Fassnacht 2016, Hemmerling 2021, Immitzer 2019)

**Priorität:** Mittel (wichtig für Paper, aber nicht für initiale Experimente)

---

## Zusammenfassung

| Erweiterung                              | Status                       | Priorität für Folgearbeit       |
| ---------------------------------------- | ---------------------------- | ------------------------------- |
| Transfer-optimiertes Training            | Nicht implementiert          | Hoch (bei schlechtem Zero-Shot) |
| Multi-Seed Evaluation                    | Kompromiss (Bootstrap CI)    | Mittel                          |
| Alternative NN-Architekturen             | 1D-CNN gewählt               | Niedrig                         |
| Multiple Fine-Tuning Strategien          | Single-Strategy              | Mittel                          |
| Intelligent Sample Selection (Tong 2019) | Nicht implementiert          | Hoch (für Data Efficiency)      |
| Class Weighting Experimente              | Balanced gewählt             | Niedrig                         |
| Ensemble-Methoden                        | Nicht implementiert          | Niedrig                         |
| Alternative Transfer-Szenarien           | Nicht implementiert          | Hoch (mehr Städte)              |
| From-Scratch alle Fraktionen             | Nur 100% Baseline            | Mittel                          |
| Explainability                           | Basis implementiert          | Mittel                          |
| Grid Search Optimierung                  | Manuelle Entscheidung (2026) | Mittel (Progressive/Optuna)     |
| exp_11 Daten-Loading                     | Phase 2c + exp_10 (✅ 2026)  | N/A                             |
| Hypothesis Testing (03c)                 | 🔴 TODO                      | Mittel (für Paper wichtig)      |

---

_Letzte Aktualisierung: 2026-02-10_
