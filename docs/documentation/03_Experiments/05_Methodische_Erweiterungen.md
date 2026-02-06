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

## Zusammenfassung

| Erweiterung                              | Status                    | Priorität für Folgearbeit       |
| ---------------------------------------- | ------------------------- | ------------------------------- |
| Transfer-optimiertes Training            | Nicht implementiert       | Hoch (bei schlechtem Zero-Shot) |
| Multi-Seed Evaluation                    | Kompromiss (Bootstrap CI) | Mittel                          |
| Alternative NN-Architekturen             | 1D-CNN gewählt            | Niedrig                         |
| Multiple Fine-Tuning Strategien          | Single-Strategy           | Mittel                          |
| Intelligent Sample Selection (Tong 2019) | Nicht implementiert       | Hoch (für Data Efficiency)      |
| Class Weighting Experimente              | Balanced gewählt          | Niedrig                         |
| Ensemble-Methoden                        | Nicht implementiert       | Niedrig                         |
| Alternative Transfer-Szenarien           | Nicht implementiert       | Hoch (mehr Städte)              |
| From-Scratch alle Fraktionen             | Nur 100% Baseline         | Mittel                          |
| Explainability                           | Basis implementiert       | Mittel                          |

---

_Letzte Aktualisierung: 2026-02-06_
