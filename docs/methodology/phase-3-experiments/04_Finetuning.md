# 04_Finetuning

# Phase 3.5: Fine-Tuning

## Einleitung

> **Wie viele lokale Trainingsdaten benötigt eine Stadt, um von einem transferierten Modell zu einem akzeptabel performenden lokalen Modell zu gelangen?**

Direkte budgetäre Implikation: Lohnt es sich, 100 Bäume manuell zu labeln, oder braucht man 1000?

---

## Forschungsfragen

1. **Sample Efficiency:** Wie viele Leipzig-Daten für X% der From-Scratch Performance?
2. **Transfer-Vorteil:** Ist Transfer + Fine-Tuning besser als From-Scratch mit gleichen Daten?
3. **Konvergenzpunkt:** Ab welcher Datenmenge ist Transfer-Vorteil vernachlässigbar?

---

## Experimentelles Design

### Beide Champions fine-tunen

Vollständiger ML vs. NN Vergleich über alle Fraktionen. Kernargument: Zero-Shot-Ranking ≠ Fine-Tuning-Ranking — ein Modell mit schlechterem Transfer kann nach Fine-Tuning besser sein.

### Fraktionen: 10%, 25%, 50%, 100%

| Fraktion | Trainings-Samples | Repräsentiert |
| --- | --- | --- |
| 10% | 8.040 | Minimal-Investment |
| 25% | 20.100 | Moderate Investment |
| 50% | 40.200 | Substantial Investment |
| 100% | 80.401 | Vollständige Nutzung |

Logarithmische Skalierung zeigt Diminishing Returns. Alle Subsets **stratifiziert** (Klassenverteilung erhalten → seltene Genus-Klassen proportional repräsentiert).

---

## Fine-Tuning Strategien

### ML (XGBoost/RF): Warm-Start

Zusätzliche Bäume auf Leipzig-Daten, ausgehend vom Berlin-Modell. Berlin-Wissen als Initialisierung; neue Bäume spezialisieren sich auf Leipzig-Muster.

### NN (1D-CNN/TabNet): Full Fine-Tune mit reduzierter LR

Learning Rate auf **10% der Original-LR** — verhindert "Vergessen" des Berlin-Wissens (0.1× Standardwert in Transfer-Learning-Literatur). Alle Weights angepasst (Full Fine-Tune statt Freeze Layers — Berlin und Leipzig unterschiedlich genug, sodass auch frühe Layers Anpassung benötigen).

**CNN Finetuning-Plateau:** Early Stopping patience auf 15 Epochen erhöht (vs. 10 beim Berlin-Training), min\_delta auf 0.001 gesenkt. Monitoring auf Validation Weighted F1, da F1 die Ziel-Metrik ist.

### Class Weighting

Neu berechnet auf **Leipzig-Verteilung** (balanced) — Fine-Tuning optimiert für Leipzig, nicht Berlin.

### Scaler-Strategie

| Szenario | Scaler | Begründung |
| --- | --- | --- |
| **Fine-Tuning** | Berlin-Scaler beibehalten | Modell-Weights erwarten Berlin-skalierte Features; neuer Scaler würde Feature-Raum verschieben |
| **From-Scratch** | Neuer Leipzig-Scaler | Kein Vorwissen, unabhängiges Modell |

---

## Datensplits

Aus Leipzig Finetune (89.335 Bäume) wird ein **10% Validierungs-Split** (stratifiziert) abgezweigt:

- **Leipzig Fine-Tune Training:** 90% (80.401) → Fraktionen 10%, 25%, 50%, 100%
- **Leipzig Validation:** 10% (8.934) → Early Stopping / Monitoring
- **Leipzig Test:** 30.480 → Finale Evaluation (unberührt während Training)

---

## From-Scratch Baseline

Komplett neu auf Leipzig trainiert, ohne Berlin-Vorwissen. Eigener Leipzig-Scaler, gleicher Val/Test-Split.

| Vergleich | Frage |
| --- | --- |
| Zero-Shot vs. From-Scratch | Ist Transfer überhaupt sinnvoll? |
| Fine-Tune 100% vs. From-Scratch | Verbessert Pretraining die 100%-Performance? |
| Fine-Tune X% vs. From-Scratch | Ab wann ist Fine-Tuning besser als From-Scratch? |

---

## Statistische Signifikanztests

### McNemar Test

Vergleicht, ob zwei Klassifikatoren **signifikant unterschiedliche Fehler** machen. α = 0.05.

**Anwendungen:** Zero-Shot vs. Fine-Tune 10%, Fine-Tune 25% vs. 50%, Fine-Tune 100% vs. From-Scratch.

### Bootstrap Confidence Intervals

Für jede Fraktion: 1000 Resamples, 95% CI für F1.

---

## Effizienz-Metriken

### Fraction to Match Scratch

Minimale Fine-Tuning-Fraktion, bei der Fine-Tune F1 ≥ From-Scratch F1.

### Fraction to 90% of Scratch

Fraktion, bei der Fine-Tune 90% der From-Scratch Performance erreicht (= "praktisch akzeptable" Performance). Niedriger Wert = Transfer sehr effizient.

### Power-Law Fit

Sample-Efficiency-Kurven folgen häufig einem **Potenzgesetz:** Performance = a × N\^b

| Exponent b | Interpretation |
| --- | --- |
| b > 0.5 | Starke Sample-Efficiency (rapide Verbesserung) |
| b ≈ 0.3 | Moderate Sample-Efficiency |
| b \< 0.2 | Schwache Sample-Efficiency → limitierte Transferierbarkeit |

**Literatur:** Tong et al. (2019, Remote Sensing): b ≈ 0.35, 50–70% Label Savings.

Ermöglicht **Extrapolation** zu Recovery-Points (z.B. 95% der From-Scratch Performance) und direkten Literatur-Vergleich. Falls b \< 0.2 → Hinweis auf limitierte Transferierbarkeit zwischen Berlin und Leipzig.

---

## Erwartete Ergebnisse

| Hypothese | Erwartung | Begründung |
| --- | --- | --- |
| Fine-Tune > Zero-Shot | 10% verbessert signifikant | Jegliche lokale Daten helfen |
| Diminishing Returns | 25%→50% weniger Gewinn als 10%→25% | Marginaler Nutzen sinkt |
| Transfer = Vorteil | Fine-Tune 50% ≈ From-Scratch 100% | Transfer spart \~50% Daten |
| Konvergenz | Fine-Tune 100% ≈ From-Scratch 100% | Bei voller Datenmenge gleicht sich aus |

---

## Praktische Implikationen

| Ergebnis | Empfehlung |  |  |
| --- | --- | --- | --- |
| 10% reicht für 90% Performance | Minimal-Investment genügt |  |  |
| 25% nötig für 90% | Moderater Labelaufwand |  |  |
| 50%+ nötig für 90% | Substantieller Aufwand, aber Transfer lohnt |  |  |
| Kein Vorteil gegenüber From-Scratch | Transfer für diese Städte nicht empfohlen |  |  |
