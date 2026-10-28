# 01_Setup_Fixierung

# Phase 3.1: Setup-Fixierung

## Einleitung

Die Setup-Fixierung bildet das Fundament aller nachfolgenden Experimente. Alle **methodischen Entscheidungen** bezüglich Datenaufbereitung und Feature-Auswahl werden vor dem Algorithmenvergleich getroffen. Dies folgt dem Prinzip der **kontrollierten Variation**: Alle Algorithmen auf identischem Setup evaluiert → Performance-Unterschiede eindeutig dem Algorithmus zuzuordnen.

---

## Forschungsfragen

1. **CHM-Strategie:** Welche CHM-Normalisierung (oder keine CHM) ist optimal?
2. **Datensatzwahl:** Baseline (alle Bäume) oder Filtered (proximity-gefiltert)?
3. **Outlier-Strategie:** Outlier entfernen oder behalten? Welche Severity-Level?
4. **Feature-Anzahl:** Wie viele Features sind optimal? (Top-30, Top-50, Top-80, alle?)

---

## Methodische Grundlagen

### Warum Setup vor Algorithmenvergleich?

Setup fixieren → dann alle Algorithmen auf gleichem Setup vergleichen. Vorteile: faire Vergleichbarkeit, weniger Experimente als kombinatorische Exploration (die 480+ Kombinationen ergäbe), und Unterschiede eindeutig dem Algorithmus zuzuordnen.

### Warum Random Forest als Baseline?

| Eigenschaft | Bedeutung für Setup-Experimente |
| --- | --- |
| Geringes Overfitting | Verzerrte Entscheidungen werden vermieden |
| Deterministisch | Reproduzierbare Ergebnisse |
| Feature Importance | Direkt verfügbar für Feature-Selektion |
| Schnelles Training | Ermöglicht 3-Fold CV |

Ein optimiertes XGBoost oder NN könnte durch Overfitting auf bestimmte Features die Setup-Entscheidungen verzerren.

---

## Experimentelle Ablationen

### 1. CHM-Ablation (exp\_08)

**Motivation:** CHM liefert strukturelle Information über Baumhöhe/Kronenstruktur. Offene Fragen: Verbessert CHM die Klassifikation? Welche Normalisierung? Schadet CHM dem Transfer (stadtspezifisches Overfitting)?

| Variante | CHM-Features | S2 Features |
| --- | --- | --- |
| no\_chm | Keine | Alle |
| raw | Unnormalisiert | Alle |
| zscore | Z-Score-normalisiert | Alle |
| percentile | Perzentil-normalisiert | Alle |
| both | Z-Score + Perzentil | Alle |

**Entscheidungslogik (pro Feature einzeln bewertet):**

1. Feature Importance > 25% → ausschließen (Overfitting-Risiko)
2. Train-Val Gap Anstieg > 5pp → ausschließen (Überanpassung)
3. F1-Gewinn > 0.03 → aufnehmen; sonst ausschließen (marginaler Gewinn)

| Schwellenwert | Begründung |
| --- | --- |
| Importance > 25% | Einzelnes Feature >25% Vorhersagekraft → Overfitting-Risiko |
| Gap-Anstieg > 5pp | >5 Prozentpunkte Gap zeigt Überanpassung |
| F1-Gewinn \< 0.03 | \<3pp praktisch insignifikant, innerhalb CV-Varianz |

**Hinweis:** Entscheidung basiert ausschließlich auf Berlin-Daten. Transfer-Effekte erst in Phase 3.4 evaluiert — Leipzig-Daten in Setup-Entscheidungen wäre Information-Leak.

---

### 2. Proximity-Filter-Ablation (exp\_08b)

**Motivation:** Sentinel-2 Pixel (10m) kann überlappende Baumkronen verschiedener Gattungen enthalten → Label-Noise durch Mischsignale. Proximity-Filter (≥5m Abstand zur nächsten anderen Gattung, exp\_06) entfernt ambige Fälle.

**Varianten:** Baseline (alle Bäume) vs. Filtered (nur isolierte Bäume)

**Entscheidungslogik:**

1. Sampleverlust > 25% → Baseline (seltene Gattungen gefährdet)
2. F1-Gewinn > 0.02 → Filtered; sonst Baseline (marginaler Gewinn)

---

### 3. Outlier-Removal-Ablation (exp\_08c)

**Severity-Framework:** Drei unabhängige Methoden (Z-Score, Mahalanobis, IQR) → Consensus-Score:

| Severity | Methoden-Übereinstimmung | Interpretation |
| --- | --- | --- |
| none | 0 von 3 | Kein Outlier |
| low | 1 von 3 | Möglicher Outlier |
| medium | 2 von 3 | Wahrscheinlicher Outlier |
| high | 3 von 3 | Sicherer Outlier |

**Varianten:** no\_removal, remove\_high, remove\_high+medium

**Entscheidungslogik:** Sampleverlust > 15% → überspringen (konservativer als Proximity-Schwellenwert); F1-Gewinn > 0.02 → annehmen. Default: no\_removal.

---

### 4. Feature-Reduktion (exp\_09)

**Methodik:** Importance-basierte Selektion via RF Gain-Importance → Subsets (Top-30, Top-50, Top-80, alle) evaluiert mit 3-Fold CV → Pareto-Analyse (Kniepunkt).

**Entscheidungsregel:** Wähle kleinstes k, sodass F1(Top-k) ≥ F1(Alle) − 0.01. 1% F1-Verlust ist praktisch irrelevant und entspricht typischer Trainingsvarianz.

**Literatur-Kontext:**

| Studie | Feature-Anzahl | Accuracy |  |  |
| --- | --- | --- | --- | --- |
| Hemmerling et al. (2021) | \~276 | 82–94% |  |  |
| Immitzer et al. (2019) | 49 | 76% |  |  |
| Dieses Projekt | 30–80 | TBD |  |  |

**Hughes-Effekt:** Falls F1(Alle) \< F1(Top-k) → Curse of Dimensionality (Fassnacht et al. 2016). Feature-Selektion dann essentiell, nicht optional.

---

## Reihenfolge der Ablationen

CHM → Proximity → Outlier → Features — jede baut auf der vorherigen auf:

1. **CHM zuerst:** Feature-Entscheidung beeinflusst alle nachfolgenden
2. **Proximity vor Outlier:** Bestimmt Basis-Datensatz für Outlier-Analyse
3. **Outlier vor Features:** Feature Importance ändert sich mit/ohne Outlier
4. **Features zuletzt:** Nutzt finalen Datensatz mit allen Entscheidungen

---

## Genus Selection Validation (exp\_10)

**Problem:** Setup-Decisions reduzieren Datensatz → einige Genera könnten unter 500-Sample-Threshold fallen.

**Analysen:**

- Sample Count Validation nach Setup-Decisions
- JM-Distance Separability Matrix (paarweise, nur Berlin Train)
- Hierarchical Clustering (Ward-Linkage) für Genus-Gruppierung
- Adaptive Threshold: 10th Percentile (P10) — konservativer Cut, nur schwächste \~10% der Genus-Paare gruppiert (datengestützter Threshold statt fixer Grenzwert)
- KL-Divergence Validation (\<0.15) für Split-Stratifizierung nach Genus-Filterung (exp\_10-spezifisch; der allgemeine Split-Validierungs-Threshold in 02c ist 0.07, relaxed von Literatur-Standard 0.05)

**Methodische Validität:** Genus-Filtering nach Spatial Splits unproblematisch — Block-Grenzen (1200m) geografisch fix, Train/Val/Test-Zuordnungen unverändert, räumliche Autokorrelations-Prevention intakt.

---

## Dual-Dataset-Strategie (Runner 03a)

Nach Abschluss aller Ablationen erstellt 03a finale ML-ready Datensätze in **zwei Varianten:**

- **XGBoost-Variante:** Reduzierte Features (Top-50) für Tree-based Models → vermeidet Dimensionality Curse
- **CNN-Variante:** Volle temporale Sequenzen (144 Features: 8 Monate × 18 Variablen — 8 Spektralbänder + 10 Vegetationsindizes) → temporale Kontinuität für Faltungsoperationen

**Begründung:** Gleiche Setup-Decisions, gleiche Samples/Labels, Unterschied nur bei Features. Tree-based Models profitieren von Feature Selection, CNNs benötigen vollständige Zeitreihen.

### Asymmetrische Proximity-Policy

- **Train/Val/Finetune:** Proximity-Filter angewendet (falls gewählt) → spektral reine Trainingssignale
- **Test-Splits (Berlin + Leipzig):** Immer ungefiltert → Real-World-Komplexität

**Begründung:** Leipzig-Zielstadt hat keine Proximity-Filterung in realen Kartierungen. Test-Metriken sind konservativer, reflektieren aber echte Einsatzbedingungen. Alternative (konsistente Filterung) wurde verworfen — optimistische Performance, in Praxis nicht erreichbar.
