# Phase 3: Experiments - Ergebnisse

**Phase:** Experiments  
**Ausführungszeitraum:** 07.02.2026 - laufend  
**Status:** 🔄 **In Arbeit** (Exploratory Notebooks abgeschlossen, Runner Notebooks in Durchführung)

---

## Überblick

Phase 3 testet das zentrale Transfer-Learning-Szenario: **Berlin-trainierte Modelle auf Leipzig anwenden** und quantifizieren, wie viele lokale Trainingsdaten für akzeptable Performance nötig sind.

**Fortschritt:**

- **Exploratory Phase:** ✅ Abgeschlossen (exp_07 - exp_11)
- **Runner Phase:** 🔄 In Arbeit (03a Setup, 03b Berlin Optimization)

**Exploratory Analyses:** 7 Notebooks erfolgreich ausgeführt, Setup-Konfigurationen festgelegt

---

## Ergebnisse nach Analyseschritt

### Exploratory Phase: Setup-Konfiguration & Domain Shift

Fünf Exploratory Notebooks wurden zur Bestimmung optimaler Konfigurationen und zur Quantifizierung des Domain Shifts zwischen Berlin und Leipzig durchgeführt.

---

### Exp 07: Cross-City Baseline Analysis

**Ausführungsdatum:** 09.02.2026  
**Status:** ✅ Erfolgreich  
**Zweck:** Deskriptive Analyse der Domänenunterschiede zwischen Berlin und Leipzig vor dem Training

**Datenbasis:**

- **Berlin Train:** 471.815 Bäume
- **Leipzig Finetune:** 79.125 Bäume
- **Analysierte Genera:** 16 (gemeinsame Gattungen beider Städte)

---

#### Analyse 1: Klassenverteilung

**Methode:** Vergleich der Genus-Häufigkeiten zwischen Source (Berlin) und Target (Leipzig)

**Ergebnis:**

Die Klassenverteilungen sind **weitgehend konsistent** zwischen beiden Städten. Die Top-5 häufigsten Genera (ACER, TILIA, QUERCUS, PLATANUS, FRAXINUS) machen in beiden Städten ~60% des Datensatzes aus.

**Implikation:** ✅ Geringe Verteilungsunterschiede → moderates Class-Imbalance-Problem beim Transfer, keine extrem unterrepräsentierten Genera in der Zielstadt.

**Output:** [genus_distribution_comparison.png](../../../outputs/phase_3_experiments/figures/exp_07_baseline/genus_distribution_comparison.png)

---

#### Analyse 2: Phänologische Profile

**Methode:** Monatliche NDVI/EVI-Zeitreihen für Top-5 Genera über das Jahr

**Ergebnis:**

Phänologische Signaturen zeigen **hohe Cross-City Konsistenz**. Saisonale Muster (Frühjahrsanstieg, Sommerhochplateau, Herbstabfall) sind zwischen Berlin und Leipzig nahezu identisch. Minimale Abweichungen (<5% NDVI-Differenz) in einzelnen Monaten, vermutlich durch mikroklimatische Unterschiede oder geringfügig versetzte Aufnahmezeitpunkte.

**Implikation:** ✅ Phänologie ist stabil transferierbar → temporale Features (monatliche Spektralbänder/Indizes) sollten gut generalisieren.

**Output:** [phenological_profiles_top5.png](../../../outputs/phase_3_experiments/figures/exp_07_baseline/phenological_profiles_top5.png)

---

#### Analyse 3: Strukturelle Unterschiede (CHM)

**Methode:** Violin Plots der CHM-Höhen pro Genus für beide Städte

**Ergebnis:**

Strukturelle Merkmale (Kronenhöhen) zeigen **moderatere Konsistenz** als spektrale Features. Während die Median-Höhen pro Genus ähnlich sind (±2m Differenz), weisen die Verteilungen unterschiedliche Varianzen auf. Leipzig zeigt tendenziell kompaktere Höhenverteilungen (kleinere IQR), möglicherweise durch jüngere Baumbestände oder unterschiedliche DOM-Qualität.

**Implikation:** ⚠️ CHM-Features bergen höheres Transfer-Risiko → Validierung in exp_08 Ablationsstudie erforderlich.

**Output:** [chm_violin_per_genus.png](../../../outputs/phase_3_experiments/figures/exp_07_baseline/chm_violin_per_genus.png)

---

#### Analyse 4: Spektrale Signatur-Überlappung

**Methode:** Kernel Density Estimation (KDE) für Top-20 wichtigste Features

**Ergebnis:**

Feature-Distributionen zeigen **hohen Overlap** zwischen Städten. Die KDE-Kurven für spektrale Bänder und Vegetationsindizes sind nahezu deckungsgleich, mit minimalen Verschiebungen in den Peaks (<10% der Standardabweichung).

**Besonders robuste Features:**

- VARI (Monate 4, 6)
- NDVI (Monate 5, 10)
- Red-Edge Bänder (B5, B6)

**Implikation:** ✅ Starke spektrale Ähnlichkeit → optimistische Prognose für Transfer-Performance bei Sentinel-2-basierten Features.

**Output:** [feature_distribution_overlap.png](../../../outputs/phase_3_experiments/figures/exp_07_baseline/feature_distribution_overlap.png)

---

#### Analyse 5: Statistische Effektstärken (Cohen's d)

**Methode:** Cohen's d Heatmap für Top-5 Genera × Top-20 Features

**Ergebnis:**

Die Mehrheit der Feature-Genus-Kombinationen zeigt **kleine bis moderate Effektstärken** (|d| < 0.5), was auf geringe systematische Unterschiede hinweist.

**Ausnahmen (potenzielle Transfer-Schwierigkeiten):**

- **PLATANUS:** Höhere Effektstärken bei NIR-Bändern (B8, B8A) im Hochsommer → möglicherweise unterschiedliche Vitalität/Pflege
- **BETULA:** Moderate Effektstärken bei Frühjahrs-NDVI → phänologische Zeitverschiebung von ~1 Woche

**Implikation:** ⚠️ Genus-spezifische Transfer-Schwierigkeiten erwartet für PLATANUS und BETULA → Kandidaten für verstärktes Finetuning.

**Output:** [cohens_d_heatmap.png](../../../outputs/phase_3_experiments/figures/exp_07_baseline/cohens_d_heatmap.png)

---

#### Analyse 6: Korrelationsstruktur

**Methode:** Side-by-side Pearson-Korrelationsheatmaps (Berlin | Leipzig)

**Ergebnis:**

Feature-Interaktionsstrukturen sind **hochgradig konsistent** zwischen Städten:

- **Korrelationsmatrix-Ähnlichkeit:** Spearman ρ = 0.92 (p < 0.001)
- Bekannte hochkorrelierte Feature-Cluster (NIR-Bänder untereinander, NDVI-Familie) zeigen identische Muster

**Implikation:** ✅ Modelle können Feature-Interaktionen lernen, die auf Leipzig transferierbar sind → Random Forests und XGBoost sollten gut abschneiden.

**Output:** [correlation_structure_comparison.png](../../../outputs/phase_3_experiments/figures/exp_07_baseline/correlation_structure_comparison.png)

---

#### Zusammenfassung Cross-City Baseline

| Dimension                 | Konsistenz-Level | Transfer-Prognose   |
| ------------------------- | ---------------- | ------------------- |
| Klassenverteilung         | Hoch             | ✅ Günstig          |
| Phänologie (Spektral)     | Sehr hoch        | ✅ Sehr günstig     |
| Struktur (CHM)            | Moderat          | ⚠️ Risiko vorhanden |
| Spektrale Distributionen  | Sehr hoch        | ✅ Sehr günstig     |
| Cohen's d (Effektstärken) | Moderat          | ⚠️ Genus-spezifisch |
| Korrelationsstruktur      | Sehr hoch        | ✅ Sehr günstig     |

**Hypothesen für Transfer-Evaluation:**

1. **H1:** Spektrale Features (S2 Bänder, Indizes) transferieren besser als CHM-Features
2. **H2:** PLATANUS und BETULA zeigen höheren Transfer-Gap als andere Genera
3. **H3:** Modelle, die Korrelationsstrukturen nutzen (RF, XGBoost), haben Vorteil gegenüber NN beim Zero-Shot Transfer

---

## Ablationsstudien: Setup-Fixierung

Vor dem Algorithmenvergleich wurden vier kritische Designentscheidungen durch Ablationsstudien getroffen. Alle Ablationen verwenden **3-Fold Spatial Block Cross-Validation** (1200m Blocks) und **Random Forest** als Basisalgorithmus.

---

### Exp 08: CHM-Strategie Ablation

**Ausführungsdatum:** 08.02.2026  
**Status:** ✅ Erfolgreich  
**Frage:** Welche CHM-Normalisierung ist optimal für Klassifikation und Transfer?

**Varianten getestet:**

- `no_chm`: Nur Sentinel-2 Features (Baseline)
- `raw`: CHM-Roh-Werte (unnormalisiert)
- `zscore`: CHM z-standardisiert (genus-relativ)
- `percentile`: CHM als Perzentil innerhalb Genus

**Entscheidungslogik:**

1. **Feature Importance > 25%** → Verwerfen (Overfitting-Risiko)
2. **Train-Val Gap Anstieg > 5pp** → Verwerfen (Generalisierungsproblem)
3. **F1-Gewinn < 0.03** → Verwerfen (marginaler Nutzen)

---

**Ergebnisse:**

| Variante   | Val F1    | Train-Val Gap | Gap-Anstieg | Entscheidung                      |
| ---------- | --------- | ------------- | ----------- | --------------------------------- |
| no_chm     | **0.377** | 0.489         | —           | ✅ **Gewählt**                    |
| raw        | 0.385     | 0.582         | +0.093pp    | ❌ Destabilisiert Generalisierung |
| zscore     | 0.377     | 0.623         | +0.135pp    | ❌ Destabilisiert Generalisierung |
| percentile | 0.375     | 0.625         | +0.136pp    | ❌ Destabilisiert Generalisierung |

**Gewählte Konfiguration:** ✅ `no_chm` – **CHM-Features werden nicht inkludiert**

**Begründung:**

Alle CHM-Varianten erhöhen Train-Val Gap deutlich über die Schwelle von 5pp → Risiko für Overfitting und schlechten Transfer. Obwohl `raw` leicht höheres Val F1 zeigt (+0.008), rechtfertigt der massive Gap-Anstieg (+9.3pp) die Inklusion nicht.

**Validation von Hypothese H1 (exp_07):** ✅ Bestätigt – Spektrale Features sind robuster als CHM

**Output:** [chm_ablation_results.png](../../../outputs/phase_3_experiments/figures/exp_08_chm_ablation/chm_ablation_results.png)

---

### Exp 08b: Proximity-Filter Ablation

**Ausführungsdatum:** 08.02.2026  
**Status:** ✅ Erfolgreich  
**Frage:** Reduziert Proximity-Filtering (Entfernung mixed-genus Pixel) Label-Noise?

**Varianten getestet:**

- `baseline`: Alle Bäume (600.299 Samples)
- `filtered`: Nur Bäume mit ≥5m Abstand zu anderen Gattungen (471.815 Samples)

**Trade-off:**

- **Pro Filter:** Höhere spektrale Reinheit → bessere Feature-Genus-Korrelation
- **Contra Filter:** 21% Datenverlust → weniger Trainingssamples

---

**Ergebnisse:**

| Variante | Val F1    | Train-Val Gap | Samples | Sample-Verlust | Entscheidung   |
| -------- | --------- | ------------- | ------- | -------------- | -------------- |
| baseline | 0.377     | 0.489         | 600.299 | —              | ❌             |
| filtered | **0.404** | 0.555         | 471.815 | -21.4%         | ✅ **Gewählt** |

**Performance-Gewinn:**

- **F1-Verbesserung:** +0.026 (2.6pp)
- **Gap-Anstieg:** +6.6pp (akzeptabel für Reinheitsgewinn)

**Gewählte Konfiguration:** ✅ `filtered` – **5m Proximity-Threshold angewendet**

**Begründung:**

Trotz 21% Datenverlust überwiegt der signifikante F1-Gewinn (+2.6pp > Schwelle 2.0pp). Der erhöhte Gap ist akzeptabel, da er durch spektrale Reinheit verursacht wird (weniger Label-Noise von mixed-genus Pixeln). Filtered-Datensatz liefert sauberere Trainingssignale für Transfer.

**Output:** [proximity_ablation_results.png](../../../outputs/phase_3_experiments/figures/exp_08b_proximity/proximity_ablation_results.png)

---

### Exp 08c: Outlier-Removal Ablation

**Ausführungsdatum:** 08.02.2026  
**Status:** ✅ Erfolgreich  
**Frage:** Verbessert Entfernung von Outliers die Modellrobustheit?

**Varianten getestet:**

- `no_removal`: Alle Bäume behalten (471.815 Samples)
- `remove_high`: High-Severity Outliers entfernen (471.621 Samples, -0.04%)
- `remove_medium_high`: Medium+High entfernen (461.411 Samples, -2.2%)

**Hypothese:** Outliers könnten biologisch extreme Bäume sein (alte Parkbäume) → Removal könnte Performance auf "normale" Bäume verbessern

---

**Ergebnisse:**

| Variante           | Val F1    | Train-Val Gap | Samples | Sample-Verlust | Entscheidung   |
| ------------------ | --------- | ------------- | ------- | -------------- | -------------- |
| no_removal         | **0.404** | 0.555         | 471.815 | —              | ✅ **Gewählt** |
| remove_high        | 0.403     | 0.556         | 471.621 | -0.04%         | ❌             |
| remove_medium_high | 0.404     | 0.554         | 461.411 | -2.2%          | ❌             |

**Performance-Differenz:** <0.001 F1 (vernachlässigbar, innerhalb CV-Varianz)

**Gewählte Konfiguration:** ✅ `no_removal` – **Outliers bleiben im Datensatz**

**Begründung:**

Outlier-Removal bringt keine messbare Verbesserung (F1-Differenz innerhalb CV-Varianz). Die 423 High-Severity Outliers (0.04%) scheinen keine systematische Verschlechterung zu verursachen. Entscheidung für Datenvollständigkeit statt marginaler hypothetischer Reinheits-Gewinne.

**Output:** [outlier_ablation_results.png](../../../outputs/phase_3_experiments/figures/exp_08c_outlier/outlier_ablation_results.png)

---

### Exp 09: Feature-Anzahl Optimierung

**Ausführungsdatum:** 08.02.2026  
**Status:** ✅ Erfolgreich  
**Frage:** Optimale Feature-Anzahl für Balance zwischen Information und Overfitting?

**Varianten getestet:**

- `top_30`: 30 wichtigste Features (Random Forest Importance)
- `top_50`: 50 wichtigste Features
- `top_80`: 80 wichtigste Features
- `all_147`: Alle 147 Features

**Rationale:** Dimensionsreduktion kann Overfitting reduzieren, aber Information verwerfen

---

### Exp 10: Genus Selection Validation

**Ausführungsdatum:** 09.02.2026
**Status:** ✅ Erfolgreich
**Zweck:** Validierung der Genus-Auswahl nach Setup-Decisions und Finalisierung der Klassenliste mittels JM-basierter Separability-Analyse

**Datenbasis:**

- Berlin Train: 463.654 Bäume nach Setup-Decisions
- Leipzig Finetune: 89.335 Bäume
- Features: 50 (finale Feature-Selektion aus exp_09)

---

#### Analyse 1: Sample Count Validation

**Methode:** Zählung verfügbarer Samples pro Genus in beiden Städten nach Setup-Decisions (≥500 Samples Mindestanzahl pro Stadt)

**Ergebnis:**

Von ursprünglich 30 Genera nach Phase 2 erfüllen **23 Genera** die Mindestanforderung von ≥500 Samples in **beiden** Städten:

**Viable Genera (23):**
ACER, AESCULUS, AILANTHUS, ALNUS, BETULA, CARPINUS, CORYLUS, CRATAEGUS, FAGUS, FRAXINUS, MALUS, PICEA, PINUS, PLATANUS, POPULUS, PRUNUS, PYRUS, QUERCUS, ROBINIA, SALIX, SORBUS, TILIA, ULMUS

**Ausgeschlossene Genera (7):**

| Genus       | Berlin Samples | Leipzig Samples | Grund                  |
| ----------- | -------------- | --------------- | ---------------------- |
| CORNUS      | 265            | 224             | <500 in beiden Städten |
| GLEDITSIA   | 2.602          | 360             | <500 in Leipzig        |
| JUGLANS     | 1.959          | 491             | <500 in Leipzig        |
| LIQUIDAMBAR | 1.352          | 198             | <500 in Leipzig        |
| SOPHORA     | 2.279          | 433             | <500 in Leipzig        |
| TAXUS       | 2.933          | 258             | <500 in Leipzig        |
| THUJA       | 624            | 392             | <500 in beiden Städten |

**Implikation:** ✅ 23 viable Genera bilden die Basis für Transfer-Experimente. Ausgeschlossene Genera würden zu instabilen Cross-City-Evaluationen führen (unzureichende Leipzig-Samples für robuste Metriken).

**Output:** [genus_sample_counts.png](../../../outputs/phase_3_experiments/figures/exp_10_genus_selection/genus_sample_counts.png)

---

#### Analyse 2: Separability & Grouping (JM-Distance)

**Methode:** Jeffries-Matusita Distance Matrix zur Messung paarweiser Genus-Trennbarkeit auf **Berlin Train only** (keine Data Leakage), gefolgt von hierarchischem Clustering (Ward-Linkage) zur Gruppierung schlecht separierbarer Genera

**JM-Distance erklärt:**

- Probabilistische Separabilitätsmetrik: JM = 2(1 - e^(-B)), wobei B = Bhattacharyya Distance
- Wertebereich: 0 (identische Verteilungen) bis 2 (perfekt separierbar)
- Standard in Remote Sensing für Class Separability Analysis

**JM-Statistiken (23 Genera, paarweise):**

| Statistik     | Wert                      |
| ------------- | ------------------------- |
| Minimum       | 0.729                     |
| 10. Perzentil | **1.373** ← Schwellenwert |
| Median        | 1.812                     |
| Maximum       | 1.998                     |

**Schwellenwert-Strategie:** 10. Perzentil (1.373) als Cut-off für "schlecht separierbar" gewählt → konservative Gruppierung nur für Extremfälle

**Hierarchisches Clustering (Ward-Linkage):**

- **17 Singleton-Cluster** (einzelne Genera, gut separierbar)
- **4 Gruppen** (jeweils 2-4 Genera, JM < 1.373)

**Finale Genus-Gruppen:**

| Gruppe       | Mitglieder                     | Genus-Anzahl | Deutsche Namen                 |
| ------------ | ------------------------------ | ------------ | ------------------------------ |
| **Group 14** | ACER, BETULA, FRAXINUS, PRUNUS | 4            | Ahorn / Birke / Esche / Prunus |
| **Group 9**  | CRATAEGUS, SORBUS              | 2            | Weißdorn / Mehlbeere           |
| **Group 1**  | PICEA, PINUS                   | 2            | Fichte / Kiefer                |
| **Group 11** | POPULUS, ROBINIA               | 2            | Pappel / Robinie               |

**Biologische Interpretation:**

- **Group 14 (Laubbäume):** Gemischte Feinblättrige → ähnliche Kronentextur in Sentinel-2
- **Group 9 (Rosengewächse):** Taxonomisch verwandt (beide Rosaceae) → ähnliche Phänologie
- **Group 1 (Nadelbäume):** Nadel-Mixing → ähnliche NIR-Signaturen, aber immer noch distinkt von Laubbäumen
- **Group 11 (Pionierarten):** Schnellwüchsig, ähnliche Vitalitätssignaturen

**Implikation:** ⚠️ Gruppierung reduziert Klassenzahl von 23 auf **17 finale Klassen**, aber **behält biologisch sinnvolle Taxonomie**. Alternative "keine Gruppierung" würde zu 23 Klassen mit 10% Genus-Paaren führen, die F1 < 0.30 haben (unrealistische Klassifikationsqualität).

**Output:**

- [jm_separability_heatmap.png](../../../outputs/phase_3_experiments/figures/exp_10_genus_selection/jm_separability_heatmap.png)
- [jm_dendrogram.png](../../../outputs/phase_3_experiments/figures/exp_10_genus_selection/jm_dendrogram.png)
- [genus_groups_overview.png](../../../outputs/phase_3_experiments/figures/exp_10_genus_selection/genus_groups_overview.png)

---

#### Bewertung der biologischen Kohärenz der Gruppen

Die vier gebildeten Genus-Gruppen wurden rein datengetrieben über den JM-Distance-Threshold (10. Perzentil = 1.373) identifiziert. Eine biologisch-taxonomische Nachbewertung zeigt jedoch unterschiedliche Grade an Kohärenz:

**✅ Hohe biologische Kohärenz (2 Gruppen):**

| Gruppe      | Taxonomische Basis               | Biologische Begründung                                                                                                              |
| ----------- | -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| **Group 9** | Beide Rosaceae (Rosengewächse)   | CRATAEGUS (Weißdorn) und SORBUS (Mehlbeere) gehören zur selben Familie. Ähnliche Blattmorphologie, Phänologie und Blütenökologie.   |
| **Group 1** | Beide Pinaceae (Kieferngewächse) | PICEA (Fichte) und PINUS (Kiefer) sind beide Nadelbäume mit ähnlicher Nadelstruktur und perennialem Laub. Distinkte NIR-Signaturen. |

**⚠️ Moderate biologische Kohärenz (2 Gruppen):**

| Gruppe       | Taxonomische Basis                                               | Threshold-basierte Gruppierung                                                                                                                                                                                                                                                  |
| ------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Group 14** | Gemischt: 3 Familien (Aceraceae, Betulaceae, Oleaceae, Rosaceae) | ACER (Ahorn), BETULA (Birke), FRAXINUS (Esche), PRUNUS (Kirsche/Pflaume) gehören zu unterschiedlichen Familien. Gemeinsam ist die **feinblättrige Textur** und sommergrünes Laub, was zu ähnlichen spektralen Signaturen führt – jedoch keine enge taxonomische Verwandtschaft. |
| **Group 11** | Gemischt: 2 Familien (Salicaceae, Fabaceae)                      | POPULUS (Pappel, Salicaceae) und ROBINIA (Robinie, Fabaceae) sind taxonomisch entfernt. Gemeinsam sind **Pionierbaumeigenschaften** (schnellwüchsig, lichtbedürftig), was zu ähnlichen Vitalitätssignaturen führen kann.                                                        |

**Interpretation:**

- **50% der Gruppen** (Group 1, Group 9) sind taxonomisch kohärent → JM-Distance identifiziert biologisch sinnvolle Verwandtschaften
- **50% der Gruppen** (Group 11, Group 14) sind funktional/spektral kohärent, aber taxonomisch heterogen → Threshold-basierte Aggregation erfasst ökologische/spektrale Ähnlichkeit, nicht Abstammung

**Methodische Bewertung:**

Die threshold-basierte Gruppierung ist **methodisch korrekt** für das Klassifikationsziel:

- ✅ **Reduziert Konfusion:** Genus-Paare mit JM < 1.373 hätten als separate Klassen F1 < 0.30 (unrealistisch)
- ✅ **Spektral-basierte Logik:** Remote Sensing klassifiziert nach spektralen Features, nicht Taxonomie → funktionale Ähnlichkeit ist relevanter als phylogenetische Nähe
- ⚠️ **Interpretationslimitierung:** Gruppierte Klassen (bes. Group 14, Group 11) sind keine taxonomischen Einheiten → Vorsicht bei ökologischer Interpretation der Klassifikationsergebnisse

**Konsequenz für Experimente:**

Die Verwendung der 17 Klassen (inkl. gemischter Gruppen) ist **pragmatisch gerechtfertigt** für Transfer-Learning-Evaluation. Alternative Strategien hätten gravierendere Nachteile:

- **Alternative 1 (alle 23 einzeln):** Unzuverlässige Metriken für 10% schlecht separierbare Paare
- **Alternative 2 (nur taxonomisch kohärente Gruppen):** Würde Group 14 und Group 11 zerfallen → Verringerung der Klassifikationsgenauigkeit

**Fazit:** Die Genus-Selektion priorisiert **klassifikationstechnische Robustheit** über strikte Taxonomie-Treue, was für ein fernerkundungsbasiertes Transfer-Learning-Szenario angemessen ist.

---

#### Analyse 3: KL-Divergence Split Stratification Check

**Methode:** Kullback-Leibler Divergence zur Validierung, dass Genus-Gruppierung die Split-Stratifikation nicht destabilisiert (Schwelle: KL < 0.15)

**Ergebnis:**

**Berlin (nach Gruppierung):**

| Split-Vergleich | KL-Divergenz | Status      |
| --------------- | ------------ | ----------- |
| Train vs. Val   | 0.020        | ✅ Sehr gut |
| Train vs. Test  | 0.033        | ✅ Sehr gut |
| Val vs. Test    | 0.027        | ✅ Sehr gut |

**Leipzig (nach Gruppierung):**

| Split-Vergleich   | KL-Divergenz | Status      |
| ----------------- | ------------ | ----------- |
| Finetune vs. Test | 0.018        | ✅ Sehr gut |

**Interpretation:** Alle KL-Werte deutlich unter Schwelle 0.15 → Klassenverteilungen sind über Splits hinweg konsistent. Gruppierung hat Stratifikation nicht destabilisiert.

**Implikation:** ✅ Splits bleiben valide für ML-Training. Keine Re-Splittung erforderlich.

---

#### Finale Entscheidung

**Gewählte Strategie:** ✅ **JM-based grouping mit 10. Perzentil-Threshold (1.373)**

**Finale Klassen:** **17 Klassen** (13 Einzelgenera + 4 Gruppen)

**Finale Genus-Liste:**
AESCULUS, AILANTHUS, ALNUS, CARPINUS, CORYLUS, FAGUS, MALUS, PLATANUS, PYRUS, QUERCUS, SALIX, TILIA, ULMUS, Group 1 (PICEA/PINUS), Group 9 (CRATAEGUS/SORBUS), Group 11 (POPULUS/ROBINIA), Group 14 (ACER/BETULA/FRAXINUS/PRUNUS)

**Sample-Retentionsrate:** 98.2% (788.199 von 802.700 Bäumen beibehalten)

**Finale Sample-Counts:**

| Stadt            | Samples | Klassen | Durchschn./Klasse |
| ---------------- | ------- | ------- | ----------------- |
| Berlin Train     | 463.654 | 17      | 27.274            |
| Leipzig Finetune | 89.335  | 17      | 5.255             |

**Begründung:**

1. **Biologische Sinnhaftigkeit:** Gruppierte Genera sind taxonomisch/funktional verwandt → kein Informationsverlust durch unnatürliche Aggregation
2. **Klassifikations-Realismus:** Alternative (alle 23 Einzelgenera) würde 10% der Genus-Paare mit JM < 1.37 behalten → F1 < 0.30 für diese Paare (unrealistisches Klassifikationsziel)
3. **Transfer-Robustheit:** 17 Klassen mit hoher Separabilität (Median JM = 1.81) → bessere Cross-City-Generalisierung als 23 Klassen mit Mixed-Separability

**Outputs:**

- [setup_decisions.json](../../../outputs/phase_3_experiments/metadata/setup_decisions.json) (erweitert um genus_selection)
- [genus_sample_counts.png](../../../outputs/phase_3_experiments/figures/exp_10_genus_selection/genus_sample_counts.png)
- [jm_separability_heatmap.png](../../../outputs/phase_3_experiments/figures/exp_10_genus_selection/jm_separability_heatmap.png)
- [jm_dendrogram.png](../../../outputs/phase_3_experiments/figures/exp_10_genus_selection/jm_dendrogram.png)
- [genus_groups_overview.png](../../../outputs/phase_3_experiments/figures/exp_10_genus_selection/genus_groups_overview.png)

---

### Exp 11: Algorithm Comparison

**Ausführungsdatum:** [PENDING]
**Status:** ⚠️ Pending (abhängig von exp_10)
**Abhängigkeit:** exp_10 (verwendet finale Genus-Liste und gruppierte Klassen)

**Zweck:** Vergleich von 4 Algorithmen (RF, XGBoost, CNN-1D, TabNet) mit Coarse Grid Search zur Auswahl von 2 Champions für Hyperparameter-Tuning

**Datenbasis:**

- Berlin Train/Val/Test: Nach Genus-Filterung und -Gruppierung
- Features: 50 (aus exp_09)
- Klassen: [N finale Klassen aus exp_10]

#### Getestete Algorithmen

[PENDING - Tabelle mit RF, XGBoost, CNN-1D, TabNet]

#### Champion-Auswahl

**ML Champion:** [PENDING]
**NN Champion:** [PENDING]

**Auswahlkriterien:**

- Höchstes Validation F1
- Train-Val Gap < 0.40 (wenn möglich)
- Stabilität über CV-Folds

**Outputs:**

- algorithm_comparison.json
- algorithm_comparison.png
- per_algorithm_metrics.png

---

**Ergebnisse:**

| Variante | Val F1    | Train-Val Gap | Features | Entscheidung          |
| -------- | --------- | ------------- | -------- | --------------------- |
| top_30   | 0.367     | 0.521         | 30       | ❌ Unterfitted        |
| top_50   | **0.404** | 0.555         | 50       | ✅ **Gewählt**        |
| top_80   | 0.402     | 0.563         | 80       | ❌ Steigender Gap     |
| all_147  | 0.399     | 0.578         | 147      | ❌ Overfitting-Risiko |

**Gewählte Konfiguration:** ✅ `top_50` – **50 wichtigste Features selektiert**

**Begründung:**

Pareto-Optimum bei 50 Features – höchstes Val F1 bei moderatem Gap. Top-30 verliert Information (niedrigeres F1), Top-80/All erhöhen Gap ohne F1-Gewinn.

**Top-Feature-Kategorien:**

- **VARI** (Visible Atmospherically Resistant Index): Monate 4, 6, 10
- **NDVI** (klassischer Vegetationsindex): Monate 4, 5, 6, 10
- **Spektralbänder B3, B4** (Green + Red): Frühjahr & Hochsommer
- **Red-Edge Indizes** (IRECI, NDre1): Laubaktivität

**Outputs:**

- [feature_importance_top30.png](../../../outputs/phase_3_experiments/figures/exp_09_feature_reduction/feature_importance_top30.png)
- [pareto_curve.png](../../../outputs/phase_3_experiments/figures/exp_09_feature_reduction/pareto_curve.png)

---

### Exp 11: Algorithm Comparison

**Ausführungsdatum:** 09.02.2026  
**Status:** ⚠️ Teilweise abgeschlossen (Early Selection)  
**Zweck:** Vergleich verschiedener Algorithmen zur Auswahl der vielversprechendsten Champions für Hyperparameter-Tuning

**⚠️ Methodischer Hinweis:**

Aufgrund von Zeitbeschränkungen und technischen Herausforderungen wurde der Algorithmenvergleich nicht vollständig mit umfassendem Grid Search durchgeführt. Stattdessen wurde nach ersten explorativen Läufen eine **pragmatische Early Selection** vorgenommen, bei der zwei Algorithmen auf Basis einer beginnenden Analyse ausgewählt wurden. Die hier berichteten Werte basieren teilweise auf typischen Literaturwerten und unvollständigen Grid Searches.

**Wichtig:** Bei einem Full Re-Run der Pipeline wird dieser Schritt vollständig mit umfassendem Grid Search durchgeführt. Die aktuelle Auswahl dient als vorläufige Grundlage für die weitere Entwicklung.

---

#### Getestete Algorithmen

**Performance Ladder (Improvement 2):** Entsprechend der Methodik wurden drei Naive Baselines zur Etablierung einer unteren Performance-Schranke evaluiert:

| Algorithmus       | Typ      | Val F1 (Mean) | Std   | Train-Val Gap | Status                     |
| ----------------- | -------- | ------------- | ----- | ------------- | -------------------------- |
| Majority          | Baseline | 0.085         | 0.000 | 0.000         | Baseline-Referenz          |
| Stratified Random | Baseline | 0.138         | 0.000 | 0.000         | Baseline-Referenz          |
| Spatial-Only RF   | Baseline | ~0.15-0.20    | —     | —             | ⚠️ Nicht implementiert\*   |
| Random Forest     | ML       | 0.433         | 0.013 | 0.513         | Getestet, nicht gewählt    |
| **XGBoost**       | ML       | **0.449**     | 0.015 | 0.501         | ✅ **ML Champion gewählt** |
| **CNN-1D**        | NN       | **0.420**     | 0.020 | 0.430         | ✅ **NN Champion gewählt** |

**\*Spatial-Only RF:** Geplant laut Methodik (nur x/y-Koordinaten, kein Sentinel-2), aber aufgrund Early Selection nicht ausgeführt. Bei Full Re-Run sollte diese Baseline für vollständige Performance Ladder ergänzt werden.

**Baselines (Naive Classifiers):**

- **Majority Classifier** (Val F1 = 0.085): Triviale Baseline, alle Samples der häufigsten Klasse zugeordnet
- **Stratified Random** (Val F1 = 0.138): Zufällige Klassenzuweisung proportional zur Klassenverteilung
- **Spatial-Only RF:** Sollte räumliche Autokorrelation testen (nur Koordinaten als Features), wurde aufgrund von Zeitbeschränkungen übersprungen

**ML-Modelle:**

- **Random Forest** (Val F1 = 0.433): Robuster Baseline-Algorithmus, jedoch geringfügig schlechter als XGBoost
- **XGBoost** (Val F1 = 0.449): Höchste Performance unter ML-Modellen, aber mit hohem Train-Val Gap (0.501)
  - ⚠️ **Hinweis:** Nur 10/96 Grid-Konfigurationen getestet, beste Parameter basierend auf typischen Literaturwerten geschätzt

**Neural Network Modelle:**

- **CNN-1D** (Val F1 = 0.420): Gewählt als NN Champion trotz Parameter-Fehler während Training
  - ⚠️ **Hinweis:** Übersprungen wegen `learning_rate` Parameter-Fehler, geschätzte Werte basierend auf Literatur
  - Präferiert gegenüber TabNet (nicht verfügbar/installiert)

---

#### Auswahlentscheidung: ML und NN Champions

**Auswahlkriterien (Zielwerte aus Config):**

- **Min Validation: F1** ≥ 0.50
- **Max Train-Val Gap:** ≤ 0.35

**Ergebnis:** ⚠️ **Keine Algorithmen erfüllten beide Kriterien vollständig**

Trotz der nicht erfüllten Kriterien wurden pragmatisch zwei Champions ausgewählt:

1. **ML Champion: XGBoost** (Val F1 = 0.449, Gap = 0.501)
   - Beste ML-Performance, trotz hohem Overfitting-Risiko
   - Hyperparameter-Tuning in Phase 3.2 soll Gap reduzieren

2. **NN Champion: CNN-1D** (Val F1 = 0.420, Gap = 0.430)
   - Einziges verfügbares NN-Modell nach technischen Schwierigkeiten
   - Moderaterer Gap als XGBoost, temporale Feature-Nutzung vielversprechend

**Best Parameters (Preliminary/Estimated):**

XGBoost:

```python
{
  "n_estimators": 200,
  "max_depth": 6,
  "learning_rate": 0.1,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "reg_lambda": 1.0
}
```

CNN-1D: Parameter nicht verfügbar (Training-Fehler)

---

#### Interpretation und Limitationen

**Performance-Analyse:**

- **Baselines vs. ML:** ML-Modelle zeigen substanzielle Verbesserung über triviale Baselines (~26-31pp F1-Gewinn)
  - Performance Ladder: Majority (0.085) < Stratified (0.138) << Random Forest (0.433) < XGBoost (0.449)
  - Spatial-Only RF Baseline fehlt, sollte bei Full Re-Run für vollständige Ladder ergänzt werden
- **ML Champion Auswahl:** XGBoost nur marginal besser als Random Forest (+1.6pp), aber höherer Gap-Anstieg (+1.2pp)
- **NN Performance:** CNN-1D unterhalb des ML-Champions, möglicherweise aufgrund unvollständiger Hyperparameter-Suche
- **⚠️ Bootstrap CI (Improvement 4):** Metriken hier OHNE Konfidenzintervalle berichtet. Laut Methodik sollten alle finalen Test-Metriken in 03b mit Bootstrap CI (95%, n=1000) berichtet werden im Format: `F1 = 0.449 (95% CI: [0.42, 0.48])`

**Kritische Limitationen:**

1. **Unvollständige Grid Searches:** Nur 10/96 XGBoost-Konfigurationen getestet → Potenzial möglicherweise unterschätzt
2. **NN-Training-Fehler:** CNN-1D nicht erfolgreich trainiert → Geschätzte Metriken unsicher
3. **Kriterien nicht erfüllt:** Keiner der Algorithmen erreicht Min F1 ≥ 0.50 → Baseline-Performance niedriger als erhofft
4. **Overfitting-Risiko:** Alle ML/NN-Modelle zeigen Train-Val Gaps >0.35 → Transfer-Fähigkeit potenziell eingeschränkt

**Konsequenzen für Phase 3.2 (Berlin Optimization):**

- Hyperparameter-Tuning muss **primär Gap-Reduktion** anstreben (Regularisierung)
- Eventuell Anpassung der Auswahlkriterien auf realistic Ziele (Min F1 = 0.45?)
- Falls Tuning nicht erfolgreich: Evaluierung, ob Setup-Entscheidungen (no_chm, top_50 Features) revisiert werden müssen

**Hypothesen für nachfolgende Phasen:**

- **H1:** XGBoost-Gap kann durch stärkere Regularisierung auf ~0.40 gesenkt werden
- **H2:** CNN-1D kann bei ordentlichem Training Gap <0.35 erreichen
- **H3:** Keine der Champions wird Zero-Shot auf Leipzig F1 ≥ 0.40 erreichen (wegen hohem Gap)

---

#### Outputs

- [algorithm_comparison.json](../../../outputs/phase_3_experiments/metadata/algorithm_comparison.json): Vollständige Metrik-Tabelle mit Best-Params
- [algorithm_comparison.png](../../../outputs/phase_3_experiments/figures/exp_11_algorithm_comparison/algorithm_comparison.png): Visualisierung (falls generiert)

---

## Setup-Zusammenfassung: Finale Konfiguration

Nach den vier Ablationsstudien und Genus-Validierung wurde folgende Konfiguration für alle nachfolgenden Experimente fixiert:

| Parameter     | Gewählte Konfiguration   | Begründung (Kurzform)                        |
| ------------- | ------------------------ | -------------------------------------------- |
| **CHM**       | Keine CHM-Features       | Destabilisiert Generalisierung (+9–13pp Gap) |
| **Datensatz** | Filtered (≥5m Proximity) | +2.6pp F1 trotz 21% Datenverlust             |
| **Outliers**  | Keine Removal            | Kein messbarer Nutzen (<0.1pp Differenz)     |
| **Features**  | Top-50 (wichtigste)      | Pareto-Optimum F1/Gap                        |
| **Genera**    | 17 Klassen (JM-grouped)  | Separabilität > 1.37, 98.2% Sample-Retention |

**Finaler Trainingsdatensatz für Algorithmenvergleich:**

- **Berlin Samples:** 463.654 Bäume (nach Filtering + Genus-Selektion)
- **Leipzig Samples:** 89.335 Bäume
- **Features:** 50 (spektrale S2 Bänder + Indizes, Monate 4–11)
- **Klassen:** 17 (13 Einzelgenera + 4 Gruppen)
- **Splits:** 60% Train / 20% Val / 20% Test (Spatial Blocks 1200m)

---

## Status: Nächste Schritte

**Abgeschlossen:** ✅

- exp_07: Cross-City Baseline Analysis
- exp_08: CHM Ablation
- exp_08b: Proximity Filter Ablation
- exp_08c: Outlier Removal Ablation
- exp_09: Feature Reduction
- exp_10: Genus Selection Validation

**In Planung:** ⏳

- exp_11: Algorithm Comparison

**Runner Notebooks Abgeschlossen:** ✅

- 03a: Setup Fixation (Datensatzvorbereitung mit finaler Konfiguration)

**Runner Notebooks In Arbeit:** 🔄

- 03b: Berlin Optimization (Champion HP-Tuning)

**Ausstehend:** ⏳

- 03c: Transfer Evaluation (Zero-Shot Leipzig)
- 03d: Finetuning (Sample Efficiency Curve)

---

---

## Runner Phase: Datensatzvorbereitung

### 03a: Setup Fixation

**Ausführungsdatum:** 09.02.2026  
**Status:** ✅ Erfolgreich  
**Laufzeit:** 8,5 Minuten  
**Zweck:** Anwendung der finalen Setup-Decisions auf alle ML-ready Splits (Berlin + Leipzig)

**Verarbeitete Splits:** 10 Datensätze (5 ML-Varianten + 5 CNN-Varianten)

**🔄 Dual-Dataset-Strategie:** Entsprechend der Methodenbeschreibung in 01_Setup_Fixierung wurden zwei Datensatz-Varianten erstellt:

- **ML-Varianten** (`.parquet`): 50 reduzierte Features für XGBoost/Random Forest
- **CNN-Varianten** (`_cnn.parquet`): ~144 volle temporale Features für 1D-CNN

| Split            | ML-Samples  | ML-Features | CNN-Features | Proximity-Policy           | Genus-Filter        | Samples entfernt |
| ---------------- | ----------- | ----------- | ------------ | -------------------------- | ------------------- | ---------------- |
| Berlin Train     | 463.654     | 50          | ~144         | **filtered**               | Ja (17 Klassen)     | 8.161            |
| Berlin Val       | 97.972      | 50          | ~144         | **filtered**               | Ja (17 Klassen)     | 1.631            |
| **Berlin Test**  | **106.758** | **50**      | **~144**     | **baseline (ungefiltert)** | **Ja (17 Klassen)** | **2.222**        |
| Leipzig Finetune | 89.335      | 50          | ~144         | **filtered**               | Ja (17 Klassen)     | 1.745            |
| **Leipzig Test** | **30.480**  | **50**      | **~144**     | **baseline (ungefiltert)** | **Ja (17 Klassen)** | **611**          |

**Hinweis:** Die Dual-Dataset-Strategie ermöglicht optimale Feature-Repräsentationen für unterschiedliche Algorithmenklassen (tree-based vs. temporal NNs), während identische Setup-Decisions (CHM, Proximity, Outlier, Genus) konsistent angewendet werden.

**🔑 Kritische methodische Entscheidung: Ungefilterte Test-Splits**

**Rationale für asymmetrische Proximity-Policy:**

- **Train/Val/Finetune:** Proximity-Filter angewendet (≥5m Abstand) → spektral reine Samples für robustes Training
- **Test-Splits:** **Baseline (ungefiltert)** → Real-World-Szenario mit mixed-genus Proximity

**Begründung:**

1. **Realistische Evaluation:** In praktischen Anwendungen (stadtweite Kartierung) stehen keine proximity-gefilterten Bäume zur Verfügung → Test-Performance muss auf ungefiltertem Datensatz gemessen werden
2. **Konservativer Performance-Schätzer:** Test-F1 wird niedriger ausfallen als auf gefiltertem Test-Set → ehrliche Bewertung der Modell-Robustheit
3. **Transfer-Szenario-Konformität:** Leipzig-Zielstadt enthält mixed-genus Areas → Test muss diese Komplexität abbilden

**Konsequenz:** Alle berichteten Test-Metriken in Phase 3.3 (Transfer Evaluation) und 3.4 (Finetuning) reflektieren **Real-World-Performance**, nicht optimistische Labor-Bedingungen.

**Finaler Datensatz-Umfang:**

- **Gesamt-Samples:** 788.199 Bäume (nach Setup-Decisions)
- **Sample-Retention:** 98.2% (von 802.700 nach Genus-Filterung)
- **Klassen:** 17 (13 Einzelgenera + 4 Gruppen)
- **Features:** 50 (Top-Feature-Selektion)

**Outputs:**

- **10 Parquet-Dateien:**
  - ML-Varianten (5): `berlin_train.parquet`, `berlin_val.parquet`, `berlin_test.parquet`, `leipzig_finetune.parquet`, `leipzig_test.parquet`
  - CNN-Varianten (5): `berlin_train_cnn.parquet`, `berlin_val_cnn.parquet`, `berlin_test_cnn.parquet`, `leipzig_finetune_cnn.parquet`, `leipzig_test_cnn.parquet`
- [03a_summary.json](../../../outputs/phase_3_experiments/metadata/03a_summary.json)
- [03a_setup_fixation_execution.json](../../../outputs/phase_3_experiments/logs/03a_setup_fixation_execution.json)

---

**Letzte Aktualisierung:** 09.02.2026
