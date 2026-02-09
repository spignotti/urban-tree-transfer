# Phase 3.1: Setup-Fixierung

## Einleitung

Die Setup-Fixierung bildet das Fundament aller nachfolgenden Experimente. In dieser Phase treffen wir alle **methodischen Entscheidungen** bezüglich Datenaufbereitung und Feature-Auswahl, bevor wir Algorithmen vergleichen und optimieren. Dies folgt dem Prinzip der **kontrollierten Variation**: Alle Algorithmen werden später auf dem identischen Setup evaluiert, sodass Performance-Unterschiede eindeutig dem Algorithmus zugeordnet werden können.

---

## Forschungsfragen

1. **CHM-Strategie:** Welche CHM-Normalisierung (oder keine CHM) ist optimal?
2. **Datensatzwahl:** Baseline (alle Bäume) oder Filtered (proximity-gefiltert)?
3. **Outlier-Strategie:** Outlier entfernen oder behalten? Wenn ja, welche Severity-Level?
4. **Feature-Anzahl:** Wie viele Features sind optimal? (Top-30, Top-50, Top-80, alle?)

---

## Methodische Begründungen

### Warum Setup-Entscheidungen vor Algorithmenvergleich?

Die Reihenfolge "Setup fixieren → dann Algorithmen vergleichen" folgt dem Prinzip der **kontrollierten Variation**:

```
Option A (gewählt):                Option B (verworfen):
──────────────────────             ─────────────────────
1. CHM-Strategie fixieren          1. Alle Algorithmen mit
2. Datensatz wählen                   allen Setups testen
   (Baseline/Filtered)             2. Kombinatorische Explosion:
3. Outlier-Strategie fixieren         5 CHM × 2 Datasets × 3 Outlier
4. Features selektieren               × 4 Features × 4 Algos
5. Dann alle Algorithmen              = 480+ Experimente
   auf GLEICHEM Setup
   vergleichen
```

**Begründung:**

- Faire Vergleichbarkeit: Alle Algorithmen nutzen identische Features
- Effizienz: Weniger Experimente nötig
- Interpretierbarkeit: Unterschiede sind eindeutig dem Algorithmus zuzuordnen

### Warum Random Forest als Baseline-Algorithmus für Setup?

Für die Setup-Entscheidungen (CHM, Features) verwenden wir **Random Forest mit Default-Parametern** als stabilen Baseline:

| Eigenschaft          | Bedeutung für Setup-Experimente           |
| -------------------- | ----------------------------------------- |
| Geringes Overfitting | Verzerrte Entscheidungen werden vermieden |
| Deterministisch      | Reproduzierbare Ergebnisse                |
| Feature Importance   | Direkt verfügbar für Feature-Selektion    |
| Schnelles Training   | Ermöglicht 3-Fold CV                      |

Ein optimiertes XGBoost oder NN könnte durch Overfitting auf bestimmte Features die CHM-Entscheidung verzerren.

---

## Experimentelle Ablationen

### 1. CHM-Ablation (exp_08)

#### Motivation

Das Canopy Height Model (CHM) liefert strukturelle Information über Baumhöhe und Kronenstruktur. Es ist jedoch unklar:

1. **Verbessert CHM die Klassifikation?** — Wenn ja, lohnt sich die zusätzliche Komplexität
2. **Welche Normalisierung ist optimal?** — Raw, Z-Score, Perzentile, oder beide?
3. **Schadet CHM dem Transfer?** — Wenn CHM stadtspezifisch ist, könnte es Overfitting auf Berlin verursachen

#### Experimentelles Design

| Variante   | CHM-Features                             | Sentinel-2 Features |
| ---------- | ---------------------------------------- | ------------------- |
| no_chm     | Keine                                    | Alle S2 Features    |
| raw        | chm_mean, chm_std, etc. (unnormalisiert) | Alle S2 Features    |
| zscore     | chm_mean_zscore, etc.                    | Alle S2 Features    |
| percentile | chm_mean_pct, etc.                       | Alle S2 Features    |
| both       | zscore + percentile                      | Alle S2 Features    |

#### Entscheidungslogik

Jedes CHM-Feature wird **einzeln** bewertet. Die Entscheidung wird pro Feature getroffen:

```
     Für jedes Feature (CHM_1m, CHM_1m_zscore, CHM_1m_percentile):

                    ┌─────────────────────────────────┐
                    │ Feature Importance > 25%?       │
                    └────────────┬────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Ja                      │ Nein
                    ▼                         ▼
         ┌──────────────────┐    ┌─────────────────────────────┐
         │ → Feature        │    │ Train-Val Gap Anstieg > 5pp?│
         │   ausschließen   │    └───────────┬─────────────────┘
         │ (Overfitting)    │                │
         └──────────────────┘   ┌────────────┴────────────┐
                                │ Ja                      │ Nein
                                ▼                         ▼
                    ┌──────────────────┐    ┌──────────────────────┐
                    │ → Feature        │    │ F1-Gewinn > 0.03?    │
                    │   ausschließen   │    └──────────┬───────────┘
                    │ (Gap-Risiko)     │               │
                    └──────────────────┘  ┌────────────┴────────────┐
                                          │ Ja                      │ Nein
                                          ▼                         ▼
                              ┌──────────────────┐    ┌──────────────────┐
                              │ → Feature         │    │ → Feature        │
                              │   aufnehmen       │    │   ausschließen   │
                              └──────────────────┘    │ (Marginaler      │
                                                      │  Gewinn)         │
                                                      └──────────────────┘
```

**Hinweis:** Diese Entscheidung basiert ausschließlich auf Berlin-Daten.
Transfer-Effekte von CHM werden erst in Phase 3.4 (Transfer-Evaluation)
evaluiert. Leipzig-Daten in Setup-Entscheidungen zu verwenden wäre ein
Information-Leak.

#### Wissenschaftliche Begründung der Schwellenwerte

| Schwellenwert               | Begründung                                                                              |
| --------------------------- | --------------------------------------------------------------------------------------- |
| Feature Importance > 25%    | Wenn ein einzelnes Feature >25% der Vorhersagekraft trägt, besteht Overfitting-Risiko   |
| Train-Val Gap Anstieg > 5pp | Mehr als 5 Prozentpunkte Gap-Anstieg zeigt, dass das Feature zu Überanpassung führt     |
| F1-Gewinn < 0.03            | Unterschiede <3 Prozentpunkte sind praktisch insignifikant und innerhalb der CV-Varianz |

#### Visualisierungen

| Abbildung                  | Zweck                               |
| -------------------------- | ----------------------------------- |
| chm_ablation_results.png   | CHM-Varianten F1-Vergleich          |
| chm_feature_importance.png | CHM Feature Importance pro Variante |
| chm_train_val_gap.png      | Train-Val Gap Vergleich             |

---

### 2. Proximity-Filter-Ablation (exp_08b)

#### Motivation

In Phase 2c wurden zwei Datensatz-Varianten erstellt:

- **Baseline:** Alle Bäume, die das Gattungsfrequenz-Kriterium erfüllen
- **Gefiltert:** Nur Bäume mit ≥20m Abstand zum nächsten Baum einer anderen Gattung

Der Filter reduziert **Label-Noise durch überlappende Baumkronen**: Wenn ein Sentinel-2 Pixel (10m Auflösung) sowohl eine Linde als auch eine Eiche enthält, ist das spektrale Signal eine Mischung beider Gattungen. Der Proximity-Filter entfernt solche ambigen Fälle.

#### Experimentelles Design

| Variante | Datensatz                       | Beschreibung                                  |
| -------- | ------------------------------- | --------------------------------------------- |
| baseline | `berlin_train.parquet`          | Alle Bäume (maximale Datenmenge)              |
| filtered | `berlin_train_filtered.parquet` | Nur isolierte Bäume (reduziertes Label-Noise) |

#### Entscheidungslogik

```
                    ┌───────────────────────────────┐
                    │ Sampleverlust > 20%?           │
                    └────────────┬──────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Ja                      │ Nein
                    ▼                         ▼
         ┌──────────────────┐    ┌─────────────────────────────┐
         │ → Baseline       │    │ F1-Gewinn > 0.02?           │
         │ (zu viel         │    └───────────┬─────────────────┘
         │  Datenverlust)   │                │
         └──────────────────┘   ┌────────────┴────────────┐
                                │ Ja                      │ Nein
                                ▼                         ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │ → Filtered       │    │ → Baseline       │
                    │                  │    │ (marginaler      │
                    └──────────────────┘    │  Gewinn)         │
                                           └──────────────────┘
```

#### Wissenschaftliche Begründung der Schwellenwerte

| Schwellenwert       | Begründung                                                                      |
| ------------------- | ------------------------------------------------------------------------------- |
| Sampleverlust > 20% | Mehr als 20% Datenverlust kann seltene Gattungen unter Minimum-Schwelle drücken |
| F1-Gewinn < 0.02    | Unter 2 Prozentpunkten überwiegt der Vorteil größerer Datenmenge                |

#### Visualisierungen

| Abbildung                   | Zweck                              |
| --------------------------- | ---------------------------------- |
| proximity_f1_comparison.png | Baseline vs. Filtered F1           |
| proximity_per_genus_f1.png  | Pro-Gattung F1 (Baseline/Filtered) |
| proximity_sample_loss.png   | Sample-Verlust pro Gattung         |

---

### 3. Outlier-Removal-Ablation (exp_08c)

#### Motivation

In Phase 2 wurden drei unabhängige Outlier-Detection-Methoden berechnet (Z-Score, Mahalanobis, IQR) und zu einem Severity-Score zusammengefasst:

| Severity | Methoden-Übereinstimmung | Interpretation                   |
| -------- | ------------------------ | -------------------------------- |
| none     | 0 von 3                  | Kein Outlier                     |
| low      | 1 von 3                  | Möglicher Outlier (eine Methode) |
| medium   | 2 von 3                  | Wahrscheinlicher Outlier         |
| high     | 3 von 3                  | Sicherer Outlier (alle einig)    |

Die Flags wurden bewusst als Metadaten erhalten (nicht entfernt), damit Phase 3 eine datengestützte Entscheidung treffen kann.

#### Experimentelles Design

| Variante           | Entfernte Severity-Level | Beschreibung                           |
| ------------------ | ------------------------ | -------------------------------------- |
| no_removal         | Keine                    | Alle Bäume (Baseline)                  |
| remove_high        | high                     | Nur sichere Outlier entfernen          |
| remove_high_medium | high, medium             | Auch wahrscheinliche Outlier entfernen |

#### Entscheidungslogik

Für jede Removal-Variante (geordnet nach Aggressivität):

```
                    ┌───────────────────────────────┐
                    │ Sampleverlust > 15%?           │
                    └────────────┬──────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │ Ja                      │ Nein
                    ▼                         ▼
         ┌──────────────────┐    ┌─────────────────────────────┐
         │ → Überspringen   │    │ F1-Gewinn > 0.02?           │
         │ (zu aggressiv)   │    └───────────┬─────────────────┘
         └──────────────────┘                │
                                ┌────────────┴────────────┐
                                │ Ja                      │ Nein
                                ▼                         ▼
                    ┌──────────────────┐    ┌──────────────────┐
                    │ → Annehmen       │    │ → Überspringen   │
                    │                  │    │ (marginaler      │
                    └──────────────────┘    │  Gewinn)         │
                                           └──────────────────┘

Default: no_removal (wenn keine Variante die Kriterien erfüllt)
```

#### Wissenschaftliche Begründung der Schwellenwerte

| Schwellenwert       | Begründung                                                                              |
| ------------------- | --------------------------------------------------------------------------------------- |
| Sampleverlust > 15% | Outlier-Removal sollte konservativer sein als Proximity-Filter                          |
| F1-Gewinn < 0.02    | Konsistent mit Proximity-Schwellenwert; bei marginalem Gewinn bevorzugen wir mehr Daten |

#### Visualisierungen

| Abbildung                         | Zweck                                 |
| --------------------------------- | ------------------------------------- |
| outlier_distribution_by_genus.png | Outlier-Severity pro Gattung          |
| outlier_tradeoff_curve.png        | F1 vs. Samplegröße Trade-off          |
| outlier_per_genus_f1.png          | Pro-Gattung F1 über Removal-Varianten |

---

### 4. Feature-Reduktion (exp_09)

#### Motivation

Die Feature-Anzahl beeinflusst:

1. **Performance:** Zu viele Features → Overfitting; zu wenige → Underfitting
2. **Interpretierbarkeit:** Weniger Features sind verständlicher
3. **Trainingszeit:** Weniger Features = schnellere Modelle
4. **Transfer-Robustheit:** Generelle Features transferieren besser als spezifische

#### Methodik: Importance-basierte Selektion

```
1. Trainiere RF mit allen Features
   └── Extrahiere Gain-basierte Importance

2. Ranke Features nach Importance
   └── Erstelle Subsets: Top-30, Top-50, Top-80, Alle

3. Evaluiere jedes Subset mit 3-Fold CV
   └── Messe: F1, Trainingszeit

4. Pareto-Analyse
   └── Finde Kniepunkt: Minimale Features bei maximaler F1

5. **Literatur-Comparison (NEW - Imp 1)**
   └── Positionierung unserer Feature-Anzahl im wissenschaftlichen Kontext

6. **Hughes-Effekt Check (NEW - Imp 1)**
   └── Prüfen ob F1(Alle) < F1(Top-k) → Curse of Dimensionality
```

#### Entscheidungslogik

**Regel:** Wähle kleinstes k, sodass F1(Top-k) ≥ F1(Alle) - 0.01

**Begründung des 1%-Schwellenwerts:**

- 1% F1-Verlust ist praktisch irrelevant
- Entspricht typischer Varianz zwischen Trainingsläufen
- Ermöglicht signifikante Feature-Reduktion ohne echten Performanceverlust

#### Literatur-Context (Improvement 1)

**Vergleich mit publizierten Studien:**

| Studie                   | Feature-Anzahl | Input-Typ               | Accuracy | Anmerkung           |
| ------------------------ | -------------- | ----------------------- | -------- | ------------------- |
| Hemmerling et al. (2021) | ~276           | 12 months × 23 features | 82-94%   | Dense time series   |
| Immitzer et al. (2019)   | 49             | Feature-selected        | 76%      | Top S2 features     |
| Dieses Projekt           | 30-80          | 8 months × selected     | TBD      | RF importance-based |

**Interpretation:** Positioniert Projekt im wissenschaftlichen Kontext. Unsere Feature-Anzahl liegt im typischen Bereich für Sentinel-2 basierte Baumklassifikation.

#### Hughes-Effekt Analyse (Improvement 1)

**Wenn F1(Alle Features) < F1(Top-k Features):**

Dokumentation:

- Expliziter Hinweis auf Hughes-Effekt (Fassnacht et al. 2016)
- Feature-Selektion ist **essentiell**, nicht optional
- Begründung: Mit limitiertem Training-Set führen zu viele Features zu Overfitting

**Literatur:** Fassnacht et al. (2016): "Curse of Dimensionality" bei zu vielen Features relativ zur Sample Size

#### Visualisierungen

| Abbildung                               | Zweck                                                 |
| --------------------------------------- | ----------------------------------------------------- |
| feature_importance_ranking.png          | Feature-Importance Ranking (alle)                     |
| pareto_curve.png                        | F1 vs. Feature-Anzahl                                 |
| feature_group_contribution.png          | Beitrag S2 vs. CHM Feature-Gruppen                    |
| **feature_pareto_curve_literature.png** | **Pareto mit Knee-Point + Literatur-Kontext (Imp 1)** |

---

## Reihenfolge der Ablationen

Die vier Setup-Ablationen werden **sequentiell** durchgeführt, da jede auf der vorherigen aufbaut:

```
exp_08 (CHM)  →  exp_08b (Proximity)  →  exp_08c (Outlier)  →  exp_09 (Features)
     │                  │                       │                      │
     ▼                  ▼                       ▼                      ▼
  Welche CHM-       Baseline oder          Outlier              Wie viele
  Features?         Filtered?              entfernen?           Features?
```

**Begründung der Reihenfolge:**

1. **CHM zuerst:** Feature-Entscheidung beeinflusst alle nachfolgenden Experimente
2. **Proximity vor Outlier:** Bestimmt den Basis-Datensatz, auf dem Outlier-Analyse stattfindet
3. **Outlier vor Feature-Reduktion:** Feature Importance ändert sich mit/ohne Outlier
4. **Feature-Reduktion zuletzt:** Nutzt den finalen Datensatz mit allen vorherigen Entscheidungen

---

## Genus Selection Validation (exp_10)

**Ausführungsdatum:** [NACH AUSFÜHRUNG AUSFÜLLEN]
**Status:** [NACH AUSFÜHRUNG AUSFÜLLEN]
**Zweck:** Validierung dass alle Genera nach Setup-Decisions noch ≥500 Samples haben

### Problem

Phase 1 filtert auf Genera mit ≥500 Samples → **30 viable genera**.
Setup-Decisions (CHM-Strategie, Proximity, Outlier, Feature-Selektion) reduzieren Datensatz weiter.

**Risiko:** Einige Genera könnten unter 500-Sample-Threshold gefallen sein.

### Analysen

1. **Sample Count Validation:** Tatsächliche Genus-Counts nach Setup-Decisions
2. **Sample Sufficiency Assessment:** Vergleich mit Literatur (RF benötigt ~100-200 Samples/Klasse)
3. **JM-Distance Separability Matrix:** Paarweise Jeffries-Matusita Distance zwischen allen Genera (sample-level, nur Berlin Train)
4. **Hierarchical Clustering:** Ward-Linkage auf JM-Matrix für Genus-Gruppierung
5. **Adaptive Threshold:** Percentile-basierte JM-Schwelle (z.B. 20th percentile) für Gruppierungsentscheidung
6. **Grouping Decision:** Schlecht separierbare Genera gruppieren basierend auf JM < Threshold
7. **KL-Divergence Validation:** Prüfung dass Splits weiterhin stratifiziert sind (Threshold: <0.15)

### Entscheidung

**[WIRD NACH AUSFÜHRUNG AUSGEFÜLLT]**

**Gewählte Strategie:** exclude_low_sample_and_group_similar

**Finale Genus-Liste:** [N Klassen]

- Ausgeschlossene Genera: [...]
- Gruppierte Genera: [...]

**Reasoning:** [...]

### Methodische Validität

**Genus-Filtering nach Spatial Splits ist methodisch unproblematisch**, weil:

1. Block-Grenzen (1200m) sind geografisch fix (nicht genus-abhängig)
2. Train/Val/Test Block-Zuordnungen bleiben unverändert
3. Räumliche Autokorrelations-Prevention bleibt intakt
4. KL-Divergence nach Filtering bestätigt Stratifizierung

### Output

- **Config:** `outputs/phase_3_experiments/metadata/setup_decisions.json` (erweitert um `genus_selection` Section)
- **Visualisierungen:**
  - `genus_sample_counts.png` - Sample-Counts mit 500-Threshold
  - `genus_separability_heatmap.png` - Genus-Distanz-Matrix im finalen Feature-Space
  - `genus_dendrogram.png` - Hierarchisches Clustering (Ward-Linkage)

---

## Runner-Notebook: 03a_setup_fixation.ipynb

Nach Abschluss aller Ablationen fasst dieses Runner-Notebook die Entscheidungen zusammen:

### Inputs

- `outputs/phase_3/metadata/setup_decisions.json` (erstellt durch exp_08, exp_08b, exp_08c, exp_09)

### Prozess

1. **Lade und validiere** `setup_decisions.json`
2. **Wende Entscheidungen an:**
   - Selektiere CHM-Features
   - Wähle Datensatz-Variante (baseline/filtered)
   - Entferne Outlier (falls entschieden)
   - Reduziere Features auf Top-k
3. **Erstelle feature-reduced Datasets** für Experimente:
   - `berlin_train.parquet` (final)
   - `berlin_val.parquet` (final)
   - `berlin_test.parquet` (final)
   - `leipzig_finetune.parquet` (final)
   - `leipzig_test.parquet` (final)

### Outputs

- Prozessierte Datasets in `data/phase_3_experiments/`
- Execution Log in `outputs/phase_3/logs/03a_setup_fixation.json`

---

## Outputs der Setup-Fixierung

### setup_decisions.json

```json
{
  "timestamp": "2026-02-03T10:30:00Z",
  "decisions": {
    "chm_strategy": {
      "selected_features": ["CHM_1m_zscore"],
      "excluded_features": ["CHM_1m", "CHM_1m_percentile"],
      "reasoning": "Z-score has 18% importance, improves F1 by 0.04, gap increase only 3pp"
    },
    "proximity_strategy": {
      "selected_dataset": "baseline",
      "reasoning": "Filtered loses 22% samples, F1 gain only 0.015 (below 0.02 threshold)"
    },
    "outlier_strategy": {
      "removal_level": "remove_high",
      "reasoning": "Removes 8% samples, improves F1 by 0.03, conservative approach"
    },
    "feature_set": {
      "n_features": 50,
      "selected_features": ["NDVI_mean", "EVI_mean", "CHM_1m_zscore", ...],
      "reasoning": "Top-50 achieves 99.2% of all-features F1, significant reduction from 120"
    }
  },
  "validation_metrics": {
    "berlin_val_f1": 0.58,
    "berlin_train_val_gap": 0.22
  }
}
```

### Alle Visualisierungen

Siehe Abschnitte oben für vollständige Liste der Ablation-spezifischen Visualisierungen.

---

_Letzte Aktualisierung: 2026-02-06_
