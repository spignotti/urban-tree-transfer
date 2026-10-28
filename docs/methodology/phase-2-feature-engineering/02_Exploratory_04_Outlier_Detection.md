# 02_Exploratory_04_Outlier_Detection

> Migriert aus Notion am 2026-10-28 (Seite `31f356451f6680059c4fd00859ed8b11`, zuletzt bearbeitet 2026-03-12). Text unverändert übernommen.

## Ziel

Validierung der Schwellenwerte für multivariate Outlier-Detektion. Etablierung eines **Severity-basierten Flagging-Systems** (keine automatische Removal) für Ablationsstudien in Phase 3.

---

## Tripartite Outlier Detection

Drei komplementäre Methoden erfassen unterschiedliche Anomalie-Typen:

| Methode | Typ | Erfasst |  |
| --- | --- | --- | --- |
| **Z-Score** | Univariat | Spektrale Extreme in einzelnen Features |  |
| **Mahalanobis** | Multivariat | Anomale Feature-Kombinationen (unter Korrelation) |  |
| **IQR (Tukey)** | Univariat | Strukturelle CHM-Höhenanomalien |  |

### 1. Z-Score

Pro Baum: Anzahl Features mit \|z\| > 3.0 zählen. Wenn ≥10 Features extrem → Outlier.

- **Threshold 3.0:** Standard 3-Sigma-Regel (99.7% Normalverteilung)
- **Min Feature Count 10:** Bei \~180 Features = \~5% → systematische Anomalie
- Erwartete Rate: \~3–7%

### 2. Mahalanobis Distance

**Genus-spezifisch:** D² zur Genus-Verteilung berechnen, Vergleich mit χ²-kritischem Wert bei α = 0.001.

- Genus-spezifische Kovarianzmatrix (nicht global), da Genera unterschiedliche spektrale Signaturen haben
- **Ledoit-Wolf Shrinkage** für numerische Stabilität bei \~180 Features
- Erwartete Rate: \~0.5–2%

### 3. IQR (Tukey's Fences)

**Genus×Stadt-spezifisch:** CHM\_1m außerhalb Q1 − 1.5×IQR / Q3 + 1.5×IQR → Outlier.

- k = 1.5: Standard (Tukey 1977)
- Genus×Stadt wegen stark variierender Höhenbereiche (QUERCUS 5–30m, MALUS 3–10m)
- Erwartete Rate: \~5–8%

---

## Consensus-Based Severity Flagging

Statt Removal → **Severity nach Consensus-Level:**

| Severity | Methoden | Erwarteter Anteil |
| --- | --- | --- |
| **high** | Alle 3 | \~1% |
| **medium** | 2 von 3 | \~2% |
| **low** | 1 von 3 | \~5% |
| **none** | Keine | \~92% |

**Rationale Flagging statt Removal:**

- Ablationsstudien ermöglichen (Phase 3 testet verschiedene Removal-Strategien)
- Nicht alle "high"-Outliers sind Fehler — könnten biologisch extreme, aber valide Bäume sein
- Methodische Transparenz: Welche Methode flaggt welchen Baum?

### Biological Context Analysis

High-severity Outliers werden zusätzlich auf biologische Erklärungen geprüft:

- **Alterskorrelation** (Mann-Whitney U): Signifikant ältere Bäume?
- **Baumtyp-Assoziation:** Überrepräsentation von Anlagenbäumen (Parks)?
- **Räumliche Clusterung** (Ripley's K): Konzentration in Parks/Denkmälern?

### Genus-Uniformitäts-Validierung

Sicherstellung, dass kein Genus überproportional betroffen ist (max. Abweichung \< 2× Mittelwert).

---

## Schlüssel-Parameter

| Parameter | Wert | Begründung |
| --- | --- | --- |
| Z-Score Threshold | 3.0σ | Standard 3-Sigma-Regel |
| Z-Score Min Features | 10 | \~5% aller Features → systematisch |
| Mahalanobis α | 0.001 | Standard für multivariate Outlier-Detection |
| IQR k | 1.5 | Standard Tukey (1977) |
| Removal | Keine | Flagging-only für Ablationsstudien |

> 📝 **Offen:** Ergebnisse nach Pipeline-Rerun validieren (Flagging-Raten, Consensus-Verteilung, Genus-Uniformität, biologischer Kontext).
