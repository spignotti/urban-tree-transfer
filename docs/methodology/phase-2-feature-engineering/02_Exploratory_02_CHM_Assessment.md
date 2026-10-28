# 02_Exploratory_02_CHM_Assessment

## Ziel

Systematische Evaluation der CHM-Features (Kronenhöhe) für Genus-Klassifikation: Diskriminationskraft, Transferierbarkeit, Plant-Year-Threshold und Genus-Scope-Entscheidung.

---

## Methodik

### 1. Discriminative Power (Welch-ANOVA η²)

**Welch-ANOVA** (`CHM_1m ~ genus_latin`) pro Stadt — robust bei ungleichen Gruppengrößen und Varianzheteroskedastizität.

- η² \< 0.06: Kleiner Effekt
- η² ∈ \[0.06, 0.14): Mittlerer Effekt
- η² ≥ 0.14: Großer Effekt → CHM ist genus-diskriminativ

### 2. Cross-City Transferability (Cohen's d)

Pro Genus: Mittelwertdifferenz CHM zwischen Berlin und Leipzig, normiert durch gepoolte Standardabweichung. Bootstrap-Konfidenzintervalle (1000 Iterationen, 95% CI).

- \|d\| \< 0.2: Geringe Differenz → gut für Transfer
- \|d\| ∈ \[0.2, 0.5): Transfer-Risiko
- \|d\| ≥ 0.5: Stadtspezifisch → schlecht für Transfer

### 3. CHM-Kataster-Korrelation

Korrelation `CHM_1m` vs. `height_m` (Kataster-Baumhöhe).

- **Erwarteter Range:** r ∈ \[0.4, 0.6\] (moderate Korrelation)
- CHM = Kronenhöhe (top of canopy), Kataster = Gesamthöhe → Differenz durch Stammhöhe, Messzeitpunkt

### 4. CHM-Feature-Engineering

Zwei Transformationen zur Reduktion absoluter Höhenunterschiede zwischen Städten:

- **Z-Score** (genus×city): `(CHM - μ_genus,city) / σ_genus,city`
- **Percentile** (genus×city): Rang innerhalb Genus und Stadt (0–100)

**Genus×City-Normalisierung** statt reiner City-Level-Normalisierung:

- City-level würde Genus-Information kodieren (große Genera → hohe Percentiles) → **Data Leakage**
- Genus×city entfernt den Genus-Mean-Effekt → Features kodieren relative Position innerhalb Gattung

### 5. Plant-Year-Threshold

Identifikation eines Schwellwerts, unterhalb dessen Bäume zu jung für Sentinel-2-Detektion sind:

- Median-CHM pro Pflanzjahr-Kohorte berechnen
- **Detection Threshold:** 2.0m — bei 10m S2-Pixeln ist eine Krone \<2m nicht als dominantes Signal detektierbar
- Cutoff = letztes Jahr mit median(CHM) ≥ 2.0m

### 6. Genus Classification Scope

Entscheidung: Nadelbäume einbeziehen oder nur Laubbäume?

- **Schwellenwert:** \< 3 Nadelbaum-Gattungen ODER \< 500 Nadelbaum-Samples → `deciduous_only`
- **Praxis:** Urbane Bestände von Laubbäumen dominiert; Nadelbäume oft \<2% der Samples

---

## Schlüssel-Parameter

| Parameter | Wert | Begründung |
| --- | --- | --- |
| CHM Detection Threshold | 2.0m | Verhältnis Kronendurchmesser zu S2-Pixelgröße |
| Min. Conifer Genera | 3 | Minimum für sinnvolle Klassifikation |
| Min. Conifer Samples | 500 | Statistische Stabilität für Splits |
| Rare Genus Fallback | \<10 Samples | City-level statt genus-level Normalisierung |
