# 02_Exploratory_01_Temporal_Selection

## Ziel

Identifikation der zeitlich diskriminativsten Monate für Genus-Klassifikation. Reduktion von 12 auf 6–10 Monate durch Analyse der monatlichen Separabilität.

---

## Methodik

### JM Distance (Jeffries-Matusita)

Maß für die spektrale Separabilität zwischen Genus-Paaren (Range 0–2):

- JM = 0: Vollständige Überlappung
- JM ≈ 1: Akzeptable Diskriminierung
- JM = 2: Perfekte Separabilität

**Berechnung:**

1. Für jedes der 23 S2-Features pro Monat (12 Monate)
2. Für alle Genus-Paare (paarweise Vergleiche)
3. Mittelung über Features → monatlicher JM-Wert
4. Mittelung über beide Städte

### Monats-Selektion

**Top-N Approach mit saisonaler Balance:**

1. Monate nach mittlerem JM sortieren (höher = besser)
2. Top 8 Monate auswählen
3. Saisonale Mindestabdeckung: Frühling ≥1, Sommer ≥2, Herbst ≥1, Winter optional

**Rationale:** Phenologische Abdeckung (Austrieb, Vollbelaubung, Laubfärbung) differenziert Genera. \~33% Feature-Reduktion durch Vermeidung redundanter Winter-Monate.

### Cross-City Consistency Validation

Sicherstellung, dass selektierte Monate in **beiden** Städten diskriminativ sind:

1. Monatliche JM-Distanz separat für Berlin und Leipzig berechnen
2. Spearman-Rangkorrelation ρ zwischen den Rankings
3. **ρ > 0.7:** Hohe Konsistenz → Standard-Selektion (Durchschnitts-JM)
4. **ρ ≤ 0.7:** Niedrige Konsistenz → nur Schnittmenge der Top-8

**Rationale:** Transfer Learning (Berlin → Leipzig) erfordert domänen-invariante Features. Stadt-spezifische Monate verschlechtern Generalisierung.

---

## Schlüssel-Parameter

| Parameter | Wert | Begründung |
| --- | --- | --- |
| Sampling | 10.000 Bäume/Genus | JM stabil ab \~5.000, 10k als Puffer |
| Top-N | 8 Monate | Balance Dimensionsreduktion vs. Informationserhalt |
| Konsistenz-Schwelle | ρ > 0.7 | Spearman-Korrelation für Rangkonsistenz |
