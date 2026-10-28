# 02_Exploratory_05_Spatial_Autocorrelation

> Migriert aus Notion am 2026-10-28 (Seite `31f356451f6680f79848d3fcf3989da5`, zuletzt bearbeitet 2026-03-12). Text unverändert übernommen.

## Ziel

Empirische Bestimmung der minimalen räumlichen Distanz, ab der keine signifikante Autokorrelation zwischen Baum-Features besteht. Definiert die **Block-Größe für Spatial Splits**, um Data Leakage durch räumliche Abhängigkeiten zu vermeiden.

---

## Methodik

### Moran's I

Misst räumliche Autokorrelation: Inwieweit sind ähnliche Feature-Werte geclustert?

- **I > 0:** Positive Autokorrelation (Clustering)
- **I ≈ 0:** Keine Autokorrelation (räumlich zufällig)
- **I \< 0:** Negative Autokorrelation (selten)

**Signifikanz:** Permutations-basiert (999 Permutationen, α = 0.05).

### Distance Lag Analysis

Sequenz von Distanz-Thresholds \[100m, 150m, 200m, …, 1200m\]. Pro Lag: Moran's I für repräsentative Features berechnen.

**Representative Features** (5 statt alle \~187):

- NDVI\_07, CHM\_1m, B8\_07, B4\_07, NDre1\_07
- Abdeckung: Spektral, strukturell, Indizes. Monat 07 (Peak Vegetation).

### Block Size Determination

1. **Decay Distance** pro Feature: Distanz, ab der I \< 0.05
2. **Maximum** über Features und Städte (konservativ)
3. **Safety Buffer:** +100m (Sampling-Unsicherheit, Edge Effects)
4. **Rounding:** Aufrundung auf 100m

### Post-Split Validation

Preliminary Splits erstellen → Moran's I within-split (\< 0.1) und between-split (\< 0.05) prüfen. Iterative Vergrößerung (+20%) falls nicht erfüllt.

---

## Schlüssel-Parameter

| Parameter | Wert | Begründung |
| --- | --- | --- |
| I-Threshold | \< 0.05 | Praktisch keine Autokorrelation |
| Safety Buffer | 100m | Sampling-Unsicherheit + Edge Effects |
| Cross-City Konsistenz | \< 50m Differenz | Hohe Konsistenz = Transfer-robust |
| Min. Blocks Berlin | ≥30 | 70/15/15 Split mit Stratifikation |
| Min. Blocks Leipzig | ≥20 | 80/20 Split |

**Literatur-Kontext:**

- Roberts et al. (2017, RSE): 500m für urbane Vegetationskartierung
- Zhu et al. (2019): Spatial Autocorrelation in Städten typisch 200–800m
- Erwartete Block Size: 400–600m

> 📝 **Offen:** Ergebnisse nach Pipeline-Rerun validieren (Decay Distances, Block Size, Post-Split Independence).
