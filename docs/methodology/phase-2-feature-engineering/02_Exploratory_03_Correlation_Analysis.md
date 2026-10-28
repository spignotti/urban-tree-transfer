# 02_Exploratory_03_Correlation_Analysis

## Ziel

Identifikation hochkorrelierter Features zur Vermeidung von Multikollinearität. Redundante Features werden für Removal markiert, unter Gewährleistung von Cross-City-Konsistenz.

---

## Methodik

### Feature-Gruppierung

Korrelationen werden **nur innerhalb** von Feature-Gruppen analysiert:

- **CHM-Features** (3): CHM\_1m, CHM\_1m\_zscore, CHM\_1m\_percentile
- **Spektrale Bänder** (10 Basen): B2–B4 (Visible), B5–B7 (Red-Edge), B8/B8A (NIR), B11/B12 (SWIR)
- **Vegetationsindizes** (13 Basen): Broadband (NDVI, EVI, GNDVI, VARI), Red-Edge (NDre1, NDVIre, CIre, IRECI, RTVIcore), Wasser (NDWI, MSI, NDII, kNDVI)

### Korrelations-Berechnung

**Dual-Metrik:** Pearson r (linear) + Spearman ρ (nichtlinear, robust bei NDVI-Sättigung). Pearson primär für Removal-Entscheidung; Spearman als Validierung. Bei Diskrepanzen → konservativ beibehalten.

**Temporal-Aggregation:** Alle Monate eines Feature-Paares werden zu einem Vektor gestackt → ein r-Wert über Raum und Zeit (keine Cross-Monat-Korrelationen).

### Redundanz-Schwellenwert

**\|r\| > 0.95** — konservativ, entfernt nur sehr hohe Redundanz.

- Dormann et al. (2013): \|r\| > 0.7–0.9 für Multikollinearitätsprobleme
- Kuhn & Johnson (2013): 0.90–0.95 für hochdimensionale Daten

**Validierung:** VIF \< 10 für verbleibende Features (Standard-Literaturwert).

### Feature-Auswahl bei Redundanz

Wenn \|r\| > 0.95 → Entfernung nach:

1. **Interpretierbarkeit:** Standard-Indices bevorzugt (NDVI > GNDVI), breitere Bänder (B8 > B8A)
2. **Diskriminative Power:** Feature mit höherer Target-Korrelation bleibt

### Konsistenz-Regeln

- **Cross-City:** Feature wird nur entfernt, wenn in **beiden** Städten redundant (Intersection-Regel → Transfer-Robustheit)
- **Temporal:** Wenn ein Feature-Basis redundant ist, werden **alle** temporalen Instanzen entfernt (keine Monats-Fragmentierung)

### Preservation-Constraints

- Spektrale Bänder: ≥7 von 10 erhalten (spektrale Diversität)
- Vegetationsindizes: ≥1 pro ökologischer Familie (physiologische Informationsvielfalt)

---

## Schlüssel-Parameter

| Parameter | Wert | Begründung |
| --- | --- | --- |
| Redundanz-Schwellenwert | \|r\| > 0.95 | Konservativ, Literatur-basiert |
| VIF-Grenze | \< 10 | Standard für Multikollinearitäts-Check |
| Sampling | 10.000/Stadt | Stabil ab \~5.000, genus-stratifiziert |
| Cross-City Agreement | > 0.80 Ziel | Hohe Konsistenz für Transfer |

**Typisch redundant (Literatur):** B8A↔B8 (NIR), GNDVI↔NDVI (Greenness), MSI↔NDWI (Wasser)
