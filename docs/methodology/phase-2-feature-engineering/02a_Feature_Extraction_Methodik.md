# 02a_Feature_Extraction_Methodik

> Migriert aus Notion am 2026-10-28 (Seite `31f356451f668008bda7c755864b1f84`, zuletzt bearbeitet 2026-03-12). Text unverändert übernommen.

## Überblick

Extraktion spektraler und struktureller Features aus Phase-1-Outputs (Baumkataster, CHM, Sentinel-2 Komposite). Alle NaN-Werte bleiben erhalten für Quality-Control in Phase 2b.

**Output pro Stadt:** 11 Metadaten + 1 CHM + 276 S2-Features = **288 Spalten** + Geometrie

---

## Tree Position Correction

Korrektur ungenauer Kataster-Koordinaten durch **Snapping zu lokalen CHM-Maxima**.

### Adaptive Radius-Bestimmung

1. Sample von 1000 Bäumen → Suche besten lokalen CHM-Peak innerhalb 5m
2. Für jeden Baum: Extrahiere CHM-Fenster im `max_radius`
   1. Perzentil der Peak-Distanzen → `max_radius = ceil(P90)`
3. Passt sich automatisch an Kataster-Qualität an (keine Stadt-spezifischen Annahmen)

### Snap-to-Peak

Pro Baum: Lokale Maxima (Footprint 3×3) im `max_radius` detektieren, Peak mit minimalem `distance / (height + 1)` wählen.

**Score-Rationale:** Balanciert Nähe und Höhe — hohe, nahe Peaks werden bevorzugt. `+1` verhindert Division durch Null und dämpft Einfluss niedriger Vegetation.

| Parameter | Wert | Begründung |
| --- | --- | --- |
| Percentile | 90.0 | Konservativ, verhindert Fehlkorrekturen bei Ausreißern |
| Sampling Radius | 5.0m | Erwartete GPS-/Kataster-Ungenauigkeit |
| Min Peak Height | 3.0m | Entfernt Bodenrauschen und niedrige Artefakte |

---

## CHM Feature Extraction

**Direct Point-Sampling** am korrigierten Standort aus 1m CHM (kein Resampling, keine Neighborhood-Aggregation).

**Verbesserung gegenüber Legacy:**

- Legacy: 10m resampled CHM mit Neighborhood-Aggregation → Cross-Tree-Leakage (r=0.638 CHM\_mean↔height\_m)
- Aktuell: 1m direkt → minimale Kontamination durch Nachbarbäume

---

## Sentinel-2 Feature Extraction

**Point-Sampling** pro Monat (Jan–Dez 2021) aus monatlichen Median-Kompositen:

- 10 spektrale Bänder × 12 Monate = 120 Features
- 13 Vegetationsindizes × 12 Monate = 156 Features
- **Gesamt: 276 S2-Features**

Cloud-masked Pixel bleiben als NaN erhalten (Quality-Control in Phase 2b).

> 📝 **Offen:** Ergebnisse nach Pipeline-Rerun validieren (Correction-Distanzen, NaN-Raten, Feature-Counts).
