# Phase 2a: Feature Extraction Methodik

**Phase:** Feature Engineering - Part 1 (Extraction)
**Letzte Aktualisierung:** 2026-01-30
**Status:** ✅ Abgeschlossen

---

## Überblick

Diese Phase extrahiert spektrale und strukturelle Features aus Phase-1-Outputs (Baumkataster, CHM, Sentinel-2 Kompositionen). Alle extrahierten Features sind Rohdaten ohne Imputation oder Transformation – NaN-Werte bleiben erhalten für spätere Quality-Control-Schritte.

**Verarbeitungs-Konstanten:**

- Projekt-CRS: EPSG:25833 (UTM Zone 33N)
- CHM-Auflösung: 1m (direkte Point-Sampling, kein Resampling)
- Sentinel-2 Auflösung: 10m (bereits in Phase 1 vorbereitet)
- Referenzjahr: 2021 (Sentinel-2 Zeitreihe)
- Extraktions-Monate: 1–12 (vollständiger Jahresverlauf)
- Batch-Größe: 50.000 Bäume (Memory-Effizienz)

**Outputs:** `data/phase_2_features/trees_with_features_{city}.gpkg` pro Stadt

---

## Tree Position Correction

### Zweck

Korrektur ungenauer Kataster-Koordinaten durch Snapping zu lokalen CHM-Maxima. Kataster-Positionen können durch GPS-Fehler, Digitalisierungsfehler oder veraltete Messungen um mehrere Meter vom tatsächlichen Kronenzentrumpunkt abweichen.

### Methodischer Ansatz

**Hybrid-Methode:** Kombination aus datengetriebenem Radius und intelligenter Peak-Auswahl.

**Phase 1 – Adaptive Radius-Bestimmung:**

1. Sample von 1000 Bäumen (reproducible, seed=42)
2. Für jeden Baum: Berechne Distanz zum nächsten CHM-Peak (Suchradius 10m)
3. Berechne 75. Perzentil (P75) dieser Distanzen
4. Setze `max_radius = ceil(P75 × safety_factor)`

**Phase 2 – Scoring-basiertes Snapping:**

1. Für jeden Baum: Extrahiere CHM-Fenster im `max_radius`
2. Score alle Pixel: `score = CHM_height × 0.7 - distance × 0.3`
3. Snap zu Pixel mit höchstem Score
4. Speichere `correction_distance` (Euklidische Distanz in Metern)

### Schlüssel-Parameter

| Parameter     | Wert | Begründung                                                             |
| ------------- | ---- | ---------------------------------------------------------------------- |
| percentile    | 75.0 | Konservativ: 75% der Bäume definieren "normale" Kataster-Genauigkeit   |
| height_weight | 0.7  | Höhe wichtiger als Nähe (höherer Baum darf weiter weg sein)            |
| safety_factor | 1.5  | Erlaubt Korrekturen bis ~85-90% Perzentil, schützt vor Outliers (>P90) |
| sample_size   | 1000 | Genug für stabile P75-Schätzung, schnell genug für große Datensätze    |

### Stadt-Adaptivität

Der adaptive Radius passt sich automatisch an die Kataster-Qualität an:

- **Berlin** (präziser Kataster): P75 ≈ 1.8m → `max_radius ≈ 3m`
- **Leipzig** (weniger präziser Kataster): P75 ≈ 2.7m → `max_radius ≈ 5m`

**Trade-off:** 85-90% der Bäume werden korrigiert, die äußersten 10-15% (wahrscheinlich Outliers) bleiben an Original-Koordinaten.

### Qualitätskriterien

- Adaptive Radius >1m und <10m (Plausibilitäts-Check)
- `correction_distance ≤ max_radius` für alle Bäume
- `correction_distance = 0.0` wenn `position_corrected = False`

### Output-Metadaten

Neue Spalten im Datensatz:

- **position_corrected** (bool): Wurde Position korrigiert?
- **correction_distance** (float): Distanz in Metern (0.0 wenn nicht korrigiert)

Zusätzliche Metadaten im Summary JSON:

```json
{
  "position_correction": {
    "adaptive_max_radius": 3.0,
    "p75_distance": 1.8,
    "safety_factor": 1.5,
    "sample_size": 1000,
    "sampled_trees": 842
  }
}
```

---

## CHM Feature Extraction

### Zweck

Extraktion der Kronenhöhe aus 1m CHM-Raster an korrigierten Baum-Standorten.

### Methodischer Ansatz

**Direct Point-Sampling:** Exakte Koordinate → 1 Pixel-Wert (kein Resampling, keine Neighborhood-Aggregation).

**Unterschied zu Legacy-Pipeline:**

- ❌ **Legacy:** 10m resampled CHM mit `CHM_mean`, `CHM_max`, `CHM_std` (Neighborhood-Aggregation)
- ✅ **Aktuell:** 1m direkt mit `CHM_1m` (keine Nachbar-Kontamination)

**Begründung:** Legacy-Methode führte zu Cross-Tree-Leakage (r=0.638 zwischen `height_m` und `CHM_mean`), da 10m-Pixel mehrere Baumkronen enthalten können.

### Schlüssel-Parameter

| Parameter  | Wert         | Begründung                                             |
| ---------- | ------------ | ------------------------------------------------------ |
| Resolution | 1m           | Maximale verfügbare Auflösung, minimiert Kontamination |
| Method     | Point-Sample | Kein Resampling, keine Aggregation                     |

### NaN-Handling

NoData-Werte (Bäume außerhalb CHM-Raster-Bounds) werden als `NaN` beibehalten für Quality-Control in Phase 2b.

**Erwartete NoData-Rate:** <2% (nur Rand-Bäume außerhalb Raster-Ausdehnung)

### Qualitätskriterien

- Feature-Name: `CHM_1m`
- Wertebereich: 0–50m (plausibel für urbane Bäume)
- CRS konsistent: EPSG:25833

### Output

Ein Feature: **CHM_1m** (Float64, may contain NaN)

---

## Sentinel-2 Feature Extraction

### Zweck

Extraktion multitemporaler spektraler Signaturen zur Erfassung phenologischer Variabilität über das Jahr.

### Datengrundlage

Monatliche Sentinel-2 Median-Kompositionen (aus Phase 1):

- 12 Monate: Januar–Dezember 2021
- 10 spektrale Bänder (B2–B8A, B11, B12)
- 13 Vegetations-Indices (berechnet in Phase 1)
- 23 Features × 12 Monate = **276 Features**

### Methodischer Ansatz

**Point-Sampling pro Monat:**

1. Für jeden Monat (01–12): Lade entsprechendes S2-Raster
2. Sample 10 Bänder + 13 Indices am Baum-Punkt
3. Speichere als `{feature_name}_{month:02d}` (z.B. `NDVI_06`, `B8_12`)

**Batch-Processing:** 50.000 Bäume pro Batch zur Vermeidung von Memory-Overflow bei großen Städten (Berlin: >800k Bäume).

### Spektrale Features

**10 Bänder:**

- **Sichtbar:** B2 (Blue), B3 (Green), B4 (Red)
- **Red-Edge:** B5 (705nm), B6 (740nm), B7 (783nm)
- **NIR:** B8 (842nm), B8A (865nm)
- **SWIR:** B11 (1610nm), B12 (2190nm)

**13 Vegetations-Indices:**

| Kategorie | Indices                              |
| --------- | ------------------------------------ |
| Broadband | NDVI, EVI, GNDVI, VARI               |
| Red-Edge  | NDre1, NDVIre, CIre, IRECI, RTVIcore |
| Water     | NDWI, MSI, NDII, kNDVI               |

### Schlüssel-Parameter

| Parameter         | Wert                 | Begründung                                                    |
| ----------------- | -------------------- | ------------------------------------------------------------- |
| Monate            | 1–12                 | Vollständiger Jahresverlauf für phenologische Differenzierung |
| Batch-Größe       | 50000                | Balance zwischen Memory-Effizienz und Performance             |
| Feature-Benennung | `{name}_{month:02d}` | Konsistente Namenskonvention, sortierbar                      |

### Phenologische Signifikanz

**Beispiel NDVI-Verlauf (Laubbäume):**

- Winter (Jan–Feb): NDVI ≈ 0.3–0.4 (laublos)
- Frühling (Mär–Apr): NDVI ↑ 0.5–0.7 (Blattaustrieb)
- Sommer (Mai–Aug): NDVI ≈ 0.7–0.85 (voll belaubt)
- Herbst (Sep–Nov): NDVI ↓ 0.6–0.4 (Laubfärbung)

→ **Unterschiedliche Gattungen zeigen charakteristische Jahresverläufe.**

### NaN-Handling

Cloud-masked Pixel oder fehlende Monate bleiben als `NaN` erhalten. Quality-Control (Phase 2b) behandelt:

- **Interior NaN** (fehlende Monate inmitten des Jahres): Linear interpoliert
- **Edge NaN** (fehlende Monate am Jahresanfang/-ende): Nearest-neighbor-fill (wenn ≤1 Monat)
- **Exzessive NaN** (>2 Monate fehlend): Baum wird entfernt

**Erwartete NoData-Rate:** <1% pro Feature (Cloud-freie Komposite aus Phase 1)

### Qualitätskriterien

- 276 Features pro Baum (23 × 12)
- Feature-Namen: `{band/index}_{01..12}`
- Wertebereich: Bands (0–10000), Indices (-1 bis +1)

### Output

276 Features: **10 Bänder × 12 Monate + 13 Indices × 12 Monate**

---

## Output-Datensatz

### Dateiformat

**Datei:** `trees_with_features_{city}.gpkg` (GeoPackage)
**CRS:** EPSG:25833 (UTM Zone 33N)
**Geometrie-Typ:** Point

### Schema

#### Metadaten-Spalten (11)

| Spalte              | Typ     | Quelle   | Beschreibung                                 |
| ------------------- | ------- | -------- | -------------------------------------------- |
| tree_id             | str     | Phase 1  | Eindeutige Baum-ID                           |
| city                | str     | Phase 1  | Stadtname (berlin/leipzig)                   |
| genus_latin         | str     | Phase 1  | Gattung lateinisch (UPPERCASE)               |
| species_latin       | str     | Phase 1  | Art lateinisch (lowercase, nullable)         |
| genus_german        | str     | Phase 1  | Gattung deutsch (nullable)                   |
| species_german      | str     | Phase 1  | Art deutsch (nullable)                       |
| plant_year          | Int64   | Phase 1  | Pflanzjahr (nullable)                        |
| height_m            | Float64 | Phase 1  | Kataster-Höhe in Metern (nullable)           |
| tree_type           | str     | Phase 1  | anlagenbaeume/strassenbaeume (nullable)      |
| position_corrected  | bool    | Phase 2a | Wurde Position korrigiert?                   |
| correction_distance | Float64 | Phase 2a | Korrektur-Distanz in Metern (0.0 wenn nicht) |

#### CHM-Feature (1)

| Feature | Typ     | Beschreibung                     |
| ------- | ------- | -------------------------------- |
| CHM_1m  | Float64 | Kronenhöhe aus 1m CHM (nullable) |

#### Sentinel-2 Features (276)

**10 Bänder × 12 Monate = 120 Features:**

- `B2_01`, `B2_02`, ..., `B2_12` (Blue)
- `B3_01`, ..., `B3_12` (Green)
- `B4_01`, ..., `B4_12` (Red)
- `B5_01`, ..., `B5_12` (Red-Edge 1)
- `B6_01`, ..., `B6_12` (Red-Edge 2)
- `B7_01`, ..., `B7_12` (Red-Edge 3)
- `B8_01`, ..., `B8_12` (NIR)
- `B8A_01`, ..., `B8A_12` (NIR narrow)
- `B11_01`, ..., `B11_12` (SWIR 1)
- `B12_01`, ..., `B12_12` (SWIR 2)

**13 Indices × 12 Monate = 156 Features:**

- `NDVI_01`, ..., `NDVI_12`
- `EVI_01`, ..., `EVI_12`
- `GNDVI_01`, ..., `GNDVI_12`
- `VARI_01`, ..., `VARI_12`
- `NDre1_01`, ..., `NDre1_12`
- `NDVIre_01`, ..., `NDVIre_12`
- `CIre_01`, ..., `CIre_12`
- `IRECI_01`, ..., `IRECI_12`
- `RTVIcore_01`, ..., `RTVIcore_12`
- `NDWI_01`, ..., `NDWI_12`
- `MSI_01`, ..., `MSI_12`
- `NDII_01`, ..., `NDII_12`
- `kNDVI_01`, ..., `kNDVI_12`

**Gesamt:** 11 Metadaten + 1 CHM + 276 S2 = **288 Spalten** + Geometrie

### Qualitätskriterien

- ✅ CRS ist EPSG:25833
- ✅ Feature-Count: 277
- ✅ Metadaten-Spalten vollständig
- ✅ Keine duplizierten tree_ids
- ✅ `correction_distance ≤ adaptive_max_radius`
- ✅ Geometrie-Typ: Point

---

## NaN-Handling

**Philosophie:** Alle NaN-Werte werden in Phase 2a **beibehalten**. Keine Imputation, keine Filterung.

**Begründung:**

1. **Transparenz:** Rohdaten bleiben unverändert, alle Entscheidungen in Quality-Control nachvollziehbar
2. **Data-Leakage-Vermeidung:** Imputation mit Train/Val/Test-Split-Awareness (Phase 2b)
3. **Qualitäts-Analyse:** NaN-Verteilung informiert über Datenqualität und systematische Probleme

**NaN-Quellen:**

- **CHM:** Bäume außerhalb Raster-Bounds (~1-2%)
- **Sentinel-2:** Cloud-masked Pixel, fehlende Szenen (<1% pro Feature)

**Nächster Schritt:** Phase 2b (Quality Control) behandelt NaN-Werte mit konservativen Methoden (within-tree interpolation, tree removal if >2 months missing).

---

## Verarbeitungs-Workflow

**Notebook:** `notebooks/runners/02a_feature_extraction.ipynb`

**Ausführung:**

1. Installation des Packages von GitHub (private repo, token-basiert)
2. Mount Google Drive
3. Laden der Feature-Config (`feature_config.yaml`)
4. Pro Stadt:
   - Tree Position Correction (adaptive radius)
   - CHM Feature Extraction (1m point sampling)
   - Sentinel-2 Feature Extraction (batch processing, 12 Monate)
   - Validierung (Schema, CRS, Feature-Count)
   - Speichern als GeoPackage
5. Generierung von Validation-Report und Execution-Log

**Idempotenz:** Skip-if-exists-Logik erlaubt sicheres Wiederholen bei Fehlern.

**Performance (geschätzt):**

- Berlin (800k Bäume): ~2-3 Stunden
- Leipzig (200k Bäume): ~30-45 Minuten

---

## Metadaten & Logs

**Output-Verzeichnisse:**

- `outputs/phase_2/metadata/` – JSON-Summaries (feature_extraction_summary.json, validation.json)
- `outputs/phase_2/logs/` – Execution logs mit Timing und Fehlerprotokollen

**Beispiel Summary JSON:**

```json
{
  "timestamp": "2026-01-30T12:34:56Z",
  "reference_year": 2021,
  "extraction_months": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
  "feature_count": 277,
  "metadata_columns": ["tree_id", "city", ...],
  "cities": {
    "berlin": {
      "status": "processed",
      "records": 842345,
      "output_path": ".../trees_with_features_berlin.gpkg",
      "summary": {
        "position_correction": {
          "corrected": 713493,
          "uncorrected": 128852,
          "correction_rate": 0.847,
          "adaptive_max_radius": 3.0,
          "p75_distance": 1.8
        },
        "chm_extraction": {
          "valid": 828234,
          "nan": 14111,
          "nan_rate": 0.017
        },
        "s2_extraction": {
          "valid_samples": 10108140,
          "nan_months_total": 84235,
          "nan_rate": 0.008
        }
      }
    }
  }
}
```

---

## Nächste Schritte

**Phase 2b – Quality Control (ausstehend):**

- Temporal Feature Selection (exp_01: JM-Distance-basierte Monats-Auswahl)
- CHM Feature Engineering (exp_02: Z-score, Percentile, Plant-Year-Threshold)
- NaN Interpolation (within-tree, no data leakage)
- NDVI Plausibility Filter (max_NDVI ≥ 0.3)
- Plant Year Filtering (Bäume zu jung für Sentinel-2 Sichtbarkeit)

**Output:** `trees_clean_{city}.gpkg` (0 NaN, quality-assured)

---

**Dokumentations-Status:** ✅ Phase 2a (Feature Extraction) vollständig dokumentiert
**Nächste Dokumentation:** Phase 2b (Quality Control) nach Implementierung der Tasks 4-7
