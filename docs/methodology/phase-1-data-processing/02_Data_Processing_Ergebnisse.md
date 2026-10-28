# 02_Data_Processing_Ergebnisse

**Phase:** Data Processing

**Ausführungsdatum:** 03.02.2026

**Ausführungszeit:** 14:45:34 - 15:46:32 (ca. 61 Minuten)

**Status:** ✅ Erfolgreich abgeschlossen

---

## Überblick

Phase 1 wurde vollständig durchgeführt und hat alle erforderlichen Eingangsdaten für Berlin und Leipzig beschafft und harmonisiert. Alle Datensätze wurden erfolgreich validiert und liegen im einheitlichen Koordinatensystem EPSG:25833 vor.

**Execution Log:** [outputs/phase\_1/logs/01\_data\_processing\_execution.json](../../../outputs/phase_1_processing/logs/01_data_processing_execution.json)

---

## Ergebnisse nach Verarbeitungsschritt

### 1. Stadtgrenzen (Boundaries)

**Ausführungszeit:** 14:46:54 - 14:47:22 (28 Sekunden)

**Status:** ✅ Erfolgreich

**Ergebnisse:**

- **Anzahl Datensätze:** 2 (Berlin, Leipzig)
- **Koordinatensystem:** EPSG:25833 ✓
- **Geometrien:** Valide Polygone, keine Null-Geometrien
- **Format:** GeoPackage

**Output:**

- Berlin: Stadtgrenze aus BKG VG250 WFS
- Leipzig: Stadtgrenze aus BKG VG250 WFS
- Beide Grenzen auf größtes Polygon reduziert (ohne Exklaven)

---

### 2. Baumkataster (Trees)

**Ausführungszeit:** 14:47:22 - 14:47:39 (17 Sekunden)

**Status:** ✅ Erfolgreich

**Ergebnisse:**

- **Anzahl Datensätze:** 1.072.999 Bäume (Berlin + Leipzig kombiniert)
- **Koordinatensystem:** EPSG:25833 ✓
- **Geometrien:** Keine Null-Geometrien (100% valide)
- **Format:** GeoPackage

**Harmonisierung:**

- Einheitliches Schema mit folgenden Attributen:
  - `tree_id`: Eindeutige Baum-ID
  - `genus_latin`: Gattung (lateinisch)
  - `species_latin`: Art (lateinisch)
  - `genus_german`: Gattung (deutsch)
  - `species_german`: Art (deutsch)
  - `plant_year`: Pflanzjahr
  - `height_m`: Baumhöhe in Metern
  - `geometry`: Punktgeometrie (EPSG:25833)

**Filterung:**

- Nur Gattungen mit ≥500 Exemplaren beibehalten (MIN\_SAMPLES\_PER\_GENUS)
- Nur Bäume innerhalb Stadtgrenze + 500m Puffer
- Ungültige oder fehlende Gattungsangaben entfernt

---

### 3. Höhenmodelle (Elevation)

**Ausführungszeit:** 14:47:42 - 15:23:41 (36 Minuten)

**Status:** ✅ Erfolgreich

**Ergebnisse:**

- **Anzahl Städte:** 2 (Berlin, Leipzig)
- **Modelle pro Stadt:** DOM (Oberflächenmodell) + DGM (Geländemodell)
- **Auflösung:** 1m
- **Format:** GeoTIFF mit LZW-Kompression

**Berlin:**

- Quelle: Berlin GDI Atom-Feed
- DOM: Digitales Oberflächenmodell (1m)
- DGM: Digitales Geländemodell (1m)
- Kacheln: Automatisch gefiltert nach räumlicher Überschneidung

**Leipzig:**

- Quelle: Sachsen GeoSN
- DOM: Digitales Oberflächenmodell (1m)
- DGM: Digitales Geländemodell (1m)
- Download: ZIP-Liste aus Konfiguration

**Verarbeitung:**

- Kacheln zu Mosaik zusammengefügt
- Reprojiziert nach EPSG:25833 (bilineare Interpolation)
- Geclippt auf Stadtgrenze + 500m Puffer
- DOM und DGM auf identischem Grid aligned
- Nodata-Wert: -9999.0

---

### 4. Canopy Height Model (CHM)

**Ausführungszeit:** 15:23:41 - 15:25:37 (2 Minuten)

**Status:** ✅ Erfolgreich

**Ergebnisse:**

- **Anzahl CHMs:** 2 (Berlin, Leipzig)
- **Auflösung:** 1m
- **Koordinatensystem:** EPSG:25833 ✓
- **Format:** GeoTIFF mit LZW-Kompression

**Berechnung:**

- CHM = DOM - DGM
- Filterung: Werte \< -2m → 0 gesetzt (Artefakt-Eliminierung)
- Filterung: Werte > 50m → Nodata (unrealistische Höhen entfernt)
- Clipping: Stadtgrenze + 500m Puffer
- Nodata-Wert: -9999.0

**Wertebereich:**

- Minimum: 0m (nach Filterung)
- Maximum: ≤50m (nach Filterung)
- Typische Baumhöhen: 5-30m

**Berlin CHM:**

- Flächendeckende Vegetation erfasst
- Urbane Bäume und Parks repräsentiert

**Leipzig CHM:**

- Flächendeckende Vegetation erfasst
- Stadtwald und Straßenbäume repräsentiert

---

### 5. Sentinel-2 Komposite

**Ausführungszeit:** Externe Verarbeitung via Google Earth Engine

**Status:** ✅ Alle Dateien vorhanden

**Ergebnisse:**

- **Anzahl Komposite:** 24 Dateien (12 Monate × 2 Städte)
- **Referenzjahr:** 2021
- **Auflösung:** 10m
- **Koordinatensystem:** EPSG:25833
- **Format:** GeoTIFF

**Spektralbänder (10):**

- B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

**Vegetationsindizes (13):**

- NDVI, EVI, GNDVI, NDre1, NDVIre, CIre, IRECI, RTVIcore, NDWI, MSI, NDII, kNDVI, VARI

**Gesamt: 23 Bänder pro Komposit**

**Cloud-Maskierung:**

- Nur SCL-Klassen 4 (Vegetation) und 5 (Bare Soil) verwendet
- Konservative Maskierung für höchste Datenqualität

**Komposite Berlin 2021:**

- Januar bis Dezember: 12 monatliche Median-Komposite
- Alle Tasks erfolgreich abgeschlossen
- Dateien: `S2_Berlin_2021_01_median.tif` bis `S2_Berlin_2021_12_median.tif`

**Komposite Leipzig 2021:**

- Januar bis Dezember: 12 monatliche Median-Komposite
- Alle Tasks erfolgreich abgeschlossen
- Dateien: `S2_Leipzig_2021_01_median.tif` bis `S2_Leipzig_2021_12_median.tif`

---

## Validierungsergebnisse

**Validation Report:** [outputs/phase\_1/metadata/validation\_report.json](../../../outputs/phase_1_processing/metadata/validation_report.json)

**Geprüfte Datensätze:** 4

| Datensatz | CRS-Validierung | Null-Geometrien | Gesamtstatus |
| --- | --- | --- | --- |
| Boundaries | ✅ EPSG:25833 | ✅ 0 von 2 | ✅ Valide |
| Trees | ✅ EPSG:25833 | ✅ 0 von 1.072.999 | ✅ Valide |
| CHM Berlin | ✅ EPSG:25833 | N/A (Raster) | ✅ Valide |
| CHM Leipzig | ✅ EPSG:25833 | N/A (Raster) | ✅ Valide |

**Validierungsstatus:** ✅ Alle Datensätze valide (4 von 4)

---

## Datenvolumen und Speicherung

**Datenspeicherort:** Google Drive (externe Speicherung)

**Metadaten:** Repository unter `outputs/phase_1/`

**Geschätztes Datenvolumen:**

- Boundaries: \~1 MB (2 GeoPackages)
- Trees: \~200 MB (1 GeoPackage mit 1,07 Mio. Features)
- Höhenmodelle (DOM/DGM): \~8 GB (4 GeoTIFFs à \~2 GB)
- CHM: \~4 GB (2 GeoTIFFs à \~2 GB)
- Sentinel-2 Komposite: \~12 GB (24 GeoTIFFs à \~500 MB)

**Gesamtvolumen:** \~24 GB

---

## Zeitaufwand nach Komponente

| Schritt | Dauer | Anteil |
| --- | --- | --- |
| Boundaries | 28 Sekunden | \<1% |
| Trees | 17 Sekunden | \<1% |
| Elevation (Download + Mosaic) | 36 Minuten | 59% |
| CHM (Berechnung) | 2 Minuten | 3% |
| Sentinel-2 (GEE) | Extern | N/A |
| **Gesamt (Lokal)** | **61 Minuten** | **100%** |

**Flaschenhals:** Elevation-Download und Kachel-Mosaizierung (36 Min = 59% der Laufzeit)

---

## Resümee: Phase 1 Abschluss

### ✅ Erfüllte Akzeptanzkriterien

1. **Datenharmonisierung:**
   - ✅ Alle Datensätze im einheitlichen CRS (EPSG:25833)
   - ✅ Baumkataster auf gemeinsames Schema harmonisiert
   - ✅ 1.072.999 Bäume erfolgreich verarbeitet
2. **Datenqualität:**
   - ✅ 100% valide Geometrien (keine Null-Geometrien)
   - ✅ CRS-Validierung für alle Datensätze bestanden
   - ✅ CHM-Wertebereich plausibel (0-50m)
3. **Vollständigkeit:**
   - ✅ Beide Städte (Berlin, Leipzig) komplett verarbeitet
   - ✅ Alle 4 Datenquellen erfolgreich integriert
   - ✅ 24 Sentinel-2 Komposite vorhanden (12 Monate × 2 Städte)
4. **Dokumentation:**
   - ✅ Execution Log erstellt und gespeichert
   - ✅ Validation Report generiert
   - ✅ Sentinel-2 Tasks dokumentiert
   - ✅ Methodik dokumentiert
5. **Technische Standards:**
   - ✅ Reproduzierbare Pipeline (Runner-Notebook)
   - ✅ Type-gecheckte Module (Pyright)
   - ✅ Konsistente Datenformate (GeoPackage, GeoTIFF)

### 🎯 Status der Phase

**Phase 1: Data Processing ist vollständig abgeschlossen.**

Alle erforderlichen Eingangsdaten für die nachfolgende Feature-Engineering-Phase (Phase 2) liegen vor. Die Datenqualität ist durchgehend hoch, und alle Validierungskriterien wurden erfüllt.

### 📋 Übergabe an Phase 2

Die folgenden Datensätze stehen für Phase 2 (Feature Engineering) bereit:

1. **Baumkataster:** 1,07 Mio. harmonisierte Bäume mit Gattungsinformation
2. **CHM:** Hochauflösende Vegetationshöhen (1m) für beide Städte
3. **Sentinel-2:** 24 monatliche Komposite mit 23 Bändern (10 spektral + 13 Indizes)
4. **Stadtgrenzen:** Definierte räumliche Ausdehnung für beide Städte

**Nächster Schritt:** Feature-Extraktion aus Sentinel-2 und CHM für jeden Baum (Phase 2)

---

**Erstellt:** 06.02.2026

**Basis-Dokumentation:** [01\_Data\_Processing\_Methodik.md](01_Data_Processing_Methodik.md)

**Execution Log:** [01\_data\_processing\_execution.json](../../../outputs/phase_1_processing/logs/01_data_processing_execution.json)
