# Phase 1: Data Processing Methodik

**Phase:** Data Processing
**Letzte Aktualisierung:** 2026-01-28
**Status:** ✅ Abgeschlossen

---

## Überblick

Diese Phase beschafft und harmonisiert alle Eingangsdaten für die Baumgattungs-Klassifikation. Alle räumlichen Daten werden in ein einheitliches Koordinatenreferenzsystem (EPSG:25833) transformiert und auf Stadtgrenzen mit 500m Puffer zugeschnitten, um Randeffekte zu erfassen.

**Verarbeitungs-Konstanten:**

- Projekt-CRS: EPSG:25833 (UTM Zone 33N)
- Grenzen-Puffer: 500m (für alle Clipping-Operationen)
- Nodata-Wert: -9999.0 (alle Raster-Outputs)
- Referenzjahr: 2021 (Sentinel-2 Daten)

**Outputs:** Siehe `outputs/metadata/` für Verarbeitungszusammenfassungen und `outputs/logs/` für Ausführungsprotokolle.

---

## Stadtgrenzen (Boundaries)

### Zweck

Beschaffung autoritativer Stadtgrenzen zur Definition der räumlichen Ausdehnung für alle Verarbeitungsschritte.

### Datenquellen

Beide Städte nutzen den BKG VG250 WFS mit stadtspezifischen Filtern. Siehe `configs/cities/*.yaml` für Quell-URLs.

### Verarbeitungsschritte

1. Download der Grenzen via WFS-Request mit OGC-Filter
2. Extraktion des größten Polygons aus MultiPolygon-Geometrien (entfernt Inseln)
3. Validierung der Geometrien mit `shapely.make_valid()`
4. Reprojektion ins Projekt-CRS

### Qualitätskriterien

- Valide Polygon-Geometrien
- CRS ist EPSG:25833
- Ein Feature pro Stadt

### Output

| Datei                                  | Beschreibung |
| -------------------------------------- | ------------ |
| `data/boundaries/city_boundaries.gpkg` | Stadtgrenzen |

---

## Baumkataster (Tree Cadastres)

### Zweck

Download und Harmonisierung der Baumkataster in ein einheitliches Schema für die Feature-Extraktion.

### Datenquellen

| Stadt   | Quelltyp | Layer                                 |
| ------- | -------- | ------------------------------------- |
| Berlin  | WFS      | Straßenbäume + Anlagenbäume (2 Layer) |
| Leipzig | WFS      | Ein Layer                             |

Attribut-Mappings sind pro Stadt in `configs/cities/*.yaml` definiert.

### Verarbeitungsschritte

1. Download des Baumkatasters via WFS (paginiert, 5000 Features pro Request)
2. Harmonisierung ins einheitliche Schema (siehe unten)
3. Reprojektion ins Projekt-CRS
4. **Räumlicher Filter:** Entfernung von Bäumen außerhalb Stadtgrenze + 500m Puffer
5. **Duplikats-Entfernung:** Entfernung doppelter tree_id Werte
6. **Viabilitäts-Filter:** Nur Gattungen mit ≥500 Samples in beiden Städten

### Einheitliches Schema

| Spalte         | Typ     | Beschreibung                         |
| -------------- | ------- | ------------------------------------ |
| tree_id        | str     | Eindeutiger Identifikator            |
| city           | str     | Stadtname                            |
| genus_latin    | str     | Gattung lateinisch (UPPERCASE)       |
| species_latin  | str     | Art lateinisch (lowercase)           |
| genus_german   | str     | Gattung deutsch                      |
| species_german | str     | Art deutsch (nullable)               |
| plant_year     | Int64   | Pflanzjahr (nullable)                |
| height_m       | Float64 | Baumhöhe in Metern (nullable)        |
| tree_type      | str     | Quell-Layer Identifikator (nullable) |
| geometry       | Point   | Standort                             |

### Schlüssel-Parameter

| Parameter             | Wert | Begründung                                  |
| --------------------- | ---- | ------------------------------------------- |
| Buffer                | 500m | Erfassung von Randbäumen für Sentinel-Pixel |
| MIN_SAMPLES_PER_GENUS | 500  | Statistische Viabilität über Städte hinweg  |

### Qualitätskriterien

- Schema identisch für beide Städte
- CRS ist EPSG:25833
- Alle Bäume innerhalb gepufferter Grenze

### Output

| Datei                                   | Beschreibung                    |
| --------------------------------------- | ------------------------------- |
| `data/trees/trees_filtered_viable.gpkg` | Harmonisierte, gefilterte Bäume |

**Metadaten:** `outputs/metadata/trees_cadastre_summary.json`

---

## Höhenmodelle (DOM/DGM)

### Zweck

Beschaffung von Digitalem Oberflächenmodell (DOM) und Digitalem Geländemodell (DGM) für die CHM-Ableitung.

### Datenquellen

| Stadt   | Quelle        | Download-Typ                            |
| ------- | ------------- | --------------------------------------- |
| Berlin  | Berlin GDI    | Atom-Feed (verschachtelte XML-Struktur) |
| Leipzig | Sachsen GeoSN | ZIP-Dateiliste                          |

### Verarbeitungsschritte

1. **Berlin Atom-Feed:**
   - Parsing des Haupt-Feeds für Dataset-Feed-URL
   - Extraktion der Kachel-Download-Links aus Dataset-Feed
   - Filterung der Kacheln nach räumlicher Überschneidung mit gepufferter Grenze (via km-Grid-Koordinaten)
   - Download und Extraktion der ZIPs
   - Konvertierung von XYZ-ASCII zu GeoTIFF (1m Auflösung)

2. **Leipzig ZIP-Liste:**
   - Laden der URLs aus Konfigurationsdatei
   - Download und Extraktion aller Kacheln

3. **Gemeinsame Schritte:**
   - Mosaik der Kacheln zu einem einzelnen Raster
   - Reprojektion ins Projekt-CRS (bilineare Interpolation)
   - Clipping auf Stadtgrenze + 500m Puffer
   - Harmonisierung DOM/DGM: Alignment DGM auf DOM-Grid

### Schlüssel-Parameter

| Parameter  | Wert     | Begründung                    |
| ---------- | -------- | ----------------------------- |
| Buffer     | 500m     | Konsistent mit Baum-Filterung |
| Nodata     | -9999.0  | Konsistent über alle Raster   |
| Resampling | Bilinear | Erhalt der Höhenkontinuität   |

### Qualitätskriterien

- DOM und DGM auf identischem Grid
- CRS ist EPSG:25833
- Konsistente Nodata-Behandlung

### Output

| Datei                              | Beschreibung       |
| ---------------------------------- | ------------------ |
| `data/elevation/{city}/dom_1m.tif` | Harmonisiertes DOM |
| `data/elevation/{city}/dgm_1m.tif` | Harmonisiertes DGM |

---

## CHM-Erstellung

### Zweck

Ableitung des Canopy Height Model (CHM) aus Höhendaten für Vegetationshöhen-Features.

### Verarbeitungsschritte

1. Berechnung CHM = DOM - DGM
2. **Filterung invalider Werte:**
   - Werte < -2m → auf 0 setzen (kleine Registrierungs-Artefakte)
   - Werte > 50m → auf Nodata setzen (unrealistische Höhen)
3. Clipping auf Stadtgrenze + 500m Puffer

### Schlüssel-Parameter

| Parameter     | Wert    | Begründung                               |
| ------------- | ------- | ---------------------------------------- |
| CHM_MIN_VALID | -2.0    | Eliminierung kleiner negativer Artefakte |
| CHM_MAX_VALID | 50.0    | Entfernung unrealistischer Werte         |
| Buffer        | 500m    | Konsistent mit anderen Prozessen         |
| Nodata        | -9999.0 | Konsistent mit Höhenmodellen             |

### Qualitätskriterien

- Keine Werte unter 0 (nach Filterung)
- Keine Werte über 50m
- CRS entspricht Projekt-CRS

### Output

| Datei                        | Beschreibung           |
| ---------------------------- | ---------------------- |
| `data/chm/CHM_1m_{city}.tif` | Finales geclipptes CHM |

**Zwischenergebnisse:** `CHM_1m_{city}_raw.tif`, `CHM_1m_{city}_filtered.tif`

---

## Sentinel-2 Komposite

### Zweck

Generierung monatlicher Median-Komposite mit Spektralbändern und Vegetationsindizes.

### Datenquelle

Sentinel-2 L2A (Surface Reflectance) via Google Earth Engine: `COPERNICUS/S2_SR_HARMONIZED`

### Verarbeitungsschritte

1. Pufferung der Stadtgrenze um 500m
2. Konvertierung nach EPSG:4326 für GEE-Filterung
3. Für jeden Monat:
   - Filterung der Collection nach Geometrie und Datum
   - **Cloud-Maskierung:** Nur SCL-Klassen 4 (Vegetation) und 5 (Bare Soil) behalten
   - Clipping der Reflektanz auf [0, 10000]
   - Berechnung von 13 Vegetationsindizes
   - Erstellung des Median-Komposits
   - Export nach Google Drive

### SCL Cloud-Maskierung

Nur Pixel mit Scene Classification Layer Werten 4 oder 5 werden behalten:

- SCL 4: Vegetation
- SCL 5: Nicht vegetiert (Bare Soil)

Alle anderen Klassen (Wolken, Schatten, Wasser, Schnee, Cirrus) werden maskiert.

### Spektralbänder

**10 Bänder:** B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12

### Vegetationsindizes

**13 Indizes:** NDVI, EVI, GNDVI, NDre1, NDVIre, CIre, IRECI, RTVIcore, NDWI, MSI, NDII, kNDVI, VARI

> **📝 Note:** Die Auswahl dieser 13 Vegetationsindizes erfolgte basierend auf einer breiten Abdeckung relevanter spektraler Eigenschaften (Greenness, Red-Edge, Moisture). Eine vollständige literaturbasi erte Begründung jedes Indexes (mit Referenzen zu Immitzer et al. 2019, Grabska et al. 2019, Hemmerling et al. 2021) steht noch aus und sollte in einer wissenschaftlichen Publikation ergänzt werden.

### Schlüssel-Parameter

| Parameter   | Wert       | Begründung                       |
| ----------- | ---------- | -------------------------------- |
| Scale       | 10m        | Sentinel-2 native Auflösung      |
| CRS         | EPSG:25833 | Projekt-Standard                 |
| Buffer      | 500m       | Konsistent mit anderen Prozessen |
| SCL classes | 4, 5       | Konservative Cloud-Maskierung    |

### Qualitätskriterien

- 23 Bänder pro Komposit (10 spektral + 13 Indizes)
- CRS entspricht Projekt-CRS
- Alle monatlichen Tasks erfolgreich abgeschlossen

### Output

| Datei                              | Beschreibung                 |
| ---------------------------------- | ---------------------------- |
| `S2_{City}_{Year}_{MM}_median.tif` | Monatliches 23-Band-Komposit |

**Beispiel:** `S2_Berlin_2021_04_median.tif` (April 2021, Berlin)

**Metadaten:** `outputs/metadata/sentinel2_tasks.json`

**Komposite:** 12 Monate × 2 Städte = 24 Dateien

---

## Validierung

Alle Outputs werden validiert auf:

- CRS entspricht EPSG:25833
- Daten innerhalb Stadtgrenze (±10m Toleranz)
- Keine Null-Geometrien (Vektordaten)
- Schema entspricht erwarteten Spalten und Datentypen

**Report:** `outputs/metadata/validation_report.json`

---

## Referenzen

- Konfiguration: `configs/cities/*.yaml`
- Konstanten: `src/urban_tree_transfer/config/constants.py`
- Runner-Notebook: `notebooks/runners/01_data_processing.ipynb`
