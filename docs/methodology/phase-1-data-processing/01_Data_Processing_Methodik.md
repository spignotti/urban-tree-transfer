# 01_Data_Processing_Methodik

> Migriert aus Notion am 2026-10-28 (Seite `31f356451f6680f2a66af10198f25342`, zuletzt bearbeitet 2026-03-25). Text unverändert übernommen.

## Überblick

Harmonisierung aller Eingangsdaten für die Baumgattungs-Klassifikation. Alle räumlichen Daten werden in EPSG:25833 (UTM Zone 33N) transformiert und auf Stadtgrenzen mit 500m Puffer zugeschnitten (Randeffekte).

**Referenzjahr:** 2021 (Sentinel-2 Daten)

---

## Stadtgrenzen

Autoritative Stadtgrenzen aus dem **BKG VG250 WFS** mit stadtspezifischen OGC-Filtern. Extraktion des größten Polygons aus MultiPolygon-Geometrien, Geometrie-Validierung, Reprojektion.

---

## Baumkataster

| Stadt | Quelltyp | Layer |
| --- | --- | --- |
| Berlin | WFS | Straßenbäume + Anlagenbäume (2 Layer) |
| Leipzig | WFS | Ein Layer |

### Harmonisierung

Beide Kataster werden auf ein einheitliches Schema gemappt: `tree_id`, `city`, `genus_latin` (UPPERCASE), `species_latin`, `genus_german`, `species_german`, `plant_year`, `height_m`, `tree_type`, `geometry` (Point).

### Filter

- **Räumlich:** Nur Bäume innerhalb Stadtgrenze + 500m Puffer
- **Duplikate:** Entfernung doppelter tree\_id-Werte
- **Viabilität:** Nur Gattungen mit ≥500 Samples in beiden Städten (statistische Mindestgröße)

### Positionskorrektur

Katasterkoordinaten werden per CHM-Peak-Snapping auf lokale Kronenmaxima korrigiert.

- **Berlin:** 95.5% korrigiert (864,781 / 905,132)
- **Leipzig:** 94.7% korrigiert
- Median-Displacement: nicht im Pipeline-Metadata erfasst (nur p90\_distance verfügbar). Bei Bedarf im nächsten Pipeline-Run als Metrik ergänzen.

---

## Höhenmodelle (DOM/DGM)

| Stadt | Quelle | Auflösung |
| --- | --- | --- |
| Berlin | Berlin GDI (Atom-Feed) | 1m |
| Leipzig | Sachsen GeoSN | 1m |

Kacheln werden mosaiziert, ins Projekt-CRS reprojiziert (bilineare Interpolation) und auf die gepufferte Stadtgrenze zugeschnitten. DGM wird auf das DOM-Grid aligned.

---

## CHM-Erstellung

**CHM = DOM − DGM**

Filterung invalider Werte:

- Werte \< −2m → 0 setzen (kleine Registrierungs-Artefakte zwischen DOM und DGM)
- Werte > 50m → Nodata (unrealistische Höhen, z.B. Hochhäuser)

---

## Sentinel-2 Komposite

**Quelle:** Sentinel-2 L2A (Surface Reflectance) via Google Earth Engine

### Verarbeitung (pro Monat, pro Stadt)

1. Filterung der Collection nach Geometrie und Datum
2. **Cloud-Maskierung:** Nur SCL-Klassen 4 (Vegetation) und 5 (Bare Soil) — konservative Maskierung, alle anderen Klassen (Wolken, Schatten, Wasser, Schnee, Cirrus) werden entfernt
3. Reflektanz-Clipping auf \[0, 10000\]
4. Berechnung von 13 Vegetationsindizes
5. **Median-Komposit** pro Monat

→ 12 Monate × 2 Städte = 24 Komposite, je 23 Bänder (10 spektral + 13 Indizes)

### Spektralbänder (10)

B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 — alle auf 10m resampelt

### Vegetationsindizes (13)

NDVI, EVI, GNDVI, NDre1, NDVIre, CIre, IRECI, RTVIcore, NDWI, MSI, NDII, kNDVI, VARI

> 📝 **Offen:** Literaturbasierte Begründung jedes Indexes (Referenzen zu Immitzer et al. 2019, Grabska et al. 2019, Hemmerling et al. 2021) für den Bericht ergänzen.

---

## Schlüssel-Parameter (Zusammenfassung)

| Parameter | Wert | Begründung |
| --- | --- | --- |
| Projekt-CRS | EPSG:25833 | Natives CRS beider Städte, metrisch |
| Grenzen-Puffer | 500m | Erfassung von Randbäumen für Sentinel-Pixel |
| S2-Auflösung | 10m | Native Sentinel-2 Auflösung |
| SCL-Klassen | 4, 5 | Konservative Cloud-Maskierung |
| MIN\_SAMPLES\_PER\_GENUS | 500 | Statistische Viabilität pro Stadt |
| CHM-Bereich | −2m bis 50m | Artefakt-Filterung |
