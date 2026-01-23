# Phase 1 Data Processing: Methodology

**Phase:** Data Processing
**Last Updated:** 2026-01-23

---

## Overview

This phase acquires and harmonizes all input datasets for tree genus classification. All spatial data are processed to a consistent coordinate reference system (EPSG:25833) and clipped to city boundaries with a 500m buffer to ensure edge effects are captured.

**Processing Constants:**
- Project CRS: EPSG:25833 (UTM Zone 33N)
- Boundary buffer: 500m (applied to all clipping operations)
- Nodata value: -9999.0 (all raster outputs)

**Outputs:** See `outputs/metadata/` for processing summaries and `outputs/logs/` for execution logs.

---

## Boundaries

### Purpose

Acquire authoritative city boundaries to define spatial extent for all processing steps.

### Data Sources

Both cities use the BKG VG250 WFS with city-specific filters. See `configs/cities/*.yaml` for source URLs.

### Processing Steps

1. Download boundaries via WFS request with CQL filter
2. Extract largest polygon from MultiPolygon geometries (removes islands)
3. Validate geometries using `shapely.make_valid()`
4. Reproject to project CRS

### Quality Criteria

- Valid polygon geometries
- CRS is EPSG:25833
- One feature per city

### Output

| File | Description |
|------|-------------|
| `data/boundaries/city_boundaries.gpkg` | City boundaries |

---

## Tree Cadastres

### Purpose

Download and harmonize tree cadastres into a unified schema for feature extraction.

### Data Sources

| City | Source Type | Layers |
|------|-------------|--------|
| Berlin | WFS | Street trees + Park trees (2 layers) |
| Leipzig | WFS | Single layer |

Attribute mappings are defined per city in `configs/cities/*.yaml`.

### Processing Steps

1. Download tree cadastre via WFS
2. Harmonize to unified schema (see below)
3. Reproject to project CRS
4. **Spatial filter:** Remove trees outside city boundary + 500m buffer
5. **Duplicate removal:** Drop duplicate tree_id values
6. **Viability filter:** Retain genera with ≥500 samples in both cities

### Unified Schema

| Column | Type | Description |
|--------|------|-------------|
| tree_id | str | Unique identifier |
| city | str | City name |
| genus_latin | str | Genus (UPPERCASE) |
| species_latin | str | Species (lowercase) |
| genus_german | str | German genus name |
| species_german | str | German species name (nullable) |
| plant_year | Int64 | Planting year (nullable) |
| height_m | Float64 | Tree height (nullable) |
| tree_type | str | Source layer identifier (nullable) |
| geometry | Point | Location |

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Buffer | 500m | Capture edge trees for Sentinel pixels |
| MIN_SAMPLES_PER_GENUS | 500 | Ensure statistical viability across cities |

### Quality Criteria

- Schema matches across cities
- CRS is EPSG:25833
- All trees within buffered boundary

### Output

| File | Description |
|------|-------------|
| `data/trees/trees_filtered_viable.gpkg` | Harmonized, filtered trees |

**Metadata:** `outputs/metadata/trees_cadastre_summary.json`

---

## Elevation (DOM/DGM)

### Purpose

Acquire Digital Orthophoto Model (DOM) and Digital Ground Model (DGM) for CHM derivation.

### Data Sources

| City | Source | Download Type |
|------|--------|---------------|
| Berlin | Berlin GDI | Atom feed (nested XML structure) |
| Leipzig | Sachsen GeoSN | ZIP file list |

### Processing Steps

1. **Berlin Atom Feed:**
   - Parse main feed for dataset feed URL
   - Extract tile download links from dataset feed
   - Filter tiles by spatial intersection with buffered boundary (using km grid coordinates)
   - Download and extract ZIPs

2. **Leipzig ZIP List:**
   - Load URLs from configuration file
   - Download and extract all tiles

3. **Common Steps:**
   - Mosaic tiles into single raster
   - Reproject to project CRS (bilinear resampling)
   - Clip to city boundary + 500m buffer
   - Harmonize DOM/DGM: align DGM to DOM grid

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Buffer | 500m | Consistent with tree filtering |
| Nodata | -9999.0 | Consistent across all rasters |
| Resampling | Bilinear | Preserve elevation continuity |

### Quality Criteria

- DOM and DGM on identical grid
- CRS is EPSG:25833
- Consistent nodata handling

### Output

| File | Description |
|------|-------------|
| `data/elevation/{city}/dom_1m.tif` | Harmonized DOM |
| `data/elevation/{city}/dgm_1m.tif` | Harmonized DGM |

---

## CHM Creation

### Purpose

Derive Canopy Height Model (CHM) from elevation data for vegetation height features.

### Processing Steps

1. Compute CHM = DOM - DGM
2. **Filter invalid values:**
   - Values < -2m → set to 0 (minor registration artifacts)
   - Values > 50m → set to nodata (unrealistic heights)
3. Clip to city boundary + 500m buffer

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| CHM_MIN_VALID | -2.0 | Clamp minor negative artifacts |
| CHM_MAX_VALID | 50.0 | Remove unrealistic values |
| Buffer | 500m | Match other processing |
| Nodata | -9999.0 | Consistent with elevation |

### Quality Criteria

- No values below 0 (after filtering)
- No values above 50m
- CRS matches project CRS

### Output

| File | Description |
|------|-------------|
| `data/chm/CHM_1m_{city}.tif` | Final clipped CHM |

**Intermediate:** `CHM_1m_{city}_raw.tif`, `CHM_1m_{city}_filtered.tif`

---

## Sentinel-2 Composites

### Purpose

Generate monthly median composites with spectral bands and vegetation indices.

### Data Source

Sentinel-2 L2A (Surface Reflectance) via Google Earth Engine: `COPERNICUS/S2_SR_HARMONIZED`

### Processing Steps

1. Buffer city boundary by 500m
2. Convert to EPSG:4326 for GEE filtering
3. For each month:
   - Filter collection by geometry and date
   - **Cloud masking:** Keep only SCL classes 4 (vegetation) and 5 (bare soil)
   - Clamp reflectance to [0, 10000]
   - Compute 13 vegetation indices
   - Create median composite
   - Export to Google Drive

### SCL Cloud Masking

Only pixels with Scene Classification Layer values 4 or 5 are retained:
- SCL 4: Vegetation
- SCL 5: Not vegetated (bare soil)

All other classes (clouds, shadows, water, snow, cirrus) are masked.

### Vegetation Indices

NDVI, EVI, GNDVI, NDre1, NDVIre, CIre, IRECI, RTVIcore, NDWI, MSI, NDII, kNDVI, VARI

### Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Scale | 10m | Sentinel-2 native resolution |
| CRS | EPSG:25833 | Project standard |
| Buffer | 500m | Match other processing |
| SCL classes | 4, 5 | Conservative cloud masking |

### Quality Criteria

- 23 bands per composite (10 spectral + 13 indices)
- CRS matches project CRS
- All monthly tasks complete successfully

### Output

| File | Description |
|------|-------------|
| `S2_{City}_{Year}_{MM}_median.tif` | Monthly 23-band composite |

**Metadata:** `outputs/metadata/sentinel2_tasks.json`

---

## Validation

All outputs are validated for:
- CRS matches EPSG:25833
- Data within city boundary (±10m tolerance)
- No null geometries (vector data)
- Schema matches expected columns and dtypes

**Report:** `outputs/metadata/validation_report.json`

---

## References

- Configuration: `configs/cities/*.yaml`
- Constants: `src/urban_tree_transfer/config/constants.py`
- Runner notebook: `notebooks/runners/01_data_processing.ipynb`
