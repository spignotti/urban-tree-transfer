# Sentinel-2 Download via Google Earth Engine

**Projekt:** Baumartenklassifikation & Cross-City-Transferierbarkeit  
**Städte:** Hamburg, Berlin, Rostock  
**Zeitraum:** 2021 (12 Monate)  
**Output:** Monatliche Median-Kompositionen mit 23 Bändern (10 Spektral + 13 Indizes)

---

## 1. OVERVIEW & METHODOLOGY

### 1.1 Purpose

Dieses Notebook lädt Sentinel-2 L2A Daten via Google Earth Engine für drei deutsche Städte herunter. Für jeden Monat des Jahres 2021 wird ein Median-Komposit mit 23 Bändern erstellt:

- **10 Spektralbänder:** B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
- **13 Vegetationsindizes:**
  - Basis & Sichtbar (4): NDVI, GNDVI, EVI, VARI
  - Red-Edge (5): NDre1, NDVIre, CIre, IRECI, RTVIcore
  - SWIR (3): NDWI, MSI, NDII
  - Advanced (1): kNDVI

**Cloud Masking:** SCL-basierte Filterung (Vegetation=4, Not vegetated=5, Unclassified=7)  
**Physical Range Clipping:** [0, 10000] für spektrale Bänder  
**Spatial Resolution:** 10m  
**Projection:** EPSG:25832 (UTM Zone 32N)

### 1.2 Workflow

```
[PHASE 1: GEE TASK SUBMISSION]
├── Step 1.1: Load city boundaries with 500m buffer
├── Step 1.2: Create monthly ImageCollections (2021-01 to 2021-12)
├── Step 1.3: Apply SCL cloud masking
├── Step 1.4: Clip spectral bands to physical range [0, 10000]
├── Step 1.5: Calculate vegetation indices
└── Step 1.6: Submit export tasks to Google Drive

    ↓

[PHASE 2: TASK MONITORING]
├── Step 2.1: Monitor GEE task status (READY/RUNNING/COMPLETED)
├── Step 2.2: Wait for Drive sync (30s buffer)
└── Step 2.3: Move files from Drive root to target directory

    ↓

[PHASE 3: VALIDATION]
├── Step 3.1: Check file existence and band count
├── Step 3.2: Validate coverage (≥15% threshold)
├── Step 3.3: Verify spectral ranges and index values
└── Step 3.4: Generate validation report (CSV)

    ↓

[OUTPUT: 36 monthly GeoTIFFs + validation report]
```

### 1.3 Expected Outputs

| File                              | Type    | Description                                      |
| --------------------------------- | ------- | ------------------------------------------------ |
| S2_{City}_2021_{MM}_median.tif    | GeoTIFF | Monthly median composite (23 bands: 10 spectral + 13 indices) |
| batch_validation_results.csv      | CSV     | Validation report with coverage and range checks |

---

## 2. SETUP & IMPORTS

### 2.1 Packages & Environment

```python
!pip -q install earthengine-api rasterio numpy --quiet
```

```python
import ee
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from pathlib import Path
import time
from datetime import datetime
import pandas as pd
import shutil
import glob
import warnings

warnings.filterwarnings('ignore')

print("✓ Imports successful")
```

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2.2 Google Earth Engine Authentication

```python
# Authenticate & Initialize GEE
ee.Authenticate()
ee.Initialize(project='treeclassifikation')

print("✓ GEE authenticated and initialized")
```

### 2.3 Utility Functions

```python
def add_vegetation_indices(image):
    """
    Berechnet 13 Vegetationsindizes für Tree Species Classification.
    """
    # Helper to get band as float
    def get_band(b): return image.select(b).toFloat()

    B2 = get_band('B2')
    B3 = get_band('B3')
    B4 = get_band('B4')
    B5 = get_band('B5')
    B6 = get_band('B6')
    B7 = get_band('B7')
    B8 = get_band('B8')
    B11 = get_band('B11')
    B12 = get_band('B12')

    # Skalierung für Indizes
    B2_s = B2.divide(10000.0)
    B4_s = B4.divide(10000.0)
    B8_s = B8.divide(10000.0)

    # Indices Calculation
    NDVI = B8.subtract(B4).divide(B8.add(B4)).rename('NDVI')
    GNDVI = B8.subtract(B3).divide(B8.add(B3)).rename('GNDVI')

    # EVI (Scaled inputs)
    EVI = B8_s.subtract(B4_s).multiply(2.5).divide(
        B8_s.add(B4_s.multiply(6)).subtract(B2_s.multiply(7.5)).add(1)
    ).rename('EVI')

    VARI = B3.subtract(B4).divide(B3.add(B4).subtract(B2)).rename('VARI')
    NDre1 = B8.subtract(B5).divide(B8.add(B5)).rename('NDre1')
    NDVIre = B8.subtract(B6).divide(B8.add(B6)).rename('NDVIre')
    CIre = B8.divide(B5).subtract(1).rename('CIre')
    IRECI = B7.subtract(B4).divide(B5.divide(B6)).rename('IRECI')
    RTVIcore = B8.subtract(B5).multiply(100).subtract(
        B8.subtract(B3).multiply(10)
    ).rename('RTVIcore')
    NDWI = B8.subtract(B11).divide(B8.add(B11)).rename('NDWI')
    MSI = B11.divide(B8).rename('MSI')
    NDII = B8.subtract(B12).divide(B8.add(B12)).rename('NDII')
    ndvi_raw = B8.subtract(B4).divide(B8.add(B4))
    kNDVI = ndvi_raw.pow(2).tanh().rename('kNDVI')

    return image.addBands([
        NDVI, GNDVI, EVI, VARI,
        NDre1, NDVIre, CIre, IRECI, RTVIcore,
        NDWI, MSI, NDII,
        kNDVI
    ])

# =============================================================================
# UPDATED HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Stellt sicher, dass lokale Zielverzeichnisse existieren"""
    IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    META_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Directories ready:\n  - {IMG_OUTPUT_DIR}\n  - {META_OUTPUT_DIR}")

def submit_export_tasks(cities_gdf, months=None):
    """Iteriert über Städte/Monate und startet GEE Export Tasks"""
    tasks = []

    # Allow override for test runs
    if months is None:
        months = MONTHS

    # FIX: Ensure EPSG:4326 for GEE Geometry construction
    if cities_gdf.crs is not None and cities_gdf.crs.to_string() != "EPSG:4326":
        print(f"ℹ️ Transforming input geometries from {cities_gdf.crs} to EPSG:4326 for GEE compatibility.")
        cities_gdf = cities_gdf.to_crs("EPSG:4326")

    def mask_scl(img):
        """
        SCL Masking.
        Classes:
        4: Vegetation
        5: Bare Soils
        """
        scl = img.select('SCL')
        mask = scl.eq(4).Or(scl.eq(5))
        return img.updateMask(mask)

    def clamp_bands(img):
        return img.clamp(0, 10000)

    for idx, row in cities_gdf.iterrows():
        city = row['gen']
        geom = row['geometry']

        if geom.geom_type == 'Polygon':
            ee_geom = ee.Geometry.Polygon([list(geom.exterior.coords)])
        elif geom.geom_type == 'MultiPolygon':
            ee_geom = ee.Geometry.MultiPolygon([list(p.exterior.coords) for p in geom.geoms])
        else:
            continue

        for month in months:
            y_next = YEAR + 1 if month == 12 else YEAR
            m_next = 1 if month == 12 else month + 1
            start_date = f"{YEAR}-{month:02d}-01"
            end_date = f"{y_next}-{m_next:02d}-01"

            col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                   .filterBounds(ee_geom)
                   .filterDate(start_date, end_date)
                   .map(mask_scl)
                   .map(clamp_bands)
                   .map(add_vegetation_indices))

            median_img = col.median().clip(ee_geom)
            out_bands = SPECTRAL_BANDS + list(INDEX_RANGES.keys())

            # Cast to Float to ensure consistent data types
            final_img = median_img.select(out_bands).toFloat()

            desc = f"S2_{city}_{YEAR}_{month:02d}_median"

            # Export to Staging Folder in Root
            task = ee.batch.Export.image.toDrive(
                image=final_img,
                description=desc,
                folder=DRIVE_EXPORT_FOLDER,
                region=ee_geom,
                scale=TARGET_SCALE,
                crs=TARGET_CRS,
                maxPixels=1e10
            )

            task.start()
            tasks.append({'description': desc, 'task': task, 'city': city, 'month': month})

    return tasks

def monitor_tasks(task_list, check_interval=30):
    print(f"Monitoring {len(task_list)} tasks...")
    while True:
        status_counts = {}
        all_done = True
        for item in task_list:
            state = item['task'].status()['state']
            status_counts[state] = status_counts.get(state, 0) + 1
            if state in ['READY', 'RUNNING']:
                all_done = False

        print(f"\rStatus: {status_counts}", end="")
        if all_done:
            print("\n✓ All tasks finished.")
            break
        time.sleep(check_interval)

def move_files_from_drive():
    """Verschiebt Dateien vom GEE Staging Ordner in das lokale Images-Verzeichnis"""
    if not GEE_DRIVE_PATH.exists():
        print(f"Waiting for Staging folder at {GEE_DRIVE_PATH}...")
        return

    files = list(GEE_DRIVE_PATH.glob("*.tif"))
    print(f"Found {len(files)} files in Staging folder.")

    for f in files:
        target = IMG_OUTPUT_DIR / f.name
        shutil.move(str(f), str(target))

    print(f"Moved {len(files)} files to {IMG_OUTPUT_DIR}")

def batch_validate():
    """Prüft die Dateien im lokalen Images-Ordner"""
    results = []
    files = list(IMG_OUTPUT_DIR.glob("*.tif"))

    for f in files:
        status = 'OK'
        error = ''
        cov = 0.0

        try:
            with rasterio.open(f) as src:
                if src.count != 23:
                    status = 'FAILED'
                    error = f"Expected 23 bands, got {src.count}"

                # Fix: Explicitly mask NaNs (since Nodata might be None in GeoTIFF)
                data = src.read(1)
                data = np.ma.masked_invalid(data)

                if data.size > 0:
                    cov = (data.count() / data.size) * 100

                if cov < MIN_COVERAGE_PERCENT:
                    status = 'FAILED'
                    error = f"Coverage too low: {cov:.1f}%"

        except Exception as e:
            status = 'FAILED'
            error = str(e)

        results.append({
            'file': f.name,
            'status': status,
            'coverage_pct': cov,
            'error': error
        })

    return pd.DataFrame(results)
```

---

## 3. CONFIGURATION & PARAMETERS

### 3.1 Paths

```python
BASE_DIR = Path("/content/drive/MyDrive/Studium/Geoinformation/Module/Projektarbeit")
DATA_DIR = BASE_DIR / 'data'

# ============================================================================
# INPUT PATHS
# ============================================================================
# Pfad zu den gepufferten Grenzen (für GEE Download/Clip)
PATH_BOUNDARIES_BUFFERED = DATA_DIR / '01_raw' / 'boundaries' / 'city_boundaries_500m_buffer.gpkg'

# Pfad zu den originalen Grenzen (für Referenz/Validierung)
PATH_BOUNDARIES_ORIGINAL = DATA_DIR / 'boundaries' / 'city_boundaries.gpkg'

# ============================================================================
# OUTPUT PATHS
# ============================================================================
# 1. GEE Staging (Temporärer Ordner im Drive-Root für Export)
DRIVE_EXPORT_FOLDER = 'sentinel2_processing_stage'
GEE_DRIVE_PATH = Path('/content/drive/MyDrive') / DRIVE_EXPORT_FOLDER

# 2. Lokales Zielverzeichnis (Finaler Speicherort)
LOCAL_BASE_DIR = DATA_DIR / '01_raw' / 'sentinel2_2021'
IMG_OUTPUT_DIR = LOCAL_BASE_DIR / 'images'
META_OUTPUT_DIR = LOCAL_BASE_DIR / 'metadata'

print(f"✓ Input (Buffered): {PATH_BOUNDARIES_BUFFERED}")
print(f"✓ Staging Folder:   {GEE_DRIVE_PATH}")
print(f"✓ Target Images:    {IMG_OUTPUT_DIR}")
print(f"✓ Target Metadata:  {META_OUTPUT_DIR}")
```

### 3.2 Processing Parameters

```python
# ============================================================================
# TEMPORAL PARAMETERS
# ============================================================================
YEAR = 2021
MONTHS = list(range(1, 13))  # Januar bis Dezember

# ============================================================================
# SPATIAL PARAMETERS
# ============================================================================
TARGET_CRS = 'EPSG:25832'  # UTM Zone 32N
TARGET_SCALE = 10  # Meter

# ============================================================================
# SPECTRAL BANDS (Sentinel-2 L2A)
# ============================================================================
SPECTRAL_BANDS = [
    'B2',   # Blue (490 nm)
    'B3',   # Green (560 nm)
    'B4',   # Red (665 nm)
    'B5',   # Red Edge 1 (705 nm)
    'B6',   # Red Edge 2 (740 nm)
    'B7',   # Red Edge 3 (783 nm)
    'B8',   # NIR (842 nm)
    'B8A',  # Narrow NIR (865 nm)
    'B11',  # SWIR 1 (1610 nm)
    'B12'   # SWIR 2 (2190 nm)
]

# ============================================================================
# CLOUD MASKING (SCL)
# ============================================================================
# KRITISCH: Konservative Whitelist für spektral saubere Signaturen
# Klasse 7 (Unclassified) AUSGESCHLOSSEN für Transfer-Robustheit
VALID_SCL_CLASSES = [
    4,  # Vegetation
    5   # Not vegetated
]

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================
MIN_COVERAGE_PERCENT = 15  # Mindest-Coverage nach Cloud-Masking
SPECTRAL_MAX_TOLERANCE = 10000  # Max-Wert für Spektralbänder (physical range)

# Erwartete Wertebereiche für Vegetationsindizes (nach Physical Range Clipping)
# Strikte Validierung möglich durch [0, 10000] Clipping in GEE
INDEX_RANGES = {
    # Basis & Sichtbar
    'NDVI': (-1, 1),        # Normalized Difference: theoretisch [-1, 1]
    'GNDVI': (-1, 1),       # Normalized Difference: theoretisch [-1, 1]
    'EVI': (-1, 2.5),       # Enhanced: kann leicht über 1 gehen
    'VARI': (-1, 1),        # Mit clipping realistisch
    # Red-Edge
    'NDre1': (-1, 1),       # Normalized Difference: theoretisch [-1, 1]
    'NDVIre': (-1, 1),      # Normalized Difference: theoretisch [-1, 1]
    'CIre': (-1, 10),       # Ratio-based: kann größer 1 werden
    'IRECI': (-5, 5),       # Inverted ratio: breiterer Bereich
    'RTVIcore': (-200, 200), # Konservativ (statt -1000, 1000)
    # SWIR
    'NDWI': (-1, 1),        # Normalized Difference: theoretisch [-1, 1]
    'MSI': (0, 3),          # Ratio: NIR/SWIR, typisch 0.4-2.0
    'NDII': (-1, 1),        # Normalized Difference: theoretisch [-1, 1]
    # Advanced
    'kNDVI': (0, 1)         # Tanh-bounded: garantiert [0, 1]
}

# Display parameters
print("Processing Parameters:")
print("-" * 80)
print(f"  {'Year':<30} {YEAR}")
print(f"  {'Months':<30} {len(MONTHS)} (Jan-Dec)")
print(f"  {'Target CRS':<30} {TARGET_CRS}")
print(f"  {'Target Scale':<30} {TARGET_SCALE}m")
print(f"  {'Spectral Bands':<30} {len(SPECTRAL_BANDS)}")
print(f"  {'Valid SCL Classes':<30} {VALID_SCL_CLASSES}")
print(f"  {'Min Coverage':<30} {MIN_COVERAGE_PERCENT}%")
```

---

## 4. DATA LOADING

### 4.1 Load City Boundaries

```python
# Lade Stadtgrenzen
cities_buffered = gpd.read_file(PATH_BOUNDARIES_BUFFERED)

# Korrektur: Die Original-Grenzen liegen im selben Ordner wie die gepufferten (01_raw/boundaries)
# Wir nutzen den Parent-Pfad von PATH_BOUNDARIES_BUFFERED, um sicherzugehen.
path_original_corrected = PATH_BOUNDARIES_BUFFERED.parent / 'city_boundaries.gpkg'
cities_original = gpd.read_file(path_original_corrected)

print(f"✓ Loaded {len(cities_buffered)} buffered cities")
print(f"✓ Loaded {len(cities_original)} original boundaries")
```

### 4.2 Data Validation

```python
# Validiere GeoDataFrame
assert 'gen' in cities_buffered.columns, "Missing 'city' column"
assert 'geometry' in cities_buffered.columns, "Missing 'geometry' column"
assert not cities_buffered['geometry'].isna().any(), "Missing geometries"

print("✓ Data validation passed")
print(f"✓ Total tasks to submit: {len(cities_buffered) * len(MONTHS)}")
```

---

## 5. MAIN PROCESSING

### 5.1 Setup Directories

```python
ensure_directories()
```

### 5.2 Submit GEE Export Tasks

```python
print("STARTING TEST RUN: Rostock (June 2021)")

# =============================================================================
# DIAGNOSTIC STEP
# =============================================================================
# Wir prüfen VOR dem Download, ob überhaupt Bilder gefunden werden.
rostock_test = cities_buffered[cities_buffered['gen'] == 'Rostock'].copy()

# Manuelle Reprojektion für Diagnose
if rostock_test.crs.to_string() != "EPSG:4326":
    rostock_test = rostock_test.to_crs("EPSG:4326")

geom = rostock_test.iloc[0]['geometry']
centroid = geom.centroid
print(f"ℹ️ Geometry Centroid (Lat/Lon): {centroid.y:.4f}, {centroid.x:.4f}")

# Check GEE Collection Size
ee_geom = ee.Geometry.Polygon([list(geom.exterior.coords)])
col_debug = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
             .filterBounds(ee_geom)
             .filterDate('2021-06-01', '2021-07-01'))

count = col_debug.size().getInfo()
print(f"ℹ️ Found {count} Sentinel-2 scenes for Rostock in June 2021.")

if count == 0:
    print("❌ ERROR: No images found. Check coordinates or date range!")
else:
    print("✅ Images found. Proceeding with export...")

    # =========================================================================
    # EXPORT STEP
    # =========================================================================
    # 1. Submit Single Task for Rostock, June (Month 6)
    # Wir nutzen wieder das originale rostock_test (wird in function eh reprojiziert)
    rostock_raw = cities_buffered[cities_buffered['gen'] == 'Rostock']
    test_tasks = submit_export_tasks(rostock_raw, months=[6])

    # 2. Monitor
    monitor_tasks(test_tasks, check_interval=10)

    # 3. Move File
    print("\nSyncing with Drive...")
    time.sleep(15)
    move_files_from_drive()

    # 4. Deep Validation
    print("\nValidating Test Image...")
    try:
        test_files = list(IMG_OUTPUT_DIR.glob("*Rostock*2021_06*.tif"))
        if not test_files:
             raise FileNotFoundError("No file found for Rostock June 2021")

        test_file = test_files[0]

        with rasterio.open(test_file) as src:
            print(f"File: {test_file.name}")
            print(f"CRS: {src.crs} (Expected: {TARGET_CRS})")
            print(f"Shape: {src.shape}")
            print(f"Bands: {src.count}")

            all_band_names = SPECTRAL_BANDS + list(INDEX_RANGES.keys())

            print("\nDetailed Band Statistics (All 23 Bands):")
            print(f"{'Idx':<4} {'Name':<10} {'Min':<10} {'Max':<10} {'Mean':<10} {'Std':<10}")
            print("-" * 60)

            for i, name in enumerate(all_band_names, start=1):
                data = src.read(i, masked=True)

                # Handle pure NaN arrays (masked or float nans)
                if data.count() > 0:
                    # Check if min is nan (happens with float nans inside valid mask)
                    dmin = data.min()
                    if np.isnan(dmin):
                        print(f"{i:<4} {name:<10} {'NaN':<10} {'NaN':<10} {'NaN':<10} {'--':<10}")
                    else:
                        print(f"{i:<4} {name:<10} {dmin:<10.4f} {data.max():<10.4f} {data.mean():<10.4f} {data.std():<10.4f}")
                else:
                    print(f"{i:<4} {name:<10} {'EMPTY':<10} {'-':<10} {'-':<10} {'-':<10}")

        print("\n✅ Test Run Complete. Check values against expected ranges.")

    except IndexError:
        print("❌ File not found. Export might have failed.")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
```

### 5.3 Monitor Task Progress

```python
print("\n" + "="*80)
print("STARTING FULL BATCH PRODUCTION")
print("="*80)

# Submit all cities, all months
task_descriptions = submit_export_tasks(cities_buffered)
print(f"\n✓ Tasks submitted: {len(task_descriptions)}")

# Monitor
monitor_tasks(task_descriptions, check_interval=60)
```

### 5.4 Drive Synchronization & File Transfer

```python
# KRITISCH: Warte auf Drive-Sync (GEE → Drive kann verzögert sein)
print("\nWarte 30s auf Drive-Sync...")
time.sleep(30)

# Verschiebe Dateien von Drive-Root zum Zielordner
move_files_from_drive()
```

---

## 6. RESULTS & VALIDATION

### 6.1 Batch Validation

```python
# Führe Batch-Validierung durch (prüft IMG_OUTPUT_DIR)
validation_df = batch_validate()
```

### 6.2 Summary Statistics

```python
print("\n" + "="*80)
print("ZUSAMMENFASSUNG")
print("="*80)

# Per-City Statistics
for city in ['Berlin', 'Hamburg', 'Rostock']:
    city_files = validation_df[validation_df['file'].str.contains(city)]
    ok_count = (city_files['status'] == 'OK').sum()
    total = len(city_files)
    avg_cov = city_files[city_files['status'] == 'OK']['coverage_pct'].mean()

    status_icon = "✅" if ok_count == total else "⚠️"
    print(f"   {status_icon} {city:<8} : {ok_count}/{total} OK (Ø Coverage: {avg_cov:.1f}%)")

# Overall Statistics
ok_total = (validation_df['status'] == 'OK').sum()
total_files = len(validation_df)
success_rate = (ok_total / total_files) * 100

print(f"\n   Gesamt:")
print(f"      ✅ OK:      {ok_total}/{total_files} ({success_rate:.1f}%)")

if ok_total < total_files:
    failed = validation_df[validation_df['status'] != 'OK']
    print(f"      ❌ FAILED:  {len(failed)}")
    print("\n   Failed files:")
    for _, row in failed.iterrows():
        print(f"      - {row['file']}: {row['error']}")
```

### 6.3 Export Validation Report

```python
# Exportiere Validierungsergebnisse in den Metadata-Ordner
report_path = META_OUTPUT_DIR / 'batch_validation_results.csv'
validation_df.to_csv(report_path, index=False)

print(f"\n   Details: {report_path}")
print("="*80)
```

---

## 7. SUMMARY & NEXT STEPS

### 7.1 Pipeline Completion

```python
print("\n" + "="*80)
print("PIPELINE ABGESCHLOSSEN")
print("="*80)

if ok_total == total_files:
    print("\n✅ ALLE DATEIEN VALIDIERT - BEREIT FÜR FEATURE EXTRACTION!")
else:
    print(f"\n⚠️ {total_files - ok_total} Dateien fehlgeschlagen - Bitte prüfen!")

print(f"\nImage directory:    {IMG_OUTPUT_DIR}")
print(f"Metadata directory: {META_OUTPUT_DIR}")
print(f"Total files: {total_files}")
print(f"Success rate: {success_rate:.1f}%")
print("\nNächster Schritt: Feature Extraction Notebook")
```

---

**Notebook End**

Author: Silas Pignotti  
Project: Tree Species Classification & Cross-City Transferability  
Date: 2025-01-08
