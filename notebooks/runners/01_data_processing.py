# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Urban Tree Transfer - Runner Notebook
#
# **Title:** Phase 1 Data Processing
#
# **Phase:** 1 - Data Processing
#
# **Step:** 01 - Data Processing
#
# **Purpose:** Download, harmonize, and validate Phase 1 spatial inputs for Berlin and Leipzig.
#
# **Input:** External city boundary/tree services, elevation sources, Google Earth Engine exports
#
# **Output:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_1_processing`
#
# **Author:** Silas Pignotti
#
# **Created:** 2025-01-15
#
# **Updated:** 2026-03-11
#
# Runner notebooks execute data processing only: input -> processing -> output.
# No analysis, no interpretation. They should be deterministic and repeatable.

# %%
# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================
# Runtime: CPU (Standard)
# GPU: Not required
# RAM: High-RAM recommended (elevation data can be large)
#
# Prerequisites:
#   - GITHUB_TOKEN in Colab Secrets (key icon in sidebar)
#   - Google Drive mounted (Cell 2)
# ============================================================================

import subprocess
from google.colab import userdata

token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        "GITHUB_TOKEN not found in Colab Secrets.\n"
        "1. Click the key icon in the left sidebar\n"
        "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
        "3. Toggle 'Notebook access' ON"
    )

repo_url = f"git+https://{token}@github.com/silas-workspace/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)
print("OK: Package installed")

# %%
# ============================================================================
# 2. GOOGLE DRIVE
# ============================================================================

from google.colab import drive

drive.mount("/content/drive")
print("OK: Google Drive mounted")

# %%
# ============================================================================
# 3. IMPORTS
# ============================================================================

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.config.loader import load_city_config
from urban_tree_transfer.data_processing import (
    # Boundaries
    clean_boundaries,
    download_city_boundary,
    validate_polygon_geometries,
    # Trees
    download_tree_cadastre,
    filter_trees_to_boundary,
    filter_viable_genera,
    harmonize_trees,
    remove_duplicate_trees,
    summarize_tree_cadastre,
    # Elevation
    download_elevation,
    harmonize_elevation,
    # CHM
    clip_chm_to_boundary,
    create_chm,
    filter_chm,
    # Sentinel
    check_task_status,
    create_gee_tasks,
    monitor_tasks,
    move_exports_to_destination,
)
from urban_tree_transfer.utils import (
    ExecutionLog,
    generate_validation_report,
    validate_dataset,
)
log = ExecutionLog("01_data_processing")

print("OK: Package imports complete")

# %%
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

# Large data files → Google Drive (not in repo)
DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
DATA_DIR = DRIVE_DIR / "data" / "phase_1_processing"

# Cache directories for fallback data (when WFS servers are unreachable)
TREES_CACHE_DIR = DATA_DIR / "trees" / "raw"

# Sentinel-2 staging folder (GEE exports here, then moved to DATA_DIR/sentinel2)
SENTINEL_STAGING_DIR = Path("/content/drive/MyDrive/sentinel2_staging")
SENTINEL_FINAL_DIR = DATA_DIR / "sentinel2"

# Google Earth Engine project ID (required for GEE API)
GEE_PROJECT = "treeclassifikation"

# Metadata & logs → inside data directory
METADATA_DIR = DATA_DIR / "metadata"
LOGS_DIR = DATA_DIR / "logs"

# Cities to process
CITIES = ["berlin", "leipzig"]

# Create directories
for d in [DATA_DIR, TREES_CACHE_DIR, SENTINEL_FINAL_DIR, METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Data (Drive):     {DATA_DIR}")
print(f"Trees cache:      {TREES_CACHE_DIR}")
print(f"Sentinel staging: {SENTINEL_STAGING_DIR}")
print(f"Sentinel final:   {SENTINEL_FINAL_DIR}")
print(f"Metadata:         {METADATA_DIR}")
print(f"Logs:             {LOGS_DIR}")
print(f"Cities:           {CITIES}")
print(f"GEE Project:      {GEE_PROJECT}")

# %%
# ============================================================================
# 5. BOUNDARIES
# ============================================================================

log.start_step("Boundaries")

try:
    boundaries_path = DATA_DIR / "boundaries" / "city_boundaries.gpkg"
    boundaries_path.parent.mkdir(parents=True, exist_ok=True)

    if boundaries_path.exists():
        print(f"Found existing boundaries: {boundaries_path}")
        boundaries_gdf = gpd.read_file(boundaries_path)
        validation = validate_dataset(boundaries_gdf)
        print(f"Validation: CRS valid={validation['crs']['valid']}")
        log.end_step(status="success", records=len(boundaries_gdf))
        print(f"Loaded: {boundaries_path}")
    else:
        boundaries = []
        for city in CITIES:
            config = load_city_config(city)
            print(f"Downloading boundary for {config['name']}...")
            
            gdf = download_city_boundary(config)
            gdf = clean_boundaries(gdf)  # Includes validate_polygon_geometries
            gdf["city"] = config["name"]
            boundaries.append(gdf)
            
            print(f"  - {len(gdf)} features, CRS: {gdf.crs}")

        boundaries_gdf = gpd.GeoDataFrame(
            pd.concat(boundaries, ignore_index=True),
            crs=PROJECT_CRS,
        )
        
        # Save boundaries
        boundaries_gdf.to_file(boundaries_path, driver="GPKG")
        
        # Validate
        validation = validate_dataset(boundaries_gdf)
        print(f"Validation: CRS valid={validation['crs']['valid']}")

        log.end_step(status="success", records=len(boundaries_gdf))
        print(f"Saved: {boundaries_path}")
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 6. TREES
# ============================================================================

log.start_step("Trees")

try:
    trees_path = DATA_DIR / "trees" / "trees_filtered_viable.gpkg"
    trees_path.parent.mkdir(parents=True, exist_ok=True)

    if trees_path.exists():
        print(f"Found existing trees: {trees_path}")
        trees_viable = gpd.read_file(trees_path)
        expected_cols = [
            "tree_id", "city", "genus_latin", "species_latin",
            "genus_german", "species_german", "plant_year",
            "height_m", "tree_type", "geometry",
        ]
        validation = validate_dataset(trees_viable, expected_columns=expected_cols)
        schema_valid = validation.get("schema", {}).get("valid", True)
        print(f"Validation: CRS={validation['crs']['valid']}, Schema={schema_valid}")
        
        # Generate metadata from existing data
        tree_metadata = []
        for city in CITIES:
            config = load_city_config(city)
            city_trees = trees_viable[trees_viable["city"] == config["name"]]
            if len(city_trees) > 0:
                tree_metadata.append({
                    "stage": "filtered",
                    **summarize_tree_cadastre(city_trees, config["name"]),
                })
        metadata_path = METADATA_DIR / "trees_cadastre_summary.json"
        metadata_path.write_text(json.dumps(tree_metadata, indent=2), encoding="utf-8")
        print(f"Metadata saved: {metadata_path}")
        
        log.end_step(status="success", records=len(trees_viable))
        print(f"Loaded: {trees_path}")
    else:
        trees = []
        tree_metadata = []
        
        for city in CITIES:
            config = load_city_config(city)
            trees_cfg = config.get("trees", {})
            
            if not trees_cfg.get("url") or not trees_cfg.get("mapping"):
                print(f"Skipping trees for {city}: config incomplete")
                continue
            
            print(f"Downloading tree cadastre for {config['name']}...")
            
            # Download raw data (with fallback to cached file if WFS fails)
            raw = download_tree_cadastre(config, cache_dir=TREES_CACHE_DIR)
            tree_metadata.append({
                "stage": "raw",
                **summarize_tree_cadastre(raw, config["name"]),
            })
            print(f"  - Raw: {len(raw)} trees")
            
            # Harmonize schema
            harmonized = harmonize_trees(raw, config)
            tree_metadata.append({
                "stage": "harmonized",
                **summarize_tree_cadastre(harmonized, config["name"]),
            })
            print(f"  - Harmonized: {len(harmonized)} trees")
            
            # Spatial filtering to boundary (with 500m buffer)
            boundary = boundaries_gdf[boundaries_gdf["city"] == config["name"]]
            filtered = filter_trees_to_boundary(harmonized, boundary, buffer_m=500)
            print(f"  - After spatial filter: {len(filtered)} trees")
            
            # Remove duplicates
            deduped = remove_duplicate_trees(filtered, by_id=True)
            tree_metadata.append({
                "stage": "filtered",
                **summarize_tree_cadastre(deduped, config["name"]),
            })
            print(f"  - After deduplication: {len(deduped)} trees")
            
            trees.append(deduped)

        if trees:
            trees_gdf = gpd.GeoDataFrame(
                pd.concat(trees, ignore_index=True),
                crs=PROJECT_CRS,
            )
            
            # Filter to viable genera (present in both cities)
            trees_viable = filter_viable_genera(trees_gdf)
            print(f"\nViable genera filter: {len(trees_gdf)} -> {len(trees_viable)} trees")
            
            # Save filtered trees
            trees_viable.to_file(trees_path, driver="GPKG")
            
            # Save metadata
            metadata_path = METADATA_DIR / "trees_cadastre_summary.json"
            metadata_path.write_text(json.dumps(tree_metadata, indent=2), encoding="utf-8")
            print(f"Metadata saved: {metadata_path}")
            
            # Validate
            expected_cols = [
                "tree_id", "city", "genus_latin", "species_latin",
                "genus_german", "species_german", "plant_year",
                "height_m", "tree_type", "geometry",
            ]
            validation = validate_dataset(trees_viable, expected_columns=expected_cols)
            print(f"Validation: CRS={validation['crs']['valid']}, Schema={validation['schema']['valid']}")

            log.end_step(status="success", records=len(trees_viable))
            print(f"Saved: {trees_path}")
        else:
            log.end_step(status="warning", warnings=["No tree cadastres processed"])
        
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# 7. ELEVATION (DOM/DGM)
# ============================================================================

log.start_step("Elevation")

try:
    processed = 0
    
    for city in CITIES:
        config = load_city_config(city)
        city_name = config["name"]
        city_slug = city_name.lower()
        
        dom_cfg = config.get("elevation", {}).get("dom", {})
        dgm_cfg = config.get("elevation", {}).get("dgm", {})
        buffer_m = config.get("elevation", {}).get("clip_buffer_m", 500)
        boundary = boundaries_gdf[boundaries_gdf["city"] == city_name]

        output_dir = DATA_DIR / "elevation" / city_slug
        dom_final = output_dir / "dom_1m.tif"
        dgm_final = output_dir / "dgm_1m.tif"

        if dom_final.exists() and dgm_final.exists():
            print(f"Skipping elevation for {city_name}: outputs already exist")
            processed += 1
            continue
        
        print(f"Processing elevation for {city_name}...")
        print(f"  - DOM type: {dom_cfg.get('type', 'N/A')}")
        print(f"  - DGM type: {dgm_cfg.get('type', 'N/A')}")
        
        dom_path = None
        dgm_path = None

        # Download DOM
        if dom_cfg.get("url") or dom_cfg.get("urls_file"):
            print("  - Downloading DOM...")
            dom_path = download_elevation(
                config, "dom", 
                boundary_gdf=boundary, 
                buffer_m=buffer_m,
            )
            print(f"    Saved: {dom_path}")
        else:
            print(f"  - Skipping DOM: config incomplete")

        # Download DGM
        if dgm_cfg.get("url") or dgm_cfg.get("urls_file"):
            print("  - Downloading DGM...")
            dgm_path = download_elevation(
                config, "dgm", 
                boundary_gdf=boundary, 
                buffer_m=buffer_m,
            )
            print(f"    Saved: {dgm_path}")
        else:
            print(f"  - Skipping DGM: config incomplete")

        # Harmonize elevation data
        if dom_path and dgm_path:
            print("  - Harmonizing elevation data...")
            output_dir.mkdir(parents=True, exist_ok=True)
            harmonize_elevation(dom_path, dgm_path, output_dir)
            print(f"    Output: {output_dir}")
            processed += 1

    if processed:
        log.end_step(status="success", records=processed)
    else:
        log.end_step(status="warning", warnings=["No elevation data processed"])
        
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 8. CHM (CANOPY HEIGHT MODEL)
# ============================================================================

log.start_step("CHM")

try:
    processed = 0
    
    for city in CITIES:
        config = load_city_config(city)
        city_name = config["name"]
        city_slug = city_name.lower()

        chm_final_path = DATA_DIR / "chm" / f"CHM_1m_{city_slug}.tif"
        if chm_final_path.exists():
            print(f"Skipping CHM for {city_name}: output already exists")
            processed += 1
            continue
        
        dom_path = DATA_DIR / "elevation" / city_slug / "dom_1m.tif"
        dgm_path = DATA_DIR / "elevation" / city_slug / "dgm_1m.tif"
        
        if not dom_path.exists() or not dgm_path.exists():
            print(f"Skipping CHM for {city_name}: missing elevation rasters")
            continue
        
        print(f"Creating CHM for {city_name}...")
        
        # Create raw CHM
        chm_raw_path = DATA_DIR / "chm" / f"CHM_1m_{city_slug}_raw.tif"
        chm_raw_path.parent.mkdir(parents=True, exist_ok=True)
        create_chm(dom_path, dgm_path, chm_raw_path)
        print(f"  - Raw CHM: {chm_raw_path}")
        
        # Filter CHM values
        chm_filtered_path = DATA_DIR / "chm" / f"CHM_1m_{city_slug}_filtered.tif"
        filter_chm(chm_raw_path, chm_filtered_path)
        print(f"  - Filtered CHM: {chm_filtered_path}")
        
        # Clip to boundary (with 500m buffer)
        boundary = boundaries_gdf[boundaries_gdf["city"] == city_name]
        clip_chm_to_boundary(chm_filtered_path, boundary, chm_final_path, buffer_m=500)
        print(f"  - Final CHM: {chm_final_path}")
        
        processed += 1

    if processed:
        log.end_step(status="success", records=processed)
    else:
        log.end_step(status="warning", warnings=["No CHM processed"])
        
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
# ============================================================================
# 9. SENTINEL-2 (GOOGLE EARTH ENGINE)
# ============================================================================

log.start_step("Sentinel-2")

try:
    import ee
    
    # Authenticate and initialize GEE with project
    try:
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE initialized with project: {GEE_PROJECT}")
    except ee.EEException:
        print("Authenticating with Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=GEE_PROJECT)
        print(f"GEE authenticated and initialized with project: {GEE_PROJECT}")
    
    all_tasks = []
    skipped_cities = []
    
    for city in CITIES:
        config = load_city_config(city)
        city_name = config["name"]
        boundary = boundaries_gdf[boundaries_gdf["city"] == city_name]
        
        year = config["sentinel2"]["year"]
        months = config["sentinel2"]["months"]

        expected_files = [
            SENTINEL_FINAL_DIR / f"S2_{city_name}_{year}_{month:02d}_median.tif"
            for month in months
        ]
        if all(path.exists() for path in expected_files):
            print(f"Skipping Sentinel-2 for {city_name}: outputs already exist")
            skipped_cities.append({"city": city_name, "year": year, "months": months, "files": len(expected_files)})
            continue
        
        print(f"Creating GEE tasks for {city_name}...")
        print(f"  - Year: {year}, Months: {months}")
        
        # Create tasks with 500m buffered boundary
        # Exports go to staging folder first
        tasks = create_gee_tasks(
            boundary=boundary,
            city=city_name,
            year=year,
            months=months,
            buffer_m=500,
            drive_folder="sentinel2_staging",
        )
        all_tasks.extend(tasks)
        print(f"  - Created {len(tasks)} export tasks")

    # Save metadata (even if skipped)
    sentinel_metadata = {
        "skipped_cities": skipped_cities,
        "tasks_created": len(all_tasks),
    }

    if not all_tasks:
        # All cities were skipped - generate metadata from existing files
        sentinel_files = list(SENTINEL_FINAL_DIR.glob("S2_*.tif"))
        sentinel_metadata["existing_files"] = [f.name for f in sorted(sentinel_files)]
        sentinel_metadata["status"] = "all_exists"
        
        task_info_path = METADATA_DIR / "sentinel2_tasks.json"
        task_info_path.write_text(json.dumps(sentinel_metadata, indent=2), encoding="utf-8")
        print(f"Metadata saved: {task_info_path}")
        
        log.end_step(status="success", records=len(sentinel_files))
        print("No new Sentinel-2 tasks needed")
    else:
        # Check initial status
        status = check_task_status(all_tasks)
        print(f"\nTask status: {status}")
        
        # Monitor tasks until completion (this can take a while)
        print("\nMonitoring export tasks...")
        results = monitor_tasks(all_tasks, interval_seconds=60, max_wait_minutes=180)
        print(f"Completed: {len(results['completed'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Pending: {len(results['pending'])}")
        
        # Move completed exports from staging to final destination
        if results['completed']:
            print(f"\nMoving exports to {SENTINEL_FINAL_DIR}...")
            moved = move_exports_to_destination(
                staging_dir=SENTINEL_STAGING_DIR,
                destination_dir=SENTINEL_FINAL_DIR,
                pattern="S2_*.tif",
            )
            print(f"  Moved {len(moved)} files")
        
        # Save task info
        sentinel_metadata.update({
            "status": "processed",
            "completed": len(results['completed']),
            "failed": len(results['failed']),
            "pending": len(results['pending']),
            "task_descriptions": [t.status().get("description", "N/A") for t in all_tasks],
        })
        task_info_path = METADATA_DIR / "sentinel2_tasks.json"
        task_info_path.write_text(json.dumps(sentinel_metadata, indent=2), encoding="utf-8")
        print(f"Metadata saved: {task_info_path}")

        log.end_step(status="success", records=len(results['completed']))
    
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
# ============================================================================
# SUMMARY
# ============================================================================

import rasterio

# Generate validation report for all outputs
print("Generating validation report...")

datasets_to_validate = {
    "boundaries": boundaries_gdf,
}

# Add trees if available
trees_path = DATA_DIR / "trees" / "trees_filtered_viable.gpkg"
if trees_path.exists():
    datasets_to_validate["trees"] = gpd.read_file(trees_path)

# Add CHM paths
for city in CITIES:
    chm_path = DATA_DIR / "chm" / f"CHM_1m_{city.lower()}.tif"
    if chm_path.exists():
        datasets_to_validate[f"chm_{city}"] = chm_path

validation_report = generate_validation_report(datasets_to_validate)
print(f"Validation summary: {validation_report['summary']}")

# Save validation report
validation_path = METADATA_DIR / "validation_report.json"
validation_path.write_text(json.dumps(validation_report, indent=2, default=str), encoding="utf-8")
print(f"Validation report saved: {validation_path}")

# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

# ============================================================================
# OUTPUT MANIFEST
# ============================================================================
print("\n" + "=" * 60)
print("OUTPUT SUMMARY")
print("=" * 60)

# --- Tree Dataset Summary ---
if trees_path.exists():
    trees_df = gpd.read_file(trees_path)
    print("\n--- TREE DATASET ---")
    print(f"Total trees: {len(trees_df):,}")
    print(f"Columns: {list(trees_df.columns)}")
    
    # Top genera
    if "genus_latin" in trees_df.columns:
        print("\nTop 10 genera (genus_latin):")
        genus_counts = trees_df["genus_latin"].value_counts().head(10)
        for genus, count in genus_counts.items():
            print(f"  {genus}: {count:,}")
    
    # Top species
    if "species_latin" in trees_df.columns:
        print("\nTop 10 species (species_latin):")
        species_counts = trees_df["species_latin"].dropna().value_counts().head(10)
        for species, count in species_counts.items():
            print(f"  {species}: {count:,}")
    
    # City distribution
    if "city" in trees_df.columns:
        print("\nTrees per city:")
        for city, count in trees_df["city"].value_counts().items():
            print(f"  {city}: {count:,}")

# --- Sentinel-2 Band Summary ---
print("\n--- SENTINEL-2 BANDS ---")
sentinel_files = list(SENTINEL_FINAL_DIR.glob("S2_*.tif"))
if sentinel_files:
    # Check one file to get band info
    sample_file = sentinel_files[0]
    with rasterio.open(sample_file) as src:
        band_count = src.count
        band_descriptions = src.descriptions
        print(f"Files found: {len(sentinel_files)}")
        print(f"Total bands per file: {band_count}")
        
        if band_descriptions and any(band_descriptions):
            print("Band names:")
            for i, desc in enumerate(band_descriptions, 1):
                print(f"  Band {i}: {desc or 'unnamed'}")
        else:
            # Default Sentinel-2 band order from our config
            default_bands = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12", "NDVI", "EVI", "NDWI", "NDBI"]
            if band_count == len(default_bands):
                print("Band names (from config):")
                spectral = [b for b in default_bands if b.startswith("B")]
                indices = [b for b in default_bands if not b.startswith("B")]
                print(f"  Spectral bands ({len(spectral)}): {', '.join(spectral)}")
                print(f"  Indices ({len(indices)}): {', '.join(indices)}")
            else:
                print(f"  (Band descriptions not embedded in file)")
    
    print("\nSentinel-2 files:")
    for f in sorted(sentinel_files):
        print(f"  {f.name}")
else:
    print("No Sentinel-2 files found in final directory")

# --- Elevation/CHM Summary ---
print("\n--- ELEVATION & CHM ---")
for city in CITIES:
    city_slug = city.lower()
    dom_path = DATA_DIR / "elevation" / city_slug / "dom_1m.tif"
    dgm_path = DATA_DIR / "elevation" / city_slug / "dgm_1m.tif"
    chm_path = DATA_DIR / "chm" / f"CHM_1m_{city_slug}.tif"
    
    print(f"\n{city.title()}:")
    print(f"  DOM: {'exists' if dom_path.exists() else 'MISSING'}")
    print(f"  DGM: {'exists' if dgm_path.exists() else 'MISSING'}")
    print(f"  CHM: {'exists' if chm_path.exists() else 'MISSING'}")

print("\n" + "=" * 60)
print("OK: NOTEBOOK COMPLETE")
print("=" * 60)
