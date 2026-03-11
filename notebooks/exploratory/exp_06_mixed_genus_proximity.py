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

# %%
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended for large datasets
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================

import subprocess
from google.colab import userdata

# Get GitHub token from Colab Secrets (for private repo access)
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        "GITHUB_TOKEN not found in Colab Secrets.\n"
        "1. Click the key icon in the left sidebar\n"
        "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
        "3. Toggle 'Notebook access' ON"
    )

# Install package from private GitHub repo
repo_url = f"git+https://{token}@github.com/SilasPignotti/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

print("OK: Package installed")


# %%
# Mount Google Drive for data files
from google.colab import drive

drive.mount("/content/drive")

print("Google Drive mounted")


# %%
# Package imports
from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
from urban_tree_transfer.utils import ExecutionLog
from urban_tree_transfer.feature_engineering.proximity import apply_proximity_filter

from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import pandas as pd
import numpy as np

log = ExecutionLog("exp_06_proximity")

print("OK: Package imports complete")


# %%
# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"

METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

CITIES = ["berlin", "leipzig"]
THRESHOLDS_TO_TEST = [5, 10, 15, 20, 30]  # meters

for d in [METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Input (Phase 2):  {INPUT_DIR}")
print(f"Output (JSONs):   {METADATA_DIR}")
print(f"Logs (Drive):     {LOGS_DIR}")
print(f"Cities:           {CITIES}")
print(f"Random seed:      {RANDOM_SEED}")



# %%
# ============================================================
# SECTION 1: Data Loading & Validation
# ============================================================

log.start_step("Data Loading")

required_columns = {"geometry", "genus_latin", "city"}
city_data = {}

total_trees = 0

for city in CITIES:
    path = INPUT_DIR / f"trees_clean_{city}.gpkg"
    print(f"Loading: {path}")
    gdf = gpd.read_file(path)

    # Ensure city column exists
    if "city" not in gdf.columns:
        gdf["city"] = city
        print(f"  Added missing city column for {city}.")

    missing = required_columns - set(gdf.columns)
    if missing:
        raise ValueError(f"Missing required columns for {city}: {missing}")

    # Validate CRS (EPSG:25833)
    if gdf.crs is None:
        raise ValueError(f"CRS missing for {city}. Expected {PROJECT_CRS}.")

    if gdf.crs.to_string() != PROJECT_CRS:
        print(f"  Reprojecting {city} to {PROJECT_CRS} (was {gdf.crs.to_string()})")
        gdf = gdf.to_crs(PROJECT_CRS)

    gdf = gdf[gdf["genus_latin"].notna()].copy()

    print(f"  {city}: {len(gdf):,} rows")
    city_data[city] = gdf
    total_trees += len(gdf)

log.end_step(status="success", records=total_trees)


# %% [markdown]
# ## Section 2: Geometric Definition and Pixel Footprint Analysis
#
# Klarstellung der geometrischen Interpretation des Proximity-Thresholds im Kontext
# der Sentinel-2 Pixelauflosung.
#
# **Sentinel-2 Kontext:**
# - Pixelgrose: 10m x 10m (B2, B3, B4, B8)
# - Pixel-Diagonale: ~14.1m
# - Baumposition: Punktgeometrie (Zentroid)
#
# **Distanz-Metrik:**
# - Euklidische Distanz zwischen Baum-Zentroiden (Punkt-zu-Punkt)
# - 20m Threshold ≈ 2-Pixel Separation (Edge-to-Edge Kontakt an der Grenze)
#
# **Kontaminationszonen:**
# - < 10m: Vollstandige Pixel-Uberlappung (HIGH)
# - 10-15m: Partielle Uberlappung (MEDIUM)
# - 15-20m: Edge-Contact (LOW)
# - > 20m: Keine Uberlappung (NONE)
#

# %%
# ============================================================
# SECTION 3: Nearest Different-Genus Distance (OPTIMIZED)
# ============================================================
from scipy.spatial import cKDTree

log.start_step("Nearest Neighbor Analysis")

np.random.seed(RANDOM_SEED)

def compute_nearest_diff_genus_fast(gdf: gpd.GeoDataFrame, genus_col: str = "genus_latin", k: int = 50) -> pd.Series:
    """
    Computes distance to the nearest neighbor of a different genus using cKDTree.
    Optimization: O(N log N) instead of O(N^2) using Spatial Indexing.
    """
    # 1. Prepare coordinates and genus codes
    # We extract X/Y coordinates for the cKDTree
    coords = np.column_stack((gdf.geometry.x, gdf.geometry.y))
    
    # Convert genus strings to integer codes for faster comparison
    genus_codes, _ = pd.factorize(gdf[genus_col])
    
    # 2. Build spatial index (extremely fast for point data)
    print(f"  Building spatial index for {len(gdf):,} trees...")
    tree = cKDTree(coords)
    
    # 3. Query k nearest neighbors (k+1 to include the point itself)
    # asking for 50 neighbors is usually enough to find a different genus in urban settings
    print("  Querying nearest neighbors...")
    dists, idxs = tree.query(coords, k=k+1, workers=-1)
    
    # 4. Find first neighbor with different genus
    # Retrieve the genus codes of the neighbors found
    neighbor_genera = genus_codes[idxs]
    self_genera = genus_codes.reshape(-1, 1)
    
    # Boolean matrix: True where neighbor genus is DIFFERENT from self
    is_diff = neighbor_genera != self_genera
    
    # Find the index of the first True value in each row
    # argmax returns the first index of the max value (True=1). 
    # If no True is found (all neighbors are same genus), it returns 0.
    first_diff_idx = np.argmax(is_diff, axis=1)
    
    # 5. Extract distances
    # Initialize with infinity (for cases where no diff genus is found within k neighbors)
    nearest_dists = np.full(len(gdf), np.inf)
    
    # valid_mask: where a different genus was actually found
    # (index > 0 because index 0 is the tree itself/same genus)
    valid_mask = first_diff_idx > 0
    
    # Assign distances using advanced indexing
    nearest_dists[valid_mask] = dists[valid_mask, first_diff_idx[valid_mask]]
    
    return pd.Series(nearest_dists, index=gdf.index)

for city, gdf in city_data.items():
    print(f"Computing nearest different-genus distance: {city} (Optimized)")
    city_data[city] = gdf.copy()
    city_data[city]["nearest_diff_genus_m"] = compute_nearest_diff_genus_fast(city_data[city])

log.end_step(status="success")

# %%
# ============================================================
# SECTION 9: Threshold Recommendation
# ============================================================

log.start_step("Threshold Recommendation")

recommended_eval = next(
    (e for e in threshold_evaluations if e["threshold"] == recommended_threshold),
    None,
)

print("THRESHOLD RECOMMENDATION")
print("=" * 60)
print(f"Recommended: {recommended_threshold}m")

for city in CITIES:
    retention = recommended_eval["per_city"][city]["retention_rate"]
    deviation = recommended_eval["per_city"][city]["max_deviation"]
    print(f"Retention ({city.title()}): {retention * 100:.1f}%")
    print(f"Genus deviation ({city.title()}): {deviation * 100:.2f}%")

print(f"Pixel coverage: {recommended_threshold / 10:.1f} pixels")
print(f"Meets retention target: {recommended_eval['passes_retention']}")
print(f"Genus uniformity: {recommended_eval['passes_uniformity']}")
print(f"Covers >=2 pixels: {recommended_eval['covers_two_pixels']}")

log.end_step(status="success")


# %%
# ============================================================
# SECTION 10: Export JSON Configuration
# ============================================================

log.start_step("Export Configuration")

impact_per_threshold = {}
for threshold in THRESHOLDS_TO_TEST:
    impact_per_threshold[str(threshold)] = {}
    for city in CITIES:
        row = sensitivity_results[
            (sensitivity_results["city"] == city)
            & (sensitivity_results["threshold"] == threshold)
        ].iloc[0]
        impact_per_threshold[str(threshold)][city] = {
            "trees_removed": int(row["trees_removed"]),
            "retention_rate": float(row["retention_rate"]),
        }

output = {
    "geometric_definition": {
        "distance_metric": "euclidean_point_to_point",
        "tree_geometry": "centroid (point)",
        "sentinel_pixel_size": "10m x 10m",
        "threshold_interpretation": "20m = approx 2-pixel separation (edge-to-edge contact)",
        "contamination_zones": {
            "high": "< 10m (full pixel overlap)",
            "medium": "10-15m (partial overlap)",
            "low": "15-20m (edge contact)",
            "none": "> 20m (no pixel overlap)",
        },
    },
    "threshold_sensitivity": {
        str(row["threshold_m"]): {
            "retention_rate": float(row["retention_rate"]),
            "removed_count": int(row["removed_count"]),
            "retained_count": int(row["retained_count"]),
        }
        for _, row in sensitivity_df.iterrows()
    },
    "version": "1.0",
    "created": datetime.now(timezone.utc).isoformat(),
    "recommended_threshold_m": int(recommended_threshold),
    "justification": (
        f"Threshold {recommended_threshold}m provides the best passing balance between "
        "spectral purity and retention, meeting retention and genus-uniformity criteria "
        "while covering at least two Sentinel-2 pixels."
    ),
    "thresholds_tested": THRESHOLDS_TO_TEST,
    "impact_per_threshold": impact_per_threshold,
    "genus_specific_uniform": bool(is_uniform.all()),
    "max_genus_deviation_percent": float(max_deviation.max() * 100),
    "distance_percentiles": stats_by_city,
    "validation": {
        "retention_exceeds_85_percent": bool(recommended_eval["passes_retention"]),
        "genus_impact_uniform": bool(recommended_eval["passes_uniformity"]),
        "covers_two_pixels": bool(recommended_eval["covers_two_pixels"]),
    },
}

json_path = METADATA_DIR / "proximity_filter.json"
json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
print(f"Saved: {json_path}")

log.end_step(status="success")


