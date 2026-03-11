# Runner Notebook Template

Standard template for all runner notebooks in this repository.

Runner notebooks execute data processing steps: input -> processing -> output.
They do not contain analysis or interpretation. They should be deterministic,
repeatable, and portfolio-ready.

**File naming convention:** `{phase_id}_{step_letter}_{name}.ipynb`

Example: `02a_feature_extraction.ipynb`

## Header Block

```python
# ============================================================================
# URBAN TREE TRANSFER - Runner Notebook
# ============================================================================
#
# Title:    [Notebook Title]
# Phase:    [1: Data Processing | 2: Feature Engineering | 3: Experiments]
# Step:     [e.g. 02a - Feature Extraction]
# Purpose:  [One sentence: what this notebook produces]
#
# Input:    [Directory/files consumed]
# Output:   [Directory/files produced]
#
# Author:   Silas Pignotti
# Created:  YYYY-MM-DD
# Updated:  YYYY-MM-DD
# ============================================================================
```

## Cell 1 - Environment Setup

```python
# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================
# Runtime: CPU (Standard) | GPU (if needed)
# RAM:     Standard | High-RAM (for large datasets)
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
```

## Cell 2 - Google Drive

```python
# ============================================================================
# 2. GOOGLE DRIVE
# ============================================================================

from google.colab import drive

drive.mount("/content/drive")
print("OK: Google Drive mounted")
```

## Cell 3 - Imports

```python
# ============================================================================
# 3. IMPORTS
# ============================================================================

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd

from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
from urban_tree_transfer.config.loader import load_city_config, load_feature_config
from urban_tree_transfer.utils import ExecutionLog, validate_dataset

# --- Notebook-specific imports ---
# from urban_tree_transfer.feature_engineering import ...
# from urban_tree_transfer.experiments import ...

NOTEBOOK_ID = "02a_feature_extraction"  # Unique identifier for this notebook
log = ExecutionLog(NOTEBOOK_ID)

print("OK: Imports complete")
```

## Cell 4 - Configuration

```python
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")

# I/O directories - adjust per notebook
INPUT_DIR = DRIVE_DIR / "data" / "phase_1_processing"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

CITIES = ["berlin", "leipzig"]

for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load configs as needed
feature_config = load_feature_config()

print(f"Input:  {INPUT_DIR}")
print(f"Output: {OUTPUT_DIR}")
print(f"Cities: {CITIES}")
print(f"Seed:   {RANDOM_SEED}")
```

## Cell 5+ - Processing Sections

Each processing step gets its own cell.

Every section follows the same pattern:

```python
# ============================================================================
# 5. [SECTION NAME] - e.g. "CHM Feature Extraction"
# ============================================================================
# Purpose: [What this section does]
# Input:   [What it reads]
# Output:  [What it writes]
# ============================================================================

log.start_step("[section_name]")

try:
    output_path = OUTPUT_DIR / "output_file.parquet"

    if output_path.exists():
        # --- Skip if already computed ---
        df = pd.read_parquet(output_path)
        validation = validate_dataset(df, expected_columns=[...])
        log.end_step(status="skipped", records=len(df))
        print(f"Existing output loaded: {output_path.name} ({len(df):,} records)")
    else:
        # --- Load inputs ---
        print("Loading inputs...")
        data = gpd.read_file(INPUT_DIR / "input_file.gpkg")

        # --- Process ---
        print("Processing...")
        result = processing_function(data, config=feature_config)

        # --- Save ---
        result.to_parquet(output_path)
        print(f"Saved: {output_path.name} ({len(result):,} records)")

        # --- Validate ---
        validation = validate_dataset(result, expected_columns=[...])
        assert validation["schema"]["valid"], f"Validation failed: {validation}"

        log.end_step(status="success", records=len(result))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise
```

## Final Cell - Summary and Export

```python
# ============================================================================
# SUMMARY
# ============================================================================

log.summary()
log.save(LOGS_DIR / f"{NOTEBOOK_ID}_execution.json")

# --- Output manifest ---
outputs = {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in OUTPUT_DIR.glob("*.parquet")}

print("\n" + "=" * 60)
print(f"{'OUTPUT MANIFEST':^60}")
print("=" * 60)
for name, size in sorted(outputs.items()):
    print(f"  {name:<45} {size:>8}")
print("=" * 60)
print(f"  Total files: {len(outputs)}")
print("=" * 60)
print("\nOK: NOTEBOOK COMPLETE")
```

## Runner Summary

| Aspect | Runner notebooks |
| --- | --- |
| Purpose | Process data |
| Output | Datasets (Parquet/GeoPackage) |
| Output directory | `data/phase_X_*` |
| Section pattern | Load -> Process -> Save -> Validate |
| Header focus | Input/Output |
| Decision cell | No |
| Skip logic | Yes |
| Final cell | Output manifest |

## Portfolio Conventions

To keep the repository clean and professional:

- Use a structured header block in every notebook.
- Keep sections numbered and visually consistent.
- Include clear purpose comments before each section.
- Keep outputs visible before commit, but remove debug noise.
- Save an execution log JSON under `outputs/*/logs/`.
- Keep runner notebooks operational, not analytical.
