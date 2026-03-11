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
# # Urban Tree Transfer - Runner Notebook Template
#
# **Title:** [Notebook Title]
#
# **Phase:** [1: Data Processing | 2: Feature Engineering | 3: Experiments]
#
# **Step:** [e.g. 02a - Feature Extraction]
#
# **Purpose:** [One sentence: what this notebook produces]
#
# **Input:** [Directory/files consumed]
#
# **Output:** [Directory/files produced]
#
# **Author:** Silas Pignotti
#
# **Created:** YYYY-MM-DD
#
# **Updated:** YYYY-MM-DD
#
# Runner notebooks execute data processing only: input -> processing -> output.
# No analysis, no interpretation. They should be deterministic and repeatable.

# %%
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

import os
import subprocess
import sys

TEST_MODE = os.environ.get("NOTEBOOK_TEST_MODE") == "1"
IN_COLAB = "google.colab" in sys.modules
RUN_NOTEBOOK = IN_COLAB and not TEST_MODE

if RUN_NOTEBOOK:
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
else:
    print("Notebook test mode or non-Colab environment: skipping repo install.")


# %%
# ============================================================================
# 2. GOOGLE DRIVE
# ============================================================================

if RUN_NOTEBOOK:
    from google.colab import drive

    drive.mount("/content/drive")
    print("OK: Google Drive mounted")


# %%
# ============================================================================
# 3. IMPORTS
# ============================================================================

if RUN_NOTEBOOK:
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

    NOTEBOOK_ID = "02a_feature_extraction"
    log = ExecutionLog(NOTEBOOK_ID)

    print("OK: Imports complete")


# %%
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

if RUN_NOTEBOOK:
    DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")

    # I/O directories - adjust per notebook
    INPUT_DIR = DRIVE_DIR / "data" / "phase_1_processing"
    OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
    METADATA_DIR = OUTPUT_DIR / "metadata"
    LOGS_DIR = OUTPUT_DIR / "logs"

    CITIES = ["berlin", "leipzig"]

    for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    feature_config = load_feature_config()

    print(f"Input:  {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Cities: {CITIES}")
    print(f"Seed:   {RANDOM_SEED}")


# %% [markdown]
# ## 5. Processing Section Template
#
# Use one cell per processing step.
#
# Pattern:
#
# - Load
# - Process
# - Save
# - Validate
#
# Include skip logic when output already exists.

# %%
# ============================================================================
# 5. [SECTION NAME]
# ============================================================================
# Purpose: [What this section does]
# Input:   [What it reads]
# Output:  [What it writes]
# ============================================================================

if RUN_NOTEBOOK:
    log.start_step("section_name")

    try:
        output_path = OUTPUT_DIR / "output_file.parquet"

        if output_path.exists():
            df = pd.read_parquet(output_path)
            validation = validate_dataset(df, expected_columns=[...])
            log.end_step(status="skipped", records=len(df))
            print(f"Existing output loaded: {output_path.name} ({len(df):,} records)")
        else:
            print("Loading inputs...")
            data = gpd.read_file(INPUT_DIR / "input_file.gpkg")

            print("Processing...")
            result = processing_function(data, config=feature_config)

            result.to_parquet(output_path)
            print(f"Saved: {output_path.name} ({len(result):,} records)")

            validation = validate_dataset(result, expected_columns=[...])
            assert validation["schema"]["valid"], f"Validation failed: {validation}"

            log.end_step(status="success", records=len(result))

    except Exception as exc:
        log.end_step(status="error", errors=[str(exc)])
        raise


# %% [markdown]
# ## Summary and Export
#
# End every runner notebook with an execution summary and output manifest.

# %%
# ============================================================================
# SUMMARY
# ============================================================================

if RUN_NOTEBOOK:
    log.summary()
    log.save(LOGS_DIR / f"{NOTEBOOK_ID}_execution.json")

    outputs = {
        p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in OUTPUT_DIR.glob("*.parquet")
    }

    print("\n" + "=" * 60)
    print(f"{'OUTPUT MANIFEST':^60}")
    print("=" * 60)
    for name, size in sorted(outputs.items()):
        print(f"  {name:<45} {size:>8}")
    print("=" * 60)
    print(f"  Total files: {len(outputs)}")
    print("=" * 60)
    print("\nOK: NOTEBOOK COMPLETE")
