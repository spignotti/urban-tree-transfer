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
# # Urban Tree Transfer - Exploratory Notebook Template
#
# **Title:** [Notebook Title]
#
# **Phase:** [1: Data Processing | 2: Feature Engineering | 3: Experiments]
#
# **Topic:** [e.g. Temporal Selection via JM-Distance]
#
# **Research Question:**
# [One sentence: what question does this notebook answer?]
#
# **Key Findings:**
# - [Update after analysis]
# - [Update after analysis]
# - [Update after analysis]
#
# **Input:** [Directory/files consumed]
#
# **Output:** [JSON configs + analysis data exported]
#
# **Author:** Silas Pignotti
#
# **Created:** YYYY-MM-DD
#
# **Updated:** YYYY-MM-DD
#
# Exploratory notebooks investigate data, test hypotheses, and determine
# parameters for downstream runner notebooks.

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
    import numpy as np
    import pandas as pd
    from scipy import stats

    from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
    from urban_tree_transfer.config.loader import load_city_config, load_feature_config
    from urban_tree_transfer.utils import ExecutionLog, validate_dataset

    # --- Notebook-specific imports ---
    # from urban_tree_transfer.feature_engineering import ...
    # from urban_tree_transfer.experiments import ...

    NOTEBOOK_ID = "02_exp_01_temporal_selection"
    log = ExecutionLog(NOTEBOOK_ID)

    print("OK: Imports complete")


# %%
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

if RUN_NOTEBOOK:
    DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")

    INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
    OUTPUT_DIR = DRIVE_DIR / "outputs" / "phase_2"
    METADATA_DIR = OUTPUT_DIR / "metadata"
    LOGS_DIR = OUTPUT_DIR / "logs"
    ANALYSIS_DIR = OUTPUT_DIR / "analysis" / "exp_01_temporal"

    CITIES = ["berlin", "leipzig"]

    for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR, ANALYSIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    feature_config = load_feature_config()

    print(f"Input:    {INPUT_DIR}")
    print(f"Analysis: {ANALYSIS_DIR}")
    print(f"Configs:  {METADATA_DIR}")
    print(f"Seed:     {RANDOM_SEED}")


# %% [markdown]
# ## 5. Data Loading
#
# Exploratory notebooks often work on the same dataset across multiple
# analyses, so data loading lives in its own cell.

# %%
# ============================================================================
# 5. DATA LOADING
# ============================================================================

if RUN_NOTEBOOK:
    log.start_step("data_loading")

    features_berlin = pd.read_parquet(INPUT_DIR / "features_berlin.parquet")
    features_leipzig = pd.read_parquet(INPUT_DIR / "features_leipzig.parquet")

    print(f"Berlin:  {len(features_berlin):,} trees, {features_berlin.shape[1]} columns")
    print(f"Leipzig: {len(features_leipzig):,} trees, {features_leipzig.shape[1]} columns")

    log.end_step(status="success", records=len(features_berlin) + len(features_leipzig))


# %% [markdown]
# ## 6. Analysis Section Template
#
# Each analysis cell should follow this pattern:
#
# Objective -> Method -> Results -> Interpretation -> Export

# %%
# ============================================================================
# 6. [ANALYSIS NAME]
# ============================================================================
# Objective: [What we want to find out]
# Method:    [Statistical method / approach used]
# ============================================================================

if RUN_NOTEBOOK:
    log.start_step("analysis_name")

    try:
        print("Computing analysis...")
        results = {}
        for city in CITIES:
            data = features_berlin if city == "berlin" else features_leipzig
            analysis_output = analysis_function(data, config=feature_config)
            results[city] = analysis_output
            print(f"  {city}: completed")

        print("\n--- Results ---")
        for city, result in results.items():
            print(f"  {city}: {result}")

        # Add inline interpretation in a markdown cell or concise print statements.

        for city, result in results.items():
            export_path = ANALYSIS_DIR / f"analysis_{city}.csv"
            pd.DataFrame(result).to_csv(export_path, index=False)
            print(f"  Exported: {export_path.name}")

        log.end_step(status="success")

    except Exception as exc:
        log.end_step(status="error", errors=[str(exc)])
        raise


# %% [markdown]
# ## Decision Cell
#
# Exploratory notebooks should end with an explicit decision that exports a
# JSON config for downstream notebooks.

# %%
# ============================================================================
# N. DECISION
# ============================================================================
# Based on the analysis above, the following parameters are determined.
# ============================================================================

if RUN_NOTEBOOK:
    config_output = {
        "notebook": NOTEBOOK_ID,
        "created": pd.Timestamp.now().isoformat(),
        "parameters": {
            "selected_months": [4, 5, 6, 7, 8, 9, 10, 11],
            "jm_threshold": 0.80,
        },
        "rationale": "Months with mean JM > 0.80 across both cities selected.",
    }

    config_path = METADATA_DIR / "temporal_selection.json"
    config_path.write_text(json.dumps(config_output, indent=2), encoding="utf-8")
    print(f"OK: Config exported: {config_path.name}")


# %% [markdown]
# ## Findings Summary
#
# End every exploratory notebook with:
#
# - research question
# - key findings
# - decisions made
# - downstream impact
# - export manifest

# %%
# ============================================================================
# FINDINGS SUMMARY
# ============================================================================

if RUN_NOTEBOOK:
    log.summary()
    log.save(LOGS_DIR / f"{NOTEBOOK_ID}_execution.json")

    analysis_files = list(ANALYSIS_DIR.glob("*"))
    config_files = list(METADATA_DIR.glob(f"*{NOTEBOOK_ID.split('_', 1)[-1]}*"))

    print("\n" + "=" * 60)
    print(f"{'EXPORT MANIFEST':^60}")
    print("=" * 60)
    print("\nAnalysis data:")
    for file_path in sorted(analysis_files):
        print(f"  {file_path.name}")
    print("\nConfig files:")
    for file_path in sorted(config_files):
        print(f"  {file_path.name}")
    print("\n" + "=" * 60)
    print("\nOK: NOTEBOOK COMPLETE")
