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
# # exp_07: Cross-City Baseline Analysis
#
# **Phase 3 - Exploratory Analysis (Optional)**
#
# Descriptive analysis of domain shift between Berlin and Leipzig datasets **before** training. Generates hypotheses for transfer evaluation (03c) by quantifying:
# - Class distribution differences
# - Phenological profile divergence
# - Structural differences (CHM)
# - Feature distribution overlap
# - Statistical effect sizes (Cohen's d)
# - Correlation structure similarity
#
# **Note:** This notebook is **purely descriptive** — no training, no JSON outputs. Results inform interpretation of transfer experiments but are not used for decisions.

# %%
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended (for correlation matrix on full feature set)
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
from urban_tree_transfer.config import RANDOM_SEED
from urban_tree_transfer.experiments import (
    data_loading,
    evaluation,
)
from urban_tree_transfer.utils import ExecutionLog

from pathlib import Path
import warnings
import gc

import pandas as pd
import numpy as np
from scipy.stats import pearsonr

log = ExecutionLog("exp_07_cross_city_baseline")

warnings.filterwarnings("ignore", category=UserWarning)

print("OK: Package imports complete")

# %%
# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_splits"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_3_experiments"

LOGS_DIR = OUTPUT_DIR / "logs"

for d in [LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")

# %%
# ============================================================
# SECTION 1: Data Loading & Memory Optimization
# ============================================================

log.start_step("Data Loading")

# Load baseline splits (before proximity filtering)
berlin_train, berlin_val, berlin_test = data_loading.load_berlin_splits(
    INPUT_DIR, variant="baseline"
)
leipzig_finetune, leipzig_test = data_loading.load_leipzig_splits(
    INPUT_DIR, variant="baseline"
)

print(f"Berlin Train:      {len(berlin_train):,} samples")
print(f"Leipzig Finetune:  {len(leipzig_finetune):,} samples")

# Memory optimization: Convert float64 → float32
print("\nMemory optimization: Converting float64 → float32...")
for df in [berlin_train, leipzig_finetune]:
    float_cols = df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
        
print(f"  Converted {len(float_cols)} float columns to float32")

log.end_step(status="success", records=len(berlin_train) + len(leipzig_finetune))

