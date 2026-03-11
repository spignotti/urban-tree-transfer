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
# Install package from GitHub
# !pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q

# Standard imports
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print('OK: Imports complete')


# %%
from google.colab import drive
drive.mount('/content/drive')


# %%
from urban_tree_transfer.utils.logging import ExecutionLog, log_step, log_success

log = ExecutionLog('NOTEBOOK_NAME')

print('OK: Initialization complete')


# %%
# Paths
BASE_DIR = Path('/content/drive/MyDrive/dev/urban-tree-transfer')
DATA_DIR = BASE_DIR / 'data'
METADATA_DIR = Path('.')

# Cities to process
CITIES = ['berlin', 'leipzig']

DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f'OK: Data directory: {DATA_DIR}')


# %%
# ============================================================
# SECTION: [Section Name]
# ============================================================

log.start_step('[Section Name]')

try:
    # Processing code here
    # ...

    log.end_step(status='success', records={'berlin': 100, 'leipzig': 50})
except Exception as e:
    log.end_step(status='error', errors=[str(e)])
    raise


# %%
# ============================================================
# SUMMARY
# ============================================================

log.summary()

log.save(METADATA_DIR / 'logs' / f'{log.notebook}_execution.json')

print('OK: Notebook complete')

