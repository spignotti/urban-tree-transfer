# Feature Engineering Workflow & Configuration Strategy

**Phase:** 2 - Feature Engineering  
**Created:** 2026-01-28

---

## Overview

Phase 2 Feature Engineering combines **Runner Notebooks** (automated processing) with **Exploratory Notebooks** (parameter determination). This document describes the workflow, configuration strategy, and the manual sync process between Google Colab and the Git repository.

---

## Workflow Architecture

### 1. Code Organization

```
src/urban_tree_transfer/
├── feature_engineering/
│   ├── extraction.py      # Reusable extraction logic
│   ├── quality.py         # Reusable quality control logic
│   ├── selection.py       # Reusable selection logic
│   ├── outliers.py        # Reusable outlier detection logic
│   └── splits.py          # Reusable split logic

notebooks/
├── runners/               # Automated processing (minimal code)
│   ├── 02a_feature_extraction.ipynb
│   ├── 02b_data_quality.ipynb
│   └── 02c_final_preparation.ipynb
└── exploratory/           # Analysis notebooks (more code allowed)
    ├── exp_01_temporal_analysis.ipynb
    ├── exp_02_chm_assessment.ipynb
    ├── exp_03_correlation_analysis.ipynb
    ├── exp_04_outlier_thresholds.ipynb
    └── exp_05_spatial_autocorrelation.ipynb
```

**Principle:**

- **Runner Notebooks:** Thin wrappers around `src/` functions (like Phase 1)
- **Exploratory Notebooks:** Can contain more inline code for analysis, but reusable analysis functions should still be in `src/feature_engineering/`

---

## Configuration Strategy

### Two Types of Configuration

#### 1. Static Configuration (`feature_config.yaml`)

**Location:** `configs/features/feature_config.yaml`  
**Purpose:** Feature definitions, initial defaults, thresholds  
**Created:** Before starting Phase 2  
**Modified:** Rarely (only when adding new features)

**Contains:**

- Metadata columns
- Feature definitions (bands, indices)
- **DEFAULT** thresholds (can be overridden by exploratory analysis)
- Genus lists
- Temporal year/months to EXTRACT (all 12 months)

**Example:**

```yaml
temporal:
  year: 2021
  extraction_months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # Extract ALL
  # selected_months: DETERMINED BY exp_01 → temporal_selection.json

quality:
  nan_month_threshold: 2 # DEFAULT - may be validated in exp_01
  ndvi_min_threshold: 0.3 # DEFAULT - may be validated in exploratory
```

#### 2. Dynamic Configuration (Exploratory JSON Outputs)

**Location:** `outputs/phase_2/metadata/*.json`  
**Purpose:** Parameters determined by exploratory analysis  
**Created:** During exploratory notebooks  
**Used by:** Subsequent runner notebooks

**Files:**

- `temporal_selection.json` - Which months to USE (subset of 12)
- `chm_assessment.json` - CHM feature decisions
- `correlation_removal.json` - Redundant features to drop
- `outlier_thresholds.json` - Validated outlier parameters
- `spatial_autocorrelation.json` - Validated block size

---

## Complete Workflow with Manual Sync

### Phase 1: Initial Setup (Local)

```bash
# 1. Create feature_config.yaml with defaults
# 2. Implement src/feature_engineering/ modules
# 3. Commit to Git
git add configs/features/feature_config.yaml
git add src/urban_tree_transfer/feature_engineering/
git commit -m "feat: Phase 2 initial configuration and modules"
git push
```

### Phase 2: Feature Extraction (Colab)

**Notebook:** `02a_feature_extraction.ipynb`

```python
# In Colab:
!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git
from google.colab import drive
drive.mount('/content/drive')

# Uses feature_config.yaml defaults
# Extracts ALL 12 months (no selection yet)
# Output: trees_with_features_{city}.gpkg
```

**No manual sync needed** - only produces data, no configs.

---

### Phase 3: Exploratory Analysis Round 1 (Colab → Manual → Git)

#### Step 1: Run Exploratory Notebooks in Colab

**Notebook:** `exp_01_temporal_analysis.ipynb`

```python
# In Colab:
from urban_tree_transfer.feature_engineering import analyze_temporal_jm

# Analysis happens (can use src/ functions OR inline code)
selected_months = [3, 4, 5, 6, 7, 8, 9, 10]  # Determined by JM analysis

# Save to Drive
output = {
    "selected_months": selected_months,
    "selection_method": "jm_threshold",
    "threshold": 1.4
}

import json
output_path = Path("/content/drive/MyDrive/urban-tree-transfer/outputs/phase_2/metadata/temporal_selection.json")
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)
```

**Notebook:** `exp_02_chm_assessment.ipynb`

```python
# Similar: Produces chm_assessment.json on Drive
```

#### Step 2: Manual Sync (Local)

```bash
# 1. Copy JSONs from Google Drive to local repo
cp ~/Google\ Drive/urban-tree-transfer/outputs/phase_2/metadata/temporal_selection.json \
   outputs/phase_2/metadata/

cp ~/Google\ Drive/urban-tree-transfer/outputs/phase_2/metadata/chm_assessment.json \
   outputs/phase_2/metadata/

# 2. Review JSONs (sanity check)
cat outputs/phase_2/metadata/temporal_selection.json

# 3. Commit to Git
git add outputs/phase_2/metadata/*.json
git commit -m "feat: Add temporal selection and CHM assessment configs from exploratory analysis"
git push
```

---

### Phase 4: Data Quality Runner (Colab)

**Notebook:** `02b_data_quality.ipynb`

```python
# In Colab:
!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git

# IMPORTANT: Repo is pulled with latest JSONs!
import json
from pathlib import Path

# Load exploratory results (NOW AVAILABLE in pulled repo)
with open("outputs/phase_2/metadata/temporal_selection.json") as f:
    temporal_config = json.load(f)
selected_months = temporal_config["selected_months"]

# Use in processing
trees_gdf = apply_temporal_selection(trees_gdf, selected_months, feature_config)
```

**Key:** The JSONs are IN the Git repo, so when Colab pulls the package, it gets them!

---

### Phase 5: Exploratory Analysis Round 2 (Colab → Manual → Git)

**Notebooks:** `exp_03`, `exp_04`, `exp_05`

Repeat the cycle:

1. Run in Colab → Produce JSONs on Drive
2. Manual sync: Copy Drive → Local → Commit → Push
3. Next runner pulls repo → Has JSONs

---

### Phase 6: Final Preparation Runner (Colab)

**Notebook:** `02c_final_preparation.ipynb`

```python
# Loads all exploratory JSONs
with open("outputs/phase_2/metadata/correlation_removal.json") as f:
    correlation_config = json.load(f)

with open("outputs/phase_2/metadata/outlier_thresholds.json") as f:
    outlier_config = json.load(f)

# Uses them for processing
```

---

## Key Principles

### 1. Git is the Single Source of Truth

- All configs (static YAML + dynamic JSONs) are in Git
- Colab notebooks PULL from Git, never push
- Manual sync ensures human review of exploratory results

### 2. Configuration Hierarchy

```
feature_config.yaml (DEFAULTS)
    ↓
Exploratory JSONs (OVERRIDE defaults)
    ↓
Runner Notebooks (USE overrides, fallback to defaults)
```

### 3. Notebook Philosophy

**Runner Notebooks (002a/b/c):**

- Minimal code, mostly imports and function calls
- Load configs, call `src/` functions, save outputs
- Like Phase 1 Data Processing pattern

**Exploratory Notebooks (exp_01-05):**

- More inline code is acceptable (analysis, visualization)
- But: Reusable analysis logic should be in `src/feature_engineering/`
- Produce JSON configs as outputs
- Generate visualizations for documentation

### 4. Why Not Automatic Sync?

**Technical:** Colab can't push to private GitHub repos (authentication complexity)  
**Advantage:** Manual review of exploratory results before committing  
**Process:** Documented, reproducible, part of workflow

---

## Example: Temporal Selection Flow

### Initial State (Before Exploration)

```yaml
# feature_config.yaml
temporal:
  extraction_months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] # Extract ALL
```

### After exp_01 (Exploratory Determines Subset)

```json
// temporal_selection.json (NEW FILE)
{
  "selected_months": [3, 4, 5, 6, 7, 8, 9, 10],
  "selection_method": "jm_threshold",
  "threshold": 1.4
}
```

### In Runner 02b (Uses Exploratory Result)

```python
# Load feature_config (has ALL 12 months)
feature_config = load_feature_config()

# Load exploratory result (OVERRIDES to 8 months)
with open("outputs/phase_2/metadata/temporal_selection.json") as f:
    temporal_config = json.load(f)
selected_months = temporal_config["selected_months"]

# Drop columns for non-selected months
trees_gdf = apply_temporal_selection(trees_gdf, selected_months, feature_config)
```

**Result:**

- Extracted 12 months (needed for exploratory)
- Use 8 months (determined by exploratory)
- feature_config.yaml unchanged (still says 12)
- Override happens via JSON

---

## File Locations Summary

### In Git Repository

```
configs/features/feature_config.yaml          # Static config (defaults)
outputs/phase_2/metadata/temporal_selection.json    # Dynamic (from exp_01)
outputs/phase_2/metadata/chm_assessment.json        # Dynamic (from exp_02)
outputs/phase_2/metadata/correlation_removal.json   # Dynamic (from exp_03)
outputs/phase_2/metadata/outlier_thresholds.json    # Dynamic (from exp_04)
outputs/phase_2/metadata/spatial_autocorrelation.json # Dynamic (from exp_05)
```

### In Google Drive (During Colab Execution)

```
/content/drive/MyDrive/urban-tree-transfer/
├── data/phase_2_features/                    # Intermediate data
├── outputs/phase_2/
│   ├── metadata/                             # JSONs produced by exploratory
│   └── figures/                              # Visualizations
```

### Sync Direction

```
Colab Drive → Manual Copy → Local Repo → Git Push → Colab Pull (next notebook)
```

---

## Checklist for Manual Sync

After running exploratory notebooks:

```bash
# 1. Check Drive for new JSON files
ls ~/Google\ Drive/urban-tree-transfer/outputs/phase_2/metadata/

# 2. Copy to local repo
cp ~/Google\ Drive/urban-tree-transfer/outputs/phase_2/metadata/*.json \
   outputs/phase_2/metadata/

# 3. Review contents
cat outputs/phase_2/metadata/temporal_selection.json

# 4. Validate JSON format
python -m json.tool outputs/phase_2/metadata/temporal_selection.json

# 5. Commit
git add outputs/phase_2/metadata/*.json
git commit -m "feat: Add exploratory analysis configs (temporal, CHM, correlation, outliers, spatial)"
git push

# 6. Verify next notebook can access
# Next Colab: pip install git+... (will pull latest with JSONs)
```

---

## Troubleshooting

### Problem: Runner notebook can't find JSON

**Symptom:**

```python
FileNotFoundError: outputs/phase_2/metadata/temporal_selection.json
```

**Solution:**

1. Check if JSON is committed to Git: `git log outputs/phase_2/metadata/`
2. Ensure Colab reinstalled package after commit: `!pip install --upgrade git+...`
3. Check file path in notebook (should be relative to package root)

### Problem: JSON has wrong format

**Symptom:**

```python
KeyError: 'selected_months'
```

**Solution:**

1. Validate JSON structure against PRD specification
2. Check exploratory notebook output logic
3. Re-run exploratory with corrected output format

### Problem: Outdated JSONs

**Symptom:** Runner uses old parameters

**Solution:**

1. Check Git commit history: `git log outputs/phase_2/metadata/`
2. Ensure latest JSONs are committed
3. Force Colab to reinstall: `!pip uninstall urban-tree-transfer && pip install git+...`

---

## Best Practices

### 1. Document Manual Sync in Notebook

```python
# MANUAL STEP REQUIRED:
# 1. This notebook produces: temporal_selection.json
# 2. Copy from Drive to local repo: outputs/phase_2/metadata/
# 3. Commit and push to Git
# 4. Next notebook (02b_data_quality) will use this config
```

### 2. Validate JSON Schema

```python
# In exploratory notebook, validate before saving
expected_keys = ["selected_months", "selection_method", "threshold"]
assert all(k in output for k in expected_keys), "Missing required keys"
```

### 3. Version JSONs

```json
{
  "version": "1.0",
  "created": "2026-01-28T14:30:00",
  "selected_months": [3, 4, 5, 6, 7, 8, 9, 10]
}
```

### 4. Keep Drive and Git in Sync

- Regular syncs (after each exploratory notebook)
- Clear naming conventions
- Git commit messages reference which exploratory notebook produced config

---

**Last Updated:** 2026-01-28
