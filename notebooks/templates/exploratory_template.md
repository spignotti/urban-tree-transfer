# Exploratory Notebook Template

Standard template for all exploratory notebooks in this repository.

Exploratory notebooks investigate data, test hypotheses, and determine
parameters for downstream runner notebooks. Every analysis should have a clear
objective, a documented result, and an exported decision artifact.

**File naming convention:** `{phase_id}_exp_{number}_{topic}.ipynb`

Example: `02_exp_01_temporal_selection.ipynb`

## Header Block

```python
# ============================================================================
# URBAN TREE TRANSFER - Exploratory Notebook
# ============================================================================
#
# Title:    [Notebook Title]
# Phase:    [1: Data Processing | 2: Feature Engineering | 3: Experiments]
# Topic:    [e.g. Temporal Selection via JM-Distance]
#
# Research Question:
#   [One sentence: what question does this notebook answer?]
#
# Key Findings:
#   - [Updated after analysis - 2-3 bullet points with main results]
#
# Input:    [Directory/files consumed]
# Output:   [JSON configs + analysis data exported]
#
# Author:   Silas Pignotti
# Created:  YYYY-MM-DD
# Updated:  YYYY-MM-DD
# ============================================================================
```

## Cells 1-3 - Same as Runner

Environment setup, Google Drive, and imports follow the same structure as the
runner template.

Additional imports for exploratory analysis:

```python
# --- Additional imports for exploratory analysis ---
import numpy as np
from scipy import stats
```

## Cell 4 - Configuration

```python
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")

INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
OUTPUT_DIR = DRIVE_DIR / "outputs" / "phase_2"
METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

# Analysis-specific export directory
ANALYSIS_DIR = OUTPUT_DIR / "analysis" / "exp_01_temporal"

CITIES = ["berlin", "leipzig"]

for d in [OUTPUT_DIR, METADATA_DIR, LOGS_DIR, ANALYSIS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

feature_config = load_feature_config()

print(f"Input:    {INPUT_DIR}")
print(f"Analysis: {ANALYSIS_DIR}")
print(f"Configs:  {METADATA_DIR}")
```

## Cell 5 - Data Loading

Exploratory notebooks usually work on the same dataset across multiple
analyses, so data loading lives in its own cell.

```python
# ============================================================================
# 5. DATA LOADING
# ============================================================================

log.start_step("data_loading")

features_berlin = pd.read_parquet(INPUT_DIR / "features_berlin.parquet")
features_leipzig = pd.read_parquet(INPUT_DIR / "features_leipzig.parquet")

print(f"Berlin:  {len(features_berlin):,} trees, {features_berlin.shape[1]} columns")
print(f"Leipzig: {len(features_leipzig):,} trees, {features_leipzig.shape[1]} columns")

log.end_step(status="success", records=len(features_berlin) + len(features_leipzig))
```

## Cell 6+ - Analysis Sections

Each analysis gets its own cell.

Every section follows this pattern:

**Objective -> Method -> Results -> Interpretation -> Export**

```python
# ============================================================================
# 6. [ANALYSIS NAME] - e.g. "JM-Distance by Month"
# ============================================================================
#
# Objective: [What we want to find out]
# Method:    [Statistical method / approach used]
# ============================================================================

log.start_step("[analysis_name]")

try:
    # --- Computation ---
    print("Computing JM distances...")
    results = {}
    for city in CITIES:
        data = features_berlin if city == "berlin" else features_leipzig
        jm_matrix = compute_jm_distance(data, feature_cols=feature_config.spectral_bands)
        results[city] = jm_matrix
        print(f"  {city}: shape {jm_matrix.shape}")

    # --- Results ---
    print("\n--- Results ---")
    for city, matrix in results.items():
        mean_jm = matrix.mean()
        print(f"  {city}: Mean JM = {mean_jm:.3f}")

    # --- Interpretation ---
    # [Inline markdown cell or print statements summarizing what the results mean]
    # Example: "Months 4-11 show JM > 0.8, confirming vegetation period as optimal window."

    # --- Export analysis data ---
    for city, matrix in results.items():
        export_path = ANALYSIS_DIR / f"jm_matrix_{city}.csv"
        pd.DataFrame(matrix).to_csv(export_path)
        print(f"  Exported: {export_path.name}")

    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise
```

## Decision Cell - Parameter Decisions

Exploratory notebooks usually end with an explicit decision that is exported as
JSON for downstream runners.

```python
# ============================================================================
# N. DECISION - [Parameter/Threshold Name]
# ============================================================================
#
# Based on the analysis above, the following parameters are determined:
#
#   - Selected months: [4, 5, 6, 7, 8, 9, 10, 11]
#   - Selection threshold: JM > 0.80
#   - Rationale: [1-2 sentences]
#
# These parameters are exported as JSON config for downstream runners.
# ============================================================================

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
```

## Final Cell - Findings Summary

```python
# ============================================================================
# FINDINGS SUMMARY
# ============================================================================
#
# Research Question: [Repeat from header]
#
# Key Findings:
#   1. [Finding with number]
#   2. [Finding with number]
#   3. [Finding with number]
#
# Decisions Made:
#   - [Parameter]: [Value] - [Rationale]
#
# Downstream Impact:
#   - Config exported to: [path]
#   - Used by: [Runner notebook name]
#
# Open Questions:
#   - [If any remain]
# ============================================================================

log.summary()
log.save(LOGS_DIR / f"{NOTEBOOK_ID}_execution.json")

# --- Export manifest ---
analysis_files = list(ANALYSIS_DIR.glob("*"))
config_files = list(METADATA_DIR.glob(f"*{NOTEBOOK_ID.split('_', 1)[-1]}*"))

print("\n" + "=" * 60)
print(f"{'EXPORT MANIFEST':^60}")
print("=" * 60)
print("\nAnalysis data:")
for f in sorted(analysis_files):
    print(f"  {f.name}")
print("\nConfig files:")
for f in sorted(config_files):
    print(f"  {f.name}")
print("\n" + "=" * 60)
print("\nOK: NOTEBOOK COMPLETE")
```

## Exploratory Summary

| Aspect | Exploratory notebooks |
| --- | --- |
| Purpose | Analyze data and determine parameters |
| Output | JSON configs + analysis CSVs |
| Output directory | `outputs/phase_X/analysis/` |
| Section pattern | Objective -> Method -> Results -> Interpretation -> Export |
| Header focus | Research Question + Key Findings |
| Decision cell | Yes |
| Skip logic | No |
| Final cell | Findings Summary + Export Manifest |

## Portfolio Conventions

To keep the repository professional:

- Use a structured header block in every notebook.
- Update the header's Key Findings after the analysis is complete.
- Add inline interpretation between code sections, not just raw output.
- Keep naming consistent across runner and exploratory notebooks.
- Keep outputs visible before commit, but remove debug prints.
- Save an execution log JSON under `outputs/*/logs/`.
