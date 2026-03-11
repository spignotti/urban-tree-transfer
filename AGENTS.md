# Urban Tree Transfer

## Project Overview

Urban Tree Transfer — genus-level classification of urban trees from multitemporal
Sentinel-2 imagery. Cross-city transfer learning study (Berlin -> Leipzig).
See `docs/PROJECT.md` for full project description, study design, and pipeline details.

**Status**: Active rework. The codebase is functional and has produced results.
We are refactoring for quality, not building from scratch.

**Owner**: Silas Pignotti
**Repository**: github.com/silas-workspace/urban-tree-transfer

---

## Critical Rules

### 1. Handle With Care
This is a **research codebase with existing results**. Every change must preserve
the ability to reproduce current outputs. Do not refactor speculatively.

- **No bulk rewrites.** Change only what the task explicitly asks for.
- **No silent behavior changes.** If a function's output would change, flag it.
- **No dependency upgrades** unless explicitly requested.
- **Preserve random seeds** (RANDOM_SEED = 42) everywhere.

### 2. Execution Environment
Code runs in **Google Colab** notebooks, NOT locally.

- You **cannot** run notebooks or execute the pipeline.
- You **can** edit Python source files under `src/` and notebook `.ipynb` files.
- You **can** run linting/typechecks if configured, but NOT the actual pipeline.
- All data lives on **Google Drive** (~24 GB), not in the repo.

### 3. Change Scope
Before making any change, verify:
- [ ] Does the task explicitly ask for this change?
- [ ] Could this affect downstream outputs?
- [ ] Is the change isolated to the specified file(s)?

If unsure, ask before proceeding.

---

## Architecture

```
urban-tree-transfer/
├── src/urban_tree_transfer/      # Installable Python package
│   ├── config/                   # Config loading, constants, experiment helpers
│   ├── configs/                  # YAML: cities/, experiments/, features/
│   ├── data_processing/          # Boundaries, trees, elevation, CHM, Sentinel-2
│   ├── feature_engineering/      # Extraction, quality, outliers, proximity, splits
│   ├── experiments/              # Models, training, ablation, transfer, evaluation
│   ├── schemas/                  # JSON schemas for pipeline output validation
│   └── utils/                    # IO, plotting, logging, validation helpers
├── notebooks/
│   ├── runners/                  # Colab runner notebooks (01_, 02a-c_, 03a-d_)
│   ├── exploratory/              # Experiment notebooks (exp_01 to exp_11)
│   └── templates/                # Runner + exploratory notebook templates (.md, .py, .ipynb)
├── tests/                        # Unit + integration tests (mirrors src/ structure)
├── docs/                         # PROJECT.md, methodology docs, PRDs
├── scripts/                      # Standalone utility scripts
├── outputs/                      # Execution logs and metadata (committed)
└── legacy/                       # Superseded code — do not import from here
```

### Package Install (Colab)
```
pip install git+https://{token}@github.com/silas-workspace/urban-tree-transfer.git
```
All shared logic lives in `src/`. Notebooks only orchestrate — they import from
the package and handle I/O with Google Drive.

### Notebook Types
Two types, each following a strict template:

| Type | Purpose | Pattern | Output |
|------|---------|---------|--------|
| **Runner** | Data processing | Load -> Process -> Save -> Validate | Parquet/GeoPackage files |
| **Exploratory** | Analysis & decisions | Objective -> Method -> Results -> Interpretation | JSON configs + CSVs |

### Key Libraries
Python 3.10+, GeoPandas, rasterio, scikit-learn, XGBoost, PyTorch, Optuna, scipy

### Data Formats
| Type | Format |
|------|--------|
| Spatial | GeoPackage (`.gpkg`) |
| ML-ready | Parquet (`.parquet`, Snappy, no geometry) |
| Config | YAML (`.yaml`) |
| Metadata | JSON (`.json`, schema-validated) |

---

## Development & Execution Workflow

This project has a split execution model: code is written and versioned locally,
but all pipeline execution happens remotely in Google Colab against data on Drive.

### The Two Environments

| | Local (this repo) | Google Colab |
|---|---|---|
| **Code** | Edited here | Installed from GitHub |
| **Notebooks** | Stored here | Opened and run here |
| **Data** | Not here (~24 GB) | Mounted from Drive |
| **Outputs** | Synced back here | Written to Drive |

### The Deploy Cycle

```
1. Edit src/ locally
2. uv run nox -s pre_commit   # lint + typecheck
3. git push origin main        # Colab installs from main
4. Open notebook in Colab
5. Run notebook -> outputs written to Drive
6. Download outputs from Drive -> outputs/ in this repo
7. git add outputs/ && git commit
```

**Critical:** Colab installs the package directly from the `main` branch via pip.
There is no practical way to install from a feature branch in Colab.
This means: **all code changes must be pushed to main before running any notebook.**
Finish and verify locally first, then push, then execute in Colab.

### Google Drive Structure

All pipeline data lives under a single Drive root (mounted at `/content/drive`):

```
MyDrive/urban-tree-transfer/
├── data/
│   ├── phase_1/                  # Boundaries, trees GeoPackages, CHM rasters
│   ├── phase_2/                  # Processed GeoPackages per split
│   └── phase_3_experiments/      # ML-ready Parquet splits, models, scalers
└── (other large files — never committed to repo)
```

Drive paths are configured via the config module — no hardcoded paths in notebooks.

### The outputs/ Directory

`outputs/` in this repo is a curated local mirror of Drive outputs.
It is committed to the repo for two reasons:

1. **Dependency tracking** — downstream notebooks read from Drive, but the outputs
   committed here document exactly what was produced and when. JSON configs like
   `setup_decisions.json` are the ground truth for what decisions were made.
2. **Audit trail** — execution logs and metadata provide a permanent record of
   each pipeline run.

**After running notebooks in Colab:**
- Download the relevant phase output folder from Drive
- Copy into the matching `outputs/<phase>/` directory here
- Commit — these outputs may be required for the next phase to run

### Output Structure

```
outputs/
├── phase_1_processing/
│   ├── logs/                     # Execution JSON logs
│   └── metadata/                 # Validation reports, task summaries
├── phase_2_features/
│   ├── figures/                  # exp_01 through exp_06 analysis figures
│   ├── logs/
│   └── metadata/                 # Exploratory JSON configs (temporal_selection.json, etc.)
├── phase_2_splits/
│   ├── figures/
│   ├── logs/
│   └── metadata/                 # phase_2_final_summary.json
└── phase_3_experiments/
    ├── figures/                  # All experiment and runner figures
    ├── logs/                     # Per-notebook execution JSONs
    ├── metadata/                 # setup_decisions.json, evaluation JSONs
    └── models/                   # Model metadata JSONs only (binaries gitignored)
```

### Inter-Phase Dependencies

Some outputs are hard dependencies for downstream phases — notebooks validate
their presence on startup and fail fast if missing. Key dependency chain:

```
exp_08 -> exp_08b -> exp_08c -> exp_09 -> setup_decisions.json (partial)
exp_10 -> extends setup_decisions.json with genus_selection
            |
            v
        03a -> berlin/leipzig *.parquet splits (on Drive)
            |
            v
        03b -> berlin_ml_champion, berlin_nn_champion, scalers, hp_tuning JSONs
            |
            v
        03c -> transfer_evaluation.json
            |
            v
        03d -> finetuning_curve.json
```

If a phase fails unexpectedly, check that its required Drive files exist and that
the corresponding output JSONs in `outputs/` are up to date with the actual run.

### What Stays Gitignored

Large binary files are never committed:
- Raw data: `*.gpkg`, `*.parquet`, `*.tif` (live on Drive only)
- Model weights: `*.pkl`, `*.pt`, `*.pth` (too large, Drive only)
- Model metadata JSONs (`*.metadata.json`) are committed — they document what was trained

---

## Code Quality

- **Line length:** 100 characters
- **Quotes:** double
- **Type hints:** required on all function signatures
- **Docstrings:** Google style for all public functions
- **CRS:** EPSG:25833 (UTM zone 33N) — always reproject before any spatial op
- **File paths:** always `pathlib.Path`, never raw strings

### Nox Sessions
```bash
uv run nox -s lint        # ruff check
uv run nox -s format      # ruff format
uv run nox -s typecheck   # pyright
uv run nox -s fix         # auto-fix lint + format
uv run nox -s test        # pytest (unit only, no integration)
uv run nox -s pre_commit  # fix + typecheck — run before every commit
uv run nox -s ci          # full pipeline
```

---

## Notebook Style

- Structured header cell with metadata (title, phase, purpose, I/O)
- Numbered sections with clear separators
- Each section: purpose comment -> code -> status output
- Final cell: execution summary + output manifest
- No visualization code in notebooks (visualizations are built separately)

---

## Communication

- If a task is ambiguous, ask for clarification before changing code.
- If a change would affect more files than specified, list them and confirm.
- After each task, summarize: files changed, what changed, what to verify in Colab.
