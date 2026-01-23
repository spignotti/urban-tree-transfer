# PRD: Phase 1 Data Processing Migration

**Status:** Draft
**Created:** 2026-01-22
**Branch:** `feat/phase-1-rework`

---

## Goal

Migrate Phase 1 Data Processing from the legacy `tree-classification` repo into the new `urban-tree-transfer` project structure. This establishes the pattern for how we transfer working legacy code into our new architecture.

**Success Criteria:**

- [ ] All data processing modules importable and type-checked
- [ ] Colab runner notebook executes end-to-end for Berlin
- [ ] Execution log JSON generated and saved to metadata
- [ ] Methodology documentation complete
- [ ] Leipzig configuration prepared (pending user-provided data sources)

---

## Context

**Read before implementation:**

- `CLAUDE.md` - Project conventions and standards
- `docs/PROJECT.md` - Project overview and architecture
- Legacy repo: `/Users/silas/Documents/projects/uni/Geo Projektarbeit/project/`
  - `scripts/config.py` - Configuration patterns
  - `scripts/boundaries/`, `scripts/tree_cadastres/`, `scripts/elevation/`, `scripts/chm/`
  - `notebooks/01_processing/` - Colab notebooks (Sentinel-2, Tree Correction)
  - `docs/documentation/01_Data_Processing/` - Existing methodology docs

---

## Scope

### What Changes

| Aspect        | Legacy                   | New                                              |
| ------------- | ------------------------ | ------------------------------------------------ |
| Cities        | Berlin, Hamburg, Rostock | Berlin, Leipzig                                  |
| CRS           | EPSG:25832 (UTM 32N)     | EPSG:25833 (UTM 33N) - native for both cities    |
| CHM Resampled | Yes (1m → 10m)           | No (bad feature, skip entirely)                  |
| Boundaries    | With/without 500m buffer | Only without buffer (buffer on-demand if needed) |
| Config        | Single Python module     | YAML per city + Python constants                 |
| Species names | Latin only               | Latin + German (genus and species)               |
| Data storage  | Local + Drive mixed      | Drive only, metadata/plots in repo               |
| Notebooks     | Multiple per phase       | Single runner per phase + template               |

### What Stays the Same

- Core processing logic (boundaries, trees, elevation, CHM, Sentinel-2)
- GeoPackage for vectors, GeoTIFF for rasters
- Google Earth Engine for Sentinel-2
- Publication plot style (seaborn-v0_8-whitegrid, 12x7, 300 DPI)

### Out of Scope

- Feature Engineering (Phase 2)
- Experiments (Phase 3)
- Leipzig data source research (user to provide)

---

## Implementation

### 1. Configuration System

#### 1.1 City YAML Configs

**File:** `configs/cities/berlin.yaml`

```yaml
name: Berlin
role: source # Training city

crs: EPSG:25833

boundaries:
  source: BKG VG250 WFS
  url: https://sgx.geodatenzentrum.de/wfs_vg250
  layer: vg250_gem
  filter: GEN = 'Berlin'

trees:
  source: Berlin WFS
  url: https://fbinter.stadt-berlin.de/fb/wfs/data/senstadt/s_wfs_baumbestand
  layers:
    - s_wfs_baumbestand_an # Anlagenbäume
    - s_wfs_baumbestand # Straßenbäume
  mapping:
    tree_id: gisid
    genus_latin: gattung
    species_latin: art_bot
    genus_german: gattung_deutsch
    species_german: art_deutsch
    plant_year: pflanzjahr
    height_m: baumhoehe

elevation:
  dom:
    source: Berlin GDI
    type: atom_feed
    url: # Berlin DOM Atom feed URL
  dgm:
    source: Berlin GDI
    type: atom_feed
    url: # Berlin DGM Atom feed URL

sentinel2:
  year: 2021
  months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

**File:** `configs/cities/leipzig.yaml`

```yaml
name: Leipzig
role: target # Transfer city

crs: EPSG:25833

boundaries:
  source: BKG VG250 WFS
  url: https://sgx.geodatenzentrum.de/wfs_vg250
  layer: vg250_gem
  filter: GEN = 'Leipzig'

trees:
  source: Leipzig WFS
  url: # TODO: User to provide
  layers: # TODO: User to provide
  mapping: # TODO: User to provide

elevation:
  dom:
    source: Sachsen GeoSN
    type: # TODO: User to provide
    url: # TODO: User to provide
  dgm:
    source: Sachsen GeoSN
    type: # TODO: User to provide
    url: # TODO: User to provide

sentinel2:
  year: 2021
  months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
```

#### 1.2 Constants Module

**File:** `src/urban_tree_transfer/config/constants.py`

```python
# Coordinate Systems
PROJECT_CRS = "EPSG:25833"  # UTM Zone 33N (Berlin & Leipzig native)

# Processing Parameters
RANDOM_SEED = 42
CHM_REFERENCE_YEAR = 2021
MIN_SAMPLES_PER_GENUS = 500

# Sentinel-2 Bands
SPECTRAL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
VEGETATION_INDICES = ["NDVI", "EVI", "GNDVI", "NDre1", "NDVIre", "CIre", "IRECI",
                      "RTVIcore", "NDWI", "MSI", "NDII", "kNDVI", "VARI"]

# CHM Parameters
CHM_MIN_VALID = -2.0  # Values below set to 0
CHM_MAX_VALID = 50.0  # Values above set to NoData

# GDAL Options
GDAL_COMPRESS_OPTIONS = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
```

---

### 2. Utilities Module

#### 2.1 Plotting

**File:** `src/urban_tree_transfer/utils/plotting.py`

```python
"""Publication-quality plotting utilities."""
import matplotlib.pyplot as plt
import seaborn as sns

PUBLICATION_STYLE = {
    "style": "seaborn-v0_8-whitegrid",
    "figsize": (12, 7),
    "dpi_export": 300,
    "palette": "Set2",
}


def setup_plotting() -> None:
    """Configure matplotlib for publication-quality plots."""
    plt.rcdefaults()
    plt.style.use(PUBLICATION_STYLE["style"])
    sns.set_palette(PUBLICATION_STYLE["palette"])
    plt.rcParams["figure.figsize"] = PUBLICATION_STYLE["figsize"]
    plt.rcParams["savefig.dpi"] = PUBLICATION_STYLE["dpi_export"]
    plt.rcParams["savefig.bbox"] = "tight"


def save_figure(fig: plt.Figure, path: Path, dpi: int = 300) -> None:
    """Save figure with consistent settings."""
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
```

#### 2.2 Logging

**File:** `src/urban_tree_transfer/utils/logging.py`

```python
"""Execution logging for notebooks."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field


def log_step(step_name: str) -> None:
    """Print formatted step header with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*70}")
    print(f"[{timestamp}] {step_name}")
    print(f"{'='*70}")


def log_success(message: str) -> None:
    """Print success message."""
    print(f"✓ {message}")


def log_warning(message: str) -> None:
    """Print warning message."""
    print(f"⚠️  {message}")


def log_error(message: str) -> None:
    """Print error message."""
    print(f"✗ {message}")


@dataclass
class StepResult:
    """Result of a processing step."""
    name: str
    status: str  # "success", "warning", "error"
    start_time: str
    end_time: str | None = None
    records: dict | int | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class ExecutionLog:
    """Track notebook execution for metadata export."""
    notebook: str
    execution_start: str = field(default_factory=lambda: datetime.now().isoformat())
    execution_end: str | None = None
    steps: list[StepResult] = field(default_factory=list)
    _current_step: StepResult | None = field(default=None, repr=False)

    def start_step(self, name: str) -> None:
        """Start tracking a new step."""
        log_step(name)
        self._current_step = StepResult(
            name=name,
            status="in_progress",
            start_time=datetime.now().isoformat()
        )

    def end_step(
        self,
        status: str = "success",
        records: dict | int | None = None,
        warnings: list[str] | None = None,
        errors: list[str] | None = None
    ) -> None:
        """Complete current step."""
        if self._current_step is None:
            return

        self._current_step.status = status
        self._current_step.end_time = datetime.now().isoformat()
        self._current_step.records = records
        self._current_step.warnings = warnings or []
        self._current_step.errors = errors or []

        self.steps.append(self._current_step)

        if status == "success":
            log_success(f"{self._current_step.name} complete")
        elif status == "warning":
            log_warning(f"{self._current_step.name} complete with warnings")
        else:
            log_error(f"{self._current_step.name} failed")

        self._current_step = None

    def save(self, path: Path) -> None:
        """Save execution log to JSON."""
        self.execution_end = datetime.now().isoformat()

        data = {
            "notebook": self.notebook,
            "execution_start": self.execution_start,
            "execution_end": self.execution_end,
            "steps": [
                {
                    "name": s.name,
                    "status": s.status,
                    "start_time": s.start_time,
                    "end_time": s.end_time,
                    "records": s.records,
                    "warnings": s.warnings,
                    "errors": s.errors,
                }
                for s in self.steps
            ]
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        log_success(f"Execution log saved to {path}")

    def summary(self) -> None:
        """Print execution summary."""
        print(f"\n{'='*70}")
        print("EXECUTION SUMMARY")
        print(f"{'='*70}")

        for step in self.steps:
            icon = "✓" if step.status == "success" else "⚠️" if step.status == "warning" else "✗"
            records_str = ""
            if step.records:
                if isinstance(step.records, dict):
                    records_str = f" ({sum(step.records.values()):,} total)"
                else:
                    records_str = f" ({step.records:,} records)"
            print(f"  {icon} {step.name}{records_str}")

        print(f"{'='*70}")
```

---

### 3. Data Processing Modules

**Location:** `src/urban_tree_transfer/data_processing/`

#### 3.1 boundaries.py

Adapt from: `legacy/scripts/boundaries/download_city_boundaries.py`

**Key changes:**

- Load city config from YAML
- Use EPSG:25833
- No buffer variants (just raw boundaries)
- Return GeoDataFrame, let caller handle saving

```python
def download_city_boundary(city_config: dict) -> gpd.GeoDataFrame:
    """Download city boundary from WFS."""
    ...

def load_boundaries(cities: list[str], config_dir: Path) -> gpd.GeoDataFrame:
    """Load boundaries for multiple cities."""
    ...
```

#### 3.2 trees.py

Adapt from: `legacy/scripts/tree_cadastres/*.py`

**Key changes:**

- Support Leipzig WFS (config-driven)
- Add German genus/species columns
- EPSG:25833
- Combine download + harmonize + filter in one module

```python
def download_tree_cadastre(city_config: dict) -> gpd.GeoDataFrame:
    """Download tree cadastre from WFS/OGC API."""
    ...

def harmonize_trees(gdf: gpd.GeoDataFrame, city_config: dict) -> gpd.GeoDataFrame:
    """Harmonize to unified schema with German names."""
    ...

def filter_viable_genera(gdf: gpd.GeoDataFrame, min_samples: int = 500) -> gpd.GeoDataFrame:
    """Filter to genera with sufficient samples in all cities."""
    ...
```

#### 3.3 elevation.py

Adapt from: `legacy/scripts/elevation/*.py`

**Key changes:**

- Config-driven download sources
- EPSG:25833
- Combine download + harmonize + validate

```python
def download_elevation(city_config: dict, data_type: str) -> Path:
    """Download DOM or DGM for city."""
    ...

def harmonize_elevation(dom_path: Path, dgm_path: Path, output_path: Path) -> None:
    """Harmonize elevation data to consistent format."""
    ...
```

#### 3.4 chm.py

Adapt from: `legacy/scripts/chm/*.py`

**Key changes:**

- NO resampling function (skip entirely)
- Just creation and filtering

```python
def create_chm(dom_path: Path, dgm_path: Path, output_path: Path) -> None:
    """Create CHM from DOM and DGM."""
    ...

def filter_chm(chm_path: Path, output_path: Path, min_val: float = -2.0, max_val: float = 50.0) -> None:
    """Filter CHM values to valid range."""
    ...
```

#### 3.5 sentinel.py

Adapt from: `legacy/notebooks/01_processing/02_sentinel2_gee_download.ipynb`

**Key changes:**

- Function-based API for Colab use
- Config-driven bands and indices

**Important Note**

- we spend a lot of time building a functioning GEE DOwnload Script, our recent version from the legacy Repo works perferct. Please be extra careful when trafering this code into our new repo, we dont want to break it!

```python
def create_gee_tasks(boundary: gpd.GeoDataFrame, city: str, year: int, months: list[int]) -> list[ee.batch.Task]:
    """Create GEE export tasks for Sentinel-2 composites."""
    ...

def check_task_status(tasks: list[ee.batch.Task]) -> dict:
    """Check status of running GEE tasks."""
    ...
```

---

### 4. Notebook Infrastructure

#### 4.1 Runner Template

**File:** `notebooks/templates/runner_template.ipynb`

**Cell 1: Setup**

```python
# Install package from GitHub
!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q

# Standard imports
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("✓ Imports complete")
```

**Cell 2: Drive Mount**

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Cell 3: Initialize**

```python
from urban_tree_transfer.utils.plotting import setup_plotting
from urban_tree_transfer.utils.logging import ExecutionLog, log_step, log_success

# Setup
setup_plotting()
log = ExecutionLog("NOTEBOOK_NAME")  # Change per notebook

print("✓ Initialization complete")
```

**Cell 4: Configuration**

```python
# Paths
BASE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
DATA_DIR = BASE_DIR / "data"
METADATA_DIR = Path(".")  # Local, will be committed to repo

# Cities to process
CITIES = ["berlin", "leipzig"]

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"✓ Data directory: {DATA_DIR}")
```

**Cell N: Processing Section (template)**

```python
# ============================================================
# SECTION: [Section Name]
# ============================================================

log.start_step("[Section Name]")

try:
    # Processing code here
    # ...

    log.end_step(status="success", records={"berlin": 100, "leipzig": 50})
except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise
```

**Final Cell: Summary**

```python
# ============================================================
# SUMMARY
# ============================================================

log.summary()

# Save execution log
log.save(METADATA_DIR / "logs" / f"{log.notebook}_execution.json")

print("\n✓ Notebook complete")
```

#### 4.2 Phase 1 Runner

**File:** `notebooks/runners/01_data_processing.ipynb`

Structure:

1. Setup & Imports
2. Drive Mount
3. Initialize (plotting, logging)
4. Configuration
5. **Section: Boundaries** - Download city boundaries
6. **Section: Trees** - Download, harmonize, filter tree cadastres
7. **Section: Elevation** - Download DOM/DGM
8. **Section: CHM** - Create and filter CHM
9. **Section: Sentinel-2** - Submit GEE tasks (separate monitoring)
10. Summary & Export Log

---

### 5. Documentation

#### 5.1 Simplified Methodology Template

**File:** `docs/templates/methodology_template.md`

```markdown
# [Component]: Methodology

**Phase:** [Data Processing / Feature Engineering / Experiments]
**Last Updated:** [Date]

---

## Purpose

[What does this step accomplish?]

## Data Sources

| City    | Source | Format | Notes |
| ------- | ------ | ------ | ----- |
| Berlin  | ...    | ...    | ...   |
| Leipzig | ...    | ...    | ...   |

## Method

[A few paragraphs describing the approach. No code. Reference config files for parameters. Write down important methodology decisions here as well.]

### Key Parameters

| Parameter | Value | Rationale |
| --------- | ----- | --------- |
| ...       | ...   | ...       |

## Output

| File | Format | Description |
| ---- | ------ | ----------- |
| ...  | ...    | ...         |

**Metadata:** See `metadata/[component]_metadata.json`

## Quality Criteria

- [ ] Criterion 1
- [ ] Criterion 2

## Limitations

- Limitation 1
- Limitation 2

---

**References:** [If applicable]
```

#### 5.2 Phase 1 Methodology

**File:** `docs/documentation/01_Data_Processing/01_Data_Processing_Methodik.md`

Apply template for each component:

- Boundaries
- Tree Cadastres
- Elevation (DOM/DGM)
- CHM Creation
- Sentinel-2 Composites

---

### 6. User Input Required

Before Leipzig processing can be implemented, the following information is needed:

**Leipzig Tree Cadastre:**

- [ ] WFS/OGC API URL
- [ ] Layer name(s)
- [ ] Attribute mapping (tree_id, genus, species, etc.)

**Leipzig Elevation Data:**

- [ ] DOM source and download method
- [ ] DGM source and download method
- [ ] Expected CRS and resolution

---

## File Checklist

### Configuration

- [ ] `configs/cities/berlin.yaml`
- [ ] `configs/cities/leipzig.yaml` (partial, awaiting user input)

### Source Code

- [ ] `src/urban_tree_transfer/config/__init__.py`
- [ ] `src/urban_tree_transfer/config/constants.py`
- [ ] `src/urban_tree_transfer/config/loader.py`
- [ ] `src/urban_tree_transfer/utils/__init__.py`
- [ ] `src/urban_tree_transfer/utils/plotting.py`
- [ ] `src/urban_tree_transfer/utils/logging.py`
- [ ] `src/urban_tree_transfer/data_processing/__init__.py`
- [ ] `src/urban_tree_transfer/data_processing/boundaries.py`
- [ ] `src/urban_tree_transfer/data_processing/trees.py`
- [ ] `src/urban_tree_transfer/data_processing/elevation.py`
- [ ] `src/urban_tree_transfer/data_processing/chm.py`
- [ ] `src/urban_tree_transfer/data_processing/sentinel.py`

### Notebooks

- [ ] `notebooks/templates/runner_template.ipynb`
- [ ] `notebooks/runners/01_data_processing.ipynb`

### Documentation

- [ ] `docs/templates/methodology_template.md`
- [ ] `docs/documentation/01_Data_Processing/01_Data_Processing_Methodik.md`

---

## Validation

1. **Type Checking:** `uv run nox -s typecheck` passes
2. **Linting:** `uv run nox -s lint` passes
3. **Import Test:** All modules importable in Python
4. **Colab Test:** Runner notebook executes for Berlin (Sentinel-2 tasks submitted)
5. **Metadata:** Execution log JSON generated
6. **Documentation:** Methodology doc complete and follows template

---

## Anti-Patterns to Avoid

- **Don't** copy legacy code verbatim - adapt to new patterns
- **Don't** hardcode city names - use config files
- **Don't** include CHM resampling - it's explicitly removed
- **Don't** create buffer variants of boundaries
- **Don't** put processing logic in notebooks - keep in src/
- **Don't** include code snippets in methodology docs - reference configs
