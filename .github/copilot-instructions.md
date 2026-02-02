# Copilot Instructions for Urban Tree Transfer

## Project Overview

Research project applying **cross-city transfer learning** for tree genus classification using Sentinel-2 satellite data. Tests how well models trained on Berlin transfer to Leipzig, and how much local fine-tuning data is needed for performance recovery.

**Key Architecture Insight:** This is a scientific ML pipeline with strict separation of concerns:

- **Phase 1 (Complete):** Data processing → Harmonized tree cadastres + CHM + Sentinel-2 composites
- **Phase 2 (In Progress):** Feature engineering → ML-ready datasets with spatial splits
- **Phase 3 (Planned):** Experiments → Model training, transfer evaluation, fine-tuning

## Essential Development Patterns

### Package Management: UV-First Workflow

This project uses **UV** exclusively for Python package management. Never use `pip` directly or edit `pyproject.toml` manually.

```bash
# ALWAYS use uv for package management
uv add pandas              # Add runtime dependency
uv add --dev pytest        # Add dev dependency
uv sync                    # Install from lockfile
uv run python script.py    # Run code in venv
uv run nox                 # Run quality checks
```

**Critical:** Dependencies are tracked in both `pyproject.toml` AND `uv.lock`. Only `uv add`/`uv remove` keeps them synchronized.

### Pre-Commit Workflow: Nox Sessions

All code quality checks run through Nox. **Never commit without running:**

```bash
uv run nox -s fix          # Auto-fix linting + formatting
uv run nox                 # Run default checks (lint + typecheck)
uv run nox -s ci           # Full CI pipeline (includes tests)
```

Available sessions: `lint`, `format`, `typecheck`, `fix`, `pre_commit`, `test`, `test_integration`, `ci`

### Configuration Architecture

**3-Level Config System:**

1. **Constants** ([src/urban_tree_transfer/config/constants.py](src/urban_tree_transfer/config/constants.py)): Global values (`PROJECT_CRS = "EPSG:25833"`, `RANDOM_SEED = 42`)
2. **City Configs** ([src/urban_tree_transfer/configs/cities/](src/urban_tree_transfer/configs/cities/)): Per-city YAML (WFS URLs, data sources, CRS)
3. **Experiment Configs** ([src/urban_tree_transfer/configs/experiments/](src/urban_tree_transfer/configs/experiments/)): Phase-specific YAML (hyperparameters, splits)

**Pattern:** Load configs via `load_city_config("berlin")`, never hardcode URLs or paths.

Example city config structure:

```yaml
name: Berlin
role: source
boundaries:
  url: https://gdi.berlin.de/services/wfs/alkis_land
  layer: alkis_land:landesgrenze
trees:
  url: https://gdi.berlin.de/services/wfs/baumbestand
  mapping:
    tree_id: gisid
    genus_latin: gattung
```

### Geospatial Processing Conventions

**Coordinate System:** All data transforms to `EPSG:25833` (UTM Zone 33N) via [constants.py](src/urban_tree_transfer/config/constants.py)

**Data Formats:**

- Spatial vectors: GeoPackage (`.gpkg`)
- ML-ready tabular: Parquet (`.parquet`)
- Rasters: GeoTIFF with LZW compression
- Configs: YAML (`.yaml`)

**Critical Patterns:**

- Use `pathlib.Path` for all file paths (never strings)
- Always use context managers for raster I/O (`with rasterio.open(...)`)
- WFS downloads are paginated (see [trees.py](src/urban_tree_transfer/data_processing/trees.py) for pagination logic)
- Polygon geometries validated with `shapely.validation.make_valid` ([boundaries.py](src/urban_tree_transfer/data_processing/boundaries.py))

### Google Colab Integration

**Package Installation Pattern:**

```python
# In Colab notebooks
!pip install git+https://github.com/SilasPignotti/urban-tree-transfer.git -q
from google.colab import drive
drive.mount('/content/drive')
```

**Key Constraints:**

- Runtime limit: ~12h continuous execution
- RAM: 12-25GB depending on runtime type
- Google Drive serves as persistent storage
- All paths must be configurable (no hardcoded local paths)

**Earth Engine Authentication:**

```python
import ee
ee.Authenticate()  # First time only
ee.Initialize(project='treeclassifikation')
```

### Testing Strategy

**Test Structure:**

- Unit tests: `tests/data_processing/`, `tests/feature_engineering/`
- Integration tests: `tests/integration/test_endpoints.py` (marked with `@pytest.mark.integration`)

**Fixtures Pattern:**

```python
@pytest.fixture
def berlin_config():
    from urban_tree_transfer.config.loader import load_city_config
    return load_city_config("berlin")
```

Run tests:

```bash
uv run nox -s test                   # Unit tests only
uv run nox -s test_integration       # Integration tests (hit external APIs)
```

**Integration Test Pattern:** Minimal data fetching (1 feature per WFS, HEAD requests for URLs) to verify configs without long downloads.

### Type Safety Requirements

- **Type hints required** for all function signatures
- **Pyright** enforces basic type checking
- Use `pandas.Int64Dtype()` and `pandas.Float64Dtype()` for nullable integers/floats (avoid numpy types that don't support NaN properly)

Example:

```python
def harmonize_trees(raw: gpd.GeoDataFrame, config: dict[str, Any]) -> gpd.GeoDataFrame:
    """Harmonize tree cadastre to standard schema."""
    harmonized = raw.copy()
    harmonized["plant_year"] = harmonized["plant_year"].astype(pd.Int64Dtype())
    return harmonized
```

### Code Quality Standards

- **Line length:** 100 characters
- **String quotes:** Double quotes (`"text"`)
- **Imports:** Sorted by `ruff` (stdlib → third-party → first-party)
- **Docstrings:** Google-style for public functions
- **Random seed:** Always use `RANDOM_SEED = 42` from constants

**Anti-patterns to avoid:**

- ❌ Manual pip edits to `pyproject.toml`
- ❌ String paths instead of `pathlib.Path`
- ❌ Hardcoded URLs/paths (use configs)
- ❌ Missing type hints
- ❌ Unvalidated polygon geometries

## Project-Specific Knowledge

### Data Processing Pipeline (Phase 1)

**Order of Operations:**

1. Boundaries → Download city polygons via WFS
2. Trees → Download cadastres, harmonize schema, filter by genus frequency
3. Elevation → Download DOM/DGM from ATOM feeds, create 1m CHM
4. Sentinel-2 → Monthly composites via Google Earth Engine (separate monitoring)

**CHM Creation:** `CHM = DOM - DGM`, where DOM is surface model, DGM is terrain model. Values clipped to `[-2, 50]` meters ([constants.py](src/urban_tree_transfer/config/constants.py)).

### WFS Pagination Pattern

Berlin/Leipzig WFS services return max 5000 features per request. See [trees.py](src/urban_tree_transfer/data_processing/trees.py) for pagination implementation:

```python
def _download_wfs_layer(url: str, layer: str, *, page_size: int = 5000) -> gpd.GeoDataFrame:
    # Fetch total count, then paginate with STARTINDEX + COUNT parameters
    # Try multiple output formats (GeoJSON, GML) for compatibility
```

### Vegetation Indices

13 indices calculated from Sentinel-2 bands (see [constants.py](src/urban_tree_transfer/config/constants.py)):
`NDVI, EVI, GNDVI, NDre1, NDVIre, CIre, IRECI, RTVIcore, NDWI, MSI, NDII, kNDVI, VARI`

These are computed in Google Earth Engine during monthly composite generation.

### Documentation Structure

- `PRDs/`: Product requirement documents (active: [002_phase2_feature_engineering.md](PRDs/002_phase2_feature_engineering.md))
- `docs/documentation/`: Methodological documentation per phase
- `CLAUDE.md`/`AGENT.md`: AI assistant guidelines (source of truth for conventions)
- `CHANGELOG.md`: Version history

**When implementing features:** Check PRDs first, they contain acceptance criteria and methodological decisions.

## Common Tasks

**Add new city:**

1. Create `configs/cities/{city}.yaml` with WFS endpoints
2. Add fixtures to `tests/conftest.py`
3. Add integration test to `tests/integration/test_endpoints.py`

**Add new data processing module:**

1. Create module in `src/urban_tree_transfer/data_processing/`
2. Add tests in `tests/data_processing/test_{module}.py`
3. Export key functions in `__init__.py`
4. Update runner notebook in `notebooks/runners/`

**Run full quality checks before PR:**

```bash
uv run nox -s ci
```

---

_Last Updated: 2026-01-28_
