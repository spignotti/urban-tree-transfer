# Urban Tree Transfer - Claude Guidelines

> **Note:** This file contains project-specific guidelines. For global development principles, see `~/.claude/CLAUDE.md`

## Project Overview

**Project Name:** Urban Tree Transfer
**Description:** Cross-City Transfer Learning for urban tree genus classification with Sentinel-2 satellite data
**Tech Stack:** Python 3.10+, GeoPandas, Rasterio, scikit-learn, XGBoost, PyTorch

## Project Structure

```
urban-tree-transfer/
├── src/urban_tree_transfer/      # Source package
│   ├── config/                   # Paths, cities, features
│   ├── data_processing/          # Boundaries, Trees, Elevation, CHM, Sentinel
│   ├── feature_engineering/      # Extraction, QC, Selection, Splits
│   ├── experiments/              # Models, Training, Evaluation
│   └── utils/                    # IO, Visualization
├── configs/                      # YAML configurations
│   ├── cities/                   # berlin.yaml, leipzig.yaml
│   └── experiments/              # phase_1.yaml, phase_2.yaml, phase_3.yaml
├── notebooks/
│   ├── runners/                  # Colab-Runner (import src/)
│   └── exploratory/              # Development notebooks
├── tests/                        # Test files
├── scripts/                      # Standalone scripts
├── docs/                         # Documentation
│   ├── PROJECT.md
│   └── documentation/
├── noxfile.py                    # Nox automation
├── pyproject.toml                # Dependencies & config
├── CLAUDE.md                     # This file
├── CHANGELOG.md                  # Version history
└── README.md
```

## Development Environment

### UV Package Management

This project uses UV for Python package and environment management.

**Essential Commands:**

```bash
# Setup
uv venv                   # Create virtual environment
uv sync                   # Install dependencies from lockfile

# Package Management
uv add <package>          # Add runtime dependency
uv add --dev <package>    # Add development dependency
uv remove <package>       # Remove dependency

# Running Code
uv run python <script>    # Run Python script
uv run nox                # Run all quality checks
```

**Important:** NEVER update dependencies directly in `pyproject.toml` - always use `uv add`

### Development Stack

| Tool        | Purpose                                              |
| ----------- | ---------------------------------------------------- |
| **UV**      | Package manager - 10-100x faster than pip            |
| **Ruff**    | Linter + Formatter (replaces flake8/black/isort)     |
| **Pyright** | Static type checker - finds type errors before runtime |
| **Nox**     | Task orchestrator - Python-based automation          |
| **Pydantic**| Runtime validation - type-safe data models           |

## Code Style & Standards

### Python Conventions

- **Style Guide:** PEP 8
- **Line Length:** 100 characters
- **String Quotes:** Double quotes (`"`)
- **Type Hints:** Required for all function signatures
- **Formatter:** `ruff format` (via nox)
- **Linter:** `ruff check` (via nox)
- **Type Checker:** `pyright` (via nox)

### Naming Conventions

- **Variables/Functions:** `snake_case`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_SNAKE_CASE`
- **Private:** `_leading_underscore`

## Project-Specific Conventions

### Geospatial Data

- Use `geopandas` for vector data
- Use `rasterio`/`rioxarray` for raster data
- Use `pathlib.Path` for file paths (not strings)
- Coordinate system: EPSG:25833 (UTM zone 33N)

### Data Formats

- **Spatial:** GeoPackage (.gpkg)
- **ML-Ready:** Parquet (.parquet)
- **Config:** YAML (.yaml)
- **Metadata:** JSON (.json)

### Machine Learning

- Random seed: `42` for all stochastic processes
- Prefer vectorized operations over loops
- Use pandas for tabular data, numpy for numerical operations

### Colab Compatibility

- Package is installable via pip from GitHub
- All paths configurable via config module
- Google Drive mounting handled in notebooks

## Nox Sessions

```bash
uv run nox -s lint        # Check code with ruff
uv run nox -s format      # Format code with ruff
uv run nox -s typecheck   # Type check with pyright
uv run nox -s fix         # Auto-fix all issues
uv run nox -s pre_commit  # Run before commit
uv run nox -s ci          # Full CI pipeline
```

## Code Review Checklist

- [ ] Type hints present for all functions
- [ ] Docstrings for public functions (Google-style)
- [ ] Error handling implemented
- [ ] No hardcoded values (use config/constants)
- [ ] Resources properly managed (context managers)
- [ ] All nox sessions pass

---

_Last Updated: 2026-01-21_
