# Product Requirements Documents (PRDs)

This directory contains all Product Requirements Documents for the urban-tree-transfer project.

## Active PRDs

### Phase 2: Feature Engineering (Current)

- **[002_phase2_feature_engineering_overview.md](002_phase2_feature_engineering_overview.md)** - Main overview PRD
- **[002_phase2/](002_phase2/)** - Modular task PRDs
  - [002a_feature_extraction.md](002_phase2/002a_feature_extraction.md) - CHM/S2 feature extraction
  - [002b_data_quality.md](002_phase2/002b_data_quality.md) - NaN handling, plausibility filters
  - [002c_final_preparation.md](002_phase2/002c_final_preparation.md) - Outliers, spatial splits
  - [002_exploratory.md](002_phase2/002_exploratory.md) - Exploratory analysis notebooks

**Working with Coding Agents:** Always provide the overview PRD + specific task PRD.

## Completed PRDs

### Phase 1: Data Processing

- **[done/001_phase1_data_processing.md](done/001_phase1_data_processing.md)** - ✅ Complete
  - Tree cadastre harmonization
  - CHM generation from DOM/DGM
  - Sentinel-2 monthly composites

## PRD Structure Philosophy

### Monolithic (Phase 1)

- Single comprehensive document
- Good for smaller, focused tasks
- Example: PRD 001

### Modular (Phase 2+)

- Main overview PRD + separate task PRDs
- Better for complex, multi-step work
- Easier for coding agent delegation
- Example: PRD 002 (overview + 002a/b/c)

## Templates

- **[templates/prd_simple.md](templates/prd_simple.md)** - Simple PRD template for small tasks

## Naming Convention

- `{NNN}_phase{N}_{topic}.md` - Main PRD (e.g., `002_phase2_feature_engineering_overview.md`)
- `{NNN}_phase{N}/{NNN}{letter}_{subtask}.md` - Task PRD (e.g., `002_phase2/002a_feature_extraction.md`)
- Completed PRDs move to `done/` folder

## Status Values

- **Draft** - In planning, not yet ready for implementation
- **In Progress** - Actively being implemented
- **Complete** - Implementation finished and validated
- **Superseded** - Replaced by newer version (moved to `done/`)

---

**Last Updated:** 2026-01-28
