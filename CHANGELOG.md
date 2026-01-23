# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Add city YAML configs for Berlin and Leipzig
- Add config constants and loader utilities
- Add plotting and execution logging utilities
- Add Phase 1 data processing modules (boundaries, trees, elevation, CHM, Sentinel-2)
- Add Phase 1 runner notebook and runner template
- Add methodology template and Phase 1 data processing methodology
- Add Leipzig WFS tree mapping with species extraction and metadata export
- Add tree_type mapping support for Berlin WFS layers
- Add zip-list elevation download with mosaic + boundary clipping for Leipzig DGM
- Add Leipzig DOM zip-list configuration
- Add Berlin DOM/DGM Atom feed URLs from legacy config
- Add `utils/geo.py` with shared geographic utilities (ensure_project_crs, buffer_boundaries, validate_geometries, clip_to_boundary)
- Add `utils/validation.py` with dataset validation utilities (validate_crs, validate_schema, validate_within_boundary, generate_validation_report)
- Add `validate_polygon_geometries()` to boundaries module using shapely make_valid
- Add `filter_trees_to_boundary()` for spatial filtering with configurable buffer
- Add `remove_duplicate_trees()` for deduplication by tree_id or proximity
- Add Atom feed parser for Berlin elevation data with spatial tile filtering
- Add `clip_chm_to_boundary()` for CHM boundary clipping
- Add `monitor_tasks()` and `batch_validate_sentinel()` for GEE task management
- Add 500m buffer to all boundary-based clipping operations
- Add `outputs/` directory for Colab-generated metadata and logs
- Add runtime settings section to runner notebook (CPU, High-RAM recommended)

### Changed
- Change documentation structure to include Phase 1 methodology
- Change nox sessions to run `uv` with active environment to avoid warnings
- Change notebook to clone repo for metadata commits (large data stays on Drive)
- Change harmonize_trees to use consistent dtypes (Int64, Float64) across cities
- Change Sentinel-2 export to use 500m buffered boundaries
- Update Phase 1 methodology with complete processing steps, parameters, and quality criteria

## [0.1.0] - YYYY-MM-DD

### Added
- Initial project setup
- Basic project structure
- Core functionality implementation
- Development environment configuration (UV, Nox, Ruff, Pyright)
- Test framework setup (pytest, coverage)
- Documentation (README.md, CLAUDE.md)

---

## Guidelines for Maintaining This Changelog

### Version Format
- Use [Semantic Versioning](https://semver.org/): MAJOR.MINOR.PATCH
  - **MAJOR**: Incompatible API changes
  - **MINOR**: Backwards-compatible new features
  - **PATCH**: Backwards-compatible bug fixes

### Categories
Use these standard categories in order:
1. **Added** - New features
2. **Changed** - Changes in existing functionality
3. **Deprecated** - Soon-to-be removed features
4. **Removed** - Removed features
5. **Fixed** - Bug fixes
6. **Security** - Security-related changes

### Writing Entries
- Write entries in **present tense** (e.g., "Add feature" not "Added feature")
- Start each entry with a **verb** (Add, Change, Fix, Remove, etc.)
- Be **specific** and **concise**
- Include **issue/PR references** when applicable: `(#123)`
- Group related changes together
- Keep the audience in mind (users, not developers)

### Examples

#### Good Entries ✅
```markdown
### Added
- Add user authentication with JWT tokens (#45)
- Add support for PostgreSQL database backend
- Add comprehensive API documentation with examples

### Fixed
- Fix memory leak in data processing pipeline (#67)
- Fix incorrect calculation in statistics module (#72)
```

#### Bad Entries ❌
```markdown
### Added
- Added stuff
- Various improvements
- Updated code

### Fixed
- Fixed bug (too vague)
- Refactored everything (not user-facing)
```

### When to Update
- Update `[Unreleased]` section as you develop
- Create a new version section when releasing
- Move items from `[Unreleased]` to the new version
- Update the version links at the bottom

### Version Links (Optional)
If using Git tags, add comparison links at the bottom:
```markdown
[unreleased]: https://github.com/username/repo/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/username/repo/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/username/repo/releases/tag/v0.1.0
```

---

## Template for New Releases

When releasing a new version, copy this template:
```markdown
## [X.Y.Z] - YYYY-MM-DD

### Added
- 

### Changed
- 

### Deprecated
- 

### Removed
- 

### Fixed
- 

### Security
- 
```

Remove empty sections before publishing.
