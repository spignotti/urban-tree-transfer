# Cross-City Transfer of Urban Tree Genus Classification Using Multitemporal Sentinel-2 Data: Methodology, Workflow, and Evaluation

## Abstract

This repository contains the full companion pipeline for a university geoinformation project on
cross-city transfer of urban tree genus classification from Berlin to Leipzig using multitemporal
Sentinel-2 data. It includes data processing, feature engineering, model training, transfer
evaluation, and fine-tuning analysis, with committed metadata outputs for reproducibility. The
project report draft is maintained in `reports/paper-draft.md` (not intended for GitHub release as
a report artifact).

## Repository Structure

```text
urban-tree-transfer/
├── src/urban_tree_transfer/      # Installable package with shared pipeline logic
├── notebooks/
│   ├── runners/                  # Colab runner notebooks (01, 02a-c, 03a-d)
│   ├── exploratory/              # Exploratory analysis notebooks (exp_01..exp_11)
│   └── templates/                # Notebook templates
├── tests/                        # Unit/integration tests for package modules
├── docs/                         # Methodology and architecture documentation
├── outputs/                      # Committed logs/metadata audit trail
├── scripts/                      # Utility scripts
└── reports/                      # Local report drafting/build assets (gitignored)
```

## Data Access

The repository does not include raw data. Data are acquired externally and processed in Google
Colab + Google Drive.

- Sentinel-2 L2A composites: Google Earth Engine collection
  `COPERNICUS/S2_SR_HARMONIZED`
  (https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED)
- Berlin tree cadastre + boundary WFS:
  - https://gdi.berlin.de/services/wfs/baumbestand
  - https://gdi.berlin.de/services/wfs/alkis_land
- Leipzig/Saxony tree cadastre + boundary WFS:
  - https://geoserver.leipzig.de/geoserver/OpenData/Baeume/wfs
  - https://geodienste.sachsen.de/aaa/public_alkis/vereinf/wfs
- CHM source rasters:
  - Berlin DOM/DGM (GDI Berlin ATOM feeds)
  - Saxony DOM/DGM tiles via GeoSN lists

Availability note: WFS endpoints are openly reachable, while some elevation products and derived
tiles may require institutional access/request workflows depending on provider policy.

## Reproducing Results

1. Install dependencies locally (`uv sync`) and ensure notebooks reference current `main`.
2. Push code changes to `main` (Colab installs package from GitHub `main`, not feature branches).
3. Open notebooks in Colab and run in dependency order:
   - Exploratory setup chain: `exp_08` -> `exp_08b` -> `exp_08c` -> `exp_09` -> `exp_10`
   - Runners: `01_data_processing` -> `02a_feature_extraction` -> `02b_data_quality` ->
     `02c_final_preparation` -> `03a_setup_fixation` -> `03b_berlin_optimization` ->
     `03c_transfer_evaluation` -> `03d_finetuning`
4. After each phase, copy generated logs/metadata from Drive into matching `outputs/<phase>/`
   folders and commit them.
5. Rebuild report figures/tables from committed outputs and local report tooling (`reports/`).

Runtime/resource note: execution is designed for Google Colab with Drive-mounted data (~24 GB).
GPU is optional for some models (notably neural-network training/tuning).

## Environment Setup

- Python: 3.10+
- Package/dependencies:

```bash
uv sync
```

- Validation commands:

```bash
uv run nox -s pre_commit
uv run nox -s test
```

- External requirements:
  - Google Earth Engine authentication (`ee.Initialize()` in Colab context)
  - Google Drive mount with project data layout
  - GitHub token in Colab secrets for private install workflows when needed

## Pipeline Overview

1. **Data Processing** (`01_data_processing`, `src/.../data_processing/`)
   - boundaries, tree cadastres, elevation inputs, Sentinel-2 task orchestration
2. **Feature Engineering** (`02a/02b/02c`, `src/.../feature_engineering/`)
   - extraction, quality control, outlier/proximity logic, split preparation
3. **Experiments** (`exp_07..exp_11`, `03a/03b`, `src/.../experiments/`)
   - setup ablations, algorithm comparison, tuning, model selection
4. **Evaluation** (`03c/03d`, `src/.../experiments/transfer.py`)
   - zero-shot transfer metrics, robustness analysis, fine-tuning curves

## Outputs

Committed outputs are organized under phase folders in `outputs/` and serve as the reproducibility
trail (logs, metadata, report-ready JSON exports).

Report figure naming convention uses stable slugs, e.g.:

- `fig-study-area` -> `reports/figures/fig-study-area.png`
- `fig-transfer-robustness` -> `reports/figures/fig-transfer-robustness.png`

Supporting metrics are stored in `outputs/phase_*/report/*.json` and
`outputs/phase_*/metadata/*.json`.

## Citation

If you use this repository, please cite the companion report/thesis.

```bibtex
@misc{pignotti2026urban_tree_transfer,
  author       = {Silas Pignotti},
  title        = {Cross-City Transfer of Urban Tree Genus Classification Using Multitemporal Sentinel-2 Data: Methodology, Workflow, and Evaluation},
  year         = {2026},
  howpublished = {University geoinformation project report, BHT Berlin},
  note         = {Companion code repository: https://github.com/spignotti/urban-tree-transfer}
}
```

## License

No repository license file is currently included. If this repository is prepared for broader public
reuse, add an explicit license (e.g., MIT) in a follow-up step.
