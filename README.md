# Cross-City Transfer of Urban Tree Genus Classification Using Multitemporal Sentinel-2 Data: Methodology, Workflow, and Evaluation

## Abstract

Urban tree cadastres are central to informed municipal tree management, yet prohibitive survey costs leave many cities without one. Satellite-based classification using freely available Sentinel-2 data offers a scalable alternative, but models trained on one city typically fail when applied to another due to domain shift. No systematic baselines currently quantify this cross-city transfer gap for urban tree genus classification. This study establishes the first such baseline with a reproducible pipeline from raw data to transfer evaluation across 17 genus-level classes in Berlin (source, 905,132 trees) and Leipzig (target, 167,867 trees). The pipeline compares two algorithmic paradigms: XGBoost for tabular machine learning (ML) and a one-dimensional Convolutional Neural Network (1D-CNN) for temporal deep learning (DL). On the source domain, XGBoost achieves weighted F1 = 0.751 under spatial block cross-validation (CV) with 1200 m blocks, exceeding published Sentinel-2-only and multi-sensor benchmarks despite higher class count and stricter evaluation. Zero-shot transfer reveals a substantial domain gap (XGBoost -49.8%, 1D-CNN -37.4%). The 1D-CNN retains more source performance, plausibly through greater reliance on temporal patterns rather than absolute spectral values, though differing feature dimensionality between paradigms (50 vs. 144 features) prevents definitive attribution. Fine-tuning with local target data recovers performance along a power-law trajectory for XGBoost (weighted F1 = 0.771 at 100% Leipzig data), while the 1D-CNN plateaus early (~0.458). This plateau reflects suboptimal fine-tuning strategy rather than an architectural limitation. A from-scratch baseline (weighted F1 = 0.786) becomes competitive at full data availability; the practical transfer advantage lies at medium data fractions (25-50%). Pipeline design decisions, particularly class balancing (+18 percentage points weighted F1), contribute more to performance than algorithm or hyperparameter choice. At the same time, improved source-domain balancing widens the transfer gap: source optimisation and cross-city robustness are competing objectives. A-priori hypothesis testing confirms genus-heterogeneous transfer loss but rejects spectral separability as a predictor of transfer robustness. Training sample size emerges as the primary moderating variable. The study provides a sample-efficiency framework directly translatable to municipal fieldwork budgets for cadastre creation.

## Methodology and Key Results

The pipeline processes multitemporal Sentinel-2 L2A composites (12 monthly medians, 10 bands + 13 vegetation indices) and municipal tree cadastres from Berlin and Leipzig. After harmonisation, quality filtering (NDVI plausibility, temporal completeness, CHM-based snap-to-peak correction), and proximity filtering (5 m minimum inter-genus distance), 782,022 trees remain (659,266 Berlin, 122,756 Leipzig) with 50 importance-ranked features (ML) or 144 full-temporal features (DL).

Four classifiers are compared across two paradigms -- RF and XGBoost (ML), 1D-CNN and TabNet (DL) -- under 3-fold spatial block CV (1200 m blocks). Champions: XGBoost (ML) and 1D-CNN (DL). Three sequential experiments evaluate source-domain performance, zero-shot cross-city transfer, and fine-tuning recovery at four data fractions (10%, 25%, 50%, 100%).

**Source domain (Berlin test set):**

- XGBoost: weighted F1 = 0.751 [95% CI: 0.749, 0.754]
- 1D-CNN: weighted F1 = 0.607 [95% CI: 0.604, 0.611]

**Zero-shot transfer (Leipzig, no adaptation):**

- XGBoost: weighted F1 = 0.377 (-49.8%)
- 1D-CNN: weighted F1 = 0.380 (-37.4%)

**Fine-tuning (XGBoost, Leipzig):**

- 10% data: 0.499 | 25%: 0.574 | 50%: 0.662 | 100%: 0.771
- From-scratch baseline: 0.786
- Transfer sweet spot: 25-50% local data

For full methodology, results, and discussion, see the companion project report ([Pignotti_Cross-City Transfer of Urban Tree Genus Classification.pdf](./Pignotti_Cross-City%20Transfer%20of%20Urban%20Tree%20Genus%20Classification.pdf)).

## Repository Structure

```text
urban-tree-transfer/
├── src/urban_tree_transfer/      # Installable Python package (shared pipeline logic)
├── notebooks/
│   ├── runners/                  # Colab runner notebooks (execution pipeline)
│   ├── exploratory/              # Exploratory analysis notebooks
│   └── templates/                # Notebook templates
├── tests/                        # Unit and integration tests
├── docs/                         # Methodology and architecture docs
├── outputs/                      # Committed logs, metadata, report-ready JSON
└── scripts/                      # Utility scripts
```

## Data Access

The repository does not include raw data. All data are acquired externally and processed in Google Colab with Google Drive.

- **Sentinel-2 L2A composites:** Google Earth Engine collection `COPERNICUS/S2_SR_HARMONIZED` ([catalog link](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED))
- **Berlin tree cadastre + boundary WFS:**
    - https://gdi.berlin.de/services/wfs/baumbestand
    - https://gdi.berlin.de/services/wfs/alkis_land
- **Leipzig/Saxony tree cadastre + boundary WFS:**
    - https://geoserver.leipzig.de/geoserver/OpenData/Baeume/wfs
    - https://geodienste.sachsen.de/aaa/public_alkis/vereinf/wfs
- **CHM source rasters:**
    - Berlin DOM/DGM (GDI Berlin ATOM feeds)
    - Saxony DOM/DGM tiles (GeoSN)

WFS endpoints are openly reachable. Some elevation products may require institutional access depending on provider policy.

## Environment Setup

**Requirements:** Python 3.10+

```bash
uv sync
```

**Validation:**

```bash
uv run nox -s pre_commit
uv run nox -s test
```

**External requirements:**

- Google Earth Engine authentication (`ee.Initialize()` in Colab)
- Google Drive mount with project data layout (~24 GB)

## Reproducing Results

Execution is designed for Google Colab with Drive-mounted data. GPU is optional (required only for neural network training/tuning).

Notebooks are organised into **exploratory analysis** (`notebooks/exploratory/`) and **runner notebooks** (`notebooks/runners/`). Exploratory notebooks document data exploration, feature analysis, and setup decisions. Runner notebooks execute the main pipeline in four phases:

1. **Data Processing** -- boundary preparation, cadastre harmonisation, elevation inputs, Sentinel-2 composite orchestration
2. **Feature Engineering** -- extraction, quality control, outlier analysis, proximity filtering, split preparation
3. **Experiments** -- setup ablation, algorithm comparison, hyperparameter optimisation, model selection
4. **Evaluation** -- zero-shot transfer metrics, fine-tuning curves, hypothesis tests

Runners must be executed in dependency order (data processing before feature engineering before experiments before evaluation). See `notebooks/runners/` for the complete set and ordering.

After each phase, copy generated logs and metadata from Drive into matching `outputs/<phase>/` folders and commit them for reproducibility.

## Outputs

Committed outputs under `outputs/` serve as the reproducibility audit trail (logs, metadata, report-ready JSON exports), organised by pipeline phase.

Report figure naming convention uses stable slugs:

- `fig-study-area` -> `reports/figures/fig-study-area.png`
- `fig-transfer-robustness` -> `reports/figures/fig-transfer-robustness.png`

Supporting metrics are stored in `outputs/phase_*/report/*.json` and `outputs/phase_*/metadata/*.json`.

## Citation

If you use this repository, please cite the companion project report:

```bibtex
@techreport{pignotti2026urban_tree_transfer,
  author      = {Silas Pignotti},
  title       = {Cross-City Transfer of Urban Tree Genus Classification
                 Using Multitemporal Sentinel-2 Data: Methodology,
                 Workflow, and Evaluation},
  year        = {2026},
  institution = {Berliner Hochschule f{\"u}r Technik (BHT)},
  type        = {Project Report},
  note        = {M.Sc. Geoinformation. Companion code repository:
                 \url{https://github.com/spignotti/urban-tree-transfer}. Final report PDF:
                 \url{https://github.com/spignotti/urban-tree-transfer/blob/main/Pignotti_Cross-City%20Transfer%20of%20Urban%20Tree%20Genus%20Classification.pdf}}
}
```

## License

This repository is currently provided without an explicit license. All rights reserved by the author. If you intend to reuse code or data artifacts, please contact the author.
