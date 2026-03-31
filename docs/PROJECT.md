# Urban Tree Transfer

**Status**: ACTIVE

**Owner**: Silas Pignotti

**Repository**: github.com/spignotti/urban-tree-transfer

**Last Updated**: 2026-03-10

---

## Project Description

Automated genus-level classification of urban trees based on multitemporal Sentinel-2 satellite
imagery (reference year 2021). The project investigates cross-city transferability of ML/DL models
trained on a data-rich source domain (Berlin) and applied to a data-scarce target domain (Leipzig).

**Research Question**: How well do ML/DL tree genus classification models transfer from Berlin to
Leipzig, and how much local data is required for performance recovery via fine-tuning?

Three sequential experiments:

1. **Berlin Optimization** -- Model training and hyperparameter optimization on Berlin data
2. **Zero-Shot Transfer** -- Direct application of Berlin-trained models to Leipzig (no retraining)
3. **Fine-Tuning Analysis** -- Incremental fine-tuning with increasing amounts of local Leipzig data

**Primary Metric**: Weighted F1-Score

---

## Study Design

### Cities

| City | Role | Data |
|------|------|------|
| **Berlin** | Source Domain (Training) | ~800k trees, DOM/DGM 2021 |
| **Leipzig** | Target Domain (Transfer) | ~57k trees, DOM/DGM 2022, ~190 km from Berlin |

### Models

| Model | Type | Description |
|-------|------|-------------|
| **Random Forest** | ML | Baseline, interpretable |
| **XGBoost** | ML | Gradient Boosting, class balancing via `scale_pos_weight` |
| **1D-CNN** | DL | Temporal convolution on feature vectors |
| **TabNet** | DL | Attention-based architecture for tabular data |

### Feature Set (147 features after selection)

- **Spectral**: 10 Sentinel-2 L2A bands (B02-B12) as monthly composites via Google Earth Engine
- **Vegetation Indices**: 13 indices (NDVI, EVI, Red-Edge variants, water indices, etc.)
- **Temporal**: 8 selected months (April-November, JM-Distance-based selection from 12)
- **Structural**: Canopy Height Model (CHM) derived from DOM/DGM difference

### CRS

EPSG:25833 (ETRS89 / UTM Zone 33N)

---

## Pipeline Overview

### Phase 1: Data Processing (complete)

Raw data acquisition and harmonization. 1,072,999 trees, ~24 GB total.

- Sentinel-2 L2A download via Google Earth Engine (monthly composites, vegetation period 2021)
- Canopy Height Model generation from DOM (Digital Surface Model) and DGM (Digital Terrain Model)
- Tree cadastre integration (Berlin WFS via gdi.berlin.de, Leipzig WFS via geodienste.leipzig.de)
- Boundary extraction and spatial subsetting

### Phase 2: Feature Engineering (complete)

Full pipeline from raw spatial data to ML-ready datasets. 983,782 trees after QC, 147 final features.

Pipeline steps in order:

1. **Temporal Selection** -- JM-Distance-based month selection (8 of 12 months retained)
2. **CHM Assessment** -- Genus-specific height validation, Welch-ANOVA, Cohen's d effect sizes
3. **Correlation Analysis** -- Pearson-based redundancy removal (|r| > 0.95)
4. **Outlier Detection** -- Tripartite approach (Z-Score, Mahalanobis, IQR), consensus-based flagging
5. **Spatial Autocorrelation** -- Moran's I analysis, block size determination (1200m)
6. **Mixed-Genus Proximity** -- 5m proximity filter, dual-dataset strategy (strict/relaxed)
7. **Feature Extraction** -- Position correction via CHM centroid, point-sampling from rasters
8. **Data Quality** -- Temporal interpolation for NaN values, CHM normalization, NDVI plausibility checks
9. **Final Preparation** -- Redundancy removal, outlier flagging, spatial block assignment, stratified splits

Key parameters: 1200m block size, 5m proximity threshold, 79.5% retention after proximity filter.

### Phase 3: Experiments (complete)

Three sequential experiments executed:

1. **Berlin Optimization**: Hyperparameter tuning (Optuna) for all 4 models on Berlin spatial-CV
2. **Zero-Shot Transfer**: Best Berlin models applied directly to Leipzig test set
3. **Fine-Tuning**: Leipzig training subsets (10%-100%) used for fine-tuning, learning curves analyzed

Evaluation: Weighted F1, per-genus F1, confusion matrices, feature importance (RF/XGBoost),
attention weights (TabNet).

---

## Architecture

```
urban-tree-transfer/
├── src/                          # Python package (installable)
│   ├── config/                   # Paths, cities, features
│   ├── data_processing/          # Boundaries, Trees, Elevation, CHM, Sentinel
│   ├── feature_engineering/      # Extraction, QC, Selection, Splits
│   ├── experiments/              # Models, Training, Evaluation
│   └── utils/                    # IO, Visualization
├── configs/                      # YAML configurations
│   ├── cities/                   # berlin.yaml, leipzig.yaml
│   └── experiments/              # phase_1.yaml, phase_2.yaml, phase_3.yaml
├── notebooks/
│   ├── runners/                  # Colab runner notebooks (import src/)
│   └── exploratory/              # Development notebooks
└── pyproject.toml
```

### External Dependencies

- **Google Earth Engine**: Sentinel-2 L2A composite download
- **Google Drive**: Data storage (~24 GB)
- **Google Colab**: GPU execution (T4/A100), ~12h runtime limit, 12-25 GB RAM

### Key Libraries

Python 3.10+, GeoPandas, rasterio, scikit-learn, XGBoost, PyTorch (1D-CNN, TabNet),
Optuna, scipy, numpy, pandas

### Data Formats

- **Spatial**: GeoPackage (.gpkg)
- **ML-Ready**: Parquet (.parquet)
- **Config**: YAML (.yaml)
- **Metadata**: JSON (.json)
- **Random Seed**: 42

---

## Known Limitations

- Single-year snapshot (2021) -- no multi-year temporal generalization
- Only two cities (Berlin -> Leipzig) -- limited geographic diversity
- Genus-level classification only (not species)
- Sentinel-2 10m resolution with single-pixel point sampling
- Global spatial block size (1200m) for heterogeneous urban landscape
- Unweighted outlier consensus across three methods
- Linear interpolation for temporal NaN values
- Computational constraints (Google Colab), single-person project

---

## Known Improvements (tagged)

Priority improvements for potential rework:

- **[Rework]**: XGBoost class balancing strategy, Welch-ANOVA assumptions, Spearman correlation
  supplement, Mahalanobis stability, visualization decoupling from notebooks, CHM documentation
  inconsistencies, KL-Divergence justification, hypothesis testing for transfer results
- **[Paper]**: Multi-pixel sampling, soft-weighting for outliers, phenological phase features,
  bootstrap confidence intervals, seasonal autocorrelation, weighted consensus, domain adaptation

Full list maintained in Notion (Improvements & Limitationen).

---

## Related

- **Documentation SSOT**: Notion workspace (full methodology documentation migrated from repo)
