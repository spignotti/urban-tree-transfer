# Google Drive Folder Structure

All pipeline data lives on Google Drive. Nothing here is committed to the repo
except metadata JSONs, logs, and figures (see `outputs/` in the repo root).

---

## Drive Structure

```
MyDrive/dev/urban-tree-transfer/
└── data/
    ├── phase_1_processing/
    │   ├── trees/
    │   │   └── trees_filtered_viable.gpkg        # Combined Berlin + Leipzig trees
    │   ├── chm/
    │   │   ├── CHM_1m_berlin.tif
    │   │   └── CHM_1m_leipzig.tif
    │   ├── sentinel2/
    │   │   ├── S2_berlin_2021_01.tif             # Monthly composites (01–12)
    │   │   └── S2_leipzig_2021_01.tif
    │   ├── metadata/
    │   └── logs/
    │
    ├── phase_2_features/
    │   ├── trees_with_features_berlin.gpkg        # After 02a
    │   ├── trees_with_features_leipzig.gpkg
    │   ├── trees_clean_berlin.gpkg                # After 02b (QC, 0 NaN)
    │   ├── trees_clean_leipzig.gpkg
    │   ├── metadata/                              # JSON configs from exp_01–06
    │   ├── logs/
    │   └── figures/
    │
    ├── phase_2_splits/
    │   ├── berlin_train.gpkg                      # Baseline splits
    │   ├── berlin_val.gpkg
    │   ├── berlin_test.gpkg
    │   ├── leipzig_finetune.gpkg
    │   ├── leipzig_test.gpkg
    │   ├── berlin_train_filtered.gpkg             # Filtered splits
    │   ├── ...
    │   ├── metadata/
    │   ├── logs/
    │   └── figures/
    │
    └── phase_3_experiments/
        ├── berlin_train.parquet                   # ML-ready splits (XGBoost)
        ├── berlin_train_cnn.parquet               # ML-ready splits (CNN1D, full features)
        ├── berlin_val.parquet
        ├── berlin_val_cnn.parquet
        ├── berlin_test.parquet
        ├── berlin_test_cnn.parquet
        ├── leipzig_finetune.parquet
        ├── leipzig_finetune_cnn.parquet
        ├── leipzig_test.parquet
        ├── leipzig_test_cnn.parquet
        ├── models/
        │   ├── berlin_ml_champion.pkl
        │   ├── berlin_nn_champion.pt
        │   ├── berlin_scaler_ml.pkl
        │   ├── berlin_scaler_nn.pkl
        │   └── label_encoder.pkl
        ├── metadata/                              # setup_decisions.json, eval JSONs
        ├── logs/
        └── figures/
```

## Data Flow

```
Phase 1 (01_data_processing)
  trees_filtered_viable.gpkg + CHM rasters + S2 composites
      |
Phase 2a (02a_feature_extraction)
  trees_with_features_*.gpkg
      |
exp_01, exp_02 -> temporal_selection.json, chm_assessment.json
      |
Phase 2b (02b_data_quality)
  trees_clean_*.gpkg
      |
exp_03–06 -> correlation_removal.json, outlier_thresholds.json,
             spatial_autocorrelation.json, proximity_filter.json
      |
Phase 2c (02c_final_preparation)
  10 split GeoPackages (baseline + filtered)
      |
exp_07–10 -> setup_decisions.json (CHM, proximity, outlier, features, genus)
      |
03a (setup_fixation)
  10 Parquet splits (XGBoost + CNN1D variants)
      |
03b (berlin_optimization) -> models, scalers, hp_tuning JSONs
      |
03c (transfer_evaluation) -> transfer_evaluation.json
      |
03d (finetuning) -> finetuning_curve.json
```

## File Naming

| Type | Pattern | Example |
|------|---------|---------|
| GeoPackage splits | `{city}_{split}[_filtered].gpkg` | `berlin_train_filtered.gpkg` |
| Parquet splits | `{city}_{split}[_cnn].parquet` | `berlin_train_cnn.parquet` |
| CHM rasters | `CHM_1m_{city}.tif` | `CHM_1m_berlin.tif` |
| Sentinel-2 | `S2_{city}_{year}_{month:02d}.tif` | `S2_berlin_2021_04.tif` |
| Exploratory JSON | `{analysis_name}.json` | `temporal_selection.json` |
| Execution log | `{notebook}_execution.json` | `02a_feature_extraction_execution.json` |

## What to Commit

After each phase, download from Drive and commit to `outputs/` in the repo:

```
outputs/<phase>/metadata/*.json   # always
outputs/<phase>/logs/*.json       # always
outputs/<phase>/figures/**/*.png  # optional, but useful for reference
```

Never commit: `.gpkg`, `.parquet`, `.tif`, `.pkl`, `.pt` — these stay on Drive.

## Related

- `docs/PROJECT.md` — research design and phase descriptions
- `AGENTS.md` — workflow, deploy cycle, `outputs/` sync procedure
