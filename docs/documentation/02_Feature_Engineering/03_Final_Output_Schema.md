# Final Output Schema (Phase 2c)

This document specifies the exact column schema for the 10 GeoPackages produced by
`notebooks/runners/02c_final_preparation.ipynb`.

## Output Files

Baseline outputs (no proximity filtering):

- `berlin_train.gpkg`
- `berlin_val.gpkg`
- `berlin_test.gpkg`
- `leipzig_finetune.gpkg`
- `leipzig_test.gpkg`

Filtered outputs (after proximity filter):

- `berlin_train_filtered.gpkg`
- `berlin_val_filtered.gpkg`
- `berlin_test_filtered.gpkg`
- `leipzig_finetune_filtered.gpkg`
- `leipzig_test_filtered.gpkg`

All files share the same schema.

## CRS

- EPSG:25833 (UTM Zone 33N)

## Column Schema

### 1) Identifiers & Metadata (11)

- `tree_id` (string, non-null)
- `city` (string, non-null, values: "berlin" or "leipzig")
- `genus_latin` (string, non-null)
- `species_latin` (string, nullable)
- `genus_german` (string, nullable)
- `species_german` (string, nullable)
- `plant_year` (Int64, nullable)
- `height_m` (float, nullable, range: >= 0)
- `tree_type` (string, nullable)
- `position_corrected` (boolean, non-null)
- `correction_distance` (float, non-null, meters, range: >= 0)

### 2) Geometry (1)

- `geometry` (Point, non-null, EPSG:25833)

### 3) CHM Features (3)

- `CHM_1m` (float, nullable, meters)
- `CHM_1m_zscore` (float, non-null)
- `CHM_1m_percentile` (float, non-null, range: 0-100)

### 4) Sentinel-2 Temporal Features (184)

Selected months are determined by `outputs/phase_2/metadata/temporal_selection.json`.
This schema assumes selected months **[04-11]**, yielding 8 months × 23 features = 184 columns.

Base features (23):
`B2`, `B3`, `B4`, `B5`, `B6`, `B7`, `B8`, `B8A`, `B11`, `B12`,
`NDVI`, `EVI`, `GNDVI`, `NDre1`, `NDVIre`, `CIre`, `IRECI`, `RTVIcore`,
`NDWI`, `MSI`, `NDII`, `kNDVI`, `VARI`.

Exact temporal columns:

- Month 04: `B2_04`, `B3_04`, `B4_04`, `B5_04`, `B6_04`, `B7_04`, `B8_04`, `B8A_04`,
  `B11_04`, `B12_04`, `NDVI_04`, `EVI_04`, `GNDVI_04`, `NDre1_04`, `NDVIre_04`,
  `CIre_04`, `IRECI_04`, `RTVIcore_04`, `NDWI_04`, `MSI_04`, `NDII_04`, `kNDVI_04`, `VARI_04`
- Month 05: `B2_05`, `B3_05`, `B4_05`, `B5_05`, `B6_05`, `B7_05`, `B8_05`, `B8A_05`,
  `B11_05`, `B12_05`, `NDVI_05`, `EVI_05`, `GNDVI_05`, `NDre1_05`, `NDVIre_05`,
  `CIre_05`, `IRECI_05`, `RTVIcore_05`, `NDWI_05`, `MSI_05`, `NDII_05`, `kNDVI_05`, `VARI_05`
- Month 06: `B2_06`, `B3_06`, `B4_06`, `B5_06`, `B6_06`, `B7_06`, `B8_06`, `B8A_06`,
  `B11_06`, `B12_06`, `NDVI_06`, `EVI_06`, `GNDVI_06`, `NDre1_06`, `NDVIre_06`,
  `CIre_06`, `IRECI_06`, `RTVIcore_06`, `NDWI_06`, `MSI_06`, `NDII_06`, `kNDVI_06`, `VARI_06`
- Month 07: `B2_07`, `B3_07`, `B4_07`, `B5_07`, `B6_07`, `B7_07`, `B8_07`, `B8A_07`,
  `B11_07`, `B12_07`, `NDVI_07`, `EVI_07`, `GNDVI_07`, `NDre1_07`, `NDVIre_07`,
  `CIre_07`, `IRECI_07`, `RTVIcore_07`, `NDWI_07`, `MSI_07`, `NDII_07`, `kNDVI_07`, `VARI_07`
- Month 08: `B2_08`, `B3_08`, `B4_08`, `B5_08`, `B6_08`, `B7_08`, `B8_08`, `B8A_08`,
  `B11_08`, `B12_08`, `NDVI_08`, `EVI_08`, `GNDVI_08`, `NDre1_08`, `NDVIre_08`,
  `CIre_08`, `IRECI_08`, `RTVIcore_08`, `NDWI_08`, `MSI_08`, `NDII_08`, `kNDVI_08`, `VARI_08`
- Month 09: `B2_09`, `B3_09`, `B4_09`, `B5_09`, `B6_09`, `B7_09`, `B8_09`, `B8A_09`,
  `B11_09`, `B12_09`, `NDVI_09`, `EVI_09`, `GNDVI_09`, `NDre1_09`, `NDVIre_09`,
  `CIre_09`, `IRECI_09`, `RTVIcore_09`, `NDWI_09`, `MSI_09`, `NDII_09`, `kNDVI_09`, `VARI_09`
- Month 10: `B2_10`, `B3_10`, `B4_10`, `B5_10`, `B6_10`, `B7_10`, `B8_10`, `B8A_10`,
  `B11_10`, `B12_10`, `NDVI_10`, `EVI_10`, `GNDVI_10`, `NDre1_10`, `NDVIre_10`,
  `CIre_10`, `IRECI_10`, `RTVIcore_10`, `NDWI_10`, `MSI_10`, `NDII_10`, `kNDVI_10`, `VARI_10`
- Month 11: `B2_11`, `B3_11`, `B4_11`, `B5_11`, `B6_11`, `B7_11`, `B8_11`, `B8A_11`,
  `B11_11`, `B12_11`, `NDVI_11`, `EVI_11`, `GNDVI_11`, `NDre1_11`, `NDVIre_11`,
  `CIre_11`, `IRECI_11`, `RTVIcore_11`, `NDWI_11`, `MSI_11`, `NDII_11`, `kNDVI_11`, `VARI_11`

### 5) Outlier Flags (5)

- `outlier_zscore` (boolean, non-null)
- `outlier_mahalanobis` (boolean, non-null)
- `outlier_iqr` (boolean, non-null)
- `outlier_severity` (string, non-null, values: "none", "low", "medium", "high")
- `outlier_method_count` (integer, non-null, range: 0-3)

### 6) Spatial Split Metadata (1)

- `block_id` (string, non-null)

## Redundancy Removal Note

The Phase 2c pipeline removes redundant temporal features using
`outputs/phase_2/metadata/correlation_removal.json`.
If a feature base is removed (e.g., `B8A`), **all** month instances are removed
(e.g., `B8A_04` ... `B8A_11`).

When `correlation_removal.json` is updated, this document should be revised to
list the exact retained temporal columns (post-removal).

## Expected Column Counts

Column counts depend on methodological decisions made in exploratory analysis:

- **Phase 2a (extraction):** 11 metadata + 1 CHM + (23 bases × 12 months S2)
- **Phase 2b (quality):** 11 metadata + 3 CHM + (23 bases × selected months)
- **Phase 2c (final):** 11 metadata + 3 CHM + (retained bases × selected months) + 5 outlier + 1 spatial

**Note:** Exact counts are determined by:

- Temporal selection (exploratory analysis exp_01)
- Correlation threshold and redundant features (exploratory analysis exp_03)
