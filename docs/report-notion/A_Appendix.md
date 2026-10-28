# A. Appendix

> Migriert aus Notion am 2026-10-28 (Seite `23f0475498004ed3b517fe1a3c6e238f`, zuletzt bearbeitet 2026-03-30). Text unverändert übernommen.

Appendix-Inhalte. Marker (`[FIGURE:]`, `[TABLE:]`) matchen die Einträge in Visualizations bzw. Tables.

---

### A.1 Hypothesis Test Summary

\[TABLE: tbl-hypothesis-tests\]

### A.2 Feature List (147 Features)

*Spectral Bands (64 features: 8 bands x 8 months, April--November):*

B2\_04, B2\_05, B2\_06, B2\_07, B2\_08, B2\_09, B2\_10, B2\_11, B3\_04, B3\_05, B3\_06, B3\_07, B3\_08, B3\_09, B3\_10, B3\_11, B4\_04, B4\_05, B4\_06, B4\_07, B4\_08, B4\_09, B4\_10, B4\_11, B5\_04, B5\_05, B5\_06, B5\_07, B5\_08, B5\_09, B5\_10, B5\_11, B8\_04, B8\_05, B8\_06, B8\_07, B8\_08, B8\_09, B8\_10, B8\_11, B8A04, B8A05, B8A06, B8A07, B8A08, B8A09, B8A10, B8A11, B11\_04, B11\_05, B11\_06, B11\_07, B11\_08, B11\_09, B11\_10, B11\_11, B12\_04, B12\_05, B12\_06, B12\_07, B12\_08, B12\_09, B12\_10, B12\_11

*Vegetation Indices (80 features: 10 indices x 8 months, April--November):*

CIre04, CIre05, CIre06, CIre07, CIre08, CIre09, CIre10, CIre11, EVI04, EVI05, EVI06, EVI07, EVI08, EVI09, EVI10, EVI11, IRECI04, IRECI05, IRECI06, IRECI07, IRECI08, IRECI09, IRECI10, IRECI11, MSI04, MSI05, MSI06, MSI07, MSI08, MSI09, MSI10, MSI11, NDVI04, NDVI05, NDVI06, NDVI07, NDVI08, NDVI09, NDVI10, NDVI11, NDVIre04, NDVIre05, NDVIre06, NDVIre07, NDVIre08, NDVIre09, NDVIre10, NDVIre11, NDWI04, NDWI05, NDWI06, NDWI07, NDWI08, NDWI09, NDWI10, NDWI11, NDre1\_04, NDre1\_05, NDre1\_06, NDre1\_07, NDre1\_08, NDre1\_09, NDre1\_10, NDre1\_11, RTVIcore04, RTVIcore05, RTVIcore06, RTVIcore07, RTVIcore08, RTVIcore09, RTVIcore10, RTVIcore11, VARI04, VARI05, VARI06, VARI07, VARI08, VARI09, VARI10, VARI11

*CHM (3 features):*

CHM\_1m, CHM1m\_percentile, CHM1m\_zscore

### A.3 Supplementary Confusion Matrices

Four normalized confusion matrices are included as supplementary figures (Figs. A1--A4): Berlin XGBoost (A1), Berlin 1D-CNN (A2), Leipzig zero-shot XGBoost + 1D-CNN as a two-panel figure (A3), and Leipzig fine-tuned 100% XGBoost (A4). All matrices use the 17-class label set defined in Section 4.2.

\[FIGURE: fig-confusion-berlin-xgb\]

\[FIGURE: fig-confusion-berlin-cnn\]

\[FIGURE: fig-confusion-leipzig-zeroshot\]

\[FIGURE: fig-confusion-leipzig-finetuned100\]

### A.4 Hyperparameter Configurations

*XGBoost (Optuna, Tree-structured Parzen Estimator (TPE) sampler, MedianPruner, 40 trials on 100k subset):*

\[TABLE: tbl-hyperparams-xgboost\]

*1D-CNN (Optuna, 20 trials on 100k subset):*

- Architecture: 3 convolutional layers (64, 128, 128 filters), kernel size 5
- 2 dense layers (128, 64 units)
- Dropout regularisation
- Early stopping on validation weighted F1 (patience: 8 epochs)
- Input: 144 features (12 bands/indices x 12 months, reshaped as temporal sequence)

\[TABLE: tbl-hyperparams-cnn\]
