---
tables:
  - table_id: tbl-study-areas
    caption: "Comparison of source and target domains (Berlin vs. Leipzig)."
    target_section: "3.1 Study Areas"
    content: |
      | Aspect | Berlin (Source Domain) | Leipzig (Target Domain) |
      | Role | Training + Validation | Zero-Shot Test + Fine-Tuning |
      | Population | ~3.8 million | ~600,000 |
      | Tree cadastre (records used) | 905,132 registered trees | 167,867 registered trees |
      | Climate zone | Cfb (Köppen), temperate | Cfb (Köppen), temperate |
      | Separation | ~150 km (straight line) -- well beyond typical spatial autocorrelation range (<1 km in urban settings) | -- |

  - table_id: tbl-hypothesis-tests
    caption: "Summary of four a-priori hypothesis tests."
    target_section: "Appendix A.1"
    notes: "Supplementary significance tests in prose (McNemar comparisons, p = 0.000)."
    content: |
      | Hypothesis | Statement | Test | Statistic | p-value | Effect Size | Result |
      | H1 | Transfer loss varies significantly across genera | Kruskal-Wallis H | H = 4953.4 | <0.001 | eta-squared = 0.163 | Confirmed |
      | H2 | Evergreen genera exhibit smaller transfer losses | Mann-Whitney U | -- | -- | -- | Inconclusive (n=1 conifer group) |
      | H3 | Higher JM-distance predicts higher transfer accuracy | Spearman rho | rho = -0.132 | 0.613 | -- | Rejected |
      | H4 | Larger zero-shot gap requires more FT data for recovery | Spearman rho | rho = 0.500 | 0.117 | -- | Rejected (positive trend, not significant at alpha = 0.05) |

  - table_id: tbl-main-results
    caption: "Summary of final weighted F1 results across source-domain, zero-shot, fine-tuned, and from-scratch settings."
    target_section: "5 Results"
    content: |
      | Model / Setting | Berlin Test weighted F1 | Leipzig Zero-Shot weighted F1 | Leipzig Fine-Tuned (100%) weighted F1 | Leipzig From-Scratch weighted F1 |
      | XGBoost | 0.751 | 0.377 | 0.771 | 0.786 |
      | 1D-CNN | 0.607 | 0.380 | 0.458 | -- |

  - table_id: tbl-hyperparams-xgboost
    caption: "Final XGBoost hyperparameter configuration selected from Optuna tuning."
    target_section: "Appendix A.4"
    content: |
      | Parameter | Value |
      | max_depth | 12 |
      | learning_rate | 0.0542 |
      | n_estimators | 358 |
      | subsample | 0.851 |
      | colsample_bytree | 0.905 |
      | min_child_weight | 2 |
      | reg_alpha | 0.0765 |
      | reg_lambda | 1.358 |

  - table_id: tbl-hyperparams-cnn
    caption: "Final 1D-CNN training and optimisation configuration selected from Optuna tuning."
    target_section: "Appendix A.4"
    content: |
      | Parameter | Value |
      | learning_rate | 0.00120 |
      | dropout_rate | 0.419 |
      | batch_size | 64 |
      | optimizer | Adam |
      | input_features | 144 (18 variables × 8 months) |
      | conv_layers | 3 (64, 128, 128 filters; kernel size 5) |
      | dense_layers | 2 (128, 64 units) |
      | early_stopping | patience = 8 (validation weighted F1) |
---

# Tables

Marker-driven tables for academic-paper build.
