# C. Tables

> Migriert aus Notion am 2026-10-28 (Seite `fbf11c9368514bc7a721305ad08ad391`, zuletzt bearbeitet 2026-03-26). Text unverändert übernommen.

Tabellen separat vom Prose definiert. Table IDs matchen `[TABLE: tbl-slug]` Marker im Draft.

---

### tbl-study-areas

- **Purpose:** Vergleich der Source- und Target-Domain (Berlin vs. Leipzig)
- **Target Section:** 3.1 Study Areas
- **Formatting Note:** Header row, 3 columns (Aspect, Berlin, Leipzig)
- **Table Content:**

| **Aspect** | **Berlin (Source Domain)** | **Leipzig (Target Domain)** |
| --- | --- | --- |
| **Role** | Training + Validation | Zero-Shot Test + Fine-Tuning |
| **Population** | \~3.8 million | \~600,000 |
| **Tree cadastre (raw)** | \~800,000 street trees | \~200,000 street trees |
| **Climate zone** | Cfb (Köppen), temperate | Cfb (Köppen), temperate |
| **Separation** | \~150 km (straight line) -- well beyond typical spatial autocorrelation range (\&lt;1 km in urban settings) |  |

---

### tbl-hypothesis-tests

- **Purpose:** Zusammenfassung der vier a-priori Hypothesentests
- **Target Section:** Appendix A.1
- **Formatting Note:** Header row, 7 columns
- **Table Content:**

| **Hypothesis** | **Statement** | **Test** | **Statistic** | **p-value** | **Effect Size** | **Result** |
| --- | --- | --- | --- | --- | --- | --- |
| H1 | Transfer loss varies significantly across genera | Kruskal-Wallis H | H = 4953.4 | 0.000 | eta-squared = 0.163 | Confirmed |
| H2 | Evergreen genera exhibit smaller transfer losses | Mann-Whitney U | -- | -- | -- | Inconclusive (n=1 conifer group) |
| H3 | Higher JM-distance predicts higher transfer accuracy | Spearman rho | rho = -0.132 | 0.613 | -- | Rejected |
| H4 | Larger zero-shot gap requires more FT data for recovery | Spearman rho | rho = 0.500 | 0.117 | -- | Rejected (positive trend, not significant at alpha = 0.05) |

Supplementary significance tests:

- McNemar fine-tuned vs. zero-shot: p = 0.000 (significant)
- McNemar fine-tuned vs. from-scratch: p = 0.000 (significant)

---

### tbl-hyperparams-xgboost

- **Purpose:** Finale Hyperparameter-Konfiguration XGBoost
- **Target Section:** Appendix A.4
- **Formatting Note:** Header row, 2 columns (Parameter, Value)
- **Source Note:** Optuna, TPE sampler, MedianPruner, 40 trials on 100k subset
- **Table Content:**

| **Parameter** | **Value** |
| --- | --- |
| max\_depth | 12 |
| learning\_rate | 0.0542 |
| n\_estimators | 358 |
| subsample | 0.851 |
| colsample\_bytree | 0.905 |
| min\_child\_weight | 2 |
| gamma | n/a (not stored) |
| reg\_alpha | 0.0765 |
| reg\_lambda | 1.358 |
| scale\_pos\_weight | n/a (not stored) |

---

### tbl-hyperparams-cnn

- **Purpose:** Finale Hyperparameter-Konfiguration CNN-1D
- **Target Section:** Appendix A.4
- **Formatting Note:** Header row, 2 columns (Parameter, Value)
- **Source Note:** Optuna, 20 trials on 100k subset. Architecture: 3 conv layers (64, 128, 128 filters), kernel size 5, 2 dense layers (128, 64 units), dropout, early stopping (patience: 8), input: 144 features (12 bands/indices x 12 months)
- **Table Content:**

| **Parameter** | **Value** |
| --- | --- |
| learning\_rate | 0.00120 |
| dropout\_rate | 0.419 |
| batch\_size | 64 |
| optimizer | Adam |
| weight\_decay | n/a (not stored) |
