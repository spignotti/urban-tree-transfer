# Report plots

This folder contains the plotting workflow for the report figures.

## Generate plots

From the repository root:

```bash
uv run python paper/plots/generate_plots.py
```

Outputs are written to `paper/figures/`.

## Generated figures

- fig-genus-distribution.png
- fig-jm-monthly.png
- fig-morans-i.png
- fig-chm-cohens-d.png
- fig-correlation.png
- fig-algorithm-comparison.png
- fig-feature-importance.png
- fig-confusion-berlin.png
- fig-confusion-berlin-xgb.png
- fig-confusion-leipzig-zeroshot.png
- fig-per-genus-transfer.png
- fig-finetuning-curve.png
- fig-per-genus-recovery.png

## Confusion matrix convention

- `fig-confusion-berlin`: Berlin confusion matrix in absolute counts (ordered by per-class Berlin F1)
- `fig-confusion-berlin-xgb`: normalized Berlin XGBoost confusion matrix (same class order)
- `fig-confusion-leipzig-zeroshot`: normalized Leipzig zero-shot XGBoost confusion matrix
  (derived from `outputs/phase_3_experiments/metadata/transfer_evaluation.json`)
