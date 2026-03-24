# Outputs Directory

This folder is the curated local mirror of notebook outputs generated in Google Colab.
It is organized by pipeline phase so downstream dependencies are easy to track.

## Structure

```text
outputs/
├── phase_1_processing/
│   ├── logs/        # Runner execution logs (JSON)
│   └── metadata/    # Validation + task summaries
├── phase_2_features/
│   ├── logs/        # 02a/02b + exp_01..exp_06 execution logs
│   ├── metadata/    # Feature engineering decision artifacts
│   └── report/      # Report-ready JSON exports
├── phase_2_splits/
│   ├── logs/        # 02c execution logs
│   └── metadata/    # Final split summary + split decisions
└── phase_3_experiments/
    ├── logs/        # exp_07..exp_11 + 03a..03d execution logs
    ├── metadata/    # Setup decisions, tuning, transfer, finetuning outputs
    └── report/      # Report-ready JSON exports
```

## What to copy from Drive after a run

After each notebook run in Colab, copy only committed artifacts (mostly JSON) into the matching
phase folder here.

- Phase 1 notebooks -> `outputs/phase_1_processing/`
- Phase 2 feature engineering notebooks (02a/02b + exp_01..exp_06) -> `outputs/phase_2_features/`
- Phase 2 split notebook (02c) -> `outputs/phase_2_splits/`
- Phase 3 exploratory + runner notebooks (exp_07..exp_11, 03a..03d) -> `outputs/phase_3_experiments/`

## Important notes

- Large binaries remain on Drive and stay gitignored (`*.gpkg`, `*.parquet`, model weights).
- `setup_decisions.json`, `transfer_evaluation.json`, and `finetuning_curve.json` are key phase-3
  checkpoints used by downstream steps.
- Keep this directory as an audit trail: no manual editing of generated JSON unless fixing a clearly
  broken export.
