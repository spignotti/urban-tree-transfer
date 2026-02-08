"""Integration tests for Phase 3 pipeline components."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from urban_tree_transfer.experiments.ablation import apply_feature_selection
from urban_tree_transfer.experiments.preprocessing import encode_genus_labels, scale_features
from urban_tree_transfer.experiments.training import create_spatial_block_cv, train_with_cv
from urban_tree_transfer.utils import (
    validate_algorithm_comparison,
    validate_evaluation_metrics,
    validate_finetuning_curve,
    validate_hp_tuning_result,
    validate_setup_decisions,
)


def test_phase3_synthetic_pipeline(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "tree_id": np.arange(30),
            "block_id": np.repeat([0, 1, 2], 10),
            "city": ["berlin"] * 30,
            "genus_latin": np.repeat(["TILIA", "ACER", "QUERCUS"], 10),
            "genus_german": np.repeat(["Linde", "Ahorn", "Eiche"], 10),
            "NDVI_06": np.random.randn(30),
            "NDVI_07": np.random.randn(30),
            "CHM_1m_zscore": np.random.randn(30),
        }
    )

    selected = ["NDVI_06", "NDVI_07", "CHM_1m_zscore"]
    reduced = apply_feature_selection(df, selected, keep_metadata=False)

    x_scaled, _, _, _ = scale_features(reduced)
    y_encoded, label_to_idx, idx_to_label = encode_genus_labels(df["genus_latin"])

    cv = create_spatial_block_cv(df, n_splits=3)
    results = train_with_cv(
        model=RandomForestClassifier(n_estimators=5, random_state=42),
        x=x_scaled,
        y=y_encoded,
        groups=df["block_id"].values,
        cv=cv,
    )
    assert "val_f1_mean" in results
    assert len(label_to_idx) == 3
    assert len(idx_to_label) == 3

    setup_decisions = {
        "chm_strategy": {"decision": "both_engineered"},
        "proximity_strategy": {"decision": "baseline"},
        "outlier_strategy": {"decision": "no_removal"},
        "feature_set": {"n_features": 3},
        "selected_features": selected,
    }
    setup_path = tmp_path / "setup_decisions.json"
    setup_path.write_text(json.dumps(setup_decisions))
    validate_setup_decisions(setup_path)

    algo_comp = {
        "algorithms": [
            {
                "name": "random_forest",
                "type": "ml",
                "val_f1_mean": 0.6,
                "train_f1_mean": 0.7,
                "train_val_gap": 0.1,
            }
        ],
        "ml_champion": {"name": "random_forest", "val_f1_mean": 0.6},
        "nn_champion": {"name": "cnn_1d", "val_f1_mean": 0.55},
    }
    algo_path = tmp_path / "algorithm_comparison.json"
    algo_path.write_text(json.dumps(algo_comp))
    validate_algorithm_comparison(algo_path)

    hp_tuning = {
        "model_name": "random_forest",
        "best_score": 0.62,
        "best_params": {"n_estimators": 200},
        "trials": [{"value": 0.6, "params": {"n_estimators": 200}}],
    }
    hp_path = tmp_path / "hp_tuning.json"
    hp_path.write_text(json.dumps(hp_tuning))
    validate_hp_tuning_result(hp_path)

    eval_metrics = {
        "metrics": {"f1_score": 0.6, "precision": 0.61, "recall": 0.59, "accuracy": 0.6}
    }
    eval_path = tmp_path / "evaluation_metrics.json"
    eval_path.write_text(json.dumps(eval_metrics))
    validate_evaluation_metrics(eval_path)

    finetune_curve = {
        "results": [
            {
                "fraction": 0.1,
                "model_type": "ml",
                "metrics": {"f1_score": 0.4, "precision": 0.4, "recall": 0.4, "accuracy": 0.4},
            }
        ]
    }
    finetune_path = tmp_path / "finetuning_curve.json"
    finetune_path.write_text(json.dumps(finetune_curve))
    validate_finetuning_curve(finetune_path)
