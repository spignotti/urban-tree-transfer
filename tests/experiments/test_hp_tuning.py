"""Unit tests for hyperparameter tuning utilities."""

from __future__ import annotations

import pytest

from urban_tree_transfer.experiments.hp_tuning import (
    build_objective,
    create_study,
    run_optuna_search,
    suggest_params_from_space,
)
from urban_tree_transfer.experiments.training import create_spatial_block_cv


class DummyTrial:
    def suggest_float(self, name: str, low: float, high: float) -> float:
        _ = name
        return float((low + high) / 2)

    def suggest_int(self, name: str, low: int, high: int) -> int:
        _ = name
        return int((low + high) // 2)

    def suggest_categorical(self, name: str, choices: list) -> object:
        _ = name
        return choices[0]


def test_suggest_params_from_space() -> None:
    space = {
        "n_estimators": [100, 200],
        "max_depth": [3, 8],
        "criterion": ["gini", "entropy"],
        "nested": {"lr": [0.01, 0.1]},
    }
    params = suggest_params_from_space(DummyTrial(), space)
    assert "n_estimators" in params
    assert "criterion" in params
    assert isinstance(params["nested"], dict)


def test_optuna_helpers(sample_dataset) -> None:
    optuna = pytest.importorskip("optuna")
    _ = optuna

    x = sample_dataset[["NDVI_06", "NDVI_07", "EVI_06"]].values
    y = sample_dataset["genus_latin"].astype("category").cat.codes.values
    groups = sample_dataset["block_id"].values
    cv = create_spatial_block_cv(sample_dataset, n_splits=3)

    space = {"n_estimators": [10, 20], "max_depth": [2, 4]}
    objective = build_objective(
        "random_forest",
        x,
        y,
        groups,
        cv,
        space,
        base_params={"n_jobs": 1},
    )
    study = create_study()
    summary = run_optuna_search(study, objective, n_trials=2)
    assert "best_score" in summary
    assert summary["trials"]
