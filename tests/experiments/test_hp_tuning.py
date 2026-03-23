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

    def set_user_attr(self, key: str, value: object) -> None:
        _ = key, value


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


def test_build_objective_routes_xgboost_learning_rate_to_model_params(sample_dataset, monkeypatch) -> None:
    _ = pytest.importorskip("optuna")

    x = sample_dataset[["NDVI_06", "NDVI_07", "EVI_06"]].values
    y = sample_dataset["genus_latin"].astype("category").cat.codes.values
    groups = sample_dataset["block_id"].values
    cv = create_spatial_block_cv(sample_dataset, n_splits=3)

    captured: dict[str, object] = {}

    def fake_create_model(model_name: str, model_params: dict, random_seed: int) -> object:
        captured["model_name"] = model_name
        captured["model_params"] = dict(model_params)
        captured["random_seed"] = random_seed
        return object()

    def fake_train_with_cv(
        model: object,
        x_data,
        y_data,
        groups_data,
        cv_obj,
        fit_params=None,
        sample_weight=None,
    ) -> dict[str, float]:
        _ = model, x_data, y_data, groups_data, cv_obj, sample_weight
        captured["fit_params"] = dict(fit_params or {})
        return {"val_f1_mean": 0.5, "train_val_gap": 0.1}

    monkeypatch.setattr("urban_tree_transfer.experiments.hp_tuning.create_model", fake_create_model)
    monkeypatch.setattr("urban_tree_transfer.experiments.hp_tuning.train_with_cv", fake_train_with_cv)

    objective = build_objective(
        model_name="xgboost",
        x=x,
        y=y,
        groups=groups,
        cv=cv,
        search_space={"learning_rate": [0.05, 0.2], "max_depth": [4, 8]},
    )

    score = objective(DummyTrial())

    assert score == 0.5
    assert captured["model_name"] == "xgboost"
    assert "learning_rate" in captured["model_params"]
    assert "learning_rate" not in captured["fit_params"]


def test_build_objective_routes_cnn_training_params_to_fit_params(sample_dataset, monkeypatch) -> None:
    _ = pytest.importorskip("optuna")

    x = sample_dataset[["NDVI_06", "NDVI_07", "EVI_06"]].values
    y = sample_dataset["genus_latin"].astype("category").cat.codes.values
    groups = sample_dataset["block_id"].values
    cv = create_spatial_block_cv(sample_dataset, n_splits=3)

    captured: dict[str, object] = {}

    def fake_create_model(model_name: str, model_params: dict, random_seed: int) -> object:
        captured["model_name"] = model_name
        captured["model_params"] = dict(model_params)
        captured["random_seed"] = random_seed
        return object()

    def fake_train_with_cv(
        model: object,
        x_data,
        y_data,
        groups_data,
        cv_obj,
        fit_params=None,
        sample_weight=None,
    ) -> dict[str, float]:
        _ = model, x_data, y_data, groups_data, cv_obj, sample_weight
        captured["fit_params"] = dict(fit_params or {})
        return {"val_f1_mean": 0.5, "train_val_gap": 0.1}

    monkeypatch.setattr("urban_tree_transfer.experiments.hp_tuning.create_model", fake_create_model)
    monkeypatch.setattr("urban_tree_transfer.experiments.hp_tuning.train_with_cv", fake_train_with_cv)

    objective = build_objective(
        model_name="cnn_1d",
        x=x,
        y=y,
        groups=groups,
        cv=cv,
        search_space={
            "learning_rate": [0.0005, 0.003],
            "batch_size": [32, 64],
            "epochs": [50, 80],
            "early_stopping_patience": [8, 15],
            "dropout": [0.2, 0.5],
        },
        base_params={"n_temporal_bases": 2, "n_months": 3, "n_static_features": 1, "n_classes": 4},
    )

    score = objective(DummyTrial())

    assert score == 0.5
    assert captured["model_name"] == "cnn_1d"
    assert "dropout" in captured["model_params"]
    assert "learning_rate" not in captured["model_params"]
    assert "learning_rate" in captured["fit_params"]
    assert "batch_size" in captured["fit_params"]
    assert "epochs" in captured["fit_params"]
    assert "early_stopping_patience" in captured["fit_params"]
