"""Tests for experiment models."""

from __future__ import annotations

import numpy as np
import pytest

from urban_tree_transfer.experiments.models import (
    CNN1D,
    create_majority_classifier,
    create_model,
    create_spatial_only_rf,
    create_stratified_random_classifier,
    train_cnn,
)


def test_create_model_all_types() -> None:
    rf = create_model("random_forest")
    assert rf is not None

    torch = pytest.importorskip("torch")
    _ = torch
    cnn = create_model(
        "cnn_1d",
        {
            "n_temporal_bases": 2,
            "n_months": 3,
            "n_static_features": 1,
            "n_classes": 3,
        },
    )
    assert isinstance(cnn, CNN1D)

    xgb = pytest.importorskip("xgboost")
    model = create_model("xgboost")
    assert model.__class__.__name__ == "XGBClassifier"
    _ = xgb

    tabnet = pytest.importorskip("pytorch_tabnet")
    model = create_model("tabnet")
    assert model.__class__.__name__ == "TabNetClassifier"
    _ = tabnet


def test_create_model_invalid_name() -> None:
    with pytest.raises(ValueError):
        create_model("not_a_model")


def test_majority_classifier() -> None:
    y_train = np.array([0, 0, 1, 0, 2])
    clf = create_majority_classifier(y_train)
    preds = clf(np.zeros((4, 2)))
    assert np.all(preds == 0)


def test_stratified_random_classifier() -> None:
    y_train = np.array([0] * 70 + [1] * 30)
    clf = create_stratified_random_classifier(y_train, random_seed=123)
    preds = clf(np.zeros((1000, 2)))
    proportion = preds.mean()
    assert 0.2 < proportion < 0.4


def test_spatial_only_rf() -> None:
    x_train = np.random.randn(50, 5)
    y_train = np.random.randint(0, 3, 50)
    model = create_spatial_only_rf(x_train, y_train, coord_indices=(1, 2))
    preds = model.predict(x_train[:, [1, 2]])
    assert preds.shape[0] == x_train.shape[0]


def test_cnn_forward_pass() -> None:
    torch = pytest.importorskip("torch")
    model = CNN1D(n_temporal_bases=2, n_months=4, n_static_features=3, n_classes=3)
    x = torch.randn(5, 2 * 4 + 3)
    out = model(x)
    assert out.shape == (5, 3)


def test_train_cnn_early_stopping() -> None:
    torch = pytest.importorskip("torch")
    _ = torch
    model = CNN1D(n_temporal_bases=2, n_months=3, n_static_features=1, n_classes=2)
    x_train = np.random.randn(40, 2 * 3 + 1)
    y_train = np.random.randint(0, 2, 40)
    x_val = np.random.randn(20, 2 * 3 + 1)
    y_val = np.random.randint(0, 2, 20)
    history = train_cnn(
        model,
        x_train,
        y_train,
        x_val=x_val,
        y_val=y_val,
        epochs=5,
        learning_rate=0.0,
        early_stopping_patience=1,
    )
    assert history["stopped_early"] is True
