"""Tests for experiment training utilities."""

from __future__ import annotations

from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier

from urban_tree_transfer.experiments.models import CNN1D
from urban_tree_transfer.experiments.training import (
    create_spatial_block_cv,
    create_stratified_subsets,
    finetune_neural_network,
    finetune_xgboost,
    load_model,
    save_model,
    train_final_model,
    train_with_cv,
)


def test_create_spatial_block_cv(sample_dataset) -> None:
    cv = create_spatial_block_cv(sample_dataset, n_splits=3)
    assert cv.n_splits == 3


def test_train_with_cv(sample_dataset) -> None:
    x = sample_dataset[["NDVI_06", "NDVI_07", "EVI_06"]].values
    y = sample_dataset["genus_latin"].astype("category").cat.codes.values
    groups = sample_dataset["block_id"].values
    cv = create_spatial_block_cv(sample_dataset, n_splits=3)
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    results = train_with_cv(model, x, y, groups, cv)
    assert len(results["train_scores"]) == 3
    assert len(results["val_scores"]) == 3
    assert results["train_f1_mean"] >= 0.0


def test_train_final_model(sample_dataset) -> None:
    x = sample_dataset[["NDVI_06", "NDVI_07"]].values
    y = sample_dataset["genus_latin"].astype("category").cat.codes.values
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    trained = train_final_model(model, x, y)
    assert hasattr(trained, "estimators_")


def test_save_load_model_sklearn(tmp_path: Path, sample_dataset) -> None:
    x = sample_dataset[["NDVI_06", "NDVI_07"]].values
    y = sample_dataset["genus_latin"].astype("category").cat.codes.values
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(x, y)

    path = tmp_path / "model.pkl"
    save_model(model, path, metadata={"note": "sklearn"})
    loaded = load_model(path)
    preds = loaded.predict(x)
    assert preds.shape[0] == x.shape[0]
    assert (tmp_path / "model.pkl.metadata.json").exists()


def test_save_load_model_pytorch(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    _ = torch
    model = CNN1D(n_temporal_bases=2, n_months=3, n_static_features=1, n_classes=2)
    path = tmp_path / "model.pt"
    save_model(model, path, metadata={"note": "pytorch"})
    loaded = load_model(
        path,
        model_class=CNN1D,
        model_params=model.get_init_params(),
    )
    assert isinstance(loaded, CNN1D)


def test_create_stratified_subsets(sample_features, sample_labels) -> None:
    """Test creation of stratified subsets at different fractions."""
    fractions = [0.1, 0.25, 0.5, 1.0]
    subsets = create_stratified_subsets(sample_features, sample_labels, fractions)

    assert len(subsets) == len(fractions)
    for frac in fractions:
        assert frac in subsets
        _x_subset, y_subset = subsets[frac]
        expected_size = int(len(sample_labels) * frac)
        assert len(y_subset) == expected_size or frac == 1.0
        if frac == 1.0:
            assert len(y_subset) == len(sample_labels)


def test_finetune_xgboost(sample_features, sample_labels) -> None:
    """Test XGBoost fine-tuning with additional trees."""
    try:
        import xgboost
    except (ImportError, Exception):
        pytest.skip("XGBoost not available or missing system dependencies")

    base_model = xgboost.XGBClassifier(n_estimators=50, random_state=42)
    base_model.fit(sample_features[:100], sample_labels[:100])

    finetuned = finetune_xgboost(
        base_model,
        sample_features[100:],
        sample_labels[100:],
        n_additional_estimators=20,
    )

    assert finetuned.n_estimators == 70
    assert hasattr(finetuned, "classes_")


def test_finetune_neural_network(sample_features, sample_labels) -> None:
    """Test neural network fine-tuning with reduced learning rate."""
    torch = pytest.importorskip("torch")
    _ = torch

    n_classes = len(set(sample_labels))

    model = CNN1D(
        n_temporal_bases=2,
        n_months=5,
        n_static_features=0,
        n_classes=n_classes,
        learning_rate=0.01,
    )

    model.fit(sample_features[:100], sample_labels[:100], epochs=5)

    original_lr = model.learning_rate

    finetuned = finetune_neural_network(
        model,
        sample_features[100:],
        sample_labels[100:],
        epochs=5,
        lr_factor=0.1,
    )

    assert finetuned.learning_rate == original_lr * 0.1
    assert isinstance(finetuned, CNN1D)
