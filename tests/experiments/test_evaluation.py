"""Unit tests for Phase 3 evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from urban_tree_transfer.experiments.evaluation import (
    bootstrap_confidence_interval,
    compute_metrics,
    compute_per_class_metrics,
)


def test_compute_metrics_matches_sklearn() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])

    metrics = compute_metrics(y_true, y_pred, average="weighted")

    assert metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert metrics["f1_score"] == f1_score(y_true, y_pred, average="weighted", zero_division=0)
    assert metrics["precision"] == precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    assert metrics["recall"] == recall_score(y_true, y_pred, average="weighted", zero_division=0)


def test_compute_per_class_metrics() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    class_names = ["ACER", "TILIA"]

    df = compute_per_class_metrics(y_true, y_pred, class_names)

    assert isinstance(df, pd.DataFrame)
    assert list(df["genus"]) == class_names
    acer = df.iloc[0]
    tili = df.iloc[1]
    assert acer["support"] == 2
    assert tili["support"] == 2
    assert np.isclose(acer["precision"], 1.0)
    assert np.isclose(acer["recall"], 0.5)
    assert np.isclose(tili["precision"], 2 / 3)
    assert np.isclose(tili["recall"], 1.0)


def test_bootstrap_confidence_interval_bounds() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 1])

    def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    point, lower, upper = bootstrap_confidence_interval(
        y_true, y_pred, metric_fn, n_bootstrap=200, confidence_level=0.9, random_seed=7
    )

    assert 0.0 <= lower <= upper <= 1.0
    assert lower <= point <= upper


def test_bootstrap_confidence_interval_invalid_inputs() -> None:
    y_true = np.array([])
    y_pred = np.array([])

    def metric_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return accuracy_score(y_true, y_pred)

    with pytest.raises(ValueError, match="y_true is empty"):
        bootstrap_confidence_interval(y_true, y_pred, metric_fn)
