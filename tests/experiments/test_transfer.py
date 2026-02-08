"""Unit tests for transfer evaluation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from urban_tree_transfer.experiments.transfer import (
    classify_transfer_robustness,
    compute_feature_stability,
    compute_transfer_gap,
    compute_transfer_metrics,
    mcnemar_test,
)


def test_compute_transfer_gap() -> None:
    gap = compute_transfer_gap(0.7, 0.56)
    assert gap.absolute_drop == pytest.approx(0.14, abs=1e-10)
    assert gap.relative_drop > 0.0


def test_classify_transfer_robustness() -> None:
    source = {"TILIA": 0.8, "ACER": 0.6}
    target = {"TILIA": 0.78, "ACER": 0.45}
    result = classify_transfer_robustness(
        source, target, robust_threshold=0.05, medium_threshold=0.2
    )
    assert result["TILIA"]["label"] == "robust"
    assert result["ACER"]["label"] in {"medium", "poor"}


def test_compute_transfer_metrics(class_labels: list[str]) -> None:
    y_true = np.array([0, 1, 2, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 2])
    result = compute_transfer_metrics(y_true, y_pred, class_labels)
    assert "metrics" in result
    assert "f1_score" in result["metrics"]


def test_compute_feature_stability() -> None:
    source = pd.DataFrame({"feature": ["A", "B", "C"], "importance": [0.3, 0.2, 0.1]})
    target = pd.DataFrame({"feature": ["A", "B", "C"], "importance": [0.31, 0.19, 0.11]})
    result = compute_feature_stability(source, target)
    assert "spearman_rho" in result
    assert result["n_features"] == 3


def test_mcnemar_test() -> None:
    y_true = np.array([0, 1, 0, 1])
    y_pred_a = np.array([0, 1, 1, 1])
    y_pred_b = np.array([0, 0, 0, 1])
    result = mcnemar_test(y_true, y_pred_a, y_pred_b)
    assert "p_value" in result
    assert 0.0 <= result["p_value"] <= 1.0
