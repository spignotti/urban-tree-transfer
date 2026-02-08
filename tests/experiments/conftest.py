"""Shared fixtures for experiment tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold


@pytest.fixture
def sample_dataset() -> pd.DataFrame:
    """Create sample dataset for testing."""
    rng = np.random.RandomState(42)
    n_samples = 200

    return pd.DataFrame(
        {
            "tree_id": np.arange(n_samples),
            "block_id": rng.randint(0, 25, n_samples),
            "city": ["berlin"] * n_samples,
            "genus_latin": rng.choice(["TILIA", "ACER", "QUERCUS"], n_samples),
            "genus_german": rng.choice(["Linde", "Ahorn", "Eiche"], n_samples),
            "species_latin": rng.choice(["TILIA CORDATA", "ACER PLATANOIDES"], n_samples),
            "species_german": rng.choice(["Winterlinde", "Spitzahorn"], n_samples),
            "tree_type": rng.choice(["Straßenbaum", "Anlage"], n_samples),
            "plant_year": rng.randint(1950, 2024, n_samples),
            "position_corrected": rng.choice([True, False], n_samples),
            "correction_distance": rng.uniform(0.0, 2.0, n_samples),
            "outlier_zscore": rng.choice([True, False], n_samples),
            "outlier_mahalanobis": rng.choice([True, False], n_samples),
            "outlier_iqr": rng.choice([True, False], n_samples),
            "outlier_severity": rng.choice(["none", "low", "medium", "high"], n_samples),
            "outlier_method_count": rng.randint(0, 4, n_samples),
            "NDVI_06": rng.randn(n_samples),
            "NDVI_07": rng.randn(n_samples),
            "EVI_06": rng.randn(n_samples),
            "CHM_1m": rng.uniform(5, 25, n_samples),
            "CHM_1m_zscore": rng.randn(n_samples),
            "CHM_1m_percentile": rng.uniform(0, 100, n_samples),
        }
    )


@pytest.fixture
def sample_setup_decisions() -> dict[str, object]:
    """Create sample setup_decisions.json structure."""
    return {
        "chm_strategy": {"decision": "both_engineered"},
        "proximity_strategy": {"decision": "baseline"},
        "outlier_strategy": {"decision": "remove_high"},
        "feature_set": {"n_features": 4},
        "selected_features": ["NDVI_06", "NDVI_07", "EVI_06", "CHM_1m_zscore"],
    }


@pytest.fixture
def sample_predictions() -> tuple[np.ndarray, np.ndarray]:
    """Create sample y_true and y_pred for testing."""
    rng = np.random.RandomState(42)
    n_samples = 500
    n_classes = 5

    y_true = rng.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    error_mask = rng.rand(n_samples) < 0.3
    y_pred[error_mask] = rng.randint(0, n_classes, error_mask.sum())

    return y_true, y_pred


@pytest.fixture
def class_labels() -> list[str]:
    """Sample class labels."""
    return ["TILIA", "ACER", "QUERCUS", "PINUS", "PICEA"]


@pytest.fixture
def german_name_map() -> dict[str, str]:
    """Sample German name mapping."""
    return {
        "TILIA": "Linde",
        "ACER": "Ahorn",
        "QUERCUS": "Eiche",
        "PINUS": "Kiefer",
        "PICEA": "Fichte",
    }


@pytest.fixture
def genus_groups() -> dict[str, list[str]]:
    """Sample genus groups."""
    return {
        "conifer": ["PINUS", "PICEA"],
        "deciduous": ["TILIA", "ACER", "QUERCUS"],
    }


@pytest.fixture
def sample_features() -> np.ndarray:
    """Create sample feature array for testing."""
    rng = np.random.RandomState(42)
    n_samples = 150
    n_features = 10
    return rng.randn(n_samples, n_features)


@pytest.fixture
def sample_labels() -> np.ndarray:
    """Create sample label array for testing."""
    rng = np.random.RandomState(42)
    n_samples = 150
    n_classes = 3
    return rng.randint(0, n_classes, n_samples)


@pytest.fixture
def sample_groups() -> np.ndarray:
    """Create sample group array for testing."""
    rng = np.random.RandomState(42)
    n_samples = 150
    return rng.randint(0, 10, n_samples)


@pytest.fixture
def sample_cv() -> GroupKFold:
    """Create sample cross-validator for testing."""
    return GroupKFold(n_splits=3)
