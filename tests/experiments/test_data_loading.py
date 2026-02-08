"""Unit tests for Phase 3 data loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from urban_tree_transfer.experiments.data_loading import (
    fix_missing_genus_german,
    get_feature_columns,
    load_parquet_dataset,
)


def _sentinel_feature_columns() -> list[str]:
    bases = [
        "B2",
        "B3",
        "B4",
        "B5",
        "B8",
        "B8A",
        "B11",
        "B12",
        "NDVI",
        "EVI",
        "VARI",
        "NDre1",
        "NDVIre",
        "CIre",
        "IRECI",
        "RTVIcore",
        "NDWI",
        "MSI",
    ]
    months = list(range(4, 12))
    return [f"{base}_{month:02d}" for base in bases for month in months]


def _make_base_df() -> pd.DataFrame:
    sentinel_cols = _sentinel_feature_columns()
    data = {
        "tree_id": ["T1", "T2"],
        "city": ["berlin", "berlin"],
        "block_id": ["b0", "b1"],
        "genus_latin": ["PRUNUS", "SOPHORA"],
        "plant_year": [2020, pd.NA],
        "species_latin": ["Prunus avium", "Sophora japonica"],
        "genus_german": [pd.NA, pd.NA],
        "species_german": ["Vogelkirsche", "Japanischer Schnurbaum"],
        "tree_type": ["strassenbaeume", "anlagenbaeume"],
        "position_corrected": [False, True],
        "correction_distance": [0.0, 2.5],
        "outlier_zscore": [False, False],
        "outlier_mahalanobis": [False, False],
        "outlier_iqr": [False, False],
        "outlier_severity": ["none", "none"],
        "outlier_method_count": [0, 0],
        "CHM_1m": [10.0, 12.0],
        "CHM_1m_zscore": [0.5, -0.1],
        "CHM_1m_percentile": [75.0, 60.0],
    }
    for col in sentinel_cols:
        data[col] = [1.0, 2.0]
    return pd.DataFrame(data)


def test_fix_missing_genus_german_fills_values() -> None:
    df = pd.DataFrame(
        {
            "genus_latin": ["PRUNUS", "SOPHORA", "ACER"],
            "genus_german": [pd.NA, pd.NA, "Ahorn"],
        }
    )

    result = fix_missing_genus_german(df)

    assert result.loc[0, "genus_german"] == "Prunus"
    assert result.loc[1, "genus_german"] == "Schnurbaum"
    assert result.loc[2, "genus_german"] == "Ahorn"


def test_fix_missing_genus_german_raises_for_unmapped() -> None:
    df = pd.DataFrame(
        {
            "genus_latin": ["ACER"],
            "genus_german": [pd.NA],
        }
    )

    with pytest.raises(ValueError, match="Unmapped genus_german values"):
        fix_missing_genus_german(df)


def test_get_feature_columns_counts() -> None:
    df = _make_base_df()

    features = get_feature_columns(df)

    assert len(features) == 147
    assert "CHM_1m" in features
    assert "CHM_1m_zscore" in features
    assert "CHM_1m_percentile" in features


def test_get_feature_columns_without_chm() -> None:
    df = _make_base_df()

    features = get_feature_columns(df, include_chm=False)

    assert len(features) == 144
    assert all(not feature.startswith("CHM") for feature in features)


def test_get_feature_columns_subset_chm() -> None:
    df = _make_base_df()

    features = get_feature_columns(df, chm_features=["CHM_1m_zscore"])

    assert len(features) == 145
    assert "CHM_1m_zscore" in features
    assert "CHM_1m" not in features


def test_get_feature_columns_invalid_chm() -> None:
    df = _make_base_df()

    with pytest.raises(ValueError, match="Invalid CHM features"):
        get_feature_columns(df, chm_features=["CHM_2m"])


def test_load_parquet_dataset_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_parquet_dataset(tmp_path / "missing.parquet")


def test_load_parquet_dataset_missing_required(tmp_path: Path) -> None:
    df = _make_base_df().drop(columns=["block_id"])
    path = tmp_path / "sample.parquet"
    df.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="Missing required columns"):
        load_parquet_dataset(path)
