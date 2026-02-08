"""Unit tests for Phase 3 evaluation analysis utilities."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from urban_tree_transfer.experiments.evaluation import (
    analyze_by_metadata,
    analyze_conifer_deciduous,
    analyze_spatial_errors,
    analyze_species_breakdown,
    analyze_worst_confused_pairs,
    bin_plant_years,
    compute_confusion_matrix,
)


def test_analyze_worst_confused_pairs(class_labels: list[str]) -> None:
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_pred = np.array([0, 1, 1, 2, 2, 0])

    df = analyze_worst_confused_pairs(y_true, y_pred, class_labels, top_k=2)

    assert df.shape[0] == 2
    assert {"true_genus", "pred_genus", "count", "confusion_rate"} <= set(df.columns)
    assert df["count"].iloc[0] >= df["count"].iloc[1]


def test_compute_confusion_matrix_normalized(
    sample_predictions: tuple[np.ndarray, np.ndarray],
) -> None:
    y_true, y_pred = sample_predictions

    cm = compute_confusion_matrix(y_true, y_pred, normalize=True)

    row_sums = cm.sum(axis=1)
    assert np.all((row_sums == 0) | np.isclose(row_sums, 1.0))


def test_analyze_conifer_deciduous(
    sample_predictions: tuple[np.ndarray, np.ndarray],
    class_labels: list[str],
    genus_groups: dict[str, list[str]],
) -> None:
    y_true, y_pred = sample_predictions

    result = analyze_conifer_deciduous(y_true, y_pred, class_labels, genus_groups)

    assert result["conifer_n"] + result["deciduous_n"] <= len(y_true)
    assert 0.0 <= result["conifer_f1"] <= 1.0
    assert 0.0 <= result["deciduous_f1"] <= 1.0


def test_analyze_by_metadata_plant_year(sample_predictions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = sample_predictions
    plant_years = pd.Series([1955, 1965, 1985, 2005, 2021] * 100)
    bins = [
        {"max_year": 1959},
        {"min_year": 1960, "max_year": 1979},
        {"min_year": 1980, "max_year": 1999},
        {"min_year": 2000, "max_year": 2019},
        {"min_year": 2020},
    ]
    labels = ["vor 1960", "1960-79", "1980-99", "2000-19", "ab 2020"]

    df = analyze_by_metadata(y_true, y_pred, plant_years, bins=bins, bin_labels=labels)

    assert df["bin_label"].tolist() == sorted(labels)
    assert df["n_samples"].sum() == len(plant_years)
    assert df["accuracy"].between(0, 1).all()


def test_analyze_spatial_errors(sample_predictions: tuple[np.ndarray, np.ndarray]) -> None:
    y_true, y_pred = sample_predictions
    block_ids = pd.Series([0, 1, 2, 3, 4] * 100)
    geometry_lookup = gpd.GeoDataFrame(
        {
            "block_id": [0, 1, 2, 3, 4],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
                Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
                Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
                Polygon([(2, 1), (3, 1), (3, 2), (2, 2)]),
            ],
        }
    )
    geometry_lookup = geometry_lookup.set_index("block_id").reset_index()

    gdf = analyze_spatial_errors(
        y_true,
        y_pred,
        block_ids,
        geometry_lookup=geometry_lookup,
    )

    assert len(gdf) == 5
    assert {"accuracy", "n_samples", "n_errors"} <= set(gdf.columns)


def test_analyze_species_breakdown(
    class_labels: list[str], german_name_map: dict[str, str]
) -> None:
    y_true = np.array([0, 0, 0, 1, 1, 2])
    y_pred = np.array([1, 0, 2, 1, 1, 2])
    species = pd.Series(["A", "A", "B", "C", "C", "D"])

    breakdown = analyze_species_breakdown(
        y_true,
        y_pred,
        class_labels,
        species_col=species,
        genus_german_map=german_name_map,
        f1_threshold=1.0,
    )

    assert breakdown
    sample_df = next(iter(breakdown.values()))
    assert {"species", "n_samples", "accuracy", "f1_score", "most_confused_with"} <= set(
        sample_df.columns
    )


def test_bin_plant_years() -> None:
    plant_years = pd.Series([1950, 1970, 1990, 2010, 2022, np.nan])
    bins = [
        {"label": "vor 1960", "max_year": 1959},
        {"label": "1960-79", "min_year": 1960, "max_year": 1979},
        {"label": "1980-99", "min_year": 1980, "max_year": 1999},
        {"label": "2000-19", "min_year": 2000, "max_year": 2019},
        {"label": "ab 2020", "min_year": 2020},
    ]

    binned = bin_plant_years(plant_years, bins)

    assert binned.iloc[0] == "vor 1960"
    assert binned.iloc[1] == "1960-79"
    assert binned.iloc[2] == "1980-99"
    assert binned.iloc[3] == "2000-19"
    assert binned.iloc[4] == "ab 2020"
    assert pd.isna(binned.iloc[5])
