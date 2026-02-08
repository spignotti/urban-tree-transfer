"""Unit tests for Phase 3 visualization utilities."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from urban_tree_transfer.experiments.visualization import (
    map_to_german_names,
    plot_ablation_comparison,
    plot_algorithm_comparison,
    plot_confusion_comparison,
    plot_confusion_matrix,
    plot_feature_group_contribution,
    plot_finetuning_curve,
    plot_pareto_curve,
    plot_per_genus_f1,
    plot_per_genus_transfer,
    plot_significance_matrix,
    plot_spatial_error_map,
    save_figure,
)

matplotlib.use("Agg")


def test_map_to_german_names() -> None:
    labels = ["TILIA", "ACER"]
    mapping = {"TILIA": "Linde", "ACER": "Ahorn"}

    mapped = map_to_german_names(labels, mapping)

    assert mapped == ["Linde", "Ahorn"]


def test_save_figure(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, _ = plt.subplots()
    output_path = tmp_path / "figures" / "plot.png"

    save_figure(fig, output_path)

    assert output_path.exists()


def test_plot_algorithm_comparison(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "algorithm": ["RF", "XGB", "CNN"],
            "val_f1_mean": [0.62, 0.58, 0.55],
            "val_f1_std": [0.03, 0.02, 0.04],
            "type": ["ml", "ml", "nn"],
        }
    )
    output_path = tmp_path / "algorithm_comparison.png"

    plot_algorithm_comparison(df, output_path, highlight_champions=["RF"])

    assert output_path.exists()


def test_plot_confusion_matrix(tmp_path: Path) -> None:
    cm = np.array([[8, 2], [1, 9]])
    labels = ["TILIA", "ACER"]
    output_path = tmp_path / "confusion_matrix.png"

    plot_confusion_matrix(cm, labels, output_path)

    assert output_path.exists()


def test_plot_per_genus_f1(tmp_path: Path) -> None:
    f1_scores = {"TILIA": 0.6, "ACER": 0.7, "QUERCUS": 0.4}
    ci_data = {"TILIA": (0.5, 0.7), "ACER": (0.6, 0.8), "QUERCUS": (0.3, 0.5)}
    output_path = tmp_path / "per_genus_f1.png"

    plot_per_genus_f1(f1_scores, output_path, ci_data=ci_data)

    assert output_path.exists()


def test_plot_spatial_error_map(tmp_path: Path) -> None:
    gdf = gpd.GeoDataFrame(
        {
            "block_id": [0, 1],
            "accuracy": [0.8, 0.6],
            "geometry": [
                Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
                Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
            ],
        },
        geometry="geometry",
    )
    output_path = tmp_path / "spatial_error_map.png"

    plot_spatial_error_map(gdf, output_path)

    assert output_path.exists()


def test_plot_ablation_comparison(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "variant": ["no_chm", "zscore_only"],
            "val_f1_mean": [0.55, 0.58],
            "val_f1_std": [0.02, 0.03],
        }
    )
    output_path = tmp_path / "ablation.png"
    plot_ablation_comparison(df, "CHM Ablation", output_path, highlight_decision="zscore_only")
    assert output_path.exists()


def test_plot_pareto_curve(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "n_features": [30, 50, 80],
            "val_f1_mean": [0.52, 0.55, 0.56],
            "val_f1_std": [0.02, 0.02, 0.03],
        }
    )
    output_path = tmp_path / "pareto.png"
    plot_pareto_curve(df, selected_k=50, output_path=output_path)
    assert output_path.exists()


def test_plot_feature_group_contribution(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature": ["A", "B", "C"],
            "importance": [0.4, 0.3, 0.2],
        }
    )
    output_path = tmp_path / "groups.png"
    plot_feature_group_contribution(df, {"A": "S2", "B": "CHM"}, output_path)
    assert output_path.exists()


def test_plot_confusion_comparison(tmp_path: Path) -> None:
    cm_a = np.array([[5, 1], [2, 6]])
    cm_b = np.array([[4, 2], [1, 7]])
    output_path = tmp_path / "confusion_compare.png"
    plot_confusion_comparison(cm_a, cm_b, ["TILIA", "ACER"], "Berlin", "Leipzig", output_path)
    assert output_path.exists()


def test_plot_per_genus_transfer(tmp_path: Path) -> None:
    transfer = {
        "TILIA": {"relative_drop": 0.03, "label": "robust"},
        "ACER": {"relative_drop": 0.2, "label": "poor"},
    }
    output_path = tmp_path / "transfer.png"
    plot_per_genus_transfer(transfer, ["TILIA", "ACER"], output_path)
    assert output_path.exists()


def test_plot_finetuning_curve(tmp_path: Path) -> None:
    results = [
        {"fraction": 0.1, "f1_score": 0.4},
        {"fraction": 0.5, "f1_score": 0.55},
        {"fraction": 1.0, "f1_score": 0.6},
    ]
    baselines = {"zero_shot": 0.35}
    output_path = tmp_path / "finetune.png"
    plot_finetuning_curve(results, baselines, output_path)
    assert output_path.exists()


def test_plot_significance_matrix(tmp_path: Path) -> None:
    df = pd.DataFrame([[0.01, 0.2], [0.03, 0.5]], index=["A", "B"], columns=["A", "B"])
    output_path = tmp_path / "significance.png"
    plot_significance_matrix(df, output_path)
    assert output_path.exists()


def test_plot_per_genus_comparison(tmp_path: Path) -> None:
    from urban_tree_transfer.experiments.visualization import plot_per_genus_comparison

    results_a = {"TILIA": 0.7, "ACER": 0.6}
    results_b = {"TILIA": 0.8, "ACER": 0.5}
    genus_labels = ["TILIA", "ACER"]
    output_path = tmp_path / "per_genus_comparison.png"
    plot_per_genus_comparison(results_a, results_b, "Berlin", "Leipzig", genus_labels, output_path)
    assert output_path.exists()


def test_plot_sample_loss_by_genus(tmp_path: Path) -> None:
    from urban_tree_transfer.experiments.visualization import plot_sample_loss_by_genus

    baseline_counts = pd.Series([100, 80], index=["TILIA", "ACER"])
    filtered_counts = pd.Series([90, 70], index=["TILIA", "ACER"])
    genus_labels = ["TILIA", "ACER"]
    output_path = tmp_path / "sample_loss_by_genus.png"
    plot_sample_loss_by_genus(baseline_counts, filtered_counts, genus_labels, output_path)
    assert output_path.exists()


def test_plot_outlier_distribution(tmp_path: Path) -> None:
    from urban_tree_transfer.experiments.visualization import plot_outlier_distribution

    severity_by_genus = pd.DataFrame(
        {"TILIA": [10, 5, 2], "ACER": [8, 6, 3]}, index=["low", "medium", "high"]
    )
    genus_labels = ["TILIA", "ACER"]
    output_path = tmp_path / "outlier_distribution.png"
    plot_outlier_distribution(severity_by_genus, genus_labels, output_path)
    assert output_path.exists()


def test_plot_performance_ladder(tmp_path: Path) -> None:
    from urban_tree_transfer.experiments.visualization import plot_performance_ladder

    results = pd.DataFrame(
        {
            "name": ["RF", "XGB"],
            "f1_score": [0.6, 0.55],
            "category": ["model", "model"],
        }
    )
    output_path = tmp_path / "performance_ladder.png"
    plot_performance_ladder(results, output_path, highlight_champions=["RF"])
    assert output_path.exists()


def test_plot_transfer_conifer_deciduous(tmp_path: Path) -> None:
    from urban_tree_transfer.experiments.visualization import plot_transfer_conifer_deciduous

    transfer_data = {
        "PINUS": {"berlin_f1": 0.7, "leipzig_f1": 0.6},
        "TILIA": {"berlin_f1": 0.8, "leipzig_f1": 0.75},
    }
    conifer_genera = {"PINUS"}
    output_path = tmp_path / "transfer_conifer_deciduous.png"
    plot_transfer_conifer_deciduous(transfer_data, conifer_genera, output_path)
    assert output_path.exists()


def test_plot_finetuning_ml_vs_nn(tmp_path: Path) -> None:
    from urban_tree_transfer.experiments.visualization import plot_finetuning_ml_vs_nn

    ml_results = [{"fraction": 0.1, "f1_score": 0.4}, {"fraction": 1.0, "f1_score": 0.6}]
    nn_results = [{"fraction": 0.1, "f1_score": 0.35}, {"fraction": 1.0, "f1_score": 0.55}]
    output_path = tmp_path / "ml_vs_nn.png"
    plot_finetuning_ml_vs_nn(ml_results, nn_results, output_path)
    assert output_path.exists()


def test_plot_feature_importance(tmp_path: Path) -> None:
    from urban_tree_transfer.experiments.visualization import plot_feature_importance

    importance_df = pd.DataFrame({"feature": ["A", "B", "C"], "importance": [0.5, 0.3, 0.2]})
    output_path = tmp_path / "feature_importance.png"
    plot_feature_importance(importance_df, output_path, top_k=3)
    assert output_path.exists()


def test_plot_optuna_history(tmp_path: Path) -> None:
    import pytest

    from urban_tree_transfer.experiments.visualization import plot_optuna_history

    try:
        import optuna
    except ImportError:
        pytest.skip("optuna not installed")

    # Create a minimal study
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1) * 0.5, n_trials=3)

    output_path = tmp_path / "optuna_history.png"
    plot_optuna_history(study, output_path)
    assert output_path.exists()
