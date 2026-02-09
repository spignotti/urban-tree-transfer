"""Tests for experiment ablation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from urban_tree_transfer.experiments.ablation import (
    apply_chm_strategy,
    apply_feature_selection,
    apply_outlier_removal,
    apply_proximity_filter,
    compute_feature_importance,
    create_feature_subsets,
    evaluate_feature_subsets,
    get_chm_features,
    get_dataset_suffix,
    get_metadata_columns,
    get_outlier_mask,
    prepare_ablation_dataset,
    select_optimal_features,
)


def test_apply_chm_strategy_all_variants(sample_dataset) -> None:
    variants = {
        "no_chm": [],
        "zscore_only": ["CHM_1m_zscore"],
        "percentile_only": ["CHM_1m_percentile"],
        "both_engineered": ["CHM_1m_zscore", "CHM_1m_percentile"],
        "raw_chm": ["CHM_1m"],
    }
    for strategy, expected in variants.items():
        df = apply_chm_strategy(sample_dataset, strategy)
        for feature in ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"]:
            if feature in expected:
                assert feature in df.columns
            else:
                assert feature not in df.columns
        assert get_chm_features(strategy) == expected


def test_apply_proximity_filter_both_variants(tmp_path: Path, sample_dataset) -> None:
    base = tmp_path
    baseline_path = base / "berlin_train.parquet"
    filtered_path = base / "berlin_train_filtered.parquet"
    sample_dataset.to_parquet(baseline_path)
    sample_dataset.to_parquet(filtered_path)

    df_base = apply_proximity_filter(base, "berlin", "train", "baseline")
    df_filtered = apply_proximity_filter(base, "berlin", "train", "filtered")

    assert len(df_base) == len(sample_dataset)
    assert len(df_filtered) == len(sample_dataset)
    assert get_dataset_suffix("baseline") == ""
    assert get_dataset_suffix("filtered") == "_filtered"


def test_apply_outlier_removal_all_strategies(sample_dataset) -> None:
    no_removal = apply_outlier_removal(sample_dataset, "no_removal")
    assert len(no_removal) == len(sample_dataset)

    remove_high = apply_outlier_removal(sample_dataset, "remove_high")
    assert "high" not in remove_high["outlier_severity"].values

    remove_high_medium = apply_outlier_removal(sample_dataset, "remove_high_medium")
    assert not np.isin(remove_high_medium["outlier_severity"], ["high", "medium"]).any()

    mask = get_outlier_mask(sample_dataset, "remove_high")
    assert mask.dtype == bool


def test_apply_feature_selection(sample_dataset) -> None:
    selected = ["NDVI_06", "NDVI_07", "CHM_1m"]
    df = apply_feature_selection(sample_dataset, selected)
    assert all(feature in df.columns for feature in selected)
    metadata = get_metadata_columns(sample_dataset)
    for col in metadata:
        assert col in df.columns


def test_prepare_ablation_dataset_full_pipeline(
    tmp_path: Path, sample_dataset, sample_setup_decisions
) -> None:
    base = tmp_path
    path = base / "berlin_train.parquet"
    sample_dataset.to_parquet(path)

    df, meta = prepare_ablation_dataset(base, "berlin", "train", sample_setup_decisions)
    assert meta["original_n_samples"] == len(sample_dataset)
    assert meta["filtered_n_samples"] == len(df)
    assert meta["outliers_removed"] >= 0
    assert meta["n_features"] == len(sample_setup_decisions["selected_features"])


def test_compute_feature_importance(sample_features, sample_labels) -> None:
    """Test feature importance computation."""
    x_train = sample_features
    y_train = sample_labels
    feature_names = [f"feature_{i}" for i in range(x_train.shape[1])]

    importance_df = compute_feature_importance(x_train, y_train, feature_names)

    assert len(importance_df) == len(feature_names)
    assert list(importance_df.columns) == ["feature", "importance", "rank"]
    assert importance_df["importance"].sum() > 0
    assert list(importance_df["rank"]) == list(range(1, len(feature_names) + 1))
    assert importance_df["importance"].is_monotonic_decreasing


def test_create_feature_subsets() -> None:
    """Test feature subset creation from importance ranking."""
    importance_df = np.array(
        [
            ["feature_a", 0.5, 1],
            ["feature_b", 0.3, 2],
            ["feature_c", 0.15, 3],
            ["feature_d", 0.05, 4],
        ]
    )
    importance_df = np.core.records.fromarrays(
        [importance_df[:, 0], importance_df[:, 1].astype(float), importance_df[:, 2].astype(int)],
        names="feature,importance,rank",
    )
    import pandas as pd

    importance_df = pd.DataFrame(importance_df)

    n_features_list = [2, 3]
    subsets = create_feature_subsets(importance_df, n_features_list)

    assert "top_2" in subsets
    assert "top_3" in subsets
    assert "all_features" in subsets
    assert subsets["top_2"] == ["feature_a", "feature_b"]
    assert subsets["top_3"] == ["feature_a", "feature_b", "feature_c"]
    assert len(subsets["all_features"]) == 4


def test_evaluate_feature_subsets(sample_features, sample_labels, sample_groups, sample_cv) -> None:
    """Test feature subset evaluation with cross-validation."""
    feature_names = [f"feature_{i}" for i in range(sample_features.shape[1])]
    feature_subsets = {
        "top_5": feature_names[:5],
        "all_features": feature_names,
    }

    results_df = evaluate_feature_subsets(
        sample_features,
        sample_labels,
        sample_groups,
        feature_names,
        feature_subsets,
        sample_cv,
    )

    assert len(results_df) == 2
    assert list(results_df.columns) == ["variant", "n_features", "val_f1_mean", "val_f1_std"]
    assert results_df["val_f1_mean"].min() > 0.0
    assert results_df["val_f1_mean"].max() <= 1.0


def test_select_optimal_features() -> None:
    """Test optimal feature set selection based on decision rules."""
    import pandas as pd

    results_df = pd.DataFrame(
        [
            {"variant": "top_10", "n_features": 10, "val_f1_mean": 0.75, "val_f1_std": 0.02},
            {"variant": "top_20", "n_features": 20, "val_f1_mean": 0.78, "val_f1_std": 0.015},
            {"variant": "top_30", "n_features": 30, "val_f1_mean": 0.79, "val_f1_std": 0.01},
            {"variant": "all_features", "n_features": 50, "val_f1_mean": 0.80, "val_f1_std": 0.01},
        ]
    )

    selected_variant, n_features = select_optimal_features(results_df, max_drop=0.02)

    assert selected_variant == "top_20"
    assert n_features == 20


def test_optimize_dtypes() -> None:
    """Test dtype optimization for memory efficiency."""
    import pandas as pd

    from urban_tree_transfer.experiments.ablation import optimize_dtypes

    # Create DataFrame with non-optimal dtypes
    df = pd.DataFrame(
        {
            "feature_1": np.array([1.0, 2.0, 3.0], dtype="float64"),
            "feature_2": np.array([0.5, 1.5, 2.5], dtype="float64"),
            "id": np.array([1, 2, 3], dtype="int64"),
            "small_int": np.array([10, 20, 30], dtype="int64"),
        }
    )

    optimized = optimize_dtypes(df)

    # Float64 should become float32
    assert optimized["feature_1"].dtype == np.float32
    assert optimized["feature_2"].dtype == np.float32

    # Large int64 should become smaller type
    assert optimized["small_int"].dtype in [np.int8, np.int16, np.int32, np.uint8, np.uint16]

    # Values should be preserved
    assert np.allclose(optimized["feature_1"].values, df["feature_1"].values, rtol=1e-5)


def test_prepare_ablation_dataset_test_split_policy(tmp_path: Path, sample_dataset) -> None:
    """Test that test splits always use baseline proximity regardless of setup."""
    # Create mock parquet files
    baseline_path = tmp_path / "berlin_test.parquet"
    filtered_path = tmp_path / "berlin_test_filtered.parquet"
    sample_dataset.to_parquet(baseline_path)
    sample_dataset.to_parquet(filtered_path)

    # Setup decisions with filtered proximity
    setup_decisions = {
        "proximity_strategy": {"decision": "filtered"},
        "outlier_strategy": {"decision": "no_removal"},
        "chm_strategy": {"decision": "both_engineered"},
        "selected_features": ["NDVI_06", "CHM_1m_zscore"],
    }

    # Process test split (should override to baseline)
    _, meta = prepare_ablation_dataset(
        base_path=tmp_path,
        city="berlin",
        split="test",
        setup_decisions=setup_decisions,
        return_metadata=True,
    )

    # Verify test split policy was applied
    assert meta["test_split_policy_applied"] is True
    assert meta["proximity_strategy_requested"] == "filtered"
    assert meta["proximity_strategy_used"] == "baseline"

    # Process non-test split (should use requested strategy)
    baseline_train = tmp_path / "berlin_train.parquet"
    filtered_train = tmp_path / "berlin_train_filtered.parquet"
    sample_dataset.to_parquet(baseline_train)
    sample_dataset.to_parquet(filtered_train)

    _, meta2 = prepare_ablation_dataset(
        base_path=tmp_path,
        city="berlin",
        split="train",
        setup_decisions=setup_decisions,
        return_metadata=True,
    )

    # Verify non-test split uses requested strategy
    assert meta2["test_split_policy_applied"] is False
    assert meta2["proximity_strategy_requested"] == "filtered"
    assert meta2["proximity_strategy_used"] == "filtered"


def test_prepare_ablation_dataset_memory_optimization(tmp_path: Path, sample_dataset) -> None:
    """Test memory optimization in prepare_ablation_dataset."""
    baseline_path = tmp_path / "berlin_train.parquet"
    sample_dataset.to_parquet(baseline_path)

    setup_decisions = {
        "proximity_strategy": {"decision": "baseline"},
        "outlier_strategy": {"decision": "no_removal"},
        "chm_strategy": {"decision": "both_engineered"},
        "selected_features": ["NDVI_06", "CHM_1m_zscore"],
    }

    # Test with optimization enabled
    df_optimized, _ = prepare_ablation_dataset(
        base_path=tmp_path,
        city="berlin",
        split="train",
        setup_decisions=setup_decisions,
        optimize_memory=True,
    )

    # Test with optimization disabled
    df_not_optimized, _ = prepare_ablation_dataset(
        base_path=tmp_path,
        city="berlin",
        split="train",
        setup_decisions=setup_decisions,
        optimize_memory=False,
    )

    # Optimized should have float32, not-optimized should have original dtypes
    float_cols_opt = df_optimized.select_dtypes(include=["float32"]).columns
    float_cols_not = df_not_optimized.select_dtypes(include=["float64"]).columns

    assert len(float_cols_opt) > 0, "Optimized should have float32 columns"
    assert len(float_cols_not) > 0, "Non-optimized should preserve float64 columns"
