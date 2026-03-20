"""Ablation utilities for applying setup decisions from PRD 003a."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import BaseCrossValidator

from urban_tree_transfer.config import RANDOM_SEED
from urban_tree_transfer.experiments.data_loading import load_parquet_dataset


def apply_chm_strategy(
    df: pd.DataFrame,
    chm_strategy: Literal[
        "no_chm",
        "zscore_only",
        "percentile_only",
        "both_engineered",
        "raw_chm",
    ],
) -> pd.DataFrame:
    """Apply CHM feature selection based on setup decision."""
    chm_features = ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"]
    selected = get_chm_features(chm_strategy)
    to_drop = [col for col in chm_features if col not in selected]
    result = df.copy()
    return result.drop(columns=[col for col in to_drop if col in result.columns])


def get_chm_features(
    chm_strategy: Literal[
        "no_chm",
        "zscore_only",
        "percentile_only",
        "both_engineered",
        "raw_chm",
    ],
) -> list[str]:
    """Get list of CHM features to include based on strategy."""
    mapping = {
        "no_chm": [],
        "zscore_only": ["CHM_1m_zscore"],
        "percentile_only": ["CHM_1m_percentile"],
        "both_engineered": ["CHM_1m_zscore", "CHM_1m_percentile"],
        "raw_chm": ["CHM_1m"],
    }
    if chm_strategy not in mapping:
        raise ValueError(f"Invalid chm_strategy: {chm_strategy}")
    return mapping[chm_strategy]


def apply_proximity_filter(
    base_path: Path,
    city: str,
    split: str,
    proximity_strategy: Literal["baseline", "filtered"],
) -> pd.DataFrame:
    """Load dataset variant based on proximity strategy."""
    suffix = get_dataset_suffix(proximity_strategy)
    file_path = base_path / f"{city}_{split}{suffix}.parquet"
    return load_parquet_dataset(file_path)


def get_dataset_suffix(proximity_strategy: Literal["baseline", "filtered"]) -> str:
    """Get filename suffix for proximity strategy."""
    if proximity_strategy == "baseline":
        return ""
    if proximity_strategy == "filtered":
        return "_filtered"
    raise ValueError(f"Invalid proximity_strategy: {proximity_strategy}")


def apply_outlier_removal(
    df: pd.DataFrame,
    outlier_strategy: Literal["no_removal", "remove_high", "remove_high_medium"],
    outlier_col: str = "outlier_severity",
) -> pd.DataFrame:
    """Remove outliers based on severity strategy."""
    if outlier_col not in df.columns:
        raise ValueError(f"Missing outlier column: {outlier_col}")

    mask = get_outlier_mask(df, outlier_strategy, outlier_col=outlier_col)
    return df.loc[mask].copy()


def get_outlier_mask(
    df: pd.DataFrame,
    outlier_strategy: Literal["no_removal", "remove_high", "remove_high_medium"],
    outlier_col: str = "outlier_severity",
) -> np.ndarray:
    """Get boolean mask for rows to keep based on outlier strategy."""
    if outlier_col not in df.columns:
        raise ValueError(f"Missing outlier column: {outlier_col}")

    if outlier_strategy == "no_removal":
        return np.ones(len(df), dtype=bool)
    if outlier_strategy == "remove_high":
        return (df[outlier_col] != "high").to_numpy()
    if outlier_strategy == "remove_high_medium":
        return (~df[outlier_col].isin(["high", "medium"])).to_numpy()

    raise ValueError(f"Invalid outlier_strategy: {outlier_strategy}")


def apply_feature_selection(
    df: pd.DataFrame,
    selected_features: list[str],
    keep_metadata: bool = True,
    metadata_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Apply feature selection from setup_decisions.json."""
    missing = sorted(feature for feature in selected_features if feature not in df.columns)
    if missing:
        raise ValueError(f"Selected features missing from DataFrame: {missing}")

    if metadata_cols is None and keep_metadata:
        metadata_cols = get_metadata_columns(df)
    if not keep_metadata:
        metadata_cols = []

    columns = list(dict.fromkeys(selected_features + (metadata_cols or [])))
    return df.loc[:, columns].copy()


def get_metadata_columns(df: pd.DataFrame) -> list[str]:
    """Auto-detect metadata columns in DataFrame."""
    fixed_names = {
        "city",
        "genus_latin",
        "genus_german",
        "species_latin",
        "species_german",
        "tree_type",
        "plant_year",
        "correction_distance",
        "position_corrected",
        "is_conifer",
    }

    metadata = []
    for col in df.columns:
        if col.endswith("_id"):
            metadata.append(col)
            continue
        if col in fixed_names:
            metadata.append(col)
            continue
        if col.startswith("outlier_"):
            metadata.append(col)

    return sorted(set(metadata))


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by converting dtypes.

    Converts float64 → float32 and optimizes integer types where possible.
    Reduces memory usage by ~50% for typical feature matrices.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with optimized dtypes
    """
    result = df.copy()

    # Float: 64 → 32
    float_cols = result.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        result[float_cols] = result[float_cols].astype("float32")

    # Int: 64 → smallest fitting type
    int_cols = result.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        col_min, col_max = result[col].min(), result[col].max()
        if col_min >= 0:  # Unsigned
            if col_max < 255:
                result[col] = result[col].astype("uint8")
            elif col_max < 65535:
                result[col] = result[col].astype("uint16")
            else:
                result[col] = result[col].astype("uint32")
        else:  # Signed
            if col_min > -128 and col_max < 127:
                result[col] = result[col].astype("int8")
            elif col_min > -32768 and col_max < 32767:
                result[col] = result[col].astype("int16")
            else:
                result[col] = result[col].astype("int32")

    return result


def prepare_ablation_dataset(
    base_path: Path,
    city: str,
    split: str,
    setup_decisions: dict[str, Any],
    return_metadata: bool = True,
    optimize_memory: bool = True,
    skip_feature_selection: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply all setup decisions to prepare dataset for experiments.

    IMPORTANT: Test splits always use baseline proximity filtering (no filtering)
    to represent real-world distribution for unbiased evaluation. This policy is
    enforced automatically when split name is "test".

    Args:
        base_path: Path to phase_2_splits directory
        city: City name ("berlin" or "leipzig")
        split: Split name ("train", "val", "test", "finetune")
        setup_decisions: Dictionary from setup_decisions.json with keys:
            - proximity_strategy: {"decision": "baseline" or "filtered"}
            - outlier_strategy: {"decision": "no_removal", "remove_high", etc.}
            - chm_strategy: {"decision": "no_chm", "zscore_only", etc.}
            - selected_features: List of feature names
        return_metadata: Whether to keep metadata columns (default: True)
        optimize_memory: Whether to optimize dtypes for memory (default: True)
        skip_feature_selection: If True, preserve all Sentinel-2 features instead of
            applying selected_features. CHM/proximity/outlier strategies still apply.
            Use for CNN1D datasets that require complete temporal sequences.

    Returns:
        Tuple of (filtered_dataframe, metadata_dict)

    Raises:
        ValueError: If setup_decisions missing required keys
        FileNotFoundError: If input parquet file not found

    Examples:
        >>> from pathlib import Path
        >>> setup = {
        ...     "proximity_strategy": {"decision": "filtered"},
        ...     "outlier_strategy": {"decision": "remove_high"},
        ...     "chm_strategy": {"decision": "both_engineered"},
        ...     "selected_features": ["NDVI_06", "NDVI_07", "CHM_1m_zscore"],
        ... }
        >>> # XGBoost dataset (reduced features)
        >>> df_xgb, meta_xgb = prepare_ablation_dataset(
        ...     Path("data/phase_2_splits"),
        ...     "berlin",
        ...     "train",
        ...     setup,
        ... )
        >>> print(f"XGBoost: {len(df_xgb)} samples, {meta_xgb['n_features']} features")
        >>> # CNN1D dataset (full features)
        >>> df_cnn, meta_cnn = prepare_ablation_dataset(
        ...     Path("data/phase_2_splits"),
        ...     "berlin",
        ...     "train",
        ...     setup,
        ...     skip_feature_selection=True,
        ... )
        >>> print(f"CNN1D: {len(df_cnn)} samples, {meta_cnn['n_features']} features")
    """
    try:
        proximity_strategy = setup_decisions["proximity_strategy"]["decision"]
        outlier_strategy = setup_decisions["outlier_strategy"]["decision"]
        chm_strategy = setup_decisions["chm_strategy"]["decision"]
        selected_features = setup_decisions["selected_features"]
    except KeyError as exc:
        raise ValueError(f"setup_decisions missing required key: {exc}") from exc

    # TEST SPLIT POLICY: Always use baseline proximity for unbiased evaluation
    is_test_split = split == "test"
    if is_test_split and proximity_strategy != "baseline":
        original_strategy = proximity_strategy
        proximity_strategy = "baseline"
        test_policy_applied = True
    else:
        original_strategy = proximity_strategy
        test_policy_applied = False

    df = apply_proximity_filter(base_path, city, split, proximity_strategy)
    original_n_samples = len(df)

    df = apply_outlier_removal(df, outlier_strategy)
    outliers_removed = original_n_samples - len(df)

    df = apply_chm_strategy(df, chm_strategy)
    chm_features_included = get_chm_features(chm_strategy)

    # Feature selection: skip for CNN1D (needs full temporal sequences)
    if skip_feature_selection:
        # Keep all feature columns (Sentinel-2 + CHM if included)
        metadata_cols = get_metadata_columns(df) if return_metadata else []
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        columns = list(dict.fromkeys(feature_cols + metadata_cols))
        df = df.loc[:, columns].copy()
        n_features = len(feature_cols)
        feature_set_type = "full"
    else:
        df = apply_feature_selection(df, selected_features, keep_metadata=return_metadata)
        n_features = len(selected_features)
        feature_set_type = "reduced"

    # Memory optimization
    if optimize_memory:
        df = optimize_dtypes(df)

    metadata = {
        "original_n_samples": original_n_samples,
        "filtered_n_samples": len(df),
        "n_features": n_features,
        "feature_set_type": feature_set_type,
        "chm_features_included": chm_features_included,
        "outliers_removed": outliers_removed,
        "test_split_policy_applied": test_policy_applied,
        "proximity_strategy_requested": original_strategy,
        "proximity_strategy_used": proximity_strategy,
    }

    return df, metadata


def evaluate_dataset_variants(
    datasets: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    feature_cols: list[str],
    target_col: str,
    cv: BaseCrossValidator,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Evaluate multiple dataset variants with cross-validation.

    Used by proximity and outlier ablation experiments to compare different
    filtering strategies.

    Args:
        datasets: Dictionary mapping variant names to (train_df, val_df) tuples
        feature_cols: List of feature column names to use for training
        target_col: Target column name (e.g., "genus_latin")
        cv: Cross-validator (e.g., StratifiedGroupKFold)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: variant, val_f1_mean, val_f1_std, n_samples
    """
    results = []

    for variant_name, (train_df, _) in datasets.items():
        x_train = train_df[feature_cols].to_numpy()
        y_train = train_df[target_col].to_numpy()
        groups = train_df["block_id"].to_numpy()

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )

        val_scores = []
        for train_idx, val_idx in cv.split(x_train, y=y_train, groups=groups):
            x_tr = x_train[train_idx]
            y_tr = y_train[train_idx]
            x_val = x_train[val_idx]
            y_val = y_train[val_idx]

            model.fit(x_tr, y_tr)
            y_pred = model.predict(x_val)
            val_scores.append(f1_score(y_val, y_pred, average="weighted"))

        results.append(
            {
                "variant": variant_name,
                "val_f1_mean": float(np.mean(val_scores)),
                "val_f1_std": float(np.std(val_scores, ddof=0)),
                "n_samples": len(train_df),
            }
        )

    return pd.DataFrame(results)


def compute_feature_importance(
    x_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    method: str = "rf_gain",
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Compute and rank feature importance.

    Args:
        x_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        feature_names: List of feature names matching x_train columns
        method: Importance method ("rf_gain" is currently supported)
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: feature, importance, rank (sorted descending by importance)

    Examples:
        >>> import numpy as np
        >>> from urban_tree_transfer.experiments import compute_feature_importance
        >>> x_train = np.random.randn(100, 5)
        >>> y_train = np.random.randint(0, 3, 100)
        >>> feature_names = ["NDVI_06", "NDVI_07", "EVI_06", "CHM_1m", "CHM_1m_zscore"]
        >>> importance_df = compute_feature_importance(x_train, y_train, feature_names)
        >>> print(importance_df.head())
                  feature  importance  rank
        0  CHM_1m_zscore        0.25     1
        1         CHM_1m        0.23     2
        2        NDVI_06        0.20     3
    """
    if method != "rf_gain":
        raise ValueError(f"Unsupported importance method: {method}. Only 'rf_gain' is supported.")

    if x_train.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: x_train has {x_train.shape[1]} columns "
            f"but {len(feature_names)} feature names provided"
        )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    return importance_df


def create_feature_subsets(
    importance_df: pd.DataFrame,
    n_features_list: list[int],
) -> dict[str, list[str]]:
    """Create feature subsets based on importance ranking.

    Args:
        importance_df: Feature importance DataFrame from compute_feature_importance()
        n_features_list: List of feature counts (e.g., [30, 50, 80])

    Returns:
        Dictionary mapping subset names to feature lists

    Examples:
        >>> import pandas as pd
        >>> from urban_tree_transfer.experiments import create_feature_subsets
        >>> importance_df = pd.DataFrame({
        ...     "feature": ["NDVI_06", "NDVI_07", "EVI_06", "CHM_1m"],
        ...     "importance": [0.4, 0.3, 0.2, 0.1],
        ...     "rank": [1, 2, 3, 4],
        ... })
        >>> subsets = create_feature_subsets(importance_df, [2, 3])
        >>> print(subsets["top_2"])
        ['NDVI_06', 'NDVI_07']
        >>> print(subsets["top_3"])
        ['NDVI_06', 'NDVI_07', 'EVI_06']
    """
    if "feature" not in importance_df.columns:
        raise ValueError("importance_df must have 'feature' column")

    if "importance" not in importance_df.columns:
        raise ValueError("importance_df must have 'importance' column")

    max_features = len(importance_df)
    if any(n > max_features for n in n_features_list):
        raise ValueError(f"Cannot select more than {max_features} features")

    subsets = {}
    for n in sorted(n_features_list):
        subset_name = f"top_{n}"
        top_features = importance_df.head(n)["feature"].tolist()
        subsets[subset_name] = top_features

    subsets["all_features"] = importance_df["feature"].tolist()

    return subsets


def evaluate_feature_subsets(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    feature_subsets: dict[str, list[str]],
    cv: BaseCrossValidator,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Evaluate each feature subset with cross-validation.

    Args:
        x: Full feature array of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        groups: Group labels for cross-validation (e.g., block_id)
        feature_names: List of all feature names matching x columns
        feature_subsets: Dictionary from create_feature_subsets()
        cv: Cross-validator
        random_seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: variant, n_features, val_f1_mean, val_f1_std

    Examples:
        >>> import numpy as np
        >>> from sklearn.model_selection import GroupKFold
        >>> from urban_tree_transfer.experiments import evaluate_feature_subsets
        >>> x = np.random.randn(150, 10)
        >>> y = np.random.randint(0, 3, 150)
        >>> groups = np.random.randint(0, 10, 150)
        >>> feature_names = [f"feature_{i}" for i in range(10)]
        >>> feature_subsets = {"top_5": feature_names[:5], "all": feature_names}
        >>> cv = GroupKFold(n_splits=3)
        >>> results = evaluate_feature_subsets(x, y, groups, feature_names, feature_subsets, cv)
        >>> print(results)
          variant  n_features  val_f1_mean  val_f1_std
        0   top_5           5         0.45        0.03
        1     all          10         0.52        0.02
    """
    if x.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature count mismatch: x has {x.shape[1]} columns "
            f"but {len(feature_names)} feature names provided"
        )

    results = []

    for subset_name, subset_features in feature_subsets.items():
        feature_indices = [feature_names.index(f) for f in subset_features]
        x_subset = x[:, feature_indices]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
        )

        val_scores = []
        for train_idx, val_idx in cv.split(x_subset, y=y, groups=groups):
            x_tr = x_subset[train_idx]
            y_tr = y[train_idx]
            x_val = x_subset[val_idx]
            y_val = y[val_idx]

            model.fit(x_tr, y_tr)
            y_pred = model.predict(x_val)
            val_scores.append(f1_score(y_val, y_pred, average="weighted"))

        results.append(
            {
                "variant": subset_name,
                "n_features": len(subset_features),
                "val_f1_mean": float(np.mean(val_scores)),
                "val_f1_std": float(np.std(val_scores, ddof=0)),
            }
        )

    return pd.DataFrame(results)


def select_optimal_features(
    results_df: pd.DataFrame,
    max_drop: float = 0.01,
) -> tuple[str, int]:
    """Select optimal feature set based on decision rules.

    Selects the smallest feature set with F1 within max_drop of the best F1.

    Args:
        results_df: Results from evaluate_feature_subsets()
        max_drop: Maximum acceptable F1 drop from best performance (default: 0.01)

    Returns:
        Tuple of (selected_variant_name, n_features)

    Examples:
        >>> import pandas as pd
        >>> from urban_tree_transfer.experiments import select_optimal_features
        >>> results_df = pd.DataFrame({
        ...     "variant": ["top_30", "top_50", "top_80", "all_features"],
        ...     "n_features": [30, 50, 80, 147],
        ...     "val_f1_mean": [0.72, 0.75, 0.76, 0.76],
        ...     "val_f1_std": [0.02, 0.015, 0.01, 0.01],
        ... })
        >>> selected_variant, n_features = select_optimal_features(results_df, max_drop=0.02)
        >>> print(selected_variant, n_features)
        top_50 50
    """
    if "variant" not in results_df.columns or "val_f1_mean" not in results_df.columns:
        raise ValueError("results_df must have 'variant' and 'val_f1_mean' columns")

    best_f1 = results_df["val_f1_mean"].max()
    threshold = best_f1 - max_drop

    candidates = results_df[results_df["val_f1_mean"] >= threshold].copy()

    if len(candidates) == 0:
        raise ValueError(f"No variants within {max_drop} of best F1 ({best_f1:.4f})")

    candidates = candidates.sort_values(by="n_features", ascending=True)  # type: ignore[call-overload]
    selected = candidates.iloc[0]

    return str(selected["variant"]), int(selected["n_features"])
