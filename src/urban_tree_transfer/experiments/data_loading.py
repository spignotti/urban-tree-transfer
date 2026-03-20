# ruff: noqa: I001
"""Data loading helpers for Phase 3 experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


_ID_COLUMNS = {
    "tree_id",
    "city",
    "block_id",
}

_TARGET_COLUMN = "genus_latin"

_METADATA_COLUMNS = {
    "plant_year",
    "species_latin",
    "genus_german",
    "species_german",
    "tree_type",
    "position_corrected",
    "correction_distance",
    "is_conifer",
}

_OUTLIER_COLUMNS = {
    "outlier_zscore",
    "outlier_mahalanobis",
    "outlier_iqr",
    "outlier_severity",
    "outlier_method_count",
}

_CHM_FEATURES = ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"]

_ALLOWED_VARIANTS = {"baseline", "filtered"}


def load_parquet_dataset(file_path: Path) -> pd.DataFrame:
    """Load ML-ready dataset from Parquet file.

    Args:
        file_path: Path to parquet file (e.g., berlin_train.parquet).

    Returns:
        DataFrame with all columns (features + metadata). CHM features are optional
        and may be absent if excluded by setup decisions.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If required columns are missing.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {file_path}")

    df = pd.read_parquet(file_path)

    required_columns = _ID_COLUMNS | {_TARGET_COLUMN} | _METADATA_COLUMNS | _OUTLIER_COLUMNS
    missing = sorted(col for col in required_columns if col not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"Loaded {len(df)} rows from {file_path}")
    return df


def fix_missing_genus_german(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing genus_german values using known mappings.

    Only 2 genera are affected:
    - PRUNUS -> "Prunus"
    - SOPHORA -> "Schnurbaum"

    Args:
        df: DataFrame with genus_latin and genus_german columns.

    Returns:
        DataFrame with filled genus_german values.

    Raises:
        ValueError: If required columns are missing or unmapped NaNs remain.
    """
    if "genus_latin" not in df.columns or "genus_german" not in df.columns:
        raise ValueError("Input DataFrame must contain genus_latin and genus_german columns.")

    result = df.copy()

    mapping = {
        "PRUNUS": "Prunus",
        "SOPHORA": "Schnurbaum",
    }

    missing_mask = result["genus_german"].isna()
    for latin, german in mapping.items():
        fill_mask = missing_mask & (result["genus_latin"] == latin)
        result.loc[fill_mask, "genus_german"] = german

    filled_count = int(missing_mask.sum() - result["genus_german"].isna().sum())
    print(f"Fixed {filled_count} missing genus_german values")

    remaining_missing = result["genus_german"].isna()
    if bool(remaining_missing.any()):
        remaining_genera = sorted(result.loc[remaining_missing, "genus_latin"].unique())
        raise ValueError(
            "Unmapped genus_german values remain for genera: "
            f"{remaining_genera} (count={remaining_missing.sum()})"
        )

    return result


def get_feature_columns(
    df: pd.DataFrame,
    include_chm: bool = True,
    chm_features: list[str] | None = None,
    expected_features: list[str] | None = None,
) -> list[str]:
    """Extract feature column names (CHM + Sentinel-2) from DataFrame.

    Args:
        df: Input DataFrame.
        include_chm: Whether to include CHM features.
        chm_features: Specific CHM features to include (default: all 3).
                      Options: ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"].
        expected_features: Optional list of expected feature columns. When provided,
            validates presence and returns columns in this order.

    Returns:
        Sorted list of feature column names, or the expected list if provided.

    Raises:
        ValueError: If expected features are missing or inputs invalid.
    """
    exclude_cols = _ID_COLUMNS | {_TARGET_COLUMN} | _METADATA_COLUMNS | _OUTLIER_COLUMNS

    if chm_features is not None and not include_chm:
        raise ValueError("chm_features provided but include_chm is False.")

    if chm_features is not None:
        invalid = [feature for feature in chm_features if feature not in _CHM_FEATURES]
        if invalid:
            raise ValueError(f"Invalid CHM features requested: {invalid}")

    all_columns = [col for col in df.columns if col not in exclude_cols]
    chm_available = [col for col in _CHM_FEATURES if col in all_columns]
    sentinel_features = [col for col in all_columns if col not in _CHM_FEATURES]

    if expected_features is not None:
        missing_expected = [col for col in expected_features if col not in all_columns]
        if missing_expected:
            raise ValueError(f"Missing expected features: {missing_expected}")
        return list(expected_features)

    if include_chm:
        if chm_features is None:
            chm_selected = chm_available
        else:
            missing_requested = [col for col in chm_features if col not in all_columns]
            if missing_requested:
                raise ValueError(f"Missing requested CHM features: {missing_requested}")
            chm_selected = chm_features
        feature_columns = sentinel_features + chm_selected
    else:
        feature_columns = sentinel_features

    return sorted(feature_columns)


def load_berlin_splits(
    data_dir: Path,
    variant: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Berlin train/val/test splits.

    Args:
        data_dir: Directory containing parquet files.
        variant: "baseline" (no filtering) or "filtered" (proximity-filtered).

    Returns:
        (train_df, val_df, test_df).

    Raises:
        ValueError: If variant is not "baseline" or "filtered".
    """
    if variant not in _ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_ALLOWED_VARIANTS)}")

    suffix = "" if variant == "baseline" else "_filtered"
    train_path = data_dir / f"berlin_train{suffix}.parquet"
    val_path = data_dir / f"berlin_val{suffix}.parquet"
    test_path = data_dir / f"berlin_test{suffix}.parquet"

    train_df = fix_missing_genus_german(load_parquet_dataset(train_path))
    val_df = fix_missing_genus_german(load_parquet_dataset(val_path))
    test_df = fix_missing_genus_german(load_parquet_dataset(test_path))

    return train_df, val_df, test_df


def load_leipzig_splits(
    data_dir: Path,
    variant: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Leipzig finetune/test splits.

    Args:
        data_dir: Directory containing parquet files.
        variant: "baseline" (no filtering) or "filtered" (proximity-filtered).

    Returns:
        (finetune_df, test_df).

    Raises:
        ValueError: If variant is not "baseline" or "filtered".
    """
    if variant not in _ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_ALLOWED_VARIANTS)}")

    suffix = "" if variant == "baseline" else "_filtered"
    finetune_path = data_dir / f"leipzig_finetune{suffix}.parquet"
    test_path = data_dir / f"leipzig_test{suffix}.parquet"

    finetune_df = fix_missing_genus_german(load_parquet_dataset(finetune_path))
    test_df = fix_missing_genus_german(load_parquet_dataset(test_path))

    return finetune_df, test_df


def load_berlin_splits_cnn(
    data_dir: Path,
    variant: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Berlin train/val/test splits with full features for CNN1D.

    Loads *_cnn.parquet files containing complete temporal sequences
    (all Sentinel-2 features + CHM if included).

    Args:
        data_dir: Directory containing parquet files.
        variant: "baseline" (no filtering) or "filtered" (proximity-filtered).

    Returns:
        (train_df, val_df, test_df).

    Raises:
        ValueError: If variant is not "baseline" or "filtered".
    """
    if variant not in _ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_ALLOWED_VARIANTS)}")

    suffix = "" if variant == "baseline" else "_filtered"
    train_path = data_dir / f"berlin_train{suffix}_cnn.parquet"
    val_path = data_dir / f"berlin_val{suffix}_cnn.parquet"
    test_path = data_dir / f"berlin_test{suffix}_cnn.parquet"

    train_df = fix_missing_genus_german(load_parquet_dataset(train_path))
    val_df = fix_missing_genus_german(load_parquet_dataset(val_path))
    test_df = fix_missing_genus_german(load_parquet_dataset(test_path))

    return train_df, val_df, test_df


def load_leipzig_splits_cnn(
    data_dir: Path,
    variant: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Leipzig finetune/test splits with full features for CNN1D.

    Loads *_cnn.parquet files containing complete temporal sequences
    (all Sentinel-2 features + CHM if included).

    Args:
        data_dir: Directory containing parquet files.
        variant: "baseline" (no filtering) or "filtered" (proximity-filtered).

    Returns:
        (finetune_df, test_df).

    Raises:
        ValueError: If variant is not "baseline" or "filtered".
    """
    if variant not in _ALLOWED_VARIANTS:
        raise ValueError(f"variant must be one of {sorted(_ALLOWED_VARIANTS)}")

    suffix = "" if variant == "baseline" else "_filtered"
    finetune_path = data_dir / f"leipzig_finetune{suffix}_cnn.parquet"
    test_path = data_dir / f"leipzig_test{suffix}_cnn.parquet"

    finetune_df = fix_missing_genus_german(load_parquet_dataset(finetune_path))
    test_df = fix_missing_genus_german(load_parquet_dataset(test_path))

    return finetune_df, test_df
