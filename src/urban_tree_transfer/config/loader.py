"""Configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Package root directory (where configs/ lives)
_PACKAGE_ROOT = Path(__file__).parent.parent


def get_config_dir() -> Path:
    """Get the configs directory path within the package."""
    return _PACKAGE_ROOT / "configs"


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def load_city_config(city: str, config_dir: Path | None = None) -> dict[str, Any]:
    """Load a single city configuration.

    Args:
        city: City name (e.g., "berlin", "leipzig")
        config_dir: Optional override for config directory.
                   Defaults to package's configs/cities/ directory.

    Returns:
        City configuration dictionary.
    """
    if config_dir is None:
        config_dir = get_config_dir() / "cities"
    path = config_dir / f"{city.lower()}.yaml"
    return load_yaml(path)


def load_city_configs(cities: list[str], config_dir: Path | None = None) -> dict[str, dict]:
    """Load multiple city configs by name."""
    return {city: load_city_config(city, config_dir=config_dir) for city in cities}


def load_feature_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load feature engineering configuration.

    Args:
        config_dir: Optional override for config directory.
                   Defaults to package's configs/features/ directory.

    Returns:
        Feature configuration dictionary.
    """
    if config_dir is None:
        config_dir = get_config_dir() / "features"
    path = config_dir / "feature_config.yaml"
    return load_yaml(path)


def load_experiment_config(config_dir: Path | None = None) -> dict[str, Any]:
    """Load Phase 3 experiment configuration.

    Args:
        config_dir: Optional override for config directory.
                   Defaults to package's configs/experiments/ directory.

    Returns:
        Experiment configuration dictionary.
    """
    if config_dir is None:
        config_dir = get_config_dir() / "experiments"
    path = config_dir / "phase3_config.yaml"
    return load_yaml(path)


def get_algorithm_config(algorithm_name: str) -> dict[str, Any]:
    """Get configuration for a specific algorithm.

    Args:
        algorithm_name: Algorithm name (e.g., "random_forest", "xgboost", "cnn_1d", "tabnet")

    Returns:
        Algorithm configuration dictionary containing type, coarse_grid, and optuna_space.
    """
    config = load_experiment_config()
    if algorithm_name not in config["algorithms"]:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    return config["algorithms"][algorithm_name]


def get_coarse_grid(algorithm_name: str) -> dict[str, list]:
    """Get coarse hyperparameter grid for an algorithm.

    Args:
        algorithm_name: Algorithm name (e.g., "random_forest", "xgboost")

    Returns:
        Dictionary mapping hyperparameter names to lists of values to try.
    """
    algo_config = get_algorithm_config(algorithm_name)
    return algo_config["coarse_grid"]


def get_optuna_space(algorithm_name: str) -> dict[str, dict]:
    """Get Optuna search space for an algorithm.

    Args:
        algorithm_name: Algorithm name (e.g., "random_forest", "xgboost", "cnn_1d", "tabnet")

    Returns:
        Dictionary mapping hyperparameter names to Optuna search space definitions.
    """
    algo_config = get_algorithm_config(algorithm_name)
    return algo_config["optuna_space"]


def get_metadata_columns(config: dict[str, Any] | None = None) -> list[str]:
    """Get list of metadata columns to preserve through pipeline.

    Args:
        config: Optional pre-loaded feature config. Loads if not provided.

    Returns:
        List of metadata column names.
    """
    if config is None:
        config = load_feature_config()
    return config["metadata_columns"]


def get_spectral_bands(config: dict[str, Any] | None = None) -> list[str]:
    """Get list of Sentinel-2 spectral band names.

    Args:
        config: Optional pre-loaded feature config. Loads if not provided.

    Returns:
        List of spectral band names (B2, B3, ..., B12).
    """
    if config is None:
        config = load_feature_config()
    return config["spectral_bands"]


def get_vegetation_indices(config: dict[str, Any] | None = None) -> list[str]:
    """Get flat list of all vegetation index names.

    Args:
        config: Optional pre-loaded feature config. Loads if not provided.

    Returns:
        List of vegetation index names (NDVI, EVI, ..., kNDVI).
    """
    if config is None:
        config = load_feature_config()
    indices = config["vegetation_indices"]
    return indices["broadband"] + indices["red_edge"] + indices["water"]


def get_all_s2_features(config: dict[str, Any] | None = None) -> list[str]:
    """Get combined list of all Sentinel-2 features (bands + indices).

    Args:
        config: Optional pre-loaded feature config. Loads if not provided.

    Returns:
        List of all S2 feature names (23 total).
    """
    if config is None:
        config = load_feature_config()
    return get_spectral_bands(config) + get_vegetation_indices(config)


def get_temporal_feature_names(
    months: list[int] | None = None,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Get list of temporal feature names for specified months.

    Feature naming convention: {feature}_{month:02d} (e.g., NDVI_04 for April NDVI)

    Args:
        months: List of months (1-12). Defaults to extraction_months from config.
        config: Optional pre-loaded feature config. Loads if not provided.

    Returns:
        List of temporal feature names (e.g., ['B2_01', 'B2_02', ..., 'kNDVI_12']).
    """
    if config is None:
        config = load_feature_config()

    months_list: list[int] = (
        months if months is not None else config["temporal"]["extraction_months"]
    )

    s2_features = get_all_s2_features(config)
    temporal_names: list[str] = []

    for month in sorted(months_list):
        for feature in s2_features:
            temporal_names.append(f"{feature}_{month:02d}")

    return temporal_names


def get_chm_feature_names(include_engineered: bool = False) -> list[str]:
    """Get list of CHM feature names.

    Args:
        include_engineered: If True, include engineered features (zscore, percentile).
                           If False, only return raw extraction feature (CHM_1m).

    Returns:
        List of CHM feature names.
    """
    if include_engineered:
        return ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"]
    return ["CHM_1m"]


def get_all_feature_names(
    months: list[int] | None = None,
    include_chm_engineered: bool = False,
    config: dict[str, Any] | None = None,
) -> list[str]:
    """Get complete list of all feature names.

    Args:
        months: List of months for temporal features. Defaults to extraction_months.
        include_chm_engineered: Include CHM engineered features (zscore, percentile).
        config: Optional pre-loaded feature config. Loads if not provided.

    Returns:
        List of all feature names (CHM + temporal S2).
    """
    chm_features = get_chm_feature_names(include_engineered=include_chm_engineered)
    temporal_features = get_temporal_feature_names(months=months, config=config)
    return chm_features + temporal_features
