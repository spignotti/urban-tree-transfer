"""Optuna hyperparameter tuning utilities for Phase 3 experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    import optuna  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore[assignment]

from urban_tree_transfer.config import RANDOM_SEED
from urban_tree_transfer.experiments.models import create_model
from urban_tree_transfer.experiments.training import train_with_cv


def create_study(
    direction: str = "maximize",
    random_seed: int = RANDOM_SEED,
    sampler_name: str = "tpe",
    pruner_name: str = "median",
    sampler: str | None = None,
    pruner: str | None = None,
) -> Any:
    """Create an Optuna study with standard settings.

    Args:
        direction: Optimization direction (maximize/minimize).
        random_seed: Random seed for reproducibility.
        sampler_name: Sampler identifier (default: "tpe").
        pruner_name: Pruner identifier (default: "median").
        sampler: Backward-compatible alias for sampler_name.
        pruner: Backward-compatible alias for pruner_name.
    """
    if optuna is None:
        raise ImportError("optuna is required for hyperparameter tuning.")

    if sampler is not None:
        sampler_name = sampler
    if pruner is not None:
        pruner_name = pruner

    sampler_obj = None
    if sampler_name == "tpe":
        sampler_obj = optuna.samplers.TPESampler(seed=random_seed)

    pruner_obj = None
    if pruner_name == "median":
        pruner_obj = optuna.pruners.MedianPruner()

    return optuna.create_study(direction=direction, sampler=sampler_obj, pruner=pruner_obj)


def build_objective(
    model_name: str,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: Any,
    search_space: dict[str, Any],
    base_params: dict[str, Any] | None = None,
    fit_params: dict[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
) -> Any:
    """Build Optuna objective for a given model and dataset."""
    if optuna is None:
        raise ImportError("optuna is required for hyperparameter tuning.")

    base_params = dict(base_params or {})
    fit_params = dict(fit_params or {})

    def objective(trial: Any) -> float:
        params = suggest_params_from_space(trial, search_space)
        model_params = {**base_params, **params}
        model = create_model(model_name, model_params, random_seed=random_seed)

        results = train_with_cv(
            model,
            x,
            y,
            groups,
            cv,
            fit_params=fit_params,
        )
        trial.set_user_attr("train_val_gap", results["train_val_gap"])
        return float(results["val_f1_mean"])

    return objective


def run_optuna_search(
    study: Any,
    objective: Any,
    n_trials: int = 50,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run Optuna study and return summary dict."""
    if optuna is None:
        raise ImportError("optuna is required for hyperparameter tuning.")

    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    trials = []
    for trial in study.trials:
        if trial.value is None:
            continue
        trials.append(
            {
                "value": float(trial.value),
                "params": trial.params,
                "train_val_gap": trial.user_attrs.get("train_val_gap"),
            }
        )

    return {
        "best_score": float(study.best_value),
        "best_params": study.best_params,
        "trials": trials,
    }


def suggest_params_from_space(trial: Any, space: dict[str, Any]) -> dict[str, Any]:
    """Suggest parameters from a config-defined search space."""
    params: dict[str, Any] = {}
    for key, value in space.items():
        if isinstance(value, dict):
            nested = suggest_params_from_space(trial, value)
            params[key] = nested
            continue

        if not isinstance(value, list):
            params[key] = value
            continue

        if len(value) == 2 and all(isinstance(v, (int, float)) or v is None for v in value):
            low, high = value
            if isinstance(low, float) or isinstance(high, float):
                params[key] = trial.suggest_float(key, float(low), float(high))
            else:
                params[key] = trial.suggest_int(key, int(low), int(high))
            continue

        params[key] = trial.suggest_categorical(key, value)

    return params
