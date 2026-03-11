"""Training infrastructure with spatial block cross-validation."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

from urban_tree_transfer.config import RANDOM_SEED
from urban_tree_transfer.experiments.models import CNN1D, DEFAULT_CNN_PARAMS, train_cnn


def create_spatial_block_cv(
    df: Any,
    n_splits: int = 3,
    block_id_col: str = "block_id",
    genus_col: str = "genus_latin",
    random_seed: int = RANDOM_SEED,
) -> StratifiedGroupKFold:
    """Create spatial block cross-validator."""
    if not hasattr(df, "columns"):
        raise ValueError("df must be a pandas DataFrame with required columns.")

    missing = [col for col in (block_id_col, genus_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    n_groups = df[block_id_col].nunique()
    if n_splits > n_groups:
        raise ValueError("n_splits cannot exceed number of unique blocks.")

    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)


def train_with_cv(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    cv: StratifiedGroupKFold,
    fit_params: dict[str, Any] | None = None,
    sample_weight: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train model with cross-validation and collect metrics."""
    fit_params = dict(fit_params or {})

    train_scores: list[float] = []
    val_scores: list[float] = []
    fold_indices: list[tuple[np.ndarray, np.ndarray]] = []

    for train_idx, val_idx in cv.split(x, y=y, groups=groups):
        fold_indices.append((train_idx, val_idx))

        x_train = x[train_idx]
        y_train = y[train_idx]
        x_val = x[val_idx]
        y_val = y[val_idx]

        if isinstance(model, CNN1D):
            if torch is None:
                raise ImportError("PyTorch is required for CNN1D training.")
            fold_model = CNN1D(**model.get_init_params())
            _ = train_cnn(
                fold_model,
                x_train,
                y_train,
                x_val=x_val,
                y_val=y_val,
                **fit_params,
            )
            fold_model.eval()
            with torch.no_grad():  # type: ignore[union-attr]
                y_train_pred = (
                    fold_model(torch.from_numpy(x_train).float())  # type: ignore[union-attr]
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )
                y_val_pred = (
                    fold_model(torch.from_numpy(x_val).float())  # type: ignore[union-attr]
                    .argmax(dim=1)
                    .cpu()
                    .numpy()
                )
        else:
            fold_model = cast(Any, clone(model))
            fold_fit_params = dict(fit_params)
            if sample_weight is not None:
                fold_fit_params["sample_weight"] = sample_weight[train_idx]
            fold_model.fit(x_train, y_train, **fold_fit_params)
            y_train_pred = fold_model.predict(x_train)
            y_val_pred = fold_model.predict(x_val)

        train_scores.append(float(f1_score(y_train, y_train_pred, average="weighted")))
        val_scores.append(float(f1_score(y_val, y_val_pred, average="weighted")))

    train_scores_arr = np.array(train_scores)
    val_scores_arr = np.array(val_scores)

    return {
        "train_scores": train_scores,
        "val_scores": val_scores,
        "train_f1_mean": float(train_scores_arr.mean()),
        "train_f1_std": float(train_scores_arr.std(ddof=0)),
        "val_f1_mean": float(val_scores_arr.mean()),
        "val_f1_std": float(val_scores_arr.std(ddof=0)),
        "train_val_gap": float(train_scores_arr.mean() - val_scores_arr.mean()),
        "fold_indices": fold_indices,
    }


def train_final_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    fit_params: dict[str, Any] | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Train final model on full training data (Train+Val combined)."""
    fit_params = dict(fit_params or {})

    if isinstance(model, CNN1D):
        if torch is None:
            raise ImportError("PyTorch is required for CNN1D training.")
        train_cnn(
            model,
            x_train,
            y_train,
            x_val=x_val,
            y_val=y_val,
            **fit_params,
        )
        return model

    if _is_tabnet(model):
        eval_set = None
        if x_val is not None and y_val is not None:
            eval_set = [(x_val, y_val)]
        model.fit(x_train, y_train, eval_set=eval_set, **fit_params)
        return model

    final_fit_params = dict(fit_params)
    if sample_weight is not None:
        final_fit_params["sample_weight"] = sample_weight
    model.fit(x_train, y_train, **final_fit_params)
    return model


def save_model(
    model: Any,
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save model to disk with metadata."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(model, CNN1D):
        if torch is None:
            raise ImportError("PyTorch is required to save CNN1D models.")
        torch.save(model.state_dict(), output_path)
    elif _is_tabnet(model):
        model.save_model(str(output_path))
    else:
        with output_path.open("wb") as handle:
            pickle.dump(model, handle)

    if metadata is not None:
        metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def load_model(
    model_path: Path,
    model_class: type | None = None,
    model_params: dict[str, Any] | None = None,
) -> Any:
    """Load model from disk."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    if model_path.suffix == ".pkl":
        with model_path.open("rb") as handle:
            return pickle.load(handle)

    if model_path.suffix == ".pt":
        if torch is None:
            raise ImportError("PyTorch is required to load CNN1D models.")
        if model_class is None:
            raise ValueError("model_class is required to load PyTorch models.")
        model_params = dict(model_params or {})
        model = model_class(**model_params)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model

    if model_path.suffix == ".zip":
        from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore[import-not-found]

        model = TabNetClassifier()
        model.load_model(str(model_path))
        return model

    raise ValueError(f"Unsupported model file extension: {model_path.suffix}")


def create_stratified_subsets(
    x: np.ndarray,
    y: np.ndarray,
    fractions: list[float],
    random_seed: int = RANDOM_SEED,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Create stratified subsets at different fractions.

    Args:
        x: Feature array of shape (n_samples, n_features)
        y: Target labels of shape (n_samples,)
        fractions: List of fractions to sample (e.g., [0.1, 0.25, 0.5, 1.0])
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary mapping fractions to (X_subset, y_subset) tuples

    Examples:
        >>> import numpy as np
        >>> from urban_tree_transfer.experiments import create_stratified_subsets
        >>> x_finetune = np.random.randn(1000, 147)
        >>> y_finetune = np.random.randint(0, 12, 1000)
        >>> fractions = [0.05, 0.1, 0.25, 0.5, 1.0]
        >>> subsets = create_stratified_subsets(x_finetune, y_finetune, fractions)
        >>> for frac, (x_sub, y_sub) in subsets.items():
        ...     print(f"{frac:.0%}: {len(y_sub)} samples")
        5%: 50 samples
        10%: 100 samples
        25%: 250 samples
        50%: 500 samples
        100%: 1000 samples
    """
    if not 0 < min(fractions) <= 1.0:
        raise ValueError("Fractions must be in range (0, 1]")
    if not all(f <= 1.0 for f in fractions):
        raise ValueError("All fractions must be <= 1.0")

    subsets: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    for frac in sorted(fractions):
        if frac == 1.0:
            subsets[frac] = (x, y)
        else:
            x_subset, _, y_subset, _ = train_test_split(
                x,
                y,
                train_size=frac,
                stratify=y,
                random_state=random_seed,
            )
            subsets[frac] = (x_subset, y_subset)  # type: ignore[assignment]

    return subsets


def finetune_xgboost(
    pretrained_model: Any,
    x_finetune: np.ndarray,
    y_finetune: np.ndarray,
    n_additional_estimators: int = 100,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
) -> Any:
    """Fine-tune XGBoost model with additional trees.

    Args:
        pretrained_model: Pre-trained XGBoost model
        x_finetune: Fine-tuning features of shape (n_samples, n_features)
        y_finetune: Fine-tuning labels of shape (n_samples,)
        n_additional_estimators: Number of additional boosting rounds

    Returns:
        Fine-tuned XGBoost model

    Examples:
        >>> import numpy as np
        >>> from xgboost import XGBClassifier
        >>> from urban_tree_transfer.experiments import finetune_xgboost
        >>> x_berlin = np.random.randn(5000, 147)
        >>> y_berlin = np.random.randint(0, 12, 5000)
        >>> x_leipzig = np.random.randn(1000, 147)
        >>> y_leipzig = np.random.randint(0, 12, 1000)
        >>> berlin_model = XGBClassifier(n_estimators=300)
        >>> berlin_model.fit(x_berlin, y_berlin)
        >>> finetuned_model = finetune_xgboost(
        ...     berlin_model, x_leipzig, y_leipzig, n_additional_estimators=100
        ... )
        >>> print(finetuned_model.n_estimators)
        400
    """
    try:
        import xgboost  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("XGBoost is required for fine-tuning XGBoost models.") from exc

    if not isinstance(pretrained_model, xgboost.XGBClassifier):
        raise TypeError("Model must be an XGBoost classifier")

    current_n_estimators = pretrained_model.n_estimators
    if current_n_estimators is None:
        current_n_estimators = 100  # XGBoost default

    params = pretrained_model.get_params()
    params.pop("n_estimators", None)

    def _parse_major_version(version_str: str) -> int:
        try:
            return int(version_str.split(".")[0])
        except (ValueError, AttributeError):
            return 0

    xgb_major = _parse_major_version(getattr(xgboost, "__version__", "0"))
    if params.get("tree_method") == "gpu_hist" and xgb_major >= 2:
        params["tree_method"] = "hist"
        params["device"] = "cuda"
        if params.get("predictor") == "gpu_predictor":
            params["predictor"] = "auto"
    finetuned_model = xgboost.XGBClassifier(
        **params,
        n_estimators=current_n_estimators + n_additional_estimators,
    )

    fit_kwargs: dict[str, Any] = {
        "xgb_model": pretrained_model.get_booster(),
        "verbose": False,
    }
    if x_val is not None and y_val is not None:
        fit_kwargs["eval_set"] = [(x_val, y_val)]
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight

    finetuned_model.fit(x_finetune, y_finetune, **fit_kwargs)

    return finetuned_model


def finetune_neural_network(
    pretrained_model: Any,
    x_finetune: np.ndarray,
    y_finetune: np.ndarray,
    epochs: int = 50,
    lr_factor: float = 0.1,
    **train_kwargs: Any,
) -> Any:
    """Fine-tune neural network with reduced learning rate.

    Args:
        pretrained_model: Pre-trained neural network (CNN1D or TabNet)
        x_finetune: Fine-tuning features of shape (n_samples, n_features)
        y_finetune: Fine-tuning labels of shape (n_samples,)
        epochs: Number of fine-tuning epochs
        lr_factor: Learning rate reduction factor (new_lr = old_lr * lr_factor)
        **train_kwargs: Additional training arguments

    Returns:
        Fine-tuned neural network model

    Examples:
        >>> import numpy as np
        >>> from urban_tree_transfer.experiments import CNN1D, finetune_neural_network
        >>> x_berlin = np.random.randn(5000, 147)
        >>> y_berlin = np.random.randint(0, 12, 5000)
        >>> x_leipzig = np.random.randn(1000, 147)
        >>> y_leipzig = np.random.randint(0, 12, 1000)
        >>> berlin_model = CNN1D(
        ...     n_temporal_bases=12, n_months=12, n_static_features=3, n_classes=12
        ... )
        >>> berlin_model.fit(x_berlin, y_berlin, epochs=100)
        >>> finetuned_model = finetune_neural_network(
        ...     berlin_model, x_leipzig, y_leipzig, epochs=50, lr_factor=0.1
        ... )
        >>> print(finetuned_model.learning_rate)
        0.001
    """
    if isinstance(pretrained_model, CNN1D):
        if torch is None:
            raise ImportError("PyTorch is required for CNN1D fine-tuning.")

        # Defensive: fallback to default if learning_rate not stored
        original_lr = getattr(
            pretrained_model, "learning_rate", DEFAULT_CNN_PARAMS["learning_rate"]
        )
        new_lr = original_lr * lr_factor

        history = train_cnn(
            pretrained_model,
            x_finetune,
            y_finetune,
            learning_rate=new_lr,
            epochs=epochs,
            **train_kwargs,
        )
        pretrained_model.finetune_history_ = history

        return pretrained_model

    elif _is_tabnet(pretrained_model):
        from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore[import-not-found]

        if not isinstance(pretrained_model, TabNetClassifier):
            raise TypeError("Model must be TabNet classifier")

        original_lr = pretrained_model.lr
        new_lr = original_lr * lr_factor

        pretrained_model.fit(
            x_finetune,
            y_finetune,
            max_epochs=epochs,
            patience=train_kwargs.get("patience", 15),
            **{k: v for k, v in train_kwargs.items() if k != "patience"},
        )
        pretrained_model.finetune_history_ = {}

        return pretrained_model

    else:
        raise TypeError(f"Unsupported model type for fine-tuning: {type(pretrained_model)}")


def _is_tabnet(model: Any) -> bool:
    return model.__class__.__name__ == "TabNetClassifier"
