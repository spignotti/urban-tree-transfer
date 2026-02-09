"""Model factory and naive baselines for Phase 3 experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import torch  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

if torch is not None:  # pragma: no cover - optional dependency
    import torch.nn as torch_nn  # type: ignore[import-not-found]

try:
    import xgboost as xgb
except Exception:  # pragma: no cover - optional dependency
    xgb = None  # type: ignore[assignment]

try:
    from pytorch_tabnet.tab_model import TabNetClassifier  # type: ignore[import-not-found]
except ImportError:
    TabNetClassifier = None  # type: ignore[assignment,misc]

from urban_tree_transfer.config import RANDOM_SEED

DEFAULT_RF_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "class_weight": "balanced",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "eval_metric": "mlogloss",
    "use_label_encoder": False,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

DEFAULT_CNN_PARAMS: dict[str, Any] = {
    "conv_filters": [64, 128, 128],
    "kernel_size": 3,
    "dropout": 0.3,
    "dense_units": [256, 128],
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 50,
    "early_stopping_patience": 10,
}

DEFAULT_TABNET_PARAMS: dict[str, Any] = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 0.001,
    "optimizer_params": {"lr": 0.02},
    "scheduler_params": {"step_size": 50, "gamma": 0.9},
    "mask_type": "entmax",
}

SUPPORTED_MODELS = ["random_forest", "xgboost", "cnn_1d", "tabnet"]


def create_model(
    model_name: str,
    model_params: dict[str, Any] | None = None,
    random_seed: int = RANDOM_SEED,
) -> Any:
    """Create model instance from name and parameters.

    Supports: 'random_forest', 'xgboost', 'cnn_1d', 'tabnet'

    Args:
        model_name: One of the supported model names.
        model_params: Model hyperparameters (merged with defaults).
        random_seed: Random seed for reproducibility.

    Returns:
        Instantiated model object (sklearn/xgboost/pytorch).

    Raises:
        ValueError: If model_name is not supported.
        ImportError: If required package not installed (e.g., pytorch_tabnet).
    """
    params = dict(model_params or {})

    if model_name == "random_forest":
        merged = {**DEFAULT_RF_PARAMS, **params, "random_state": random_seed}
        return RandomForestClassifier(**merged)

    if model_name == "xgboost":
        if xgb is None:
            raise ImportError("xgboost is required for model_name='xgboost'.")
        merged = {**DEFAULT_XGB_PARAMS, **params, "random_state": random_seed}
        return xgb.XGBClassifier(**merged)

    if model_name == "cnn_1d":
        if torch is None:
            raise ImportError("PyTorch is required for model_name='cnn_1d'.")
        required = {"n_temporal_bases", "n_months", "n_static_features", "n_classes"}
        missing = sorted(required - set(params))
        if missing:
            raise ValueError(f"Missing CNN1D parameters: {missing}")
        cnn_params = {
            "n_temporal_bases": params.pop("n_temporal_bases"),
            "n_months": params.pop("n_months"),
            "n_static_features": params.pop("n_static_features"),
            "n_classes": params.pop("n_classes"),
        }
        merged = {**DEFAULT_CNN_PARAMS, **params}
        return CNN1D(**cnn_params, **_filter_cnn_init_params(merged))

    if model_name == "tabnet":
        if TabNetClassifier is None:
            raise ImportError("pytorch-tabnet is required for model_name='tabnet'.")
        merged = {**DEFAULT_TABNET_PARAMS, **params}
        return TabNetClassifier(**merged)

    raise ValueError(f"Unsupported model_name: {model_name}. Supported: {SUPPORTED_MODELS}")


def create_majority_classifier(y_train: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Create majority class baseline.

    Returns a callable that always predicts the most frequent class.

    Args:
        y_train: Training labels (encoded integers).

    Returns:
        Callable that takes X_test and returns predictions (all same class).
    """
    if y_train.size == 0:
        raise ValueError("y_train must not be empty.")

    counts = np.bincount(y_train.astype(int))
    majority_class = int(np.argmax(counts))

    def predict(x_test: np.ndarray) -> np.ndarray:
        return np.full((x_test.shape[0],), majority_class, dtype=int)

    return predict


def create_stratified_random_classifier(
    y_train: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> Callable[[np.ndarray], np.ndarray]:
    """Create stratified random baseline.

    Predictions weighted by class distribution in training data.

    Args:
        y_train: Training labels (encoded integers).
        random_seed: Random seed for reproducibility.

    Returns:
        Callable that takes X_test and returns random predictions.
    """
    if y_train.size == 0:
        raise ValueError("y_train must not be empty.")

    counts = np.bincount(y_train.astype(int))
    probabilities = counts / counts.sum()
    classes = np.arange(len(counts))
    rng = np.random.RandomState(random_seed)

    def predict(x_test: np.ndarray) -> np.ndarray:
        return rng.choice(classes, size=x_test.shape[0], p=probabilities)

    return predict


def create_spatial_only_rf(
    x_train: np.ndarray,
    y_train: np.ndarray,
    coord_indices: tuple[int, int] | None = None,
    n_estimators: int = 100,
    max_depth: int = 10,
    random_seed: int = RANDOM_SEED,
) -> Any:
    """Create Random Forest trained only on spatial coordinates.

    Baseline to test if spectral features add value over spatial distribution.

    Args:
        X_train: Training features (full feature matrix).
        y_train: Training labels.
        coord_indices: Indices of x/y coordinate columns. If None, assumes first 2 cols.
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        random_seed: Random seed.

    Returns:
        Trained RandomForestClassifier (fitted on coordinates only).

    Raises:
        ValueError: If coordinates cannot be extracted.
    """
    if x_train.ndim != 2:
        raise ValueError("x_train must be 2D.")

    if coord_indices is None:
        coord_indices = (0, 1)

    if max(coord_indices) >= x_train.shape[1]:
        raise ValueError("coord_indices out of bounds for x_train.")

    coords = x_train[:, list(coord_indices)]
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_seed,
        n_jobs=-1,
    )
    model.fit(coords, y_train)
    return model


if torch is None:  # pragma: no cover - optional dependency

    class CNN1D:  # type: ignore[no-redef]
        """Placeholder CNN1D when PyTorch is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError("PyTorch is required to use CNN1D.")

        def get_init_params(self) -> dict[str, Any]:
            raise ImportError("PyTorch is required to use CNN1D.")

else:
    assert torch is not None

    class CNN1D(torch_nn.Module):
        """1D CNN for temporal feature classification."""

        def __init__(
            self,
            n_temporal_bases: int,
            n_months: int,
            n_static_features: int,
            n_classes: int,
            conv_filters: list[int] | None = None,
            kernel_size: int = 3,
            dropout: float = 0.3,
            dense_units: list[int] | None = None,
        ) -> None:
            """Initialize 1D-CNN."""
            super().__init__()
            self.n_temporal_bases = n_temporal_bases
            self.n_months = n_months
            self.n_static_features = n_static_features
            self.n_classes = n_classes

            if conv_filters is None:
                conv_filters = DEFAULT_CNN_PARAMS["conv_filters"]
            if dense_units is None:
                dense_units = DEFAULT_CNN_PARAMS["dense_units"]
            assert conv_filters is not None
            assert dense_units is not None

            self._init_params = {
                "n_temporal_bases": n_temporal_bases,
                "n_months": n_months,
                "n_static_features": n_static_features,
                "n_classes": n_classes,
                "conv_filters": list(conv_filters),
                "kernel_size": kernel_size,
                "dropout": dropout,
                "dense_units": list(dense_units),
            }

            # Store learning_rate as instance attribute for fine-tuning (not an __init__ param)
            self.learning_rate = DEFAULT_CNN_PARAMS["learning_rate"]

            if n_temporal_bases <= 0 or n_months <= 0 or n_classes <= 1:
                raise ValueError("n_temporal_bases, n_months, and n_classes must be positive.")

            padding = kernel_size // 2
            conv_layers: list[Any] = []
            in_channels = n_temporal_bases

            for out_channels in conv_filters:
                conv_layers.extend(
                    [
                        torch_nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                        torch_nn.BatchNorm1d(out_channels),
                        torch_nn.ReLU(),
                        torch_nn.MaxPool1d(kernel_size=2),
                    ]
                )
                in_channels = out_channels

            self.conv_stack = torch_nn.Sequential(*conv_layers)

            conv_out_size = _infer_conv_output_size(self.conv_stack, n_temporal_bases, n_months)
            dense_input = conv_out_size + n_static_features

            dense_layers: list[Any] = []
            prev_units = dense_input
            for units in dense_units:
                dense_layers.append(torch_nn.Linear(prev_units, units))
                dense_layers.append(torch_nn.ReLU())
                dense_layers.append(torch_nn.Dropout(dropout))
                prev_units = units

            self.dense_stack = torch_nn.Sequential(*dense_layers)
            self.output_layer = torch_nn.Linear(prev_units, n_classes)

        def forward(self, x: Any) -> Any:
            """Forward pass."""
            assert torch is not None
            if x.ndim != 2:
                raise ValueError("Input must be a 2D tensor (batch_size, n_features).")

            temporal_size = self.n_temporal_bases * self.n_months
            if x.shape[1] < temporal_size:
                raise ValueError("Input feature size smaller than temporal feature count.")

            temporal = x[:, :temporal_size]
            static = x[:, temporal_size:]

            temporal = temporal.view(-1, self.n_temporal_bases, self.n_months)
            conv_out = self.conv_stack(temporal)
            conv_flat = conv_out.view(conv_out.size(0), -1)

            if self.n_static_features > 0:
                combined = torch.cat([conv_flat, static], dim=1)
            else:
                combined = conv_flat

            if len(self.dense_stack) > 0:
                combined = self.dense_stack(combined)
            return self.output_layer(combined)

        def predict(
            self,
            x_test: np.ndarray,
            batch_size: int = 256,
            device: str | None = None,
        ) -> np.ndarray:
            """Predict class labels for CNN1D."""
            if torch is None:
                raise ImportError("PyTorch is required to run CNN1D predictions.")
            assert torch is not None
            use_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            self.to(use_device)
            self.eval()
            dataset = torch.utils.data.TensorDataset(torch.from_numpy(x_test).float())
            loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
            preds: list[np.ndarray] = []
            with torch.no_grad():
                for (batch_x,) in loader:
                    batch_x = batch_x.to(use_device)
                    logits = self(batch_x)
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy())
            return np.concatenate(preds)

        def get_init_params(self) -> dict[str, Any]:
            """Return initialization parameters for cloning the model."""
            return dict(self._init_params)


def train_cnn(
    model: CNN1D,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    batch_size: int = 64,
    epochs: int = 50,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    device: str = "cpu",
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Train 1D-CNN with early stopping."""
    if torch is None:
        raise ImportError("PyTorch is required to train CNN1D.")
    assert torch is not None
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x_train).float(),
        torch.from_numpy(y_train).long(),
    )
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    has_val = x_val is not None and y_val is not None
    if has_val:
        val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(x_val).float(),  # type: ignore[arg-type]
            torch.from_numpy(y_val).long(),
        )
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_loader = None

    history: dict[str, Any] = {
        "train_loss": [],
        "val_loss": [],
        "best_epoch": 0,
        "stopped_early": False,
    }

    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        epoch_losses: list[float] = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        history["train_loss"].append(float(np.mean(epoch_losses)))

        if not has_val:
            continue

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:  # type: ignore[union-attr]
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
                val_losses.append(float(loss.item()))

        val_loss = float(np.mean(val_losses))
        history["val_loss"].append(val_loss)

        if val_loss < best_loss - 1e-6:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history["best_epoch"] = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            history["stopped_early"] = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Store actual learning rate used for fine-tuning reference
    model.learning_rate = learning_rate

    return history


def _filter_cnn_init_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "conv_filters": params.get("conv_filters", DEFAULT_CNN_PARAMS["conv_filters"]),
        "kernel_size": params.get("kernel_size", DEFAULT_CNN_PARAMS["kernel_size"]),
        "dropout": params.get("dropout", DEFAULT_CNN_PARAMS["dropout"]),
        "dense_units": params.get("dense_units", DEFAULT_CNN_PARAMS["dense_units"]),
    }


def _infer_conv_output_size(conv_stack: Any, n_temporal_bases: int, n_months: int) -> int:
    assert torch is not None
    with torch.no_grad():
        dummy = torch.zeros(1, n_temporal_bases, n_months)
        out = conv_stack(dummy)
        return int(out.numel())
