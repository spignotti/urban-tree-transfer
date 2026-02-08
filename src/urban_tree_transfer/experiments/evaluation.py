"""Evaluation utilities for Phase 3 experiments."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "weighted",
) -> dict:
    """Compute classification metrics.

    Args:
        y_true: True labels (encoded integers).
        y_pred: Predicted labels (encoded integers).
        average: "weighted", "macro", or "micro".

    Returns:
        Dict with keys: f1_score, precision, recall, accuracy.

    Raises:
        ValueError: If average is invalid or inputs are empty.
    """
    if y_true.size == 0:
        raise ValueError("y_true is empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if average not in {"weighted", "macro", "micro"}:
        raise ValueError("average must be 'weighted', 'macro', or 'micro'.")

    metrics = {
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division="warn"),
        "precision": precision_score(y_true, y_pred, average=average, zero_division="warn"),
        "recall": recall_score(y_true, y_pred, average=average, zero_division="warn"),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    """Compute per-class precision, recall, and F1.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_names: List of genus names in encoded order.

    Returns:
        DataFrame with columns: genus, precision, recall, f1_score, support.

    Raises:
        ValueError: If class_names length does not match label range.
    """
    if not class_names:
        raise ValueError("class_names must be non-empty.")

    labels = list(range(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division="warn",
    )
    report_dict = cast(dict[str, dict[str, float]], report)

    rows = []
    for name in class_names:
        if name not in report_dict:
            raise ValueError(f"Missing class '{name}' in classification report.")
        rows.append(
            {
                "genus": name,
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1_score": report_dict[name]["f1-score"],
                "support": report_dict[name]["support"],
            }
        )

    return pd.DataFrame(rows)


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        metric_fn: Function that takes (y_true, y_pred) and returns float.
        n_bootstrap: Number of bootstrap samples.
        confidence_level: Confidence level (0.95 = 95%).
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, lower_bound, upper_bound).

    Raises:
        ValueError: If inputs are invalid.
    """
    if y_true.size == 0:
        raise ValueError("y_true is empty.")
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be > 0.")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(random_seed)
    n_samples = y_true.shape[0]

    boot_metrics = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_samples, size=n_samples)
        boot_metrics.append(metric_fn(y_true[indices], y_pred[indices]))

    point_estimate = metric_fn(y_true, y_pred)
    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_metrics, 100 * alpha))
    upper = float(np.percentile(boot_metrics, 100 * (1 - alpha)))

    return point_estimate, lower, upper


def analyze_worst_confused_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    genus_german_map: dict[str, str] | None = None,
    top_k: int = 10,
) -> pd.DataFrame:
    """Extract top-k most confused genus pairs from confusion matrix.

    Args:
        y_true: True labels (encoded integers).
        y_pred: Predicted labels (encoded integers).
        class_labels: List of genus names (latin) in encoded order.
        genus_german_map: Optional mapping from latin to German names.
        top_k: Number of worst pairs to return.

    Returns:
        DataFrame with columns:
        - true_genus: True genus (German if map provided, else Latin)
        - pred_genus: Predicted genus (German if map provided, else Latin)
        - count: Number of misclassifications
        - confusion_rate: count / total_true_genus_samples

        Sorted by count descending, excludes diagonal (correct predictions).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if not class_labels:
        raise ValueError("class_labels must be non-empty.")
    if top_k <= 0:
        raise ValueError("top_k must be > 0.")

    labels = list(range(len(class_labels)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1)

    rows: list[dict[str, object]] = []
    for true_idx in range(cm.shape[0]):
        for pred_idx in range(cm.shape[1]):
            if true_idx == pred_idx:
                continue
            count = int(cm[true_idx, pred_idx])
            if count == 0:
                continue
            denom = row_sums[true_idx]
            confusion_rate = float(count / denom) if denom else 0.0
            true_label = class_labels[true_idx]
            pred_label = class_labels[pred_idx]
            if genus_german_map is not None:
                true_label = genus_german_map.get(true_label, true_label)
                pred_label = genus_german_map.get(pred_label, pred_label)
            rows.append(
                {
                    "true_genus": true_label,
                    "pred_genus": pred_label,
                    "count": count,
                    "confusion_rate": confusion_rate,
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["count", "confusion_rate"], ascending=False).head(top_k)
    return df.reset_index(drop=True)


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Compute confusion matrix with optional normalization.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        labels: List of label values (defaults to unique labels).
        normalize: If True, normalize by true class counts (row-wise).

    Returns:
        Confusion matrix (2D array).
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if not normalize:
        return cm

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        )
    normalized[row_sums.squeeze() == 0] = 0.0
    return normalized


def analyze_conifer_deciduous(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    genus_groups: dict[str, list[str]],
) -> dict[str, float]:
    """Compute F1 for conifers vs. deciduous trees.

    Args:
        y_true: True labels (encoded integers).
        y_pred: Predicted labels (encoded integers).
        class_labels: List of genus names (latin) in encoded order.
        genus_groups: Dict with keys 'conifer', 'deciduous' mapping to genus lists.

    Returns:
        Dict with keys:
        - 'conifer_f1': Weighted F1 for conifer genera
        - 'deciduous_f1': Weighted F1 for deciduous genera
        - 'conifer_n': Number of conifer samples
        - 'deciduous_n': Number of deciduous samples

    Raises:
        ValueError: If genus_groups missing required keys.
    """
    if "conifer" not in genus_groups or "deciduous" not in genus_groups:
        raise ValueError("genus_groups must include 'conifer' and 'deciduous'.")

    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    conifer_indices = [label_to_idx[g] for g in genus_groups["conifer"] if g in label_to_idx]
    deciduous_indices = [label_to_idx[g] for g in genus_groups["deciduous"] if g in label_to_idx]

    conifer_mask = np.isin(y_true, conifer_indices)
    deciduous_mask = np.isin(y_true, deciduous_indices)

    conifer_f1 = (
        f1_score(
            y_true[conifer_mask],
            y_pred[conifer_mask],
            labels=conifer_indices,
            average="weighted",
            zero_division="warn",
        )
        if conifer_mask.any()
        else 0.0
    )
    deciduous_f1 = (
        f1_score(
            y_true[deciduous_mask],
            y_pred[deciduous_mask],
            labels=deciduous_indices,
            average="weighted",
            zero_division="warn",
        )
        if deciduous_mask.any()
        else 0.0
    )

    return {
        "conifer_f1": float(conifer_f1),
        "deciduous_f1": float(deciduous_f1),
        "conifer_n": int(conifer_mask.sum()),
        "deciduous_n": int(deciduous_mask.sum()),
    }


def analyze_by_metadata(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata_col: pd.Series,
    bins: list[dict] | None = None,
    bin_labels: list[str] | None = None,
) -> pd.DataFrame:
    """Analyze accuracy by metadata column (plant_year, tree_type, CHM bins, etc.).

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        metadata_col: Metadata column (e.g., plant_year, tree_type).
        bins: Optional list of bin dicts with 'min_year'/'max_year' or single 'value'.
        bin_labels: Optional labels for bins (must match bins length).

    Returns:
        DataFrame with columns: bin_label, accuracy, f1_weighted, n_samples.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(metadata_col) != len(y_true):
        raise ValueError("metadata_col must align with y_true length.")

    rows: list[dict[str, object]] = []

    if bins is None:
        unique_values = pd.Series(metadata_col.dropna().unique()).tolist()
        for value in unique_values:
            mask = metadata_col == value
            if not mask.any():
                continue
            y_true_bin = y_true[mask.to_numpy()]
            y_pred_bin = y_pred[mask.to_numpy()]
            rows.append(
                {
                    "bin_label": str(value),
                    "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
                    "f1_weighted": float(
                        f1_score(
                            y_true_bin,
                            y_pred_bin,
                            average="weighted",
                            zero_division="warn",
                        )
                    ),
                    "n_samples": int(mask.sum()),
                }
            )
    else:
        if bin_labels is not None and len(bin_labels) != len(bins):
            raise ValueError("bin_labels must match bins length.")
        for idx, bin_def in enumerate(bins):
            label = bin_labels[idx] if bin_labels is not None else str(bin_def)
            if "value" in bin_def:
                mask = metadata_col == bin_def["value"]
            else:
                mask = pd.Series(True, index=metadata_col.index)
                min_year = bin_def.get("min_year", None)
                max_year = bin_def.get("max_year", None)
                if min_year is not None:
                    mask &= metadata_col >= min_year
                if max_year is not None:
                    mask &= metadata_col <= max_year
            if not mask.any():
                rows.append(
                    {
                        "bin_label": label,
                        "accuracy": 0.0,
                        "f1_weighted": 0.0,
                        "n_samples": 0,
                    }
                )
                continue
            y_true_bin = y_true[mask.to_numpy()]
            y_pred_bin = y_pred[mask.to_numpy()]
            rows.append(
                {
                    "bin_label": label,
                    "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
                    "f1_weighted": float(
                        f1_score(
                            y_true_bin,
                            y_pred_bin,
                            average="weighted",
                            zero_division="warn",
                        )
                    ),
                    "n_samples": int(mask.sum()),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("bin_label").reset_index(drop=True)


def bin_plant_years(
    plant_years: pd.Series,
    decade_bins: list[dict],
) -> pd.Series:
    """Bin plant years into decades.

    Args:
        plant_years: Series of plant years (can contain NaN).
        decade_bins: List of bin dicts with 'min_year'/'max_year' and 'label'.

    Returns:
        Series with decade labels (NaN for missing plant_year).
    """
    labels = [bin_def["label"] for bin_def in decade_bins]
    result = pd.Series(pd.NA, index=plant_years.index, dtype="object")

    for bin_def in decade_bins:
        label = bin_def["label"]
        mask = pd.Series(True, index=plant_years.index)
        if "min_year" in bin_def:
            mask &= plant_years >= bin_def["min_year"]
        if "max_year" in bin_def:
            mask &= plant_years <= bin_def["max_year"]
        result.loc[mask] = label

    result = result.astype(pd.CategoricalDtype(categories=labels, ordered=True))
    return result


def analyze_spatial_errors(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    block_ids: pd.Series,
    geometry_lookup: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Compute per-block accuracy and return spatial error map.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        block_ids: Series of block IDs for each sample.
        geometry_lookup: GeoDataFrame with block_id and geometry columns.

    Returns:
        GeoDataFrame with per-block accuracy, sample counts, and geometry.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(block_ids) != len(y_true):
        raise ValueError("block_ids must align with y_true length.")
    missing_cols = {"block_id", "geometry"} - set(geometry_lookup.columns)
    if missing_cols:
        raise ValueError("geometry_lookup must include 'block_id' and 'geometry' columns.")

    correctness = y_true == y_pred
    data = pd.DataFrame({"block_id": block_ids, "correct": correctness})
    grouped = data.groupby("block_id", dropna=False)
    summary = grouped["correct"].agg(
        accuracy="mean",
        n_samples="count",
        n_errors=lambda x: int((~x).sum()),
    )
    summary = summary.reset_index()

    merged = geometry_lookup.merge(summary, on="block_id", how="left")
    merged["accuracy"] = merged["accuracy"].fillna(0.0)
    merged["n_samples"] = merged["n_samples"].fillna(0).astype(int)
    merged["n_errors"] = merged["n_errors"].fillna(0).astype(int)
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=geometry_lookup.crs)


def analyze_species_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    species_col: pd.Series,
    genus_german_map: dict[str, str] | None = None,
    f1_threshold: float = 0.50,
) -> dict[str, pd.DataFrame]:
    """For genera with F1 below threshold, break down errors by species.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_labels: List of genus names (latin) in encoded order.
        species_col: Species latin names for each sample.
        genus_german_map: Optional mapping to German names.
        f1_threshold: Only analyze genera with F1 below this.

    Returns:
        Dict mapping genus name to a DataFrame of per-species metrics.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape.")
    if len(species_col) != len(y_true):
        raise ValueError("species_col must align with y_true length.")

    metrics_df = compute_per_class_metrics(y_true, y_pred, class_labels)
    low_f1 = metrics_df[metrics_df["f1_score"] < f1_threshold]["genus"].tolist()

    label_to_idx = {label: idx for idx, label in enumerate(class_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    results: dict[str, pd.DataFrame] = {}
    for genus in low_f1:
        genus_idx = label_to_idx[genus]
        mask = y_true == genus_idx
        if not mask.any():
            continue
        species_values = species_col.to_numpy()
        species_series = pd.Series(species_values[mask], index=species_col.index[mask])
        y_true_subset = y_true[mask]
        y_pred_subset = y_pred[mask]

        rows: list[dict[str, object]] = []
        for species_value in species_series.dropna().unique().tolist():
            species_mask = species_series == species_value
            if not species_mask.any():
                continue
            y_true_species = y_true_subset[species_mask.to_numpy()]
            y_pred_species = y_pred_subset[species_mask.to_numpy()]
            accuracy = float(accuracy_score(y_true_species, y_pred_species))
            f1_value = float(
                f1_score(
                    y_true_species,
                    y_pred_species,
                    labels=[genus_idx],
                    average="macro",
                    zero_division="warn",
                )
            )
            misclassified = y_pred_species[y_pred_species != genus_idx]
            if misclassified.size > 0:
                most_common = int(pd.Series(misclassified).mode().iloc[0])
                most_confused = idx_to_label[most_common]
                if genus_german_map is not None:
                    most_confused = genus_german_map.get(most_confused, most_confused)
            else:
                most_confused = "-"

            rows.append(
                {
                    "species": str(species_value),
                    "n_samples": int(species_mask.sum()),
                    "accuracy": accuracy,
                    "f1_score": f1_value,
                    "most_confused_with": most_confused,
                }
            )

        if rows:
            genus_label = str(genus)
            if genus_german_map is not None:
                genus_label = genus_german_map.get(genus_label, genus_label)
            results[genus_label] = pd.DataFrame(rows)

    return results


def fit_power_law(
    x_data: np.ndarray,
    y_data: np.ndarray,
    target_y: float | None = None,
) -> dict[str, float | None]:
    """Fit power-law curve y = a * x^b to fine-tuning data.

    Args:
        x_data: Sample sizes (independent variable).
        y_data: F1 scores (dependent variable).
        target_y: Optional target F1 to extrapolate required samples.

    Returns:
        Dict with keys: a, b, r_squared, extrapolated_x (None if target not provided or not achievable).
    """
    from scipy.optimize import curve_fit

    def power_law(x: np.ndarray, a: float, b: float) -> np.ndarray:
        return a * np.power(x, b)

    # Fit power law
    try:
        params, _ = curve_fit(power_law, x_data, y_data, p0=[0.1, 0.5], maxfev=10000)
        a, b = params
    except RuntimeError:
        # Fitting failed
        return {"a": 0.0, "b": 0.0, "r_squared": 0.0, "extrapolated_x": None}

    # Compute R²
    y_pred = power_law(x_data, a, b)
    ss_res = np.sum((y_data - y_pred) ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Extrapolate to target if provided
    extrapolated_x = None
    if target_y is not None and b != 0 and a > 0 and target_y > 0:
        # Solve: target_y = a * x^b => x = (target_y / a)^(1/b)
        extrapolated_x = float(np.power(target_y / a, 1 / b))
        # Check if extrapolation is reasonable (within 10x of max x_data)
        if extrapolated_x > 10 * x_data.max() or extrapolated_x < x_data.min():
            extrapolated_x = None

    return {
        "a": float(a),
        "b": float(b),
        "r_squared": float(r_squared),
        "extrapolated_x": extrapolated_x,
    }
