"""Visualization utilities for Phase 3 experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.figure import Figure

DEFAULT_DPI = 300
DEFAULT_FORMAT = "png"
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_DOUBLE = (14, 6)
FIGSIZE_TABLE = (10, 4)
FIGSIZE_LARGE = (12, 8)

COLORMAP_SEQUENTIAL = "viridis"
COLORMAP_DIVERGING = "RdYlGn"
COLORMAP_CATEGORICAL = "tab10"


def _get_palette() -> list[tuple[float, float, float]]:
    try:
        import seaborn as sns  # pyright: ignore[reportMissingImports,reportMissingModuleSource]

        sns.set_theme(style="whitegrid", context="paper")
        return list(sns.color_palette(COLORMAP_CATEGORICAL))
    except ModuleNotFoundError:
        pass
    cmap = plt.get_cmap(COLORMAP_CATEGORICAL)
    return [cmap(i)[:3] for i in range(cmap.N)]


def map_to_german_names(labels: list[str], german_map: dict[str, str] | None) -> list[str]:
    """Map latin names to German names if map provided.

    Args:
        labels: List of latin genus names.
        german_map: Optional mapping latin → German.

    Returns:
        German names if map provided, else original labels.
    """
    if german_map is None:
        return labels
    return [german_map.get(label, label) for label in labels]


def save_figure(fig: Figure, output_path: Path, dpi: int = DEFAULT_DPI) -> None:
    """Save figure with standard settings.

    Args:
        fig: Matplotlib figure.
        output_path: Path to save.
        dpi: Resolution (default: 300).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_comparison(
    results: pd.DataFrame,
    title: str,
    output_path: Path,
    highlight_decision: str | None = None,
) -> None:
    """Generic bar chart for ablation study F1 comparison."""
    data = results.copy()
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    bars = ax.bar(data["variant"], data["val_f1_mean"], color="steelblue")
    if "val_f1_std" in data.columns:
        ax.errorbar(
            data["variant"],
            data["val_f1_mean"],
            yerr=data["val_f1_std"],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
    if highlight_decision:
        for bar, label in zip(bars, data["variant"], strict=False):
            if label == highlight_decision:
                bar.set_edgecolor("black")
                bar.set_linewidth(2)
    ax.set_ylabel("Validation F1")
    ax.set_xlabel("Variant")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    save_figure(fig, output_path)


def plot_per_genus_comparison(
    results_a: dict[str, float],
    results_b: dict[str, float],
    label_a: str,
    label_b: str,
    genus_labels: list[str],
    output_path: Path,
) -> None:
    """Grouped bar chart comparing per-genus F1 between two variants."""
    values_a = [results_a.get(genus, 0.0) for genus in genus_labels]
    values_b = [results_b.get(genus, 0.0) for genus in genus_labels]
    x = np.arange(len(genus_labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    ax.bar(x - width / 2, values_a, width=width, label=label_a, color="#4C72B0")
    ax.bar(x + width / 2, values_b, width=width, label=label_b, color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(genus_labels, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Genus Comparison")
    ax.legend()
    save_figure(fig, output_path)


def plot_sample_loss_by_genus(
    baseline_counts: pd.Series,
    filtered_counts: pd.Series,
    genus_labels: list[str],
    output_path: Path,
) -> None:
    """Stacked bar: samples retained vs. removed per genus."""
    base = baseline_counts.reindex(genus_labels).fillna(0).astype(float)
    filt = filtered_counts.reindex(genus_labels).fillna(0).astype(float)
    removed = (base - filt).clip(lower=0)
    x = np.arange(len(genus_labels))

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    ax.bar(x, filt, label="Retained", color="#4C72B0")
    ax.bar(x, removed, bottom=filt, label="Removed", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels(genus_labels, rotation=45, ha="right")
    ax.set_ylabel("Samples")
    ax.set_title("Sample Loss by Genus")
    ax.legend()
    save_figure(fig, output_path)


def plot_outlier_distribution(
    severity_by_genus: pd.DataFrame,
    genus_labels: list[str],
    output_path: Path,
) -> None:
    """Stacked bar: outlier severity distribution by genus."""
    data = severity_by_genus.reindex(genus_labels).fillna(0)
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    bottoms = np.zeros(len(genus_labels))
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#8172B2"]
    for idx, col in enumerate(data.columns):
        values = data[col].to_numpy()
        ax.bar(genus_labels, values, bottom=bottoms, label=col, color=colors[idx % len(colors)])
        bottoms += values
    ax.set_ylabel("Count")
    ax.set_title("Outlier Severity Distribution")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend()
    save_figure(fig, output_path)


def plot_tradeoff_curve(
    points: list[dict[str, Any]],
    x_label: str,
    y_label: str,
    output_path: Path,
) -> None:
    """Scatter/line plot: F1 vs. data retained."""
    df = pd.DataFrame(points)
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(df["samples_retained"], df["f1"], marker="o", color="steelblue")
    for idx, name in enumerate(df["name"]):
        ax.text(df["samples_retained"].iloc[idx], df["f1"].iloc[idx], str(name), fontsize=8)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title("Trade-off Curve")
    save_figure(fig, output_path)


def plot_pareto_curve(
    results: pd.DataFrame,
    selected_k: int,
    output_path: Path,
) -> None:
    """F1 vs. feature count with knee point and selection marker."""
    data = results.sort_values("n_features")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(data["n_features"], data["val_f1_mean"], marker="o", color="steelblue")
    if "val_f1_std" in data.columns:
        ax.errorbar(
            data["n_features"],
            data["val_f1_mean"],
            yerr=data["val_f1_std"],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
    ax.axvline(selected_k, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Number of Features")
    ax.set_ylabel("Validation F1")
    ax.set_title("Pareto Curve")
    save_figure(fig, output_path)


def plot_feature_group_contribution(
    importance_df: pd.DataFrame,
    group_mapping: dict[str, str],
    output_path: Path,
) -> None:
    """Pie/bar chart showing feature importance by group."""
    data = importance_df.copy()
    data["group"] = data["feature"].map(lambda feature: group_mapping.get(feature, "Other"))
    grouped = cast(pd.Series, data.groupby("group")["importance"].sum()).sort_values(
        ascending=False
    )
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.bar(grouped.index, grouped.to_numpy(dtype=float), color="steelblue")
    ax.set_ylabel("Total Importance")
    ax.set_title("Feature Group Contribution")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    save_figure(fig, output_path)


def plot_algorithm_comparison(
    results: pd.DataFrame,
    output_path: Path,
    metric: str = "val_f1_mean",
    include_baselines: bool = True,
    highlight_champions: list[str] | None = None,
) -> None:
    """Bar chart: Algorithm F1 comparison with error bars.

    Args:
        results: DataFrame with algorithm metrics.
        output_path: Path to save figure.
        metric: Metric column to plot.
        include_baselines: Whether to include baseline rows.
        highlight_champions: Optional list of algorithms to highlight.
    """
    data = results.copy()
    if not include_baselines:
        data = data[data["type"] != "baseline"]
    data = cast(pd.DataFrame, data).sort_values(by=[metric], ascending=False)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    palette_values = _get_palette()
    palette = {
        "ml": palette_values[0],
        "nn": palette_values[1],
        "baseline": palette_values[2],
    }
    colors = [palette.get(t, palette_values[3]) for t in data["type"]]
    bars = ax.bar(data["algorithm"], data[metric], color=colors)
    if "val_f1_std" in data.columns:
        ax.errorbar(
            data["algorithm"],
            data[metric],
            yerr=data["val_f1_std"],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel(metric)
    ax.set_xlabel("Algorithm")
    ax.set_title("Algorithm Comparison")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")

    if highlight_champions:
        for bar, name in zip(bars, data["algorithm"], strict=False):
            if name in highlight_champions:
                bar.set_edgecolor("black")
                bar.set_linewidth(2)

    save_figure(fig, output_path)


def plot_performance_ladder(
    results: pd.DataFrame,
    output_path: Path,
    include_ci: bool = False,
    highlight_champions: list[str] | None = None,
) -> None:
    """Sorted ranking plot: Baselines → Models → Champions.

    Args:
        results: DataFrame with name, f1_score, and category columns.
        output_path: Path to save figure.
        include_ci: Whether to use CI columns for error bars.
        highlight_champions: Optional list of model names to highlight.
    """
    data = results.sort_values("f1_score", ascending=True)
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    palette_values = _get_palette()
    palette = {
        "baseline": palette_values[2],
        "model": palette_values[0],
        "champion": palette_values[1],
    }
    colors = [palette.get(cat, palette_values[3]) for cat in data["category"]]
    y_positions = np.arange(len(data))
    ax.barh(y_positions, data["f1_score"], color=colors)

    if include_ci and {"f1_ci_lower", "f1_ci_upper"}.issubset(data.columns):
        lower = data["f1_score"] - data["f1_ci_lower"]
        upper = data["f1_ci_upper"] - data["f1_score"]
        ax.errorbar(data["f1_score"], y_positions, xerr=[lower, upper], fmt="none", ecolor="black")
    elif "f1_std" in data.columns:
        ax.errorbar(data["f1_score"], y_positions, xerr=data["f1_std"], fmt="none", ecolor="black")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["name"])
    ax.set_xlabel("F1 Score")
    ax.set_title("Performance Ladder")

    for idx, (score, name) in enumerate(zip(data["f1_score"], data["name"], strict=False)):
        ax.text(score + 0.005, idx, f"{score:.3f}", va="center", fontsize=8)
        if highlight_champions and name in highlight_champions:
            ax.get_yticklabels()[idx].set_fontweight("bold")

    save_figure(fig, output_path)


def plot_train_val_gap(
    results: pd.DataFrame,
    output_path: Path,
    threshold: float = 0.30,
) -> None:
    """Bar chart: Train-Val gap comparison across algorithms.

    Args:
        results: DataFrame with algorithm and train_val_gap columns.
        output_path: Path to save figure.
        threshold: Highlight gaps above this threshold.
    """
    data = results.sort_values("train_val_gap", ascending=False)
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    colors = ["red" if gap > threshold else "steelblue" for gap in data["train_val_gap"]]
    ax.bar(data["algorithm"], data["train_val_gap"], color=colors)
    ax.axhline(threshold, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Train-Val Gap")
    ax.set_xlabel("Algorithm")
    ax.set_title("Train-Val Gap Comparison")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    save_figure(fig, output_path)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_labels: list[str],
    output_path: Path,
    normalize: bool = True,
    use_german_names: bool = True,
    german_name_map: dict[str, str] | None = None,
) -> None:
    """Confusion matrix heatmap with German labels.

    Args:
        cm: Confusion matrix.
        class_labels: List of latin genus names.
        output_path: Path to save figure.
        normalize: If True, normalize row-wise.
        use_german_names: If True, map labels via german_name_map.
        german_name_map: Mapping latin → German names.
    """
    labels = class_labels
    if use_german_names:
        labels = map_to_german_names(class_labels, german_name_map)

    if normalize:
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = np.divide(
                cm_plot, row_sums, out=np.zeros_like(cm_plot, dtype=float), where=row_sums != 0
            )
        cm_plot[row_sums.squeeze() == 0] = 0.0
    else:
        cm_plot = cm

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
    try:
        import seaborn as sns  # pyright: ignore[reportMissingImports,reportMissingModuleSource]

        sns.set_theme(style="whitegrid", context="paper")
        sns.heatmap(
            cm_plot,
            ax=ax,
            cmap=COLORMAP_SEQUENTIAL,
            xticklabels=labels,
            yticklabels=labels,
            annot=True,
            fmt=".2f" if normalize else "d",
            cbar_kws={"label": "Anteil" if normalize else "Count"},
        )
    except ModuleNotFoundError:
        im = ax.imshow(cm_plot, cmap=COLORMAP_SEQUENTIAL)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, label="Anteil" if normalize else "Count")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.set_title("Confusion Matrix")
    save_figure(fig, output_path)


def plot_confusion_comparison(
    cm_source: np.ndarray,
    cm_target: np.ndarray,
    class_labels: list[str],
    source_name: str,
    target_name: str,
    output_path: Path,
    normalize: bool = True,
    use_german_names: bool = True,
    german_name_map: dict[str, str] | None = None,
) -> None:
    """Side-by-side confusion matrix comparison."""
    labels = class_labels
    if use_german_names:
        labels = map_to_german_names(class_labels, german_name_map)

    def _normalize(cm: np.ndarray) -> np.ndarray:
        if not normalize:
            return cm
        cm_plot = cm.astype(float)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            cm_plot = np.divide(
                cm_plot, row_sums, out=np.zeros_like(cm_plot, dtype=float), where=row_sums != 0
            )
        cm_plot[row_sums.squeeze() == 0] = 0.0
        return cm_plot

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DOUBLE)
    for ax, cm, title in zip(
        axes,
        [_normalize(cm_source), _normalize(cm_target)],
        [source_name, target_name],
        strict=False,
    ):
        im = ax.imshow(cm, cmap=COLORMAP_SEQUENTIAL)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.8)
    save_figure(fig, output_path)


def plot_per_genus_f1(
    f1_scores: dict[str, float],
    output_path: Path,
    german_name_map: dict[str, str] | None = None,
    ci_data: dict[str, tuple[float, float]] | None = None,
    threshold: float | None = 0.50,
) -> None:
    """Bar chart: Per-genus F1 (sorted) with optional CIs.

    Args:
        f1_scores: Mapping genus → F1 score.
        output_path: Path to save figure.
        german_name_map: Optional mapping latin → German names.
        ci_data: Optional mapping genus → (ci_lower, ci_upper).
        threshold: Optional threshold for highlighting low scores.
    """
    data = pd.DataFrame(
        {"genus": list(f1_scores.keys()), "f1_score": list(f1_scores.values())}
    ).sort_values("f1_score", ascending=True)
    data["label"] = map_to_german_names(data["genus"].tolist(), german_name_map)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    colors = []
    for score in data["f1_score"]:
        if threshold is not None and score < threshold:
            colors.append("firebrick")
        else:
            colors.append("steelblue")
    y_positions = np.arange(len(data))
    ax.barh(y_positions, data["f1_score"], color=colors)

    if ci_data:
        lower = []
        upper = []
        for idx, genus in enumerate(data["genus"]):
            ci_lower, ci_upper = ci_data.get(genus, (0.0, 0.0))
            f1_value = float(data.iloc[idx]["f1_score"])
            lower.append(f1_value - ci_lower)
            upper.append(ci_upper - f1_value)
        ax.errorbar(
            data["f1_score"],
            y_positions,
            xerr=[lower, upper],
            fmt="none",
            ecolor="black",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["label"])
    ax.set_xlabel("F1 Score")
    ax.set_title("Per-Genus F1")

    for idx, score in enumerate(data["f1_score"]):
        ax.text(score + 0.005, idx, f"{score:.3f}", va="center", fontsize=8)

    if threshold is not None:
        ax.axvline(threshold, color="gray", linestyle="--", linewidth=1)

    save_figure(fig, output_path)


def plot_per_genus_metrics_table(
    metrics_df: pd.DataFrame,
    output_path: Path,
    german_name_map: dict[str, str] | None = None,
) -> None:
    """Render per-genus metrics as publication-quality table.

    Args:
        metrics_df: DataFrame with genus, precision, recall, f1_score, support.
        output_path: Path to save figure.
        german_name_map: Optional mapping latin → German names.
    """
    df = metrics_df.copy()
    if german_name_map is not None:
        df["genus"] = map_to_german_names(df["genus"].tolist(), german_name_map)
    df = df.sort_values("f1_score", ascending=False)
    df_display = df.copy()
    df_display["precision"] = df_display["precision"].map(lambda x: f"{x:.3f}")
    df_display["recall"] = df_display["recall"].map(lambda x: f"{x:.3f}")
    df_display["f1_score"] = df_display["f1_score"].map(lambda x: f"{x:.3f}")
    df_display["support"] = df_display["support"].astype(int).astype(str)

    fig, ax = plt.subplots(figsize=FIGSIZE_TABLE)
    ax.axis("off")
    table = ax.table(
        cellText=df_display.values.tolist(),
        colLabels=df_display.columns.tolist(),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)

    f1_values = df["f1_score"].to_numpy()
    norm = Normalize(vmin=f1_values.min(), vmax=f1_values.max())
    cmap = plt.get_cmap(COLORMAP_DIVERGING)
    for row_idx, value in enumerate(f1_values, start=1):
        color = cmap(norm(value))
        for col_idx in range(len(df_display.columns)):
            table[(row_idx, col_idx)].set_facecolor(color)

    save_figure(fig, output_path)


def plot_confusion_pairs(
    pairs_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 10,
) -> None:
    """Bar chart: Top-10 confused genus pairs.

    Args:
        pairs_df: DataFrame from analyze_worst_confused_pairs.
        output_path: Path to save figure.
        top_k: Number of pairs to plot.
    """
    data = pairs_df.head(top_k).copy()
    data["pair"] = data["true_genus"] + " → " + data["pred_genus"]
    data = data.sort_values("count", ascending=True)

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    y_positions = np.arange(len(data))
    ax.barh(y_positions, data["count"], color="steelblue")
    ax.set_yticks(y_positions)
    ax.set_yticklabels(data["pair"])
    ax.set_xlabel("Count")
    ax.set_title("Top Confused Pairs")

    for idx, (count, rate) in enumerate(zip(data["count"], data["confusion_rate"], strict=False)):
        ax.text(count + 0.5, idx, f"{rate:.1%}", va="center", fontsize=8)

    save_figure(fig, output_path)


def plot_conifer_deciduous_comparison(
    conifer_f1: float,
    deciduous_f1: float,
    output_path: Path,
    conifer_n: int | None = None,
    deciduous_n: int | None = None,
) -> None:
    """Bar chart: Conifer vs. Deciduous F1 comparison.

    Args:
        conifer_f1: F1 for conifer group.
        deciduous_f1: F1 for deciduous group.
        output_path: Path to save figure.
        conifer_n: Optional sample count for conifers.
        deciduous_n: Optional sample count for deciduous.
    """
    labels = ["Nadelbäume", "Laubbäume"]
    values = [conifer_f1, deciduous_f1]
    if conifer_n is not None and deciduous_n is not None:
        labels = [f"Nadelbäume (n={conifer_n})", f"Laubbäume (n={deciduous_n})"]

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.bar(labels, values, color=["#4C72B0", "#55A868"])
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    ax.set_ylim(0, min(1.0, max(values) + 0.1))
    ax.set_ylabel("F1 Score")
    ax.set_title("Conifer vs. Deciduous")
    save_figure(fig, output_path)


def plot_metadata_impact(
    metadata_df: pd.DataFrame,
    output_path: Path,
    x_col: str = "bin_label",
    y_col: str = "accuracy",
    title: str = "Accuracy by Metadata",
    xlabel: str = "Metadata Bin",
    ylabel: str = "Accuracy",
) -> None:
    """Generic bar chart for metadata impact analysis.

    Args:
        metadata_df: DataFrame with bin_label, accuracy, and n_samples.
        output_path: Path to save figure.
        x_col: Column name for x-axis.
        y_col: Column name for y-axis.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.bar(metadata_df[x_col], metadata_df[y_col], color="steelblue")
    for idx, n_samples in enumerate(metadata_df["n_samples"]):
        y_value = float(metadata_df.iloc[idx][y_col])
        ax.text(idx, y_value + 0.01, f"n={n_samples}", ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    save_figure(fig, output_path)


def plot_tree_type_comparison(
    tree_type_metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    """Grouped bar: Tree type comparison per genus."""
    data = tree_type_metrics.copy()
    genera = sorted(data["genus"].unique())
    types = sorted(data["tree_type"].unique())
    x = np.arange(len(genera))
    width = 0.35 if len(types) <= 2 else 0.8 / len(types)

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    for idx, tree_type in enumerate(types):
        subset = data[data["tree_type"] == tree_type]
        values = [subset.loc[subset["genus"] == genus, "f1"].mean() for genus in genera]
        ax.bar(x + idx * width, values, width=width, label=tree_type)
    ax.set_xticks(x + width * (len(types) - 1) / 2)
    ax.set_xticklabels(genera, rotation=45, ha="right")
    ax.set_ylabel("F1 Score")
    ax.set_title("Tree Type Comparison")
    ax.legend()
    save_figure(fig, output_path)


def plot_plant_year_impact(
    pred_df: pd.DataFrame,
    decade_bins: list[int] | list[dict[str, int | str | None]],
    output_path: Path,
) -> None:
    """Line/bar chart: prediction accuracy by plant year decade."""
    df = pred_df.copy()
    if "plant_year" not in df.columns or "correct" not in df.columns:
        raise ValueError("pred_df must contain plant_year and correct columns.")

    if decade_bins and isinstance(decade_bins[0], dict):
        bins = cast(list[dict[str, int | str | None]], decade_bins)
        labels = []
        ranges = []
        for bin_item in bins:
            min_year_value = bin_item.get("min_year")
            max_year_value = bin_item.get("max_year")
            min_year = int(min_year_value) if min_year_value is not None else None
            max_year = int(max_year_value) if max_year_value is not None else None
            label_value = bin_item.get("label")
            label: str | None = None
            if label_value is not None:
                label = str(label_value)
            if label is None:
                if min_year is None:
                    label = f"<= {max_year}"
                elif max_year is None:
                    label = f">= {min_year}"
                else:
                    label = f"{min_year}-{max_year}"
            labels.append(label)
            ranges.append((min_year, max_year))
    else:
        boundaries = sorted(int(x) for x in cast(list[int], decade_bins))
        labels = []
        ranges = []
        for idx, start in enumerate(boundaries):
            end = boundaries[idx + 1] - 1 if idx + 1 < len(boundaries) else None
            label = f"{start}-{end}" if end else f">= {start}"
            labels.append(label)
            ranges.append((start, end))

    rows = []
    for label, (min_year, max_year) in zip(labels, ranges, strict=False):
        mask = pd.Series(True, index=df.index)
        if min_year is not None:
            mask &= df["plant_year"] >= min_year
        if max_year is not None:
            mask &= df["plant_year"] <= max_year
        subset = df[mask]
        if subset.empty:
            continue
        rows.append(
            {
                "bin_label": label,
                "accuracy": float(subset["correct"].mean()),
                "n_samples": int(subset.shape[0]),
            }
        )
    plot_metadata_impact(pd.DataFrame(rows), output_path, title="Accuracy by Plant Year")


def plot_species_breakdown(
    species_metrics: pd.DataFrame,
    problematic_genera: list[str],
    output_path: Path,
) -> None:
    """Per-species F1 bar charts for problematic genera."""
    data = species_metrics.copy()
    if data.empty:
        raise ValueError("species_metrics is empty.")
    data = data[data["genus"].isin(problematic_genera)]
    if data.empty:
        raise ValueError("No matching genera in species_metrics.")

    genera = sorted(cast(pd.Series, data["genus"]).unique())
    fig, axes = plt.subplots(len(genera), 1, figsize=(FIGSIZE_DOUBLE[0], 4 * len(genera)))
    if len(genera) == 1:
        axes = [axes]
    for ax, genus in zip(axes, genera, strict=False):
        subset = cast(pd.DataFrame, data[data["genus"] == genus]).sort_values(by="f1_score")
        ax.barh(subset["species"], subset["f1_score"], color="steelblue")
        ax.set_title(f"Species Breakdown: {genus}")
        ax.set_xlabel("F1 Score")
    save_figure(fig, output_path)


def plot_chm_impact(
    pred_df: pd.DataFrame,
    output_path: Path,
    chm_col: str = "CHM_1m",
) -> None:
    """Binned accuracy plot: prediction accuracy as a function of CHM value."""
    df = pred_df.copy()
    if chm_col not in df.columns or "correct" not in df.columns:
        raise ValueError("pred_df must contain CHM column and correct column.")

    df = df[df[chm_col].notna()].copy()
    bins = np.linspace(df[chm_col].min(), df[chm_col].max(), 11)
    df["bin"] = pd.cut(df[chm_col], bins=bins, include_lowest=True)
    grouped = df.groupby("bin")["correct"].agg(["mean", "count"]).reset_index()

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(grouped["bin"].astype(str), grouped["mean"], marker="o", color="steelblue")
    ax.set_xlabel("CHM Bin")
    ax.set_ylabel("Accuracy")
    ax.set_title("CHM Impact on Accuracy")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    save_figure(fig, output_path)


def plot_per_genus_transfer(
    transfer_data: dict[str, dict],
    genus_labels: list[str],
    output_path: Path,
) -> None:
    """Per-genus transfer robustness visualization."""
    df = (
        pd.DataFrame.from_dict(transfer_data, orient="index")
        .reset_index()
        .rename(columns={"index": "genus"})
    )
    df = df.set_index("genus").reindex(genus_labels).reset_index()

    label_colors = {"robust": "#4C72B0", "medium": "#DD8452", "poor": "#C44E52"}
    colors = [label_colors.get(lbl, "gray") for lbl in df["label"]]

    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    ax.bar(df["genus"], df["relative_drop"], color=colors)
    ax.set_ylabel("Relative Drop")
    ax.set_title("Per-Genus Transfer Robustness")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    save_figure(fig, output_path)


def plot_transfer_conifer_deciduous(
    transfer_data: dict[str, dict],
    conifer_genera: set[str],
    output_path: Path,
) -> None:
    """Aggregate transfer performance: conifer vs. deciduous."""
    conifer_scores = []
    deciduous_scores = []
    for genus, stats in transfer_data.items():
        target = stats.get("target_f1", 0.0)
        if genus in conifer_genera:
            conifer_scores.append(target)
        else:
            deciduous_scores.append(target)

    conifer_f1 = float(np.mean(conifer_scores)) if conifer_scores else 0.0
    deciduous_f1 = float(np.mean(deciduous_scores)) if deciduous_scores else 0.0
    plot_conifer_deciduous_comparison(conifer_f1, deciduous_f1, output_path)


def plot_finetuning_curve(
    results: list[dict],
    baselines: dict[str, float],
    output_path: Path,
) -> None:
    """Sample efficiency curve with zero-shot + from-scratch baselines."""
    df = pd.DataFrame(results).sort_values("fraction")
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(df["fraction"], df["f1_score"], marker="o", label="Fine-tune")
    for name, value in baselines.items():
        ax.axhline(value, linestyle="--", linewidth=1, label=name)
    ax.set_xlabel("Leipzig Fraction")
    ax.set_ylabel("F1 Score")
    ax.set_title("Fine-tuning Curve")
    ax.legend()
    save_figure(fig, output_path)


def plot_finetuning_ml_vs_nn(
    ml_results: list[dict],
    nn_results: list[dict],
    output_path: Path,
) -> None:
    """Side-by-side ML vs. NN fine-tuning curves."""
    ml_df = pd.DataFrame(ml_results).sort_values("fraction")
    nn_df = pd.DataFrame(nn_results).sort_values("fraction")

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(ml_df["fraction"], ml_df["f1_score"], marker="o", label="ML")
    ax.plot(nn_df["fraction"], nn_df["f1_score"], marker="o", label="NN")
    ax.set_xlabel("Leipzig Fraction")
    ax.set_ylabel("F1 Score")
    ax.set_title("Fine-tuning ML vs. NN")
    ax.legend()
    save_figure(fig, output_path)


def plot_finetuning_per_genus_recovery(
    per_genus_by_fraction: dict[float, dict[str, float]],
    genus_labels: list[str],
    output_path: Path,
) -> None:
    """Heatmap: per-genus F1 at each fine-tuning fraction."""
    fractions = sorted(per_genus_by_fraction.keys())
    data = np.array(
        [[per_genus_by_fraction[f].get(genus, 0.0) for genus in genus_labels] for f in fractions]
    )

    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
    im = ax.imshow(data, cmap=COLORMAP_SEQUENTIAL, aspect="auto")
    ax.set_xticks(np.arange(len(genus_labels)))
    ax.set_yticks(np.arange(len(fractions)))
    ax.set_xticklabels(genus_labels, rotation=45, ha="right")
    ax.set_yticklabels([str(f) for f in fractions])
    ax.set_xlabel("Genus")
    ax.set_ylabel("Fraction")
    ax.set_title("Per-Genus Recovery")
    fig.colorbar(im, ax=ax, label="F1 Score")
    save_figure(fig, output_path)


def plot_significance_matrix(
    p_values: pd.DataFrame,
    output_path: Path,
) -> None:
    """Heatmap of McNemar test p-values across comparisons."""
    fig, ax = plt.subplots(figsize=FIGSIZE_TABLE)
    im = ax.imshow(p_values.values, cmap="viridis_r", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(p_values.shape[1]))
    ax.set_yticks(np.arange(p_values.shape[0]))
    ax.set_xticklabels(p_values.columns, rotation=45, ha="right")
    ax.set_yticklabels(p_values.index)
    ax.set_title("McNemar Significance Matrix")
    fig.colorbar(im, ax=ax, label="p-value")
    save_figure(fig, output_path)


def plot_spatial_error_map(
    error_gdf: gpd.GeoDataFrame,
    output_path: Path,
    accuracy_col: str = "accuracy",
    city_name: str = "Berlin",
    cmap: str = COLORMAP_DIVERGING,
) -> None:
    """Choropleth map: Per-block accuracy.

    Args:
        error_gdf: GeoDataFrame with geometry and accuracy column.
        output_path: Path to save figure.
        accuracy_col: Column name to visualize.
        city_name: City name for title.
        cmap: Colormap name.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_LARGE)
    error_gdf.plot(column=accuracy_col, cmap=cmap, legend=True, ax=ax)
    ax.set_title(f"Classification Accuracy by Block - {city_name}")
    ax.set_axis_off()
    ax.set_aspect("equal")
    save_figure(fig, output_path)


def plot_misclassification_sankey(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_labels: list[str],
    output_path: Path,
    german_name_map: dict[str, str] | None = None,
    max_flows: int = 20,
) -> None:
    """Sankey diagram: True genus → Predicted genus (errors only).

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        class_labels: List of latin genus names.
        output_path: Path to save figure.
        german_name_map: Optional mapping latin → German names.
        max_flows: Maximum number of flows to show.
    """
    try:
        import plotly.graph_objects as go  # pyright: ignore[reportMissingImports]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "plotly is required for Sankey visualization; install plotly + kaleido."
        ) from exc

    mis_mask = y_true != y_pred
    if not mis_mask.any():
        raise ValueError("No misclassifications available for Sankey diagram.")

    true_labels = np.array(class_labels)[y_true[mis_mask]]
    pred_labels = np.array(class_labels)[y_pred[mis_mask]]
    flow_df = (
        pd.DataFrame({"true": true_labels, "pred": pred_labels})
        .value_counts()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(max_flows)
    )

    labels = sorted(set(flow_df["true"]).union(set(flow_df["pred"])))
    mapped = map_to_german_names(labels, german_name_map)
    label_map = {label: idx for idx, label in enumerate(labels)}

    sources = [label_map[val] for val in flow_df["true"]]
    targets = [label_map[val] for val in flow_df["pred"]]
    values = flow_df["count"].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                node={"label": mapped},
                link={"source": sources, "target": targets, "value": values},
            )
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(output_path), format=DEFAULT_FORMAT, scale=2)


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: Path,
    top_k: int = 20,
    importance_col: str = "importance",
    feature_col: str = "feature",
    method: str = "gain",
) -> None:
    """Bar chart: Top-k feature importance.

    Args:
        importance_df: DataFrame with feature and importance columns.
        output_path: Path to save figure.
        top_k: Number of features to show.
        importance_col: Column name for importance values.
        feature_col: Column name for feature names.
        method: Importance method (for title).
    """
    data = importance_df.sort_values(importance_col, ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=FIGSIZE_DOUBLE)
    ax.barh(data[feature_col], data[importance_col], color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top-{top_k} Features ({method.capitalize()} Importance)")
    ax.invert_yaxis()
    save_figure(fig, output_path)


def plot_optuna_history(
    study: Any,
    output_path: Path,
) -> None:
    """Optimization history: F1 vs. trial number.

    Args:
        study: Optuna study object.
        output_path: Path to save figure.
    """
    trials = [t.value for t in study.trials if t.value is not None]
    if not trials:
        raise ValueError("Study contains no completed trials.")

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax.plot(range(1, len(trials) + 1), trials, marker="o", linewidth=1)
    best_idx = int(np.argmax(trials))
    best_value = trials[best_idx]
    ax.axhline(best_value, color="gray", linestyle="--", linewidth=1)
    ax.scatter(best_idx + 1, best_value, color="red", zorder=5)
    ax.set_xlabel("Trial")
    ax.set_ylabel("Objective Value")
    ax.set_title("Optuna Optimization History")

    if len(trials) >= 10:
        rolling = pd.Series(trials).rolling(window=10, min_periods=1).mean()
        ax.plot(range(1, len(trials) + 1), rolling, color="black", linewidth=1, alpha=0.7)

    save_figure(fig, output_path)
