"""Generate publication-ready report plots from paper/data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from style import CITY_COLORS, PALETTE, apply_report_style

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
OUTPUTS_DIR = ROOT.parent / "outputs" / "phase_3_experiments" / "metadata"

GROUP_MEMBERS = {
    "Group 1": {"PICEA", "PINUS"},
    "Group 9": {"CRATAEGUS", "SORBUS"},
    "Group 11": {"POPULUS", "ROBINIA"},
    "Group 14": {"ACER", "BETULA", "FRAXINUS", "PRUNUS"},
}
FINAL_SINGLE_CLASSES = {
    "AESCULUS",
    "AILANTHUS",
    "ALNUS",
    "CARPINUS",
    "CORYLUS",
    "FAGUS",
    "MALUS",
    "PLATANUS",
    "PYRUS",
    "QUERCUS",
    "SALIX",
    "TILIA",
    "ULMUS",
}

MONTH_MAP = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def _read_matrix_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    first = df.columns[0]
    return df.rename(columns={first: "label"}).set_index("label")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_matrix_csv(matrix: pd.DataFrame, path: Path) -> None:
    out = matrix.copy()
    out.index.name = "label"
    out.reset_index().to_csv(path, index=False)


def _ordered_labels_by_berlin_f1(candidates: list[str]) -> list[str]:
    path = OUTPUTS_DIR / "berlin_evaluation.json"
    if not path.exists():
        return candidates

    data = _read_json(path)
    f1_by_label = {row["genus"]: row["f1_score"] for row in data.get("per_class", [])}
    return sorted(candidates, key=lambda label: f1_by_label.get(label, -1.0), reverse=True)


def _feature_label(raw: str) -> str:
    if "_" not in raw:
        return raw
    token, month = raw.rsplit("_", maxsplit=1)
    if not month.isdigit():
        return raw
    month_name = MONTH_MAP.get(int(month), month)
    return f"{token} ({month_name})"


def _feature_type(raw: str) -> str:
    token = raw.split("_", maxsplit=1)[0]
    if token.startswith("B") and token[1:].replace("A", "").isdigit():
        return "Spectral band"
    return "Vegetation index"


def _build_berlin_n_for_final_classes() -> dict[str, int]:
    df = pd.read_csv(DATA_DIR / "genus_distribution_filtered.csv")
    n_by_genus = {row.genus: int(row.berlin_count) for row in df.itertuples(index=False)}

    out: dict[str, int] = {}
    for genus in FINAL_SINGLE_CLASSES:
        out[genus] = n_by_genus.get(genus, 0)
    for group_name, members in GROUP_MEMBERS.items():
        out[group_name] = int(sum(n_by_genus.get(g, 0) for g in members))
    return out


def _annotated_heatmap(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    cbar_label: str,
    fmt: str = ".2f",
    text_size: float = 7.0,
    mask_upper: bool = False,
    hide_diagonal: bool = False,
    suppress_leq: float | None = None,
    norm: mcolors.Normalize | None = None,
) -> None:
    values = matrix.to_numpy(dtype=float)
    mask = np.zeros_like(values, dtype=bool)
    if mask_upper:
        mask |= np.triu(np.ones_like(values, dtype=bool), k=1)
    if hide_diagonal:
        mask |= np.eye(values.shape[0], dtype=bool)

    values_masked = np.ma.array(values, mask=mask)
    if norm is not None:
        im = ax.imshow(values_masked, cmap=cmap, aspect="equal", norm=norm)
    else:
        im = ax.imshow(values_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    value_norm = norm if norm is not None else mcolors.Normalize(vmin=vmin, vmax=vmax)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if mask[i, j]:
                continue
            v = float(values[i, j])
            if suppress_leq is not None and abs(v) <= suppress_leq:
                continue
            text_color = "white" if value_norm(v) >= 0.58 else PALETTE["ink"]
            ax.text(
                j, i, format(v, fmt), ha="center", va="center", color=text_color, fontsize=text_size
            )

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.86)
    cbar.set_label(cbar_label)


def plot_genus_distribution() -> None:
    df = pd.read_csv(DATA_DIR / "genus_distribution_filtered.csv").copy()
    df = df.sort_values("berlin_count", ascending=False)

    merged_lookup = {genus: group for group, members in GROUP_MEMBERS.items() for genus in members}

    def status_for_row(row: pd.Series) -> str:
        if row["berlin_count"] < 500 or row["leipzig_count"] < 500:
            return "Excluded"
        return "Retained"

    df["status"] = df.apply(status_for_row, axis=1)
    y = np.arange(len(df))
    h = 0.38

    fig, ax = plt.subplots(figsize=(10.6, 10.8))
    for idx, row in enumerate(df.itertuples(index=False)):
        alpha = 0.9 if row.status != "Excluded" else 0.3
        ax.barh(
            y[idx] - h / 2,
            row.berlin_count,
            height=h,
            color=CITY_COLORS["berlin"],
            alpha=alpha,
            edgecolor="white",
        )
        ax.barh(
            y[idx] + h / 2,
            row.leipzig_count,
            height=h,
            color=CITY_COLORS["leipzig"],
            alpha=alpha,
            edgecolor="white",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(df["genus"])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Number of trees (log scale)")
    ax.set_ylabel("Genus")
    ax.axvline(500, color=PALETTE["slate"], linestyle="--", linewidth=1.2)
    ax.grid(axis="x")

    x_annot = float(df[["berlin_count", "leipzig_count"]].to_numpy().max()) * 1.4
    for yi, genus in enumerate(df["genus"]):
        group_tag = merged_lookup.get(genus)
        if group_tag:
            ax.text(x_annot, yi, group_tag, va="center", fontsize=8.2, color=PALETTE["slate"])

    ax.set_xlim(80, x_annot * 1.2)
    city_handles = [
        mpatches.Patch(color=CITY_COLORS["berlin"], label="Berlin"),
        mpatches.Patch(color=CITY_COLORS["leipzig"], label="Leipzig"),
    ]
    ax.legend(handles=city_handles, loc="lower right")
    ax.text(
        500 * 1.07,
        len(df) - 0.55,
        "<500 in one city: excluded",
        fontsize=8.9,
        color=PALETTE["slate"],
        va="top",
    )
    _save(fig, "fig-genus-distribution.png")


def plot_jm_monthly() -> None:
    df = pd.read_csv(DATA_DIR / "jm_monthly_mean_wide.csv")
    labels = [MONTH_MAP.get(int(m), str(m)) for m in df["month"]]

    fig, ax = plt.subplots(figsize=(8.4, 4.9))
    ax.axvspan(3.5, 11.5, color=PALETTE["cloud"], alpha=0.85, zorder=0)
    ax.plot(
        df["month"],
        df["berlin"],
        color=CITY_COLORS["berlin"],
        marker="o",
        label="Berlin",
    )
    ax.plot(
        df["month"],
        df["leipzig"],
        color=CITY_COLORS["leipzig"],
        marker="^",
        linestyle="--",
        label="Leipzig",
    )
    ax.set_xticks(df["month"])
    ax.set_xticklabels(labels)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean JM distance")
    min_v = float(df[["berlin", "leipzig"]].to_numpy().min())
    max_v = float(df[["berlin", "leipzig"]].to_numpy().max())
    pad = (max_v - min_v) * 0.2
    ax.set_ylim(min_v - pad, max_v + pad)
    ax.text(
        0.02,
        0.08,
        "Spearman rho = 0.951, p < 0.001",
        transform=ax.transAxes,
        fontsize=8.5,
        color=PALETTE["ink"],
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": PALETTE["cloud"]},
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y")
    _save(fig, "fig-jm-monthly.png")


def plot_morans_i() -> None:
    df = pd.read_csv(DATA_DIR / "morans_i_decay_wide.csv")

    fig, ax = plt.subplots(figsize=(8.0, 4.9))
    ax.plot(
        df["distance_m"],
        df["berlin"],
        color=CITY_COLORS["berlin"],
        marker="o",
        label="Berlin",
    )
    ax.plot(
        df["distance_m"],
        df["leipzig"],
        color=CITY_COLORS["leipzig"],
        marker="^",
        linestyle="--",
        label="Leipzig",
    )
    ax.set_xlabel("Spatial lag distance (m)")
    ax.set_ylabel("Moran's I")
    ax.set_xlim(float(df["distance_m"].min()), float(df["distance_m"].max()))
    min_v = float(df[["berlin", "leipzig"]].to_numpy().min())
    max_v = float(df[["berlin", "leipzig"]].to_numpy().max())
    ax.set_ylim(max(0.0, min_v * 0.85), max_v * 1.08)
    ax.axvline(1200, color=PALETTE["slate"], linestyle=":", linewidth=1.1)

    end_x = int(df["distance_m"].iloc[-1])
    end_berlin = float(df["berlin"].iloc[-1])
    end_leipzig = float(df["leipzig"].iloc[-1])
    ax.annotate(
        f"{end_berlin:.3f}",
        xy=(end_x, end_berlin),
        xytext=(8, 4),
        textcoords="offset points",
        color=CITY_COLORS["berlin"],
        fontsize=8.5,
    )
    ax.annotate(
        f"{end_leipzig:.3f}",
        xy=(end_x, end_leipzig),
        xytext=(8, -10),
        textcoords="offset points",
        color=CITY_COLORS["leipzig"],
        fontsize=8.5,
    )

    ax.legend(loc="best")
    ax.grid(axis="y")
    _save(fig, "fig-morans-i.png")


def plot_chm_cohens_d() -> None:
    df = pd.read_csv(DATA_DIR / "chm_assessment_sorted.csv").copy()
    df = df.sort_values("cohens_d")
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(10.4, 10.2))
    for yi, row in enumerate(df.itertuples(index=False)):
        color = CITY_COLORS["berlin"] if row.cohens_d >= 0 else CITY_COLORS["leipzig"]
        ax.barh(yi, row.cohens_d, color=color, alpha=0.9, edgecolor="white")
        ax.errorbar(
            row.cohens_d,
            yi,
            xerr=[[row.cohens_d - row.ci_low], [row.ci_high - row.cohens_d]],
            fmt="none",
            ecolor=PALETTE["ink"],
            elinewidth=0.8,
            capsize=2,
        )

    lim = float(max(abs(df["ci_low"].min()), abs(df["ci_high"].max())) * 1.15)
    ax.set_xlim(-lim, lim)
    for ref in [0.2, 0.5, 0.8]:
        ax.axvline(ref, color=PALETTE["slate"], linestyle="--", linewidth=0.85, alpha=0.8)
        ax.axvline(-ref, color=PALETTE["slate"], linestyle="--", linewidth=0.85, alpha=0.8)
    ax.axvline(0.0, color=PALETTE["ink"], linestyle="-", linewidth=0.95)
    ax.set_yticks(y)
    ax.set_yticklabels(df["genus"])
    ax.set_xlabel("Cohen's d of canopy height (Berlin minus Leipzig)")
    ax.set_ylabel("Genus")
    ax.grid(axis="x")
    _save(fig, "fig-chm-cohens-d.png")


def plot_correlation() -> None:
    mat = _read_matrix_csv(DATA_DIR / "correlation_spectral_bands.csv")

    fig, ax = plt.subplots(figsize=(7.4, 6.3))
    _annotated_heatmap(
        ax,
        mat,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        cbar_label="Pearson r",
        fmt=".2f",
        text_size=7.3,
        mask_upper=False,
        hide_diagonal=False,
        suppress_leq=None,
    )
    ax.set_xlabel("Spectral band")
    ax.set_ylabel("Spectral band")
    _save(fig, "fig-correlation.png")


def plot_algorithm_comparison() -> None:
    df = pd.read_csv(DATA_DIR / "algorithm_comparison.csv").copy()
    display_order = [
        "majority",
        "stratified_random",
        "random_forest",
        "xgboost",
        "cnn_1d",
        "tabnet",
    ]
    labels = {
        "majority": "Majority",
        "stratified_random": "Stratified random",
        "random_forest": "Random Forest",
        "xgboost": "XGBoost",
        "cnn_1d": "1D-CNN",
        "tabnet": "TabNet",
    }
    group_color = {
        "baseline": PALETTE["slate"],
        "ml": CITY_COLORS["berlin"],
        "dl": CITY_COLORS["leipzig"],
        "nn": CITY_COLORS["leipzig"],
    }

    df["order"] = df["algorithm"].map({name: i for i, name in enumerate(display_order)})
    df = df.sort_values("order")
    x = np.arange(len(df))
    w = 0.38

    colors = [group_color[t] for t in df["type"]]
    fig, ax = plt.subplots(figsize=(9.1, 5.1))
    bars_train = ax.bar(
        x - w / 2,
        df["train_f1_mean"],
        width=w,
        color=colors,
        alpha=0.45,
        hatch="//",
        edgecolor="white",
        label="_nolegend_",
    )
    bars_val = ax.bar(
        x + w / 2,
        df["val_f1_mean"],
        width=w,
        color=colors,
        edgecolor="white",
        label="_nolegend_",
    )

    ax.axvline(1.5, color=PALETTE["slate"], linestyle=":", linewidth=1.1)
    ax.text(0.16, 0.98, "Naive baselines", transform=ax.transAxes, va="top", fontsize=8.2)
    ax.text(0.56, 0.98, "ML / DL models", transform=ax.transAxes, va="top", fontsize=8.2)

    for i, (b_train, b_val) in enumerate(zip(bars_train, bars_val, strict=False)):
        train_v = float(df["train_f1_mean"].iloc[i])
        val_v = float(df["val_f1_mean"].iloc[i])
        ax.text(
            b_val.get_x() + b_val.get_width() / 2,
            val_v + 0.016,
            f"{val_v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.6,
        )
        if df["type"].iloc[i] != "baseline":
            gap = train_v - val_v
            ax.text(
                b_train.get_x() + b_train.get_width() / 2,
                max(train_v, val_v) + 0.045,
                f"Δ={gap:.3f}",
                ha="center",
                va="bottom",
                fontsize=7.3,
                color=PALETTE["slate"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels([labels[a] for a in df["algorithm"]], rotation=20, ha="right")
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y")
    style_handles = [
        mpatches.Patch(
            facecolor=PALETTE["cloud"],
            edgecolor=PALETTE["ink"],
            hatch="//",
            label="Train",
        ),
        mpatches.Patch(
            facecolor=PALETTE["cloud"],
            edgecolor=PALETTE["ink"],
            label="Validation",
        ),
    ]
    type_handles = [
        mpatches.Patch(facecolor=PALETTE["slate"], label="Baseline paradigm"),
        mpatches.Patch(facecolor=CITY_COLORS["berlin"], label="ML paradigm"),
        mpatches.Patch(facecolor=CITY_COLORS["leipzig"], label="DL paradigm"),
    ]
    spacer = Line2D([0], [0], color="none", label="")
    ax.legend(
        handles=[*style_handles, spacer, *type_handles],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        handlelength=2.2,
    )
    _save(fig, "fig-algorithm-comparison.png")


def plot_feature_importance() -> None:
    df = pd.read_csv(DATA_DIR / "feature_importances_top20.csv").copy()
    df = df.sort_values("importance", ascending=True)
    df["feature_type"] = df["feature"].map(_feature_type)
    df["feature_label"] = df["feature"].map(_feature_label)

    color_map = {"Spectral band": PALETTE["steel"], "Vegetation index": PALETTE["forest"]}
    colors = [color_map[t] for t in df["feature_type"]]

    fig, ax = plt.subplots(figsize=(8.9, 7.2))
    ax.barh(df["feature_label"], df["importance"], color=colors, alpha=0.94)
    ax.set_xlim(left=0.0)
    ax.set_xlabel("Feature importance (gain)")
    ax.set_ylabel("Feature")
    ax.grid(axis="x")
    ax.legend(
        handles=[
            mpatches.Patch(facecolor=color_map["Spectral band"], label="Spectral bands"),
            mpatches.Patch(facecolor=color_map["Vegetation index"], label="Vegetation indices"),
        ],
        loc="lower right",
    )
    _save(fig, "fig-feature-importance.png")


def _plot_confusion(
    *,
    input_name: str,
    output_name: str,
    normalized: bool,
    sort_by_f1: bool,
) -> None:
    mat = _read_matrix_csv(DATA_DIR / input_name)
    labels = list(mat.index)

    if sort_by_f1:
        ordered = _ordered_labels_by_berlin_f1(labels)
        mat = mat.loc[ordered, ordered]

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    if normalized:
        _annotated_heatmap(
            ax,
            mat,
            cmap="Blues",
            vmin=0,
            vmax=1,
            cbar_label="Normalized proportion",
            fmt=".2f",
            text_size=5.8,
            suppress_leq=0.01,
            norm=mcolors.PowerNorm(gamma=0.55, vmin=0, vmax=1),
        )
    else:
        vmax = float(mat.to_numpy().max())
        _annotated_heatmap(
            ax,
            mat,
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            cbar_label="Count",
            fmt=".0f",
            text_size=5.4,
            suppress_leq=0.0,
        )

    ax.set_xlabel("Predicted genus")
    ax.set_ylabel("True genus")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, output_name)


def _load_leipzig_zeroshot_normalized_matrix() -> pd.DataFrame:
    path = OUTPUTS_DIR / "transfer_evaluation.json"
    data = _read_json(path)

    labels = [str(row["genus"]) for row in data.get("per_class", [])]
    matrix = np.asarray(data.get("confusion_matrix", []), dtype=float)

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("transfer_evaluation.json confusion_matrix is not square.")
    if len(labels) != matrix.shape[0]:
        raise ValueError(
            "transfer_evaluation.json label count does not match confusion_matrix shape."
        )

    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_norm = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)
    mat = pd.DataFrame(matrix_norm, index=labels, columns=labels)

    ordered = _ordered_labels_by_berlin_f1(labels)
    return mat.loc[ordered, ordered]


def plot_confusion_berlin() -> None:
    _plot_confusion(
        input_name="confusion_matrix_berlin_raw.csv",
        output_name="fig-confusion-berlin.png",
        normalized=False,
        sort_by_f1=True,
    )


def plot_confusion_berlin_xgb() -> None:
    _plot_confusion(
        input_name="confusion_matrix_berlin_norm.csv",
        output_name="fig-confusion-berlin-xgb.png",
        normalized=True,
        sort_by_f1=True,
    )


def plot_confusion_leipzig_zeroshot() -> None:
    mat = _load_leipzig_zeroshot_normalized_matrix()
    _write_matrix_csv(mat, DATA_DIR / "confusion_matrix_leipzig_zeroshot_norm.csv")

    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    _annotated_heatmap(
        ax,
        mat,
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar_label="Normalized proportion",
        fmt=".2f",
        text_size=5.8,
        suppress_leq=0.01,
        norm=mcolors.PowerNorm(gamma=0.55, vmin=0, vmax=1),
    )
    ax.set_xlabel("Predicted genus")
    ax.set_ylabel("True genus")
    ax.tick_params(axis="x", rotation=45)
    ax.tick_params(axis="y", rotation=0)
    _save(fig, "fig-confusion-leipzig-zeroshot.png")


def plot_per_genus_transfer() -> None:
    df = pd.read_csv(DATA_DIR / "transfer_metrics_sorted.csv").copy()
    df = df.sort_values("relative_drop", ascending=False)
    counts = _build_berlin_n_for_final_classes()

    y = np.arange(len(df))
    y_labels = [f"{g} (n={counts.get(g, 0):,})" for g in df["genus"]]
    is_conifer = df["is_conifer"].astype(bool)
    bar_colors = [PALETTE["amber"] if flag else PALETTE["steel"] for flag in is_conifer]

    fig, (ax_drop, ax_abs) = plt.subplots(
        ncols=2,
        figsize=(12.2, 7.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    ax_drop.barh(y, df["relative_drop"] * 100, color=bar_colors, alpha=0.95)
    ax_drop.set_yticks(y)
    ax_drop.set_yticklabels(y_labels)
    ax_drop.invert_yaxis()
    ax_drop.set_xlabel("Relative transfer drop (%)")
    ax_drop.set_ylabel("Genus (Berlin sample size)")
    ax_drop.axvline(5, color=PALETTE["slate"], linestyle="--", linewidth=1.0)
    ax_drop.axvline(15, color=PALETTE["ink"], linestyle="--", linewidth=1.0)
    ax_drop.grid(axis="x")

    ax_abs.hlines(
        y, df["leipzig_f1"], df["berlin_f1"], color=PALETTE["slate"], linewidth=1.2, alpha=0.75
    )
    ax_abs.scatter(
        df["berlin_f1"],
        y,
        color=CITY_COLORS["berlin"],
        marker="o",
        s=28,
        label="Berlin F1",
        zorder=3,
    )
    ax_abs.scatter(
        df["leipzig_f1"],
        y,
        color=CITY_COLORS["leipzig"],
        marker="^",
        s=30,
        label="Leipzig zero-shot F1",
        zorder=3,
    )
    ax_abs.set_xlim(0.0, 1.0)
    ax_abs.set_xlabel("Absolute per-genus F1")
    ax_abs.grid(axis="x")

    legend_handles = [
        mpatches.Patch(facecolor=PALETTE["amber"], label="Conifers"),
        mpatches.Patch(facecolor=PALETTE["steel"], label="Deciduous"),
    ]
    ax_drop.legend(handles=legend_handles, loc="lower right")
    ax_abs.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    _save(fig, "fig-per-genus-transfer.png")


def plot_finetuning_curve() -> None:
    df = pd.read_csv(DATA_DIR / "finetuning_curves.csv")
    ml = df[df["model_type"] == "ml"].sort_values("n_samples")
    nn = df[df["model_type"] == "nn"].sort_values("n_samples")

    meta = _read_json(OUTPUTS_DIR / "finetuning_curve.json")
    a = float(meta["powerlaw_fit"]["a"])
    b = float(meta["powerlaw_fit"]["b"])

    x_dense = np.linspace(float(ml["n_samples"].min()), float(ml["n_samples"].max()), 200)
    y_fit = a * np.power(x_dense, b)

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    ax.plot(
        ml["n_samples"],
        ml["f1_score"],
        color=CITY_COLORS["berlin"],
        marker="o",
        label="XGBoost fine-tuned",
    )
    ax.plot(
        nn["n_samples"],
        nn["f1_score"],
        color=CITY_COLORS["leipzig"],
        marker="^",
        linestyle="--",
        label="1D-CNN fine-tuned",
    )
    ax.fill_between(
        ml["n_samples"], ml["ci_low"], ml["ci_high"], color=CITY_COLORS["berlin"], alpha=0.15
    )
    ax.fill_between(
        nn["n_samples"], nn["ci_low"], nn["ci_high"], color=CITY_COLORS["leipzig"], alpha=0.15
    )
    ax.plot(
        x_dense,
        y_fit,
        color=CITY_COLORS["berlin"],
        linestyle=":",
        linewidth=1.6,
        label="Power-law fit (b=0.194)",
    )

    refs = [
        (0.751, PALETTE["ink"], "--", "Berlin test XGBoost (0.751)"),
        (0.607, PALETTE["slate"], "--", "Berlin test 1D-CNN (0.607)"),
        (0.377, CITY_COLORS["berlin"], "-.", "Zero-shot XGBoost (0.377)"),
        (0.380, CITY_COLORS["leipzig"], "-.", "Zero-shot 1D-CNN (0.380)"),
        (0.786, PALETTE["forest"], "--", "From-scratch XGBoost (0.786)"),
    ]
    x_right = float(ml["n_samples"].max())
    min_gap = 0.018
    sorted_refs = sorted(refs, key=lambda item: item[0])
    adjusted_y: list[float] = []
    for y_ref, *_ in sorted_refs:
        target = y_ref if not adjusted_y else max(y_ref, adjusted_y[-1] + min_gap)
        adjusted_y.append(target)

    text_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for (y_ref, color, ls, label), y_text in zip(sorted_refs, adjusted_y, strict=False):
        ax.axhline(y_ref, color=color, linestyle=ls, linewidth=1.0, alpha=0.9)
        ax.text(
            1.12,
            y_text,
            label,
            transform=text_transform,
            fontsize=7.0,
            color=color,
            va="center",
            ha="left",
            clip_on=False,
        )

    ticks = [int(x) for x in ml["n_samples"].tolist()]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{x:,}" for x in ticks])
    ax.set_xlabel("Leipzig training samples (absolute)")
    ax.set_ylabel("Weighted F1")
    ax.set_ylim(0.34, 0.83)
    ax.set_xlim(float(ml["n_samples"].min()) * 0.95, x_right * 1.1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=7.2)
    ax.grid(axis="y")
    _save(fig, "fig-finetuning-curve.png")


def plot_per_genus_recovery() -> None:
    mat = _read_matrix_csv(DATA_DIR / "finetuning_per_genus_matrix.csv")
    transfer = pd.read_csv(DATA_DIR / "transfer_metrics_sorted.csv").set_index("genus")

    mat["0.0"] = transfer.loc[mat.index, "leipzig_f1"].astype(float)
    mat["berlin_ref"] = transfer.loc[mat.index, "berlin_f1"].astype(float)
    mat["from_scratch_ref"] = 0.786

    ordered_cols = ["0.0", "0.1", "0.25", "0.5", "1.0", "berlin_ref", "from_scratch_ref"]
    mat = mat[ordered_cols]

    mat["recovery_gain"] = mat["1.0"] - mat["0.0"]
    mat = mat.sort_values("recovery_gain", ascending=False)
    mat = mat.drop(columns=["recovery_gain"])

    fig, ax = plt.subplots(figsize=(9.3, 8.0))
    _annotated_heatmap(
        ax,
        mat,
        cmap="BuGn",
        vmin=0,
        vmax=1,
        cbar_label="Per-genus F1",
        fmt=".2f",
        text_size=6.8,
    )
    ax.set_xlabel("Fine-tuning fraction and references")
    ax.set_ylabel("Genus")
    ax.set_xticklabels(["0%", "10%", "25%", "50%", "100%", "Berlin\nref", "From-scratch\noverall"])
    _save(fig, "fig-per-genus-recovery.png")


def main() -> None:
    apply_report_style()

    plot_genus_distribution()
    plot_jm_monthly()
    plot_morans_i()
    plot_chm_cohens_d()
    plot_correlation()
    plot_algorithm_comparison()
    plot_feature_importance()
    plot_confusion_berlin()
    plot_confusion_berlin_xgb()
    plot_confusion_leipzig_zeroshot()
    plot_per_genus_transfer()
    plot_finetuning_curve()
    plot_per_genus_recovery()


if __name__ == "__main__":
    main()
