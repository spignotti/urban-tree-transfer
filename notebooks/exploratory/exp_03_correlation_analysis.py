# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Urban Tree Transfer - Exploratory Notebook
#
# **Title:** Correlation Analysis
#
# **Phase:** 2 - Feature Engineering
#
# **Topic:** Correlation-Based Feature Reduction
#
# **Research Question:**
# Which spectral and index features are redundant across Berlin and Leipzig and should be removed before final model preparation?
#
# **Key Findings:**
# - Redundant features are identified within feature families.
# - Removal decisions are exported as `correlation_removal.json`.
# - Cross-city consistency is used to avoid city-specific over-pruning.
#
# **Input:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_2_features`
#
# **Output:** `correlation_removal.json` plus analysis outputs on Google Drive
#
# **Author:** Silas Pignotti
#
# **Created:** 2025-01-15
#
# **Updated:** 2026-03-11

# %%
# ============================================================================
# 1. ENVIRONMENT SETUP
# ============================================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended for large datasets
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================================

import subprocess
from google.colab import userdata

# Get GitHub token from Colab Secrets (for private repo access)
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        "GITHUB_TOKEN not found in Colab Secrets.\n"
        "1. Click the key icon in the left sidebar\n"
        "2. Add a secret named 'GITHUB_TOKEN' with your GitHub token\n"
        "3. Toggle 'Notebook access' ON"
    )

# Install package from private GitHub repo
repo_url = f"git+https://{token}@github.com/silas-workspace/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

print("OK: Package installed")


# %%
# ============================================================================
# 2. GOOGLE DRIVE
# ============================================================================
from google.colab import drive

drive.mount("/content/drive")

print("OK: Google Drive mounted")


# %%
# ============================================================================
# 3. IMPORTS
# ============================================================================
from urban_tree_transfer.config import RANDOM_SEED
from urban_tree_transfer.config.loader import (
    load_feature_config,
    get_spectral_bands,
    get_vegetation_indices,
)
from urban_tree_transfer.utils import ExecutionLog

from pathlib import Path
from datetime import datetime, timezone
import json
import re

import geopandas as gpd
import pandas as pd
import numpy as np

from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

log = ExecutionLog("exp_03_correlation_analysis")

print("OK: Package imports complete")


# %%
# ============================================================================
# 4. CONFIGURATION
# ============================================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"
REPORT_DIR = DRIVE_DIR / "outputs" / "report"

CITIES = ["berlin", "leipzig"]
SAMPLE_SIZE = 10_000
CORRELATION_THRESHOLD = 0.95

for d in [METADATA_DIR, LOGS_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

feature_config = load_feature_config()
spectral_bands = get_spectral_bands(feature_config)
vegetation_indices = get_vegetation_indices(feature_config)
chm_features = ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"]

index_families = feature_config["vegetation_indices"]

print(f"Input (Phase 2):  {INPUT_DIR}")
print(f"Output (JSONs):   {METADATA_DIR}")
print(f"Logs (Drive):     {LOGS_DIR}")
print(f"Cities:           {CITIES}")
print(f"Random seed:      {RANDOM_SEED}")

# %% [markdown]
# ## Correlation Analysis Overview
# This notebook identifies redundant features within *groups only* (CHM, bands, indices)
# using Pearson correlation on **standardized** values. Correlations are computed by
# stacking all temporal instances per feature base (no cross-month correlation).
#

# %%
# ============================================================================
# SECTION 1: Load and sample data
# ============================================================================

log.start_step("Data Loading & Sampling")

def stratified_sample(df: pd.DataFrame, label_col: str, n: int, seed: int) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    y = df[label_col]
    idx = next(splitter.split(df, y))[0]
    return df.iloc[idx].copy()

def resolve_base_alias(base: str, columns: set[str]) -> str:
    if any(col.startswith(f"{base}_") for col in columns):
        return base
    if base.startswith("B") and base[1:].isdigit():
        alt = f"B{int(base[1:]):02d}"
        if any(col.startswith(f"{alt}_") for col in columns):
            return alt
    base_lower = base.lower()
    if any(col.startswith(f"{base_lower}_") for col in columns):
        return base_lower
    return base

def get_months_for_base(columns: list[str], base: str) -> list[int]:
    pattern = re.compile(rf"^{re.escape(base)}_(\d{{1,2}})$")
    months = []
    for col in columns:
        match = pattern.match(col)
        if match:
            months.append(int(match.group(1)))
    return sorted(set(months))

def build_base_columns(
    base: str,
    months: list[int],
    columns: set[str],
    alias_map: dict[str, str] | None = None,
) -> list[str]:
    alias = alias_map.get(base, base) if alias_map else base
    col_list = []
    for m in months:
        candidates = [f"{alias}_{m:02d}", f"{alias}_{m}"]
        for c in candidates:
            if c in columns:
                col_list.append(c)
                break
    return col_list

city_data = {}
all_columns = None

for city in CITIES:
    path = INPUT_DIR / f"trees_clean_{city}.gpkg"
    print(f"Loading: {path}")
    gdf = gpd.read_file(path)
    print(f"  {city}: {len(gdf):,} rows, {len(gdf.columns)} columns")

    sampled = stratified_sample(gdf, label_col="genus_latin", n=SAMPLE_SIZE, seed=RANDOM_SEED)
    print(f"  {city}: sampled {len(sampled):,} rows")

    city_data[city] = sampled
    if all_columns is None:
        all_columns = set(sampled.columns)
    else:
        all_columns = all_columns & set(sampled.columns)

if all_columns is None:
    raise ValueError("No data loaded. Check input paths.")

base_aliases = {
    base: resolve_base_alias(base, all_columns) for base in spectral_bands + vegetation_indices
}

# Determine common temporal months across all S2 bases
months_by_base = {}
for base in spectral_bands + vegetation_indices:
    alias = base_aliases.get(base, base)
    months_by_base[base] = get_months_for_base(list(all_columns), alias)

common_months = None
for base, months in months_by_base.items():
    if not months:
        continue
    months_set = set(months)
    if common_months is None:
        common_months = months_set
    else:
        common_months = common_months & months_set

if not common_months:
    temporal_like = sorted(
        [c for c in all_columns if re.search(r"_(\d{1,2})$", c)]
    )
    sample = ", ".join(temporal_like[:10]) if temporal_like else "<none>"
    raise ValueError(
        "No common months found across S2 features. "
        f"Sample temporal-like columns: {sample}"
    )

common_months = sorted(common_months)
print(f"Common months used for correlation: {common_months}")

log.end_step(status="success", records=sum(len(df) for df in city_data.values()))


# %%
# ============================================================================
# Helper functions for correlation and selection
# ============================================================================

def standardize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    scaler = StandardScaler()
    values = scaler.fit_transform(df[cols])
    return pd.DataFrame(values, columns=cols, index=df.index)

def stacked_vector(df_std: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return df_std[cols].to_numpy().ravel(order="C")

def compute_base_correlations(
    df: pd.DataFrame,
    bases: list[str],
    months: list[int],
    alias_map: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    base_cols = {
        base: build_base_columns(base, months, set(df.columns), alias_map) for base in bases
    }
    # Remove bases without columns
    base_cols = {k: v for k, v in base_cols.items() if v}
    all_cols = [c for cols in base_cols.values() for c in cols]

    if not all_cols:
        raise ValueError("No columns found for correlation.")

    df_std = standardize_columns(df, all_cols)

    base_vectors = {
        base: stacked_vector(df_std, cols) for base, cols in base_cols.items()
    }

    bases_list = list(base_vectors.keys())
    corr = {base: {} for base in bases_list}
    for i, base_i in enumerate(bases_list):
        corr[base_i][base_i] = 1.0
        for j in range(i + 1, len(bases_list)):
            base_j = bases_list[j]
            r, _ = pearsonr(base_vectors[base_i], base_vectors[base_j])
            corr[base_i][base_j] = float(r)
            corr[base_j][base_i] = float(r)
    return corr


def compute_spearman_base_correlations(
    df: pd.DataFrame,
    bases: list[str],
    months: list[int],
    alias_map: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compute Spearman rank correlations between feature bases."""
    base_cols = {
        base: build_base_columns(base, months, set(df.columns), alias_map)
        for base in bases
    }
    base_cols = {k: v for k, v in base_cols.items() if v}
    all_cols = [c for cols in base_cols.values() for c in cols]
    if not all_cols:
        return {}
    df_std = standardize_columns(df, all_cols)
    base_vectors = {
        base: stacked_vector(df_std, cols) for base, cols in base_cols.items()
    }
    bases_list = list(base_vectors.keys())
    corr = {base: {} for base in bases_list}
    for i, base_i in enumerate(bases_list):
        corr[base_i][base_i] = 1.0
        for j in range(i + 1, len(bases_list)):
            base_j = bases_list[j]
            r, _ = spearmanr(base_vectors[base_i], base_vectors[base_j])
            corr[base_i][base_j] = float(r)
            corr[base_j][base_i] = float(r)
    return corr


def compute_chm_correlations(df: pd.DataFrame, features: list[str]) -> dict[str, dict[str, float]]:
    cols = [c for c in features if c in df.columns]
    df_std = standardize_columns(df, cols)
    corr = {c: {} for c in cols}
    for i, col_i in enumerate(cols):
        corr[col_i][col_i] = 1.0
        for j in range(i + 1, len(cols)):
            col_j = cols[j]
            r, _ = pearsonr(df_std[col_i].to_numpy(), df_std[col_j].to_numpy())
            corr[col_i][col_j] = float(r)
            corr[col_j][col_i] = float(r)
    return corr

def corr_dict_to_df(corr: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(corr)
    return df.loc[df.columns, df.columns]

def compute_mean_abs_target_corr(
    df: pd.DataFrame,
    base_cols: dict[str, list[str]],
    label_col: str = "genus_latin",
) -> dict[str, float]:
    target = df[label_col].astype(str)
    classes = sorted(target.unique())
    base_scores = {}

    for base, cols in base_cols.items():
        if not cols:
            continue
        df_std = standardize_columns(df, cols)
        feature_vec = stacked_vector(df_std, cols)
        # Repeat labels per month to match stacked vector
        labels_repeated = np.repeat(target.to_numpy(), len(cols))

        scores = []
        for cls in classes:
            cls_vec = (labels_repeated == cls).astype(int)
            r, _ = pearsonr(feature_vec, cls_vec)
            scores.append(abs(r))

        base_scores[base] = float(np.mean(scores))

    return base_scores

def compute_vif(df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    values = df[cols].to_numpy()
    # Standardize for stability
    values = StandardScaler().fit_transform(values)
    vifs = {}
    for i, col in enumerate(cols):
        y = values[:, i]
        X = np.delete(values, i, axis=1)
        # Solve linear regression via least squares
        coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coef
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
        vifs[col] = float(vif)
    return vifs

def reorder_by_clustering(corr_df: pd.DataFrame) -> list[str]:
    # Distance = 1 - |r|
    dist = 1 - np.abs(corr_df.values)
    # Ensure zero diagonal
    np.fill_diagonal(dist, 0)
    linkage_matrix = linkage(squareform(dist, checks=False), method="average")
    order = leaves_list(linkage_matrix)
    return corr_df.index[order].tolist()



# %%
# ============================================================================
# SECTION 2: CHM correlation analysis
# ============================================================================

log.start_step("CHM Correlation Analysis")

chm_corr = {}
for city, df in city_data.items():
    chm_corr[city] = compute_chm_correlations(df, chm_features)
    print(f"CHM correlation computed for {city}")

log.end_step(status="success", records=len(chm_features))


# %%
# ============================================================================
# SECTION 3: Spectral bands correlation analysis
# ============================================================================

log.start_step("Spectral Bands Correlation Analysis")

bands_corr = {}
bands_base_cols = {}
bands_target_corr = {}

for city, df in city_data.items():
    bands_corr[city] = compute_base_correlations(df, spectral_bands, common_months, base_aliases)
    bands_base_cols[city] = {
        base: build_base_columns(base, common_months, set(df.columns), base_aliases) for base in spectral_bands
    }
    bands_target_corr[city] = compute_mean_abs_target_corr(df, bands_base_cols[city])
    print(f"Bands correlation computed for {city}")

log.end_step(status="success", records=len(spectral_bands))


# %%
# ============================================================================
# SECTION 4: Vegetation indices correlation analysis
# ============================================================================

log.start_step("Vegetation Indices Correlation Analysis")

indices_corr = {}
indices_base_cols = {}
indices_target_corr = {}

for city, df in city_data.items():
    indices_corr[city] = compute_base_correlations(df, vegetation_indices, common_months, base_aliases)
    indices_base_cols[city] = {
        base: build_base_columns(base, common_months, set(df.columns), base_aliases) for base in vegetation_indices
    }
    indices_target_corr[city] = compute_mean_abs_target_corr(df, indices_base_cols[city])
    print(f"Indices correlation computed for {city}")

log.end_step(status="success", records=len(vegetation_indices))


# %%
# ============================================================================
# SECTION 5: Cross-city consistency & redundancy selection
# ============================================================================

log.start_step("Cross-city Consistency")

def redundant_pairs(corr_df: pd.DataFrame, threshold: float) -> set[tuple[str, str]]:
    pairs = set()
    cols = corr_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if abs(corr_df.iloc[i, j]) > threshold:
                a, b = cols[i], cols[j]
                pairs.add(tuple(sorted((a, b))))
    return pairs

def redundant_bases_from_pairs(pairs: set[tuple[str, str]]) -> set[str]:
    bases = set()
    for a, b in pairs:
        bases.add(a)
        bases.add(b)
    return bases

# Interpretability preference (higher = keep)
interpretability_score = {
    "NDVI": 5,
    "EVI": 4,
    "VARI": 3,
    "NDWI": 4,
    "NDII": 3,
    "NDre1": 3,
    "NDVIre": 2,
    "GNDVI": 1,
    "MSI": 1,
    "CIre": 1,
    "IRECI": 1,
    "RTVIcore": 1,
    "kNDVI": 1,
}

band_preference = {
    "B8": 5,
    "B8A": 2,
}

def choose_removal(a: str, b: str, target_scores: dict[str, float]) -> str:
    score_a = interpretability_score.get(a, 0) + band_preference.get(a, 0)
    score_b = interpretability_score.get(b, 0) + band_preference.get(b, 0)
    if score_a != score_b:
        return b if score_a > score_b else a
    # Tie-breaker: lower mean abs correlation with target
    ta = target_scores.get(a, 0.0)
    tb = target_scores.get(b, 0.0)
    return a if ta < tb else b

def select_removals(
    corr_city: dict[str, dict[str, float]],
    corr_city_2: dict[str, dict[str, float]],
    bases: list[str],
    target_scores: dict[str, float],
    min_keep: int | None = None,
    family_map: dict[str, str] | None = None,
) -> tuple[set[str], set[str], set[str]]:
    df1 = corr_dict_to_df(corr_city)
    df2 = corr_dict_to_df(corr_city_2)
    pairs1 = redundant_pairs(df1, CORRELATION_THRESHOLD)
    pairs2 = redundant_pairs(df2, CORRELATION_THRESHOLD)
    pairs_both = pairs1 & pairs2

    removals = set()
    retained = set(bases)

    # Sort by average absolute correlation strength
    def pair_strength(pair: tuple[str, str]) -> float:
        a, b = pair
        return (abs(df1.loc[a, b]) + abs(df2.loc[a, b])) / 2

    for a, b in sorted(pairs_both, key=pair_strength, reverse=True):
        if a in removals or b in removals:
            continue
        drop = choose_removal(a, b, target_scores)

        # Constraint checks
        if min_keep is not None and (len(retained) - 1) < min_keep:
            continue
        if family_map is not None:
            fam = family_map.get(drop)
            if fam:
                remaining = {x for x in retained if family_map.get(x) == fam}
                if len(remaining) <= 1:
                    continue

        removals.add(drop)
        retained.remove(drop)

    return removals, retained, pairs_both

# Build target scores averaged across cities
avg_bands_target = {}
for base in spectral_bands:
    vals = [bands_target_corr[c].get(base, 0.0) for c in CITIES]
    avg_bands_target[base] = float(np.mean(vals))

avg_indices_target = {}
for base in vegetation_indices:
    vals = [indices_target_corr[c].get(base, 0.0) for c in CITIES]
    avg_indices_target[base] = float(np.mean(vals))

# Family map for vegetation indices
family_map = {}
for fam, items in index_families.items():
    for item in items:
        family_map[item] = fam

bands_removed, bands_retained, bands_pairs_both = select_removals(
    bands_corr["berlin"],
    bands_corr["leipzig"],
    spectral_bands,
    avg_bands_target,
    min_keep=7,
)

indices_removed, indices_retained, indices_pairs_both = select_removals(
    indices_corr["berlin"],
    indices_corr["leipzig"],
    vegetation_indices,
    avg_indices_target,
    min_keep=None,
    family_map=family_map,
)

chm_removed = set()
chm_retained = set(chm_features)

log.end_step(status="success", records=len(bands_removed) + len(indices_removed))


# %%
# ============================================================================
# SECTION 6: VIF validation (retained features only)
# ============================================================================

log.start_step("VIF Validation")

# Build retained temporal columns
retained_bases = sorted(bands_retained | indices_retained)
retained_temporal_cols = []
for base in retained_bases:
    retained_temporal_cols.extend(build_base_columns(base, common_months, set(all_columns), base_aliases))

vif_results = {}
for city, df in city_data.items():
    vif_results[city] = compute_vif(df, retained_temporal_cols)
    max_vif = max(vif_results[city].values()) if vif_results[city] else 0.0
    print(f"{city}: max VIF = {max_vif:.2f}")

log.end_step(status="success", records=len(retained_temporal_cols))


# %%
# ============================================================================
# SECTION 8: Save JSON config
# ============================================================================

def build_rationale(removed_bases: set[str], corr_b: pd.DataFrame, corr_l: pd.DataFrame) -> dict[str, str]:
    rationale = {}
    for base in sorted(removed_bases):
        # Find strongest correlated partner in each city
        partners = [c for c in corr_b.columns if c != base]
        if not partners:
            continue
        best_partner = max(partners, key=lambda c: abs(corr_b.loc[base, c]))
        r_b = corr_b.loc[base, best_partner]
        r_l = corr_l.loc[base, best_partner]
        rationale[base] = (
            f"Highly correlated with {best_partner} (r_berlin={r_b:.2f}, r_leipzig={r_l:.2f})."
        )
    return rationale

bands_corr_df_berlin = corr_dict_to_df(bands_corr["berlin"])
bands_corr_df_leipzig = corr_dict_to_df(bands_corr["leipzig"])
indices_corr_df_berlin = corr_dict_to_df(indices_corr["berlin"])
indices_corr_df_leipzig = corr_dict_to_df(indices_corr["leipzig"])

bands_rationale = build_rationale(bands_removed, bands_corr_df_berlin, bands_corr_df_leipzig)
indices_rationale = build_rationale(indices_removed, indices_corr_df_berlin, indices_corr_df_leipzig)

# Redundant bases per city
bands_pairs_berlin = redundant_pairs(bands_corr_df_berlin, CORRELATION_THRESHOLD)
bands_pairs_leipzig = redundant_pairs(bands_corr_df_leipzig, CORRELATION_THRESHOLD)
indices_pairs_berlin = redundant_pairs(indices_corr_df_berlin, CORRELATION_THRESHOLD)
indices_pairs_leipzig = redundant_pairs(indices_corr_df_leipzig, CORRELATION_THRESHOLD)

redundant_berlin = sorted(
    redundant_bases_from_pairs(bands_pairs_berlin | indices_pairs_berlin)
)
redundant_leipzig = sorted(
    redundant_bases_from_pairs(bands_pairs_leipzig | indices_pairs_leipzig)
)

removed_bases = sorted(bands_removed | indices_removed)
removed_set = set(removed_bases)

union_redundant = set(redundant_berlin) | set(redundant_leipzig)
agreement_rate = len(removed_set) / len(union_redundant) if union_redundant else 1.0

city_specific_kept = sorted(union_redundant - removed_set)

# Temporal removal list
removed_temporal_features = []
for base in removed_bases:
    for m in common_months:
        removed_temporal_features.append(f"{base}_{m:02d}")

# Feature reduction summary
n_months = len(common_months)
total_bands = len(spectral_bands) * n_months
total_indices = len(vegetation_indices) * n_months
total_chm = len(chm_features)

removed_bands = len(bands_removed) * n_months
removed_indices = len(indices_removed) * n_months

after_bands = total_bands - removed_bands
after_indices = total_indices - removed_indices

before_total = total_bands + total_indices + total_chm
after_total = after_bands + after_indices + total_chm
reduction_pct = ((before_total - after_total) / before_total) * 100

max_vif = max(max(v.values()) for v in vif_results.values()) if vif_results else 0.0

output_json = {
    "analysis_date": datetime.now(timezone.utc).isoformat(),
    "sample_size_per_city": SAMPLE_SIZE,
    "correlation_threshold": CORRELATION_THRESHOLD,
    "correlation_metric": "pearson",
    "feature_groups": {
        "chm": {
            "analyzed": chm_features,
            "removed": sorted(chm_removed),
            "retained": sorted(chm_retained),
            "rationale": "All three serve different purposes (absolute, standardized, rank-based)",
        },
        "spectral_bands": {
            "analyzed": spectral_bands,
            "removed": sorted(bands_removed),
            "retained": sorted(bands_retained),
            "rationale_per_feature": bands_rationale,
        },
        "vegetation_indices": {
            "analyzed": vegetation_indices,
            "removed": sorted(indices_removed),
            "retained": sorted(indices_retained),
            "rationale_per_feature": indices_rationale,
        },
    },
    "temporal_removal": {
        "explanation": "Features removed across ALL temporal instances",
        "removed_temporal_features": removed_temporal_features,
        "total_removed": len(removed_temporal_features),
    },
    "cross_city_consistency": {
        "redundant_in_berlin": redundant_berlin,
        "redundant_in_leipzig": redundant_leipzig,
        "removed": removed_bases,
        "city_specific_kept": city_specific_kept,
        "agreement_rate": agreement_rate,
        "interpretation": (
            "High consistency (>=0.75)" if agreement_rate >= 0.75 else "Below target consistency"
        ),
    },
    "feature_reduction_summary": {
        "before": {
            "chm": total_chm,
            "spectral_bands": total_bands,
            "vegetation_indices": total_indices,
            "total": before_total,
        },
        "after": {
            "chm": total_chm,
            "spectral_bands": after_bands,
            "vegetation_indices": after_indices,
            "total": after_total,
        },
        "reduction_pct": round(reduction_pct, 2),
    },
    "validation": {
        "max_vif_retained_features": round(max_vif, 2),
        "vif_threshold": 10.0,
        "vif_check_passed": max_vif < 10.0,
    },
}


# %%
# ============================================================================
# SECTION 9: Spearman Supplementary Check
# ============================================================================

log.start_step("Spearman Supplementary Check")

spearman_divergent = []
combined_sample = pd.concat(city_data.values(), ignore_index=True)

bands_spearman = compute_spearman_base_correlations(
    combined_sample, spectral_bands, common_months, base_aliases
)
indices_spearman = compute_spearman_base_correlations(
    combined_sample, vegetation_indices, common_months, base_aliases
)

combined_pearson_bands = compute_base_correlations(
    combined_sample, spectral_bands, common_months, base_aliases
)
combined_pearson_indices = compute_base_correlations(
    combined_sample, vegetation_indices, common_months, base_aliases
)

for group_name, pearson_corr, spearman_corr in [
    ("spectral_bands", combined_pearson_bands, bands_spearman),
    ("vegetation_indices", combined_pearson_indices, indices_spearman),
]:
    bases_list = sorted(set(pearson_corr.keys()) & set(spearman_corr.keys()))
    for i, base_i in enumerate(bases_list):
        for base_j in bases_list[i + 1:]:
            r_p = pearson_corr.get(base_i, {}).get(base_j, 0.0)
            r_s = spearman_corr.get(base_i, {}).get(base_j, 0.0)
            divergence = abs(r_p - r_s)
            if divergence > 0.1:
                spearman_divergent.append({
                    "group": group_name,
                    "feature_a": base_i,
                    "feature_b": base_j,
                    "pearson_r": round(r_p, 4),
                    "spearman_r": round(r_s, 4),
                    "divergence": round(divergence, 4),
                })

if spearman_divergent:
    print(f"Found {len(spearman_divergent)} divergent pairs (|r_p - r_s| > 0.1):")
    for pair in spearman_divergent:
        print(
            f"  {pair['feature_a']} vs {pair['feature_b']}: "
            f"Pearson={pair['pearson_r']}, Spearman={pair['spearman_r']}"
        )
else:
    print("No divergent pairs found. Pearson and Spearman agree within 0.1.")

output_json["spearman_supplement"] = {
    "divergent_pairs": spearman_divergent,
    "max_divergence": (
        max(p["divergence"] for p in spearman_divergent)
        if spearman_divergent else 0.0
    ),
    "note": "Pearson remains primary metric; Spearman as supplementary check",
}

log.end_step(status="success", records=len(spearman_divergent))


# %%
# ============================================================================
# SECTION 10: JSON Output
# ============================================================================

log.start_step("JSON Output")

json_path = METADATA_DIR / "correlation_removal.json"
json_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")
print(f"Saved JSON: {json_path}")

log.end_step(status="success", records=len(removed_temporal_features))


# %%
try:
    report_path = REPORT_DIR / "correlation_matrix.json"
    report_output = {
        "pearson": {
            "spectral_bands": combined_pearson_bands,
            "vegetation_indices": combined_pearson_indices,
        },
        "spearman": {
            "spectral_bands": bands_spearman,
            "vegetation_indices": indices_spearman,
        },
        "disagreements": output_json["spearman_supplement"]["divergent_pairs"],
    }
    report_path.write_text(json.dumps(report_output, indent=2), encoding="utf-8")
    print(f"Saved report JSON: {report_path}")
except Exception as e:
    print(f"WARNING: Failed to export report JSON: {e}")


# %%
# ============================================================================
# FINDINGS SUMMARY
# ============================================================================

log.summary()
log.save(LOGS_DIR / f"{log.notebook}_execution.json")

analysis_files = []
config_files = [json_path] if json_path.exists() else []

print("\n" + "=" * 60)
print(f"{'EXPORT MANIFEST':^60}")
print("=" * 60)
print("\nAnalysis data:")
for file_path in sorted(analysis_files):
    print(f"  {file_path.name}")
print("\nConfig files:")
for file_path in sorted(config_files):
    print(f"  {file_path.name}")
print("\n" + "=" * 60)
print("\nOK: NOTEBOOK COMPLETE")
