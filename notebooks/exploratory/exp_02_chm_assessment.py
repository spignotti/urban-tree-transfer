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
# **Title:** CHM Assessment
#
# **Phase:** 2 - Feature Engineering
#
# **Topic:** CHM Quality Assessment and Threshold Selection
#
# **Research Question:**
# How useful are CHM-derived features for genus discrimination, and which CHM-related thresholds should be carried forward?
#
# **Key Findings:**
# - CHM feature quality is assessed for both cities.
# - Threshold decisions are exported for downstream notebooks.
# - Results are summarized as a JSON config on Google Drive.
#
# **Input:** `/content/drive/MyDrive/dev/urban-tree-transfer/data/phase_2_features`
#
# **Output:** `chm_assessment.json` plus analysis outputs on Google Drive
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
from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
from urban_tree_transfer.utils import ExecutionLog

from pathlib import Path
from datetime import datetime, timezone
import json

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.stats import alexandergovern, pearsonr, bootstrap

# Genus classification (deciduous/coniferous)
DECIDUOUS_GENERA = [
    "ACER", "AESCULUS", "AILANTHUS", "ALNUS", "BETULA", "CARPINUS",
    "CORNUS", "CORYLUS", "CRATAEGUS", "FAGUS", "FRAXINUS", "GLEDITSIA",
    "JUGLANS", "LIQUIDAMBAR", "MALUS", "PLATANUS", "POPULUS", "PRUNUS",
    "PYRUS", "QUERCUS", "ROBINIA", "SALIX", "SOPHORA", "SORBUS",
    "TILIA", "ULMUS",
]
CONIFEROUS_GENERA = [
    "ABIES", "LARIX", "PICEA", "PINUS", "TAXUS", "THUJA",
]

log = ExecutionLog("exp_02_chm_assessment")

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

CITIES = ["berlin", "leipzig"]
MIN_SAMPLES_PER_GENUS = 500
# Empirically based on Sentinel-2 spatial resolution: at 10m pixel size,
# trees below ~2m CHM are indistinguishable from ground-level vegetation
DETECTION_THRESHOLD_M = 2.0

for d in [METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

print(f"Input:  {INPUT_DIR}")
print(f"Output: {METADATA_DIR}")
print(f"Cities: {CITIES}")
print(f"Random seed: {RANDOM_SEED}")

# %% [markdown]
# ## Data Loading & Validation
#
# Load Phase 2a feature data for Berlin and Leipzig and validate expected schema and CRS.
#

# %%
log.start_step("Data Loading & Validation")

try:
    city_data = {}
    required_cols = ["tree_id", "city", "genus_latin", "CHM_1m", "height_m", "plant_year", "geometry"]

    for city in CITIES:
        path = INPUT_DIR / f"trees_with_features_{city}.gpkg"
        if not path.exists():
            fallback_path = INPUT_DIR / f"trees_with_features_{city.title()}.gpkg"
            if fallback_path.exists():
                print(
                    "Warning: non-normalized city filename detected. "
                    f"Using {fallback_path.name} instead of {path.name}."
                )
                path = fallback_path
        print(f"Loading: {path}")
        df = gpd.read_file(path)

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in {city}: {missing}")

        if df.crs is None or df.crs.to_epsg() != int(str(PROJECT_CRS).split(":")[-1]):
            raise ValueError(f"Invalid CRS for {city}: {df.crs}. Expected {PROJECT_CRS}.")

        df["genus_latin"] = df["genus_latin"].astype(str).str.upper().str.strip()
        df["city"] = city

        city_data[city] = df
        print(f"Loaded {city}: {len(df):,} rows")

    log.end_step(status="success", records=sum(len(df) for df in city_data.values()))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Discriminative Power (ANOVA η²)
#
# Compute one-way ANOVA and η² for CHM_1m by genus per city.
#

# %%
log.start_step("Discriminative Power (ANOVA eta-squared)")

try:
    eta2_by_city = {}
    viable_genera_by_city = {}

    for city, df in city_data.items():
        df_valid = df[df["CHM_1m"].notna()].copy()
        genus_counts = df_valid["genus_latin"].value_counts()
        viable_genera = genus_counts[genus_counts >= MIN_SAMPLES_PER_GENUS].index.tolist()
        viable_genera_by_city[city] = sorted(viable_genera)

        groups = [
            df_valid[df_valid["genus_latin"] == g]["CHM_1m"].values
            for g in viable_genera
        ]
        if len(groups) < 2:
            raise ValueError(f"Not enough viable genera for ANOVA in {city}.")

        result = alexandergovern(*groups)
        p_value = result.pvalue
        grand_mean = df_valid["CHM_1m"].mean()
        ss_between = sum(
            len(df_valid[df_valid["genus_latin"] == g])
            * (df_valid[df_valid["genus_latin"] == g]["CHM_1m"].mean() - grand_mean) ** 2
            for g in viable_genera
        )
        ss_total = ((df_valid["CHM_1m"] - grand_mean) ** 2).sum()
        eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0.0

        eta2_by_city[city] = {
            "eta_squared": eta_squared,
            "a_stat": float(result.statistic),
            "test": "alexander_govern_welch_anova",
            "p_value": float(p_value),
            "n_genera": len(viable_genera),
            "n_samples": int(len(df_valid)),
        }

        print(f"{city.title()}: eta^2={eta_squared:.4f}, n_genera={len(viable_genera)}")

    log.end_step(status="success", records=len(eta2_by_city))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Cross-City Consistency (Cohen's d)
#
# Compute Cohen's d effect sizes and confidence intervals to assess cross-city CHM consistency.

# %%
def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d using pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * group1.std(ddof=1) ** 2 + (n2 - 1) * group2.std(ddof=1) ** 2)
        / (n1 + n2 - 2)
    )
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def cohens_d_ci(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute Cohen's d with bootstrap CI."""
    def stat(g1, g2):
        # Return as 1D array instead of scalar to avoid concatenation error
        return np.atleast_1d(cohens_d(g1, g2))

    rng = np.random.default_rng(RANDOM_SEED)
    res = bootstrap(
        (group1, group2),
        stat,
        n_resamples=n_bootstrap,
        confidence_level=confidence,
        random_state=rng,
        method="percentile",
    )
    # Use .item() to extract scalar from 0-d array and avoid deprecation warning
    return res.confidence_interval.low.item(), res.confidence_interval.high.item()


log.start_step("Cross-City Consistency (Cohen's d)")

try:
    berlin = city_data["berlin"]
    leipzig = city_data["leipzig"]

    viable_common = sorted(
        set(viable_genera_by_city.get("berlin", []))
        & set(viable_genera_by_city.get("leipzig", []))
    )

    cohens_records = []
    for genus in viable_common:
        g1 = berlin[berlin["genus_latin"] == genus]["CHM_1m"].dropna().values
        g2 = leipzig[leipzig["genus_latin"] == genus]["CHM_1m"].dropna().values
        if len(g1) < MIN_SAMPLES_PER_GENUS or len(g2) < MIN_SAMPLES_PER_GENUS:
            continue

        d_value = cohens_d(g1, g2)
        ci_low, ci_high = cohens_d_ci(g1, g2)

        cohens_records.append(
            {
                "genus": genus,
                "cohens_d": d_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_berlin": int(len(g1)),
                "n_leipzig": int(len(g2)),
            }
        )

    cohens_df = pd.DataFrame(cohens_records).sort_values("cohens_d")
    mean_abs_d = float(cohens_df["cohens_d"].abs().mean()) if not cohens_df.empty else 0.0

    if mean_abs_d < 0.2:
        transfer_interpretation = "low transfer risk"
    elif mean_abs_d < 0.5:
        transfer_interpretation = "medium transfer risk"
    else:
        transfer_interpretation = "high transfer risk"

    print(f"Common viable genera: {len(viable_common)}")
    print(f"Mean |d|: {mean_abs_d:.3f} ({transfer_interpretation})")

    log.end_step(status="success", records=len(cohens_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Feature Engineering Validation
#
# Validate CHM_1m against cadastre height, and compute Z-score/percentile transforms.
#

# %%
log.start_step("Feature Engineering Validation")

try:
    combined = pd.concat(city_data.values(), ignore_index=True)
    combined = combined[combined["CHM_1m"].notna()].copy()

    valid_corr = combined[combined["height_m"].notna()].copy()
    r_value, p_value = pearsonr(valid_corr["CHM_1m"], valid_corr["height_m"])

    combined["CHM_1m_zscore"] = combined.groupby("genus_latin")["CHM_1m"].transform(
        lambda x: (x - x.mean()) / x.std(ddof=1) if x.std(ddof=1) > 0 else 0.0
    )
    combined["CHM_1m_percentile"] = combined.groupby("genus_latin")["CHM_1m"].transform(
        lambda x: x.rank(pct=True) * 100.0
    )

    print(f"CHM vs height correlation: r={r_value:.3f} (p={p_value:.3g})")

    log.end_step(status="success", records=len(combined))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Plant Year Threshold Analysis
#
# Determine the latest planting year with median CHM above the detection threshold.
#

# %%
log.start_step("Plant Year Threshold Analysis")

try:
    df_with_year = combined[combined["plant_year"].notna()].copy()
    df_with_year["plant_year"] = df_with_year["plant_year"].astype(int)

    median_by_year = (
        df_with_year.groupby("plant_year")["CHM_1m"]
        .median()
        .sort_index()
    )

    min_year = int(median_by_year.index.min())
    max_year = int(median_by_year.index.max())
    years_below = median_by_year[median_by_year < DETECTION_THRESHOLD_M]
    if not years_below.empty:
        cutoff_year = int(years_below.index.min())
        recommended_max_year = max(min_year, cutoff_year - 1)
    else:
        recommended_max_year = max_year

    df_with_year["year_cohort"] = (df_with_year["plant_year"] // 2) * 2

    print(f"Recommended max plant year: {recommended_max_year}")

    log.end_step(status="success", records=len(median_by_year))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Genus Inventory & Classification
#
# Create genus inventory, classify deciduous/coniferous, and determine analysis scope.
#

# %%
log.start_step("Genus Inventory & Classification")

try:
    genus_inventory = {
        city: city_data[city]["genus_latin"].value_counts().to_dict()
        for city in CITIES
    }

    combined_counts = pd.Series(genus_inventory["berlin"]).add(
        pd.Series(genus_inventory["leipzig"]), fill_value=0
    ).sort_values(ascending=False)

    def classify_genus(genus: str) -> str:
        if genus in DECIDUOUS_GENERA:
            return "deciduous"
        if genus in CONIFEROUS_GENERA:
            return "coniferous"
        return "unknown"

    genus_types = {g: classify_genus(g) for g in combined_counts.index}
    unclassified_genera = [g for g, t in genus_types.items() if t == "unknown"]

    viable_genera_overall = [
        g for g, count in combined_counts.items() if count >= MIN_SAMPLES_PER_GENUS
    ]
    conifer_genera_count = len([g for g in viable_genera_overall if g in CONIFEROUS_GENERA])
    conifer_sample_count = int(
        combined[combined["genus_latin"].isin(CONIFEROUS_GENERA)]["genus_latin"].count()
    )

    if conifer_genera_count < 3 or conifer_sample_count < 500:
        analysis_scope = "deciduous_only"
        reason = (
            f"n_genera={conifer_genera_count} < 3"
            if conifer_genera_count < 3
            else f"n_samples={conifer_sample_count} < 500"
        )
        include_conifers = False
    else:
        analysis_scope = "all"
        reason = "Sufficient conifer diversity and samples"
        include_conifers = True

    print(f"Analysis scope: {analysis_scope} ({reason})")

    log.end_step(status="success", records=len(combined_counts))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Visualizations
#
# Generate required plots for CHM assessment.
#

# %%
log.start_step("Output Validation")

try:
    # Validate eta^2 values
    for city, stats in eta2_by_city.items():
        eta = stats["eta_squared"]
        if not (0.0 <= eta <= 1.0):
            raise ValueError(f"Invalid eta^2 for {city}: {eta} (must be in [0, 1])")

    # Validate Cohen's d CIs
    if not cohens_df.empty:
        invalid_ci = cohens_df[cohens_df["ci_low"] > cohens_df["ci_high"]]
        if not invalid_ci.empty:
            raise ValueError(
                f"Invalid CI bounds for genera: {invalid_ci['genus'].tolist()}"
            )

    # Validate correlation
    if not (-1.0 <= r_value <= 1.0):
        raise ValueError(f"Invalid correlation: {r_value} (must be in [-1, 1])")

    # Validate plant year threshold
    if not (min_year <= recommended_max_year <= max_year):
        raise ValueError(
            f"Plant year threshold {recommended_max_year} outside data range [{min_year}, {max_year}]"
        )

    # Validate all genera classified
    if unclassified_genera:
        print(
            f"Warning: {len(unclassified_genera)} unclassified genera: {unclassified_genera[:5]}"
        )

    print("✓ All validation checks passed")
    log.end_step(status="success", records=4)

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## JSON Output
#
# Save CHM assessment configuration for downstream use.
#

# %%
log.start_step("JSON Output")

try:
    include_chm = (
        eta2_by_city.get("berlin", {}).get("eta_squared", 0.0) >= 0.06
        and eta2_by_city.get("leipzig", {}).get("eta_squared", 0.0) >= 0.06
        and mean_abs_d < 0.2
    )

    median_chm_by_year = {str(int(k)): float(v) for k, v in median_by_year.items()}
    justification = (
        f"Trees planted after {recommended_max_year} have median CHM < "
        f"{DETECTION_THRESHOLD_M}m in 2021 imagery"
    )

    output = {
        "analysis_date": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "chm_features": ["CHM_1m", "CHM_1m_zscore", "CHM_1m_percentile"],
        "include_chm": include_chm,
        "discriminative_power": {
            "berlin_eta2": round(eta2_by_city["berlin"]["eta_squared"], 4),
            "leipzig_eta2": round(eta2_by_city["leipzig"]["eta_squared"], 4),
        },
        "transfer_risk": {
            "cohens_d_mean": round(mean_abs_d, 4),
            "interpretation": transfer_interpretation,
        },
        "validation": {
            "chm_cadastre_correlation": round(float(r_value), 4),
        },
        "plant_year_analysis": {
            "detection_threshold_m": DETECTION_THRESHOLD_M,
            "median_chm_by_year": median_chm_by_year,
            "recommended_max_plant_year": int(recommended_max_year),
            "justification": justification,
        },
        "genus_inventory": {
            "berlin": {k: int(v) for k, v in genus_inventory["berlin"].items()},
            "leipzig": {k: int(v) for k, v in genus_inventory["leipzig"].items()},
            "classification": {
                "deciduous": DECIDUOUS_GENERA,
                "coniferous": CONIFEROUS_GENERA,
            },
            "unclassified_genera": unclassified_genera,
            "conifer_analysis": {
                "n_genera": int(conifer_genera_count),
                "n_samples": int(conifer_sample_count),
                "include_in_analysis": bool(include_conifers),
                "reason": reason,
            },
            "analysis_scope": analysis_scope,
        },
    }

    output_path = METADATA_DIR / "chm_assessment.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved: {output_path}")

    log.end_step(status="success", records=1)

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Findings Summary
#


# %%
# ============================================================================
# FINDINGS SUMMARY
# ============================================================================

log.summary()
log.save(LOGS_DIR / f"{log.notebook}_execution.json")

analysis_files = []
config_files = [output_path] if output_path.exists() else []

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
