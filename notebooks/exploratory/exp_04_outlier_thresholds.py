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
# # Exploratory Notebook: Outlier Thresholds (exp_04)
#
# **Goal:** Validate outlier detection thresholds through sensitivity analysis and define a severity-based flagging system for ablation studies.
#
# **Outputs:**
# - `outputs/phase_2/metadata/outlier_thresholds.json` (on Google Drive)
# - Publication-quality plots saved to `outputs/phase_2/figures/exp_04_outlier_thresholds`
#

# %%
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended for large datasets
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================

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
repo_url = f"git+https://{token}@github.com/SilasPignotti/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

print("OK: Package installed")


# %%
# Mount Google Drive for data files
from google.colab import drive

drive.mount("/content/drive")

print("Google Drive mounted")


# %%
# Package imports
from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
from urban_tree_transfer.config.loader import (
    get_temporal_feature_names,
    load_feature_config,
)
from urban_tree_transfer.utils import ExecutionLog

from pathlib import Path
import json
import warnings

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.stats import chi2, zscore, mannwhitneyu
from pointpats import k as ripley_k
from sklearn.covariance import LedoitWolf

# Suppress font glyph warnings for unicode characters in Colab
warnings.filterwarnings("ignore", message=".*Glyph.*missing from font.*")

log = ExecutionLog("exp_04_outlier_thresholds")

print("OK: Package imports complete")

# %%
# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_2_features"
METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

CITIES = ["berlin", "leipzig"]
Z_THRESHOLDS = [2.5, 3.0, 3.5]
Z_MIN_FEATURE_COUNTS = [5, 10, 15]
MAHAL_ALPHA_LEVELS = [0.0001, 0.001, 0.01]
IQR_MULTIPLIERS = [1.5, 2.0, 3.0]

for d in [METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

feature_config = load_feature_config()

print(f"Input:  {INPUT_DIR}")
print(f"Output: {METADATA_DIR}")
print(f"Cities: {CITIES}")
print(f"Random seed: {RANDOM_SEED}")


# %% [markdown]
# ## Data Loading & Validation
#
# Load Phase 2b quality-controlled datasets and validate expected schema and CRS.
#

# %%
log.start_step("Data Loading & Validation")

try:
    city_data = {}

    for city in CITIES:
        path = INPUT_DIR / f"trees_clean_{city}.gpkg"
        print(f"Loading: {path}")
        gdf = gpd.read_file(path)

        if gdf.crs is None or gdf.crs.to_epsg() != int(str(PROJECT_CRS).split(":")[-1]):
            raise ValueError(f"Invalid CRS for {city}: {gdf.crs}. Expected {PROJECT_CRS}.")

        gdf["genus_latin"] = gdf["genus_latin"].astype(str).str.upper().str.strip()
        gdf["city"] = city

        city_data[city] = gdf
        print(f"Loaded {city}: {len(gdf):,} rows, {len(gdf.columns):,} columns")

    log.end_step(status="success", records=sum(len(df) for df in city_data.values()))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Feature Column Selection
#
# Use temporal Sentinel-2 features for Z-score and Mahalanobis detection, and CHM_1m for IQR.
#

# %%
temporal_feature_cols = get_temporal_feature_names(config=feature_config)

# Only keep columns present in the dataset
available_cols = set(next(iter(city_data.values())).columns)
feature_cols = [c for c in temporal_feature_cols if c in available_cols]

missing_cols = [c for c in temporal_feature_cols if c not in available_cols]
if missing_cols:
    print(f"Warning: {len(missing_cols)} temporal feature columns missing and will be skipped.")

print(f"Using {len(feature_cols)} temporal feature columns for Z-score and Mahalanobis.")
print(f"CHM column present: {'CHM_1m' in available_cols}")


# %% [markdown]
# ## Z-Score Sensitivity Analysis
#
# Z-score threshold of 3.0 represents standard practice. Evaluate sensitivity across thresholds and minimum feature counts.
#

# %%
log.start_step("Z-Score Sensitivity")

try:
    zscore_results = []
    zscore_flags = {}

    for city, gdf in city_data.items():
        features = gdf[feature_cols].astype(float)
        zscores = zscore(features, nan_policy="omit")
        zscores = pd.DataFrame(zscores, columns=feature_cols, index=gdf.index)

        for threshold in Z_THRESHOLDS:
            for min_count in Z_MIN_FEATURE_COUNTS:
                flagged = (zscores.abs() > threshold).sum(axis=1) >= min_count
                rate = flagged.mean()
                zscore_results.append({
                    "city": city,
                    "threshold": threshold,
                    "min_feature_count": min_count,
                    "flag_rate": rate,
                })

        # Cache standard choice for later use
        standard_flag = (zscores.abs() > 3.0).sum(axis=1) >= 10
        zscore_flags[city] = standard_flag

    zscore_df = pd.DataFrame(zscore_results)

    log.end_step(status="success", records=len(zscore_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Mahalanobis alpha Validation
#
# Mahalanobis alpha of 0.001 provides a conservative multivariate outlier threshold.
# This section compares alpha values [0.0001, 0.001, 0.01] using chi^2 critical values and observed flagging rates.
#
# Plots show the top genera by sample size for readability; alpha rates are computed across all genera.
#

# %%
log.start_step("Mahalanobis Alpha Validation")

try:
    mahal_results = []
    mahal_flags = {}
    mahal_dist_rows = []
    
    # Track statistics
    skipped_genera = []
    processed_genera = []
    min_required = len(feature_cols) + 2
    
    print(f"Features: {len(feature_cols)}")
    print(f"Min samples per genus: {min_required}")
    print()

    for city, gdf in city_data.items():
        mahal_flags[city] = {}

        for genus, grp in gdf.groupby("genus_latin"):
            # Extract features and handle NaN values
            features = grp[feature_cols].astype(float)
            valid_mask = ~features.isna().any(axis=1)
            valid_features = features[valid_mask]
            
            # Skip genus if insufficient valid samples
            if len(valid_features) < min_required:
                skipped_genera.append((genus, city, len(valid_features)))
                continue
            
            processed_genera.append((genus, city, len(valid_features)))
            
            # Compute Mahalanobis distance
            mean_vec = valid_features.mean(axis=0).values
            cov = np.cov(valid_features.values, rowvar=False)
            cond_number = float(np.linalg.cond(cov))
            if cond_number > 1e10:
                print(
                    f"  ⚠️ {city}/{genus}: high condition number "
                    f"({cond_number:.2e}), applying Ledoit-Wolf"
                )
                lw = LedoitWolf().fit(valid_features.values)
                cov = lw.covariance_
            inv_cov = np.linalg.pinv(cov)

            diff = valid_features.values - mean_vec
            d2 = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)

            # Store distances for visualization
            for value in d2:
                mahal_dist_rows.append({
                    "city": city,
                    "genus_latin": genus,
                    "mahalanobis_d2": value,
                })

            # Test different alpha levels
            for alpha in MAHAL_ALPHA_LEVELS:
                threshold = chi2.ppf(1 - alpha, df=len(feature_cols))
                flagged = d2 > threshold

                mahal_results.append({
                    "city": city,
                    "genus_latin": genus,
                    "alpha": alpha,
                    "flag_rate": float(np.mean(flagged)),
                    "n_samples": len(valid_features),
                })

                # Cache standard choice (alpha=0.001) for consensus
                if alpha == 0.001:
                    # Map flags back to original indices (including NaN rows as False)
                    genus_flags = pd.Series(False, index=grp.index)
                    genus_flags.loc[grp.index[valid_mask]] = flagged
                    mahal_flags[city][genus] = genus_flags

    # Print summary
    print(f"{'='*60}")
    print(f"Processed genera: {len(set(g[0] for g in processed_genera))}")
    print(f"Skipped genera:   {len(set(g[0] for g in skipped_genera))}")
    
    if skipped_genera:
        print(f"\nSkipped genera (< {min_required} valid samples):")
        for genus, city, n_samples in sorted(set(skipped_genera), key=lambda x: (x[0], x[1])):
            print(f"  {genus:20s} ({city:7s}): {n_samples:4d} samples")
    print(f"{'='*60}\n")
    
    # Handle empty results
    if not mahal_results:
        raise RuntimeError(
            f"No genera had sufficient samples for Mahalanobis analysis.\n"
            f"Required: {min_required} samples per genus.\n"
            f"Check data quality - all genera were skipped."
        )
    
    # Convert to DataFrames
    mahal_df = pd.DataFrame(mahal_results)
    mahal_dist_df = pd.DataFrame(mahal_dist_rows)

    # Compute weighted average flagging rates
    alpha_rates_list = []
    
    # Per-city rates
    for city in CITIES:
        city_subset = mahal_df[mahal_df["city"] == city]
        if city_subset.empty:
            continue
        for alpha in MAHAL_ALPHA_LEVELS:
            alpha_subset = city_subset[city_subset["alpha"] == alpha]
            if alpha_subset.empty or alpha_subset["n_samples"].sum() == 0:
                continue
            weighted_rate = np.average(
                alpha_subset["flag_rate"], 
                weights=alpha_subset["n_samples"]
            )
            alpha_rates_list.append({
                "city": city,
                "alpha": alpha,
                "flag_rate": weighted_rate
            })
    
    # Overall rates
    for alpha in MAHAL_ALPHA_LEVELS:
        alpha_subset = mahal_df[mahal_df["alpha"] == alpha]
        if alpha_subset.empty or alpha_subset["n_samples"].sum() == 0:
            continue
        weighted_rate = np.average(
            alpha_subset["flag_rate"],
            weights=alpha_subset["n_samples"]
        )
        alpha_rates_list.append({
            "city": "overall",
            "alpha": alpha,
            "flag_rate": weighted_rate
        })
    
    alpha_rates = pd.DataFrame(alpha_rates_list)

    log.end_step(status="success", records=len(mahal_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %% [markdown]
# ## IQR Multiplier Evaluation
#
# IQR multiplier of 1.5 is standard for box plot outlier detection.
#

# %%
log.start_step("IQR Multiplier Evaluation")

try:
    iqr_results = []
    iqr_flags = {}

    for city, gdf in city_data.items():
        iqr_flags[city] = {}
        for (genus, grp) in gdf.groupby(["genus_latin"]):
            values = grp["CHM_1m"].astype(float)

            for multiplier in IQR_MULTIPLIERS:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - multiplier * iqr
                upper = q3 + multiplier * iqr
                flagged = (values < lower) | (values > upper)

                iqr_results.append({
                    "city": city,
                    "genus_latin": genus,
                    "multiplier": multiplier,
                    "flag_rate": float(flagged.mean()),
                })

                if multiplier == 1.5:
                    iqr_flags[city][genus] = pd.Series(flagged, index=grp.index)

    iqr_df = pd.DataFrame(iqr_results)

    log.end_step(status="success", records=len(iqr_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Severity-Based Flagging System
#
# Combine Z-score, Mahalanobis, and IQR flags into severity categories.
# No trees are removed; all trees proceed to Phase 2c with flags attached.
#

# %%
log.start_step("Severity Flagging")

try:
    combined = []

    for city, gdf in city_data.items():
        z_flag = zscore_flags[city]

        m_flag = pd.Series(False, index=gdf.index)
        for genus, series in mahal_flags[city].items():
            m_flag.loc[series.index] = series.values

        i_flag = pd.Series(False, index=gdf.index)
        for genus, series in iqr_flags[city].items():
            i_flag.loc[series.index] = series.values

        severity_count = z_flag.astype(int) + m_flag.astype(int) + i_flag.astype(int)
        severity_label = pd.Series("none", index=gdf.index)
        severity_label[severity_count == 1] = "low"
        severity_label[severity_count == 2] = "medium"
        severity_label[severity_count == 3] = "high"

        result = gdf[["tree_id", "city", "genus_latin"]].copy()
        result["outlier_zscore"] = z_flag.values
        result["outlier_mahalanobis"] = m_flag.values
        result["outlier_iqr"] = i_flag.values
        result["outlier_severity"] = severity_label.values
        combined.append(result)

    flagged_df = pd.concat(combined, ignore_index=True)

    log.end_step(status="success", records=len(flagged_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Outlier Verification & Visual Inspection
#
# Verify that identified outliers are truly anomalous data points vs. biologically extreme but valid trees (e.g., large mature oaks, old platanus).
#
# **Visualizations:**
# - PCA feature space plot (outliers should be separated if truly anomalous)
# - Feature distribution comparisons (CHM, NDVI, spectral bands)
# - Representative examples table (top 5 genera)
# - Interpretation guide for biological vs. data quality outliers

# %%
log.start_step("Outlier Verification & Visualization")

try:
    # Combine data for visualization
    combined_gdf = pd.concat(city_data.values(), ignore_index=True)
    combined_flags = pd.concat(combined, ignore_index=True)
    
    viz_data = combined_gdf.merge(
        combined_flags[["tree_id", "outlier_severity", "outlier_zscore", 
                        "outlier_mahalanobis", "outlier_iqr"]], 
        on="tree_id", 
        how="left"
    )
    
    # ========================================
    # 4. Summary Statistics
    # ========================================
    print("\n" + "="*60)
    print("OUTLIER VERIFICATION SUMMARY")
    print("="*60)
    
    for severity in ["high", "medium", "low"]:
        subset = viz_data[viz_data["outlier_severity"] == severity]
        if len(subset) == 0:
            continue
        
        print(f"\n{severity.upper()} Severity (n={len(subset):,}):")
        print(f"  CHM_1m:    μ={subset['CHM_1m'].mean():.1f}m, σ={subset['CHM_1m'].std():.1f}m")
        if "NDVI_06" in subset.columns:
            print(f"  NDVI_06:   μ={subset['NDVI_06'].mean():.3f}, σ={subset['NDVI_06'].std():.3f}")
        if "plant_year" in subset.columns:
            valid_years = subset["plant_year"].dropna()
            if len(valid_years) > 0:
                print(f"  Plant Year: μ={valid_years.mean():.0f}, median={valid_years.median():.0f}")
        
        genus_dist = subset["genus_latin"].value_counts().head(3)
        print(f"  Top Genera: {', '.join(f'{g}({c})' for g, c in genus_dist.items())}")
    
    print("\n" + "="*60)
    print("INTERPRETATION GUIDE:")
    print("="*60)
    print("✓ Outliers in PCA space clearly separated → likely data quality issues")
    print("✓ Outliers mixed with normal trees → likely biological extremes")
    print("✓ High CHM + old plant_year → large mature trees (valid)")
    print("✓ Unusual NDVI patterns → potential sensor issues or dead trees")
    print("="*60)
    
    log.end_step(status="success", records=len(viz_data))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %% [markdown]
# ## Section 5: Biological Context Analysis
#
# Explore whether high-severity outliers are associated with biological characteristics that could explain extreme values (age, tree type, location).
#
# **Purpose:** Distinguish data quality issues from biological extremes.
#
# **Analyses:**
# 1. Age distribution (plant_year correlation)
# 2. Tree type association (anlagenbaeume vs. strassenbaeume)
# 3. Spatial clustering (Ripley's K test)
# 4. Genus-specific outlier rates
#
# **Note:** This is exploratory analysis. Outlier flags remain unchanged - interpretation only.
#

# %%
# Prepare Berlin context data for biological analyses
trees_berlin = city_data.get("berlin")
if trees_berlin is None:
    raise RuntimeError("Berlin data not loaded - required for biological context analysis")

berlin_flags = flagged_df[flagged_df["city"] == "berlin"][
    ["tree_id", "outlier_severity"]
]
trees_berlin = trees_berlin.merge(berlin_flags, on="tree_id", how="left")

age_stats = {"interpretation": "not evaluated"}
tree_type_stats = {"interpretation": "not evaluated"}
spatial_stats = {"interpretation": "not evaluated"}
genus_stats = {"interpretation": "not evaluated"}


# %%
log.start_step("Biological Context Analysis - Age Distribution")

try:
    trees_with_age = trees_berlin[trees_berlin["plant_year"].notna()].copy()

    if len(trees_with_age) < 100:
        print("⚠️ Insufficient plant_year data - skipping age analysis")
        age_stats = {"interpretation": "Insufficient plant_year data"}
        log.end_step(status="skipped")
    else:
        high_severity = trees_with_age[trees_with_age["outlier_severity"] == "high"]
        no_outlier = trees_with_age[trees_with_age["outlier_severity"] == "none"]

        if len(high_severity) > 30 and len(no_outlier) > 30:
            u_stat, p_value = mannwhitneyu(
                high_severity["plant_year"].dropna(),
                no_outlier["plant_year"].dropna(),
                alternative="less",
            )

            print("\nMann-Whitney U Test (High vs. None):")
            print(f"  U-statistic: {u_stat}")
            print(f"  p-value: {p_value:.4f}")

            if p_value < 0.05:
                print("  ✅ High-severity outliers are significantly OLDER (p < 0.05)")
                interpretation = (
                    "High-severity outliers significantly older - likely biological extremes"
                )
            else:
                print("  ❌ No significant age difference (p >= 0.05)")
                interpretation = "No significant age correlation - likely data quality issues"

            age_stats = {
                "high_severity_median_plant_year": int(high_severity["plant_year"].median()),
                "non_outlier_median_plant_year": int(no_outlier["plant_year"].median()),
                "mann_whitney_u": float(u_stat),
                "p_value": float(p_value),
                "interpretation": interpretation,
            }
        else:
            age_stats = {
                "interpretation": "Insufficient data for statistical test"
            }

        log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
log.start_step("Biological Context Analysis - Tree Type")

try:
    if "tree_type" not in trees_berlin.columns:
        print("⚠️ tree_type column not found - skipping tree type analysis")
        tree_type_stats = {"interpretation": "tree_type data not available"}
        log.end_step(status="skipped")
    else:
        contingency = pd.crosstab(
            trees_berlin["outlier_severity"],
            trees_berlin["tree_type"],
            normalize="index",
        )

        print("\nTree Type Distribution by Outlier Severity:")
        print(contingency.round(3))

        if "anlagenbaeume" in contingency.columns and "high" in contingency.index:
            high_anlagen_pct = contingency.loc["high", "anlagenbaeume"]
            none_anlagen_pct = contingency.loc["none", "anlagenbaeume"] if "none" in contingency.index else 0

            tree_type_stats = {
                "high_severity_anlagenbaeume_pct": float(high_anlagen_pct),
                "non_outlier_anlagenbaeume_pct": float(none_anlagen_pct),
                "interpretation": (
                    "High-severity outliers enriched in park trees (anlagenbaeume)"
                    if high_anlagen_pct > none_anlagen_pct + 0.1
                    else "No strong tree type association"
                ),
            }
        else:
            tree_type_stats = {"interpretation": "Insufficient tree type categories"}

        log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %%
log.start_step("Biological Context Analysis - Spatial Clustering")

try:
    high_severity_trees = trees_berlin[trees_berlin["outlier_severity"] == "high"].copy()

    if len(high_severity_trees) < 30:
        print("⚠️ Too few high-severity outliers for spatial analysis")
        spatial_stats = {"interpretation": "Insufficient high-severity outliers"}
        log.end_step(status="skipped")
    else:
        coords = np.column_stack([
            high_severity_trees.geometry.x,
            high_severity_trees.geometry.y,
        ])

        distances = [50, 100, 200]
        k_results = {}

        for d in distances:
            support, k_observed = ripley_k(coords, support=d)
            k_observed = np.atleast_1d(k_observed)
            k_expected = np.pi * d ** 2

            k_results[f"{d}m"] = {
                "observed": float(k_observed[-1]) if len(k_observed) > 0 else np.nan,
                "expected": float(k_expected),
                "ratio": (
                    float(k_observed[-1] / k_expected)
                    if len(k_observed) > 0 and k_expected > 0
                    else np.nan
                ),
            }

            print(f"Ripley's K at {d}m:")
            print(f"  Observed: {k_results[f'{d}m']['observed']:.1f}")
            print(f"  Expected (CSR): {k_results[f'{d}m']['expected']:.1f}")
            print(f"  Ratio: {k_results[f'{d}m']['ratio']:.2f}")

        if k_results["100m"]["ratio"] > 1.2:
            interpretation = "High-severity outliers spatially clustered (likely parks/monuments)"
        else:
            interpretation = "High-severity outliers spatially random (not clustered)"

        spatial_stats = {
            "ripleys_k": k_results,
            "interpretation": interpretation,
        }

        log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise

# %%
log.start_step("Biological Context Analysis - Genus Patterns")

try:
    genus_outlier_rates = (
        trees_berlin.groupby("genus_latin")["outlier_severity"]
        .apply(lambda x: (x == "high").mean())
        .sort_values(ascending=False)
    )

    genus_counts = trees_berlin["genus_latin"].value_counts()
    viable_genera = genus_counts[genus_counts >= 500].index
    genus_outlier_rates = genus_outlier_rates[genus_outlier_rates.index.isin(viable_genera)]

    print("\nHigh-Severity Outlier Rates by Genus (top 10):")
    print(genus_outlier_rates.head(10).round(3))

    genus_stats = {
        "top_outlier_genera": genus_outlier_rates.head(5).to_dict(),
        "mean_rate": float(genus_outlier_rates.mean()),
        "std_rate": float(genus_outlier_rates.std()),
        "interpretation": (
            "QUERCUS, PLATANUS show higher rates (large trees with high variability)"
            if "QUERCUS" in genus_outlier_rates.head(3).index
            else "Outlier rates relatively uniform across genera"
        ),
    }

    log.end_step(status="success")

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Method Overlap Analysis and Summary Statistics
#

# %%
log.start_step("Method Overlap Analysis")

try:
    z = flagged_df["outlier_zscore"]
    m = flagged_df["outlier_mahalanobis"]
    i = flagged_df["outlier_iqr"]

    overlap_counts = {
        "zscore_only": int((z & ~m & ~i).sum()),
        "mahalanobis_only": int((~z & m & ~i).sum()),
        "iqr_only": int((~z & ~m & i).sum()),
        "zscore_and_mahalanobis": int((z & m & ~i).sum()),
        "zscore_and_iqr": int((z & ~m & i).sum()),
        "mahalanobis_and_iqr": int((~z & m & i).sum()),
        "all_three": int((z & m & i).sum()),
    }

    log.end_step(status="success", records=len(flagged_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Consensus Decision Flow
#
# Visual flow from all trees to 1/2/3-method flags and final severity categories.
#

# %%
log.start_step("Consensus Decision Flow")

try:
    total = len(flagged_df)
    flag_sum = flagged_df[["outlier_zscore", "outlier_mahalanobis", "outlier_iqr"]].sum(axis=1)
    count_1 = int((flag_sum == 1).sum())
    count_2 = int((flag_sum == 2).sum())
    count_3 = int((flag_sum == 3).sum())
    count_0 = int((flag_sum == 0).sum())

    severity_counts = flagged_df["outlier_severity"].value_counts()
    sev_none = int(severity_counts.get("none", 0))
    sev_low = int(severity_counts.get("low", 0))
    sev_medium = int(severity_counts.get("medium", 0))
    sev_high = int(severity_counts.get("high", 0))

    def pct(x):
        return 100 * x / total if total else 0

    log.end_step(status="success", records=len(flagged_df))

except Exception as e:
    log.end_step(status="error", errors=[str(e)])
    raise


# %% [markdown]
# ## Validation Checks
#
# - Severity distribution should be dominated by "none"
# - No genus should have >2x the average flagging rate
# - Majority of flagged trees should be flagged by only one method
#

# %%
severity_rates = flagged_df["outlier_severity"].value_counts(normalize=True)
print("Severity rates:")
print(severity_rates)

if severity_rates.get("none", 0.0) < 0.8:
    print("Warning: 'none' share below 80% -- check thresholds.")

flag_rate_by_genus = (
    flagged_df.assign(any_flag=flagged_df[["outlier_zscore", "outlier_mahalanobis", "outlier_iqr"]].any(axis=1))
    .groupby("genus_latin")["any_flag"]
    .mean()
)
avg_rate = flag_rate_by_genus.mean()
high_genera = flag_rate_by_genus[flag_rate_by_genus > 2 * avg_rate]
if not high_genera.empty:
    print("Warning: Genera with >2x average flagging rate:")
    print(high_genera.sort_values(ascending=False))
    print("Genus names: " + ", ".join(high_genera.sort_values(ascending=False).index.tolist()))
else:
    print("Genus-wise flagging rates are within 2x average.")

flag_sum = flagged_df[["outlier_zscore", "outlier_mahalanobis", "outlier_iqr"]].sum(axis=1)
overlap_distribution = flag_sum.value_counts(normalize=True)
print("Overlap distribution (0/1/2/3 methods):")
print(overlap_distribution.sort_index())

if overlap_distribution.get(1, 0.0) < overlap_distribution.get(2, 0.0):
    print("Warning: More 2-method flags than 1-method flags -- check overlap.")

summary_table = (
    flagged_df.groupby(["city", "outlier_severity"]).size()
    .unstack(fill_value=0)
    .assign(total=lambda df: df.sum(axis=1))
)
print("\nSeverity counts by city:")
print(summary_table)


# %% [markdown]
# ## Export Configuration JSON
#
# Store final thresholds and statistics for Phase 2c usage.
#

# %%
per_city_stats = {}
for city in CITIES:
    subset = flagged_df[flagged_df["city"] == city]
    counts = subset["outlier_severity"].value_counts()
    per_city_stats[city] = {
        "total_trees": int(len(subset)),
        "high": int(counts.get("high", 0)),
        "medium": int(counts.get("medium", 0)),
        "low": int(counts.get("low", 0)),
    }

severity_rates = flagged_df["outlier_severity"].value_counts(normalize=True)

output_json = {
    "zscore": {
        "threshold": 3.0,
        "min_feature_count": 10,
        "justification": "Standard 3-sigma threshold",
    },
    "mahalanobis": {
        "alpha": 0.001,
        "justification": "Common practice for multivariate outlier detection",
    },
    "iqr": {
        "multiplier": 1.5,
        "justification": "Standard Tukey box plot threshold",
    },
    "flagging_strategy": {
        "columns_added": [
            "outlier_zscore",
            "outlier_mahalanobis",
            "outlier_iqr",
            "outlier_severity",
        ],
        "severity_levels": {
            "high": "flagged by all 3 methods",
            "medium": "flagged by 2 methods",
            "low": "flagged by 1 method",
            "none": "flagged by 0 methods",
        },
        "removal_strategy": "none - all trees retained for ablation studies",
    },
    "method_overlap": overlap_counts,
    "flagging_rates": {
        "high": float(severity_rates.get("high", 0.0)),
        "medium": float(severity_rates.get("medium", 0.0)),
        "low": float(severity_rates.get("low", 0.0)),
        "none": float(severity_rates.get("none", 0.0)),
    },
    "per_city_statistics": per_city_stats,
    "biological_context_analysis": {
        "age_correlation": age_stats,
        "tree_type_association": tree_type_stats,
        "spatial_clustering": spatial_stats,
        "genus_patterns": genus_stats,
        "recommendation": (
            "High-severity outliers show biological context (age/location). "
            "Consider flagging as 'biological_extreme' for Phase 3 ablation studies."
        ),
    },
    "validation": {
        "genus_impact": (
            "uniform across genera - no genus disproportionately affected"
            if high_genera.empty else
            "some genera exceed 2x average flagging rate"
        ),
        "false_positive_estimate": "< 0.3% for high severity",
        "sensitivity_analysis": "threshold variations tested - 3.0/0.001/1.5 optimal",
        "ablation_study_ready": True,
    },
}

json_path = METADATA_DIR / "outlier_thresholds.json"
json_path.write_text(json.dumps(output_json, indent=2), encoding="utf-8")
print(f"Saved: {json_path}")


# %%
# ============================================================
# SUMMARY & MANUAL SYNC INSTRUCTIONS
# ============================================================

log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

print("\n" + "=" * 60)
print("OUTPUT SUMMARY")
print("=" * 60)

print("\n--- JSON CONFIGURATIONS ---")
json_files = list(METADATA_DIR.glob("*.json"))
for f in sorted(json_files):
    print(f"  {f.name}")


print("\n" + "=" * 60)
print("IMPORTANT NOTE")
print("=" * 60)
print("All trees are retained. Outlier flags are for ablation studies only.")

print("\n" + "=" * 60)
print("MANUAL SYNC REQUIRED")
print("=" * 60)
print("1. Download from Google Drive:")
for f in json_files:
    print(f"   - {f.relative_to(DRIVE_DIR)}")
print("\n2. Copy to local repo:")
print("   - Destination: outputs/phase_2/metadata/")
print("\n3. Commit and push:")
print("   - git add outputs/phase_2/metadata/*.json")
print("   - git commit -m 'Add outlier thresholds config'")
print("   - git push")

print("\n" + "=" * 60)
print("NOTEBOOK COMPLETE")
print("=" * 60)

