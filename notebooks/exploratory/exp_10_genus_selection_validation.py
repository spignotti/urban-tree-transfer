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
#     name: python3
# ---

# %% [markdown] id="nVP7bQwfsIlt"
# # exp_10: Genus Selection Validation
#
# **Purpose:** Validate genus sample counts after applying setup decisions and define final genus classes for Phase 3.
#
# **Approach:**
# 1. Load setup decisions from exp_09 (CHM, proximity, outlier, feature selection)
# 2. Apply all filtering strategies to all Phase 2 splits
# 3. Validate sample counts across all splits (≥500 per genus in both cities)
# 4. Compute JM-based separability on **Berlin train only** (no leakage)
# 5. Group poorly separable genera using hierarchical clustering
# 6. Validate split stratification after grouping (KL-divergence)
# 7. Extend `setup_decisions.json` with `genus_selection`
#
# **Output:**
# - Updates `setup_decisions.json` (genus_selection section)
# - Figures: `genus_sample_counts.png`, `jm_separability_heatmap.png`, `jm_dendrogram.png`, `genus_groups_overview.png`
#

# %% colab={"base_uri": "https://localhost:8080/"} id="Xh_DviNesIlw" outputId="42c9a06f-1977-4c56-bd11-e99b50b8c153"
# ============================================================
# RUNTIME SETTINGS
# ============================================================
# Required: CPU (Standard)
# GPU: Not required
# High-RAM: Recommended (for hierarchical clustering)
#
# SETUP: Add GITHUB_TOKEN to Colab Secrets (key icon in sidebar)
# ============================================================

import subprocess
from google.colab import userdata

# Get GitHub token from Colab Secrets (for private repo access)
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise ValueError(
        """GITHUB_TOKEN not found in Colab Secrets.
1. Click the key icon in the left sidebar
2. Add a secret named 'GITHUB_TOKEN' with your GitHub token
3. Toggle 'Notebook access' ON"""
    )

# Install package from private GitHub repo
repo_url = f"git+https://{token}@github.com/SilasPignotti/urban-tree-transfer.git"
subprocess.run(["pip", "install", repo_url, "-q"], check=True)

print("OK: Package installed")

# %% colab={"base_uri": "https://localhost:8080/"} id="1kKs9Uo2sIly" outputId="d880691a-afde-418d-99ca-e49b226f260b"
# Mount Google Drive for data files
from google.colab import drive

drive.mount("/content/drive")

print("Google Drive mounted")


# %% colab={"base_uri": "https://localhost:8080/"} id="zEWDUVpFsIly" outputId="79e950ff-e651-447e-83e6-f83f86246b66"
# Package imports
from urban_tree_transfer.config import MIN_SAMPLES_PER_GENUS, RANDOM_SEED
from urban_tree_transfer.experiments import ablation, data_loading
from urban_tree_transfer.utils import ExecutionLog

from pathlib import Path
from datetime import datetime, timezone
import json
import warnings

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import entropy

log = ExecutionLog("exp_10_genus_selection_validation")

warnings.filterwarnings("ignore", category=UserWarning)

print("OK: Package imports complete")


# %% colab={"base_uri": "https://localhost:8080/"} id="EhUhYIJLsIlz" outputId="dda80409-1c36-4137-d39a-4b4815e03c0f"
# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_DIR = Path("/content/drive/MyDrive/dev/urban-tree-transfer")
INPUT_DIR = DRIVE_DIR / "data" / "phase_2_splits"
OUTPUT_DIR = DRIVE_DIR / "data" / "phase_3_experiments"

METADATA_DIR = OUTPUT_DIR / "metadata"
LOGS_DIR = OUTPUT_DIR / "logs"

for d in [METADATA_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Analysis parameters
MIN_SAMPLES = MIN_SAMPLES_PER_GENUS  # 500
CLUSTERING_METHOD = "ward"  # Ward linkage for hierarchical clustering
PERCENTILE = 10  # Default: lower 10% grouped
KL_THRESHOLD = 0.15  # Maximum KL-divergence after filtering

print(f"Input (Phase 2 Splits): {INPUT_DIR}")
print(f"Output (Phase 3):       {OUTPUT_DIR}")
print(f"Metadata:               {METADATA_DIR}")
print(f"Logs:                   {LOGS_DIR}")
print(f"Random seed:            {RANDOM_SEED}")
print(f"Min samples per genus:  {MIN_SAMPLES}")


# %% [markdown] id="3w-kbWKesIl0"
# # ============================================================
# # SECTION 1: Load Setup Decisions & All Splits
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="t2eWKxdasIl0" outputId="a9b090a4-d927-4c6b-9ee9-93151e8020b6"
log.start_step("Load Setup & Data")

# Load setup decisions from exp_09
setup_path = METADATA_DIR / "setup_decisions.json"

if not setup_path.exists():
    raise FileNotFoundError(
        f"""setup_decisions.json not found at {setup_path}
"""
        "Run exploratory notebooks exp_08 → exp_08b → exp_08c → exp_09 first."
    )

with open(setup_path) as f:
    setup_config = json.load(f)

# Validate required keys
required_keys = ["chm_strategy", "proximity_strategy", "outlier_strategy", "feature_set", "selected_features"]
missing = [k for k in required_keys if k not in setup_config]
if missing:
    raise ValueError(
        f"""Missing setup decisions: {missing}
"""
        "Run exp_08 → exp_08b → exp_08c → exp_09 in order."
    )

print("✅ Loaded setup decisions:")
print(f"   CHM Strategy: {setup_config['chm_strategy']['decision']}")
print(f"   Proximity Strategy: {setup_config['proximity_strategy']['decision']}")
print(f"   Outlier Strategy: {setup_config['outlier_strategy']['decision']}")
print(f"   Selected Features: {len(setup_config['selected_features'])}")

log.end_step(status="success")


# %% [markdown] id="pk4T2S_MsIl0"
# # ============================================================
# # SECTION 2: Apply Setup Decisions to All Splits
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="AzAJVWYUsIl1" outputId="f44b2859-1b4b-476a-ce27-b26b026dcd5c"
log.start_step("Apply Setup Decisions")

print("Applying setup decisions to all splits...")

city_splits = [
    ("berlin", "train"),
    ("berlin", "val"),
    ("berlin", "test"),
    ("leipzig", "finetune"),
    ("leipzig", "test"),
]

processed_splits = {}
meta_by_split = {}

for city, split in city_splits:
    split_name = f"{city}_{split}"
    df, meta = ablation.prepare_ablation_dataset(
        base_path=INPUT_DIR,
        city=city,
        split=split,
        setup_decisions=setup_config,
        return_metadata=True,
        optimize_memory=True,
    )

    # Fix missing German names (PRUNUS/SOPHORA)
    df = data_loading.fix_missing_genus_german(df)

    processed_splits[split_name] = df
    meta_by_split[split_name] = meta
    print(f"   {split_name}: {meta['original_n_samples']:,} → {len(df):,} samples")

# Extract feature columns (exclude metadata)
metadata_cols = ablation.get_metadata_columns(processed_splits["berlin_train"])
feature_cols = [col for col in processed_splits["berlin_train"].columns if col not in metadata_cols]

print(f"\n✅ Setup decisions applied")
print(f"   Feature columns: {len(feature_cols)}")
print(f"   Total processed samples: {sum(len(df) for df in processed_splits.values()):,}")

log.end_step(status="success")


# %% [markdown] id="EF9EzVEWsIl1"
# # ============================================================
# # SECTION 3: Sample Count Validation (All Splits)
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="685anOpysIl2" outputId="e9790133-7665-4f8b-a3f3-990c4faa76c7"
log.start_step("Sample Count Validation")

print("Validating sample counts across ALL splits...\n")

# Count samples per genus in each split
genus_counts = {}
for split_name, split_df in processed_splits.items():
    genus_counts[split_name] = split_df["genus_latin"].value_counts()

# Get all unique genera
all_genera = sorted(set().union(*[set(counts.index) for counts in genus_counts.values()]))

# Create comprehensive count table
count_df = pd.DataFrame(
    {
        split_name: [genus_counts[split_name].get(g, 0) for g in all_genera]
        for split_name in processed_splits.keys()
    },
    index=all_genera,
)

# Add aggregate columns
count_df["Berlin_Total"] = count_df[["berlin_train", "berlin_val", "berlin_test"]].sum(axis=1)
count_df["Leipzig_Total"] = count_df[["leipzig_finetune", "leipzig_test"]].sum(axis=1)
count_df["Min_Per_City"] = count_df[["Berlin_Total", "Leipzig_Total"]].min(axis=1)
count_df["Viable"] = count_df["Min_Per_City"] >= MIN_SAMPLES

print("Sample Counts per Genus (All Splits):")
print(count_df.to_string())

# Identify viable and excluded genera
viable_genera = count_df[count_df["Viable"]].index.tolist()
excluded_genera = count_df[~count_df["Viable"]].index.tolist()

print(f"\n✅ Sample count validation complete")
print(f"   Viable genera (≥{MIN_SAMPLES} in both cities): {len(viable_genera)}")
print(f"   Excluded genera (<{MIN_SAMPLES} in at least one city): {len(excluded_genera)}")
if excluded_genera:
    print(f"\n   Excluded genera:")
    for genus in excluded_genera:
        berlin_count = count_df.loc[genus, "Berlin_Total"]
        leipzig_count = count_df.loc[genus, "Leipzig_Total"]
        print(f"      {genus}: Berlin={berlin_count}, Leipzig={leipzig_count}")

log.end_step(status="success")


# %% [markdown] id="_0jkAfhnsIl2"
# # ============================================================
# # SECTION 4: Genus Separability Analysis (Berlin Train Only)
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="7fjhuHgHsIl3" outputId="e46c7c1f-138e-4256-b8cb-6f3f7d849c73"
log.start_step("Separability Analysis")

# Filter to viable genera only (Berlin train only for JM analysis)
berlin_train_viable = processed_splits["berlin_train"][
    processed_splits["berlin_train"]["genus_latin"].isin(viable_genera)
].copy()

print("Filtered to viable genera:")
print(f"   Berlin Train: {len(berlin_train_viable):,} samples ({len(viable_genera)} genera)")

# IMPORTANT: Only use Berlin train for separability analysis (no leakage, no city bias)
print("\n✅ Using Berlin train only for JM separability analysis")

log.end_step(status="success")


# %% colab={"base_uri": "https://localhost:8080/"} id="TxQC8zqasIl3" outputId="4e1ae093-7c37-46b9-e3e6-a141e57d9408"
# JM distance implementation (Bhattacharyya → Jeffries-Matusita)

def bhattacharyya_distance(X_i: np.ndarray, X_j: np.ndarray) -> float:
    # Compute Bhattacharyya distance between two multivariate Gaussians.
    mu_i = np.mean(X_i, axis=0)
    mu_j = np.mean(X_j, axis=0)

    cov_i = np.cov(X_i, rowvar=False)
    cov_j = np.cov(X_j, rowvar=False)

    cov_pool = (cov_i + cov_j) / 2.0
    cov_pool = cov_pool + np.eye(cov_pool.shape[0]) * 1e-6

    diff = mu_i - mu_j
    try:
        inv_cov = np.linalg.inv(cov_pool)
    except np.linalg.LinAlgError:
        inv_cov = np.linalg.pinv(cov_pool)

    mahal = float(diff.T @ inv_cov @ diff)

    det_pool = float(np.linalg.det(cov_pool))
    det_i = float(np.linalg.det(cov_i))
    det_j = float(np.linalg.det(cov_j))

    if det_pool <= 0 or det_i <= 0 or det_j <= 0:
        det_term = 0.0
    else:
        det_term = float(np.log(det_pool / np.sqrt(det_i * det_j)))

    return 0.125 * mahal + 0.5 * det_term


def jeffries_matusita_distance(X_i: np.ndarray, X_j: np.ndarray) -> float:
    # Compute Jeffries-Matusita distance (bounded [0, 2]).
    b_distance = bhattacharyya_distance(X_i, X_j)
    jm = 2.0 * (1.0 - np.exp(-b_distance))
    return float(np.clip(jm, 0.0, 2.0))


def compute_jm_matrix(df: pd.DataFrame, feature_cols: list[str], class_col: str = "genus_latin") -> pd.DataFrame:
    # Compute pairwise JM distances for all classes.
    classes = sorted(df[class_col].unique())
    n_classes = len(classes)
    jm_matrix = np.zeros((n_classes, n_classes))

    for i, class_i in enumerate(classes):
        X_i = df[df[class_col] == class_i][feature_cols].values
        for j, class_j in enumerate(classes):
            if i >= j:
                continue
            X_j = df[df[class_col] == class_j][feature_cols].values
            jm = jeffries_matusita_distance(X_i, X_j)
            jm_matrix[i, j] = jm
            jm_matrix[j, i] = jm

    return pd.DataFrame(jm_matrix, index=classes, columns=classes)

print("Computing JM matrix...")

jm_matrix = compute_jm_matrix(berlin_train_viable, feature_cols, class_col="genus_latin")

jm_values = jm_matrix.values[np.triu_indices_from(jm_matrix.values, k=1)]

jm_stats = {
    "min": float(np.min(jm_values)),
    "q10": float(np.percentile(jm_values, 10)),
    "q25": float(np.percentile(jm_values, 25)),
    "median": float(np.median(jm_values)),
    "q75": float(np.percentile(jm_values, 75)),
    "max": float(np.max(jm_values)),
}

# Adaptive threshold strategy
if jm_stats["q10"] >= 1.5:
    threshold = 0.0
    threshold_strategy = "none_all_separable"
    print("✅ All genera well separable (Q10 >= 1.5) - no grouping required")
elif jm_stats["median"] < 1.0:
    threshold = float(np.percentile(jm_values, 30))
    threshold_strategy = "percentile_30_low_median"
    print(f"⚠️ Many genera poorly separable (median < 1.0) - threshold: {threshold:.2f}")
else:
    threshold = float(np.percentile(jm_values, PERCENTILE))
    threshold_strategy = f"percentile_{PERCENTILE}"
    print(f"ℹ️ Threshold (P{PERCENTILE}): {threshold:.2f}")

print("\nJM Statistics:")
for k, v in jm_stats.items():
    print(f"  {k:>6}: {v:.3f}")

# %% [markdown] id="zj-rgv3ssIl4"
# # ============================================================
# # SECTION 5: Hierarchical Clustering & Grouping
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="aHi3849usIl5" outputId="25fdac3e-0d45-4e27-ac60-aec46c3283ee"
# Use JM distances directly (Lower JM = more similar = closer in clustering)
clustering_distances = jm_matrix.values.copy()
np.fill_diagonal(clustering_distances, 0) # Ensure diagonal is strictly zero
condensed = squareform(clustering_distances)

linkage_matrix = linkage(condensed, method=CLUSTERING_METHOD)
max_dist = linkage_matrix[:, 2].max()

# Cluster assignment
if threshold <= 0:
    cluster_labels = list(range(1, len(jm_matrix.index) + 1))
    print("No grouping applied - all genera kept separate")
else:
    # Use threshold directly
    distance_threshold = threshold
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

# Create genus -> cluster mapping
genus_to_cluster = dict(zip(jm_matrix.index, cluster_labels))

# Group genera by cluster
cluster_to_genera = {}
for genus, cluster_id in genus_to_cluster.items():
    cluster_to_genera.setdefault(cluster_id, []).append(genus)

# Identify groups (clusters with >1 genus)
genus_groups = {}
for cluster_id, genera_list in cluster_to_genera.items():
    if len(genera_list) > 1:
        genera_list_sorted = sorted(genera_list)
        group_name = f"Group {cluster_id}"
        genus_groups[group_name] = genera_list_sorted

print("\n✅ Clustering complete")
print(f"   Total clusters: {len(cluster_to_genera)}")
print(f"   Singleton genera: {sum(1 for g in cluster_to_genera.values() if len(g) == 1)}")
print(f"   Multi-genus groups: {len(genus_groups)}")

if genus_groups:
    print("\nIdentified genus groups:")
    for group_name, genera_list in genus_groups.items():
        print(f"   {group_name}: {', '.join(genera_list)} ({len(genera_list)} genera)")
else:
    print("\nNo genus groups identified - all genera sufficiently separable")


# %% [markdown] id="QrqIAcRasIl6"
# # ============================================================
# # SECTION 6: Create Final Genus List
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="ld14v_P6sIl6" outputId="e77adbc7-243f-4dbd-9382-f6deeafea2b2"
# Create final genus list (replace grouped genera with group names)
final_genera_list = []
genus_to_final = {}

# Add singleton genera
for cluster_id, genera_list in cluster_to_genera.items():
    if len(genera_list) == 1:
        genus = genera_list[0]
        final_genera_list.append(genus)
        genus_to_final[genus] = genus

# Add groups
for group_name, genera_list in genus_groups.items():
    final_genera_list.append(group_name)
    for genus in genera_list:
        genus_to_final[genus] = group_name

final_genera_list = sorted(final_genera_list)

print("✅ Final genus list created")
print(f"   Original viable genera: {len(viable_genera)}")
print(f"   Final classes (after grouping): {len(final_genera_list)}")
print(f"   Reduction: {len(viable_genera) - len(final_genera_list)} genera merged")

# Build German name mapping
# Ensure missing German names fixed before mapping
# Already fixed in Section 2, but re-applying for safety
berlin_train_viable = data_loading.fix_missing_genus_german(berlin_train_viable)

# Base mapping for singletons
base_german = (
    berlin_train_viable.groupby("genus_latin")["genus_german"]
    .first()
    .to_dict()
)

genus_german_mapping = {}

# Singletons
for genus in viable_genera:
    if genus_to_final[genus] == genus:
        genus_german_mapping[genus] = base_german.get(genus, genus)

# Groups
for group_name, genera_list in genus_groups.items():
    german_names = [base_german.get(g, g) for g in genera_list]
    genus_german_mapping[group_name] = " / ".join(german_names)

# Validate mapping completeness
missing_german = [g for g in final_genera_list if g not in genus_german_mapping]
if missing_german:
    raise ValueError(f"Missing German names for: {missing_german}")

print("\n✅ German name mapping created")
print(f"   Mapped classes: {len(genus_german_mapping)}")


# %% colab={"base_uri": "https://localhost:8080/"} id="YqsL_eRUsIl7" outputId="88678108-f876-42c4-8299-aa28b22ecbbb"
# Calculate sample counts for final classes (Berlin train + Leipzig finetune)
print("Calculating sample counts for final classes...")

berlin_train_mapped = berlin_train_viable.copy()
berlin_train_mapped["final_class"] = berlin_train_mapped["genus_latin"].map(genus_to_final)

leipzig_finetune_viable = processed_splits["leipzig_finetune"][
    processed_splits["leipzig_finetune"]["genus_latin"].isin(viable_genera)
].copy()
leipzig_finetune_mapped = leipzig_finetune_viable.copy()
leipzig_finetune_mapped["final_class"] = leipzig_finetune_mapped["genus_latin"].map(genus_to_final)

berlin_final_counts = berlin_train_mapped["final_class"].value_counts().sort_index()
leipzig_final_counts = leipzig_finetune_mapped["final_class"].value_counts().sort_index()

final_counts_df = pd.DataFrame(
    {
        "Berlin": berlin_final_counts,
        "Leipzig": leipzig_final_counts,
    }
)
final_counts_df["Total"] = final_counts_df["Berlin"] + final_counts_df["Leipzig"]

print("✅ Sample counts for final classes:")
print(final_counts_df.to_string())
print(f"Total samples (train+finetune): {final_counts_df['Total'].sum():,}")


# %% [markdown] id="QVhtJ-UwsIl7"
# # ============================================================
# # SECTION 7: Validate Split Stratification (KL-Divergence)
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="ENYWEnLcsIl8" outputId="a786aa90-717f-4af1-f1c3-65f9684703c0"
log.start_step("KL-Divergence Validation")

print("Validating that genus filtering maintains split stratification...\n")

# Filter all splits to viable genera and apply genus mapping
splits_mapped = {}
for split_name, split_df in processed_splits.items():
    df = split_df[split_df["genus_latin"].isin(viable_genera)].copy()
    df["genus_latin"] = df["genus_latin"].map(genus_to_final)
    df["genus_german"] = df["genus_latin"].map(genus_german_mapping)

    if df["genus_german"].isna().any():
        missing = df[df["genus_german"].isna()]["genus_latin"].unique().tolist()
        raise ValueError(f"Missing German names after mapping in {split_name}: {missing}")

    splits_mapped[split_name] = df
    print(f"   {split_name}: {len(split_df):,} → {len(df):,} samples")

# Calculate KL-divergence between splits

def compute_kl_divergence(split1_df, split2_df, class_col="genus_latin"):
    # Compute KL-divergence between class distributions.
    dist1 = split1_df[class_col].value_counts(normalize=True).sort_index()
    dist2 = split2_df[class_col].value_counts(normalize=True).sort_index()

    all_classes = sorted(set(dist1.index) | set(dist2.index))
    dist1 = dist1.reindex(all_classes, fill_value=1e-10)
    dist2 = dist2.reindex(all_classes, fill_value=1e-10)

    return float(entropy(dist1, dist2))

print("\nComputing KL-divergence between splits...")

kl_berlin_train_val = compute_kl_divergence(splits_mapped["berlin_train"], splits_mapped["berlin_val"])
kl_berlin_train_test = compute_kl_divergence(splits_mapped["berlin_train"], splits_mapped["berlin_test"])
kl_berlin_val_test = compute_kl_divergence(splits_mapped["berlin_val"], splits_mapped["berlin_test"])

kl_leipzig_finetune_test = compute_kl_divergence(
    splits_mapped["leipzig_finetune"],
    splits_mapped["leipzig_test"],
)

kl_results = {
    "berlin": {
        "train_vs_val": kl_berlin_train_val,
        "train_vs_test": kl_berlin_train_test,
        "val_vs_test": kl_berlin_val_test,
    },
    "leipzig": {
        "finetune_vs_test": kl_leipzig_finetune_test,
    },
}

print(f"\nKL-Divergence Results (Threshold: {KL_THRESHOLD}):")
print("\nBerlin:")
print(f"   Train vs Val:  {kl_berlin_train_val:.4f} {'✅' if kl_berlin_train_val < KL_THRESHOLD else '❌'}")
print(f"   Train vs Test: {kl_berlin_train_test:.4f} {'✅' if kl_berlin_train_test < KL_THRESHOLD else '❌'}")
print(f"   Val vs Test:   {kl_berlin_val_test:.4f} {'✅' if kl_berlin_val_test < KL_THRESHOLD else '❌'}")

print("\nLeipzig:")
print(f"   Finetune vs Test: {kl_leipzig_finetune_test:.4f} {'✅' if kl_leipzig_finetune_test < KL_THRESHOLD else '❌'}")

all_kl_values = [
    kl_berlin_train_val,
    kl_berlin_train_test,
    kl_berlin_val_test,
    kl_leipzig_finetune_test,
]
if all(kl < KL_THRESHOLD for kl in all_kl_values):
    print("\n✅ All KL-divergences below threshold - stratification preserved")
else:
    print("\n⚠️ Some KL-divergences exceed threshold - review grouping impact")

log.end_step(status="success")


# %% [markdown] id="UZJMtJ2GsIl8"
# # ============================================================
# # SECTION 8: Export Final Configuration
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="UrQ-7JvRsIl9" outputId="db27ecc3-4525-4740-9165-11cc82fd4568"
# Build final configuration JSON
output_config = {
    "version": "2.0",
    "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "notebook": "exp_10_genus_selection_validation.ipynb",
    "method": "jm_distance_based_separability",
    "config": {
        "min_samples_per_genus": MIN_SAMPLES,
        "sample_count_validation": "all_splits",
        "separability_analysis": "berlin_train_only",
        "clustering_method": CLUSTERING_METHOD,
        "distance_metric": "jeffries_matusita",
        "threshold_strategy": threshold_strategy,
        "threshold_value": float(threshold),
        "kl_threshold": KL_THRESHOLD,
    },
    "jm_statistics": jm_stats,
    "validation_results": {
        "viable_genera": viable_genera,
        "excluded_genera": excluded_genera,
        "n_viable": len(viable_genera),
        "n_excluded": len(excluded_genera),
        "exclusion_reasons": {
            genus: f"Berlin: {count_df.loc[genus, 'Berlin_Total']}, Leipzig: {count_df.loc[genus, 'Leipzig_Total']}"
            for genus in excluded_genera
        } if excluded_genera else {},
    },
    "grouping_analysis": {
        "n_clusters": len(cluster_to_genera),
        "n_groups": len(genus_groups),
        "genus_groups": genus_groups,
        "genus_to_final_mapping": genus_to_final,
        "genus_german_mapping": genus_german_mapping,
    },
    "decision": {
        "strategy_applied": "jm_based_grouping_relative_threshold",
        "final_genera_count": len(final_genera_list),
        "final_genera_list": final_genera_list,
        "grouping_applied": len(genus_groups) > 0,
        "reasoning": (
            f"{len(viable_genera)} genera have ≥{MIN_SAMPLES} samples in both cities. "
            f"JM-based separability (threshold={threshold:.2f}) identified {len(genus_groups)} groups. "
            f"Final dataset has {len(final_genera_list)} classes."
        ),
    },
    "impact_assessment": {
        "samples_retained": sum(len(df) for df in splits_mapped.values()),
        "retention_rate": sum(len(df) for df in splits_mapped.values())
        / sum(len(df) for df in processed_splits.values()),
        "final_class_counts": {
            "berlin_train": berlin_final_counts.to_dict(),
            "leipzig_finetune": leipzig_final_counts.to_dict(),
        },
    },
    "kl_divergence_validation": kl_results,
}

# Extend setup_decisions.json with genus selection
setup_config["genus_selection"] = output_config

with open(setup_path, "w") as f:
    json.dump(setup_config, f, indent=2)

print("✅ Extended setup_decisions.json with genus_selection")
print("Final Summary:")
print(f"   Original genera (Phase 1): {len(all_genera)}")
print(f"   Viable genera (≥{MIN_SAMPLES}): {len(viable_genera)}")
print(f"   Excluded genera: {len(excluded_genera)}")
print(f"   Genus groups formed: {len(genus_groups)}")
print(f"   Final classes: {len(final_genera_list)}")
print(f"   Total samples retained: {sum(len(df) for df in splits_mapped.values()):,}")
print(f"   Retention rate: {output_config['impact_assessment']['retention_rate'] * 100:.1f}%")


# %% [markdown] id="KwPXYYb-sIl9"
# # ============================================================
# # SECTION 9: Summary & Next Steps
# # ============================================================
#

# %% colab={"base_uri": "https://localhost:8080/"} id="4K_KnnsZsIl9" outputId="57cb9ba9-93a4-456a-896b-9d2d0fc1b212"
# Save execution log
log.summary()
log_path = LOGS_DIR / f"{log.notebook}_execution.json"
log.save(log_path)
print(f"Execution log saved: {log_path}")

print("=" * 80)
print("GENUS SELECTION VALIDATION - COMPLETE")
print("=" * 80)

print("\nResults:")
print(f"   Original genera (Phase 1): {len(all_genera)}")
print(f"   Viable after setup decisions: {len(viable_genera)}")
print(f"   Final classes (after grouping): {len(final_genera_list)}")
print(f"   Samples retained: {output_config['impact_assessment']['retention_rate'] * 100:.1f}%")

print("\nOutputs:")
print("   • Extended setup_decisions.json (genus_selection section)")
print("   • genus_sample_counts.png")
print("   • jm_separability_heatmap.png")
print("   • jm_dendrogram.png")
print("   • genus_groups_overview.png")

print("\nNext Steps:")
print("   1. Review genus groups for biological plausibility")
print("   2. Run 03a_setup_fixation to apply genus mapping")
print("   3. Run exp_11_algorithm_comparison.ipynb")
print("=" * 80)

