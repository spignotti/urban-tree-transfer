"""Spatial block creation and stratified splitting.

This module handles:
- Regular grid (block) creation
- Spatial assignment of trees to blocks
- Stratified train/val/test splitting with spatial disjointness
"""

from __future__ import annotations

from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.special import rel_entr
from sklearn.model_selection import StratifiedGroupKFold

from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED
from urban_tree_transfer.utils.strings import normalize_city_name


def create_spatial_blocks(
    trees_gdf: gpd.GeoDataFrame,
    block_size_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Create regular spatial grid and assign trees to blocks using vectorized operations.

    Optimized for large datasets (1M+ trees) by avoiding spatial joins.
    Uses direct mathematical grid index calculation: O(N) instead of O(N*M).

    Args:
        trees_gdf: Input trees (must have geometry in projected CRS).
        block_size_m: Block size in meters (from exp_05 analysis).

    Returns:
        GeoDataFrame with new column 'block_id' (str).
    """
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")
    if block_size_m <= 0:
        raise ValueError("block_size_m must be > 0.")
    if trees_gdf.empty:
        raise ValueError("trees_gdf must not be empty.")
    if "city" not in trees_gdf.columns:
        raise ValueError("Missing required column: city")

    result = trees_gdf.copy()
    city_groups: list[gpd.GeoDataFrame] = []

    for city, group in result.groupby("city"):
        if group.empty:
            continue

        city_slug = normalize_city_name(str(city))[:8]
        minx, miny, _, _ = group.total_bounds

        # Vectorized grid index calculation (O(N))
        # Formula: floor((coordinate - origin) / block_size)
        grid_x = np.floor((group.geometry.x.values - minx) / block_size_m).astype(np.int32)
        grid_y = np.floor((group.geometry.y.values - miny) / block_size_m).astype(np.int32)

        # Vectorized string formatting using pandas (faster than list comprehension)
        block_ids = pd.Series(
            [f"{city_slug}_{x:04d}_{y:04d}" for x, y in zip(grid_x, grid_y, strict=True)],
            index=group.index,
        )

        group_copy = cast(gpd.GeoDataFrame, group.copy())
        group_copy["block_id"] = block_ids
        city_groups.append(group_copy)

    if not city_groups:
        raise ValueError("No city groups could be created from input data.")

    return cast(gpd.GeoDataFrame, pd.concat(city_groups, ignore_index=False))


def _assign_folds(
    trees_gdf: gpd.GeoDataFrame,
    *,
    n_splits: int,
    random_seed: int,
    stratify_column: str,
    group_column: str,
) -> pd.Series:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if trees_gdf[stratify_column].nunique(dropna=True) < 2:
        raise ValueError("Stratification requires at least two classes.")

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    fold_ids = pd.Series(-1, index=trees_gdf.index, dtype=int)
    for fold, (_, test_idx) in enumerate(
        splitter.split(trees_gdf, y=trees_gdf[stratify_column], groups=trees_gdf[group_column])
    ):
        fold_ids.iloc[test_idx] = fold

    if (fold_ids < 0).any():
        raise ValueError("Failed to assign all samples to folds.")
    return fold_ids


def create_stratified_splits_berlin(
    gdf: gpd.GeoDataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = RANDOM_SEED,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create spatially disjoint Train/Val/Test splits for Berlin.

    Args:
        gdf: Input trees with block_id and genus_latin columns.
        train_ratio: Proportion for training set.
        val_ratio: Proportion for validation set.
        test_ratio: Proportion for test set.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_gdf, val_gdf, test_gdf).
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {gdf.crs}")
    if "block_id" not in gdf.columns:
        raise ValueError("Missing required column: block_id")
    if "genus_latin" not in gdf.columns:
        raise ValueError("Missing required column: genus_latin")
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0.")

    n_blocks = int(gdf["block_id"].nunique())
    if n_blocks < 3:
        raise ValueError("Not enough blocks to create splits.")

    n_splits = int(min(10, n_blocks))
    fold_ids = _assign_folds(
        gdf,
        n_splits=n_splits,
        random_seed=random_seed,
        stratify_column="genus_latin",
        group_column="block_id",
    )
    holdout_folds = max(1, round((1 - train_ratio) * n_splits))
    holdout_fold_ids = set(range(holdout_folds))

    holdout_mask = fold_ids.isin(holdout_fold_ids)
    train_gdf = gdf.loc[~holdout_mask].copy()
    holdout_gdf = gdf.loc[holdout_mask].copy()

    holdout_blocks = holdout_gdf["block_id"].nunique()
    if holdout_blocks < 2:
        raise ValueError("Not enough blocks in holdout set for val/test split.")

    val_test_splitter = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=random_seed)
    for val_idx, test_idx in val_test_splitter.split(
        holdout_gdf,
        y=holdout_gdf["genus_latin"],
        groups=holdout_gdf["block_id"],
    ):
        val_gdf = holdout_gdf.iloc[val_idx].copy()
        test_gdf = holdout_gdf.iloc[test_idx].copy()
        break

    return train_gdf, val_gdf, test_gdf


def create_stratified_splits_leipzig(
    gdf: gpd.GeoDataFrame,
    finetune_ratio: float = 0.80,
    test_ratio: float = 0.20,
    random_seed: int = RANDOM_SEED,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create spatially disjoint Finetune/Test splits for Leipzig.

    Args:
        gdf: Input trees with block_id and genus_latin columns.
        finetune_ratio: Proportion for fine-tuning pool.
        test_ratio: Proportion for test set.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (finetune_gdf, test_gdf).
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {gdf.crs}")
    if "block_id" not in gdf.columns:
        raise ValueError("Missing required column: block_id")
    if "genus_latin" not in gdf.columns:
        raise ValueError("Missing required column: genus_latin")
    if not np.isclose(finetune_ratio + test_ratio, 1.0):
        raise ValueError("finetune_ratio + test_ratio must equal 1.0.")

    n_blocks = int(gdf["block_id"].nunique())
    if n_blocks < 2:
        raise ValueError("Not enough blocks to create splits.")

    n_splits = int(min(5, n_blocks))
    fold_ids = _assign_folds(
        gdf,
        n_splits=n_splits,
        random_seed=random_seed,
        stratify_column="genus_latin",
        group_column="block_id",
    )
    test_folds = max(1, round(test_ratio * n_splits))
    test_fold_ids = set(range(test_folds))

    test_mask = fold_ids.isin(test_fold_ids)
    test_gdf = gdf.loc[test_mask].copy()
    finetune_gdf = gdf.loc[~test_mask].copy()

    return finetune_gdf, test_gdf

# KL threshold of 0.07 is empirically derived: spatial block-based splitting
# with genus stratification produces KL-divergences in the range 0.01-0.05
# for well-stratified splits. The 0.07 threshold allows for variance from
# spatial blocking constraints while catching poorly stratified splits.
def validate_split_stratification(
    *split_gdfs: gpd.GeoDataFrame,
    split_names: list[str] | None = None,
    kl_threshold: float = 0.07,
) -> dict[str, Any]:
    """Validate split quality and spatial disjointness for multiple splits.

    Args:
        *split_gdfs: Variable number of split GeoDataFrames.
        split_names: Names for each split (e.g., ['train', 'val', 'test']).
        kl_threshold: Maximum allowed KL-divergence between splits (default 0.07 for spatial blocking + filtering).

    Returns:
        Dictionary with validation metrics.
    """
    if not split_gdfs:
        raise ValueError("At least one split GeoDataFrame must be provided.")

    if split_names is None:
        split_names = [f"split_{idx}" for idx in range(len(split_gdfs))]
    if len(split_names) != len(split_gdfs):
        raise ValueError("split_names length must match number of splits.")

    for split_gdf in split_gdfs:
        if split_gdf.crs is None:
            raise ValueError("GeoDataFrame has no CRS defined.")
        if str(split_gdf.crs) != PROJECT_CRS:
            raise ValueError(f"Expected CRS {PROJECT_CRS}, got {split_gdf.crs}")
        if "block_id" not in split_gdf.columns:
            raise ValueError("Missing required column: block_id")
        if "genus_latin" not in split_gdf.columns:
            raise ValueError("Missing required column: genus_latin")

    genus_distributions: dict[str, dict[str, int]] = {}
    block_counts: dict[str, int] = {}
    sample_sizes: dict[str, int] = {}

    all_genera: set[str] = set()
    for name, split_gdf in zip(split_names, split_gdfs, strict=True):
        counts = split_gdf["genus_latin"].value_counts().to_dict()
        genus_distributions[name] = {str(k): int(v) for k, v in counts.items()}
        block_counts[name] = int(split_gdf["block_id"].nunique())
        sample_sizes[name] = len(split_gdf)
        all_genera.update(counts.keys())

    kl_divergences: dict[str, float] = {}
    split_probs: dict[str, np.ndarray] = {}
    genus_list = sorted(all_genera)
    for name, split_gdf in zip(split_names, split_gdfs, strict=True):
        counts = split_gdf["genus_latin"].value_counts()
        probs = np.array([counts.get(genus, 0) for genus in genus_list], dtype=float)
        probs = np.zeros_like(probs) if probs.sum() == 0 else probs / probs.sum()
        split_probs[name] = probs

    epsilon = 1e-12
    for i, name_i in enumerate(split_names):
        for name_j in split_names[i + 1 :]:
            p = split_probs[name_i] + epsilon
            q = split_probs[name_j] + epsilon
            kl = float(np.sum(rel_entr(p, q)))
            kl_divergences[f"{name_i}_vs_{name_j}"] = kl

    block_sets = {
        name: set(split_gdf["block_id"].dropna().unique())
        for name, split_gdf in zip(split_names, split_gdfs, strict=True)
    }
    overlap_blocks: set[str] = set()
    for i, name_i in enumerate(split_names):
        for name_j in split_names[i + 1 :]:
            overlap_blocks |= block_sets[name_i].intersection(block_sets[name_j])

    if kl_divergences:
        max_kl = max(kl_divergences.values())
        if max_kl > kl_threshold:
            raise ValueError(
                "Stratification quality check failed: "
                f"max KL={max_kl:.4f} > {kl_threshold:.4f}. "
                "Increase n_splits or check genus distributions."
            )

    return {
        "genus_distributions": genus_distributions,
        "kl_divergences": kl_divergences,
        "spatial_overlap": len(overlap_blocks),
        "block_counts": block_counts,
        "sample_sizes": sample_sizes,
    }
