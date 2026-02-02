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
from shapely.geometry import box
from sklearn.model_selection import StratifiedGroupKFold

from urban_tree_transfer.config import PROJECT_CRS, RANDOM_SEED


def create_spatial_blocks(
    trees_gdf: gpd.GeoDataFrame,
    block_size_m: float = 500.0,
) -> gpd.GeoDataFrame:
    """Create regular spatial grid and assign trees to blocks.

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

    grids: list[gpd.GeoDataFrame] = []
    for city, group in trees_gdf.groupby("city"):
        city_name = str(city)
        if group.empty:
            continue
        minx, miny, maxx, maxy = group.total_bounds
        xs = np.arange(minx, maxx + block_size_m, block_size_m)
        ys = np.arange(miny, maxy + block_size_m, block_size_m)

        records: list[dict[str, Any]] = []
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                records.append(
                    {
                        "city": city,
                        "grid_x": ix,
                        "grid_y": iy,
                        "block_id": f"{city_name[:8]}_{ix:04d}_{iy:04d}",
                        "geometry": box(x, y, x + block_size_m, y + block_size_m),
                    }
                )
        grid_gdf = gpd.GeoDataFrame(records, crs=PROJECT_CRS)
        grids.append(grid_gdf)

    if not grids:
        raise ValueError("No grids could be created from input data.")

    blocks_gdf = gpd.GeoDataFrame(
        pd.concat(grids, ignore_index=True),
        geometry="geometry",
        crs=PROJECT_CRS,
    )
    blocks_subset = gpd.GeoDataFrame(
        blocks_gdf[["block_id", "geometry"]],
        geometry="geometry",
        crs=PROJECT_CRS,
    )

    joined = cast(
        gpd.GeoDataFrame,
        gpd.sjoin(
            trees_gdf,
            blocks_subset,
            how="left",
            predicate="within",
        ),
    )

    if "index_right" in joined.columns:
        joined = joined.drop(columns=["index_right"])

    missing_mask = joined["block_id"].isna()
    if bool(missing_mask.any()):
        nearest = gpd.sjoin_nearest(
            joined.loc[missing_mask].drop(columns=["block_id"]),
            blocks_subset,
            how="left",
        )
        joined.loc[missing_mask, "block_id"] = nearest["block_id"].values
        joined = joined.drop(columns=[col for col in joined.columns if col.startswith("index_")])

    return cast(gpd.GeoDataFrame, joined)


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


def validate_split_stratification(
    *split_gdfs: gpd.GeoDataFrame,
    split_names: list[str] | None = None,
    kl_threshold: float = 0.01,
) -> dict[str, Any]:
    """Validate split quality and spatial disjointness for multiple splits.

    Args:
        *split_gdfs: Variable number of split GeoDataFrames.
        split_names: Names for each split (e.g., ['train', 'val', 'test']).
        kl_threshold: Maximum allowed KL-divergence between splits.

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
