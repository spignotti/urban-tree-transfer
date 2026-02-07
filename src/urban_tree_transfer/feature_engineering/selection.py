"""Feature selection and redundancy removal.

This module handles:
- Correlation analysis within feature groups
- Redundant feature identification and removal
- Parquet export for ML-ready datasets
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import geopandas as gpd
import numpy as np
import pandas as pd

from urban_tree_transfer.config import PROJECT_CRS
from urban_tree_transfer.utils.final_validation import validate_zero_nan
from urban_tree_transfer.utils.schema_validation import validate_phase2c_output

# Columns to drop when exporting to Parquet (keep metadata for Phase 3 analysis)
_PARQUET_DROP_COLUMNS = [
    "geometry",  # Available in geometry_lookup.parquet
    "height_m",  # Redundant with CHM features (cadastral height not needed)
]
# KEEP for Phase 3 analysis:
# - Categorical: species_latin, genus_german, species_german, tree_type
# - Temporal: plant_year (error metrics analysis)
# - QC: position_corrected, correction_distance (position quality analysis)


def compute_feature_correlations(
    trees_gdf: gpd.GeoDataFrame,
    feature_groups: dict[str, list[str]],
) -> dict[str, pd.DataFrame]:
    """Compute correlation matrices for feature groups.

    Args:
        trees_gdf: GeoDataFrame with features.
        feature_groups: Dict mapping group name to feature list.
                       Example: {'spectral': ['B2_04', 'B3_04', ...]}

    Returns:
        Dict mapping group name to correlation DataFrame.
    """
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")

    if not feature_groups:
        raise ValueError("feature_groups must be a non-empty dict.")

    correlations: dict[str, pd.DataFrame] = {}
    for group_name, columns in feature_groups.items():
        if not columns:
            raise ValueError(f"feature_groups['{group_name}'] must be non-empty.")
        missing = [col for col in columns if col not in trees_gdf.columns]
        if missing:
            raise ValueError(f"Missing columns for group '{group_name}': {missing}")
        correlations[group_name] = pd.DataFrame(trees_gdf[columns]).corr()

    return correlations


def identify_redundant_features(
    correlation_matrices: dict[str, pd.DataFrame],
    threshold: float = 0.95,
    feature_importance: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Identify redundant feature pairs based on correlation threshold.

    For each highly correlated pair (|r| > threshold), marks one for removal.
    Selection criteria: If feature_importance provided, keep the more important
    feature; otherwise default to removing feature_b.

    Args:
        correlation_matrices: From compute_feature_correlations.
        threshold: Correlation threshold for redundancy.
        feature_importance: Optional dict of feature -> importance score.

    Returns:
        List of dicts with keys: feature_to_remove, correlated_with, correlation, reason.
    """
    if not correlation_matrices:
        raise ValueError("correlation_matrices must be a non-empty dict.")
    if threshold <= 0 or threshold >= 1:
        raise ValueError("threshold must be between 0 and 1.")

    redundant: list[dict[str, Any]] = []
    for group_name, matrix in correlation_matrices.items():
        if matrix.empty:
            continue
        cols = list(matrix.columns)
        for i, feature_a in enumerate(cols):
            for feature_b in cols[i + 1 :]:
                corr_value = float(matrix.loc[feature_a, feature_b])
                if abs(corr_value) > threshold:
                    to_remove = feature_b
                    correlated_with = feature_a
                    if feature_importance is not None:
                        importance_a = feature_importance.get(feature_a, 0.0)
                        importance_b = feature_importance.get(feature_b, 0.0)
                        if importance_b > importance_a:
                            to_remove = feature_a
                            correlated_with = feature_b
                    redundant.append(
                        {
                            "feature_to_remove": to_remove,
                            "correlated_with": correlated_with,
                            "correlation": corr_value,
                            "reason": f"group={group_name}, |r|>{threshold}",
                        }
                    )
    return redundant


def compute_vif(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Compute Variance Inflation Factor (VIF) for features.

    Args:
        trees_gdf: GeoDataFrame with feature columns.
        feature_columns: Feature columns to compute VIF for.

    Returns:
        DataFrame with columns: feature, vif.
    """
    if not feature_columns:
        raise ValueError("feature_columns must be a non-empty list.")
    missing = [col for col in feature_columns if col not in trees_gdf.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    data = trees_gdf[feature_columns].dropna()
    if data.empty:
        raise ValueError("No valid rows available for VIF computation.")

    values = data.to_numpy(dtype=float)
    n_features = values.shape[1]
    vifs: list[float] = []

    for i in range(n_features):
        y = values[:, i]
        if n_features == 1:
            vifs.append(1.0)
            continue

        x = np.delete(values, i, axis=1)
        x = np.column_stack([np.ones(x.shape[0]), x])
        coeffs, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_pred = x @ coeffs

        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 0.0 if ss_tot == 0 else 1.0 - (ss_res / ss_tot)

        if r_squared >= 1.0:
            vifs.append(float("inf"))
        else:
            vifs.append(float(1.0 / (1.0 - r_squared)))

    return pd.DataFrame({"feature": feature_columns, "vif": vifs})


def validate_final_preparation_output(
    trees_gdf: gpd.GeoDataFrame,
    feature_columns: list[str],
) -> gpd.GeoDataFrame:
    """Validate Phase 2c output schema after final preparation steps."""
    validate_phase2c_output(trees_gdf, feature_columns)
    validate_zero_nan(trees_gdf, feature_columns, dataset_name="phase2c")
    return trees_gdf


def remove_redundant_features(
    trees_gdf: gpd.GeoDataFrame,
    features_to_remove: list[str],
) -> gpd.GeoDataFrame:
    """Remove redundant features from dataset.

    Args:
        trees_gdf: Input GeoDataFrame.
        features_to_remove: List of column names to drop.

    Returns:
        GeoDataFrame with redundant columns removed.
    """
    if trees_gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined.")
    if str(trees_gdf.crs) != PROJECT_CRS:
        raise ValueError(f"Expected CRS {PROJECT_CRS}, got {trees_gdf.crs}")
    if features_to_remove is None:
        raise ValueError("features_to_remove must not be None.")

    if not features_to_remove:
        return trees_gdf.copy()

    safe_remove = [col for col in features_to_remove if col != "geometry"]
    if len(safe_remove) < len(features_to_remove):
        warnings.warn("Skipping 'geometry' in features_to_remove.", stacklevel=2)

    missing = [col for col in safe_remove if col not in trees_gdf.columns]
    if missing:
        warnings.warn(f"Columns not found for removal: {missing}", stacklevel=2)

    existing = [col for col in safe_remove if col in trees_gdf.columns]
    return cast(gpd.GeoDataFrame, trees_gdf.drop(columns=existing).copy())


def export_splits_to_parquet(
    splits: dict[str, gpd.GeoDataFrame],
    output_dir: Path,
    drop_metadata_columns: list[str] | None = None,
) -> dict[str, Path]:
    """Export GeoDataFrame splits to Parquet files (without geometry).

    Drops geometry and specified metadata columns, then saves each split
    as a Parquet file with Snappy compression.

    Args:
        splits: Mapping of split name (e.g. "berlin_train") to GeoDataFrame.
        output_dir: Directory to write Parquet files into.
        drop_metadata_columns: Additional columns to drop beyond geometry. If None, uses default
            list of non-ML columns (species, plant_year, height_m, etc.).

    Returns:
        Mapping of split name to output Parquet path.
    """
    if not splits:
        raise ValueError("splits must be a non-empty dict.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if drop_metadata_columns is None:
        drop_metadata_columns = _PARQUET_DROP_COLUMNS

    parquet_paths: dict[str, Path] = {}

    for split_name, gdf in splits.items():
        if gdf.empty:
            raise ValueError(f"Split '{split_name}' is empty.")
        if gdf.crs is None:
            raise ValueError(f"Split '{split_name}' has no CRS defined.")
        if str(gdf.crs) != PROJECT_CRS:
            raise ValueError(f"Split '{split_name}' has CRS {gdf.crs}, expected {PROJECT_CRS}.")

        # Drop columns (only those that exist)
        cols_to_drop = [col for col in drop_metadata_columns if col in gdf.columns]

        # Convert to regular DataFrame (drops geometry automatically if removed)
        df = pd.DataFrame(gdf.drop(columns=cols_to_drop))

        # Ensure geometry is not present
        if "geometry" in df.columns:
            df = df.drop(columns=["geometry"])

        # Export to Parquet with Snappy compression
        output_path = output_dir / f"{split_name}.parquet"
        df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

        parquet_paths[split_name] = output_path

    return parquet_paths


def export_geometry_lookup(
    splits: dict[str, gpd.GeoDataFrame],
    output_path: Path,
) -> int:
    """Export geometry lookup table for all trees across splits.

    Creates a single Parquet file with tree_id, city, split, filtered, x, y
    for later visualization of predictions on maps.

    Args:
        splits: Mapping of split name to GeoDataFrame. Names must follow pattern
            "{city}_{split}" or "{city}_{split}_filtered".
        output_path: Path for the output Parquet file.

    Returns:
        Number of unique trees in the lookup.
    """
    if not splits:
        raise ValueError("splits must be a non-empty dict.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lookup_records: list[dict[str, Any]] = []

    for split_name, gdf in splits.items():
        if gdf.empty:
            continue
        if gdf.crs is None:
            raise ValueError(f"Split '{split_name}' has no CRS defined.")
        if str(gdf.crs) != PROJECT_CRS:
            raise ValueError(f"Split '{split_name}' has CRS {gdf.crs}, expected {PROJECT_CRS}.")

        # Required columns
        if "tree_id" not in gdf.columns:
            raise ValueError(f"Split '{split_name}' missing 'tree_id' column.")
        if "city" not in gdf.columns:
            raise ValueError(f"Split '{split_name}' missing 'city' column.")

        # Parse filtered flag from split name
        is_filtered = split_name.endswith("_filtered")

        # Extract coordinates
        for _, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            lookup_records.append(
                {
                    "tree_id": row["tree_id"],
                    "city": row["city"],
                    "split": split_name,
                    "filtered": is_filtered,
                    "x": float(geom.x),
                    "y": float(geom.y),
                }
            )

    if not lookup_records:
        raise ValueError("No valid geometry records found in splits.")

    # Create DataFrame (no deduplication - trees appear once per split they belong to)
    lookup_df = pd.DataFrame(lookup_records)

    # Export to Parquet
    lookup_df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)

    return len(lookup_df)
