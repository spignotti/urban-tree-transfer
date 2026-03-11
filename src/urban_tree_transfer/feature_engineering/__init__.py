"""Feature engineering exports."""

from .extraction import (
    correct_tree_positions,
    extract_all_features,
    extract_chm_features,
    extract_sentinel_features,
)
from .outliers import (
    apply_consensus_outlier_filter,
    detect_iqr_outliers,
    detect_mahalanobis_outliers,
    detect_zscore_outliers,
)
from .proximity import (
    analyze_genus_specific_impact,
    apply_proximity_filter,
    compute_nearest_different_genus_distance,
)
from .quality import (
    add_is_conifer_column,
    analyze_nan_distribution,
    apply_temporal_selection,
    compute_chm_engineered_features,
    filter_by_plant_year,
    filter_deciduous_genera,
    filter_nan_trees,
    filter_ndvi_plausibility,
    interpolate_features_within_tree,
    run_quality_pipeline,
)
from .selection import (
    compute_feature_correlations,
    compute_vif,
    identify_redundant_features,
    remove_redundant_features,
    validate_final_preparation_output,
)
from .splits import (
    create_spatial_blocks,
    create_stratified_splits_berlin,
    create_stratified_splits_leipzig,
    validate_split_stratification,
)

__all__ = [
    "add_is_conifer_column",
    "analyze_genus_specific_impact",
    "analyze_nan_distribution",
    "apply_consensus_outlier_filter",
    "apply_proximity_filter",
    "apply_temporal_selection",
    "compute_chm_engineered_features",
    "compute_feature_correlations",
    "compute_nearest_different_genus_distance",
    "compute_vif",
    "correct_tree_positions",
    "create_spatial_blocks",
    "create_stratified_splits_berlin",
    "create_stratified_splits_leipzig",
    "detect_iqr_outliers",
    "detect_mahalanobis_outliers",
    "detect_zscore_outliers",
    "extract_all_features",
    "extract_chm_features",
    "extract_sentinel_features",
    "filter_by_plant_year",
    "filter_deciduous_genera",
    "filter_nan_trees",
    "filter_ndvi_plausibility",
    "identify_redundant_features",
    "interpolate_features_within_tree",
    "remove_redundant_features",
    "run_quality_pipeline",
    "validate_final_preparation_output",
    "validate_split_stratification",
]
