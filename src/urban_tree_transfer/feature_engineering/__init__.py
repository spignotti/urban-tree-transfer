"""Feature engineering exports."""

from .extraction import (
    correct_tree_positions,
    extract_all_features,
    extract_chm_features,
    extract_sentinel_features,
)
from .outliers import (
    classify_outliers_by_consensus,
    detect_iqr_outliers,
    detect_mahalanobis_outliers,
    detect_zscore_outliers,
    remove_outliers,
)
from .quality import (
    analyze_nan_distribution,
    apply_temporal_selection,
    compute_chm_engineered_features,
    filter_by_plant_year,
    filter_deciduous_genera,
    filter_nan_trees,
    filter_ndvi_plausibility,
    interpolate_features_within_tree,
)
from .selection import (
    compute_feature_correlations,
    identify_redundant_features,
    remove_redundant_features,
)
from .splits import (
    assign_trees_to_blocks,
    create_spatial_blocks,
    create_stratified_spatial_splits,
    validate_splits,
)

__all__ = [
    "analyze_nan_distribution",
    "apply_temporal_selection",
    "assign_trees_to_blocks",
    "classify_outliers_by_consensus",
    "compute_chm_engineered_features",
    "compute_feature_correlations",
    "correct_tree_positions",
    "create_spatial_blocks",
    "create_stratified_spatial_splits",
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
    "remove_outliers",
    "remove_redundant_features",
    "validate_splits",
]
