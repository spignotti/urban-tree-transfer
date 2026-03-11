"""Utility exports."""

from .final_validation import validate_zero_nan
from .geo import (
    buffer_boundaries,
    clip_to_boundary,
    ensure_project_crs,
    validate_geometries,
)
from .json_validation import (
    validate_algorithm_comparison,
    validate_chm_assessment,
    validate_correlation_removal,
    validate_evaluation_metrics,
    validate_finetuning_curve,
    validate_hp_tuning_result,
    validate_outlier_thresholds,
    validate_proximity_filter,
    validate_setup_decisions,
    validate_spatial_autocorrelation,
    validate_temporal_selection,
)
from .logging import ExecutionLog, log_error, log_step, log_success, log_warning
from .schema_validation import (
    validate_phase2a_output,
    validate_phase2b_output,
    validate_phase2c_output,
)
from .strings import normalize_city_name
from .validation import (
    generate_validation_report,
    validate_crs,
    validate_dataset,
    validate_no_null_geometries,
    validate_schema,
    validate_within_boundary,
)

__all__ = [
    "ExecutionLog",
    "buffer_boundaries",
    "clip_to_boundary",
    "ensure_project_crs",
    "generate_validation_report",
    "log_error",
    "log_step",
    "log_success",
    "log_warning",
    "normalize_city_name",
    "validate_algorithm_comparison",
    "validate_chm_assessment",
    "validate_correlation_removal",
    "validate_crs",
    "validate_dataset",
    "validate_evaluation_metrics",
    "validate_finetuning_curve",
    "validate_geometries",
    "validate_hp_tuning_result",
    "validate_no_null_geometries",
    "validate_outlier_thresholds",
    "validate_phase2a_output",
    "validate_phase2b_output",
    "validate_phase2c_output",
    "validate_proximity_filter",
    "validate_schema",
    "validate_setup_decisions",
    "validate_spatial_autocorrelation",
    "validate_temporal_selection",
    "validate_within_boundary",
    "validate_zero_nan",
]
