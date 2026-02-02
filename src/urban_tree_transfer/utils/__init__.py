"""Utility exports."""

from typing import TYPE_CHECKING

from .geo import (
    buffer_boundaries,
    clip_to_boundary,
    ensure_project_crs,
    validate_geometries,
)
from .logging import ExecutionLog, log_error, log_step, log_success, log_warning
from .strings import normalize_city_name

if TYPE_CHECKING:
    from .plotting import save_figure, setup_plotting
else:
    try:
        from .plotting import save_figure, setup_plotting
    except ModuleNotFoundError as exc:
        _PLOTTING_IMPORT_ERROR = exc

        def save_figure(fig, path, dpi: int = 300) -> None:  # type: ignore[no-untyped-def]
            raise ModuleNotFoundError(
                "matplotlib is required for save_figure; install plotting dependencies."
            ) from _PLOTTING_IMPORT_ERROR

        def setup_plotting() -> None:  # type: ignore[no-untyped-def]
            raise ModuleNotFoundError(
                "matplotlib is required for setup_plotting; install plotting dependencies."
            ) from _PLOTTING_IMPORT_ERROR


from .final_validation import validate_zero_nan
from .json_validation import (
    validate_chm_assessment,
    validate_correlation_removal,
    validate_outlier_thresholds,
    validate_proximity_filter,
    validate_spatial_autocorrelation,
    validate_temporal_selection,
)
from .schema_validation import (
    validate_phase2a_output,
    validate_phase2b_output,
    validate_phase2c_output,
)
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
    "save_figure",
    "setup_plotting",
    "validate_chm_assessment",
    "validate_correlation_removal",
    "validate_crs",
    "validate_dataset",
    "validate_geometries",
    "validate_no_null_geometries",
    "validate_outlier_thresholds",
    "validate_phase2a_output",
    "validate_phase2b_output",
    "validate_phase2c_output",
    "validate_proximity_filter",
    "validate_schema",
    "validate_spatial_autocorrelation",
    "validate_temporal_selection",
    "validate_within_boundary",
    "validate_zero_nan",
]
