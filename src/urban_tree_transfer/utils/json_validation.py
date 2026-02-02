"""JSON schema validation utilities for Phase 2 exploratory outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _load_schema(schema_name: str) -> dict[str, Any]:
    schema_path = _SCHEMA_DIR / schema_name
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    if not isinstance(schema, dict):
        raise ValueError(f"Expected JSON schema object in {schema_path}")
    return schema


def _validate_type(value: Any, expected_type: str) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    return True


def _validate_schema(data: Any, schema: dict[str, Any], path: str = "$") -> None:
    expected_type = schema.get("type")
    if expected_type and not _validate_type(data, expected_type):
        raise ValueError(f"{path}: expected type {expected_type}")

    if expected_type == "object":
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                raise ValueError(f"{path}: missing required key '{key}'")
        properties = schema.get("properties", {})
        if isinstance(properties, dict):
            for key, subschema in properties.items():
                if key in data:
                    _validate_schema(data[key], subschema, path=f"{path}.{key}")

    if expected_type == "array":
        if "minItems" in schema and len(data) < int(schema["minItems"]):
            raise ValueError(f"{path}: expected at least {schema['minItems']} items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(data):
                _validate_schema(item, item_schema, path=f"{path}[{idx}]")

    if expected_type == "integer":
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            raise ValueError(f"{path}: value {data} below minimum {minimum}")
        if maximum is not None and data > maximum:
            raise ValueError(f"{path}: value {data} above maximum {maximum}")

    if expected_type == "number":
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            raise ValueError(f"{path}: value {data} below minimum {minimum}")
        if maximum is not None and data > maximum:
            raise ValueError(f"{path}: value {data} above maximum {maximum}")


def _load_and_validate(json_path: Path, schema_name: str) -> dict[str, Any]:
    data = _load_json(json_path)
    schema = _load_schema(schema_name)
    _validate_schema(data, schema)
    return data


def validate_temporal_selection(json_path: Path) -> dict[str, Any]:
    """Load and validate temporal_selection.json."""
    return _load_and_validate(json_path, "temporal_selection.schema.json")


def validate_chm_assessment(json_path: Path) -> dict[str, Any]:
    """Load and validate chm_assessment.json."""
    return _load_and_validate(json_path, "chm_assessment.schema.json")


def validate_correlation_removal(json_path: Path) -> dict[str, Any]:
    """Load and validate correlation_removal.json."""
    return _load_and_validate(json_path, "correlation_removal.schema.json")


def validate_outlier_thresholds(json_path: Path) -> dict[str, Any]:
    """Load and validate outlier_thresholds.json."""
    return _load_and_validate(json_path, "outlier_thresholds.schema.json")


def validate_spatial_autocorrelation(json_path: Path) -> dict[str, Any]:
    """Load and validate spatial_autocorrelation.json."""
    return _load_and_validate(json_path, "spatial_autocorrelation.schema.json")


def validate_proximity_filter(json_path: Path) -> dict[str, Any]:
    """Load and validate proximity_filter.json."""
    return _load_and_validate(json_path, "proximity_filter.schema.json")
