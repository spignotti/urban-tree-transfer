"""String normalization utilities."""

from __future__ import annotations


def normalize_city_name(value: str) -> str:
    """Normalize city names for identifiers and comparisons."""
    return value.strip().lower()
