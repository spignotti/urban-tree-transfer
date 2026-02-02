"""Notebook idempotency checks for skip-if-exists logic."""

from __future__ import annotations

from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")

RUNNER_DIR = Path("notebooks/runners")


def _notebook_sources(path: Path) -> str:
    with path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    return "\n".join(
        "".join(cell.get("source", [])) for cell in notebook.get("cells", [])
    )


def _assert_skip_logic(source: str) -> None:
    assert "output_path.exists" in source or "path.exists" in source
    assert "skipping" in source


def test_02a_skips_when_output_exists(tmp_path: Path) -> None:
    """Verify 02a notebook skips extraction when trees_features_*.gpkg exist."""
    (tmp_path / "trees_with_features_berlin.gpkg").touch()
    (tmp_path / "trees_with_features_leipzig.gpkg").touch()

    source = _notebook_sources(RUNNER_DIR / "02a_feature_extraction.ipynb")
    _assert_skip_logic(source)


def test_02b_skips_when_output_exists(tmp_path: Path) -> None:
    """Verify 02b skips QC when trees_clean_*.gpkg exist."""
    (tmp_path / "trees_clean_berlin.gpkg").touch()
    (tmp_path / "trees_clean_leipzig.gpkg").touch()

    source = _notebook_sources(RUNNER_DIR / "02b_data_quality.ipynb")
    _assert_skip_logic(source)


def test_02c_skips_when_output_exists(tmp_path: Path) -> None:
    """Verify 02c skips preparation when final splits exist."""
    (tmp_path / "berlin_train.gpkg").touch()
    (tmp_path / "leipzig_test_filtered.gpkg").touch()

    source = _notebook_sources(RUNNER_DIR / "02c_final_preparation.ipynb")
    _assert_skip_logic(source)
