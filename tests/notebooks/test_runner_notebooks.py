"""Notebook execution smoke tests for runner notebooks."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
ExecutePreprocessor = pytest.importorskip("nbconvert.preprocessors").ExecutePreprocessor

NOTEBOOK_DIR = Path("notebooks/runners")


@pytest.mark.skipif(
    os.environ.get("EXECUTE_NOTEBOOKS") != "1",
    reason="Set EXECUTE_NOTEBOOKS=1 to run notebook execution tests.",
)
@pytest.mark.parametrize(
    "notebook_name",
    [
        "02a_feature_extraction.ipynb",
        "02b_data_quality.ipynb",
        "02c_final_preparation.ipynb",
    ],
)
def test_runner_notebook_executes(notebook_name: str) -> None:
    notebook_path = NOTEBOOK_DIR / notebook_name
    with notebook_path.open("r", encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    ep.preprocess(notebook, {"metadata": {"path": str(NOTEBOOK_DIR)}})
