"""Publication-quality plotting utilities."""

from __future__ import annotations

# pyright: reportMissingImports=false, reportMissingModuleSource=false
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

PUBLICATION_STYLE = {
    "style": "seaborn-v0_8-whitegrid",
    "figsize": (12, 7),
    "dpi_export": 300,
    "palette": "Set2",
}


def setup_plotting() -> None:
    """Configure matplotlib for publication-quality plots."""
    plt.rcdefaults()
    plt.style.use(PUBLICATION_STYLE["style"])
    sns.set_palette(PUBLICATION_STYLE["palette"])
    plt.rcParams["figure.figsize"] = PUBLICATION_STYLE["figsize"]
    plt.rcParams["savefig.dpi"] = PUBLICATION_STYLE["dpi_export"]
    plt.rcParams["savefig.bbox"] = "tight"
    # Configure font fallback for unicode characters (e.g., checkmarks)
    plt.rcParams["font.family"] = "DejaVu Sans"


def save_figure(fig: Figure, path: Path, dpi: int = 300) -> None:
    """Save figure with consistent settings."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
