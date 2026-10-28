"""Shared plotting style for report figures."""

from __future__ import annotations

import matplotlib as mpl

PALETTE = {
    "forest": "#2D6A4F",
    "teal": "#40916C",
    "amber": "#E07A2F",
    "steel": "#457B9D",
    "sand": "#D4A276",
    "ink": "#1D3557",
    "slate": "#6C757D",
    "cloud": "#F8F9FA",
}

# Okabe-Ito colors (colorblind-safe) for city-comparison figures.
CITY_COLORS = {
    "berlin": "#009E73",  # bluish green / teal
    "leipzig": "#E69F00",  # orange
}


def apply_report_style() -> None:
    """Apply consistent publication-oriented plotting defaults."""
    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": PALETTE["ink"],
            "axes.labelcolor": PALETTE["ink"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "grid.color": PALETTE["slate"],
            "grid.alpha": 0.18,
            "grid.linewidth": 0.6,
            "font.family": "serif",
            "font.serif": ["Libertinus Serif", "Times New Roman", "DejaVu Serif"],
            "font.size": 9.5,
            "axes.titlesize": 10,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.8,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 5.2,
        }
    )
