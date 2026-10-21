"""Shared publication palette for manuscript and supplemental figures."""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "A": "#66C2A5",       # ACP precursor
    "H": "#009E73",       # Matrix-associated apatite
    "matrix": "#009E73",
    "Q": "#D55E00",       # Solution-driven mineral
    "total": "#333333",
    "C": "#0072B2",
    "P": "#E69F00",
    "reference": "#999999",
}

REGIME_COLORS = (COLORS["reference"], COLORS["matrix"], COLORS["Q"])


def sequential_shades(color: str, count: int) -> list[tuple[float, float, float]]:
    """Ordered, distinguishable tints of one hue for a parameter sweep."""
    base = np.asarray(mcolors.to_rgb(color))
    return [tuple((1.0 - strength) * np.ones(3) + strength * base)
            for strength in np.linspace(0.38, 1.0, count)]


def configure_publication_style(**overrides: object) -> None:
    settings: dict[str, object] = {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#E5E5E5",
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "image.cmap": "viridis",
    }
    settings.update(overrides)
    plt.rcParams.update(settings)
