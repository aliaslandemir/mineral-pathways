"""
Generate the day-30 protocol prediction heatmap.

python -m src.protocol_main_prediction_figure --figure-formats pdf,png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np

from src.config import RUN_DEFAULTS, build_base_params
from src.io_utils import OUTPUT_ROOT, parse_figure_formats, save_figure_to_directory
from src.model import simulate


FIGURE_DIR = OUTPUT_ROOT / "protocol_optimization" / "figures"
FIGURE_STEM = "protocol_prediction_day30_apatite_heatmap"
ENDPOINT_DAY = 30.0
INTERVALS = (1.0, 2.0, 3.0)
BETA_GP_LEVELS = tuple(float(value) for value in range(1, 11))
BASELINE_INTERVAL = 2.0
BASELINE_BETA_GP = 10.0


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def simulate_endpoint_grid(dt: float = RUN_DEFAULTS.dt) -> np.ndarray:
    base = build_base_params()
    values = np.empty((len(INTERVALS), len(BETA_GP_LEVELS)), dtype=float)
    for row, interval in enumerate(INTERVALS):
        for column, beta_gp in enumerate(BETA_GP_LEVELS):
            params = dict(base)
            params["beta_GP"] = beta_gp
            result = simulate(params, t_end=ENDPOINT_DAY, dt=dt, media_interval=interval)
            values[row, column] = float(result.H[-1])
    return values


def make_figure(
    formats: tuple[str, ...] = ("pdf", "png"),
    *,
    dt: float = RUN_DEFAULTS.dt,
) -> tuple[Path, ...]:
    configure_matplotlib()
    values = simulate_endpoint_grid(dt)
    cmap = LinearSegmentedColormap.from_list(
        "apatite_heatmap",
        ("#f7f7f7", "#dedede", "#b7d7d0", "#62b6a4", "#23867f", "#0b4f6c"),
    )
    norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(values)), 1e-12))

    fig, ax = plt.subplots(figsize=(6.1, 4.2))
    image = ax.imshow(values, cmap=cmap, norm=norm, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(BETA_GP_LEVELS)), [f"{value:g}" for value in BETA_GP_LEVELS])
    ax.set_yticks(np.arange(len(INTERVALS)), [f"{value:g}" for value in INTERVALS])
    ax.set_xlabel(r"$\beta$-GP (mM)")
    ax.set_ylabel("Medium-renewal interval (days)")
    ax.set_title("Day-30 apatite-like mineral")

    text_threshold = 0.55 * float(np.nanmax(values))
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            color = "white" if value >= text_threshold else "#111111"
            ax.text(column, row, f"{value:.2g}", ha="center", va="center", color=color, fontsize=8.2)

    baseline_row = INTERVALS.index(BASELINE_INTERVAL)
    baseline_column = BETA_GP_LEVELS.index(BASELINE_BETA_GP)
    ax.add_patch(
        Rectangle(
            (baseline_column - 0.45, baseline_row - 0.45),
            0.9,
            0.9,
            fill=False,
            edgecolor="#d95f02",
            linewidth=2.1,
            label="Baseline protocol",
        )
    )
    ax.set_xticks(np.arange(-0.5, len(BETA_GP_LEVELS), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(INTERVALS), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.legend(frameon=True, loc="lower left", fontsize=8.0)
    colorbar = fig.colorbar(image, ax=ax, pad=0.025)
    colorbar.set_label("Apatite H (mM phosphate equivalents)")
    fig.tight_layout(pad=0.8)
    return save_figure_to_directory(fig, FIGURE_DIR, FIGURE_STEM, formats, dpi=600, close=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the day-30 protocol prediction heatmap.")
    parser.add_argument("--dt", type=float, default=RUN_DEFAULTS.dt)
    parser.add_argument("--figure-formats", default="pdf,png", help="Comma-separated: pdf,png; empty skips figures.")
    args = parser.parse_args()
    saved = make_figure(parse_figure_formats(args.figure_formats), dt=args.dt)
    print("Saved protocol heatmap:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
