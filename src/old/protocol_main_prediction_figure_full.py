"""Archived full protocol-prediction figure suite."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

from src.config import RUN_DEFAULTS, build_base_params
from src.io_utils import OUTPUT_ROOT, parse_figure_formats, save_figure_to_directory
from src.model import simulate


WORKFLOW_DIR = OUTPUT_ROOT / "protocol_optimization"
DATA_DIR = WORKFLOW_DIR / "protocol_optimization"
FIGURE_DIR = WORKFLOW_DIR / "figures"
FIGURE_STEM = "protocol_prediction_supersaturation_endpoint"
PANEL_B_STEM = "protocol_prediction_day28_apatite_heatmap"

INTERVALS = (1.0, 2.0, 3.0)
CALENDAR_BETA_GP = (4.0, 6.0, 8.0, 10.0)
HEATMAP_BETA_GP = tuple(float(v) for v in range(1, 11))
BASELINE_BETA_GP = 10.0
BASELINE_INTERVAL = 2.0
H_THRESHOLD = 0.5
LABEL_THRESHOLD = 0.01


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.1,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.6,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_protocol_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    trajectories = pd.read_csv(DATA_DIR / "protocol_phase_trajectories.csv")
    summary = pd.read_csv(DATA_DIR / "protocol_phase_crossing_summary.csv")
    return trajectories, summary


def simulate_protocol_outputs(*, K_A: float, t_end: float = RUN_DEFAULTS.t_end, dt: float = RUN_DEFAULTS.dt) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = build_base_params()
    params["K_A"] = float(K_A)
    trajectory_rows: list[pd.DataFrame] = []
    summary_rows: list[dict[str, float]] = []
    for interval in INTERVALS:
        for beta_gp in HEATMAP_BETA_GP:
            run_params = dict(params)
            run_params["beta_GP"] = float(beta_gp)
            sim = simulate(run_params, t_end=float(t_end), dt=float(dt), media_interval=float(interval))
            trajectory_rows.append(
                pd.DataFrame(
                    {
                        "time_days": sim.t,
                        "beta_GP_mM": float(beta_gp),
                        "media_interval_days": float(interval),
                        "S_A": sim.S_A,
                    }
                )
            )
            summary_rows.append(
                {
                    "beta_GP_mM": float(beta_gp),
                    "media_interval_days": float(interval),
                    "H_final_model_units": float(sim.H[-1]),
                }
            )
    trajectories = pd.concat(trajectory_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["media_interval_days", "beta_GP_mM"], ignore_index=True)
    return trajectories, summary


def true_runs(time: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return [(float(time[start]), float(time[stop])) for start, stop in zip(starts, stops) if stop >= start]


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.08, 1.035, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=12.0, fontweight="bold")


def plot_supersaturation_calendar(ax: plt.Axes, trajectories: pd.DataFrame) -> None:
    colors = {1.0: "#24736f", 2.0: "#7d70b2", 3.0: "#d7832f"}
    row_labels: list[str] = []
    row_positions: list[float] = []
    y = 0.0
    group_centers: list[tuple[float, float]] = []

    for interval in INTERVALS:
        group_start = y
        for beta_gp in CALENDAR_BETA_GP:
            subset = trajectories[
                np.isclose(trajectories["media_interval_days"], interval)
                & np.isclose(trajectories["beta_GP_mM"], beta_gp)
            ].sort_values("time_days")
            if subset.empty:
                continue
            time = subset["time_days"].to_numpy(dtype=float)
            above = subset["S_A"].to_numpy(dtype=float) >= 1.0
            for start, stop in true_runs(time, above):
                ax.broken_barh(
                    [(start, max(stop - start, 0.04))],
                    (y - 0.34, 0.68),
                    facecolors=colors[interval],
                    edgecolors="none",
                    alpha=0.92,
                )
            row_positions.append(y)
            row_labels.append(f"{beta_gp:g}")
            y += 1.0
        group_centers.append((interval, 0.5 * (group_start + y - 1.0)))
        if interval != INTERVALS[-1]:
            ax.axhline(y - 0.5, color="#cfcfcf", linewidth=0.75)
            y += 0.45

    ax.set_yticks(row_positions, row_labels)
    ax.set_xlabel("Culture time (days)")
    ax.set_ylabel(r"$\beta$-GP (mM)")
    ax.set_xlim(0.0, 30.0)
    ax.set_ylim(-0.65, y - 0.35)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e6eaed", linewidth=0.65)
    ax.set_title("Supersaturation calendar", loc="left")
    for interval, center in group_centers:
        ax.text(28.65, center, f"{interval:g}-day\nrenewal", ha="left", va="center", fontsize=7.6, color="#333333")
    ax.legend(
        handles=[Patch(facecolor=colors[v], edgecolor="none", label=f"{v:g}-day renewal") for v in INTERVALS],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.01),
        ncol=3,
        handlelength=1.4,
        columnspacing=1.0,
        borderaxespad=0.0,
    )
    panel_label(ax, "A")


def lowest_beta_gp_reaching_threshold(summary: pd.DataFrame) -> pd.Series | None:
    feasible = summary.loc[summary["H_final_model_units"] >= H_THRESHOLD].copy()
    if feasible.empty:
        return None
    feasible = feasible.sort_values(["beta_GP_mM", "media_interval_days"], ascending=[True, True])
    return feasible.iloc[0]


def plot_endpoint_heatmap(fig: plt.Figure, ax: plt.Axes, summary: pd.DataFrame, *, title: str = "Day-28 apatite-like pool") -> None:
    table = (
        summary.pivot(index="media_interval_days", columns="beta_GP_mM", values="H_final_model_units")
        .reindex(index=list(INTERVALS), columns=list(HEATMAP_BETA_GP))
    )
    values = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = LinearSegmentedColormap.from_list(
        "light_gray_teal",
        ["#f7f7f7", "#e2e2e2", "#b7d7d0", "#62b6a4", "#23867f", "#0b4f6c"],
    )
    cmap.set_bad("#eeeeee")
    norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(values)), H_THRESHOLD))
    im = ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", aspect="auto")

    ax.set_xticks(np.arange(len(HEATMAP_BETA_GP)), [f"{v:g}" for v in HEATMAP_BETA_GP])
    ax.set_yticks(np.arange(len(INTERVALS)), [f"{v:g}" for v in INTERVALS])
    ax.set_xlabel(r"$\beta$-GP (mM)")
    ax.set_ylabel("Renewal interval (days)")
    ax.set_title(title, loc="left")

    for i, interval in enumerate(INTERVALS):
        for j, beta_gp in enumerate(HEATMAP_BETA_GP):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            if value >= LABEL_THRESHOLD:
                ax.text(j, i, f"{value:.2g}", ha="center", va="center", color="#111111", fontsize=7.8)
            if value >= H_THRESHOLD:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="white",
                        linewidth=1.15,
                        linestyle=(0, (2.5, 1.7)),
                    )
                )

    baseline_i = list(INTERVALS).index(BASELINE_INTERVAL)
    baseline_j = list(HEATMAP_BETA_GP).index(BASELINE_BETA_GP)
    ax.add_patch(
        Rectangle(
            (baseline_j - 0.46, baseline_i - 0.46),
            0.92,
            0.92,
            fill=False,
            edgecolor="#d95f02",
            linewidth=2.0,
        )
    )

    selected = lowest_beta_gp_reaching_threshold(summary)
    if selected is not None:
        selected_i = list(INTERVALS).index(float(selected["media_interval_days"]))
        selected_j = list(HEATMAP_BETA_GP).index(float(selected["beta_GP_mM"]))
        ax.add_patch(
            Rectangle(
                (selected_j - 0.38, selected_i - 0.38),
                0.76,
                0.76,
                fill=False,
                edgecolor="#e7298a",
                linewidth=2.1,
            )
        )

    ax.set_xticks(np.arange(-0.5, len(HEATMAP_BETA_GP), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(INTERVALS), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    panel_label(ax, "B")

    cbar = fig.colorbar(im, ax=ax, pad=0.026, fraction=0.052)
    cbar.set_label("Day-28 H (mM phosphate equivalents)", labelpad=7)

    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="#d95f02", linewidth=1.8, label="Baseline protocol"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="#e7298a", linewidth=1.8, label=r"Lowest $\beta$-GP with $H(28)\geq0.5$"),
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="#777777", linewidth=1.2, linestyle=(0, (2.5, 1.7)), label=r"$H(28)\geq0.5$"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.20), borderaxespad=0.0)


def plot_endpoint_heatmap_panel_b(fig: plt.Figure, ax: plt.Axes, summary: pd.DataFrame) -> None:
    table = (
        summary.pivot(index="media_interval_days", columns="beta_GP_mM", values="H_final_model_units")
        .reindex(index=list(INTERVALS), columns=list(HEATMAP_BETA_GP))
    )
    values = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = LinearSegmentedColormap.from_list(
        "light_gray_teal_panel_b",
        ["#f7f7f7", "#e3e3e3", "#bdd7d2", "#72bea9", "#319488", "#0b5871"],
    )
    cmap.set_bad("#eeeeee")
    norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(values)), H_THRESHOLD))
    im = ax.imshow(masked, cmap=cmap, norm=norm, origin="lower", aspect="auto")

    ax.set_xticks(np.arange(len(HEATMAP_BETA_GP)), [f"{v:g}" for v in HEATMAP_BETA_GP])
    ax.set_yticks(np.arange(len(INTERVALS)), [f"{v:g}" for v in INTERVALS])
    ax.set_xlabel(r"$\beta$-GP (mM)")
    ax.set_ylabel("Renewal interval (days)")
    ax.set_title("Day-28 apatite-like mineral", loc="center", pad=7)

    for i, interval in enumerate(INTERVALS):
        for j, beta_gp in enumerate(HEATMAP_BETA_GP):
            value = values[i, j]
            if not np.isfinite(value):
                continue
            if value >= LABEL_THRESHOLD:
                ax.text(j, i, f"{value:.2g}", ha="center", va="center", color="#111111", fontsize=9.2)
            if value >= H_THRESHOLD:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1.0,
                        1.0,
                        fill=False,
                        edgecolor="#ffffff",
                        linewidth=1.1,
                        linestyle=(0, (2.5, 1.7)),
                    )
                )

    baseline_i = list(INTERVALS).index(BASELINE_INTERVAL)
    baseline_j = list(HEATMAP_BETA_GP).index(BASELINE_BETA_GP)
    baseline_patch = Rectangle(
        (baseline_j - 0.45, baseline_i - 0.45),
        0.90,
        0.90,
        fill=False,
        edgecolor="#d95f02",
        linewidth=2.0,
    )
    ax.add_patch(baseline_patch)

    ax.set_xticks(np.arange(-0.5, len(HEATMAP_BETA_GP), 1.0), minor=True)
    ax.set_yticks(np.arange(-0.5, len(INTERVALS), 1.0), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.0)
    ax.tick_params(which="minor", bottom=False, left=False)
    panel_label(ax, "B")

    cbar = fig.colorbar(im, ax=ax, pad=0.018, fraction=0.045)
    cbar.set_label("Day-28 H (arbitrary units)", labelpad=7)

    handles = [
        Rectangle((0, 0), 1, 1, fill=False, edgecolor="#d95f02", linewidth=1.8, label="Baseline protocol"),
    ]
    legend = ax.legend(
        handles=handles,
        frameon=True,
        loc="lower left",
        bbox_to_anchor=(0.012, 0.02),
        borderaxespad=0.0,
        fontsize=7.8,
        handlelength=1.7,
        labelspacing=0.4,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#d4d4d4")
    legend.get_frame().set_linewidth(0.7)
    legend.get_frame().set_alpha(0.88)


def make_panel_b_figure(formats: tuple[str, ...] = ("pdf", "png")) -> tuple[Path, ...]:
    configure_matplotlib()
    _, summary = load_protocol_outputs()
    fig, ax = plt.subplots(1, 1, figsize=(5.7, 4.15))
    plot_endpoint_heatmap_panel_b(fig, ax, summary)
    fig.subplots_adjust(left=0.12, right=0.88, top=0.89, bottom=0.13)
    return save_figure_to_directory(fig, FIGURE_DIR, PANEL_B_STEM, formats, dpi=600, close=True, pad_inches=0.06)


def write_caption(path: Path, summary: pd.DataFrame, *, K_A: float | None = None) -> None:
    selected = lowest_beta_gp_reaching_threshold(summary)
    selected_text = "No protocol reached H(28) >= 0.5 mM phosphate equivalents."
    if selected is not None:
        selected_text = (
            f"The lowest beta-GP protocol reaching H(28) >= 0.5 mM phosphate equivalents was "
            f"{float(selected['beta_GP_mM']):g} mM beta-GP with renewal every "
            f"{float(selected['media_interval_days']):g} days."
        )
    k_a_text = "" if K_A is None else f" Simulations in this figure used K_A={float(K_A):g}."
    caption = (
        "Protocol-dependent supersaturation exposure and apatite-like mineral accumulation. "
        "Panel A shows model-predicted time intervals with S_A >= 1, interpreted as "
        "precursor-forming conditions. Panel B shows simulated day-28 apatite-like pool "
        "H(28) for each discrete beta-GP and renewal-interval protocol. Dashed cell "
        "outlines indicate H(28) >= 0.5 mM phosphate equivalents. The orange outline "
        "marks the manuscript baseline protocol, 10 mM beta-GP with renewal every 2 days. "
        f"{selected_text}{k_a_text}"
    )
    path.write_text(caption + "\n", encoding="utf-8")


def make_figure(
    formats: tuple[str, ...] = ("pdf", "png"),
    *,
    trajectories: pd.DataFrame | None = None,
    summary: pd.DataFrame | None = None,
    stem: str = FIGURE_STEM,
    title_suffix: str = "",
    K_A: float | None = None,
) -> tuple[Path, ...]:
    configure_matplotlib()
    if trajectories is None or summary is None:
        trajectories, summary = load_protocol_outputs()
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.8, 5.15),
        gridspec_kw={"width_ratios": (1.20, 1.0), "wspace": 0.34},
    )
    fig.suptitle(f"Protocol-dependent supersaturation exposure and apatite-like mineral accumulation{title_suffix}", y=0.985, fontsize=11.2)
    plot_supersaturation_calendar(axes[0], trajectories)
    plot_endpoint_heatmap(fig, axes[1], summary)
    fig.subplots_adjust(left=0.06, right=0.94, top=0.84, bottom=0.25)
    write_caption(FIGURE_DIR / f"{stem}_caption.txt", summary, K_A=K_A)
    return save_figure_to_directory(fig, FIGURE_DIR, stem, formats, dpi=600, close=True, pad_inches=0.08)


def make_k_a_sweep_figures(K_A_values: tuple[float, ...], formats: tuple[str, ...] = ("pdf", "png")) -> tuple[Path, ...]:
    saved: list[Path] = []
    for value in K_A_values:
        trajectories, summary = simulate_protocol_outputs(K_A=float(value))
        stem = f"protocol_prediction_supersaturation_endpoint_K_A_{value:g}".replace(".", "p")
        saved.extend(
            make_figure(
                formats,
                trajectories=trajectories,
                summary=summary,
                stem=stem,
                title_suffix=fr" ($K_A$={value:g})",
                K_A=float(value),
            )
        )
    return tuple(saved)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Create the two-panel protocol-prediction figure.")
    parser.add_argument("--figure-formats", default="pdf,png")
    parser.add_argument("--K-A-values", "--K_A_values", dest="K_A_values", default=None, help="Optional comma-separated K_A values for additional sweep figures.")
    parser.add_argument("--panel-b-only", action="store_true", help="Export only the day-28 heatmap panel.")
    args = parser.parse_args()
    formats = parse_figure_formats(args.figure_formats)
    if args.panel_b_only:
        saved = list(make_panel_b_figure(formats))
    else:
        saved = list(make_figure(formats))
    if args.K_A_values:
        values = tuple(float(v.strip()) for v in args.K_A_values.split(",") if v.strip())
        saved.extend(make_k_a_sweep_figures(values, formats))
    print("Saved protocol-prediction figure:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
