"""Archived exploratory plots for protocol optimization outputs."""

from __future__ import annotations

import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from src.io_utils import OUTPUT_ROOT, parse_figure_formats, save_figure_to_directory


WORKFLOW_DIR = OUTPUT_ROOT / "protocol_optimization"
DATA_DIR = WORKFLOW_DIR / "protocol_optimization"
FIGURE_DIR = WORKFLOW_DIR / "figures"
DEFAULT_FORMATS = ("pdf", "png")

ACP = "#24527a"
HAP = "#2a9d8f"
ORANGE = "#d95f02"
PURPLE = "#6a3d9a"
DARK = "#252525"
GREY = "#d9d9d9"
GRID = "#e8ecef"


def configure() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 8.6,
            "axes.titlesize": 9.6,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.8,
            "axes.linewidth": 0.75,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load() -> tuple[pd.DataFrame, pd.DataFrame]:
    traj = pd.read_csv(DATA_DIR / "protocol_phase_trajectories.csv")
    summary = pd.read_csv(DATA_DIR / "protocol_phase_crossing_summary.csv")
    return traj, summary


def _save(fig: plt.Figure, stem: str, formats: tuple[str, ...]) -> tuple[Path, ...]:
    return save_figure_to_directory(fig, FIGURE_DIR, stem, formats, dpi=600, close=True)


def _interval_label(value: float) -> str:
    unit = "day" if np.isclose(value, 1.0) else "days"
    return f"Every {value:g} {unit}"


def _dose_count(beta_gp: float, interval: float, t: float = 28.0) -> float:
    return float(beta_gp) * (1.0 + np.floor((t - 1e-9) / float(interval)))


def _pivot(summary: pd.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = summary.pivot(index="media_interval_days", columns="beta_GP_mM", values=value).sort_index()
    return table.columns.to_numpy(dtype=float), table.index.to_numpy(dtype=float), table.to_numpy(dtype=float)


def plot_01_ridgeline_apatite(traj: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.1), sharex=True)
    cmap = plt.get_cmap("cividis")
    for ax, interval in zip(axes, sorted(traj["media_interval_days"].unique())):
        sub = traj[np.isclose(traj["media_interval_days"], interval)]
        doses = sorted(sub["beta_GP_mM"].unique())
        max_h = max(float(sub["H_apatite_like_model_units"].max()), 1e-12)
        offset = 0.74
        for j, dose in enumerate(doses):
            d = sub[np.isclose(sub["beta_GP_mM"], dose)]
            y = d["H_apatite_like_model_units"].to_numpy(dtype=float) / max_h
            base = j * offset
            color = cmap((dose - min(doses)) / (max(doses) - min(doses)))
            ax.fill_between(d["time_days"], base, base + y, color=color, alpha=0.72, linewidth=0)
            ax.plot(d["time_days"], base + y, color=color, linewidth=1.05)
        ax.set_yticks(np.arange(len(doses)) * offset, [f"{d:g}" for d in doses])
        ax.set_title(_interval_label(interval))
        ax.set_xlabel("Time (days)")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_ylabel(r"$\beta$-GP (mM), ridgeline offset")
    fig.suptitle("Apatite-like pool ridgelines", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "creative_01_apatite_ridgeline", formats)


def plot_02_phase_portrait(traj: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.55), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis")
    norm = Normalize(traj["beta_GP_mM"].min(), traj["beta_GP_mM"].max())
    for ax, interval in zip(axes, sorted(traj["media_interval_days"].unique())):
        sub = traj[np.isclose(traj["media_interval_days"], interval)]
        for dose in sorted(sub["beta_GP_mM"].unique()):
            d = sub[np.isclose(sub["beta_GP_mM"], dose)]
            ax.plot(
                d["A_ACP_like_model_units"],
                d["H_apatite_like_model_units"],
                color=cmap(norm(dose)),
                linewidth=1.2,
                alpha=0.82,
            )
            ax.scatter(d["A_ACP_like_model_units"].iloc[-1], d["H_apatite_like_model_units"].iloc[-1], s=10, color=cmap(norm(dose)))
        lim = max(float(sub["A_ACP_like_model_units"].max()), float(sub["H_apatite_like_model_units"].max()))
        ax.plot([0, lim], [0, lim], color="#777777", linestyle="--", linewidth=0.85, label=r"$H=A$")
        ax.set_title(_interval_label(interval))
        ax.set_xlabel("ACP-like A (model units)")
        ax.grid(color=GRID, linewidth=0.55)
    axes[0].set_ylabel("Apatite-like H (model units)")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, pad=0.018, fraction=0.035)
    cbar.set_label(r"$\beta$-GP (mM)")
    fig.suptitle("Phase portrait of precursor-to-apatite conversion", y=1.02)
    return _save(fig, "creative_02_phase_portrait_A_vs_H", formats)


def plot_03_endpoint_heatmap(summary: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    x, y, z = _pivot(summary, "H_final_model_units")
    fig, ax = plt.subplots(figsize=(6.5, 3.35))
    im = ax.imshow(z, aspect="auto", origin="lower", cmap="magma", extent=(x.min() - 0.5, x.max() + 0.5, y.min() - 0.5, y.max() + 0.5))
    for i, interval in enumerate(y):
        for j, dose in enumerate(x):
            if z[i, j] > 0:
                ax.text(dose, interval, f"{z[i,j]:.2g}", color="white" if z[i, j] > np.nanmax(z) * 0.45 else DARK, ha="center", va="center", fontsize=7.2)
    ax.set_xlabel(r"$\beta$-GP (mM)")
    ax.set_ylabel("Renewal interval (days)")
    ax.set_title("Day-28 apatite-like pool")
    cbar = fig.colorbar(im, ax=ax, pad=0.025)
    cbar.set_label("H at day 28 (model units)")
    fig.tight_layout()
    return _save(fig, "creative_03_endpoint_H_heatmap", formats)


def plot_04_bubble_map(summary: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    fig, ax = plt.subplots(figsize=(6.65, 3.55))
    s = 45 + 650 * summary["H_final_model_units"] / max(float(summary["H_final_model_units"].max()), 1e-12)
    c = summary["phi_H_final"].fillna(0.0)
    sc = ax.scatter(summary["beta_GP_mM"], summary["media_interval_days"], s=s, c=c, cmap="cividis", edgecolor="white", linewidth=0.8)
    not_reached = ~np.isfinite(summary["phase_dominance_time_days"])
    ax.scatter(summary.loc[not_reached, "beta_GP_mM"], summary.loc[not_reached, "media_interval_days"], marker="x", color="#777777", s=45, linewidth=1.1)
    ax.set_xlabel(r"$\beta$-GP (mM)")
    ax.set_ylabel("Renewal interval (days)")
    ax.set_yticks(sorted(summary["media_interval_days"].unique()))
    ax.set_title("Endpoint maturity and yield map")
    ax.grid(color=GRID, linewidth=0.6)
    cbar = fig.colorbar(sc, ax=ax, pad=0.025)
    cbar.set_label(r"Day-28 $\phi_H$")
    ax.legend(handles=[Line2D([], [], marker="o", color="none", markerfacecolor="#999999", markeredgecolor="white", markersize=8, label="Larger bubble = higher H"), Line2D([], [], marker="x", color="#777777", linestyle="none", label="No H>A by day 28")], frameon=False, loc="upper left")
    fig.tight_layout()
    return _save(fig, "creative_04_endpoint_bubble_map", formats)


def plot_05_dominance_lollipop(summary: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.35), sharey=True)
    colors = {1.0: "#1b9e77", 2.0: "#7570b3", 3.0: "#d95f02"}
    for ax, interval in zip(axes, sorted(summary["media_interval_days"].unique())):
        sub = summary[np.isclose(summary["media_interval_days"], interval)].sort_values("beta_GP_mM")
        reached = np.isfinite(sub["phase_dominance_time_days"])
        ax.hlines(sub.loc[reached, "beta_GP_mM"], 0, sub.loc[reached, "phase_dominance_time_days"], color=colors[interval], linewidth=1.4, alpha=0.68)
        ax.scatter(sub.loc[reached, "phase_dominance_time_days"], sub.loc[reached, "beta_GP_mM"], s=34, color=colors[interval], edgecolor="white", linewidth=0.8)
        ax.scatter(np.full(np.count_nonzero(~reached), 28.0), sub.loc[~reached, "beta_GP_mM"], marker="x", s=36, color="#777777", linewidth=1.2)
        ax.set_xlim(0, 28.5)
        ax.set_xlabel(r"First $H>A$ day")
        ax.set_title(_interval_label(interval))
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[0].set_ylabel(r"$\beta$-GP (mM)")
    fig.suptitle("Time to apatite dominance", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "creative_05_dominance_lollipop", formats)


def plot_06_endpoint_composition(summary: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35), sharey=True)
    for ax, interval in zip(axes, sorted(summary["media_interval_days"].unique())):
        sub = summary[np.isclose(summary["media_interval_days"], interval)].sort_values("beta_GP_mM")
        x = sub["beta_GP_mM"].to_numpy()
        a = sub["A_final_model_units"].to_numpy()
        h = sub["H_final_model_units"].to_numpy()
        ax.bar(x, a, color=ACP, width=0.72, label="ACP-like A")
        ax.bar(x, h, bottom=a, color=HAP, width=0.72, label="Apatite-like H")
        ax.set_title(_interval_label(interval))
        ax.set_xlabel(r"$\beta$-GP (mM)")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    axes[0].set_ylabel("Day-28 mineral pool (model units)")
    axes[-1].legend(frameon=False, loc="upper left")
    fig.suptitle("Endpoint phase composition", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "creative_06_endpoint_composition_bars", formats)


def plot_07_surface_contours(summary: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    x, y, z = _pivot(summary, "H_final_model_units")
    fig, ax = plt.subplots(figsize=(6.6, 3.45))
    xx, yy = np.meshgrid(x, y)
    levels = np.linspace(0, max(float(np.nanmax(z)), 1e-12), 11)
    cf = ax.contourf(xx, yy, z, levels=levels, cmap="viridis")
    cs = ax.contour(xx, yy, z, levels=levels[2::2], colors="white", linewidths=0.75, alpha=0.9)
    ax.clabel(cs, inline=True, fontsize=6.7, fmt="%.1f")
    ax.scatter(summary["beta_GP_mM"], summary["media_interval_days"], s=8, color=DARK, alpha=0.45)
    ax.set_xlabel(r"$\beta$-GP (mM)")
    ax.set_ylabel("Renewal interval (days)")
    ax.set_title("Interpolated endpoint H contours")
    cbar = fig.colorbar(cf, ax=ax, pad=0.025)
    cbar.set_label("H at day 28 (model units)")
    fig.tight_layout()
    return _save(fig, "creative_07_endpoint_H_contours", formats)


def plot_08_supersaturation_calendar(traj: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 5.3), sharex=True)
    doses = [4.0, 6.0, 8.0, 10.0]
    cmap = plt.get_cmap("plasma")
    for ax, interval in zip(axes, sorted(traj["media_interval_days"].unique())):
        sub = traj[np.isclose(traj["media_interval_days"], interval)]
        for j, dose in enumerate(doses):
            d = sub[np.isclose(sub["beta_GP_mM"], dose)]
            if d.empty:
                continue
            above = d["S_A"].to_numpy(dtype=float) >= 1.0
            y0 = j - 0.34
            for start, stop in _true_runs(d["time_days"].to_numpy(dtype=float), above):
                ax.broken_barh([(start, stop - start)], (y0, 0.62), facecolors=cmap(j / max(len(doses) - 1, 1)), edgecolors="none", alpha=0.88)
        ax.set_yticks(range(len(doses)), [f"{d:g} mM" for d in doses])
        ax.set_ylabel(_interval_label(interval))
        ax.grid(axis="x", color=GRID, linewidth=0.6)
    axes[-1].set_xlabel("Time with precursor-forming conditions, S_A >= 1 (days)")
    fig.suptitle("Supersaturation calendar by protocol", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, "creative_08_supersaturation_calendar", formats)


def _true_runs(time: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    starts = np.flatnonzero(mask & np.r_[True, ~mask[:-1]])
    stops = np.flatnonzero(mask & np.r_[~mask[1:], True])
    return [(float(time[s]), float(time[e])) for s, e in zip(starts, stops) if e >= s]


def plot_09_dose_yield_frontier(summary: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    fig, ax = plt.subplots(figsize=(6.6, 3.65))
    colors = {1.0: "#1b9e77", 2.0: "#7570b3", 3.0: "#d95f02"}
    for interval in sorted(summary["media_interval_days"].unique()):
        sub = summary[np.isclose(summary["media_interval_days"], interval)].sort_values("beta_GP_mM").copy()
        sub["cumulative_beta_gp_mM"] = [_dose_count(b, interval) for b in sub["beta_GP_mM"]]
        ax.plot(sub["cumulative_beta_gp_mM"], sub["H_final_model_units"], marker="o", color=colors[interval], linewidth=1.7, label=_interval_label(interval))
        for row in sub.itertuples(index=False):
            if row.H_final_model_units > 0:
                ax.text(row.cumulative_beta_gp_mM, row.H_final_model_units, f"{row.beta_GP_mM:g}", fontsize=6.6, ha="center", va="bottom", color=colors[interval])
    ax.set_xlabel("Cumulative beta-GP supplied by day 28 (mM)")
    ax.set_ylabel("Day-28 apatite-like H (model units)")
    ax.set_title("Dose-yield frontier")
    ax.grid(color=GRID, linewidth=0.6)
    ax.legend(frameon=False, title="Renewal")
    fig.tight_layout()
    return _save(fig, "creative_09_dose_yield_frontier", formats)


def plot_10_polar_endpoint(summary: pd.DataFrame, formats: tuple[str, ...]) -> tuple[Path, ...]:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.75), subplot_kw={"projection": "polar"})
    max_h = max(float(summary["H_final_model_units"].max()), 1e-12)
    for ax, interval in zip(axes, sorted(summary["media_interval_days"].unique())):
        sub = summary[np.isclose(summary["media_interval_days"], interval)].sort_values("beta_GP_mM")
        theta = 2.0 * np.pi * (sub["beta_GP_mM"].to_numpy(dtype=float) - 1.0) / 10.0
        radius = sub["H_final_model_units"].to_numpy(dtype=float) / max_h
        phi = sub["phi_H_final"].fillna(0.0).to_numpy(dtype=float)
        width = 2.0 * np.pi / 12.0
        bars = ax.bar(theta, radius, width=width, bottom=0.0, color=plt.get_cmap("cividis")(phi), edgecolor="white", linewidth=0.55)
        del bars
        ax.set_title(_interval_label(interval), pad=13)
        ax.set_xticks(theta, [f"{d:g}" for d in sub["beta_GP_mM"]], fontsize=7)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0], ["", "0.5", "", "1.0"], fontsize=7)
        ax.set_ylim(0, 1.0)
        ax.grid(color="#dddddd", linewidth=0.55)
    sm = plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap="cividis")
    cbar = fig.colorbar(sm, ax=axes, pad=0.055, fraction=0.035)
    cbar.set_label(r"Day-28 $\phi_H$")
    fig.suptitle("Circular endpoint fingerprint: radius = normalized H", y=1.02)
    return _save(fig, "creative_10_polar_endpoint_fingerprint", formats)


PLOTTERS = (
    plot_01_ridgeline_apatite,
    plot_02_phase_portrait,
    plot_03_endpoint_heatmap,
    plot_04_bubble_map,
    plot_05_dominance_lollipop,
    plot_06_endpoint_composition,
    plot_07_surface_contours,
    plot_08_supersaturation_calendar,
    plot_09_dose_yield_frontier,
    plot_10_polar_endpoint,
)


def export_creative_protocol_plots(formats: tuple[str, ...] = DEFAULT_FORMATS) -> list[Path]:
    configure()
    traj, summary = _load()
    saved: list[Path] = []
    saved.extend(plot_01_ridgeline_apatite(traj, formats))
    saved.extend(plot_02_phase_portrait(traj, formats))
    saved.extend(plot_03_endpoint_heatmap(summary, formats))
    saved.extend(plot_04_bubble_map(summary, formats))
    saved.extend(plot_05_dominance_lollipop(summary, formats))
    saved.extend(plot_06_endpoint_composition(summary, formats))
    saved.extend(plot_07_surface_contours(summary, formats))
    saved.extend(plot_08_supersaturation_calendar(traj, formats))
    saved.extend(plot_09_dose_yield_frontier(summary, formats))
    saved.extend(plot_10_polar_endpoint(summary, formats))
    return saved


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate ten diverse protocol-optimization exploratory plots.")
    parser.add_argument("--figure-formats", default="pdf,png")
    args = parser.parse_args()
    saved = export_creative_protocol_plots(parse_figure_formats(args.figure_formats))
    print("Saved creative protocol plots:")
    for path in saved:
        print(f"- {path}")


if __name__ == "__main__":
    main()
