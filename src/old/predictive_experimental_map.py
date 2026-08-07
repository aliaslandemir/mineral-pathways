"""Archived exploratory predictive experimental map."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from src.config import MODEL_DEFAULTS, RUN_DEFAULTS, STOCHASTIC_DEFAULTS, build_base_params
from src.io_utils import OUTPUT_ROOT, parse_figure_formats, save_figure, save_figure_to_directory
from src.model import beta_gp_hydrolysis_rate_constant, simulate
from src.regime_map import analytical_q_pi_crit


FIGURE_STEM = "main_fig_predictive_experimental_map"
WORKFLOW_DIR = OUTPUT_ROOT / "predictive_experimental_map"

BASELINE_BETA_GP = float(MODEL_DEFAULTS.beta_GP)
BASELINE_MEDIA_INTERVAL = float(RUN_DEFAULTS.media_interval)
MINERAL_POOL_EPS = 1e-12
PERSISTENCE_DAYS = 0.25
ENDPOINT_DAY = float(RUN_DEFAULTS.t_end)
NONNEGATIVE_TOLERANCE = 1e-8

BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
DARK = "#303030"
MID_GREY = "#777777"
LIGHT_BLUE = "#DCEAF2"


def configure_matplotlib() -> None:
    """Apply restrained full-width PLOS Computational Biology figure styling."""
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            # Arimo is the installed metric-compatible Arial substitute.
            "font.family": "Arimo",
            "font.size": 8.0,
            "axes.titlesize": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.25,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _validated_arrays(
    time: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.asarray(time, dtype=float)
    A = np.asarray(A, dtype=float)
    H = np.asarray(H, dtype=float)
    if time.ndim != 1 or A.ndim != 1 or H.ndim != 1 or not (time.size == A.size == H.size):
        raise ValueError("time, A, and H must be one-dimensional arrays of equal length.")
    if time.size < 2 or not np.all(np.isfinite(time)) or not np.all(np.isfinite(A)) or not np.all(np.isfinite(H)):
        raise ValueError("time, A, and H must contain at least two finite values.")
    if np.any(np.diff(time) < -1e-12):
        raise ValueError("time must be nondecreasing.")
    if float(np.min(A)) < -NONNEGATIVE_TOLERANCE or float(np.min(H)) < -NONNEGATIVE_TOLERANCE:
        raise ValueError("A and H must be non-negative within numerical tolerance.")
    return time, np.maximum(A, 0.0), np.maximum(H, 0.0)


def _interpolate_crossing(
    t0: float,
    t1: float,
    y0: float,
    y1: float,
    threshold: float,
) -> float:
    if t1 <= t0 or not np.isfinite(y0 + y1) or np.isclose(y1, y0):
        return float(t1)
    fraction = np.clip((threshold - y0) / (y1 - y0), 0.0, 1.0)
    return float(t0 + fraction * (t1 - t0))


def compute_persistent_crossing(
    time: np.ndarray,
    values: np.ndarray,
    threshold: float,
    persistence_days: float = PERSISTENCE_DAYS,
    *,
    start_time: float = 0.0,
) -> float:
    """Return the first upward crossing sustained for the requested duration."""
    time = np.asarray(time, dtype=float)
    values = np.asarray(values, dtype=float)
    if time.ndim != 1 or values.ndim != 1 or time.size != values.size:
        raise ValueError("time and values must be one-dimensional arrays of equal length.")
    if time.size < 2 or not np.all(np.isfinite(time)) or np.any(np.diff(time) < -1e-12):
        raise ValueError("time must be a finite, nondecreasing array with at least two values.")
    valid = np.isfinite(values) & (values >= float(threshold)) & (time >= float(start_time))
    starts = np.flatnonzero(valid & np.r_[True, ~valid[:-1]])
    for start in starts:
        crossing = float(time[start])
        if start > 0 and np.isfinite(values[start - 1]) and time[start - 1] >= start_time - 1e-12:
            crossing = _interpolate_crossing(
                float(time[start - 1]),
                float(time[start]),
                float(values[start - 1]),
                float(values[start]),
                float(threshold),
            )
        invalid_after = np.flatnonzero(~valid[start:])
        if invalid_after.size:
            stop = start + int(invalid_after[0])
            last_valid = stop - 1
            exit_time = float(time[last_valid])
            if np.isfinite(values[stop]):
                exit_time = _interpolate_crossing(
                    float(time[last_valid]),
                    float(time[stop]),
                    float(values[last_valid]),
                    float(values[stop]),
                    float(threshold),
                )
        else:
            exit_time = float(time[-1])
        if exit_time - crossing >= float(persistence_days) - 1e-12:
            return crossing
    return float("nan")


def compute_protocol_metrics(
    time: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    S_A: np.ndarray,
    *,
    beta_GP: float,
    media_interval: float,
    endpoint_day: float = ENDPOINT_DAY,
) -> dict[str, float]:
    """Calculate timing and endpoint metrics without calibrated mineral-amount cutoffs."""
    time, A, H = _validated_arrays(time, A, H)
    S_A = np.asarray(S_A, dtype=float)
    if S_A.ndim != 1 or S_A.size != time.size or not np.all(np.isfinite(S_A)):
        raise ValueError("S_A must be a finite one-dimensional array with the same length as time.")
    total = A + H
    phi = np.divide(
        H,
        total,
        out=np.full_like(total, np.nan),
        where=total > MINERAL_POOL_EPS,
    )
    finite_phi = phi[np.isfinite(phi)]
    if finite_phi.size and (np.min(finite_phi) < -NONNEGATIVE_TOLERANCE or np.max(finite_phi) > 1.0 + NONNEGATIVE_TOLERANCE):
        raise ValueError("Apatite fraction must remain between 0 and 1.")

    onset = compute_persistent_crossing(time, S_A, 1.0, PERSISTENCE_DAYS)
    dominance = float("nan")
    if np.isfinite(onset):
        dominance = compute_persistent_crossing(
            time,
            phi,
            0.5,
            PERSISTENCE_DAYS,
            start_time=onset,
        )
        if np.isfinite(dominance) and dominance < onset - 1e-9:
            raise ValueError("Apatite dominance cannot precede precursor-threshold onset.")

    if endpoint_day > float(time[-1]) + 1e-9:
        raise ValueError(f"Simulation does not reach day {endpoint_day:g}.")
    A_day = float(np.interp(endpoint_day, time, A))
    H_day = float(np.interp(endpoint_day, time, H))
    total_day = A_day + H_day
    phi_day = H_day / total_day if total_day > MINERAL_POOL_EPS else float("nan")
    return {
        "beta_GP": float(beta_GP),
        "media_interval": float(media_interval),
        "precursor_threshold_time": float(onset),
        "onset_time": float(onset),
        "apatite_dominance_time": float(dominance),
        "ACP_window_duration": float(dominance - onset) if np.isfinite(onset) and np.isfinite(dominance) else float("nan"),
        "total_mineral_day28": total_day,
        "phi_H_day28": float(np.clip(phi_day, 0.0, 1.0)) if np.isfinite(phi_day) else float("nan"),
    }


def _simulate_condition(
    base_params: dict[str, float | str],
    beta_gp: float,
    media_interval: float,
    *,
    t_end: float,
    dt: float,
) -> tuple[object, dict[str, float]]:
    params = dict(base_params)
    params["beta_GP"] = float(beta_gp)
    sim = simulate(params, t_end=float(t_end), dt=float(dt), media_interval=float(media_interval))
    metrics = compute_protocol_metrics(
        sim.t,
        sim.A,
        sim.H,
        sim.S_A,
        beta_GP=float(beta_gp),
        media_interval=float(media_interval),
    )
    return sim, metrics


def analytical_minimum_beta_gp(
    base_params: dict[str, float | str],
    media_interval: np.ndarray | float,
) -> np.ndarray:
    """Beta-GP needed to reach the model's analytical S_A=1 threshold in one cycle."""
    intervals = np.asarray(media_interval, dtype=float)
    release_fraction = 1.0 - np.exp(-beta_gp_hydrolysis_rate_constant(base_params) * intervals)
    return analytical_q_pi_crit(base_params) / np.maximum(release_fraction, 1e-12)


def find_minimum_beta_gp(
    base_params: dict[str, float | str],
    media_interval: float,
    *,
    t_end: float,
    dt: float,
    beta_max: float = 20.0,
    iterations: int = 16,
) -> float:
    """Find the minimum dose giving persistent precursor-forming conditions by day 28."""
    _, high_metrics = _simulate_condition(base_params, beta_max, media_interval, t_end=t_end, dt=dt)
    if not np.isfinite(high_metrics["onset_time"]):
        return float("nan")
    low = max(0.0, 0.98 * float(analytical_minimum_beta_gp(base_params, media_interval)))
    high = float(beta_max)
    for _ in range(int(iterations)):
        middle = 0.5 * (low + high)
        _, metrics = _simulate_condition(base_params, middle, media_interval, t_end=t_end, dt=dt)
        if np.isfinite(metrics["onset_time"]):
            high = middle
        else:
            low = middle
    return high


def build_predictive_outputs(
    base_params: dict[str, float | str],
    *,
    t_end: float,
    dt: float,
    beta_points: int = 81,
    threshold_points: int = 17,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if t_end < ENDPOINT_DAY:
        raise ValueError("The predictive figure requires a simulation horizon of at least 28 days.")
    beta_min = min(float(v) for v in STOCHASTIC_DEFAULTS.beta_gp_levels)
    beta_max = max(float(v) for v in STOCHASTIC_DEFAULTS.beta_gp_levels)
    beta_grid = np.unique(np.r_[np.linspace(beta_min, beta_max, max(21, int(beta_points))), BASELINE_BETA_GP])
    response_rows: list[dict[str, float]] = []
    for beta_gp in beta_grid:
        _, metrics = _simulate_condition(
            base_params,
            float(beta_gp),
            BASELINE_MEDIA_INTERVAL,
            t_end=ENDPOINT_DAY,
            dt=dt,
        )
        response_rows.append(metrics)

    interval_grid = np.unique(
        np.r_[np.linspace(1.0, 5.0, max(9, int(threshold_points))), BASELINE_MEDIA_INTERVAL]
    )
    numerical_threshold = [
        find_minimum_beta_gp(
            base_params,
            float(interval),
            t_end=ENDPOINT_DAY,
            dt=dt,
        )
        for interval in interval_grid
    ]
    threshold = pd.DataFrame(
        {
            "media_interval": interval_grid,
            "minimum_beta_GP_numerical_SA": numerical_threshold,
            "minimum_beta_GP_analytical": analytical_minimum_beta_gp(base_params, interval_grid),
        }
    )
    response = pd.DataFrame(response_rows).sort_values("beta_GP", ignore_index=True)
    baseline = response.loc[np.isclose(response["beta_GP"], BASELINE_BETA_GP)]
    if baseline.empty:
        raise ValueError("The baseline beta-GP=10 mM response is unavailable.")
    return response, threshold


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.04, label, transform=ax.transAxes, fontsize=10.0, fontweight="bold", ha="left", va="bottom")


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out")


SELECTED_INTERVALS = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
ENDPOINT_DOSES = np.array(STOCHASTIC_DEFAULTS.beta_gp_levels, dtype=float)
SELECTED_DOSES = np.array([value for value in ENDPOINT_DOSES if value >= 5.0], dtype=float)


def _nearest_rows(frame: pd.DataFrame, column: str, values: np.ndarray) -> pd.DataFrame:
    source = frame[column].to_numpy(dtype=float)
    return pd.DataFrame([frame.iloc[int(np.argmin(np.abs(source - value)))] for value in values]).reset_index(drop=True)


def plot_dose_threshold(ax: plt.Axes, threshold: pd.DataFrame) -> None:
    selected = _nearest_rows(threshold, "media_interval", SELECTED_INTERVALS)
    analytical = selected["minimum_beta_GP_analytical"].to_numpy(dtype=float)
    numerical = selected["minimum_beta_GP_numerical_SA"].to_numpy(dtype=float)
    y = np.arange(len(selected), dtype=float)
    for yi, x0, x1 in zip(y, analytical, numerical):
        ax.plot([x0, x1], [yi, yi], color="#AEB8BE", linewidth=2.2, solid_capstyle="round", zorder=1)
    ax.scatter(analytical, y, s=25, facecolor="white", edgecolor=DARK, linewidth=0.9, zorder=3)
    ax.scatter(numerical, y, s=29, facecolor=BLUE, edgecolor=BLUE, linewidth=0.8, zorder=4)
    for yi, value in zip(y, numerical):
        ax.text(value + 0.17, yi - 0.14, f"{value:.2f}", color=BLUE, fontsize=6.3, ha="left", va="center")
    baseline_y = int(np.where(np.isclose(SELECTED_INTERVALS, BASELINE_MEDIA_INTERVAL))[0][0])
    ax.scatter(
        [BASELINE_BETA_GP],
        [baseline_y],
        s=49,
        facecolor="white",
        edgecolor=ORANGE,
        linewidth=1.25,
        zorder=5,
    )
    ax.annotate(
        "10 mM baseline",
        xy=(BASELINE_BETA_GP, baseline_y),
        xytext=(9.65, baseline_y + 0.43),
        fontsize=6.4,
        color=ORANGE,
        ha="right",
        arrowprops={"arrowstyle": "-", "lw": 0.55, "color": ORANGE},
    )
    ax.set_yticks(y, [f"{value:g} d" for value in SELECTED_INTERVALS])
    ax.invert_yaxis()
    ax.set_xlim(0.0, 11.0)
    ax.set_xlabel(r"Minimum $\beta$-GP (mM)")
    ax.set_ylabel("Renewal interval")
    ax.set_title("Minimum dose by renewal", loc="left", fontweight="bold")
    ax.legend(
        handles=[
            Line2D([], [], marker="o", linestyle="none", markerfacecolor=BLUE, markeredgecolor=BLUE, label=r"Numerical $S_A\geq1$"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="white", markeredgecolor=DARK, label=r"Analytical $S_A=1$"),
        ],
        frameon=False,
        loc="lower right",
        handletextpad=0.45,
        borderaxespad=0.1,
    )
    _panel_label(ax, "A")


def plot_sampling_schedule(ax: plt.Axes, response: pd.DataFrame) -> None:
    selected = _nearest_rows(response, "beta_GP", SELECTED_DOSES)
    onset = selected["onset_time"].to_numpy(dtype=float)
    dominance = selected["apatite_dominance_time"].to_numpy(dtype=float)
    y = np.arange(len(selected), dtype=float)
    baseline_y = int(np.where(np.isclose(SELECTED_DOSES, BASELINE_BETA_GP))[0][0])
    ax.axhspan(baseline_y - 0.35, baseline_y + 0.35, color="#FAF3EC", linewidth=0.0, zorder=0)
    for yi, start, stop, dose in zip(y, onset, dominance, SELECTED_DOSES):
        if not np.isfinite(start) or not np.isfinite(stop):
            continue
        width = 5.4 if np.isclose(dose, BASELINE_BETA_GP) else 4.0
        ax.plot([start, stop], [yi, yi], color=LIGHT_BLUE, linewidth=width, solid_capstyle="round", zorder=1)
        ax.scatter([start], [yi], s=29, facecolor="white", edgecolor=BLUE, linewidth=1.0, zorder=3)
        ax.scatter([stop], [yi], s=29, marker="s", facecolor="white", edgecolor=ORANGE, linewidth=1.0, zorder=3)
        ax.text(stop + 0.25, yi, f"{stop - start:.1f} d", fontsize=6.7, color=MID_GREY, va="center")
    ax.text(onset[0], -0.43, "onset", color=BLUE, fontsize=6.5, ha="center")
    ax.text(dominance[0], -0.43, "apatite dominance", color=ORANGE, fontsize=6.5, ha="center")
    ax.set_yticks(y, [f"{dose:g} mM" for dose in SELECTED_DOSES])
    ax.get_yticklabels()[baseline_y].set_fontweight("bold")
    ax.get_yticklabels()[baseline_y].set_color(ORANGE)
    ax.invert_yaxis()
    ax.set_ylim(len(y) - 0.5, -0.62)
    ax.set_xlim(0.0, 11.5)
    ax.set_xlabel("Culture time (days)")
    ax.set_ylabel(r"$\beta$-GP")
    ax.set_title("ACP-rich sampling schedule", loc="left", fontweight="bold")
    _panel_label(ax, "B")


def plot_endpoint_lollipops(fig: plt.Figure, parent_spec, response: pd.DataFrame) -> tuple[plt.Axes, plt.Axes]:
    selected = _nearest_rows(response, "beta_GP", ENDPOINT_DOSES)
    beta = selected["beta_GP"].to_numpy(dtype=float)
    total = selected["total_mineral_day28"].to_numpy(dtype=float)
    phi = selected["phi_H_day28"].to_numpy(dtype=float)
    subgrid = parent_spec.subgridspec(2, 1, hspace=0.10)
    ax_total = fig.add_subplot(subgrid[0, 0])
    ax_phi = fig.add_subplot(subgrid[1, 0], sharex=ax_total)
    ax_total.plot(response["beta_GP"], response["total_mineral_day28"], color=BLUE, linewidth=0.8, alpha=0.38)
    ax_phi.plot(response["beta_GP"], response["phi_H_day28"], color=ORANGE, linewidth=0.8, alpha=0.38)
    ax_total.vlines(beta, 0.0, total, color="#B9CFDB", linewidth=1.1)
    ax_total.scatter(beta, total, s=24, facecolor="white", edgecolor=BLUE, linewidth=0.95, zorder=3)
    finite = np.isfinite(phi)
    ax_phi.vlines(beta[finite], 0.5, phi[finite], color="#EBC4AE", linewidth=1.1)
    ax_phi.scatter(beta[finite], phi[finite], s=24, facecolor="white", edgecolor=ORANGE, linewidth=0.95, zorder=3)
    ax_phi.axhline(0.5, color=MID_GREY, linestyle=(0, (4, 2.5)), linewidth=0.7)
    baseline = selected.iloc[int(np.argmin(np.abs(beta - BASELINE_BETA_GP)))]
    ax_total.scatter([BASELINE_BETA_GP], [baseline["total_mineral_day28"]], s=47, facecolor="white", edgecolor=BLUE, linewidth=1.3, zorder=4)
    ax_phi.scatter([BASELINE_BETA_GP], [baseline["phi_H_day28"]], s=47, facecolor="white", edgecolor=ORANGE, linewidth=1.3, zorder=4)
    ax_total.annotate(f"{baseline['total_mineral_day28']:.2f}", (BASELINE_BETA_GP, baseline["total_mineral_day28"]), xytext=(5, 5), textcoords="offset points", fontsize=6.4, color=BLUE)
    ax_phi.annotate(f"{baseline['phi_H_day28']:.3f}", (BASELINE_BETA_GP, baseline["phi_H_day28"]), xytext=(5, -10), textcoords="offset points", fontsize=6.4, color=ORANGE)
    ax_total.set_ylabel("Total mineral\n(model units)")
    ax_phi.set_ylabel(r"Apatite fraction, $\phi_H$")
    ax_phi.set_xlabel(r"$\beta$-GP (mM)")
    ax_phi.set_xticks(ENDPOINT_DOSES, [f"{value:g}" for value in ENDPOINT_DOSES])
    ax_total.set_ylim(bottom=0.0)
    ax_phi.set_ylim(0.0, 1.0)
    ax_total.tick_params(labelbottom=False)
    ax_total.set_title("Day-28 outcomes", loc="left", fontweight="bold")
    _clean_axis(ax_total)
    _clean_axis(ax_phi)
    _panel_label(ax_total, "C")
    return ax_total, ax_phi


def make_final_predictive_figure(
    base_params: dict[str, float | str],
    *,
    t_end: float = ENDPOINT_DAY,
    dt: float = RUN_DEFAULTS.dt,
    beta_points: int = 81,
    threshold_points: int = 17,
) -> tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    configure_matplotlib()
    response, threshold = build_predictive_outputs(
        base_params,
        t_end=t_end,
        dt=dt,
        beta_points=beta_points,
        threshold_points=threshold_points,
    )
    fig = plt.figure(figsize=(7.09, 3.55), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(0.93, 1.13, 1.04), wspace=0.08)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    plot_dose_threshold(ax_a, threshold)
    plot_sampling_schedule(ax_b, response)
    ax_c1, ax_c2 = plot_endpoint_lollipops(fig, grid[0, 2], response)
    _clean_axis(ax_a)
    _clean_axis(ax_b)
    _clean_axis(ax_c1)
    _clean_axis(ax_c2)
    return fig, response, threshold


def write_predictive_caption(output_dir: Path) -> None:
    caption = (
        "Predictive experimental guidance from the ACP-to-apatite model. "
        "(A) Minimum beta-glycerophosphate (beta-GP) concentration required to produce "
        "persistent precursor-forming conditions by day 28 for experimentally selected "
        "medium-renewal intervals. The numerical criterion is S_A >= 1 for at least "
        "0.25 day. The dashed curve is the analytical precursor-formation threshold "
        "S_A=1; the open orange circle marks the baseline protocol. "
        "(B) Predicted ACP-rich sampling schedules for selected beta-GP concentrations "
        "at the baseline 2-day renewal interval. Each bar extends from precursor-threshold onset "
        "to persistent apatite dominance (phi_H >= 0.5 for at least 0.25 day). "
        "(C) Day-28 total mineral and apatite fraction for selected beta-GP concentrations "
        "at the same renewal interval. Enlarged symbols and numeric labels mark the "
        "baseline beta-GP concentration of 10 mM. Other model "
        "parameters were fixed to the baseline values. A and H are uncalibrated model "
        "pools, so absolute mineral-pool amount thresholds were not used."
    )
    (Path(output_dir) / "predictive_experimental_map_caption.txt").write_text(caption + "\n", encoding="utf-8")


def _save_figure(
    fig: plt.Figure,
    output_dir: Path,
    formats: tuple[str, ...],
) -> None:
    output_dir = Path(output_dir)
    if output_dir.name.startswith("run_"):
        save_figure(fig, output_dir, FIGURE_STEM, formats, dpi=600, close=True, pad_inches=0.03)
    else:
        save_figure_to_directory(
            fig,
            output_dir / "figures",
            FIGURE_STEM,
            formats,
            dpi=600,
            close=True,
            pad_inches=0.03,
        )


def export_predictive_experimental_map(
    output_dir: Path,
    base_params: dict[str, float | str],
    *,
    t_end: float,
    dt: float,
    formats: tuple[str, ...],
    beta_points: int = 81,
    tmedia_points: int = 17,
    source_csv: Path | None = None,
) -> Path:
    """Export the three-panel predictive experimental-guidance figure and data."""
    if source_csv is not None:
        warnings.warn("source_csv is ignored because the figure is regenerated from the supplied model parameters.", RuntimeWarning)
    output_dir = Path(output_dir)
    data_dir = output_dir / "predictive_experimental_map"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig, response, threshold = make_final_predictive_figure(
        base_params,
        t_end=t_end,
        dt=dt,
        beta_points=beta_points,
        threshold_points=tmedia_points,
    )
    response_path = data_dir / "predictive_experimental_map.csv"
    response.to_csv(response_path, index=False)
    threshold.to_csv(data_dir / "minimum_beta_gp_by_renewal.csv", index=False)
    write_predictive_caption(output_dir)
    if formats:
        _save_figure(fig, output_dir, formats)
    else:
        plt.close(fig)

    baseline = response.loc[np.isclose(response["beta_GP"], BASELINE_BETA_GP)].iloc[0]
    print("Predictive figure baseline (10 mM beta-GP, 2-day renewal):")
    print(
        baseline[
            [
                "precursor_threshold_time",
                "apatite_dominance_time",
                "ACP_window_duration",
                "total_mineral_day28",
                "phi_H_day28",
            ]
        ].to_string(float_format=lambda value: f"{value:.4g}")
    )
    print("\nMinimum beta-GP for persistent S_A >= 1 by renewal interval:")
    print(threshold.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    return response_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the three-panel predictive experimental-guidance figure.")
    parser.add_argument("--parameters-json", type=Path, help="Baseline parameters from a completed main-model run.")
    parser.add_argument("--output-dir", type=Path, default=WORKFLOW_DIR)
    parser.add_argument("--beta-points", type=int, default=81)
    parser.add_argument("--threshold-points", type=int, default=17)
    parser.add_argument("--figure-formats", default="pdf,png", help="Comma-separated: pdf,png.")
    parser.add_argument("--t-end", type=float, default=ENDPOINT_DAY)
    parser.add_argument("--dt", type=float, default=RUN_DEFAULTS.dt)
    args = parser.parse_args()

    if args.parameters_json is None:
        params = build_base_params()
    else:
        params = json.loads(args.parameters_json.read_text(encoding="utf-8"))
    formats = parse_figure_formats(args.figure_formats)
    csv_path = export_predictive_experimental_map(
        args.output_dir,
        params,
        t_end=float(args.t_end),
        dt=float(args.dt),
        formats=formats,
        beta_points=int(args.beta_points),
        tmedia_points=int(args.threshold_points),
    )
    print(f"Saved predictive data: {csv_path}")


if __name__ == "__main__":
    main()
