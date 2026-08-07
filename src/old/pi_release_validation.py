"""
Archived validation of beta-GP phosphate release against digitized
short-time Pi-release data.

Example:
python -m src.pi_release_validation --figure-formats pdf,png
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_DEFAULTS, PLOT_DEFAULTS, RUN_DEFAULTS, build_base_params
from src.io_utils import create_run_directory, parse_figure_formats, save_figure
from src.model import (
    beta_gp_available_phosphate,
    hydrolysis_fraction_from_rate,
    synchronize_hydrolysis_parameters,
)
from src.visualization import clean_axis, configure_matplotlib


TIME_H = np.array([0, 1, 2, 3, 4, 5, 6], dtype=float)
CONTROL_PI_MM = np.array([1.35, 1.43, 1.53, 1.52, 1.47, 1.18, 1.67], dtype=float)
BETA_GP_PI_MM = np.array([1.35, 2.15, 2.70, 3.50, 4.30, 4.90, 5.80], dtype=float)

ERRORBAR_TIME_H = np.array([2, 3, 4, 5, 6], dtype=float)
ERRORBAR_LOW_MM = np.array([2.25, 3.20, 3.90, 4.75, 5.45], dtype=float)
ERRORBAR_HIGH_MM = np.array([3.05, 3.80, 4.65, 5.05, 6.15], dtype=float)

REPORTED_INTERCEPT_MM = 1.35
REPORTED_SLOPE_MM_PER_H = 0.73
VALIDATION_END_H = 7.0


def reported_linear_pi(time_h: np.ndarray | float) -> np.ndarray:
    return REPORTED_INTERCEPT_MM + REPORTED_SLOPE_MM_PER_H * np.asarray(time_h, dtype=float)


def model_release_pi(time_h: np.ndarray, params: dict[str, float]) -> np.ndarray:
    days = np.asarray(time_h, dtype=float) / 24.0
    return np.asarray([beta_gp_available_phosphate(params, duration_days=float(day)) for day in days])


def finite_difference_slope(time_h: np.ndarray, values: np.ndarray) -> float:
    x = np.asarray(time_h, dtype=float)
    y = np.asarray(values, dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def write_validation_csv(
    out_dir: Path,
    time_h: np.ndarray,
    model_total_aligned: np.ndarray,
    model_total_native: np.ndarray,
    model_released: np.ndarray,
) -> None:
    path = out_dir / "pi_release_validation_7h.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "time_h",
                "literature_control_Pi_mM",
                "literature_betaGP_10mM_Pi_mM",
                "literature_reported_fit_Pi_mM",
                "model_released_from_betaGP_mM",
                "model_total_Pi_aligned_to_literature_initial_mM",
                "model_total_Pi_native_medium_mM",
            ],
        )
        writer.writeheader()
        for i, h in enumerate(time_h):
            nearest = np.where(np.isclose(TIME_H, h))[0]
            lit_control = CONTROL_PI_MM[int(nearest[0])] if nearest.size else np.nan
            lit_beta_gp = BETA_GP_PI_MM[int(nearest[0])] if nearest.size else np.nan
            writer.writerow(
                {
                    "time_h": f"{h:.10g}",
                    "literature_control_Pi_mM": f"{lit_control:.10g}" if np.isfinite(lit_control) else "",
                    "literature_betaGP_10mM_Pi_mM": f"{lit_beta_gp:.10g}" if np.isfinite(lit_beta_gp) else "",
                    "literature_reported_fit_Pi_mM": f"{float(reported_linear_pi(h)):.10g}",
                    "model_released_from_betaGP_mM": f"{model_released[i]:.10g}",
                    "model_total_Pi_aligned_to_literature_initial_mM": f"{model_total_aligned[i]:.10g}",
                    "model_total_Pi_native_medium_mM": f"{model_total_native[i]:.10g}",
                }
            )


def write_metrics_csv(out_dir: Path, params: dict[str, float], model_7h_total_aligned: float) -> dict[str, float]:
    model_release_7h = float(model_7h_total_aligned - REPORTED_INTERCEPT_MM)
    reported_7h_total = float(reported_linear_pi(VALIDATION_END_H))
    reported_release_7h = reported_7h_total - REPORTED_INTERCEPT_MM
    model_initial_slope = float(params["beta_GP"]) * float(params["k_hyd"]) / 24.0
    model_0_7h_slope = model_release_7h / VALIDATION_END_H
    eta_7h = hydrolysis_fraction_from_rate(float(params["k_hyd"]), VALIDATION_END_H / 24.0)

    metrics = {
        "k_hyd_day^-1": float(params["k_hyd"]),
        "eta_48h": float(params["eta_48h"]),
        "eta_7h": eta_7h,
        "model_initial_slope_mM_h^-1": model_initial_slope,
        "model_mean_slope_0_7h_mM_h^-1": model_0_7h_slope,
        "reported_slope_mM_h^-1": REPORTED_SLOPE_MM_PER_H,
        "model_total_Pi_7h_aligned_mM": model_7h_total_aligned,
        "reported_fit_Pi_7h_mM": reported_7h_total,
        "model_release_7h_mM": model_release_7h,
        "reported_release_7h_mM": reported_release_7h,
        "reported_to_model_7h_release_ratio": reported_release_7h / model_release_7h,
    }

    with (out_dir / "pi_release_validation_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics))
        writer.writeheader()
        writer.writerow({key: f"{value:.10g}" for key, value in metrics.items()})
    return metrics


def plot_validation(
    out_dir: Path,
    params: dict[str, float],
    metrics: dict[str, float],
    figure_formats: tuple[str, ...],
) -> None:
    t_fit = np.linspace(0.0, VALIDATION_END_H, 300)
    model_fit_release = model_release_pi(t_fit, params)
    model_fit_total_aligned = REPORTED_INTERCEPT_MM + model_fit_release
    model_points_release = model_release_pi(TIME_H, params)
    model_points_total_aligned = REPORTED_INTERCEPT_MM + model_points_release

    err_idx = [int(np.where(np.isclose(TIME_H, h))[0][0]) for h in ERRORBAR_TIME_H]
    err_center = BETA_GP_PI_MM[err_idx]
    yerr = np.vstack((err_center - ERRORBAR_LOW_MM, ERRORBAR_HIGH_MM - err_center))

    fig, ax = plt.subplots(figsize=(6.6, 4.2), constrained_layout=True)
    clean_axis(ax)

    ax.plot(
        t_fit,
        reported_linear_pi(t_fit),
        color="#2f2f2f",
        linewidth=1.9,
        label=r"Reported linear fit, $0.73\ \mathrm{mM\,h^{-1}}$",
        zorder=2,
    )
    ax.errorbar(
        ERRORBAR_TIME_H,
        err_center,
        yerr=yerr,
        fmt="none",
        ecolor="#b23a48",
        elinewidth=1.0,
        capsize=3,
        zorder=2,
    )
    ax.scatter(
        TIME_H,
        BETA_GP_PI_MM,
        s=34,
        marker="o",
        color="#b23a48",
        edgecolor="white",
        linewidth=0.6,
        label=r"Digitized $10\ \mathrm{mM}\ \beta$-GP",
        zorder=3,
    )
    ax.scatter(
        TIME_H,
        CONTROL_PI_MM,
        s=28,
        marker="s",
        color="#7b8794",
        edgecolor="white",
        linewidth=0.6,
        label="Digitized control",
        zorder=3,
    )
    ax.plot(
        t_fit,
        model_fit_total_aligned,
        color="#1f6f8b",
        linewidth=2.1,
        label=rf"Model hydrolysis, $\eta_{{48h}}={float(params['eta_48h']):.2f}$",
        zorder=4,
    )
    ax.scatter(
        TIME_H,
        model_points_total_aligned,
        s=25,
        marker="D",
        color="#1f6f8b",
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
        label="_nolegend_",
    )

    ax.set_xlim(-0.15, VALIDATION_END_H + 0.15)
    ax.set_ylim(0.95, max(float(reported_linear_pi(VALIDATION_END_H)), float(np.max(BETA_GP_PI_MM))) + 0.35)
    ax.set_xticks(np.arange(0, VALIDATION_END_H + 1, 1))
    ax.set_xlabel("Time after beta-GP addition (h)")
    ax.set_ylabel("Soluble phosphate, Pi (mM)")
    ax.set_title("Short-time beta-GP phosphate release", loc="left", pad=8)
    ax.legend(loc="upper left", frameon=True, borderpad=0.45, handlelength=2.2)

    ax.text(
        0.98,
        0.93,
        (
            rf"7 h release: model {metrics['model_release_7h_mM']:.2f} mM; "
            rf"reported fit {metrics['reported_release_7h_mM']:.2f} mM"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.2,
        color="#222222",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#d8d8d8",
            "linewidth": 0.6,
            "alpha": 0.92,
        },
    )

    save_figure(fig, out_dir, "pi_release_validation_7h", figure_formats, dpi=350)


def run_validation(figure_formats: tuple[str, ...] = ("pdf", "png")) -> Path:
    configure_matplotlib(PLOT_DEFAULTS)
    out_dir = create_run_directory("validation", "pi_release")

    params = build_base_params()
    synchronize_hydrolysis_parameters(params, RUN_DEFAULTS.dt, RUN_DEFAULTS.media_interval)

    time_h = np.arange(0.0, VALIDATION_END_H + 1.0, 1.0)
    model_released = model_release_pi(time_h, params)
    model_total_aligned = REPORTED_INTERCEPT_MM + model_released
    model_total_native = float(params["P_ext"]) + model_released

    write_validation_csv(out_dir, time_h, model_total_aligned, model_total_native, model_released)
    metrics = write_metrics_csv(out_dir, params, float(model_total_aligned[-1]))
    plot_validation(out_dir, params, metrics, figure_formats)

    print(f"Saved Pi-release validation to {out_dir}")
    print(
        "7 h release: "
        f"model={metrics['model_release_7h_mM']:.3f} mM, "
        f"reported fit={metrics['reported_release_7h_mM']:.3f} mM, "
        f"ratio={metrics['reported_to_model_7h_release_ratio']:.1f}x"
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure-formats",
        default="pdf,png",
        help="Comma-separated figure formats: pdf,png",
    )
    args = parser.parse_args()
    run_validation(parse_figure_formats(args.figure_formats))


if __name__ == "__main__":
    main()
