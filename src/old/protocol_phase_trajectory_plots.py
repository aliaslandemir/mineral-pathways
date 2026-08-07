"""
Archived direct trajectory plots for protocol comparison.

Examples:
  python -m src.old.protocol_phase_trajectory_plots --eta-48h 0.22 --M-max 2.0 --k-m 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import MODEL_DEFAULTS, RUN_DEFAULTS, STOCHASTIC_DEFAULTS, build_base_params
from src.io_utils import OUTPUT_ROOT, parse_figure_formats, save_figure_to_directory
from src.model import beta_gp_hydrolysis_rate_constant, simulate, synchronize_hydrolysis_parameters
MINERAL_POOL_EPS = 1e-12


WORKFLOW_DIR = OUTPUT_ROOT / "protocol_optimization"
INTERVALS_DAYS = (1.0, 2.0, 3.0)
CONCENTRATIONS_MM = tuple(float(v) for v in range(1, 11))
BASELINE_BETA_GP_MM = float(MODEL_DEFAULTS.beta_GP)
BASELINE_INTERVAL_DAYS = float(RUN_DEFAULTS.media_interval)
TARGET_PHI_H = 0.50

ACP_COLOR = "#1f4e79"
HAP_COLOR = "#2a9d8f"
CROSS_COLOR = "#c96b12"
TARGET_COLOR = "#7a3b9d"
GRID_COLOR = "#e6ebef"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.2,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
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


def parse_float_list(text: str) -> list[float]:
    values = [v.strip() for v in text.split(",") if v.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of numbers.")
    try:
        return [float(v) for v in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric list: {text}") from exc


def apply_parameter_overrides(params: dict[str, float | str], args: argparse.Namespace) -> dict[str, float | str]:
    p = dict(params)
    for name in [
        "C_ext",
        "P_ext",
        "beta_GP",
        "eta_48h",
        "alpha",
        "beta",
        "n",
        "K_A",
        "k_f",
        "M_max",
        "k_m",
        "a_AA0",
        "t_AA_half_life",
        "solver_rtol",
        "solver_atol",
        "media_renewal_fraction",
    ]:
        if hasattr(args, name):
            p[name] = float(getattr(args, name))
    if hasattr(args, "r_A_min") and hasattr(args, "r_A_max"):
        p["r_A"] = 0.5 * (float(args.r_A_min) + float(args.r_A_max))
    if hasattr(args, "r_H_min") and hasattr(args, "r_H_max"):
        p["r_H"] = 0.5 * (float(args.r_H_min) + float(args.r_H_max))
    if hasattr(args, "Xi"):
        p["Xi"] = float(args.Xi)
    p.pop("_k_hyd_explicit", None)
    p.pop("_eta_dt_explicit", None)
    if getattr(args, "k_hyd", None) is not None:
        p["k_hyd"] = float(args.k_hyd)
        p["_k_hyd_explicit"] = True
    elif getattr(args, "eta_dt", None) is not None:
        p["eta_dt"] = float(args.eta_dt)
        p["eta_dt_interval_days"] = float(args.dt)
        p["_eta_dt_explicit"] = True
    else:
        p.pop("k_hyd", None)
    p["solver_method"] = str(getattr(args, "solver_method", RUN_DEFAULTS.solver_method))
    synchronize_hydrolysis_parameters(p, float(args.dt), float(args.media_interval))
    return p


def first_h_exceeds_a_time(
    t: np.ndarray,
    a: np.ndarray,
    h: np.ndarray,
    *,
    mineral_threshold: float = MINERAL_POOL_EPS,
) -> float:
    diff = np.asarray(h, dtype=float) - np.asarray(a, dtype=float)
    total = np.asarray(a, dtype=float) + np.asarray(h, dtype=float)
    idx = np.flatnonzero((diff > 0.0) & (total >= float(mineral_threshold)))
    if idx.size == 0:
        return float("nan")
    j = int(idx[0])
    if j == 0:
        return float(t[j])
    t0, t1 = float(t[j - 1]), float(t[j])
    d0, d1 = float(diff[j - 1]), float(diff[j])
    m0, m1 = float(total[j - 1]), float(total[j])
    candidates = [t1]
    if d0 <= 0.0 < d1 and not np.isclose(d0, d1):
        candidates.append(float(t0 + (0.0 - d0) * (t1 - t0) / (d1 - d0)))
    if m0 < float(mineral_threshold) <= m1 and not np.isclose(m0, m1):
        candidates.append(float(t0 + (float(mineral_threshold) - m0) * (t1 - t0) / (m1 - m0)))
    return float(max(candidates))


def first_phase_dominance_time(
    t: np.ndarray,
    a: np.ndarray,
    h: np.ndarray,
    *,
    target_phi_h: float = TARGET_PHI_H,
) -> float:
    total = np.asarray(a, dtype=float) + np.asarray(h, dtype=float)
    phi_h = np.divide(h, total, out=np.full_like(total, np.nan), where=total > MINERAL_POOL_EPS)
    idx = np.flatnonzero(phi_h >= float(target_phi_h))
    return float(t[int(idx[0])]) if idx.size else float("nan")


def simulate_condition(
    base_params: dict[str, float | str],
    *,
    beta_gp: float,
    media_interval: float,
    t_end: float,
    dt: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    params = dict(base_params)
    params["beta_GP"] = float(beta_gp)
    sim = simulate(params, t_end=float(t_end), dt=float(dt), media_interval=float(media_interval))
    total = sim.A + sim.H
    phi_h = np.divide(sim.H, total, out=np.full_like(sim.H, np.nan), where=total > MINERAL_POOL_EPS)
    crossing_time = first_h_exceeds_a_time(sim.t, sim.A, sim.H)
    dominance_time = first_phase_dominance_time(sim.t, sim.A, sim.H)
    rows = pd.DataFrame(
        {
            "time_days": sim.t,
            "beta_GP_mM": float(beta_gp),
            "media_interval_days": float(media_interval),
            "A_ACP_like_model_units": sim.A,
            "H_apatite_like_model_units": sim.H,
            "total_mineral_model_units": total,
            "phi_H": phi_h,
            "S_A": sim.S_A,
            "reset_status": sim.reset_status,
        }
    )
    summary = {
        "beta_GP_mM": float(beta_gp),
        "media_interval_days": float(media_interval),
        "H_exceeds_A_time_days": crossing_time,
        "phase_dominance_time_days": dominance_time,
        "A_final_model_units": float(sim.A[-1]),
        "H_final_model_units": float(sim.H[-1]),
        "phi_H_final": float(phi_h[-1]) if np.isfinite(phi_h[-1]) else float("nan"),
        "max_A_model_units": float(np.max(sim.A)),
        "max_H_model_units": float(np.max(sim.H)),
    }
    return rows, summary


def _format_crossing(value: float) -> str:
    return "no H>A" if not np.isfinite(value) else f"H>A: day {value:.1f}"


def _format_dominance(value: float) -> str:
    return "H>A: not reached" if not np.isfinite(value) else f"H>A: day {value:.1f}"


def _style_axis(ax: plt.Axes, x_max: float, y_max: float) -> None:
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_media_interval_comparison(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    beta_gp: float,
    t_end: float,
    output_dir: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    subset = frame[np.isclose(frame["beta_GP_mM"], float(beta_gp))]
    y_max = 1.08 * float(np.nanmax(subset[["A_ACP_like_model_units", "H_apatite_like_model_units"]].to_numpy()))
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0), sharex=True, sharey=True)
    for idx, (ax, interval) in enumerate(zip(axes, INTERVALS_DAYS)):
        data = subset[np.isclose(subset["media_interval_days"], interval)]
        meta = summary[
            np.isclose(summary["beta_GP_mM"], float(beta_gp))
            & np.isclose(summary["media_interval_days"], float(interval))
        ].iloc[0]
        ax.plot(data["time_days"], data["A_ACP_like_model_units"], color=ACP_COLOR, linewidth=2.0, label="ACP-like precursor")
        ax.plot(data["time_days"], data["H_apatite_like_model_units"], color=HAP_COLOR, linewidth=2.0, label="Apatite-like mineral")
        dominance = float(meta["phase_dominance_time_days"])
        if np.isfinite(dominance):
            h_target = float(np.interp(dominance, data["time_days"], data["H_apatite_like_model_units"]))
            ax.axvline(dominance, color=TARGET_COLOR, linewidth=1.0, linestyle="--", alpha=0.85)
            ax.scatter([dominance], [h_target], s=34, color=TARGET_COLOR, edgecolor="white", linewidth=0.7, zorder=5)
        ax.text(
            0.05,
            0.93,
            _format_dominance(dominance),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#d2d2d2", "alpha": 0.86},
        )
        title_unit = "day" if np.isclose(interval, 1.0) else "days"
        ax.set_title(f"Every {interval:.0f} {title_unit}", loc="center", pad=8)
        ax.text(-0.02, 1.04, chr(ord("A") + idx), transform=ax.transAxes, fontsize=12, fontweight="bold")
        _style_axis(ax, float(t_end), y_max)
        ax.set_xlabel("Time (days)")
        if idx == 0:
            ax.set_ylabel("Mineral pool (model units)")
    handles = [
        plt.Line2D([0], [0], color=ACP_COLOR, linewidth=2.0, label="ACP-like precursor"),
        plt.Line2D([0], [0], color=HAP_COLOR, linewidth=2.0, label="Apatite-like mineral"),
        plt.Line2D([0], [0], color=TARGET_COLOR, linewidth=1.0, linestyle="--", label=r"First $H>A$"),
    ]
    fig.suptitle(f"Mineral phase amounts at {beta_gp:g} mM beta-GP", y=0.995, fontsize=11.0)
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.925), ncol=4, handlelength=2.0, columnspacing=1.2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    return save_figure_to_directory(fig, Path(output_dir) / "figures", "protocol_media_interval_phase_trajectories", formats, dpi=600)


def plot_beta_gp_comparison(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    media_interval: float,
    t_end: float,
    output_dir: Path,
    formats: tuple[str, ...],
    concentrations: tuple[float, ...] = CONCENTRATIONS_MM,
    stem: str = "protocol_beta_gp_phase_trajectories",
    title_suffix: str = "",
    shared_y: bool = True,
    y_limit: float | None = None,
) -> tuple[Path, ...]:
    subset = frame[np.isclose(frame["media_interval_days"], float(media_interval))]
    subset = subset[subset["beta_GP_mM"].isin(concentrations)]
    y_max_global = 1.08 * float(np.nanmax(subset[["A_ACP_like_model_units", "H_apatite_like_model_units"]].to_numpy()))
    ncols = 5 if len(concentrations) >= 10 else 4
    nrows = int(np.ceil(len(concentrations) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11.0, 1.82 * nrows + 0.35), sharex=True, sharey=shared_y)
    axes_flat = axes.ravel()
    for ax in axes_flat[len(concentrations) :]:
        ax.set_visible(False)
    for ax, concentration in zip(axes_flat, concentrations):
        data = subset[np.isclose(subset["beta_GP_mM"], float(concentration))]
        meta = summary[
            np.isclose(summary["beta_GP_mM"], float(concentration))
            & np.isclose(summary["media_interval_days"], float(media_interval))
        ].iloc[0]
        ax.plot(data["time_days"], data["A_ACP_like_model_units"], color=ACP_COLOR, linewidth=1.55)
        ax.plot(data["time_days"], data["H_apatite_like_model_units"], color=HAP_COLOR, linewidth=1.55)
        dominance = float(meta["phase_dominance_time_days"])
        if np.isfinite(dominance):
            h_target = float(np.interp(dominance, data["time_days"], data["H_apatite_like_model_units"]))
            ax.axvline(dominance, color=TARGET_COLOR, linewidth=0.8, linestyle="--", alpha=0.78)
            ax.scatter([dominance], [h_target], s=20, color=TARGET_COLOR, edgecolor="white", linewidth=0.45, zorder=5)
        label = f"{concentration:.0f} mM"
        if np.isfinite(dominance):
            label += f"\nH>A {dominance:.1f} d"
        else:
            label += "\nno H>A"
        ax.text(
            0.05,
            0.92,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
        local_max = 1.08 * float(np.nanmax(data[["A_ACP_like_model_units", "H_apatite_like_model_units"]].to_numpy()))
        panel_y_max = float(y_limit) if y_limit is not None else (y_max_global if shared_y else local_max)
        _style_axis(ax, float(t_end), panel_y_max)
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (days)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Mineral pool\n(model units)")
    handles = [
        plt.Line2D([0], [0], color=ACP_COLOR, linewidth=2.0, label="ACP-like precursor"),
        plt.Line2D([0], [0], color=HAP_COLOR, linewidth=2.0, label="Apatite-like mineral"),
        plt.Line2D([0], [0], color=TARGET_COLOR, linewidth=1.0, linestyle="--", label=r"First $H>A$"),
    ]
    interval_unit = "day" if np.isclose(media_interval, 1.0) else "days"
    fig.suptitle(f"Mineral phase amounts across beta-GP concentrations{title_suffix}, renewal every {media_interval:g} {interval_unit}", y=0.995, fontsize=11.0)
    fig.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=4, handlelength=2.0)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.925))
    return save_figure_to_directory(fig, Path(output_dir) / "figures", stem, formats, dpi=600)


def plot_target_summary(
    summary: pd.DataFrame,
    *,
    media_interval: float,
    output_dir: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    panel = summary[np.isclose(summary["media_interval_days"], float(media_interval))].copy()
    panel = panel[panel["beta_GP_mM"].isin(CONCENTRATIONS_MM)].sort_values("beta_GP_mM")
    x = panel["beta_GP_mM"].to_numpy(dtype=float)
    dominance = panel["phase_dominance_time_days"].to_numpy(dtype=float)
    h_final = panel["H_final_model_units"].to_numpy(dtype=float)
    a_final = panel["A_final_model_units"].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15))
    ax_time, ax_final = axes
    reached = np.isfinite(dominance)
    ax_time.plot(x[reached], dominance[reached], color=TARGET_COLOR, marker="o", markersize=4.2, linewidth=1.7)
    if np.any(~reached):
        ax_time.scatter(x[~reached], np.full(np.count_nonzero(~reached), 28.0), marker="x", s=34, color="#686868", linewidth=1.2, label="Not reached by day 28")
    ax_time.set_xlabel(r"$\beta$-GP concentration (mM)")
    ax_time.set_ylabel(r"First $H>A$ day")
    ax_time.set_xticks(x)
    ax_time.set_ylim(7.0, 28.8)
    ax_time.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax_time.legend(frameon=False, loc="upper right", fontsize=7.5)

    ax_final.plot(x, h_final, color=HAP_COLOR, marker="o", markersize=4.0, linewidth=1.7, label="Apatite-like mineral")
    ax_final.plot(x, a_final, color=ACP_COLOR, marker="o", markersize=4.0, linewidth=1.5, label="ACP-like precursor")
    ax_final.set_xlabel(r"$\beta$-GP concentration (mM)")
    ax_final.set_ylabel("Day-28 mineral pool\n(model units)")
    ax_final.set_xticks(x)
    ax_final.set_ylim(0.0, max(1.0, 1.08 * float(np.nanmax([h_final.max(), a_final.max()]))))
    ax_final.grid(True, color=GRID_COLOR, linewidth=0.6)
    ax_final.legend(frameon=False, loc="upper left", fontsize=7.5)

    interval_unit = "day" if np.isclose(media_interval, 1.0) else "days"
    fig.suptitle(f"Phase-dominance timing and day-28 mineral pools, renewal every {media_interval:g} {interval_unit}", y=0.99, fontsize=10.5)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return save_figure_to_directory(fig, Path(output_dir) / "figures", "protocol_beta_gp_target_summary_1to10", formats, dpi=600)


def plot_apatite_overlay(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    media_interval: float,
    t_end: float,
    output_dir: Path,
    formats: tuple[str, ...],
) -> tuple[Path, ...]:
    panel = frame[np.isclose(frame["media_interval_days"], float(media_interval))]
    panel = panel[panel["beta_GP_mM"].isin(CONCENTRATIONS_MM)]
    meta = summary[np.isclose(summary["media_interval_days"], float(media_interval))]
    meta = meta[meta["beta_GP_mM"].isin(CONCENTRATIONS_MM)]

    cmap = plt.get_cmap("cividis")
    norm = plt.Normalize(vmin=min(CONCENTRATIONS_MM), vmax=max(CONCENTRATIONS_MM))
    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    for concentration in CONCENTRATIONS_MM:
        data = panel[np.isclose(panel["beta_GP_mM"], concentration)]
        color = cmap(norm(concentration))
        ax.plot(data["time_days"], data["H_apatite_like_model_units"], color=color, linewidth=1.65)
        row = meta[np.isclose(meta["beta_GP_mM"], concentration)].iloc[0]
        dominance = float(row["phase_dominance_time_days"])
        if np.isfinite(dominance):
            h_target = float(np.interp(dominance, data["time_days"], data["H_apatite_like_model_units"]))
            ax.scatter([dominance], [h_target], s=22, color=color, edgecolor="white", linewidth=0.55, zorder=5)
    ax.set_xlim(0.0, float(t_end))
    ax.set_ylim(0.0, 1.08 * float(panel["H_apatite_like_model_units"].max()))
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Apatite-like mineral\n(model units)")
    ax.grid(True, color=GRID_COLOR, linewidth=0.6)
    interval_unit = "day" if np.isclose(media_interval, 1.0) else "days"
    ax.set_title(f"Apatite trajectories for 1-10 mM beta-GP, renewal every {media_interval:g} {interval_unit}", pad=7)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.018, fraction=0.05)
    cbar.set_label(r"$\beta$-GP (mM)")
    cbar.set_ticks(CONCENTRATIONS_MM)
    fig.tight_layout()
    return save_figure_to_directory(fig, Path(output_dir) / "figures", "protocol_beta_gp_apatite_overlay_1to10", formats, dpi=600)


def write_parameter_manifest(output_dir: Path, params: dict[str, float | str], args: argparse.Namespace) -> Path:
    manifest = {
        "t_end": float(args.t_end),
        "dt": float(args.dt),
        "baseline_beta_GP_mM": float(args.beta_GP),
        "baseline_media_interval_days": float(args.media_interval),
        "media_renewal_fraction": float(params["media_renewal_fraction"]),
        "eta_48h": float(params["eta_48h"]),
        "k_hyd": beta_gp_hydrolysis_rate_constant(params),
        "K_A": float(params["K_A"]),
        "k_f": float(params["k_f"]),
        "M_max": float(params["M_max"]),
        "k_m": float(params["k_m"]),
        "a_AA0": float(params["a_AA0"]),
        "t_AA_half_life": float(params["t_AA_half_life"]),
        "C_ext": float(params["C_ext"]),
        "P_ext": float(params["P_ext"]),
        "intervals_days": ",".join(f"{v:g}" for v in INTERVALS_DAYS),
        "concentrations_mM": ",".join(f"{v:g}" for v in CONCENTRATIONS_MM),
        "H_exceeds_A_requires_total_mineral_model_units": MINERAL_POOL_EPS,
        "target_phi_H": TARGET_PHI_H,
        "mineral_pool_units_note": "A and H are uncalibrated model pools; no absolute A/H amount threshold is used.",
    }
    path = Path(output_dir) / "protocol_phase_trajectory_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["parameter", "value"])
        writer.writeheader()
        for key, value in manifest.items():
            writer.writerow({"parameter": key, "value": value})
    print("Resolved trajectory-plot parameters:")
    for key, value in manifest.items():
        print(f"- {key}: {value}")
    return path


def export_protocol_phase_trajectory_plots(
    output_dir: Path,
    params: dict[str, float | str],
    *,
    args: argparse.Namespace,
    formats: tuple[str, ...],
) -> None:
    configure_matplotlib()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[pd.DataFrame] = []
    summaries: list[dict[str, float]] = []

    print("Running full model trajectories...")
    for interval in INTERVALS_DAYS:
        for concentration in CONCENTRATIONS_MM:
            rows, summary = simulate_condition(
                params,
                beta_gp=concentration,
                media_interval=interval,
                t_end=float(args.t_end),
                dt=float(args.dt),
            )
            all_rows.append(rows)
            summaries.append(summary)
    frame = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summaries).sort_values(["media_interval_days", "beta_GP_mM"], ignore_index=True)

    data_dir = output_dir / "protocol_optimization"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / "protocol_phase_trajectories.csv", index=False)
    summary.to_csv(data_dir / "protocol_phase_crossing_summary.csv", index=False)
    write_parameter_manifest(data_dir, params, args)

    interval_paths = plot_media_interval_comparison(
        frame,
        summary,
        beta_gp=float(args.beta_GP),
        t_end=float(args.t_end),
        output_dir=output_dir,
        formats=formats,
    )
    beta_paths = plot_beta_gp_comparison(
        frame,
        summary,
        media_interval=float(args.media_interval),
        t_end=float(args.t_end),
        output_dir=output_dir,
        formats=formats,
    )
    target_summary_paths = plot_target_summary(
        summary,
        media_interval=float(args.media_interval),
        output_dir=output_dir,
        formats=formats,
    )
    overlay_paths = plot_apatite_overlay(
        frame,
        summary,
        media_interval=float(args.media_interval),
        t_end=float(args.t_end),
        output_dir=output_dir,
        formats=formats,
    )
    compatibility_paths = plot_beta_gp_comparison(
        frame,
        summary,
        media_interval=float(args.media_interval),
        t_end=float(args.t_end),
        output_dir=output_dir,
        formats=formats,
        concentrations=CONCENTRATIONS_MM,
        stem="protocol_beta_gp_phase_trajectories_low_dose_zoom",
        title_suffix=" (1-10 mM)",
        shared_y=True,
    )

    print("\nFirst phase-dominance time (H > A; no absolute mineral-amount cutoff):")
    selected = summary[
        np.isclose(summary["beta_GP_mM"], float(args.beta_GP))
        | np.isclose(summary["media_interval_days"], float(args.media_interval))
    ].sort_values(["media_interval_days", "beta_GP_mM"])
    for row in selected.itertuples(index=False):
        dominance = row.phase_dominance_time_days
        dominance_text = "not reached" if not np.isfinite(dominance) else f"day {dominance:.2f}"
        print(f"- beta-GP {row.beta_GP_mM:g} mM, renewal every {row.media_interval_days:g} d: {dominance_text}")
    print("\nSaved figures:")
    for path in (*interval_paths, *beta_paths, *target_summary_paths, *overlay_paths, *compatibility_paths):
        print(f"- {path}")


def build_parser() -> argparse.ArgumentParser:
    defaults = build_base_params()
    parser = argparse.ArgumentParser(
        description="Plot direct ACP-like and apatite-like mineral trajectories for protocol comparisons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=WORKFLOW_DIR)
    parser.add_argument("--t-end", "--t_end", dest="t_end", type=float, default=RUN_DEFAULTS.t_end)
    parser.add_argument("--dt", type=float, default=RUN_DEFAULTS.dt)
    parser.add_argument("--media-interval", "--media_interval", dest="media_interval", type=float, default=RUN_DEFAULTS.media_interval)
    parser.add_argument("--media-renewal-fraction", "--media_renewal_fraction", dest="media_renewal_fraction", type=float, default=defaults["media_renewal_fraction"])
    parser.add_argument("--C-ext", "--C_ext", dest="C_ext", type=float, default=defaults["C_ext"])
    parser.add_argument("--P-ext", "--P_ext", dest="P_ext", type=float, default=defaults["P_ext"])
    parser.add_argument("--beta-GP", "--beta_GP", dest="beta_GP", type=float, default=defaults["beta_GP"])
    parser.add_argument("--eta-48h", "--eta_48h", dest="eta_48h", type=float, default=defaults["eta_48h"])
    hydrolysis = parser.add_mutually_exclusive_group()
    hydrolysis.add_argument("--eta-dt", "--eta_dt", dest="eta_dt", type=float, default=None)
    hydrolysis.add_argument("--k-hyd", "--k_hyd", dest="k_hyd", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=defaults["alpha"])
    parser.add_argument("--beta", type=float, default=defaults["beta"])
    parser.add_argument("--n", type=float, default=defaults["n"])
    parser.add_argument("--K-A", "--K_A", dest="K_A", type=float, default=defaults["K_A"])
    parser.add_argument("--k-f", "--k_f", dest="k_f", type=float, default=defaults["k_f"])
    parser.add_argument("--M-max", "--M_max", dest="M_max", type=float, default=defaults["M_max"])
    parser.add_argument("--k-m", "--k_m", dest="k_m", type=float, default=defaults["k_m"])
    parser.add_argument("--a-AA0", "--a_AA0", dest="a_AA0", type=float, default=defaults["a_AA0"])
    parser.add_argument("--t-AA-half-life", "--t_AA_half_life", dest="t_AA_half_life", type=float, default=defaults["t_AA_half_life"])
    parser.add_argument("--r-A-min", "--r_A_min", dest="r_A_min", type=float, default=STOCHASTIC_DEFAULTS.r_A_min)
    parser.add_argument("--r-A-max", "--r_A_max", dest="r_A_max", type=float, default=STOCHASTIC_DEFAULTS.r_A_max)
    parser.add_argument("--r-H-min", "--r_H_min", dest="r_H_min", type=float, default=STOCHASTIC_DEFAULTS.r_H_min)
    parser.add_argument("--r-H-max", "--r_H_max", dest="r_H_max", type=float, default=STOCHASTIC_DEFAULTS.r_H_max)
    parser.add_argument("--Xi", type=float, default=STOCHASTIC_DEFAULTS.Xi)
    parser.add_argument("--solver-method", "--solver_method", dest="solver_method", type=str, default=RUN_DEFAULTS.solver_method)
    parser.add_argument("--solver-rtol", "--solver_rtol", dest="solver_rtol", type=float, default=RUN_DEFAULTS.solver_rtol)
    parser.add_argument("--solver-atol", "--solver_atol", dest="solver_atol", type=float, default=RUN_DEFAULTS.solver_atol)
    parser.add_argument("--figure-formats", default="pdf,png")
    return parser


def main() -> None:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring main.py-only arguments:", " ".join(unknown))
    params = apply_parameter_overrides(build_base_params(), args)
    formats = parse_figure_formats(args.figure_formats)
    export_protocol_phase_trajectory_plots(args.output_dir, params, args=args, formats=formats)


if __name__ == "__main__":
    main()
