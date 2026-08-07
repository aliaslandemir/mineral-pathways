from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import save_figure


def configure_matplotlib(plot: Any) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": plot.figure_dpi,
            "savefig.dpi": plot.save_dpi,
            "font.family": "DejaVu Sans",
            "font.size": plot.font_size,
            "axes.titlesize": plot.axes_title_size,
            "axes.labelsize": plot.axes_label_size,
            "legend.fontsize": plot.legend_font_size,
            "xtick.labelsize": plot.tick_label_size,
            "ytick.labelsize": plot.tick_label_size,
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


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.grid(axis=grid_axis, color="#e6e6e6", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")


def _save(fig: plt.Figure, out_dir: Path, stem: str, plot: Any) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="This figure includes Axes that are not compatible with tight_layout")
        fig.tight_layout(pad=0.8)
    save_figure(fig, out_dir, stem, plot.figure_formats, dpi=plot.save_dpi)


def _mean_std(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=float)
    return np.nanmean(values, axis=0), np.nanstd(values, axis=0)


def _phase_lines(ax: plt.Axes, plot: Any) -> None:
    for day in plot.phase_lines_days:
        ax.axvline(day, color="#cfcfcf", linewidth=0.55, linestyle=":", zorder=0)


def _set_axis_text_size(ax: plt.Axes, *, label_size: float, tick_size: float, title_size: float | None = None) -> None:
    ax.xaxis.label.set_size(label_size)
    ax.yaxis.label.set_size(label_size)
    ax.tick_params(axis="both", labelsize=tick_size)
    if title_size is not None:
        ax.title.set_size(title_size)


def _mean_band(
    ax: plt.Axes,
    time: np.ndarray,
    runs: np.ndarray,
    *,
    color: str,
    label: str,
    draw_errorbars: bool = False,
    errorbar_count: int = 9,
    band_alpha: float = 0.22,
    interval: str = "sd",
    line_width: float = 1.8,
    band_label: str | None = None,
) -> None:
    mean, std = _mean_std(runs)
    if interval == "95ci":
        spread = 1.96 * std
    elif interval == "sd":
        spread = std
    else:
        raise ValueError(f"Unsupported interval: {interval}")
    ax.fill_between(
        time,
        mean - spread,
        mean + spread,
        color=color,
        alpha=band_alpha,
        linewidth=0,
        label=band_label,
    )
    ax.plot(time, mean, color=color, linewidth=line_width, label=label)
    if draw_errorbars:
        sample_count = max(2, min(int(errorbar_count), len(time)))
        idx = np.unique(np.linspace(0, len(time) - 1, sample_count, dtype=int))
        ax.errorbar(
            time[idx],
            mean[idx],
            yerr=spread[idx],
            fmt="none",
            ecolor=color,
            elinewidth=0.8,
            capsize=2.2,
            alpha=0.9,
            zorder=4,
        )


def _relative_endpoint_stats(
    runs: np.ndarray,
    baseline_final_runs: np.ndarray,
    eps: float = 1e-12,
) -> tuple[float, float]:
    final = np.asarray(runs, dtype=float)[:, -1]
    baseline = np.asarray(baseline_final_runs, dtype=float)
    if final.shape != baseline.shape:
        baseline = np.full_like(final, float(np.nanmean(baseline)))
    relative = (final + eps) / (baseline + eps)
    return float(np.nanmean(relative)), float(np.nanstd(relative))


def _endpoint_panel(
    ax: plt.Axes,
    levels: list[float],
    runs_by_level: dict[float, np.ndarray],
    baseline_final_runs: np.ndarray,
    *,
    title: str,
    xlabel: str,
    baseline_level: float,
    y_limits: tuple[float, float],
    xticks: list[float] | None = None,
) -> None:
    x = np.asarray(levels, dtype=float)
    stats = np.asarray(
        [_relative_endpoint_stats(runs_by_level[level], baseline_final_runs) for level in levels],
        dtype=float,
    )
    mean, std = stats[:, 0], stats[:, 1]
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
    ax.plot(x, mean, color="#2a9d8f", linewidth=1.4)
    ax.scatter(x, mean, s=28, color="#2a9d8f", zorder=3)
    ax.errorbar(x, mean, yerr=std, fmt="none", ecolor="#202020", elinewidth=0.8, capsize=2.2)
    baseline_idx = int(np.argmin(np.abs(x - float(baseline_level))))
    ax.scatter(
        [x[baseline_idx]],
        [mean[baseline_idx]],
        s=52,
        facecolor="white",
        edgecolor="#202020",
        linewidth=1.1,
        zorder=4,
    )
    clean_axis(ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Relative endpoint apatite")
    ax.set_ylim(*y_limits)
    shown_ticks = np.asarray(xticks if xticks is not None else x, dtype=float)
    ax.set_xticks(shown_ticks)
    ax.set_xticklabels([f"{value:g}" for value in shown_ticks], rotation=35, ha="right", fontsize=7)


def generate_stochastic_figure_set(
    *,
    out_dir: Path,
    t_days: np.ndarray,
    ions_C: np.ndarray,
    ions_P: np.ndarray,
    mineral_A: np.ndarray,
    mineral_H: np.ndarray,
    rates_Rf: np.ndarray,
    rates_Rm: np.ndarray,
    t_AA_half_life_levels: list[float],
    t_AA_half_life_h_runs: dict[float, np.ndarray],
    t_AA_half_life_rm_runs: dict[float, np.ndarray],
    beta_gp_levels: list[float],
    beta_gp_h_runs: dict[float, np.ndarray],
    eta_48h_levels: list[float],
    eta_48h_h_runs: dict[float, np.ndarray],
    K_A_levels: list[float],
    K_A_h_runs: dict[float, np.ndarray],
    k_f_levels: list[float],
    k_f_h_runs: dict[float, np.ndarray],
    M_max_levels: list[float],
    M_max_h_runs: dict[float, np.ndarray],
    k_m_levels: list[float],
    k_m_h_runs: dict[float, np.ndarray],
    baseline_h_final_runs: np.ndarray,
    baseline_beta_gp: float,
    baseline_eta_48h: float,
    baseline_K_A: float,
    baseline_k_f: float,
    baseline_M_max: float,
    baseline_k_m: float,
    plot_config: Any,
    saturation_A: np.ndarray,
    ions_P_reference: np.ndarray | None = None,
) -> None:
    plot = plot_config
    x_max = float(np.nanmax(t_days))
    blue = "#1f4e79"
    teal = "#2a9d8f"

    fig = plt.figure(figsize=(7.7, 6.7))
    grid = fig.add_gridspec(3, 1, height_ratios=(1.65, 0.82, 1.0), hspace=0.22)
    ax_ions = fig.add_subplot(grid[0])
    ax_removed = fig.add_subplot(grid[1], sharex=ax_ions)
    ax_saturation = fig.add_subplot(grid[2], sharex=ax_ions)
    ax_phosphate = ax_ions.twinx()
    _mean_band(ax_ions, t_days, ions_C, color=blue, label="Calcium")
    _mean_band(ax_phosphate, t_days, ions_P, color=teal, label="Phosphate")
    p_mean, _ = _mean_std(ions_P)
    if ions_P_reference is not None:
        p_reference, _ = _mean_std(ions_P_reference)
        ax_phosphate.plot(t_days, p_reference, color=teal, linewidth=1.1, linestyle="--", label="Phosphate input only")
        p_removed = np.maximum(0.0, p_reference - p_mean)
    else:
        p_removed = np.zeros_like(p_mean)
    ax_removed.plot(t_days, p_removed, color=teal, linewidth=1.7)
    _mean_band(ax_saturation, t_days, saturation_A, color="#4d4d4d", label="Formation index")
    ax_saturation.axhline(1.0, color="#9b2226", linewidth=1.0, linestyle=":", label="Threshold")
    for ax in (ax_ions, ax_removed, ax_saturation):
        _phase_lines(ax, plot)
        clean_axis(ax)
        ax.set_xlim(0.0, x_max)
    ax_ions.set_ylabel("Calcium (mM)", color=blue, labelpad=5)
    ax_phosphate.set_ylabel("Phosphate (mM)", color=teal, labelpad=5)
    ax_removed.set_ylabel("P removed\n(mM)")
    ax_saturation.set_ylabel("Formation index")
    ax_saturation.set_xlabel("Time (days)")
    ax_ions.set_title("Soluble ion dynamics under medium renewal", pad=20)
    for axis in (ax_ions, ax_phosphate, ax_removed, ax_saturation):
        _set_axis_text_size(axis, label_size=8.8, tick_size=8.0, title_size=9.6)
    lines_left, labels_left = ax_ions.get_legend_handles_labels()
    lines_right, labels_right = ax_phosphate.get_legend_handles_labels()
    ax_ions.legend(
        lines_left + lines_right,
        labels_left + labels_right,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=7.8,
        handlelength=2.4,
        columnspacing=1.4,
    )
    ax_saturation.legend(frameon=False, ncol=2, loc="upper left", fontsize=7.8)
    _save(fig, out_dir, "main_fig02_soluble_ion_dynamics", plot)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    _mean_band(
        ax,
        t_days,
        mineral_A,
        color=blue,
        label="ACP-like precursor",
        band_alpha=0.24,
        interval="sd",
        line_width=1.35,
    )
    _mean_band(
        ax,
        t_days,
        mineral_H,
        color=teal,
        label="Apatite-like mineral",
        band_alpha=0.24,
        interval="sd",
        line_width=1.35,
    )
    _phase_lines(ax, plot)
    clean_axis(ax)
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mineral pool (mM phosphate equivalents)")
    ax.set_title("ACP-like precursor and apatite-like mineral dynamics")
    ax.legend(frameon=False, loc="upper left", handlelength=2.6)
    _save(fig, out_dir, "main_fig03_acp_apatite_dynamics", plot)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    _mean_band(ax, t_days, rates_Rf, color=blue, label="ACP formation rate")
    _mean_band(ax, t_days, rates_Rm, color=teal, label="Apatite maturation rate")
    _phase_lines(ax, plot)
    clean_axis(ax)
    ax.set_xlim(0.0, x_max)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Rate (model units day$^{-1}$)")
    ax.set_title("Precursor formation and maturation rates")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, out_dir, "supp_fig02_formation_maturation_rates", plot)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(t_AA_half_life_levels)))
    for level, color in zip(t_AA_half_life_levels, colors):
        rm_mean, _ = _mean_std(t_AA_half_life_rm_runs[level])
        h_mean, _ = _mean_std(t_AA_half_life_h_runs[level])
        label = rf"$t_{{1/2,\mathrm{{Asc}}}}={level:g}$ h"
        axes[0].plot(t_days, rm_mean, color=color, linewidth=1.5, label=label)
        axes[1].plot(t_days, h_mean, color=color, linewidth=1.5, label=label)
    for ax in axes:
        _phase_lines(ax, plot)
        clean_axis(ax)
        ax.set_xlim(0.0, x_max)
        ax.set_xlabel("Time (days)")
    axes[0].set_ylabel("Maturation rate (model units day$^{-1}$)")
    axes[0].set_title("AA2P half-life and maturation")
    axes[1].set_ylabel("Apatite H (model units)")
    axes[1].set_title("Apatite accumulation")
    axes[1].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    _save(fig, out_dir, "supp_fig03_ascorbate_matrix_support_sweep", plot)

    endpoint_specs = (
        (r"$\beta$-GP", r"$\beta$-GP (mM)", beta_gp_levels, beta_gp_h_runs, baseline_beta_gp, [0.0, 5.0, 10.0, 15.0, 20.0]),
        (r"$\eta_{48h}$", r"$\eta_{48h}$ (dimensionless)", eta_48h_levels, eta_48h_h_runs, baseline_eta_48h, None),
        (r"$K_A$", r"$K_A$ (mM$^5$)", K_A_levels, K_A_h_runs, baseline_K_A, None),
        (r"$k_f$", "$k_f$\n(mM P-eq day$^{-1}$)", k_f_levels, k_f_h_runs, baseline_k_f, None),
        (r"$M_{\max}$", "$M_{\\max}$\n(mM phosphate equivalents)", M_max_levels, M_max_h_runs, baseline_M_max, None),
        (r"$k_m$", r"$k_m$ (day$^{-1}$)", k_m_levels, k_m_h_runs, baseline_k_m, None),
    )
    endpoint_upper = []
    for _, _, levels, h_runs, _, _ in endpoint_specs:
        stats = np.asarray([_relative_endpoint_stats(h_runs[level], baseline_h_final_runs) for level in levels], dtype=float)
        endpoint_upper.extend((stats[:, 0] + stats[:, 1]).tolist())
    shared_y_max = max(1.4, float(np.nanmax(endpoint_upper)) * 1.08)
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.2), sharey=True)
    for idx, (ax, (title, xlabel, levels, h_runs, baseline_level, xticks)) in enumerate(zip(axes.ravel(), endpoint_specs)):
        _endpoint_panel(
            ax,
            levels,
            h_runs,
            baseline_h_final_runs,
            title=title,
            xlabel=xlabel,
            baseline_level=baseline_level,
            y_limits=(0.0, shared_y_max),
            xticks=xticks,
        )
        if idx % 3 != 0:
            ax.set_ylabel("")
    fig.suptitle("Endpoint responses relative to baseline", y=1.01)
    fig.text(
        0.5,
        0.01,
        "Open circles mark baseline settings; error bars show +/-1 SD from prescribed Ca/P-ratio sampling.",
        ha="center",
        va="bottom",
        fontsize=8.2,
        color="#333333",
    )
    _save(fig, out_dir, "main_fig06_endpoint_parameter_sweep", plot)
