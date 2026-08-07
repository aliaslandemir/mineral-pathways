"""Archived full figure suite retained for reproducibility."""

from pathlib import Path
from typing import Any, Dict, List, Tuple
import os
import warnings
import csv

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
import matplotlib.pyplot as plt
import numpy as np

from .io_utils import figure_format_dir, figure_path, save_figure


FIGURE_EXPORT_DIR = "figures"
PDF_DIR = "pdf"
PNG_DIR = "png"

FIGURE_MANIFEST: dict[str, tuple[str, str, str]] = {
    "fig1_ions_mean_std": ("main", "main_fig02_soluble_ion_dynamics", "Soluble ions and effective precursor-formation index"),
    "fig2_minerals_mean_std": ("main", "main_fig03_acp_apatite_dynamics", "ACP-like precursor and apatite-like mineral dynamics"),
    "fig4_aa_presence_apatite": ("main", "main_fig04_ascorbate_presence_apatite", "Effect of ascorbate matrix support on apatite-like mineral accumulation"),
    "fig11d_sweep_endpoint_lollipop": ("main", "main_fig06_endpoint_parameter_sweep", "Combined endpoint parameter-sweep responses"),
    "regime_map": ("main", "main_fig07_model_derived_regime_map", "Model-derived mineralization regimes at days 10 and 28"),
    "main_fig_predictive_experimental_map": ("main", "main_fig_predictive_experimental_map", "Predictive experimental guidance for dose, sampling, and endpoint selection"),
    "fig2b_mineral_ca_equivalents_mean_std": ("supp", "supp_fig01_mineral_calcium_equivalents", "Calcium equivalents in mineral phases"),
    "fig3_rates_mean_std": ("supp", "supp_fig02_formation_maturation_rates", "ACP formation and apatite maturation rates"),
    "fig5_aa_sensitivity_mean_std": ("supp", "supp_fig03_ascorbate_matrix_support_sweep", "Ascorbate matrix-support sensitivity"),
    "fig6_beta_gp_sensitivity_mean_std": ("supp", "supp_fig04_beta_gp_trajectory_sweep", "Beta-GP concentration trajectory sensitivity"),
    "fig7_eta_48h_sensitivity_mean_std": ("supp", "supp_fig05_beta_gp_hydrolysis_trajectory_sweep", "Beta-GP hydrolysis trajectory sensitivity"),
    "fig8_K_A_sensitivity_mean_std": ("supp", "supp_fig06_precursor_formation_index_sweep", "Effective precursor-formation index sweep"),
    "fig9_k_f_sensitivity_mean_std": ("supp", "supp_fig08_acp_formation_rate_sweep", "ACP formation-rate sweep"),
    "fig10_M_max_sensitivity_mean_std": ("supp", "supp_fig09_mineral_capacity_sweep", "Retained mineral capacity sweep"),
    "fig10_k_m_sensitivity_mean_std": ("supp", "supp_fig10_apatite_maturation_rate_sweep", "Apatite maturation-rate sweep"),
    "fig12_totals_mean_std": ("supp", "supp_fig11_mineral_bound_element_uptake", "Mineral-bound element uptake"),
    "phosphate_accounting_schaeck_inspired": ("supp", "supp_fig12_phosphate_accounting_schaeck_inspired", "Phosphate accounting"),
}


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
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.55,
            "legend.framealpha": 1.0,
            "legend.edgecolor": "#d9d9d9",
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(axis=grid_axis, color="#e6e6e6", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _save(fig: plt.Figure, out_dir: Path, name: str, plot: Any) -> None:
    if not fig.get_constrained_layout():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout(pad=0.7)
    save_figure(fig, out_dir, name, plot.figure_formats, dpi=plot.save_dpi)


def organize_article_figures(run_dir: Path) -> None:
    rows: list[dict[str, str]] = []
    for suffix in (PDF_DIR, PNG_DIR):
        format_dir = figure_format_dir(run_dir, suffix)
        if not format_dir.exists():
            continue
        for src in sorted(format_dir.glob(f"*.{suffix}")):
            category, article_stem, title = FIGURE_MANIFEST.get(
                src.stem,
                ("supp", f"supp_extra_{src.stem}", "Unclassified supplementary output"),
            )
            dest = format_dir / f"{article_stem}.{suffix}"
            if src.name != dest.name:
                src.replace(dest)
            rows.append(
                {
                    "category": category,
                    "source_stem": src.stem,
                    "export_stem": article_stem,
                    "format": suffix,
                    "title": title,
                }
            )

    manifest_path = run_dir / FIGURE_EXPORT_DIR / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["category", "source_stem", "export_stem", "format", "title"])
        writer.writeheader()
        writer.writerows(rows)


def _phase_lines(ax: plt.Axes, plot: Any) -> None:
    for x in plot.phase_lines_days:
        ax.axvline(x, color="#c7ced6", linestyle="--", linewidth=0.65, alpha=0.45, zorder=0)


def _clean_axis(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.grid(axis=grid_axis, color="#e6e6e6", linewidth=0.6, alpha=0.8)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)


def _clean_twin_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_color("#333333")
    ax.spines["right"].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.8)


def _set_panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(title, loc="left", pad=6, fontweight="normal")


def _legend_above(ax: plt.Axes, ncol: int = 2, x: float = 1.0) -> None:
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(x, 1.02),
        ncol=ncol,
        borderaxespad=0.0,
        columnspacing=1.4,
        handlelength=2.0,
    )


def _mean_std(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def _level_colors(levels: List[float], cmap_name: str = "viridis") -> List[tuple[float, float, float, float]]:
    if len(levels) <= 1:
        return [plt.get_cmap(cmap_name)(0.55)]
    cmap = plt.get_cmap(cmap_name)
    return [cmap(x) for x in np.linspace(0.12, 0.86, len(levels))]


def _compact_label(name: str, value: float, units: str = "") -> str:
    pretty_names = {
        "a_AA0": r"$a_{\mathrm{AA},0}$",
        "t_AA_half_life": r"$t_{1/2,\mathrm{Asc}}$",
        "beta-GP": "β-GP",
        "eta": "η48h",
        "eta_48h": "η48h",
        "K_A": r"$K_A$",
        "k_f": r"$k_f$",
        "M_max": r"$M_{\max}$",
        "k_m": r"$k_m$",
    }
    pretty_name = pretty_names.get(name, name)
    text = f"{pretty_name} = {value:g}"
    if units:
        text += f" {units}"
    return text


def _format_log2fc(value: float) -> str:
    if abs(value) >= 10.0:
        return "<-10" if value < 0 else ">10"
    if abs(value) >= 1.0:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _log2fc_stats(runs: np.ndarray, baseline_final_runs: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    final = np.asarray(runs, dtype=float)[:, -1]
    baseline = np.asarray(baseline_final_runs, dtype=float)
    if final.shape == baseline.shape:
        denom = baseline
    else:
        denom = np.full_like(final, float(np.nanmean(baseline)))
    vals = np.log2((final + eps) / (denom + eps))
    return float(np.nanmean(vals)), float(np.nanstd(vals))


def _bar_width(levels: List[float]) -> float:
    x = np.asarray(levels, dtype=float)
    if x.size < 2:
        return 0.5
    diffs = np.diff(np.sort(np.unique(x)))
    if diffs.size == 0:
        return 0.5
    return float(max(np.nanmin(diffs) * 0.55, 1e-4))


def _plot_log2fc_endpoint(
    ax: plt.Axes,
    levels: List[float],
    h_runs: Dict[float, np.ndarray],
    baseline_h_final_runs: np.ndarray,
    color: str,
    floor: float = -10.0,
    title: str = "Endpoint response",
    xlabel: str = "Sweep level",
    ylabel: str = "Log2 fold change in day-28 apatite",
    annotate: bool = False,
) -> None:
    x = np.asarray(levels, dtype=float)
    mu_sd = np.array([_log2fc_stats(h_runs[level], baseline_h_final_runs) for level in levels], dtype=float)
    mu = mu_sd[:, 0]
    sd = mu_sd[:, 1]
    shown_mu = np.maximum(mu, floor)
    shown_low = np.maximum(mu - sd, floor)
    shown_high = np.maximum(mu + sd, floor)
    yerr = np.vstack([shown_mu - shown_low, shown_high - shown_mu])
    ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":")
    ax.bar(x, shown_mu, width=_bar_width(levels), color=color, alpha=0.78, edgecolor="none")
    ax.errorbar(x, shown_mu, yerr=yerr, fmt="o", color="#202020", markersize=2.8, elinewidth=0.8, capsize=2.2)
    clipped = mu < floor
    if np.any(clipped):
        yrange = max(float(np.nanmax(shown_mu)) - floor, 1.0)
        for xi in x[clipped]:
            ax.text(xi, floor + 0.035 * yrange, "<-10", ha="center", va="bottom", fontsize=7.2, rotation=90, color="#333333")
    if annotate:
        yrange = max(float(np.nanmax(shown_high)) - float(np.nanmin(shown_low)), 1.0)
        for idx, (xi, yi) in enumerate(zip(x, shown_mu)):
            dy = 0.03 * yrange
            if abs(yi) < 0.15:
                dy *= 1.6
                va = "bottom" if idx % 2 == 0 else "top"
                y_text = yi + dy if va == "bottom" else yi - dy
            else:
                va = "bottom" if yi >= 0 else "top"
                y_text = yi + dy if yi >= 0 else yi - dy
            ax.text(xi, y_text, _format_log2fc(float(yi)), ha="center", va=va, fontsize=6.8, color="#1f1f1f")
    _clean_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_delta_endpoint(
    ax: plt.Axes,
    levels: List[float],
    h_runs: Dict[float, np.ndarray],
    baseline_h_final_runs: np.ndarray,
    color: str,
    title: str = "Endpoint response",
    xlabel: str = "Sweep level",
    ylabel: str = "Change in day-28 apatite vs baseline",
    annotate: bool = False,
) -> None:
    x = np.asarray(levels, dtype=float)
    mu_sd = np.array(
        [
            (
                float(np.nanmean(np.asarray(h_runs[level], dtype=float)[:, -1] - np.asarray(baseline_h_final_runs, dtype=float))),
                float(np.nanstd(np.asarray(h_runs[level], dtype=float)[:, -1] - np.asarray(baseline_h_final_runs, dtype=float))),
            )
            for level in levels
        ],
        dtype=float,
    )
    mu = mu_sd[:, 0]
    sd = mu_sd[:, 1]
    ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle=":")
    ax.bar(x, mu, width=_bar_width(levels), color=color, alpha=0.78, edgecolor="none")
    ax.errorbar(x, mu, yerr=sd, fmt="o", color="#202020", markersize=2.8, elinewidth=0.8, capsize=2.2)
    if annotate:
        yrange = max(float(np.nanmax(mu + sd)) - float(np.nanmin(mu - sd)), 1.0)
        for idx, (xi, yi) in enumerate(zip(x, mu)):
            dy = 0.03 * yrange
            if abs(yi) < 0.0005:
                dy *= 1.6
                va = "bottom" if idx % 2 == 0 else "top"
                y_text = yi + dy if va == "bottom" else yi - dy
            else:
                va = "bottom" if yi >= 0 else "top"
                y_text = yi + dy if yi >= 0 else yi - dy
            ax.text(xi, y_text, f"{yi:.4f}", ha="center", va=va, fontsize=6.8, color="#1f1f1f")
    _clean_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_absolute_endpoint(
    ax: plt.Axes,
    levels: List[float],
    h_runs: Dict[float, np.ndarray],
    color: str,
    title: str = "Endpoint response",
    xlabel: str = "Sweep level",
    ylabel: str = "Day-28 apatite H (model units)",
    annotate: bool = True,
) -> None:
    x = np.asarray(levels, dtype=float)
    mu = np.array([float(np.nanmean(np.asarray(h_runs[level], dtype=float)[:, -1])) for level in levels], dtype=float)
    sd = np.array([float(np.nanstd(np.asarray(h_runs[level], dtype=float)[:, -1])) for level in levels], dtype=float)
    ax.bar(x, mu, width=_bar_width(levels), color=color, alpha=0.82, edgecolor="none")
    ax.errorbar(x, mu, yerr=sd, fmt="o", color="#202020", markersize=2.8, elinewidth=0.8, capsize=2.2)
    if annotate:
        yrange = max(float(np.nanmax(mu + sd)) - float(np.nanmin(mu - sd)), 1e-12)
        for idx, (xi, yi) in enumerate(zip(x, mu)):
            dy = 0.03 * yrange
            if yi > 0:
                y_text = yi + dy
                va = "bottom"
            else:
                y_text = yi - dy
                va = "top"
            ax.text(xi, y_text, f"{yi:.4f}", ha="center", va=va, fontsize=6.8, color="#1f1f1f")
    ax.axhline(float(np.nanmean(mu)), color="#666666", linewidth=0.8, linestyle="--", alpha=0.5, zorder=0)
    _clean_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _relative_endpoint_stats(runs: np.ndarray, baseline_final_runs: np.ndarray, eps: float = 1e-12) -> Tuple[float, float]:
    final = np.asarray(runs, dtype=float)[:, -1]
    baseline = np.asarray(baseline_final_runs, dtype=float)
    if final.shape != baseline.shape:
        baseline = np.full_like(final, float(np.nanmean(baseline)))
    rel = (final + eps) / (baseline + eps)
    return float(np.nanmean(rel)), float(np.nanstd(rel))


def _plot_relative_endpoint(
    ax: plt.Axes,
    levels: List[float],
    h_runs: Dict[float, np.ndarray],
    baseline_h_final_runs: np.ndarray,
    color: str,
    title: str = "Endpoint response",
    xlabel: str = "Sweep level",
    ylabel: str = "Relative day-28 apatite (baseline = 1)",
    annotate: bool = True,
    bar_alpha: float = 0.82,
) -> None:
    x = np.asarray(levels, dtype=float)
    stats = np.array([_relative_endpoint_stats(h_runs[level], baseline_h_final_runs) for level in levels], dtype=float)
    mu = stats[:, 0]
    sd = stats[:, 1]
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
    ax.bar(x, mu, width=_bar_width(levels), color=color, alpha=bar_alpha, edgecolor="none")
    ax.errorbar(x, mu, yerr=sd, fmt="o", color="#202020", markersize=2.8, elinewidth=0.8, capsize=2.2)
    if annotate:
        yrange = max(float(np.nanmax(mu + sd)) - float(np.nanmin(mu - sd)), 1e-12)
        for idx, (xi, yi) in enumerate(zip(x, mu)):
            dy = 0.025 * yrange
            if abs(yi - 1.0) < 0.02:
                dy *= 1.7
                va = "bottom" if idx % 2 == 0 else "top"
                y_text = yi + dy if va == "bottom" else yi - dy
            else:
                va = "bottom" if yi >= 1.0 else "top"
                y_text = yi + dy if yi >= 1.0 else yi - dy
            ax.text(xi, y_text, f"{yi:.3f}", ha="center", va=va, fontsize=6.8, color="#1f1f1f")
    _clean_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _plot_relative_endpoint_lollipop(
    ax: plt.Axes,
    levels: List[float],
    h_runs: Dict[float, np.ndarray],
    baseline_h_final_runs: np.ndarray,
    color: str,
    title: str,
    xlabel: str = "Sweep level",
    ylabel: str = "Relative day-28 apatite (baseline = 1)",
    mark_baseline: bool = True,
) -> None:
    x = np.asarray(levels, dtype=float)
    stats = np.array([_relative_endpoint_stats(h_runs[level], baseline_h_final_runs) for level in levels], dtype=float)
    mu = stats[:, 0]
    sd = stats[:, 1]
    ax.axhline(1.0, color="#333333", linewidth=0.8, linestyle=":")
    ax.plot(x, mu, color=color, linewidth=1.4, alpha=0.9)
    ax.scatter(x, mu, s=28, color=color, zorder=3)
    ax.errorbar(x, mu, yerr=sd, fmt="none", ecolor="#202020", elinewidth=0.8, capsize=2.2, alpha=0.9)
    if mark_baseline and len(x):
        baseline_idx = int(np.nanargmin(np.abs(mu - 1.0)))
        ax.scatter(
            [x[baseline_idx]],
            [mu[baseline_idx]],
            s=70,
            facecolors="none",
            edgecolors="#202020",
            linewidths=1.1,
            zorder=4,
        )
    _clean_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def _set_sweep_xticks(ax: plt.Axes, levels: List[float], fontsize: float = 6.7) -> None:
    ax.set_xticks(np.asarray(levels, dtype=float))
    ax.set_xticklabels([f"{level:g}" for level in levels], fontsize=fontsize, rotation=35, ha="right", rotation_mode="anchor")


def _plot_mean_std(
    ax: plt.Axes,
    x: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    color: str,
    label: str,
    band_label: str | None = None,
    draw_errorbars: bool = False,
    errorbar_sample_divisor: int = 8,
    band_alpha: float = 0.28,
    line_width: float = 1.9,
) -> None:
    ax.fill_between(x, mu - sd, mu + sd, color=color, alpha=band_alpha, linewidth=0, label=band_label, zorder=1)
    ax.plot(x, mu, color=color, lw=line_width, label=label, zorder=2)
    if draw_errorbars:
        step = max(1, len(x) // errorbar_sample_divisor)
        xi = x[::step]
        mui = mu[::step]
        sdi = sd[::step]
        ax.errorbar(
            xi,
            mui,
            yerr=sdi,
            fmt="none",
            ecolor=color,
            elinewidth=1.0,
            capsize=2.5,
            alpha=0.75,
        )


def generate_stochastic_figure_set(
    out_dir: Path,
    t_days: np.ndarray,
    ions_C: np.ndarray,
    ions_P: np.ndarray,
    mineral_A: np.ndarray,
    mineral_H: np.ndarray,
    total_P: np.ndarray,
    total_Ca: np.ndarray,
    rates_Rf: np.ndarray,
    rates_Rm: np.ndarray,
    t_AA_half_life_levels: List[float],
    t_AA_half_life_h_runs: Dict[float, np.ndarray],
    t_AA_half_life_rm_runs: Dict[float, np.ndarray],
    beta_gp_levels: List[float],
    beta_gp_p_runs: Dict[float, np.ndarray],
    beta_gp_a_runs: Dict[float, np.ndarray],
    beta_gp_h_runs: Dict[float, np.ndarray],
    beta_gp_rf_runs: Dict[float, np.ndarray],
    eta_48h_levels: List[float],
    eta_48h_p_runs: Dict[float, np.ndarray],
    eta_48h_h_runs: Dict[float, np.ndarray],
    eta_48h_rf_runs: Dict[float, np.ndarray],
    K_A_levels: List[float],
    K_A_s_runs: Dict[float, np.ndarray],
    K_A_h_runs: Dict[float, np.ndarray],
    K_A_rf_runs: Dict[float, np.ndarray],
    k_f_levels: List[float],
    k_f_a_runs: Dict[float, np.ndarray],
    k_f_h_runs: Dict[float, np.ndarray],
    k_f_rf_runs: Dict[float, np.ndarray],
    M_max_levels: List[float],
    M_max_a_runs: Dict[float, np.ndarray],
    M_max_h_runs: Dict[float, np.ndarray],
    M_max_rf_runs: Dict[float, np.ndarray],
    k_m_levels: List[float],
    k_m_a_runs: Dict[float, np.ndarray],
    k_m_h_runs: Dict[float, np.ndarray],
    k_m_rm_runs: Dict[float, np.ndarray],
    baseline_h_final_runs: np.ndarray,
    plot_config: Any,
    saturation_A: np.ndarray | None = None,
    ions_C_reference: np.ndarray | None = None,
    ions_P_reference: np.ndarray | None = None,
    mineral_A_ca: np.ndarray | None = None,
    mineral_H_ca: np.ndarray | None = None,
) -> None:
    del ions_C_reference
    plot = plot_config

    # restrained, publication-style palette
    c_main = "#1f4e79"
    c_alt = "#2a9d8f"
    c_ref = "#333333"
    x_max = float(np.nanmax(t_days))

    # 1) Ions: mean ± std
    # Use separate y-axes so the calcium variation is visible alongside
    # phosphate without splitting the figure into multiple panels.
    ax_head = None
    if saturation_A is None and ions_P_reference is None:
        fig, ax = plt.subplots(figsize=(7.6, 4.5), constrained_layout=True)
        ax_s = None
        ax_pu = None
    elif saturation_A is None:
        fig, (ax, ax_pu) = plt.subplots(
            2,
            1,
            figsize=(7.6, 5.35),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.0, 0.9]},
        )
        ax_s = None
    elif ions_P_reference is not None:
        fig = plt.figure(figsize=(7.7, 7.4), constrained_layout=True)
        gs = fig.add_gridspec(4, 1, height_ratios=[0.34, 2.0, 0.82, 1.05])
        ax_head = fig.add_subplot(gs[0])
        ax = fig.add_subplot(gs[1])
        ax_pu = fig.add_subplot(gs[2], sharex=ax)
        ax_s = fig.add_subplot(gs[3], sharex=ax)
        ax_head.axis("off")
    else:
        fig, (ax, ax_s) = plt.subplots(
            2,
            1,
            figsize=(7.6, 5.95),
            sharex=True,
            constrained_layout=True,
            gridspec_kw={"height_ratios": [2.0, 1.0]},
        )
        ax_pu = None
    c_mu, c_sd = _mean_std(ions_C)
    p_mu, p_sd = _mean_std(ions_P)
    ax_p = ax.twinx()

    _plot_mean_std(ax, t_days, c_mu, c_sd, c_main, "Calcium", band_alpha=0.18, line_width=1.65)
    _plot_mean_std(ax_p, t_days, p_mu, p_sd, c_alt, "Phosphate", band_alpha=0.18, line_width=1.65)
    if ions_P_reference is not None:
        p_ref_mu, p_ref_sd = _mean_std(ions_P_reference)
        ax_p.plot(t_days, p_ref_mu, color=c_alt, lw=1.25, linestyle="--", alpha=0.75, label="Phosphate input only")
    _phase_lines(ax, plot)
    _clean_axis(ax)
    _clean_twin_axis(ax_p)
    ax.set_xlim(0, x_max)
    if ax_s is None and ax_pu is None:
        ax.set_xlabel("Time (days)")
    ax.set_ylabel("Calcium (mM)", color=c_main)
    ax_p.set_ylabel("Phosphate (mM)", color=c_alt)
    c_lo = float(np.nanmin(c_mu - c_sd))
    c_hi = float(np.nanmax(c_mu + c_sd))
    c_span = max(c_hi - c_lo, 1e-4)
    ax.set_ylim(c_lo - 0.25 * c_span, c_hi + 0.25 * c_span)
    p_ylim_parts = [p_mu - p_sd, p_mu + p_sd]
    if ions_P_reference is not None:
        p_ylim_parts.extend([p_ref_mu - p_ref_sd, p_ref_mu + p_ref_sd])
    p_lo = float(np.nanmin(np.concatenate(p_ylim_parts)))
    p_hi = float(np.nanmax(np.concatenate(p_ylim_parts)))
    p_span = max(p_hi - p_lo, 1e-4)
    ax_p.set_ylim(max(0.0, p_lo - 0.08 * p_span), p_hi + 0.08 * p_span)
    if ax_head is not None:
        ax_head.text(
            0.5,
            0.82,
            "Soluble ion dynamics under medium replenishment",
            ha="center",
            va="center",
            fontsize=plot.axes_title_size,
        )
    else:
        ax.set_title("Soluble ion dynamics under medium replenishment", loc="center", pad=7, fontweight="normal")
    ax.tick_params(axis="y", colors=c_main)
    ax_p.tick_params(axis="y", colors=c_alt)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_p.get_legend_handles_labels()
    legend_parent = ax_head if ax_head is not None else ax
    legend_parent.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        frameon=False,
        loc="lower center" if ax_head is not None else "lower center",
        bbox_to_anchor=(0.5, 0.05 if ax_head is not None else 1.02),
        ncol=3,
        borderaxespad=0.0,
        columnspacing=1.4,
        handlelength=2.0,
    )
    if ax_pu is not None and ions_P_reference is not None:
        p_ref_mu, _ = _mean_std(ions_P_reference)
        p_removed = np.maximum(0.0, p_ref_mu - p_mu)
        ax_pu.plot(t_days, p_removed, color=c_alt, lw=1.8)
        _phase_lines(ax_pu, plot)
        _clean_axis(ax_pu)
        ax_pu.set_xlim(0, x_max)
        ax_pu.set_ylim(0.0, max(1e-3, float(np.nanmax(p_removed)) * 1.18))
        ax_pu.set_ylabel("P removed (mM)", color=c_alt)
        ax_pu.tick_params(axis="y", colors=c_alt)
        _set_panel_title(ax_pu, "Phosphate")
        if ax_s is None:
            ax_pu.set_xlabel("Time (days)")
    if ax_s is not None:
        s_mu, s_sd = _mean_std(saturation_A)
        _plot_mean_std(ax_s, t_days, s_mu, s_sd, "#4d4d4d", "Formation index")
        ax_s.axhline(1.0, color="#9b2226", linewidth=1.0, linestyle=":", label="Formation threshold")
        ax_s.fill_between(
            t_days,
            1.0,
            s_mu,
            where=s_mu >= 1.0,
            color="#9b2226",
            alpha=0.08,
            linewidth=0,
            interpolate=True,
        )
        _phase_lines(ax_s, plot)
        _clean_axis(ax_s)
        ax_s.set_xlim(0, x_max)
        ax_s.set_ylim(0.0, max(1.2, float(np.nanmax(s_mu + s_sd)) * 1.12))
        ax_s.set_ylabel("Effective formation index")
        _set_panel_title(ax_s, "Effective precursor-formation index")
        ax_s.legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.02, 0.98),
            ncol=2,
            borderaxespad=0.0,
            columnspacing=1.1,
            handlelength=2.0,
        )
        ax_s.set_xlabel("Time (days)")
    _save(fig, out_dir, "fig1_ions_mean_std", plot)

    # 2) Minerals: mean ± std
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    a_mu, a_sd = _mean_std(mineral_A)
    h_mu, h_sd = _mean_std(mineral_H)
    _plot_mean_std(ax, t_days, a_mu, a_sd, c_main, "ACP-like precursor", "ACP-like mean ± SD", draw_errorbars=True, errorbar_sample_divisor=5, band_alpha=0.58, line_width=1.15)
    _plot_mean_std(ax, t_days, h_mu, h_sd, c_alt, "Apatite-like mineral", "Apatite-like mean ± SD", draw_errorbars=True, errorbar_sample_divisor=5, band_alpha=0.58, line_width=1.15)
    _phase_lines(ax, plot)
    _clean_axis(ax)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mineral pool (model units)")
    ax.set_title("Mineral phase amounts", loc="center", pad=8, fontweight="normal")
    ax.legend(frameon=False, loc="upper left", fontsize=7.5, ncol=2, handlelength=1.8, columnspacing=0.9)
    _save(fig, out_dir, "fig2_minerals_mean_std", plot)

    if mineral_A_ca is not None and mineral_H_ca is not None:
        fig, ax = plt.subplots(figsize=(7.0, 4.1))
        a_ca_mu, a_ca_sd = _mean_std(mineral_A_ca)
        h_ca_mu, h_ca_sd = _mean_std(mineral_H_ca)
        _plot_mean_std(ax, t_days, a_ca_mu, a_ca_sd, c_main, "ACP-bound calcium", "ACP Ca mean ± SD from sampled effective Ca/P ratios")
        _plot_mean_std(ax, t_days, h_ca_mu, h_ca_sd, c_alt, "Apatite-bound calcium", "Apatite Ca mean ± SD from sampled effective Ca/P ratios")
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Calcium-equivalent concentration (mM)")
        _set_panel_title(ax, "Calcium equivalents in mineral phases")
        _legend_above(ax, ncol=2)
        _save(fig, out_dir, "fig2b_mineral_ca_equivalents_mean_std", plot)

    # 3) Rates: mean ± std
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    rf_mu, rf_sd = _mean_std(rates_Rf)
    rm_mu, rm_sd = _mean_std(rates_Rm)
    _plot_mean_std(ax, t_days, rf_mu, rf_sd, c_main, "ACP formation rate", "Formation mean ± SD from sampled effective Ca/P ratios", draw_errorbars=True, errorbar_sample_divisor=6)
    _plot_mean_std(ax, t_days, rm_mu, rm_sd, c_alt, "Apatite maturation rate", "Maturation mean ± SD from sampled effective Ca/P ratios", draw_errorbars=True, errorbar_sample_divisor=6)
    _clean_axis(ax)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Mineral-pool rate (model units day$^{-1}$)")
    _set_panel_title(ax, "Formation and maturation rates")
    ax.legend(frameon=False, loc="upper left", fontsize=7.7, ncol=2, handlelength=1.8, columnspacing=0.9)
    _save(fig, out_dir, "fig3_rates_mean_std", plot)

    # 5) AA2P/ascorbate half-life sensitivity.
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1))
    aa_colors = _level_colors(t_AA_half_life_levels, "viridis")
    for i, t_half in enumerate(t_AA_half_life_levels):
        h_mu, _ = _mean_std(t_AA_half_life_h_runs[t_half])
        rm_mu, _ = _mean_std(t_AA_half_life_rm_runs[t_half])
        color = aa_colors[i]
        label = _compact_label("t_AA_half_life", t_half, "h")
        axes[0].plot(t_days, rm_mu, color=color, lw=1.55, label=label)
        axes[1].plot(t_days, h_mu, color=color, lw=1.55, label=label)
    for ax in axes:
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
    axes[0].set_ylabel("Apatite maturation rate (model units day$^{-1}$)")
    axes[0].set_title("AA2P half-life and maturation")
    axes[1].set_ylabel("Apatite H (model units)")
    axes[1].set_title("Half-life effect on mineral accumulation")
    axes[1].legend(frameon=False, fontsize=7.0, ncol=2, loc="upper left", handlelength=1.6, columnspacing=0.8)
    _save(fig, out_dir, "fig5_aa_sensitivity_mean_std", plot)

    # 5b) Main manuscript comparison: fast decay vs 24 h baseline.
    baseline_t_half = min(t_AA_half_life_levels, key=lambda level: abs(float(level) - 24.0))
    fast_t_half = min(t_AA_half_life_levels)
    comparison_specs = []
    if fast_t_half in t_AA_half_life_h_runs and fast_t_half != baseline_t_half:
        comparison_specs.append((fast_t_half, f"Fast AA2P decay ({fast_t_half:g} h)", "#8f8f8f"))
    if baseline_t_half in t_AA_half_life_h_runs:
        comparison_specs.append((baseline_t_half, r"Baseline AA2P half-life ($24$ h)", c_alt))
    if comparison_specs:
        fig, ax = plt.subplots(figsize=(6.8, 4.15), constrained_layout=True)
        for level, label, color in comparison_specs:
            h_mu, h_sd = _mean_std(t_AA_half_life_h_runs[level])
            ax.fill_between(t_days, h_mu - h_sd, h_mu + h_sd, color=color, alpha=0.16, linewidth=0)
            ax.plot(t_days, h_mu, color=color, lw=1.9, label=label)
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Apatite-like mineral H (model units)")
        ax.set_title("Ascorbate support enables apatite-like mineral accumulation", loc="center", pad=8, fontweight="normal")
        ax.legend(frameon=True, loc="upper left", borderpad=0.45, handlelength=2.0)
        _save(fig, out_dir, "fig4_aa_presence_apatite", plot)

    # 6) beta-GP sensitivity: phosphate supply and downstream mineral formation.
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9))
    colors = _level_colors(beta_gp_levels, "viridis")
    for i, level in enumerate(beta_gp_levels):
        p_mu, _ = _mean_std(beta_gp_p_runs[level])
        rf_mu, _ = _mean_std(beta_gp_rf_runs[level])
        h_mu, _ = _mean_std(beta_gp_h_runs[level])
        color = colors[i]
        label = _compact_label("beta-GP", level, "mM")
        axes[0].plot(t_days, p_mu, color=color, lw=1.55, label=label)
        axes[1].plot(t_days, rf_mu, color=color, lw=1.55, label=label)
        axes[2].plot(t_days, h_mu, color=color, lw=1.55, label=label)
    for ax in axes[:3]:
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
    _plot_log2fc_endpoint(axes[3], beta_gp_levels, beta_gp_h_runs, baseline_h_final_runs, c_alt)
    axes[3].set_xlabel("β-GP concentration (mM)")
    axes[0].set_ylabel("Phosphate (mM)")
    axes[0].set_title("Soluble phosphate")
    axes[1].set_ylabel("ACP formation rate (model units day$^{-1}$)")
    axes[1].set_title("ACP formation")
    axes[2].set_ylabel("Apatite H (model units)")
    axes[2].set_title("Apatite accumulation")
    axes[2].legend(frameon=False, fontsize=6.8, ncol=1, loc="upper left", handlelength=1.5)
    _save(fig, out_dir, "fig6_beta_gp_sensitivity_mean_std", plot)

    # 7) beta-GP first-order hydrolysis sweep.
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9))
    colors = _level_colors(eta_48h_levels, "viridis")
    for i, level in enumerate(eta_48h_levels):
        p_mu, _ = _mean_std(eta_48h_p_runs[level])
        rf_mu, _ = _mean_std(eta_48h_rf_runs[level])
        h_mu, _ = _mean_std(eta_48h_h_runs[level])
        color = colors[i]
        label = _compact_label("eta", level)
        axes[0].plot(t_days, p_mu, color=color, lw=1.45, label=label)
        axes[1].plot(t_days, rf_mu, color=color, lw=1.45, label=label)
        axes[2].plot(t_days, h_mu, color=color, lw=1.45, label=label)
    for ax in axes[:3]:
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
    _plot_log2fc_endpoint(axes[3], eta_48h_levels, eta_48h_h_runs, baseline_h_final_runs, c_alt)
    axes[3].set_xlabel("48 h β-GP hydrolysis fraction")
    axes[0].set_ylabel("Phosphate (mM)")
    axes[0].set_title("β-GP hydrolysis")
    axes[1].set_ylabel("ACP formation rate (model units day$^{-1}$)")
    axes[1].set_title("ACP formation")
    axes[2].set_ylabel("Apatite H (model units)")
    axes[2].set_title("Apatite accumulation")
    axes[2].legend(frameon=False, fontsize=6.4, ncol=2, loc="upper left", handlelength=1.4, columnspacing=0.6)
    _save(fig, out_dir, "fig7_eta_48h_sensitivity_mean_std", plot)

    # 8) Supersaturation scale sweep.
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9))
    colors = _level_colors(K_A_levels, "viridis")
    for i, level in enumerate(K_A_levels):
        s_mu, _ = _mean_std(K_A_s_runs[level])
        rf_mu, _ = _mean_std(K_A_rf_runs[level])
        h_mu, _ = _mean_std(K_A_h_runs[level])
        color = colors[i]
        label = _compact_label("K_A", level)
        axes[0].plot(t_days, s_mu, color=color, lw=1.55, label=label)
        axes[1].plot(t_days, rf_mu, color=color, lw=1.55, label=label)
        axes[2].plot(t_days, h_mu, color=color, lw=1.55, label=label)
    axes[0].axhline(1.0, color="#9b2226", linewidth=1.0, linestyle=":", label="threshold")
    for ax in axes[:3]:
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
    _plot_log2fc_endpoint(axes[3], K_A_levels, K_A_h_runs, baseline_h_final_runs, c_alt)
    axes[3].set_xlabel(r"$K_A$")
    axes[0].set_ylabel("Effective formation index")
    axes[0].set_title("Precursor-formation index")
    axes[1].set_ylabel("ACP formation rate (model units day$^{-1}$)")
    axes[1].set_title("ACP formation")
    axes[2].set_ylabel("Apatite H (model units)")
    axes[2].set_title("Apatite accumulation")
    axes[2].legend(frameon=False, fontsize=6.8, ncol=1, loc="upper left", handlelength=1.5)
    _save(fig, out_dir, "fig8_K_A_sensitivity_mean_std", plot)

    # 9) Precursor formation scale sweep.
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9))
    colors = _level_colors(k_f_levels, "viridis")
    for i, level in enumerate(k_f_levels):
        rf_mu, _ = _mean_std(k_f_rf_runs[level])
        a_mu, _ = _mean_std(k_f_a_runs[level])
        h_mu, _ = _mean_std(k_f_h_runs[level])
        color = colors[i]
        label = _compact_label("k_f", level)
        axes[0].plot(t_days, rf_mu, color=color, lw=1.55, label=label)
        axes[1].plot(t_days, a_mu, color=color, lw=1.55, label=label)
        axes[2].plot(t_days, h_mu, color=color, lw=1.55, label=label)
    for ax in axes[:3]:
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
    _plot_log2fc_endpoint(axes[3], k_f_levels, k_f_h_runs, baseline_h_final_runs, c_alt)
    axes[3].set_xlabel(r"$k_f$")
    axes[0].set_ylabel("ACP formation rate (model units day$^{-1}$)")
    axes[0].set_title("ACP formation constant")
    axes[1].set_ylabel("ACP A (model units)")
    axes[1].set_title("ACP pool")
    axes[2].set_ylabel("Apatite H (model units)")
    axes[2].set_title("Apatite accumulation")
    axes[2].legend(frameon=False, fontsize=6.8, ncol=1, loc="upper left", handlelength=1.5)
    _save(fig, out_dir, "fig9_k_f_sensitivity_mean_std", plot)

    # 10) Retained mineral capacity sweep.
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9))
    colors = _level_colors(M_max_levels, "viridis")
    for i, level in enumerate(M_max_levels):
        rf_mu, _ = _mean_std(M_max_rf_runs[level])
        a_mu, _ = _mean_std(M_max_a_runs[level])
        h_mu, _ = _mean_std(M_max_h_runs[level])
        color = colors[i]
        label = _compact_label("M_max", level)
        axes[0].plot(t_days, rf_mu, color=color, lw=1.55, label=label)
        axes[1].plot(t_days, a_mu, color=color, lw=1.55, label=label)
        axes[2].plot(t_days, h_mu, color=color, lw=1.55, label=label)
    for ax in axes[:3]:
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
    _plot_log2fc_endpoint(axes[3], M_max_levels, M_max_h_runs, baseline_h_final_runs, c_alt)
    axes[3].set_xlabel(r"$M_{\max}$")
    axes[0].set_ylabel("ACP formation rate (model units day$^{-1}$)")
    axes[0].set_title("Capacity-limited ACP formation")
    axes[1].set_ylabel("ACP A (model units)")
    axes[1].set_title("ACP pool")
    axes[2].set_ylabel("Apatite H (model units)")
    axes[2].set_title("Apatite accumulation")
    axes[2].legend(frameon=False, fontsize=6.8, ncol=1, loc="upper left", handlelength=1.5)
    _save(fig, out_dir, "fig10_M_max_sensitivity_mean_std", plot)

    # 11) Maturation rate scale sweep.
    fig, axes = plt.subplots(1, 4, figsize=(14.2, 3.9))
    colors = _level_colors(k_m_levels, "viridis")
    for i, level in enumerate(k_m_levels):
        rm_mu, _ = _mean_std(k_m_rm_runs[level])
        a_mu, _ = _mean_std(k_m_a_runs[level])
        h_mu, _ = _mean_std(k_m_h_runs[level])
        color = colors[i]
        label = _compact_label("k_m", level)
        axes[0].plot(t_days, rm_mu, color=color, lw=1.55, label=label)
        axes[1].plot(t_days, a_mu, color=color, lw=1.55, label=label)
        axes[2].plot(t_days, h_mu, color=color, lw=1.55, label=label)
    for ax in axes[:3]:
        _phase_lines(ax, plot)
        _clean_axis(ax)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Time (days)")
    _plot_log2fc_endpoint(axes[3], k_m_levels, k_m_h_runs, baseline_h_final_runs, c_alt)
    axes[3].set_xlabel(r"$k_m$")
    axes[0].set_ylabel("Apatite maturation rate (model units day$^{-1}$)")
    axes[0].set_title("Apatite maturation constant")
    axes[1].set_ylabel("ACP A (model units)")
    axes[1].set_title("ACP pool")
    axes[2].set_ylabel("Apatite H (model units)")
    axes[2].set_title("Apatite accumulation")
    axes[2].legend(frameon=False, fontsize=6.8, ncol=1, loc="upper left", handlelength=1.5)
    _save(fig, out_dir, "fig10_k_m_sensitivity_mean_std", plot)

    # 11b) Single endpoint summary used as the manuscript sweep figure.
    summary_specs = [
        ("β-GP", "β-GP concentration (mM)", beta_gp_levels, beta_gp_h_runs),
        (r"$\eta_{48h}$", r"$\eta_{48h}$", eta_48h_levels, eta_48h_h_runs),
        (r"$K_A$", r"$K_A$", K_A_levels, K_A_h_runs),
        (r"$k_f$", r"$k_f$", k_f_levels, k_f_h_runs),
        (r"$M_{\max}$", r"$M_{\max}$", M_max_levels, M_max_h_runs),
        (r"$k_m$", r"$k_m$ (day$^{-1}$)", k_m_levels, k_m_h_runs),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.2), sharex=False)
    for ax, (title, xlabel, levels, h_runs) in zip(axes.ravel(), summary_specs):
        _plot_relative_endpoint_lollipop(
            ax,
            levels,
            h_runs,
            baseline_h_final_runs,
            c_alt,
            title=title,
            xlabel=xlabel,
            ylabel="Relative day-28 apatite (baseline = 1)",
        )
        _set_sweep_xticks(ax, levels, fontsize=6.7)
    for ax in axes.ravel()[len(summary_specs):]:
        ax.axis("off")
    fig.suptitle("Sweep endpoint responses relative to baseline", y=0.995)
    _save(fig, out_dir, "fig11d_sweep_endpoint_lollipop", plot)

    # 11) Mineral-bound element uptake. Soluble pools are strongly affected by
    # medium exchange and beta-GP release, so compare mineral-bound phosphate
    # with mineral-bound calcium.
    fig, ax = plt.subplots(figsize=(7.0, 4.1))
    ax_ca = ax.twinx()
    mineral_bound_p = total_P - ions_P
    p_bound_mu, p_bound_sd = _mean_std(mineral_bound_p)
    mineral_bound_ca = total_Ca - ions_C
    ca_bound_mu, ca_bound_sd = _mean_std(mineral_bound_ca)
    _plot_mean_std(ax, t_days, p_bound_mu, p_bound_sd, c_main, "Mineral-bound phosphate", "Phosphate mean ± SD from sampled effective Ca/P ratios")
    _plot_mean_std(ax_ca, t_days, ca_bound_mu, ca_bound_sd, c_alt, "Mineral-bound calcium", "Calcium mean ± SD from sampled effective Ca/P ratios")
    _phase_lines(ax, plot)
    _clean_axis(ax)
    _clean_twin_axis(ax_ca)
    ax.set_xlim(0, x_max)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Model-derived bound phosphate equivalent", color=c_main)
    ax_ca.set_ylabel("Model-derived bound calcium equivalent", color=c_alt)
    ax.tick_params(axis="y", colors=c_main)
    ax_ca.tick_params(axis="y", colors=c_alt)
    _set_panel_title(ax, "Mineral-bound element uptake")
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_ca.get_legend_handles_labels()
    ax.legend(
        lines_l + lines_r,
        labels_l + labels_r,
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.0, 1.02),
        ncol=2,
        borderaxespad=0.0,
        columnspacing=1.4,
        handlelength=2.0,
    )
    _save(fig, out_dir, "fig12_totals_mean_std", plot)
