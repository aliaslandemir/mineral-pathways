"""Archived model-derived regime-map workflow."""

import csv
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from .io_utils import save_figure
from .model import MIN_DENOMINATOR, beta_gp_available_phosphate, simulate


REGION_SUBTHRESHOLD = "Subthreshold"
REGION_ACP_RICH = "ACP-dominated transition"
REGION_APATITE_RICH = "Apatite-rich"
REGION_DISPLAY = {
    REGION_SUBTHRESHOLD: "Subthreshold",
    REGION_ACP_RICH: "ACP-dominated",
    REGION_APATITE_RICH: "Apatite-rich",
}
REGION_TO_CODE = {
    REGION_SUBTHRESHOLD: 0,
    REGION_ACP_RICH: 1,
    REGION_APATITE_RICH: 2,
}


@dataclass(frozen=True)
class RegimeMapResult:
    q_pi_crit: float
    region_counts: dict[str, int]
    csv_rows: int
    output_dir: Path


def maturation_fraction(A_final: float, H_final: float) -> float:
    total = float(A_final) + float(H_final)
    if total <= MIN_DENOMINATOR:
        return float("nan")
    return float(H_final) / total


def classify_regime(max_s_a: float, phi_h: float, mineral_total: float = 1.0, eps: float = MIN_DENOMINATOR) -> str:
    if max_s_a <= 1.0 or mineral_total <= eps:
        return REGION_SUBTHRESHOLD
    if np.isnan(phi_h) or phi_h < 0.5:
        return REGION_ACP_RICH
    return REGION_APATITE_RICH


def analytical_q_pi_crit(params: dict[str, float]) -> float:
    c_ext = max(float(params["C_ext"]), MIN_DENOMINATOR)
    p_ext = float(params["P_ext"])
    k_a = max(float(params["K_A"]), MIN_DENOMINATOR)
    return max(0.0, float(np.sqrt(k_a / c_ext**3) - p_ext))


def _validate_finite(values: dict[str, np.ndarray], q_pi: float, k_m: float) -> None:
    for name, arr in values.items():
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Non-finite {name} values at q_Pi={q_pi:.6g}, k_m={k_m:.6g}.")


def _value_at_time(t: np.ndarray, values: np.ndarray, day: float) -> float:
    return float(np.interp(float(day), np.asarray(t, dtype=float), np.asarray(values, dtype=float)))


def _evaluate_regime_point(task):
    row_idx, col_idx, q_pi, k_m, base_params, beta_gp_fixed, horizons, dt, media_interval = task
    params = dict(base_params)
    params["beta_GP"] = beta_gp_fixed
    params["beta_GP_cycle"] = beta_gp_fixed
    params["_beta_GP_cycle_explicit"] = True
    release_fraction = min(max(float(q_pi) / beta_gp_fixed, 0.0), 1.0 - MIN_DENOMINATOR)
    params["k_hyd"] = float(-np.log1p(-release_fraction) / max(float(media_interval), MIN_DENOMINATOR))
    params["eta_48h"] = float(1.0 - np.exp(-2.0 * params["k_hyd"]))
    params["_k_hyd_explicit"] = True
    params.pop("_eta_dt_explicit", None)
    params["k_m"] = float(k_m)
    sim = simulate(params, t_end=max(horizons), dt=dt, media_interval=media_interval)
    _validate_finite(
        {"C": sim.C, "P": sim.P, "A": sim.A, "H": sim.H, "Rf": sim.Rf, "Rm": sim.Rm, "S_A": sim.S_A},
        float(q_pi),
        float(k_m),
    )
    rows = []
    for day in horizons:
        horizon_mask = sim.t <= float(day) + MIN_DENOMINATOR
        max_s_a = float(np.max(sim.S_A[horizon_mask]))
        a_day = _value_at_time(sim.t, sim.A, day)
        h_day = _value_at_time(sim.t, sim.H, day)
        phi_h = maturation_fraction(a_day, h_day)
        mineral_total = a_day + h_day
        region = classify_regime(max_s_a, phi_h, mineral_total)
        rows.append(
            {
                "day": float(day),
                "q_Pi": float(q_pi),
                "eta_48h": float(params["eta_48h"]),
                "k_hyd": float(params["k_hyd"]),
                "k_m": float(k_m),
                "max_SA": max_s_a,
                "A": a_day,
                "H": h_day,
                "phi_H": phi_h,
                "mineral_total": mineral_total,
                "region": region,
                "regime": REGION_TO_CODE[region],
            }
        )
    return row_idx, col_idx, rows


def _write_caption(path: Path, q_pi_crit: float, grid_shape: tuple[int, int], horizons: tuple[float, ...]) -> None:
    horizon_text = " and ".join(f"day {day:g}" for day in horizons)
    caption = (
        f"Mineralization-regime maps for the reduced ACP-to-apatite model at {horizon_text}. "
        "Regions are model-defined from the maximum effective precursor-formation index and "
        "the endpoint maturation fraction phi_H(t)=H(t)/(A(t)+H(t)). Subthreshold points "
        "have max_t S_A(t) <= 1; above-threshold points are classified as an "
        "ACP-dominated transition when phi_H(t) < 0.5 and apatite-rich when "
        "phi_H(t) >= 0.5. The 0.5 boundary is an operational pool-dominance "
        "criterion, not a biological constant. Maturation is first-order in the "
        "existing ACP-like precursor pool, so the lower-k_m portion of the map "
        "captures delayed conversion rather than an imposed apatite threshold. "
        "The near-vertical onset boundary "
        "indicates that threshold crossing is primarily controlled by phosphate "
        "availability in this model, whereas k_m mainly determines post-threshold "
        "partitioning between ACP-like precursor and apatite-like pools. The dashed "
        "line marks the analytical precursor-formation threshold "
        f"q_Pi,crit = sqrt(K_A/C_ext^3) - P_ext = {q_pi_crit:.4g} mM cycle^-1. "
        "The q_Pi axis is implemented through first-order beta-GP hydrolysis by "
        "converting each per-cycle release target to k_hyd and eta_48h. "
        f"The exported grid contains {grid_shape[0]} x {grid_shape[1]} deterministic "
        "simulations with other parameters at baseline values."
    )
    path.write_text(caption + "\n", encoding="utf-8")


def _plot_regime_map(
    output_dir: Path,
    q_values: np.ndarray,
    k_m_values: np.ndarray,
    region_codes_by_day: dict[float, np.ndarray],
    baseline_q_pi: float,
    baseline_k_m: float,
    q_pi_crit: float,
    plot: Any,
) -> None:
    colors = {
        REGION_SUBTHRESHOLD: "#d9d9d9",
        REGION_ACP_RICH: "#56b4e9",
        REGION_APATITE_RICH: "#e69f00",
    }
    q_min = float(q_values.min())
    q_max = float(q_values.max())
    k_min = float(k_m_values.min())
    k_max = float(k_m_values.max())
    fig, axes = plt.subplots(1, len(region_codes_by_day), figsize=(7.35, 4.15), constrained_layout=False, sharex=True, sharey=True)
    if len(region_codes_by_day) == 1:
        axes = [axes]
    present_codes_all: set[int] = set()
    k_dense = np.linspace(k_min, k_max, 600)

    for ax, (day, region_codes) in zip(axes, region_codes_by_day.items()):
        present_codes = set(int(code) for code in np.unique(region_codes))
        present_codes_all.update(present_codes)
        apatite_boundary = np.full_like(k_m_values, q_max, dtype=float)
        for idx, codes_at_k in enumerate(region_codes):
            apatite_idx = np.where(codes_at_k >= REGION_TO_CODE[REGION_APATITE_RICH])[0]
            if apatite_idx.size:
                first = int(apatite_idx[0])
                if first == 0:
                    apatite_boundary[idx] = float(q_values[0])
                else:
                    apatite_boundary[idx] = 0.5 * (float(q_values[first - 1]) + float(q_values[first]))

        degree = min(3, max(1, len(k_m_values) - 1))
        coeffs = np.polyfit(k_m_values, apatite_boundary, deg=degree)
        apatite_boundary_dense = np.polyval(coeffs, k_dense)
        apatite_boundary_dense = np.clip(apatite_boundary_dense, q_pi_crit, q_max)
        transition_left = np.full_like(k_dense, q_pi_crit)
        transition_right = np.maximum(apatite_boundary_dense, transition_left)
        if REGION_TO_CODE[REGION_ACP_RICH] not in present_codes:
            transition_right = transition_left.copy()

        ax.fill_betweenx(k_dense, q_min, q_pi_crit, color=colors[REGION_SUBTHRESHOLD], linewidth=0)
        ax.fill_betweenx(k_dense, transition_left, transition_right, color=colors[REGION_ACP_RICH], linewidth=0)
        ax.fill_betweenx(k_dense, transition_right, q_max, color=colors[REGION_APATITE_RICH], linewidth=0)
        if REGION_TO_CODE[REGION_ACP_RICH] in present_codes and REGION_TO_CODE[REGION_APATITE_RICH] in present_codes:
            ax.plot(
                transition_right,
                k_dense,
                color="#4b2e83",
                linestyle="-",
                linewidth=1.35,
                alpha=0.92,
                zorder=4,
            )
        ax.axvline(q_pi_crit, color="#111111", linestyle=(0, (5, 2.5)), linewidth=1.15, zorder=5)
        ax.scatter(
            [baseline_q_pi],
            [baseline_k_m],
            marker="o",
            s=42,
            facecolor="white",
            edgecolor="#202020",
            linewidth=1.05,
            zorder=6,
        )
        ax.set_xlim(q_min, q_max)
        ax.set_ylim(k_min, k_max)
        ax.set_title(f"Day {day:g}", fontsize=11.2, pad=7)
        ax.tick_params(length=3.5, width=0.85, labelsize=9.2)
        ax.spines["left"].set_linewidth(0.9)
        ax.spines["bottom"].set_linewidth(0.9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if REGION_TO_CODE[REGION_ACP_RICH] in present_codes:
            locations = np.argwhere(region_codes == REGION_TO_CODE[REGION_ACP_RICH])
            row = int(np.round(np.median(locations[:, 0]))); col = int(np.round(np.median(locations[:, 1])))
            ax.text(q_values[col], max(k_m_values[row], k_min + 0.08 * (k_max - k_min)), "ACP", ha="center", va="center", fontsize=8.8)
        if REGION_TO_CODE[REGION_APATITE_RICH] in present_codes:
            ax.text(0.5 * (q_pi_crit + q_max), k_min + 0.88 * (k_max - k_min), "Apatite", ha="center", va="center", fontsize=8.8)

    axes[0].set_ylabel(r"Maturation rate, $k_m$ (day$^{-1}$)", fontsize=10.4)
    fig.supxlabel(r"Effective phosphate release, $q_{\mathrm{Pi}}$ (mM cycle$^{-1}$)", fontsize=10.4, y=0.19)
    fig.suptitle("Model-derived mineralization regimes", fontsize=11.8, y=0.965)
    legend_handles = [
        Patch(facecolor=colors[name], edgecolor="none", label=REGION_DISPLAY[name])
        for name, code in REGION_TO_CODE.items() if code in present_codes_all
    ]
    legend_handles.append(Line2D([0], [0], color="#111111", linestyle=(0, (5, 2.5)), linewidth=1.35, label="Analytical onset threshold"))
    if REGION_TO_CODE[REGION_ACP_RICH] in present_codes_all and REGION_TO_CODE[REGION_APATITE_RICH] in present_codes_all:
        legend_handles.append(Line2D([0], [0], color="#4b2e83", linewidth=1.35, label=r"$\phi_H=0.5$ boundary"))
    fig.legend(
        handles=legend_handles,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        borderaxespad=0.0,
        columnspacing=1.0,
        handlelength=1.5,
        handletextpad=0.6,
        prop={"size": 8.5},
    )
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.34, top=0.84, wspace=0.10)
    save_figure(
        fig,
        output_dir,
        "regime_map",
        formats=plot.figure_formats,
        dpi=plot.save_dpi,
        pad_inches=0.04,
    )


def generate_regime_map(
    base_params: dict[str, float],
    output_dir: Path,
    t_end: float,
    dt: float,
    media_interval: float,
    plot: Any,
    q_pi_range: tuple[float, float] = (0.3, 3.0),
    k_m_range: tuple[float, float] = (0.02, 0.80),
    grid_size: int = 30,
    jobs: int = 1,
) -> RegimeMapResult:
    if grid_size < 10:
        raise ValueError("Regime-map grid_size must be at least 10.")

    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_release = beta_gp_available_phosphate(base_params, media_interval)
    q_pi_range = (q_pi_range[0], max(q_pi_range[1], 1.15 * baseline_release))
    q_values = np.linspace(q_pi_range[0], q_pi_range[1], grid_size)
    k_m_values = np.linspace(k_m_range[0], k_m_range[1], grid_size)
    horizons = tuple(float(day) for day in (min(10.0, t_end), t_end))
    horizons = tuple(dict.fromkeys(horizons))
    beta_gp_fixed = float(base_params["beta_GP"])
    if beta_gp_fixed <= MIN_DENOMINATOR:
        raise ValueError("beta_GP must be positive for q_Pi-to-k_hyd conversion.")

    rows: list[dict[str, float | str]] = []
    region_counts: Counter[str] = Counter()

    tasks = [
        (row_idx, col_idx, float(q_pi), float(k_m), base_params, beta_gp_fixed, horizons, dt, media_interval)
        for row_idx, k_m in enumerate(k_m_values)
        for col_idx, q_pi in enumerate(q_values)
    ]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as executor:
            results = executor.map(_evaluate_regime_point, tasks, chunksize=max(1, len(tasks) // (jobs * 8)))
            evaluated = list(results)
    else:
        evaluated = [_evaluate_regime_point(task) for task in tasks]

    region_codes_by_day = {day: np.zeros((grid_size, grid_size), dtype=float) for day in horizons}
    for row_idx, col_idx, day_rows in evaluated:
        for row in day_rows:
            day = float(row["day"])
            region = str(row["region"])
            region_counts[f"day{day:g} {region}"] += 1
            region_codes_by_day[day][row_idx, col_idx] = REGION_TO_CODE[region]
            rows.append(row)

    assert REGION_TO_CODE == {REGION_SUBTHRESHOLD: 0, REGION_ACP_RICH: 1, REGION_APATITE_RICH: 2}
    for day in horizons:
        for name in REGION_TO_CODE:
            if region_counts[f"day{day:g} {name}"] == 0:
                warnings.warn(f"Regime-map class absent from day {day:g} grid: {REGION_DISPLAY[name]}", RuntimeWarning)
        print(
            f"Regime-map class counts day {day:g}: "
            + ", ".join(f"{REGION_DISPLAY[k]}={region_counts[f'day{day:g} {k}']}" for k in REGION_TO_CODE)
        )

    csv_path = output_dir / "regime_map_data.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["day", "q_Pi", "eta_48h", "k_hyd", "k_m", "max_SA", "A", "H", "phi_H", "mineral_total", "region", "regime"],
        )
        writer.writeheader()
        writer.writerows(rows)

    q_pi_crit = analytical_q_pi_crit(base_params)
    _plot_regime_map(
        output_dir=output_dir,
        q_values=q_values,
        k_m_values=k_m_values,
        region_codes_by_day=region_codes_by_day,
        baseline_q_pi=baseline_release,
        baseline_k_m=float(base_params["k_m"]),
        q_pi_crit=q_pi_crit,
        plot=plot,
    )
    _write_caption(output_dir / "regime_map_caption.txt", q_pi_crit, (grid_size, grid_size), horizons)

    return RegimeMapResult(
        q_pi_crit=q_pi_crit,
        region_counts={name: int(region_counts.get(name, 0)) for name in sorted(region_counts)},
        csv_rows=len(rows),
        output_dir=output_dir,
    )
