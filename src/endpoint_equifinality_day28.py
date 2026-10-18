"""Endpoint-equifinality analysis for final day-28 mineral.

Run from the repository root:

    python -m src.endpoint_equifinality_day28 \
      --output-dir output/extrafolder \
      --rerun-search \
      --n-samples 2000 \
      --n-refine 50 \
      --tolerance 0.02 \
      --formats pdf,png \
      --jobs 6
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mineral-pathways-matplotlib")
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.use(os.environ["MPLBACKEND"], force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc, spearmanr

from src.model import (
    DEFAULT_OUTPUT_DIR,
    EPS,
    MineralizationModel,
    ModelParameters,
    SimulationSettings,
    resolve_output_path,
)


FREE_PARAMETERS = ("eta_48h", "K_A", "K_Q", "k_f", "M_max", "k_b", "Q_max")
DISPLAY_NAMES = {
    "eta_48h": r"$\eta_{48h}$",
    "K_A": r"$K_A$",
    "K_Q": r"$K_Q$",
    "k_f": r"$k_f$",
    "M_max": r"$M_{mat,max}$",
    "k_b": r"$k_b$",
    "Q_max": r"$Q_{max}$",
}
CSV_NAMES = {"M_max": "M_mat_max"}

# Current Sobol bounds from figures.py, restricted here to the free
# endpoint-identifiability parameters requested for this analysis.
SOBOL_BOUNDS = {
    "eta_48h": (0.10, 0.50),
    "K_A": (24.0, 48.0),
    "K_Q": (50.0, 132.0),
    "k_f": (0.05, 1.00),
    "M_max": (0.25, 4.0),
    "k_b": (0.03, 3.0),
    "Q_max": (0.01, 4.0),
}
CSV_FREE_PARAMETERS = ("eta_48h", "K_A", "K_Q", "k_f", "M_mat_max", "k_b", "Q_max")
CSV_DISPLAY_NAMES = {
    "eta_48h": r"$\eta_{48h}$",
    "K_A": r"$K_A$",
    "K_Q": r"$K_Q$",
    "k_f": r"$k_f$",
    "M_mat_max": r"$M_{mat,max}$",
    "k_b": r"$k_b$",
    "Q_max": r"$Q_{max}$",
}
COMPENSATION_PAIRS = (
    ("eta_48h", "K_Q"),
    ("eta_48h", "k_b"),
    ("eta_48h", "Q_max"),
    ("K_Q", "k_b"),
    ("K_Q", "Q_max"),
    ("k_b", "Q_max"),
    ("M_mat_max", "Q_max"),
    ("K_A", "k_f"),
)
BASELINE_VALUES = {
    "eta_48h": 0.30,
    "K_A": 33.0,
    "K_Q": 66.0,
    "k_f": 0.2,
    "M_mat_max": 1.0,
    "k_b": 0.2,
    "Q_max": 1.0,
}
DEFAULT_EXTRA_OUTPUT_DIR = DEFAULT_OUTPUT_DIR / "extrafolder"
ENDPOINT_DAY = 28.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9.5,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def log_bounds() -> np.ndarray:
    return np.asarray([[np.log(SOBOL_BOUNDS[name][0]), np.log(SOBOL_BOUNDS[name][1])] for name in FREE_PARAMETERS], dtype=float)


def vector_to_changes(log_values: np.ndarray) -> dict[str, float]:
    return {name: float(value) for name, value in zip(FREE_PARAMETERS, np.exp(log_values))}


def endpoint_from_changes(task: tuple[dict[str, float], dict[str, float], SimulationSettings, float]) -> dict[str, float | str]:
    base_values, changes, settings, baseline_mtot = task
    values = dict(base_values)
    values.update(changes)
    try:
        params = ModelParameters(**values)
        result = MineralizationModel(params).simulate(settings)
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        row: dict[str, float | str] = {CSV_NAMES.get(name, name): float(changes[name]) for name in FREE_PARAMETERS}
        row.update({"status": str(exc), "Mmat_28": np.nan, "Q_28": np.nan, "Mtot_28": np.nan, "relative_error": np.inf, "fmat_28": np.nan})
        return row
    mtot = float(result.M_tot[-1])
    mmat = float(result.M_mat[-1])
    q_28 = float(result.Q[-1])
    row = {CSV_NAMES.get(name, name): float(changes[name]) for name in FREE_PARAMETERS}
    row.update(
        {
            "status": "accepted" if abs(mtot - baseline_mtot) / max(abs(baseline_mtot), EPS) <= 0.02 else "sampled",
            "A_28": float(result.A[-1]),
            "H_28": float(result.H[-1]),
            "Mmat_28": mmat,
            "Q_28": q_28,
            "Mtot_28": mtot,
            "relative_error": abs(mtot - baseline_mtot) / max(abs(baseline_mtot), EPS),
            "fmat_28": mmat / mtot if mtot > EPS else np.nan,
        }
    )
    return row


def lhs_log_samples(n_samples: int, seed: int) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=len(FREE_PARAMETERS), seed=seed)
    unit = sampler.random(n_samples)
    bounds = log_bounds()
    return bounds[:, 0] + unit * (bounds[:, 1] - bounds[:, 0])


def evaluate_rows(
    log_samples: np.ndarray,
    base_values: dict[str, float],
    settings: SimulationSettings,
    baseline_mtot: float,
    jobs: int,
) -> list[dict[str, float | str]]:
    tasks = [(base_values, vector_to_changes(row), settings, baseline_mtot) for row in log_samples]
    if jobs <= 1:
        return [endpoint_from_changes(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        return list(executor.map(endpoint_from_changes, tasks))


def objective_factory(base_values: dict[str, float], settings: SimulationSettings, baseline_mtot: float):
    lower_upper = log_bounds()
    bounds = tuple((float(low), float(high)) for low, high in lower_upper)

    def objective(log_values: np.ndarray) -> float:
        row = endpoint_from_changes((base_values, vector_to_changes(log_values), settings, baseline_mtot))
        rel = float(row["relative_error"])
        if not np.isfinite(rel):
            return 1.0e6
        return rel * rel

    return objective, bounds


def refine_endpoint_matches(
    rows: list[dict[str, float | str]],
    base_values: dict[str, float],
    settings: SimulationSettings,
    baseline_mtot: float,
    n_refine: int,
) -> list[dict[str, float | str]]:
    finite = [row for row in rows if np.isfinite(float(row["relative_error"]))]
    finite.sort(key=lambda row: float(row["relative_error"]))
    starts = []
    for row in finite[:n_refine]:
        starts.append(np.log([float(row[CSV_NAMES.get(name, name)]) for name in FREE_PARAMETERS]))
    if not starts:
        return []

    objective, bounds = objective_factory(base_values, settings, baseline_mtot)
    refined: list[dict[str, float | str]] = []
    for start in starts:
        opt = minimize(objective, np.asarray(start, dtype=float), method="L-BFGS-B", bounds=bounds, options={"maxiter": 80, "ftol": 1.0e-12})
        candidate = np.asarray(opt.x if opt.success or np.isfinite(opt.fun) else start, dtype=float)
        row = endpoint_from_changes((base_values, vector_to_changes(candidate), settings, baseline_mtot))
        row["status"] = "refined"
        refined.append(row)
    return refined


def accepted_unique(rows: Iterable[dict[str, float | str]], tolerance: float) -> list[dict[str, float | str]]:
    accepted = []
    seen: set[tuple[float, ...]] = set()
    for row in rows:
        if float(row["relative_error"]) > tolerance:
            continue
        key = tuple(round(float(row[CSV_NAMES.get(name, name)]), 8) for name in FREE_PARAMETERS)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(row)
    return accepted


def save_accepted_csv(path: Path, rows: list[dict[str, float | str]], baseline_mtot: float) -> None:
    fields = [
        "eta_48h",
        "K_A",
        "K_Q",
        "k_f",
        "M_mat_max",
        "k_b",
        "Q_max",
        "A_28",
        "H_28",
        "Mmat_28",
        "Q_28",
        "Mtot_28",
        "relative_error",
        "fmat_28",
        "baseline_Mtot_28",
        "status",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = {field: row.get(field, "") for field in fields}
            out["baseline_Mtot_28"] = baseline_mtot
            writer.writerow(out)


def generate_accepted_sets(
    output_dir: Path,
    n_samples: int,
    n_refine: int,
    jobs: int,
    seed: int,
    tolerance: float,
) -> tuple[list[dict[str, float | str]], float]:
    params = ModelParameters()
    settings = SimulationSettings(t_end=ENDPOINT_DAY)
    baseline = MineralizationModel(params).simulate(settings)
    baseline_mtot = float(baseline.M_tot[-1])
    base_values = asdict(params)

    sampled = evaluate_rows(lhs_log_samples(n_samples, seed), base_values, settings, baseline_mtot, jobs)
    refined = refine_endpoint_matches(sampled, base_values, settings, baseline_mtot, n_refine) if n_refine > 0 else []
    accepted = accepted_unique([*sampled, *refined], tolerance)
    if not accepted:
        raise RuntimeError(
            f"No endpoint-equifinal sets found within tolerance {tolerance:g}; "
            "increase --n-samples, increase --n-refine, or relax --tolerance."
        )

    accepted.sort(key=lambda row: float(row["fmat_28"]))
    save_accepted_csv(output_dir / "accepted_endpoint_equifinal_sets.csv", accepted, baseline_mtot)
    return accepted, baseline_mtot


def load_accepted_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"No accepted endpoint-equifinal sets found in {path}")
    return rows


def accepted_csv_matches_current_baseline(rows: list[dict[str, float | str]], baseline_mtot: float) -> bool:
    try:
        csv_baseline_mtot = float(rows[0]["baseline_Mtot_28"])
    except (KeyError, TypeError, ValueError):
        return False
    return abs(csv_baseline_mtot - baseline_mtot) / max(abs(baseline_mtot), EPS) <= 1.0e-6


def normalized_parameter_values(rows: list[dict[str, float | str]], name: str) -> np.ndarray:
    return np.asarray([float(row[name]) / BASELINE_VALUES[name] for row in rows], dtype=float)


def baseline_endpoint_row() -> dict[str, float | str]:
    params = ModelParameters()
    settings = SimulationSettings(t_end=ENDPOINT_DAY)
    result = MineralizationModel(params).simulate(settings)
    mtot = float(result.M_tot[-1])
    mmat = float(result.M_mat[-1])
    return {
        "eta_48h": params.eta_48h,
        "K_A": params.K_A,
        "K_Q": params.K_Q,
        "k_f": params.k_f,
        "M_mat_max": params.M_max,
        "k_b": params.k_b,
        "Q_max": params.Q_max,
        "status": "baseline",
        "A_28": float(result.A[-1]),
        "H_28": float(result.H[-1]),
        "Mmat_28": mmat,
        "Q_28": float(result.Q[-1]),
        "Mtot_28": mtot,
        "relative_error": 0.0,
        "fmat_28": mmat / mtot if mtot > EPS else np.nan,
        "baseline_Mtot_28": mtot,
    }


def representative_rows(rows: list[dict[str, float | str]], baseline_row: dict[str, float | str]) -> list[tuple[str, dict[str, float | str]]]:
    fmat = np.asarray([float(row["fmat_28"]) for row in rows], dtype=float)
    selected = {
        "Solution-rich": int(np.argmin(fmat)),
        "Matrix-only": int(np.argmax(fmat)),
    }

    for label, quantile in (("Mixed", 0.25), ("Matrix-rich", 0.75)):
        target = float(np.quantile(fmat, quantile))
        available = [idx for idx in range(len(rows)) if idx not in selected.values()]
        selected[label] = min(available, key=lambda idx: abs(float(fmat[idx]) - target))

    ordered = ("Solution-rich", "Mixed", "Baseline", "Matrix-rich", "Matrix-only")
    return [(label, baseline_row if label == "Baseline" else rows[selected[label]]) for label in ordered]


def parameter_intervals(rows: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    intervals: list[dict[str, float | str]] = []
    for name in CSV_FREE_PARAMETERS:
        values = np.log10(normalized_parameter_values(rows, name))
        q025, q25, median, q75, q975 = np.percentile(values, [2.5, 25.0, 50.0, 75.0, 97.5])
        intervals.append(
            {
                "parameter": name,
                "display": CSV_DISPLAY_NAMES[name],
                "q025": float(q025),
                "q25": float(q25),
                "median": float(median),
                "q75": float(q75),
                "q975": float(q975),
            }
        )
    intervals.sort(key=lambda row: float(row["q975"]) - float(row["q025"]), reverse=True)
    return intervals


def write_caption(path: Path) -> None:
    caption = (
        "Endpoint equifinality at day 28. Each column represents a distinct accepted parameterization reproducing "
        "the day-28 total retained mineral endpoint within +/-2%; the true baseline simulation is inserted at its "
        "actual matrix-associated fraction as a black reference column. Panel A shows that accepted parameterizations retain "
        "nearly the same total mineral while spanning very different matrix-associated and solution-driven pathway "
        "partitions. Panel B shows the corresponding parameter fingerprints in the same parameterization order, revealing "
        "compensating parameter combinations underlying endpoint equifinality rather than a unique parameter set.\n"
    )
    path.write_text(caption, encoding="utf-8")


def endpoint_arrays(rows: list[dict[str, float | str]], baseline_row: dict[str, float | str]) -> dict[str, np.ndarray | float]:
    baseline_mtot = float(baseline_row["Mtot_28"])
    return {
        "mmat": np.asarray([float(row["Mmat_28"]) for row in rows], dtype=float),
        "q": np.asarray([float(row["Q_28"]) for row in rows], dtype=float),
        "mtot": np.asarray([float(row["Mtot_28"]) for row in rows], dtype=float),
        "fmat": np.asarray([float(row["fmat_28"]) for row in rows], dtype=float),
        "error": np.asarray([float(row["relative_error"]) for row in rows], dtype=float),
        "baseline_mmat": float(baseline_row["Mmat_28"]),
        "baseline_q": float(baseline_row["Q_28"]),
        "baseline_mtot": baseline_mtot,
        "baseline_fmat": float(baseline_row["fmat_28"]),
    }


def sorted_log_parameter_matrix(rows: list[dict[str, float | str]]) -> tuple[np.ndarray, list[str], tuple[float, float]]:
    parameter_order = ("eta_48h", "K_A", "k_f", "M_mat_max", "K_Q", "k_b", "Q_max")
    matrix = np.asarray([np.log10(normalized_parameter_values(rows, name)) for name in parameter_order], dtype=float)
    labels = [CSV_DISPLAY_NAMES[name] for name in parameter_order]
    limit = max(2.0, float(np.nanmax(np.abs(matrix))) * 1.05)
    return matrix, labels, (-limit, limit)


def display_rows_with_baseline(
    accepted_sorted: list[dict[str, float | str]],
    baseline_row: dict[str, float | str],
    max_accepted: int = 24,
) -> tuple[list[dict[str, float | str]], int]:
    if len(accepted_sorted) > max_accepted:
        indices = np.unique(np.linspace(0, len(accepted_sorted) - 1, max_accepted, dtype=int))
        accepted_sorted = [accepted_sorted[int(index)] for index in indices]
    fmat_values = [float(row["fmat_28"]) for row in accepted_sorted]
    insert_at = int(np.searchsorted(fmat_values, float(baseline_row["fmat_28"])))
    return accepted_sorted[:insert_at] + [baseline_row] + accepted_sorted[insert_at:], insert_at


def style_axes(*axes: plt.Axes) -> None:
    for ax in axes:
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(color="0.91", linewidth=0.75)
        ax.set_axisbelow(True)


def panel_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.15, 1.035, letter, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom")


def relative_ticks(ax: plt.Axes) -> None:
    ax.set_xticks([-2, -1, 0, 1, 2], ["0.01×", "0.1×", "1×", "10×", "100×"])


def save_named_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: tuple[str, ...]) -> None:
    for extension in formats:
        dpi = 600 if extension.lower() == "png" else None
        fig.savefig(output_dir / f"{stem}.{extension}", bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def plot_design_1(output_dir: Path, rows: list[dict[str, float | str]], formats: tuple[str, ...]) -> None:
    baseline_row = baseline_endpoint_row()
    arrays = endpoint_arrays(rows, baseline_row)
    parameter_matrix, labels, x_limits = sorted_log_parameter_matrix(rows)
    rng = np.random.default_rng(8721)
    matrix_color = "#4C78A8"

    fig = plt.figure(figsize=(7.35, 3.4))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.92, 1.18), wspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    endpoint_percent = 100.0 * arrays["mtot"] / float(arrays["baseline_mtot"])
    fmat_percent = 100.0 * arrays["fmat"]
    ax_a.axhspan(98.0, 102.0, color=matrix_color, alpha=0.10, linewidth=0)
    ax_a.axhline(100.0, color="black", linewidth=0.95)
    ax_a.vlines(fmat_percent, 96.25, 96.95, color="0.68", alpha=0.55, linewidth=0.65)
    ax_a.scatter(fmat_percent, endpoint_percent + rng.uniform(-0.10, 0.10, size=len(endpoint_percent)), s=22, color="0.66", alpha=0.54, linewidths=0)
    ax_a.scatter([100.0 * float(arrays["baseline_fmat"])], [100.0], s=105, color="black", marker="*", linewidths=0, zorder=4)
    ax_a.text(3.0, 101.35, "+/-2% endpoint band", fontsize=7.3, color=matrix_color)
    ax_a.text(3.0, 100.28, "baseline total", fontsize=7.3, color="black")
    for index, dx in ((int(np.nanargmin(arrays["fmat"])), 12.0), (int(np.nanargmax(arrays["fmat"])), -30.0)):
        percent = float(fmat_percent[index])
        ax_a.annotate(f"{percent:.0f}% matrix", xy=(percent, endpoint_percent[index]), xytext=(percent + dx, endpoint_percent[index] + 1.05), arrowprops={"arrowstyle": "-", "color": "0.35", "linewidth": 0.7}, fontsize=7.5, color="0.15")
    ax_a.set_xlim(0.0, 103.0)
    ax_a.set_ylim(96.0, 103.0)
    ax_a.set_xlabel("Matrix fraction at day 28 (%)")
    ax_a.set_ylabel(r"$M_{tot,28}$ / baseline (%)")
    ax_a.set_title("Endpoint band", fontweight="bold", pad=7)

    y = np.arange(parameter_matrix.shape[0])
    violins = ax_b.violinplot(list(parameter_matrix), positions=y, vert=False, widths=0.72, showmeans=False, showmedians=False, showextrema=False)
    for body in violins["bodies"]:
        body.set_facecolor(matrix_color)
        body.set_edgecolor("none")
        body.set_alpha(0.16)
    q025, q25, median, q75, q975 = np.percentile(parameter_matrix, [2.5, 25.0, 50.0, 75.0, 97.5], axis=1)
    for row_index, values in enumerate(parameter_matrix):
        ax_b.scatter(values, np.full_like(values, y[row_index], dtype=float) + rng.uniform(-0.13, 0.13, size=len(values)), s=10, color="0.58", alpha=0.40, linewidths=0)
    ax_b.hlines(y, q025, q975, color="0.48", linewidth=1.0)
    ax_b.hlines(y, q25, q75, color=matrix_color, linewidth=4.2)
    ax_b.scatter(median, y, s=28, color="black", zorder=4)
    ax_b.axvline(0.0, color="black", linewidth=0.95)
    ax_b.set_yticks(y, labels)
    ax_b.invert_yaxis()
    ax_b.set_xlim(*x_limits)
    relative_ticks(ax_b)
    ax_b.set_xlabel("Parameter value relative to baseline")
    ax_b.set_title("Parameter rainclouds", fontweight="bold", pad=7)

    style_axes(ax_a, ax_b)
    panel_letter(ax_a, "A")
    panel_letter(ax_b, "B")
    fig.subplots_adjust(top=0.86, bottom=0.22, left=0.09, right=0.985)
    save_named_figure(fig, output_dir, "endpoint_equifinality_day28_design1", formats)


def plot_design_2(output_dir: Path, rows: list[dict[str, float | str]], formats: tuple[str, ...]) -> None:
    baseline_row = baseline_endpoint_row()
    arrays = endpoint_arrays(rows, baseline_row)
    parameter_matrix, labels, x_limits = sorted_log_parameter_matrix(rows)
    order = np.argsort(arrays["fmat"])
    matrix_color = "#4C78A8"

    fig = plt.figure(figsize=(7.4, 3.55))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.9, 1.1), wspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0], projection="polar")
    ax_b = fig.add_subplot(grid[0, 1])

    theta = (0.08 + 0.84 * arrays["fmat"]) * np.pi
    radius = 1.0 + 7.5 * arrays["error"]
    colors = plt.cm.Blues(0.28 + 0.58 * arrays["fmat"])
    ax_a.scatter(theta, radius, s=34, color=colors, alpha=0.78, edgecolor="white", linewidth=0.4)
    baseline_theta = (0.08 + 0.84 * float(arrays["baseline_fmat"])) * np.pi
    ax_a.scatter([baseline_theta], [1.0], s=130, color="black", marker="*", zorder=5)
    ax_a.set_theta_zero_location("W")
    ax_a.set_theta_direction(-1)
    ax_a.set_thetamin(0)
    ax_a.set_thetamax(180)
    ax_a.set_rlim(0.95, 1.17)
    ax_a.set_rticks([1.0, 1.1], ["0%", "1.3%"])
    ax_a.set_rlabel_position(92)
    ax_a.set_xticks([0.08 * np.pi, 0.5 * np.pi, 0.92 * np.pi], ["solution", "mixed", "matrix"])
    ax_a.spines["polar"].set_color("0.25")
    ax_a.grid(color="0.88", linewidth=0.75)
    ax_a.set_title("Pathway compass", fontweight="bold", pad=10)
    ax_a.text(-0.20, 1.07, "A", transform=ax_a.transAxes, fontsize=11, fontweight="bold", va="bottom")

    sorted_matrix = parameter_matrix[:, order]
    image = ax_b.imshow(sorted_matrix, aspect="auto", cmap="RdBu_r", vmin=x_limits[0], vmax=x_limits[1], interpolation="nearest")
    ax_b.axvline(int(np.argmin(np.abs(arrays["fmat"][order] - float(arrays["baseline_fmat"])))), color="black", linewidth=1.1)
    ax_b.set_yticks(np.arange(len(labels)), labels)
    ax_b.set_xticks([0, len(order) - 1], [f"{100.0 * arrays['fmat'][order[0]]:.0f}% matrix", f"{100.0 * arrays['fmat'][order[-1]]:.0f}% matrix"])
    ax_b.set_xlabel("Accepted sets sorted by pathway")
    ax_b.set_title("Parameter barcode", fontweight="bold", pad=7)
    cbar = fig.colorbar(image, ax=ax_b, fraction=0.045, pad=0.025)
    cbar.set_label("log10(value/baseline)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    style_axes(ax_b)
    ax_b.grid(False)
    panel_letter(ax_b, "B")
    fig.subplots_adjust(top=0.84, bottom=0.23, left=0.07, right=0.97)
    save_named_figure(fig, output_dir, "endpoint_equifinality_day28_design2", formats)


def plot_design_3(output_dir: Path, rows: list[dict[str, float | str]], formats: tuple[str, ...]) -> None:
    baseline_row = baseline_endpoint_row()
    arrays = endpoint_arrays(rows, baseline_row)
    parameter_matrix, labels, x_limits = sorted_log_parameter_matrix(rows)
    order = np.argsort(arrays["fmat"])
    matrix_color = "#2B6CB0"
    solution_color = "#D98C24"

    fig = plt.figure(figsize=(7.45, 3.55))
    grid = fig.add_gridspec(1, 2, width_ratios=(0.95, 1.15), wspace=0.42)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])

    x = np.linspace(0.0, 1.0, len(order))
    mmat = arrays["mmat"][order]
    q = arrays["q"][order]
    total = arrays["mtot"][order]
    ax_a.fill_between(x, 0.0, mmat, color=matrix_color, alpha=0.88, linewidth=0)
    ax_a.fill_between(x, mmat, total, color=solution_color, alpha=0.88, linewidth=0)
    ax_a.plot(x, total, color="black", linewidth=1.0)
    ax_a.axhspan(0.98 * float(arrays["baseline_mtot"]), 1.02 * float(arrays["baseline_mtot"]), color="0.85", alpha=0.35, linewidth=0, zorder=0)
    baseline_x = float(np.interp(float(arrays["baseline_fmat"]), arrays["fmat"][order], x))
    ax_a.scatter([baseline_x], [float(arrays["baseline_mtot"])], s=105, color="black", marker="*", zorder=4)
    ax_a.text(0.06, 0.66 * float(np.nanmax(total)), "matrix", color="white", fontweight="bold", fontsize=8.0)
    ax_a.text(0.12, 0.91 * float(np.nanmax(total)), "solution", color="0.25", fontweight="bold", fontsize=8.0)
    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(0.0, 1.08 * float(np.nanmax(total)))
    ax_a.set_xticks([0.0, 0.5, 1.0], [f"{100.0 * arrays['fmat'][order[0]]:.0f}%", "pathway continuum", f"{100.0 * arrays['fmat'][order[-1]]:.0f}%"])
    ax_a.set_ylabel("Day-28 mineral (mM P-eq.)")
    ax_a.set_title("Pathway stack ribbon", fontweight="bold", pad=7)

    parameter_x = np.arange(parameter_matrix.shape[0])
    for col in order:
        color = plt.cm.Blues(0.24 + 0.62 * arrays["fmat"][col])
        ax_b.plot(parameter_x, parameter_matrix[:, col], color=color, alpha=0.32, linewidth=1.0)
    ax_b.plot(parameter_x, np.zeros_like(parameter_x, dtype=float), color="black", linewidth=1.0)
    medians = np.median(parameter_matrix, axis=1)
    ax_b.scatter(parameter_x, medians, s=28, color="black", zorder=4)
    ax_b.set_xticks(parameter_x, labels, rotation=35, ha="right")
    ax_b.set_ylim(*x_limits)
    ax_b.set_ylabel("Parameter value relative to baseline")
    ax_b.set_title("Parameter compensation bundle", fontweight="bold", pad=7)
    ax_b.set_yticks([-2, -1, 0, 1, 2], ["0.01×", "0.1×", "1×", "10×", "100×"])
    style_axes(ax_a, ax_b)
    panel_letter(ax_a, "A")
    panel_letter(ax_b, "B")
    fig.subplots_adjust(top=0.86, bottom=0.26, left=0.08, right=0.985)
    save_named_figure(fig, output_dir, "endpoint_equifinality_day28_design3", formats)


def plot_figure(
    output_dir: Path,
    rows: list[dict[str, float | str]],
    formats: tuple[str, ...],
) -> None:
    configure_style()
    baseline_row = baseline_endpoint_row()

    matrix_color = "#56B4E9"
    solution_color = "#CC79A7"
    baseline_mtot = float(baseline_row["Mtot_28"])
    parameter_order = ("eta_48h", "K_A", "k_f", "M_mat_max", "K_Q", "k_b", "Q_max")
    parameter_labels = [CSV_DISPLAY_NAMES[name] for name in parameter_order]

    accepted_sorted = sorted(rows, key=lambda row: float(row["fmat_28"]))
    column_rows, baseline_col = display_rows_with_baseline(accepted_sorted, baseline_row)

    x_pos = np.arange(len(column_rows))
    mmat = np.asarray([float(row["Mmat_28"]) for row in column_rows], dtype=float)
    q_28 = np.asarray([float(row["Q_28"]) for row in column_rows], dtype=float)
    mtot = mmat + q_28
    fmat = np.asarray([float(row["fmat_28"]) for row in column_rows], dtype=float)
    heatmap = np.asarray([[np.log10(float(row[name]) / BASELINE_VALUES[name]) for row in column_rows] for name in parameter_order], dtype=float)
    color_limit = max(1.0, float(np.nanmax(np.abs(heatmap))))

    fig = plt.figure(figsize=(7.2, 4.55))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.0, 0.035), height_ratios=(0.72, 0.92), hspace=0.50, wspace=0.05)
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[1, 0], sharex=ax_a)
    cax = fig.add_subplot(grid[1, 1])
    fig.add_subplot(grid[0, 1]).axis("off")

    bar_widths = np.full(len(column_rows), 0.72)
    bar_widths[baseline_col] = 0.92
    edge_colors = np.full(len(column_rows), "none", dtype=object)
    line_widths = np.zeros(len(column_rows))

    band_low = 0.98 * baseline_mtot
    band_high = 1.02 * baseline_mtot
    band_color = "#E8EEF3"
    ax_a.axhspan(band_low, band_high, color=band_color, alpha=0.95, linewidth=0, zorder=0)
    ax_a.axhline(band_low, color="0.35", linewidth=0.75, linestyle="--", zorder=1)
    ax_a.axhline(band_high, color="0.35", linewidth=0.75, linestyle="--", zorder=1)
    ax_a.bar(x_pos, mmat, width=bar_widths, color=matrix_color, edgecolor=edge_colors, linewidth=line_widths, label="Matrix", zorder=2)
    ax_a.bar(x_pos, q_28, bottom=mmat, width=bar_widths, color=solution_color, edgecolor=edge_colors, linewidth=line_widths, label="Solution", zorder=2)
    ax_a.plot(x_pos, mtot, color="black", linewidth=0.75, zorder=4)
    ax_a.axhline(baseline_mtot, color="black", linewidth=0.85, zorder=3)
    ax_a.add_patch(plt.Rectangle((baseline_col - 0.53, 0.0), 1.06, mtot[baseline_col], fill=False, edgecolor="black", linewidth=1.05, zorder=5))
    ax_a.text(
        0.985,
        1.055,
        "+/-2%",
        transform=ax_a.transAxes,
        ha="right",
        va="center",
        fontsize=7.2,
        color="black",
        clip_on=False,
        bbox={"boxstyle": "square,pad=0.14", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.96},
    )
    legend = ax_a.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=True,
        framealpha=0.96,
        facecolor="white",
        edgecolor="0.75",
        ncol=2,
        handlelength=1.0,
        columnspacing=1.0,
        borderpad=0.25,
    )
    legend.set_zorder(10)
    ax_a.set_ylabel("Day-28 mineral\n(mM P-eq.)", labelpad=4)
    ax_a.set_ylim(0.0, max(float(np.nanmax(mtot)), 1.02 * baseline_mtot) * 1.12)
    ax_a.set_title("A  Mineral source", loc="left", fontweight="bold", fontsize=9.0, pad=6)
    ax_a.tick_params(axis="x", labelbottom=False, length=0)

    image = ax_b.imshow(heatmap, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-color_limit, vmax=color_limit, extent=(-0.5, len(column_rows) - 0.5, len(parameter_order) - 0.5, -0.5))
    ax_b.axvline(baseline_col, color="black", linewidth=0.95, zorder=4)
    ax_b.add_patch(plt.Rectangle((baseline_col - 0.5, -0.5), 1.0, len(parameter_order), fill=False, edgecolor="black", linewidth=0.8, zorder=5))
    ax_b.set_yticks(np.arange(len(parameter_order)), parameter_labels)
    ax_b.set_title("B  Parameter changes", loc="left", fontweight="bold", fontsize=9.0, pad=6)
    ax_b.set_xlabel("Accepted sets ordered by matrix fraction")

    tick_positions = [0, baseline_col, len(column_rows) - 1]
    tick_labels = [f"{100.0 * fmat[0]:.0f}%", f"Baseline\n{100.0 * fmat[baseline_col]:.0f}%", f"{100.0 * fmat[-1]:.0f}%"]
    ax_b.set_xticks(tick_positions, tick_labels)
    ax_b.set_xlim(-0.5, len(column_rows) - 0.5)

    for ax in (ax_a, ax_b):
        ax.set_facecolor("white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
    ax_a.grid(axis="y", color="0.91", linewidth=0.65)
    for boundary in np.arange(0.5, len(parameter_order) - 0.5, 1.0):
        ax_b.axhline(boundary, color="white", linewidth=0.55, alpha=0.75, zorder=3)

    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("Change vs baseline", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)
    ax_b.xaxis.labelpad = 10
    fig.subplots_adjust(top=0.93, bottom=0.18, left=0.10, right=0.965)

    save_named_figure(fig, output_dir, "endpoint_equifinality_day28", formats)
    write_caption(output_dir / "endpoint_equifinality_day28_caption.txt")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Day-28 endpoint identifiability figure and accepted-set export.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRA_OUTPUT_DIR)
    parser.add_argument("--n-samples", type=int, default=2000, help="Latin-hypercube samples inside the current Sobol bounds.")
    parser.add_argument("--n-refine", type=int, default=0, help="Optional number of best sampled starts refined by bounded endpoint optimization.")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--tolerance", type=float, default=0.02)
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--rerun-search", action="store_true", help="Regenerate accepted sets instead of using the existing accepted-set CSV.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    output_dir = resolve_output_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = tuple(value.strip().lower() for value in args.formats.split(",") if value.strip())
    accepted_csv = output_dir / "accepted_endpoint_equifinal_sets.csv"

    baseline_mtot_current = float(baseline_endpoint_row()["Mtot_28"])
    should_generate = args.rerun_search or not accepted_csv.exists()
    accepted: list[dict[str, float | str]]
    if not should_generate:
        accepted = load_accepted_csv(accepted_csv)
        should_generate = not accepted_csv_matches_current_baseline(accepted, baseline_mtot_current)
        if should_generate:
            csv_baseline = accepted[0].get("baseline_Mtot_28", "unknown")
            print(f"Existing accepted-set CSV baseline ({csv_baseline}) does not match current baseline ({baseline_mtot_current:.8g}); regenerating.")

    if should_generate:
        accepted, baseline_mtot = generate_accepted_sets(output_dir, args.n_samples, args.n_refine, args.jobs, args.seed, args.tolerance)
        print(f"Generated {len(accepted)} endpoint-equifinal sets at {accepted_csv}.")
    else:
        baseline_mtot = float(accepted[0]["baseline_Mtot_28"])
        print(f"Loaded {len(accepted)} endpoint-equifinal sets from {accepted_csv}.")

    plot_figure(output_dir, accepted, formats)
    print(f"Baseline Mtot_28={baseline_mtot:.8g}")
    print(f"Saved outputs to {output_dir}")


if __name__ == "__main__":
    main()
